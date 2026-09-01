//! GPU 백엔드 (cubecl + HIP/ROCm) — universal 모드.
//!
//! 커널은 Rust(cubecl 매크로)로 작성, hipRTC가 gfx1151용으로 JIT 컴파일.
//! 같은 커널이 CUDA/PTX(sm_80)로도 컴파일되어 CMP 170HX에 직결 (ADR-0009).
//!
//! `GpuMatmul`: `Accelerator` 구현 — 양자화 가중치 원시 바이트를 GPU에 상주시키고
//! dequant+FMA 없는 mul+add GEMM을 수행. 블록/요소 순서는 CPU 참조(quant.rs·matmul.rs)와
//! 동일한 누산 순서를 유지해 수치 일치 확률을 최대화한다 (ADR-0005 strict FP).

use cubecl::prelude::*;

mod gemm2;
mod attn;
use cubecl::zspace::{Shape, Strides};
use cubecl_runtime::server::Handle;
use llm170_core::matmul::{Accelerator, Weight};
use llm170_gguf::GgmlType;
use llm170_profiler::profile_span;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

// 위상별 누적 시간(µs) — LLM170_GPU_TIME=1일 때 infer 종료 시 보고.
pub static T_UP: AtomicU64 = AtomicU64::new(0);
pub static T_LAUNCH: AtomicU64 = AtomicU64::new(0);
pub static T_READ: AtomicU64 = AtomicU64::new(0);
pub static N_OP: AtomicU64 = AtomicU64::new(0);

/// 계측 요약 출력 (server가 infer 완료 후 호출).
pub fn timing_report() {
    if T_UP.load(Ordering::Relaxed) == 0 && T_READ.load(Ordering::Relaxed) == 0 {
        return;
    }
    let (u, l, r, n) = (
        T_UP.load(Ordering::Relaxed),
        T_LAUNCH.load(Ordering::Relaxed),
        T_READ.load(Ordering::Relaxed),
        N_OP.load(Ordering::Relaxed),
    );
    eprintln!("# gpu-timing: up={:.1}s launch={:.1}s read={:.1}s ops={n}", u as f64 / 1e6, l as f64 / 1e6, r as f64 / 1e6);
}

#[inline]
fn acc(a: &AtomicU64, d: std::time::Duration) {
    if std::env::var_os("LLM170_GPU_TIME").is_some() {
        a.fetch_add(d.as_micros() as u64, Ordering::Relaxed);
    }
}

/// x를 tlen(≤8의 2의 거듭제곱)행으로 패딩 — 디코드 q3 커널의 토큰 상각용.
fn pad_x(xs: &[Vec<f32>], n_in: usize, t: usize) -> (Vec<f32>, usize) {
    let tlen = if t <= 8 {
        t.max(1).next_power_of_two().min(8)
    } else {
        t.div_ceil(16) * 16 // q4 타일 정렬
    };
    let mut xf = Vec::with_capacity(tlen * n_in);
    for r in xs {
        xf.extend_from_slice(r);
    }
    for _ in t..tlen {
        xf.resize(xf.len() + n_in, 0.0);
    }
    (xf, tlen)
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// f32 스모크 커널 (원형 유지 — gpu-smoke 서브커맨드)
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
/// f32 GEMV: out[o] = Σ_i x[i]·W[o,i] — W: [n_out, n_in] 행 우선, 큐브당 1 출력 행.
fn gemv_f32(x: &Tensor<f32>, w: &Tensor<f32>, out: &mut Tensor<f32>, #[comptime] n_in: usize) {
    let o = ABSOLUTE_POS_X as usize;
    if o >= out.len() {
        terminate!();
    }
    let mut acc = 0.0f32;
    let base = o * n_in;
    for i in 0..n_in {
        acc += x[i] * w[base + i];
    }
    out[o] = acc;
}

/// 디바이스 프로브.
pub fn probe() -> Result<String, String> {
    let device = hanzo_cubecl_hip::AmdDevice::new(0);
    let client = hanzo_cubecl_hip::HipRuntime::client(&device);
    let _ = client.properties();
    Ok("hip device 0 ready".to_string())
}

/// wgpu/Vulkan 디바이스 프로브.
pub fn probe_vulkan() -> Result<String, String> {
    use cubecl::wgpu::{RuntimeOptions, Vulkan, WgpuDevice, WgpuRuntime, init_device, init_setup};
    let device = WgpuDevice::DefaultDevice;
    let setup = init_setup::<Vulkan>(&device, RuntimeOptions::default());
    let dev = init_device(setup, RuntimeOptions::default());
    let client = WgpuRuntime::client(&dev);
    let _ = client.properties();
    Ok("wgpu/vulkan device ready".to_string())
}

/// GEMV 스모크: 결정적 입력으로 CPU 대조.
pub fn smoke_gemv(n_in: usize, n_out: usize) -> Result<Vec<f32>, String> {
    let device = hanzo_cubecl_hip::AmdDevice::new(0);
    let client = hanzo_cubecl_hip::HipRuntime::client(&device);

    let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.37) % 1.0 - 0.5).collect();
    let w: Vec<f32> = (0..n_in * n_out)
        .map(|i| ((i * 31 + 7) as f32 * 0.11 % 1.0) - 0.5)
        .collect();

    let x_gpu = client.create_from_slice(bytemuck::cast_slice(&x));
    let w_gpu = client.create_from_slice(bytemuck::cast_slice(&w));
    let out_gpu = client.empty(n_out * 4);

    // SAFETY: 커널은 ABSOLUTE_POS_X >= out.len() 시 terminate!() — OOB 없음.
    unsafe {
        let shape_x: Shape = [n_in].into();
        let shape_w: Shape = [n_in * n_out].into();
        let shape_o: Shape = [n_out].into();
        let strides_x: Strides = [1].into();
        let strides_w: Strides = [1].into();
        let strides_o: Strides = [1].into();
        gemv_f32::launch_unchecked(
            &client,
            CubeCount::Static(n_out as u32, 1, 1),
            CubeDim::new_2d(64, 4), // 256 threads/cube — GEMV 행 병렬
            TensorArg::from_raw_parts(x_gpu, strides_x, shape_x),
            TensorArg::from_raw_parts(w_gpu, strides_w, shape_w),
            TensorArg::from_raw_parts(out_gpu.clone(), strides_o, shape_o),
            n_in,
        );
    }

    let bytes = client.read_one(out_gpu).map_err(|e| e.to_string())?;
    let v: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();

    let mut ok = true;
    for o in 0..n_out {
        let mut acc = 0.0f32;
        for i in 0..n_in {
            acc += x[i] * w[o * n_in + i];
        }
        if (acc - v[o]).abs() > 1e-3 {
            ok = false;
        }
    }
    if !ok {
        return Err("GEMV GPU 결과 불일치".into());
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// Accelerator 구현 — core::matmul::Accelerator (server --backend gpu)
// ---------------------------------------------------------------------------

mod buffers;
mod gemm5;
mod gdn_kernel;
use buffers::{ScratchPool, WeightStore, WRef};

/// GPU matmul 가속기 — 런타임 제네릭(HIP 또는 wgpu/Vulkan).
/// 가중치는 데이터 포인터 키로 1회 업로드 후 상주. 커널 소스는 런타임 공유 (ADR-0009).
pub struct GpuMatmul<R: Runtime> {
    client: ComputeClient<R>,
    /// 가중치 상주 저장소 (P0 아레나 — dummy 핸들 부류 원천 제거).
    weights: WeightStore,
    /// 크기별 재사용 스크래치/휘발업로드 풀 — 해제 없는 영속 보관.
    bufs: ScratchPool,
    ktab: Handle,  // iq4_nl 룩업 (f32×16)
    grid3: Handle, // iq3_s 그리드 (u32×512)
}

impl GpuMatmul<hanzo_cubecl_hip::HipRuntime> {
    /// ROCm/HIP 런타임 (기본).
    pub fn new_hip() -> Result<Self, String> {
        let device = hanzo_cubecl_hip::AmdDevice::new(0);
        let client = hanzo_cubecl_hip::HipRuntime::client(&device);
        Self::with_client(client)
    }
}

impl GpuMatmul<cubecl::wgpu::WgpuRuntime> {
    /// wgpu/Vulkan 런타임 — 같은 커널을 WGSL/SPIR-V로 컴파일.
    pub fn new_vulkan() -> Result<Self, String> {
        use cubecl::wgpu::{
            RuntimeOptions, Vulkan, WgpuDevice, WgpuRuntime, init_device, init_setup,
        };
        // DefaultDevice = 고성능 우선 선택 — 개발기 iGPU(8060S)·CMP dGPU 모두 대응
        // (DiscreteGpu 필터는 APU에서 설계상 페닉)
        let device = WgpuDevice::DefaultDevice;
        let setup = init_setup::<Vulkan>(&device, RuntimeOptions::default());
        let dev = init_device(setup, RuntimeOptions::default());
        let client = WgpuRuntime::client(&dev);
        Self::with_client(client)
    }
}

impl<R: Runtime> GpuMatmul<R> {
    fn with_client(client: ComputeClient<R>) -> Result<Self, String> {
        // 워크어라운드(2026-08-31): 장문 혼합 워크로드에서 DSD 스레드의
        // hipHccModuleLaunchKernel이 GPF(고정 IP, libamdhip64 7.2.2) — 런치와
        // 병행 할당의 경합 의심. 동기 런치로 소멸 실측(런치당 ~0.1ms).
        // LLM170_HIP_ASYNC=1이면 비활성(재현·재검증용).
        if std::env::var_os("LLM170_HIP_ASYNC").is_none() {
            // SAFETY: 모든 스레드 시작 전 초기화 경로에서 1회 — 경합 없음
            unsafe { std::env::set_var("HIP_LAUNCH_BLOCKING", "1") };
        }
        // 모든 할당을 영속 모드로 — cubecl 지연 해제가 큐 잔여 참조와 경합해
        // "Memory page 0 doesn't exist"(GPF)·가비지 판독(NaN)을 유발한다
        // (2026-09-01 실측). 해제를 전면 금지하고 재사용은 ScratchPool이
        // 담당 — VRAM 상한은 풀 계측(POOL_TOTAL)·가중치 예산(W_CAP)으로 관리.
        // SAFETY: 클라이언트 생성 직후 단일 스레드 — 이후 다른 allocation_mode
        // 호출 없음. "누수"는 의도적(전량 재사용).
        unsafe { client.allocation_mode(cubecl::MemoryAllocationMode::Persistent) };
        let ktab: Vec<f32> = llm170_core::KVALUES_IQ4NL
            .iter()
            .map(|&v| v as f32)
            .collect();
        let grid3 = llm170_core::IQ3S_GRID;
        let ktab = client.create_from_slice(bytemuck::cast_slice(&ktab));
        let grid3 = client.create_from_slice(bytemuck::cast_slice(&grid3));
        Ok(GpuMatmul {
            client,
            weights: WeightStore::new(),
            bufs: ScratchPool::new(),
            ktab,
            grid3,
        })
    }

    /// MoE 전문가 배치 down — K전문가 1런치 (P2-2). x는 [K·n_in] 평탄,
    /// ws는 전문가 스택 전체 뷰, outs [K][n_out].
    pub fn moe_down_gpu(
        &self,
        xs: &[Vec<f32>],
        ws: &Weight,
        expert_ids: &[u32],
        n_expert_stack: usize,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let k = xs.len();
        if k == 0 || outs.len() != k || expert_ids.len() != k {
            return Err(format!(
                "moe_down: 형상 불일치 k={k} outs={} ids={}",
                outs.len(),
                expert_ids.len()
            ));
        }
        let d = self.dev_weight(ws)?;
        if d.is_host() {
            return Err("moe_down: 스택 가중치 호스트 폴백 (예산 초과)".into());
        }
        let (n_in, n_out_full) = d.shape();
        if std::env::var_os("LLM170_DEBUG_MOE").is_some() {
            eprintln!("# moe_down: k={k} ids={expert_ids:?} stack={n_expert_stack} n_in={n_in} n_out_full={n_out_full}");
        }
        // 스택 [ne0=n_in][ne1=rows][ne2=experts] — w4 뷰는 n_out=rows·experts.
        let n_out = n_out_full / n_expert_stack;
        let wtype = d.ty() as u32 as usize;
        let wwords = d.words();
        let exp_bytes = wwords * 4 / n_expert_stack;
        // x 평탄 업로드 (t=1행 × K)
        let mut xf = Vec::with_capacity(k * n_in);
        for row in xs {
            if row.len() != n_in {
                return Err("moe_down: x 행 길이 불일치".into());
            }
            xf.extend_from_slice(row);
        }
        let t0 = std::time::Instant::now();
        let xg = self.client.create_from_slice(bytemuck::cast_slice(&xf));
        let eg = self.client.create_from_slice(bytemuck::cast_slice(expert_ids));
        let og = self.acquire_buf(k * n_out * 4)?;
        let pg = self.acquire_buf(k * n_out * 64 * 4)?;
        acc(&T_UP, t0.elapsed());
        let t1 = std::time::Instant::now();
        let gx = n_out.min(65535);
        let gz = n_out.div_ceil(gx);
        // SAFETY: 그리드 (gx, K, gz)·64레인 — o 접힘 포함 상한 내.
        unsafe {
            gemm5::gemm_q5::launch_unchecked(
                &self.client,
                CubeCount::Static(gx as u32, k as u32, gz as u32),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(xg.clone(), [1].into(), [xf.len()].into()),
                TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [wwords].into()),
                TensorArg::from_raw_parts(pg.clone(), [1].into(), [k * n_out * 64].into()),
                TensorArg::from_raw_parts(eg, [1].into(), [expert_ids.len()].into()),
                TensorArg::from_raw_parts(self.ktab.clone(), [1].into(), [16].into()),
                TensorArg::from_raw_parts(self.grid3.clone(), [1].into(), [512].into()),
                n_in,
                n_out,
                exp_bytes,
                gx,
                wtype,
            );
            gemm2::reduce_parts::launch_unchecked(
                &self.client,
                CubeCount::Static(gx as u32, k as u32, gz as u32),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(pg.clone(), [1].into(), [k * n_out * 64].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [k * n_out].into()),
                n_out,
                k,
                gx,
                64,
            );
        }
        acc(&T_LAUNCH, t1.elapsed());
        let t2 = std::time::Instant::now();
        let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
        acc(&T_READ, t2.elapsed());
        let res: &[f32] = bytemuck::cast_slice(&raw);
        for (e, out) in outs.iter_mut().enumerate() {
            out.copy_from_slice(&res[e * n_out..(e + 1) * n_out]);
        }
        self.release_bufs(&[
            (xg, xf.len() * 4),
            (og, k * n_out * 4),
            (pg, k * n_out * 64 * 4),
        ]);
        let _ = n_out_full;
        Ok(())
    }

    /// GDN AR 단일 토큰 상태 갱신 — S 업로드→커널→S·o 판독.
    /// e^g·β는 호스트 사전 계산(beta_ge), q는 scale 사전 곱.
    pub fn gdn_ar_gpu(
        &self,
        q_scaled: &[f32],
        k: &[f32],
        v: &[f32],
        beta_ge: &[f32],
        states: &mut [f32],
        out: &mut [f32],
        n_seqs: usize,
        h_k: usize,
        h_v: usize,
        d: usize,
    ) -> Result<(), String> {
        let k_stride = h_k * d;
        let v_stride = h_v * d;
        let n_pairs = n_seqs * h_v;
        let t0 = std::time::Instant::now();
        let sg = self.client.create_from_slice(bytemuck::cast_slice(states));
        let qg = self.client.create_from_slice(bytemuck::cast_slice(q_scaled));
        let kg = self.client.create_from_slice(bytemuck::cast_slice(k));
        let vg = self.client.create_from_slice(bytemuck::cast_slice(v));
        let bg = self.client.create_from_slice(bytemuck::cast_slice(beta_ge));
        let og = self.acquire_buf(out.len() * 4)?;
        acc(&T_UP, t0.elapsed());
        let t1 = std::time::Instant::now();
        // SAFETY: 그리드 (n_pairs,1,1) — 큐브당 pair 1개, 128레인 중 u≥d 종료.
        unsafe {
            gdn_kernel::gdn_ar::launch_unchecked(
                &self.client,
                CubeCount::Static(n_pairs as u32, 1, 1),
                CubeDim::new_1d(128),
                TensorArg::from_raw_parts(sg.clone(), [1].into(), [states.len()].into()),
                TensorArg::from_raw_parts(qg.clone(), [1].into(), [q_scaled.len()].into()),
                TensorArg::from_raw_parts(kg.clone(), [1].into(), [k.len()].into()),
                TensorArg::from_raw_parts(vg.clone(), [1].into(), [v.len()].into()),
                TensorArg::from_raw_parts(bg, [1].into(), [beta_ge.len()].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [out.len()].into()),
                d,
                k_stride,
                v_stride,
                h_v,
                h_k,
            );
        }
        acc(&T_LAUNCH, t1.elapsed());
        let t2 = std::time::Instant::now();
        let raw_s = self.client.read_one(sg.clone()).map_err(|e| e.to_string())?;
        let raw_o = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
        acc(&T_READ, t2.elapsed());
        states.copy_from_slice(bytemuck::cast_slice(&raw_s));
        out.copy_from_slice(bytemuck::cast_slice(&raw_o));
        // 휘발 업로드 풀 반납 — 해제 금지 (ADR-0014).
        self.release_bufs(&[
            (sg, states.len() * 4),
            (qg, q_scaled.len() * 4),
            (kg, k.len() * 4),
            (vg, v.len() * 4),
            (og, out.len() * 4),
        ]);
        Ok(())
    }

    /// 큐 완결 동기화 — 풀 재사용 안전성의 증명 지점. read_one이 커널
    /// 완결까지 보장하지 않는 결함(2026-09-01 NaN 실측) 대응: 스테이지
    /// 경계에서 호출해 모든 비행 중 연산 종료를 확정한다.
    pub fn barrier(&self) {
        let _ = cubecl_common::future::block_on(self.client.sync());
    }
    /// 풀 획득 위임.
    fn acquire_buf(&self, bytes: usize) -> Result<Handle, String> {
        self.bufs.acquire(&self.client, bytes)
    }

    /// read 동기화 후 반납 (크기 명시 — Handle은 크기 비공개).
    /// 휘발 업로드 포함 전부 풀에 영속 보관 — drop·지연 해제가 큐 잔여
    /// 연산과 경합해 가비지 판독(NaN, 2026-09-01 실측)하므로 해제 자체를
    /// 하지 않는다 (해제 경합 원인 제거 후 barrier 불필요).
    fn release_bufs(&self, hs: &[(Handle, usize)]) {
        self.bufs.release(hs);
    }

    /// 가중치 상주 예산 — 초과분은 호스트 연산. PLE(26.8GB)은 애초에 matmul
    /// 대상이 아니고 본체+전문가 ~83GB는 96GB VRAM에 들어감(llama.cpp와 동일
    /// 배치). GPF 원인은 청킹 부재였고 해소됨 — 2026-08-31 실측 재조정.
    /// 가중치 조회 위임 — P0 아레나. WRef::Host는 gpu()가 Err.
    fn dev_weight(&self, w: &Weight) -> Result<WRef, String> {
        self.weights.get(&self.client, w)
    }

    /// W4A8 GPU GEMM — x를 q8로 양자화해 전송, gemm_q6 + reduce. [t][n_out] 플랫.
    pub fn matmul_w4a8_gpu(&self, x: &[f32], w: &Weight) -> Result<Vec<f32>, String> {
        let y = llm170_core::quant::quantize_row_q8_ref(x);
        let t = 1usize; // 단일 벡터(디코드) 변형
        let d = self.dev_weight(w)?;
        let (n_in, n_out) = d.shape();
        // qs → 평탄 i8 → u32 워드, d → f32 배열
        let mut flat_q = Vec::with_capacity(y.len() * 32);
        for b in &y {
            for q in b.qs {
                flat_q.push(q);
            }
        }
        flat_q.resize(x.len(), 0);
        let mut qs_words = Vec::with_capacity(flat_q.len().div_ceil(4));
        for c in flat_q.chunks(4) {
            let mut word = 0u32;
            for (i, b) in c.iter().enumerate() {
                word |= (*b as u8 as u32) << (8 * i);
            }
            qs_words.push(word);
        }
        let ds: Vec<f32> = y.iter().map(|b| b.d).collect();
        let xq = self.client.create_from_slice(bytemuck::cast_slice(&qs_words));
        let xd = self.client.create_from_slice(bytemuck::cast_slice(&ds));
        let og = self.acquire_buf(t * n_out * 4)?;
        let pg = self.acquire_buf(t * n_out * 64 * 4)?;
        // SAFETY: 그리드 (n_out,1)·(n_out,t), 시작부 가드 — 상한 내.
        unsafe {
            gemm2::gemm_q6::launch_unchecked(
                &self.client,
                CubeCount::Static(n_out as u32, 1, 1),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(xq, [1].into(), [flat_q.len().div_ceil(4)].into()),
                TensorArg::from_raw_parts(xd, [1].into(), [ds.len()].into()),
                TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                TensorArg::from_raw_parts(pg.clone(), [1].into(), [t * n_out * 64].into()),
                TensorArg::from_raw_parts(self.ktab.clone(), [1].into(), [16].into()),
                TensorArg::from_raw_parts(self.grid3.clone(), [1].into(), [512].into()),
                n_in,
                n_out,
                t,
                d.ty() as u32 as usize,
            );
            let r_gx = n_out.min(65535);
            let r_gz = n_out.div_ceil(r_gx);
            gemm2::reduce_parts::launch_unchecked(
                &self.client,
                CubeCount::Static(r_gx as u32, t as u32, r_gz as u32),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(pg.clone(), [1].into(), [t * n_out * 64].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [t * n_out].into()),
                n_out,
                t,
                r_gx,
                64,
            );
        }
        let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
        self.release_bufs(&[(og, t * n_out * 4), (pg, t * n_out * 64 * 4)]);
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }

    /// QSA 마스크드 밀집 GQA — GPU 상주 (attn::qsa_score + qsa_mix).
    /// q: [t][n_head*2*hd] (norm·rope 완료, q‖gate 인터리브),
    /// ck/cv: 캐시 [n_past*n_kv*hd], mask: [t*n_past] u32 0/1,
    /// 반환 out: [t][n_head*hd] 게이트 적용 완료.
    fn qsa_attention_inner(
        &self,
        q: &[f32],
        ck: &[f32],
        cv: &[f32],
        mask: &[u32],
        kq_scale: f32,
        n_past: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<Vec<f32>, String> {
        // kq_scale을 ck에 사전 곱 (커널 f32 스칼라 인수 미지원) — q에 곱하면
        // q‖gate 인터리브에서 게이트까지 오염됨 (실측 2026-08-31)
        let cks: Vec<f32> = ck.iter().map(|v| v * kq_scale).collect();
        let qg = self.client.create_from_slice(bytemuck::cast_slice(q));
        let ckg = self.client.create_from_slice(bytemuck::cast_slice(&cks));
        let cvg = self.client.create_from_slice(bytemuck::cast_slice(cv));
        let mg = self.client.create_from_slice(bytemuck::cast_slice(mask));
        let up_bytes = (
            q.len() * 4,
            cks.len() * 4,
            cv.len() * 4,
            mask.len() * 4,
        );
        let sg = self.acquire_buf(t * n_head * n_past * 4)?;
        let og = self.acquire_buf(t * n_head * hd * 4)?;
        // SAFETY: 두 커널 모두 그리드가 (n_past|hd, n_head, t)를 정확히 덮고
        // 시작부 범위 가드 — 인덱스 상한 내, 무한루프 없음.
        unsafe {
            attn::qsa_score::launch_unchecked(
                &self.client,
                CubeCount::Static(n_past as u32, n_head as u32, t as u32),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(qg.clone(), [1].into(), [q.len()].into()),
                TensorArg::from_raw_parts(ckg.clone(), [1].into(), [ck.len()].into()),
                TensorArg::from_raw_parts(mg.clone(), [1].into(), [mask.len()].into()),
                TensorArg::from_raw_parts(sg.clone(), [1].into(), [t * n_head * n_past].into()),
                n_past,
                n_head,
                n_kv,
                hd,
                t,
            );
            attn::qsa_mix::launch_unchecked(
                &self.client,
                CubeCount::Static(hd as u32, n_head as u32, t as u32),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(qg.clone(), [1].into(), [q.len()].into()),
                TensorArg::from_raw_parts(sg.clone(), [1].into(), [t * n_head * n_past].into()),
                TensorArg::from_raw_parts(cvg.clone(), [1].into(), [cv.len()].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [t * n_head * hd].into()),
                n_past,
                n_head,
                n_kv,
                hd,
                t,
            );
        }
        let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
        // 휘발 업로드(q/ck/cv/mask)도 drop 대신 풀 반납 — 지연 해제 경합 제거.
        let (qb, ckb, cvb, mb) = up_bytes;
        self.release_bufs(&[
            (qg, qb),
            (ckg, ckb),
            (cvg, cvb),
            (mg, mb),
            (og, t * n_head * hd * 4),
            (sg, t * n_head * n_past * 4),
        ]);
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }

    /// 디버그: 텐서 블록 0 요소별 디양자화 값 (gpu-de).
    pub fn debug_dequant_block(&self, w: &Weight) -> Result<Vec<f32>, String> {
        let d = self.dev_weight(w)?;
        let (blck, _) = d.ty().block_info();
        let n = blck as usize;
        let og = self.client.empty(n * 4);
        // SAFETY: 그리드가 정확히 요소 수만큼 — 범위 가드 내.
        unsafe {
            gemm2::debug_de::launch_unchecked(
                &self.client,
                CubeCount::Static(n as u32, 1, 1),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [n].into()),
                TensorArg::from_raw_parts(self.ktab.clone(), [1].into(), [16].into()),
                TensorArg::from_raw_parts(self.grid3.clone(), [1].into(), [512].into()),
                d.ty() as u32 as usize,
                n,
            );
        }
        let raw = self.client.read_one(og).map_err(|e| e.to_string())?;
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }

    /// 디버그: 블록 0 원시 바이트 (gpu-de-bytes).
    pub fn debug_block_bytes(&self, w: &Weight) -> Result<Vec<f32>, String> {
        let d = self.dev_weight(w)?;
        let n = 110.min(d.words() * 4);
        let og = self.client.empty(n * 4);
        // SAFETY: 그리드 = 요소 수, 가드 내.
        unsafe {
            gemm2::debug_bytes::launch_unchecked(
                &self.client,
                CubeCount::Static(n as u32, 1, 1),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [n].into()),
                n,
            );
        }
        let raw = self.client.read_one(og).map_err(|e| e.to_string())?;
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }

    /// 디버그: q3_K 중간값 (mode).
    pub fn debug_q3(&self, w: &Weight, mode: usize) -> Result<Vec<f32>, String> {
        let d = self.dev_weight(w)?;
        let n = 256;
        let og = self.client.empty(n * 4);
        // SAFETY: 그리드 = 요소 수, 가드 내.
        unsafe {
            gemm2::debug_q3::launch_unchecked(
                &self.client,
                CubeCount::Static(n as u32, 1, 1),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [n].into()),
                n,
                mode,
            );
        }
        let raw = self.client.read_one(og).map_err(|e| e.to_string())?;
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }

    /// GEMM 런치 (비동기, v2 k-레인) — x는 외부에서 업로드해 전달. 반환: out 핸들.
    /// d는 GPU 상주분만(WRef::Host면 Err — dummy GEMM 원천 차단).
    fn launch_gemm(
        &self,
        d: &WRef,
        xg: Handle,
        t: usize,
    ) -> Result<(Handle, Handle, usize, usize), String> {
        let WRef::Gpu { h: wh, ty: wty, n_in, n_out, bytes } = d else {
            return Err("launch_gemm: 호스트 폴백 가중치 (아레나 위반)".into());
        };
        let (wh, wty, n_in, n_out, wbytes) = (wh.clone(), *wty, *n_in, *n_out, *bytes);
        // 스크래치 예산 3GiB(q4 prefill) — decode는 512MiB {64,16,4}
        // part 스크래치 예산 512MiB — q2/q4 공용. 대형 t에서 slices=64는
        // 텐서당 수 GB를 잡아 VRAM 초과 fault 유도(2026-08-31 실측).
        let slices: usize = if t * n_out * 64 * 4 <= 512 << 20 {
            64
        } else if t * n_out * 16 * 4 <= 512 << 20 {
            16
        } else {
            4
        };
        // 디코드(t ≤ 8) — q3 토큰 상각. prefill — q2(안정) + 엔진 층위 1024토큰
        // 청킹이 정답으로 확정(2026-08-31). 타일 커널 q4는 두 형상 모두 불안정
        // (콜드 런 비결정 / 모듈 로드 STATUS 700)해 제거됨.
        let decode = t <= 8 && slices == 64;
        // wgpu 그리드 X 상한 65,535 — o 차원을 Z로 접는다 (Vulkan 이식성).
        let gx = n_out.min(65535);
        let gz = n_out.div_ceil(gx);
        let og = self.acquire_buf(t * n_out * 4)?;
        let pg = self.acquire_buf(t * n_out * slices * 4)?;
        // SAFETY: 두 경로 모두 그리드가 (n_out, t)를 덮고 시작부 범위 가드 —
        // 인덱스 상한 내, 무한루프 없음.
        unsafe {
            if decode {
                gemm2::gemm_q3::launch_unchecked(
                    &self.client,
                    CubeCount::Static(gx as u32, 1, gz as u32),
                    CubeDim::new_1d(64),
                    TensorArg::from_raw_parts(xg, [1].into(), [t * n_in].into()),
                    TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [t * n_out * 64].into()),
                    TensorArg::from_raw_parts(self.ktab.clone(), [1].into(), [16].into()),
                    TensorArg::from_raw_parts(self.grid3.clone(), [1].into(), [512].into()),
                    n_in,
                    n_out,
                    gx,
                    t,
                    d.ty() as u32 as usize,
                );
            } else {
                let gy = t.div_ceil(4) as u32;
                gemm2::gemm_q2::launch_unchecked(
                    &self.client,
                    CubeCount::Static(gx as u32, gy, gz as u32),
                    CubeDim::new_2d(64, 4),
                    TensorArg::from_raw_parts(xg, [1].into(), [t * n_in].into()),
                    TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [t * n_out * slices].into()),
                    TensorArg::from_raw_parts(self.ktab.clone(), [1].into(), [16].into()),
                    TensorArg::from_raw_parts(self.grid3.clone(), [1].into(), [512].into()),
                    n_in,
                    n_out,
                    t,
                    gx,
                    d.ty() as u32 as usize,
                    slices,
                );
            }
            gemm2::reduce_parts::launch_unchecked(
                &self.client,
                CubeCount::Static(gx as u32, t as u32, gz as u32),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(pg.clone(), [1].into(), [t * n_out * slices].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [t * n_out].into()),
                n_out,
                t,
                gx,
                slices,
            );
        }
        Ok((og, pg, t * n_out * 4, t * n_out * slices * 4))
    }

    /// GEMM 실행 → [t][n_out] 플랫 결과 (tlen 패딩은 호출부 책임).
    fn run_gemm(&self, d: &WRef, xf: &[f32], t: usize) -> Result<Vec<f32>, String> {
        let t0 = std::time::Instant::now();
        let xg = self.client.create_from_slice(bytemuck::cast_slice(xf));
        acc(&T_UP, t0.elapsed());
        let t1 = std::time::Instant::now();
        let (og, pg, ob, pb) = self.launch_gemm(d, xg.clone(), t)?;
        acc(&T_LAUNCH, t1.elapsed());
        let t2 = std::time::Instant::now();
        let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
        acc(&T_READ, t2.elapsed());
        // xg는 drop 금지 — cubecl 지연 해제가 큐 잔여 연산과 경합해 가비지
        // 판독(NaN)을 유발(2026-09-01 실측). 풀 반납으로 해제 자체를 제거.
        self.release_bufs(&[(xg, xf.len() * 4), (og, ob), (pg, pb)]);
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }
}

impl<R: Runtime> Accelerator for GpuMatmul<R> {
    fn qsa_attention(
        &self,
        q: &[f32],
        ck: &[f32],
        cv: &[f32],
        mask: &[u32],
        kq_scale: f32,
        n_past: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<Vec<f32>, String> {
        self.qsa_attention_inner(q, ck, cv, mask, kq_scale, n_past, n_head, n_kv, hd, t)
    }

    fn matmul_batch(
        &self,
        xs: &[Vec<f32>],
        w: &Weight,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        profile_span!("gpu::matmulB");
        let t = xs.len();
        let n_in = w.n_in as usize;
        if xs.iter().any(|r| r.len() != n_in) {
            return Err(format!("matmul_batch: x 행 길이 != n_in ({n_in})"));
        }
        let d = self.dev_weight(w)?;
        if d.is_host() {
            llm170_core::matmul::matmul_batch(xs, w, outs);
            return Ok(());
        }
        let mut xf = Vec::with_capacity(t * n_in);
        for r in xs {
            xf.extend_from_slice(r);
        }
        let res = self.run_gemm(&d, &xf, t)?;
        for (ti, out) in outs.iter_mut().enumerate() {
            out.copy_from_slice(&res[ti * d.shape().1..(ti + 1) * d.shape().1]);
        }
        Ok(())
    }

    fn matmul(&self, x: &[f32], w: &Weight, out: &mut [f32]) -> Result<(), String> {
        profile_span!("gpu::matmul1");
        let d = self.dev_weight(w)?;
        if d.is_host() {
            llm170_core::matmul::matmul(x, w, out);
            return Ok(());
        }
        if x.len() != d.shape().0 {
            return Err(format!("matmul: x 길이 {} != n_in {}", x.len(), d.shape().0));
        }
        let res = self.run_gemm(&d, x, 1)?;
        out.copy_from_slice(&res[..d.shape().1]);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_down(
        &self,
        xs: &[Vec<f32>],
        ws: &Weight,
        expert_ids: &[u32],
        n_expert_stack: usize,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        self.moe_down_gpu(xs, ws, expert_ids, n_expert_stack, outs)
    }

    fn gdn_ar(
        &self,
        q_scaled: &[f32],
        k: &[f32],
        v: &[f32],
        beta_ge: &[f32],
        states: &mut [f32],
        out: &mut [f32],
        n_seqs: usize,
        h_k: usize,
        h_v: usize,
        d: usize,
    ) -> Result<(), String> {
        self.gdn_ar_gpu(q_scaled, k, v, beta_ge, states, out, n_seqs, h_k, h_v, d)
    }

    /// 짝 GEMM — 가중치마다 다른 1행 x (MoE down). 런치 배치 + 단일 동기화.
    fn matmul_paired(
        &self,
        xs: &[Vec<f32>],
        ws: &[Weight],
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        profile_span!("gpu::matmulP");
        if ws.is_empty() || ws.len() != xs.len() || ws.len() != outs.len() {
            return Err(format!(
                "matmul_paired: 형상 불일치 ws={} xs={} outs={}",
                ws.len(),
                xs.len(),
                outs.len()
            ));
        }
        let n_in = ws[0].n_in as usize;
        if xs.iter().any(|r| r.len() != n_in) {
            return Err(format!("matmul_paired: x 행 길이 != n_in ({n_in})"));
        }
        if ws.iter().any(|w| w.n_in as usize != n_in) {
            return Err("matmul_paired: 그룹 내 n_in 불일치".into());
        }
        // host 폴백은 CPU — dummy 핸들 방지.
        let mut idx_gpu: Vec<usize> = Vec::new();
        for (i, w) in ws.iter().enumerate() {
            let d = self.dev_weight(w)?;
            if d.is_host() {
                llm170_core::matmul::matmul(&xs[i], w, &mut outs[i]);
            } else {
                idx_gpu.push(i);
            }
        }
        if idx_gpu.is_empty() {
            return Ok(());
        }
        let devs: Vec<WRef> = idx_gpu
            .iter()
            .map(|&i| self.dev_weight(&ws[i]))
            .collect::<Result<_, _>>()?;
        for (d, &i) in devs.iter().zip(idx_gpu.iter()) {
            if outs[i].len() != d.shape().1 {
                return Err("matmul_paired: outs 형상 불일치".into());
            }
        }
        let t0 = std::time::Instant::now();
        let xgs: Vec<(Handle, usize)> = idx_gpu
            .iter()
            .map(|&i| {
                let (xf, _tlen) = pad_x(std::slice::from_ref(&xs[i]), n_in, 1);
                let bytes = xf.len() * 4;
                let h = self.client.create_from_slice(bytemuck::cast_slice(&xf));
                Ok::<(Handle, usize), String>((h, bytes))
            })
            .collect::<Result<Vec<_>, String>>()?;
        acc(&T_UP, t0.elapsed());
        let t1 = std::time::Instant::now();
        let pairs: Vec<_> = devs
            .iter()
            .zip(xgs.iter().map(|(g, _)| g.clone()))
            .map(|(d, xg)| self.launch_gemm(d, xg, 1))
            .collect::<Result<Vec<_>, _>>()?;
        acc(&T_LAUNCH, t1.elapsed());
        let t2 = std::time::Instant::now();
        let mut to_release: Vec<(Handle, usize)> = Vec::with_capacity(pairs.len() * 3);
        to_release.extend(xgs.iter().cloned());
        for (((og, pg, ob, pb), d), &i) in pairs.iter().zip(devs.iter()).zip(idx_gpu.iter()) {
            let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
            let res: &[f32] = bytemuck::cast_slice(&raw);
            outs[i].copy_from_slice(&res[..d.shape().1]);
            to_release.push((og.clone(), *ob));
            to_release.push((pg.clone(), *pb));
        }
        acc(&T_READ, t2.elapsed());
        self.release_bufs(&to_release);
        N_OP.fetch_add(devs.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn matmul_group(
        &self,
        xs: &[Vec<f32>],
        ws: &[Weight],
        outs: &mut [Vec<Vec<f32>>],
    ) -> Result<(), String> {
        profile_span!("gpu::matmulG");
        let t = xs.len();
        if ws.is_empty() || ws.len() != outs.len() {
            return Err(format!("matmul_group: 잘못된 구성 ws={} outs={}", ws.len(), outs.len()));
        }
        let n_in = ws[0].n_in as usize;
        if xs.iter().any(|r| r.len() != n_in) {
            return Err(format!("matmul_group: x 행 길이 != n_in ({n_in})"));
        }
        if ws.iter().any(|w| w.n_in as usize != n_in) {
            return Err("matmul_group: 그룹 내 n_in 불일치".into());
        }
        // host 폴백 가중치는 CPU로 — dev_weight가 dummy(4B) 핸들을 주기 때문.
        let mut idx_gpu: Vec<usize> = Vec::new();
        for (i, w) in ws.iter().enumerate() {
            let d = self.dev_weight(w)?;
            if d.is_host() {
                llm170_core::matmul::matmul_batch(xs, w, &mut outs[i]);
            } else {
                idx_gpu.push(i);
            }
        }
        if idx_gpu.is_empty() {
            return Ok(());
        }
        let devs: Vec<WRef> = idx_gpu
            .iter()
            .map(|&i| self.dev_weight(&ws[i]))
            .collect::<Result<_, _>>()?;
        for (d, &i) in devs.iter().zip(idx_gpu.iter()) {
            let out = &outs[i];
            if out.len() != t || out.iter().any(|r| r.len() != d.shape().1) {
                return Err("matmul_group: outs 형상 불일치".into());
            }
        }
        // x 1회 업로드 → K개 런치(비동기 FIFO) → 순차 read(첫 read가 스트림 동기화,
        // 이후 read는 이미 완료 상태라 즉시 반환) — 동기화 지점 1회로 파이프라이닝.
        let (xf, tlen) = pad_x(xs, n_in, t);
        let t0 = std::time::Instant::now();
        let xg = self.client.create_from_slice(bytemuck::cast_slice(&xf));
        acc(&T_UP, t0.elapsed());
        let t1 = std::time::Instant::now();
        let pairs: Vec<_> = devs
            .iter()
            .map(|d| self.launch_gemm(d, xg.clone(), tlen))
            .collect::<Result<Vec<_>, _>>()?;
        acc(&T_LAUNCH, t1.elapsed());
        let t2 = std::time::Instant::now();
        let mut to_release: Vec<(Handle, usize)> = Vec::with_capacity(pairs.len() * 2 + 1);
        to_release.push((xg, xf.len() * 4));
        for (((og, pg, ob, pb), d), &i) in pairs.iter().zip(devs.iter()).zip(idx_gpu.iter()) {
            let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
            let res: &[f32] = bytemuck::cast_slice(&raw);
            for (ti, out) in outs[i].iter_mut().enumerate() {
                out.copy_from_slice(&res[ti * d.shape().1..(ti + 1) * d.shape().1]);
            }
            to_release.push((og.clone(), *ob));
            to_release.push((pg.clone(), *pb));
        }
        acc(&T_READ, t2.elapsed());
        self.release_bufs(&to_release);
        let _ = tlen;
        N_OP.fetch_add(devs.len() as u64, Ordering::Relaxed);
        Ok(())
    }
}
