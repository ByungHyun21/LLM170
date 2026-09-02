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
mod ew;
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
/// 프레임 op/mm 호스트 구간(마샬링+암시 대기 포함) — P2 계측.
pub static T_FOP: AtomicU64 = AtomicU64::new(0);
pub static N_FOP: AtomicU64 = AtomicU64::new(0);

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
    eprintln!("# gpu-timing: up={:.1}s launch={:.1}s read={:.1}s ops={n} fop={:.1}s fn={}", u as f64 / 1e6, l as f64 / 1e6, r as f64 / 1e6, T_FOP.load(Ordering::Relaxed) as f64 / 1e6, N_FOP.load(Ordering::Relaxed));
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

/// 지정 스트림에서 클로저 실행 — 그 안의 커널 런치가 별도 hipStream로
/// 향한다 (cubecl 0.10 멀티스트림, 스트림별 독립 메모리 풀). value는
/// 스트림 식별자: 기본 스레드가 0, 충돌 회피 권장값 100+. 스트림 간
/// 데이터 의존은 자동 추적되지 않는다(할당 커서만) — 호출자 책임.
pub fn on_stream<T>(value: u64, f: impl FnOnce() -> T) -> T {
    cubecl_common::stream_id::StreamId { value }.executes(f)
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
    ktab2: Handle, // iq4_nl 바이트쌍 룩업 (u32×256) — W4A8 정수 경로
    grid3: Handle, // iq3_s 그리드 (u32×512)
    frames: Mutex<std::collections::HashMap<u64, (Handle, usize)>>,
    frame_next: AtomicU64,
}

impl<R: Runtime> GpuMatmul<R> {
    /// 프레임 레지스트리 조회.
    fn frame_get(&self, h: u64) -> Result<(Handle, usize), String> {
        self.frames
            .lock()
            .map_err(|_| "frame lock poisoned")?
            .get(&h)
            .cloned()
            .ok_or_else(|| format!("frame_get: 알 수 없는 핸들 {h}"))
    }

    /// launch_gemm + 프레임 출력 핸들 주입 (og 신규 획득 회피).
    fn launch_gemm_into(
        &self,
        d: &WRef,
        xg: Handle,
        t: usize,
        og: Handle,
    ) -> Result<(Handle, usize), String> {
        let WRef::Gpu { h: wh, ty: wty, n_in, n_out, bytes: _ } = d else {
            return Err("launch_gemm_into: 호스트 폴백 가중치 (아레나 위반)".into());
        };
        let (wh, wty, n_in, n_out) = (wh.clone(), *wty, *n_in, *n_out);
        let slices: usize = if t * n_out * 64 * 4 <= 512 << 20 {
            64
        } else if t * n_out * 16 * 4 <= 512 << 20 {
            16
        } else {
            4
        };
        let decode = t <= 8 && slices == 64;
        let gx = n_out.min(65535);
        let gz = n_out.div_ceil(gx);
        let pg = self.acquire_buf(t * n_out * slices * 4)?;
        // SAFETY: launch_gemm과 동일 그리드/가드 — og만 호출부 제공.
        unsafe {
            if decode {
                crate::gemm2::gemm_q3::launch_unchecked(
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
                crate::gemm2::gemm_q2::launch_unchecked(
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
            crate::gemm2::reduce_parts::launch_unchecked(
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
        self.release_bufs(&[(pg, t * n_out * slices * 4)]);
        Ok((og, t * n_out * 4))
    }
}

impl GpuMatmul<hanzo_cubecl_hip::HipRuntime> {
    /// ROCm/HIP runtime (default). Weight budget derives from the measured
    /// VRAM carve-out total (sysfs), not a per-machine constant.
    pub fn new_hip() -> Result<Self, String> {
        let device = hanzo_cubecl_hip::AmdDevice::new(0);
        let client = hanzo_cubecl_hip::HipRuntime::client(&device);
        let mem = buffers::sysfs_mem_total("vram").unwrap_or(32 << 30);
        Self::with_client_mem(client, mem)
    }
}

impl GpuMatmul<cubecl::wgpu::WgpuRuntime> {
    /// wgpu/Vulkan runtime — the same kernels compile to WGSL/SPIR-V.
    /// Weights land in GTT (host-visible), so the budget keys off GTT total.
    pub fn new_vulkan() -> Result<Self, String> {
        use cubecl::wgpu::{
            RuntimeOptions, Vulkan, WgpuDevice, WgpuRuntime, init_device, init_setup,
        };
        // DefaultDevice picks the high-performance adapter — works on both
        // the dev iGPU (8060S) and the CMP dGPU (DiscreteGpu filters would
        // panic by design on APUs).
        let device = WgpuDevice::DefaultDevice;
        let setup = init_setup::<Vulkan>(&device, RuntimeOptions::default());
        let dev = init_device(setup, RuntimeOptions::default());
        let client = WgpuRuntime::client(&dev);
        let mem = buffers::sysfs_mem_total("gtt").unwrap_or(32 << 30);
        Self::with_client_mem(client, mem)
    }
}

impl<R: Runtime> GpuMatmul<R> {
    fn with_client_mem(client: ComputeClient<R>, mem_total: usize) -> Result<Self, String> {
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
        // LLM170_NO_PERSISTENT=1이면 생략 — PersistentPool 경로가 HIP에서
        // "Memory page 0 doesn't exist"를 유발하는지 격리용 (2026-09-01).
        if std::env::var_os("LLM170_NO_PERSISTENT").is_none() {
            unsafe { client.allocation_mode(cubecl::MemoryAllocationMode::Persistent) };
        }
        let ktab: Vec<f32> = llm170_core::KVALUES_IQ4NL
            .iter()
            .map(|&v| v as f32)
            .collect();
        let ktab2: Vec<u32> = (0..256u32)
            .map(|b| {
                let lo = llm170_core::KVALUES_IQ4NL[(b & 0xF) as usize] as u8 as u32;
                let hi = llm170_core::KVALUES_IQ4NL[(b >> 4) as usize] as u8 as u32;
                lo | (hi << 8)
            })
            .collect();
        let ktab2 = client.create_from_slice(bytemuck::cast_slice(&ktab2));
        let grid3 = llm170_core::IQ3S_GRID;
        let ktab = client.create_from_slice(bytemuck::cast_slice(&ktab));
        let grid3 = client.create_from_slice(bytemuck::cast_slice(&grid3));
        Ok(GpuMatmul {
            client,
            weights: WeightStore::new(mem_total),
            bufs: ScratchPool::new(),
            ktab,
            ktab2,
            grid3,
            frames: Mutex::new(std::collections::HashMap::new()),
            frame_next: AtomicU64::new(1),
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

    /// GDN depthwise conv + ring — 값 스타일 (qwen35 디코드, 02-2).
    pub fn gdn_conv_gpu(
        &self,
        qkv: &[f32],
        conv_w: &[f32],
        state: &mut [f32],
        out: &mut [f32],
        ch: usize,
        k: usize,
    ) -> Result<(), String> {
        let t_len = qkv.len() / ch;
        let t0 = std::time::Instant::now();
        let qg = self.client.create_from_slice(bytemuck::cast_slice(qkv));
        let wg = self.client.create_from_slice(bytemuck::cast_slice(conv_w));
        let sg = self.client.create_from_slice(bytemuck::cast_slice(state));
        let og = self.acquire_buf(out.len() * 4)?;
        acc(&T_UP, t0.elapsed());
        let t1 = std::time::Instant::now();
        // SAFETY: 큐브당 1채널, 유닛 0만 실행 — ew.rs gdn_conv 가드.
        unsafe {
            ew::gdn_conv::launch_unchecked(
                &self.client,
                CubeCount::Static(ch as u32, 1, 1),
                CubeDim::new_1d(32),
                TensorArg::from_raw_parts(qg.clone(), [1].into(), [qkv.len()].into()),
                TensorArg::from_raw_parts(wg, [1].into(), [conv_w.len()].into()),
                TensorArg::from_raw_parts(sg.clone(), [1].into(), [state.len()].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [out.len()].into()),
                ch,
                k,
                t_len,
            );
        }
        acc(&T_LAUNCH, t1.elapsed());
        let t2 = std::time::Instant::now();
        let raw_s = self.client.read_one(sg.clone()).map_err(|e| e.to_string())?;
        let raw_o = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
        acc(&T_READ, t2.elapsed());
        state.copy_from_slice(bytemuck::cast_slice(&raw_s));
        out.copy_from_slice(bytemuck::cast_slice(&raw_o));
        self.release_bufs(&[(qg, qkv.len() * 4), (og, out.len() * 4)]);
        Ok(())
    }

    /// GDN 청크 프리필(t>1) — 값 스타일 (03 §3.1). q/k는 l2 완료·무스케일.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_chunk_gpu(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        g: &[f32],
        states: &mut [f32],
        out: &mut [f32],
        t_len: usize,
        h_k: usize,
        h_v: usize,
        d: usize,
    ) -> Result<(), String> {
        if d > 128 {
            return Err("gdn_chunk: d>128 미지원".into());
        }
        // 제로 패딩 — 커널은 n_chunks·CS 행을 무조건 판독 (패딩 기여 0).
        let t_pad = t_len.div_ceil(gdn_kernel::CS_K) * gdn_kernel::CS_K;
        let mut qp = q.to_vec();
        let mut kp = k.to_vec();
        let mut vp = v.to_vec();
        let mut bp = beta.to_vec();
        let mut gp = g.to_vec();
        qp.resize(t_pad * h_k * d, 0.0);
        kp.resize(t_pad * h_k * d, 0.0);
        vp.resize(t_pad * h_v * d, 0.0);
        bp.resize(t_pad * h_v, 0.0);
        gp.resize(t_pad * h_v, 0.0);
        let mut outp = vec![0.0f32; t_pad * h_v * d];
        let t0 = std::time::Instant::now();
        let qg = self.client.create_from_slice(bytemuck::cast_slice(&qp));
        let kg = self.client.create_from_slice(bytemuck::cast_slice(&kp));
        let vg = self.client.create_from_slice(bytemuck::cast_slice(&vp));
        let bg = self.client.create_from_slice(bytemuck::cast_slice(&bp));
        let gg = self.client.create_from_slice(bytemuck::cast_slice(&gp));
        let sg = self.client.create_from_slice(bytemuck::cast_slice(states));
        let og = self.acquire_buf(outp.len() * 4)?;
        acc(&T_UP, t0.elapsed());
        let t1 = std::time::Instant::now();
        let n_chunks = t_pad / gdn_kernel::CS_K;
        let mat_len = h_v * n_chunks * gdn_kernel::CS_K * gdn_kernel::CS_K;
        let ag = self.acquire_buf(mat_len * 4)?;
        let kqg = self.acquire_buf(mat_len * 4)?;
        // SAFETY: kkt 그리드 (CS²/64 큐브, h_v, n_chunks) — pair 가드 j≤i.
        unsafe {
            gdn_kernel::gdn_chunk_kkt::launch_unchecked(
                &self.client,
                CubeCount::Static(
                    (gdn_kernel::CS_K * gdn_kernel::CS_K).div_ceil(64) as u32,
                    h_v as u32,
                    n_chunks as u32,
                ),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(qg.clone(), [1].into(), [qp.len()].into()),
                TensorArg::from_raw_parts(kg.clone(), [1].into(), [kp.len()].into()),
                TensorArg::from_raw_parts(bg.clone(), [1].into(), [bp.len()].into()),
                TensorArg::from_raw_parts(gg.clone(), [1].into(), [gp.len()].into()),
                TensorArg::from_raw_parts(ag.clone(), [1].into(), [mat_len].into()),
                TensorArg::from_raw_parts(kqg.clone(), [1].into(), [mat_len].into()),
                t_len,
                h_k,
                h_v,
                d,
            );
        }
        // SAFETY: 그리드 (h_v,1,1) — 큐브당 헤드, 유닛 64가 dv 분담·가드.
        unsafe {
            gdn_kernel::gdn_chunk::launch_unchecked(
                &self.client,
                CubeCount::Static(h_v as u32, 1, 1),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(vg, [1].into(), [vp.len()].into()),
                TensorArg::from_raw_parts(kg, [1].into(), [kp.len()].into()),
                TensorArg::from_raw_parts(qg, [1].into(), [qp.len()].into()),
                TensorArg::from_raw_parts(bg, [1].into(), [bp.len()].into()),
                TensorArg::from_raw_parts(gg, [1].into(), [gp.len()].into()),
                TensorArg::from_raw_parts(ag.clone(), [1].into(), [mat_len].into()),
                TensorArg::from_raw_parts(kqg.clone(), [1].into(), [mat_len].into()),
                TensorArg::from_raw_parts(sg.clone(), [1].into(), [states.len()].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [outp.len()].into()),
                t_len,
                h_k,
                h_v,
                d,
            );
        }
        self.release_bufs(&[(ag, mat_len * 4), (kqg, mat_len * 4)]);
        acc(&T_LAUNCH, t1.elapsed());
        let t2 = std::time::Instant::now();
        let raw_s = self.client.read_one(sg.clone()).map_err(|e| e.to_string())?;
        let raw_o = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
        acc(&T_READ, t2.elapsed());
        states.copy_from_slice(bytemuck::cast_slice(&raw_s));
        out.copy_from_slice(bytemuck::cast_slice(&raw_o[..out.len() * 4]));
        self.release_bufs(&[(og, outp.len() * 4)]);
        Ok(())
    }

    /// GDN β/e^g 사전 계산 — 값 스타일. 출력 [h·2] 인터리브.
    pub fn gdn_beta_g_gpu(
        &self,
        b: &[f32],
        a: &[f32],
        dtb: &[f32],
        sa: &[f32],
        bg: &mut [f32],
    ) -> Result<(), String> {
        let n_h = b.len();
        let bg_size = n_h * 2;
        let t0 = std::time::Instant::now();
        let bh = self.client.create_from_slice(bytemuck::cast_slice(b));
        let ah = self.client.create_from_slice(bytemuck::cast_slice(a));
        let dh = self.client.create_from_slice(bytemuck::cast_slice(dtb));
        let sh = self.client.create_from_slice(bytemuck::cast_slice(sa));
        let gh = self.acquire_buf(bg_size * 4)?;
        acc(&T_UP, t0.elapsed());
        let t1 = std::time::Instant::now();
        // SAFETY: ABSOLUTE_POS 가드 h<n_h.
        unsafe {
            ew::gdn_beta_g::launch_unchecked(
                &self.client,
                CubeCount::Static(n_h.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(bh, [1].into(), [n_h].into()),
                TensorArg::from_raw_parts(ah, [1].into(), [n_h].into()),
                TensorArg::from_raw_parts(dh, [1].into(), [n_h].into()),
                TensorArg::from_raw_parts(sh, [1].into(), [n_h].into()),
                TensorArg::from_raw_parts(gh.clone(), [1].into(), [bg_size].into()),
                n_h,
            );
        }
        acc(&T_LAUNCH, t1.elapsed());
        let t2 = std::time::Instant::now();
        let raw = self.client.read_one(gh.clone()).map_err(|e| e.to_string())?;
        acc(&T_READ, t2.elapsed());
        bg.copy_from_slice(bytemuck::cast_slice(&raw));
        self.release_bufs(&[(gh, bg_size * 4)]);
        Ok(())
    }

    /// GDN norm_gated silu 게이트 (qwen35) — 값 스타일. w는 [n_h·d] 타일.
    pub fn gdn_norm_gated_silu_gpu(
        &self,
        o: &[f32],
        z: &[f32],
        w: &[f32],
        out: &mut [f32],
        eps: f32,
        d: usize,
    ) -> Result<(), String> {
        let rows = o.len() / d;
        let n_h = w.len() / d;
        let t0 = std::time::Instant::now();
        let oh = self.client.create_from_slice(bytemuck::cast_slice(o));
        let zh = self.client.create_from_slice(bytemuck::cast_slice(z));
        let wh = self.client.create_from_slice(bytemuck::cast_slice(w));
        let og = self.acquire_buf(out.len() * 4)?;
        let pg = self.client.create_from_slice(bytemuck::cast_slice(&[eps]));
        acc(&T_UP, t0.elapsed());
        let t1 = std::time::Instant::now();
        // SAFETY: 큐브당 1행, 유닛 0만 실행.
        unsafe {
            ew::norm_gated_rows_silu::launch_unchecked(
                &self.client,
                CubeCount::Static(rows as u32, 1, 1),
                CubeDim::new_1d(32),
                TensorArg::from_raw_parts(oh, [1].into(), [o.len()].into()),
                TensorArg::from_raw_parts(zh, [1].into(), [z.len()].into()),
                TensorArg::from_raw_parts(wh, [1].into(), [w.len()].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [out.len()].into()),
                TensorArg::from_raw_parts(pg, [1].into(), [1].into()),
                d,
                n_h,
            );
        }
        acc(&T_LAUNCH, t1.elapsed());
        let t2 = std::time::Instant::now();
        let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
        acc(&T_READ, t2.elapsed());
        out.copy_from_slice(bytemuck::cast_slice(&raw));
        self.release_bufs(&[(og, out.len() * 4)]);
        Ok(())
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


    /// 활성 f32 → q8 양자화 (GPU) — rust quantize_row_q8_ref 비트 미러.
    /// 반환: (qs 워드 [n/8], 블록 d [n/32]).
    pub fn quant_q8_gpu(&self, x: &[f32]) -> Result<(Vec<u32>, Vec<f32>), String> {
        let n = x.len();
        if n % 32 != 0 {
            return Err("quant_q8_gpu: n%32 != 0".into());
        }
        let xs = self.client.create_from_slice(bytemuck::cast_slice(x));
        let qg = self.acquire_buf(n / 4)?;
        let dg = self.acquire_buf(n / 32 * 4)?;
        // SAFETY: 그리드가 n/32 블록을 덮음, 유닛당 1블록 순차 — 상한 내.
        unsafe {
            gemm2::quant_q8::launch_unchecked(
                &self.client,
                CubeCount::Static((n / 32).div_ceil(32) as u32, 1, 1),
                CubeDim::new_1d(32),
                TensorArg::from_raw_parts(xs, [1].into(), [n].into()),
                TensorArg::from_raw_parts(qg.clone(), [1].into(), [n / 4].into()),
                TensorArg::from_raw_parts(dg.clone(), [1].into(), [n / 32].into()),
                n,
                127.0f32,
            );
        }
        let qraw = self.client.read_one(qg.clone()).map_err(|e| e.to_string())?;
        let draw = self.client.read_one(dg.clone()).map_err(|e| e.to_string())?;
        self.release_bufs(&[(qg, n / 4), (dg, n / 32 * 4)]);
        Ok((bytemuck::cast_slice(&qraw).to_vec(), bytemuck::cast_slice(&draw).to_vec()))
    }

    /// W4A8 배치 GEMM (iq4_xs, 1≤t≤32) — xs [t][n_in], 반환 [t][n_out] 평탄.
    /// 프리필용: 가중치 로드를 t 토큰이 공유.
    pub fn matmul_w4a8_b_gpu(&self, xs: &[Vec<f32>], w: &Weight) -> Result<Vec<f32>, String> {
        use llm170_gguf::GgmlType;
        if w.ty != GgmlType::Iq4Xs {
            return Err("matmul_w4a8_b_gpu: iq4_xs 전용 (현 단계)".into());
        }
        let t = xs.len();
        if t == 0 {
            return Err("matmul_w4a8_b_gpu: t=0".into());
        }
        // 임의 t: 32행 서브배치로 절단 — 행 독립이라 비트 동일.
        if t > 32 || t & (t - 1) != 0 {
            let mut out: Vec<f32> = Vec::with_capacity(t * w.n_out as usize);
            for ch in xs.chunks(32) {
                let mut sub = ch.to_vec();
                while sub.len() & (sub.len() - 1) != 0 {
                    sub.pop(); // 2의 거듭제곱으로 — 마지막 조각은 개별 처리
                }
                if sub.is_empty() {
                    sub.push(ch[0].clone());
                }
                let r = self.matmul_w4a8_b_gpu(&sub, w)?;
                out.extend_from_slice(&r);
                for extra in &ch[sub.len()..] {
                    let r1 = self.matmul_w4a8_b_gpu(std::slice::from_ref(extra), w)?;
                    out.extend_from_slice(&r1);
                }
            }
            out.truncate(t * w.n_out as usize);
            return Ok(out);
        }
        let d = self.dev_weight(w)?;
        let (n_in, n_out) = d.shape();
        let mut qs_words = Vec::with_capacity(t * n_in / 4);
        let mut ds: Vec<f32> = Vec::with_capacity(t * n_in / 32);
        for row in xs {
            let y = llm170_core::quant::quantize_row_q8_ref(row);
            for c in y.iter().flat_map(|b| b.qs.iter()).collect::<Vec<_>>().chunks(4) {
                let mut word = 0u32;
                for (i, b) in c.iter().enumerate() {
                    word |= (**b as u8 as u32) << (8 * i);
                }
                qs_words.push(word);
            }
            ds.extend(y.iter().map(|b| b.d));
        }
        let xq = self.client.create_from_slice(bytemuck::cast_slice(&qs_words));
        let xd = self.client.create_from_slice(bytemuck::cast_slice(&ds));
        let og = self.acquire_buf(t * n_out * 4)?;
        let pg = self.acquire_buf(t * n_out * 64 * 8)?;
        let gx = n_out.min(65535);
        let gz = n_out.div_ceil(gx);
        let wh = d.gpu()?.clone();
        macro_rules! launch_b {
            ($tl:expr) => {{
                // SAFETY: 그리드 (n_out,1,gz)·가드 — 상한 내.
                unsafe {
                    gemm2::gemm_q8i_b_xs::launch_unchecked(
                        &self.client,
                        CubeCount::Static(gx as u32, 1, gz as u32),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(xq.clone(), [1].into(), [qs_words.len()].into()),
                        TensorArg::from_raw_parts(xd.clone(), [1].into(), [ds.len()].into()),
                        TensorArg::from_raw_parts(wh.clone(), [1].into(), [d.words()].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [t * n_out * 64].into()),
                        TensorArg::from_raw_parts(self.ktab2.clone(), [1].into(), [256].into()),
                        n_in,
                        n_out,
                        gx,
                        $tl,
                    );
                }
            }};
        }
        match t {
            1 => launch_b!(1usize),
            2 => launch_b!(2usize),
            4 => launch_b!(4usize),
            8 => launch_b!(8usize),
            16 => launch_b!(16usize),
            _ => launch_b!(32usize),
        }
        // SAFETY: (gx, t, gz)가 [t·n_out]을 덮음.
        unsafe {
            gemm2::reduce_parts_f64_batch::launch_unchecked(
                &self.client,
                CubeCount::Static(gx as u32, t as u32, gz as u32),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(pg.clone(), [1].into(), [t * n_out * 64].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [t * n_out].into()),
                n_out,
                gx,
                t,
            );
        }
        let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
        self.release_bufs(&[(og, t * n_out * 4), (pg, t * n_out * 64 * 8)]);
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }

    /// W4A8 정수 GEMM (iq4_xs 전용, t=1) — gemm_q8i + reduce_parts_f64.
    /// CPU dot_row_w4a8_iq4xs_lane과 비트 일치. iters>1이면 런치만 반복
    /// (업로드·판독 1회)해 순수 커널 시간 측정을 겸한다. (f64, Vec<f64)> 반환.
    pub fn matmul_w4a8_int_gpu(
        &self,
        x: &[f32],
        w: &Weight,
        iters: usize,
    ) -> Result<(Vec<f32>, std::time::Duration), String> {
        use llm170_core::quant::quantize_row_q8_ref;
        if !llm170_core::matmul::w4a8_ty(w.ty) {
            return Err("matmul_w4a8_int_gpu: 미지원 타입".into());
        }
        let y = quantize_row_q8_ref(x);
        let d = self.dev_weight(w)?;
        let (n_in, n_out) = d.shape();
        let mut qs_words = Vec::with_capacity(n_in.div_ceil(4));
        for c in y.iter().flat_map(|b| b.qs.iter()).collect::<Vec<_>>().chunks(4) {
            let mut word = 0u32;
            for (i, b) in c.iter().enumerate() {
                word |= (**b as u8 as u32) << (8 * i);
            }
            qs_words.push(word);
        }
        let ds: Vec<f32> = y.iter().map(|b| b.d).collect();
        let xq = self.client.create_from_slice(bytemuck::cast_slice(&qs_words));
        let xd = self.client.create_from_slice(bytemuck::cast_slice(&ds));
        let og = self.acquire_buf(n_out * 4)?;
        let pg = self.acquire_buf(n_out * 64 * 8)?;
        let gx = n_out.min(65535);
        let gz = n_out.div_ceil(gx);
        let t0 = std::time::Instant::now();
        for _ in 0..iters.max(1) {
            // SAFETY: 그리드 (n_out,1,gz)·시작부 가드 — 상한 내.
            unsafe {
                if w.ty == llm170_gguf::GgmlType::Q3K {
                    gemm2::gemm_q8i_q3k::launch_unchecked(
                        &self.client,
                        CubeCount::Static(gx as u32, 1, gz as u32),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(xq.clone(), [1].into(), [qs_words.len()].into()),
                        TensorArg::from_raw_parts(xd.clone(), [1].into(), [ds.len()].into()),
                        TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                        n_in,
                        n_out,
                        gx,
                    );
                } else if w.ty == llm170_gguf::GgmlType::Q5K {
                    gemm2::gemm_q8i_q5k::launch_unchecked(
                        &self.client,
                        CubeCount::Static(gx as u32, 1, gz as u32),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(xq.clone(), [1].into(), [qs_words.len()].into()),
                        TensorArg::from_raw_parts(xd.clone(), [1].into(), [ds.len()].into()),
                        TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                        n_in,
                        n_out,
                        gx,
                    );
                } else if w.ty == llm170_gguf::GgmlType::Q4K {
                    gemm2::gemm_q8i_q4k::launch_unchecked(
                        &self.client,
                        CubeCount::Static(gx as u32, 1, gz as u32),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(xq.clone(), [1].into(), [qs_words.len()].into()),
                        TensorArg::from_raw_parts(xd.clone(), [1].into(), [ds.len()].into()),
                        TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                        n_in,
                        n_out,
                        gx,
                    );
                } else if w.ty == llm170_gguf::GgmlType::Q8_0 {
                    gemm2::gemm_q8i_q8_0::launch_unchecked(
                        &self.client,
                        CubeCount::Static(gx as u32, 1, gz as u32),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(xq.clone(), [1].into(), [qs_words.len()].into()),
                        TensorArg::from_raw_parts(xd.clone(), [1].into(), [ds.len()].into()),
                        TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                        n_in,
                        n_out,
                        gx,
                    );
                } else if w.ty == llm170_gguf::GgmlType::Q6K {
                    gemm2::gemm_q8i_q6k::launch_unchecked(
                        &self.client,
                        CubeCount::Static(gx as u32, 1, gz as u32),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(xq.clone(), [1].into(), [qs_words.len()].into()),
                        TensorArg::from_raw_parts(xd.clone(), [1].into(), [ds.len()].into()),
                        TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                        n_in,
                        n_out,
                        gx,
                    );
                } else if w.ty == llm170_gguf::GgmlType::Iq4Nl {
                    gemm2::gemm_q8i_iq4nl::launch_unchecked(
                        &self.client,
                        CubeCount::Static(gx as u32, 1, gz as u32),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(xq.clone(), [1].into(), [qs_words.len()].into()),
                        TensorArg::from_raw_parts(xd.clone(), [1].into(), [ds.len()].into()),
                        TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                        TensorArg::from_raw_parts(self.ktab2.clone(), [1].into(), [256].into()),
                        n_in,
                        n_out,
                        gx,
                    );
                } else {
                    gemm2::gemm_q8i::launch_unchecked(
                        &self.client,
                        CubeCount::Static(gx as u32, 1, gz as u32),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(xq.clone(), [1].into(), [qs_words.len()].into()),
                        TensorArg::from_raw_parts(xd.clone(), [1].into(), [ds.len()].into()),
                        TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                        TensorArg::from_raw_parts(self.ktab2.clone(), [1].into(), [256].into()),
                        n_in,
                        n_out,
                        gx,
                    );
                }
                gemm2::reduce_parts_f64::launch_unchecked(
                    &self.client,
                    CubeCount::Static(gx as u32, 1, gz as u32),
                    CubeDim::new_1d(64),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                    TensorArg::from_raw_parts(og.clone(), [1].into(), [n_out].into()),
                    n_out,
                    gx,
                );
            }
        }
        let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
        let dt = t0.elapsed();
        self.release_bufs(&[(og, n_out * 4), (pg, n_out * 64 * 8)]);
        Ok((bytemuck::cast_slice(&raw).to_vec(), dt))
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
            } else if slices == 64 {
                // 토큰-블록 상각 (PP) — 가중치 1회 디양자화로 16토큰 누산.
                let gy = t.div_ceil(16) as u32;
                gemm2::gemm_q7::launch_unchecked(
                    &self.client,
                    CubeCount::Static(gx as u32, gy, gz as u32),
                    CubeDim::new_1d(64),
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

    /// gpu-ew-check — ew 커널 전종 GPU↔CPU 상호검증 (합성, LCG 고정).
    /// 반환: (커널명, max_rel, 비트일치율) — norm류는 비트일치 기대(f64 경로),
    /// 활성화류는 libm 차이 수준(≤1e-6) 허용. 판정은 서버측.
    pub fn ew_check(&self) -> Result<Vec<(&'static str, f64, f64, f64)>, String> {
        use llm170_core::ops::{l2_norm, rms_norm, sigmoid, silu};
        let mut seed = 0x1234_5678u64;
        let mut lcg = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        };
        let up = |v: &[f32]| self.client.create_from_slice(bytemuck::cast_slice(v));
        let mut rels: Vec<(&'static str, f64, f64, f64)> = Vec::new();
        let mut trash: Vec<(Handle, usize)> = Vec::new();
        // 비교기: max_rel + max_abs + 비트일치율 — abs는 소폭 출력에서의
        // rel 증폭(libm σ 편차 ~1e-7이 분모 1e-3에서 1e-4로 보이는 것) 판별용.
        let cmp = |name: &'static str, g: &[f32], c: &[f32], rels: &mut Vec<(&'static str, f64, f64, f64)>| {
            let (mut mr, mut ma, mut eq) = (0.0f64, 0.0f64, 0usize);
            for (a, b) in g.iter().zip(c) {
                let d = (*a - *b).abs() as f64;
                ma = ma.max(d);
                mr = mr.max(d / b.abs().max(1e-3) as f64);
                eq += (a.to_bits() == b.to_bits()) as usize;
            }
            rels.push((name, mr, ma, eq as f64 / c.len() as f64));
        };

        // ── 요소별 활성화 ──
        let n = 256usize;
        let src: Vec<f32> = (0..n).map(|_| lcg() * 3.0).collect();
        // silu — in-place: 업로드 핸들을 커널이 직접 변형
        {
            let tg = up(&src);
            unsafe {
                ew::ew_silu::launch_unchecked(
                    &self.client,
                    CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
                    CubeDim::new_1d(64),
                    TensorArg::from_raw_parts(tg.clone(), [1].into(), [n].into()),
                    n,
                );
            }
            let raw = self.client.read_one(tg.clone()).map_err(|e| e.to_string())?;
            let gv: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();
            trash.push((tg, n * 4));
            let cv: Vec<f32> = src.iter().map(|v| silu(*v)).collect();
            cmp("ew_silu", &gv, &cv, &mut rels);
        }
        // sigmoid
        {
            let tg = up(&src);
            unsafe {
                ew::ew_sigmoid::launch_unchecked(
                    &self.client,
                    CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
                    CubeDim::new_1d(64),
                    TensorArg::from_raw_parts(tg.clone(), [1].into(), [n].into()),
                    n,
                );
            }
            let raw = self.client.read_one(tg.clone()).map_err(|e| e.to_string())?;
            let gv: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();
            trash.push((tg, n * 4));
            let cv: Vec<f32> = src.iter().map(|v| sigmoid(*v)).collect();
            cmp("ew_sigmoid", &gv, &cv, &mut rels);
        }
        // silu_div (hc=4)
        {
            let hc = 4.0f32;
            let pg = up(&[hc]);
            let tg = up(&src);
            unsafe {
                ew::ew_silu_div::launch_unchecked(
                    &self.client,
                    CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
                    CubeDim::new_1d(64),
                    TensorArg::from_raw_parts(tg.clone(), [1].into(), [n].into()),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                    n,
                );
            }
            let raw = self.client.read_one(tg.clone()).map_err(|e| e.to_string())?;
            let gv: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();
            trash.push((tg, n * 4));
            trash.push((pg, 4));
            let cv: Vec<f32> = src.iter().map(|v| silu(*v / hc)).collect();
            cmp("ew_silu_div", &gv, &cv, &mut rels);
        }
        // silu_mul (GLU)
        {
            let g_in: Vec<f32> = (0..n).map(|_| lcg() * 2.0).collect();
            let u_in: Vec<f32> = (0..n).map(|_| lcg() * 2.0).collect();
            let gg = up(&g_in);
            let ug = up(&u_in);
            let og = self.acquire_buf(n * 4)?;
            unsafe {
                ew::ew_silu_mul::launch_unchecked(
                    &self.client,
                    CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
                    CubeDim::new_1d(64),
                    TensorArg::from_raw_parts(gg.clone(), [1].into(), [n].into()),
                    TensorArg::from_raw_parts(ug.clone(), [1].into(), [n].into()),
                    TensorArg::from_raw_parts(og.clone(), [1].into(), [n].into()),
                    n,
                );
            }
            let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
            let gv: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();
            trash.push((og, n * 4));
            trash.push((gg, n * 4));
            trash.push((ug, n * 4));
            let cv: Vec<f32> = g_in.iter().zip(&u_in).map(|(g, u)| silu(*g) * u).collect();
            cmp("ew_silu_mul", &gv, &cv, &mut rels);
        }

        // ── norm류 (비트일치 기대) ──
        let (nn, w_reps, rows) = (256usize, 4usize, 8usize);
        let x_in: Vec<f32> = (0..rows * nn).map(|_| lcg()).collect();
        let w_in: Vec<f32> = (0..w_reps * nn).map(|_| lcg() + 1.0).collect();
        let eps = 1e-5f32;
        {
            let xg = up(&x_in);
            let wg = up(&w_in);
            let og = self.acquire_buf(rows * nn * 4)?;
            let pg = up(&[eps]);
            let p64 = self.acquire_buf(rows * 32 * 8)?;
            unsafe {
                ew::rms_rows_part::launch_unchecked(
                    &self.client,
                    CubeCount::Static(rows as u32, 1, 1),
                    CubeDim::new_1d(32),
                    TensorArg::from_raw_parts(xg.clone(), [1].into(), [rows * nn].into()),
                    TensorArg::from_raw_parts(p64.clone(), [1].into(), [rows * 32].into()),
                    nn,
                    32,
                );
                ew::rms_rows_finish::launch_unchecked(
                    &self.client,
                    CubeCount::Static(rows as u32, 1, 1),
                    CubeDim::new_1d(32),
                    TensorArg::from_raw_parts(xg.clone(), [1].into(), [rows * nn].into()),
                    TensorArg::from_raw_parts(wg.clone(), [1].into(), [w_reps * nn].into()),
                    TensorArg::from_raw_parts(p64.clone(), [1].into(), [rows * 32].into()),
                    TensorArg::from_raw_parts(og.clone(), [1].into(), [rows * nn].into()),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                    nn,
                    w_reps,
                    32,
                );
            }
            trash.push((p64, rows * 32 * 8));
            let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
            let gv: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();
            trash.push((og, rows * nn * 4));
            trash.push((xg, rows * nn * 4));
            trash.push((wg, w_reps * nn * 4));
            trash.push((pg, 4));
            let cv: Vec<f32> = (0..rows)
                .flat_map(|r| {
                    rms_norm(
                        &x_in[r * nn..(r + 1) * nn],
                        &w_in[(r % w_reps) * nn..(r % w_reps + 1) * nn],
                        eps,
                    )
                })
                .collect();
            cmp("rms_rows", &gv, &cv, &mut rels);
        }
        // norm_gated
        {
            let (d, n_h) = (64usize, 6usize);
            let o_in: Vec<f32> = (0..rows * d).map(|_| lcg()).collect();
            let z_in: Vec<f32> = (0..rows * d).map(|_| lcg() * 2.0).collect();
            let og_ = self.acquire_buf(rows * d * 4)?;
            let ogt = up(&o_in);
            let zgt = up(&z_in);
            let wgt = up(&w_in[..n_h * d]);
            let pg = up(&[eps]);
            unsafe {
                ew::norm_gated_rows::launch_unchecked(
                    &self.client,
                    CubeCount::Static(rows as u32, 1, 1),
                    CubeDim::new_1d(32),
                    TensorArg::from_raw_parts(ogt.clone(), [1].into(), [rows * d].into()),
                    TensorArg::from_raw_parts(zgt.clone(), [1].into(), [rows * d].into()),
                    TensorArg::from_raw_parts(wgt.clone(), [1].into(), [n_h * d].into()),
                    TensorArg::from_raw_parts(og_.clone(), [1].into(), [rows * d].into()),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                    d,
                    n_h,
                );
            }
            let raw = self.client.read_one(og_.clone()).map_err(|e| e.to_string())?;
            let gv: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();
            trash.push((og_, rows * d * 4));
            trash.push((ogt, rows * d * 4));
            trash.push((zgt, rows * d * 4));
            trash.push((wgt, n_h * d * 4));
            trash.push((pg, 4));
            let cv: Vec<f32> = (0..rows)
                .flat_map(|r| {
                    let nr = rms_norm(&o_in[r * d..(r + 1) * d], &w_in[(r % n_h) * d..(r % n_h + 1) * d], eps);
                    nr.iter()
                        .zip(&z_in[r * d..(r + 1) * d])
                        .map(|(v, zz)| v * sigmoid(*zz))
                        .collect::<Vec<f32>>()
                })
                .collect();
            cmp("norm_gated_rows", &gv, &cv, &mut rels);
        }
        // l2_rows
        {
            let d = 64usize;
            let mut x_l2: Vec<f32> = (0..rows * d).map(|_| lcg()).collect();
            let cv: Vec<f32> = (0..rows)
                .flat_map(|r| l2_norm(&x_l2[r * d..(r + 1) * d].to_vec(), eps))
                .collect();
            let xg = up(&x_l2);
            let pg = up(&[eps]);
            unsafe {
                ew::l2_rows::launch_unchecked(
                    &self.client,
                    CubeCount::Static(rows as u32, 1, 1),
                    CubeDim::new_1d(32),
                    TensorArg::from_raw_parts(xg.clone(), [1].into(), [rows * d].into()),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                    d,
                );
            }
            let raw = self.client.read_one(xg.clone()).map_err(|e| e.to_string())?;
            let gv: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();
            trash.push((xg, rows * d * 4));
            trash.push((pg, 4));
            let _ = &mut x_l2;
            cmp("l2_rows", &gv, &cv, &mut rels);
        }

        // ── hyper-connection ──
        let (t_n, hc, ne) = (3usize, 4usize, 256usize);
        {
            let xn: Vec<f32> = (0..t_n * hc * ne).map(|_| lcg()).collect();
            let gate: Vec<f32> = (0..t_n * hc * ne).map(|_| lcg() * 2.0).collect();
            let xg = up(&xn);
            let gg = up(&gate);
            let og = self.acquire_buf(t_n * ne * 4)?;
            let pg = up(&[hc as f32]);
            unsafe {
                ew::hc_gate_mean::launch_unchecked(
                    &self.client,
                    CubeCount::Static(ne.div_ceil(64) as u32, t_n as u32, 1),
                    CubeDim::new_1d(64),
                    TensorArg::from_raw_parts(xg.clone(), [1].into(), [xn.len()].into()),
                    TensorArg::from_raw_parts(gg.clone(), [1].into(), [gate.len()].into()),
                    TensorArg::from_raw_parts(og.clone(), [1].into(), [t_n * ne].into()),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                    ne,
                    hc,
                );
            }
            let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
            let gv: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();
            trash.push((og, t_n * ne * 4));
            trash.push((xg, xn.len() * 4));
            trash.push((gg, gate.len() * 4));
            trash.push((pg, 4));
            let cv: Vec<f32> = (0..t_n)
                .flat_map(|t| {
                    (0..ne)
                        .map(|i| {
                            let mut acc = 0.0f32;
                            for s in 0..hc {
                                let g = gate[t * hc * ne + s * ne + i];
                                acc += xn[t * hc * ne + s * ne + i] * sigmoid(g);
                            }
                            acc / hc as f32
                        })
                        .collect::<Vec<f32>>()
                })
                .collect();
            cmp("hc_gate_mean", &gv, &cv, &mut rels);
        }
        // hc_combine
        {
            let mut res: Vec<f32> = (0..t_n * hc * ne).map(|_| lcg()).collect();
            let outv: Vec<f32> = (0..t_n * ne).map(|_| lcg()).collect();
            let inj: Vec<f32> = (0..t_n * hc).map(|_| lcg() * 2.0).collect();
            let rg = up(&res);
            let ogt = up(&outv);
            let ig = up(&inj);
            let pg = up(&[hc as f32]);
            let total = t_n * hc * ne;
            unsafe {
                ew::hc_combine::launch_unchecked(
                    &self.client,
                    CubeCount::Static(total.div_ceil(64) as u32, 1, 1),
                    CubeDim::new_1d(64),
                    TensorArg::from_raw_parts(rg.clone(), [1].into(), [total].into()),
                    TensorArg::from_raw_parts(ogt.clone(), [1].into(), [t_n * ne].into()),
                    TensorArg::from_raw_parts(ig.clone(), [1].into(), [t_n * hc].into()),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                    ne,
                    hc,
                    total,
                );
            }
            let raw = self.client.read_one(rg.clone()).map_err(|e| e.to_string())?;
            let gv: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();
            trash.push((rg, total * 4));
            trash.push((ogt, t_n * ne * 4));
            trash.push((ig, t_n * hc * 4));
            trash.push((pg, 4));
            for t in 0..t_n {
                for s in 0..hc {
                    let w = 2.0 * sigmoid(inj[t * hc + s] / hc as f32);
                    for (i, ov) in outv[t * ne..(t + 1) * ne].iter().enumerate() {
                        res[t * hc * ne + s * ne + i] += ov * w;
                    }
                }
            }
            cmp("hc_combine", &gv, &res, &mut rels);
        }

        // ── GDN ──
        {
            let (n_h,) = (48usize,);
            let b_in: Vec<f32> = (0..n_h).map(|_| lcg() * 2.0).collect();
            let a_in: Vec<f32> = (0..n_h).map(|_| lcg() * 3.0).collect();
            let dtb: Vec<f32> = (0..n_h).map(|_| lcg()).collect();
            let sa: Vec<f32> = (0..n_h).map(|_| -lcg().abs() - 0.01).collect();
            let bgg = self.acquire_buf(n_h * 2 * 4)?;
            let bgt = up(&b_in);
            let agt = up(&a_in);
            let dtg = up(&dtb);
            let sag = up(&sa);
            unsafe {
                ew::gdn_beta_g::launch_unchecked(
                    &self.client,
                    CubeCount::Static(n_h.div_ceil(64) as u32, 1, 1),
                    CubeDim::new_1d(64),
                    TensorArg::from_raw_parts(bgt.clone(), [1].into(), [n_h].into()),
                    TensorArg::from_raw_parts(agt.clone(), [1].into(), [n_h].into()),
                    TensorArg::from_raw_parts(dtg.clone(), [1].into(), [n_h].into()),
                    TensorArg::from_raw_parts(sag.clone(), [1].into(), [n_h].into()),
                    TensorArg::from_raw_parts(bgg.clone(), [1].into(), [n_h * 2].into()),
                    n_h,
                );
            }
            let raw = self.client.read_one(bgg.clone()).map_err(|e| e.to_string())?;
            let gv: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();
            trash.push((bgg, n_h * 2 * 4));
            trash.push((bgt, n_h * 4));
            trash.push((agt, n_h * 4));
            trash.push((dtg, n_h * 4));
            trash.push((sag, n_h * 4));
            let cv: Vec<f32> = (0..n_h * 2)
                .map(|i| {
                    let h = i / 2;
                    if i % 2 == 0 {
                        sigmoid(b_in[h])
                    } else {
                        let x = a_in[h] + dtb[h];
                        let sp = if x > 20.0 { x } else { x.exp().ln_1p() };
                        (sp * sa[h]).exp()
                    }
                })
                .collect();
            cmp("gdn_beta_g", &gv, &cv, &mut rels);
        }
        // gdn_conv
        {
            let (ch, k, t_len) = (96usize, 4usize, 5usize);
            let qkv: Vec<f32> = (0..t_len * ch).map(|_| lcg()).collect();
            let cwv: Vec<f32> = (0..ch * k).map(|_| lcg()).collect();
            let mut st: Vec<f32> = (0..(k - 1) * ch).map(|_| lcg()).collect();
            let st_orig = st.clone();
            let qg = up(&qkv);
            let cwg = up(&cwv);
            let sgt = up(&st_orig);
            let og = self.acquire_buf(t_len * ch * 4)?;
            unsafe {
                ew::gdn_conv::launch_unchecked(
                    &self.client,
                    CubeCount::Static(ch as u32, 1, 1),
                    CubeDim::new_1d(32),
                    TensorArg::from_raw_parts(qg.clone(), [1].into(), [t_len * ch].into()),
                    TensorArg::from_raw_parts(cwg.clone(), [1].into(), [ch * k].into()),
                    TensorArg::from_raw_parts(sgt.clone(), [1].into(), [(k - 1) * ch].into()),
                    TensorArg::from_raw_parts(og.clone(), [1].into(), [t_len * ch].into()),
                    ch,
                    k,
                    t_len,
                );
            }
            let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
            let gv: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();
            trash.push((og, t_len * ch * 4));
            trash.push((qg, t_len * ch * 4));
            trash.push((cwg, ch * k * 4));
            trash.push((sgt, (k - 1) * ch * 4));
            let mut cv = vec![0.0f32; t_len * ch];
            for t in 0..t_len {
                for c in 0..ch {
                    let mut sum = cwv[c * k + (k - 1)] * qkv[t * ch + c];
                    for j in 0..k - 1 {
                        sum += cwv[c * k + j] * st[j * ch + c];
                    }
                    let oc = silu(sum);
                    for j in 0..k - 2 {
                        st[j * ch + c] = st[(j + 1) * ch + c];
                    }
                    st[(k - 2) * ch + c] = qkv[t * ch + c];
                    cv[t * ch + c] = oc;
                }
            }
            cmp("gdn_conv", &gv, &cv, &mut rels);
        }

        // ── MoE top10 ──
        {
            let (n_exp, k_sel, t_len) = (64usize, 10usize, 3usize);
            let mut route: Vec<f32> = (0..t_len * n_exp).map(|_| lcg() * 4.0).collect();
            let rg = up(&route);
            let idg = self.acquire_buf(t_len * k_sel * 4)?;
            let wtg = self.acquire_buf(t_len * k_sel * 4)?;
            unsafe {
                ew::moe_top10::launch_unchecked(
                    &self.client,
                    CubeCount::Static(t_len as u32, 1, 1),
                    CubeDim::new_1d(32),
                    TensorArg::from_raw_parts(rg.clone(), [1].into(), [t_len * n_exp].into()),
                    TensorArg::from_raw_parts(idg.clone(), [1].into(), [t_len * k_sel].into()),
                    TensorArg::from_raw_parts(wtg.clone(), [1].into(), [t_len * k_sel].into()),
                    n_exp,
                    k_sel,
                );
            }
            let raw_i = self.client.read_one(idg.clone()).map_err(|e| e.to_string())?;
            let raw_w = self.client.read_one(wtg.clone()).map_err(|e| e.to_string())?;
            let gids: Vec<u32> = bytemuck::cast_slice(&raw_i).to_vec();
            let gws: Vec<f32> = bytemuck::cast_slice(&raw_w).to_vec();
            trash.push((rg, t_len * n_exp * 4));
            trash.push((idg, t_len * k_sel * 4));
            trash.push((wtg, t_len * k_sel * 4));
            // CPU 기준 — moe.rs 40-61
            let (mut ids_ok, mut wmr) = (true, 0.0f64);
            for t in 0..t_len {
                let logits = &mut route[t * n_exp..(t + 1) * n_exp];
                let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut zs = 0.0f32;
                for v in logits.iter_mut() {
                    *v = (*v - mx).exp();
                    zs += *v;
                }
                for v in logits.iter_mut() {
                    *v /= zs;
                }
                let mut idx: Vec<usize> = (0..n_exp).collect();
                idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
                let sel = &idx[..k_sel];
                let mut wsum: f32 = sel.iter().map(|&e| logits[e]).sum();
                wsum = wsum.max(6.103515625e-5);
                for (s, &e) in sel.iter().enumerate() {
                    let w = logits[e] / wsum;
                    if gids[t * k_sel + s] != e as u32 {
                        ids_ok = false;
                    }
                    let d = (gws[t * k_sel + s] - w).abs() as f64;
                    wmr = wmr.max(d / w.abs().max(1e-3) as f64);
                }
            }
            rels.push(("moe_top10_ids", if ids_ok { 0.0 } else { 1.0 }, 0.0, 1.0));
            rels.push(("moe_top10_w", wmr, 0.0, 1.0));
        }

        self.release_bufs(&trash);
        Ok(rels)
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
        // W4A8 — CPU matmul_batch(전 타입 int)와 동일 비트.
        // iq4_xs: 배치 커널. 타 나머지: 행루프 t=1 (배치 커널 확장 전 임시).
        if llm170_core::matmul::w4a8_enabled() && llm170_core::matmul::w4a8_ty(w.ty) {
            if w.ty == llm170_gguf::GgmlType::Iq4Xs {
                let g = self.matmul_w4a8_b_gpu(xs, w)?;
                for (ti, out) in outs.iter_mut().enumerate() {
                    let base = ti * w.n_out as usize;
                    out.copy_from_slice(&g[base..base + w.n_out as usize]);
                }
            } else {
                for (ti, out) in outs.iter_mut().enumerate() {
                    let (g, _) = self.matmul_w4a8_int_gpu(&xs[ti], w, 1)?;
                    out.copy_from_slice(&g);
                }
            }
            return Ok(());
        }
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
        // W4A8 디코드(t=1) — CPU·프레임 경로와 동일 비트
        if llm170_core::matmul::w4a8_enabled() && llm170_core::matmul::w4a8_ty(w.ty) {
            let (g, _dt) = self.matmul_w4a8_int_gpu(x, w, 1)?;
            let n = out.len().min(g.len());
            out[..n].copy_from_slice(&g[..n]);
            return Ok(());
        }
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
    fn gdn_conv(
        &self,
        qkv: &[f32],
        conv_w: &[f32],
        state: &mut [f32],
        out: &mut [f32],
        ch: usize,
        k: usize,
    ) -> Result<(), String> {
        self.gdn_conv_gpu(qkv, conv_w, state, out, ch, k)
    }

    fn gdn_beta_g(
        &self,
        b: &[f32],
        a: &[f32],
        dtb: &[f32],
        sa: &[f32],
        bg: &mut [f32],
    ) -> Result<(), String> {
        self.gdn_beta_g_gpu(b, a, dtb, sa, bg)
    }

    fn gdn_norm_gated_silu(
        &self,
        o: &[f32],
        z: &[f32],
        w: &[f32],
        out: &mut [f32],
        eps: f32,
        d: usize,
    ) -> Result<(), String> {
        self.gdn_norm_gated_silu_gpu(o, z, w, out, eps, d)
    }

    fn gdn_chunk(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        g: &[f32],
        states: &mut [f32],
        out: &mut [f32],
        t_len: usize,
        h_k: usize,
        h_v: usize,
        d: usize,
    ) -> Result<(), String> {
        self.gdn_chunk_gpu(q, k, v, beta, g, states, out, t_len, h_k, h_v, d)
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
            } else if llm170_core::matmul::w4a8_enabled()
                && llm170_core::matmul::w4a8_ty(w.ty)
            {
                if w.ty == llm170_gguf::GgmlType::Iq4Xs {
                    let g = self.matmul_w4a8_b_gpu(xs, w)?;
                    for (ti, out) in outs[i].iter_mut().enumerate() {
                        let base = ti * w.n_out as usize;
                        out.copy_from_slice(&g[base..base + w.n_out as usize]);
                    }
                } else {
                    for (ti, out) in outs[i].iter_mut().enumerate() {
                        let (g, _) = self.matmul_w4a8_int_gpu(&xs[ti], w, 1)?;
                        out.copy_from_slice(&g);
                    }
                }
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
        // K개 출력을 1회 read(vec)로 — read당 ~550µs 고정 채널 왕복이
        // 그룹 크기만큼 중복되던 것을 1회로 (2026-09-02 실측, op당 고정
        // 오버헤드 rows=1에서 0.56ms — 전부 read phase).
        let raws = self.client.read(pairs.iter().map(|p| p.0.clone()).collect());
        for ((((og, pg, ob, pb), d), &i), raw) in pairs
            .iter()
            .zip(devs.iter())
            .zip(idx_gpu.iter())
            .zip(raws.iter())
        {
            let res: &[f32] = bytemuck::cast_slice(raw);
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

    // ─── 프레임(활성화 GPU 상주) — P2-4. 값 경로와 병행, readback 없음. ───

    fn frame_alloc(&self, len: usize) -> Result<u64, String> {
        if len == 0 {
            return Err("frame_alloc: 길이 0".into());
        }
        let h = self.acquire_buf(len * 4)?;
        let id = self.frame_next.fetch_add(1, Ordering::Relaxed);
        self.frames
            .lock()
            .map_err(|_| "frame lock poisoned")?
            .insert(id, (h, len));
        Ok(id)
    }

    fn frame_free(&self, h: u64) -> Result<(), String> {
        let (handle, len) = self
            .frames
            .lock()
            .map_err(|_| "frame lock poisoned")?
            .remove(&h)
            .ok_or_else(|| format!("frame_free: 알 수 없는 핸들 {h}"))?;
        self.release_bufs(&[(handle, len * 4)]);
        Ok(())
    }

    /// 기록 = 신규 업로드 + 레지스트리 교체 (cubecl에 기존 핸들 write API 없음).
    /// 구 핸들은 풀 반납 — 해제 경로 없음 (ADR-0014).
    fn frame_write_u32(&self, h: u64, data: &[u32]) -> Result<(), String> {
        let (old, old_len) = self.frame_get(h)?;
        if old_len != data.len() {
            return Err(format!(
                "frame_write_u32: 길이 불일치 핸들={old_len} 데이터={}",
                data.len()
            ));
        }
        let new = self.client.create_from_slice(bytemuck::cast_slice(data));
        self.frames
            .lock()
            .map_err(|_| "frame lock poisoned")?
            .insert(h, (new, data.len()));
        self.release_bufs(&[(old, old_len * 4)]);
        Ok(())
    }
    fn frame_write(&self, h: u64, data: &[f32]) -> Result<(), String> {
        let (old, old_len) = self.frame_get(h)?;
        if old_len != data.len() {
            return Err(format!(
                "frame_write: 길이 불일치 핸들={old_len} 데이터={}",
                data.len()
            ));
        }
        let new = self.client.create_from_slice(bytemuck::cast_slice(data));
        self.frames
            .lock()
            .map_err(|_| "frame lock poisoned")?
            .insert(h, (new, data.len()));
        self.release_bufs(&[(old, old_len * 4)]);
        Ok(())
    }

    fn frame_read(&self, h: u64, out: &mut [f32]) -> Result<(), String> {
        let (handle, len) = self.frame_get(h)?;
        if len != out.len() {
            return Err(format!("frame_read: 길이 불일치 핸들={len} out={}", out.len()));
        }
        let raw = self.client.read_one(handle).map_err(|e| e.to_string())?;
        out.copy_from_slice(bytemuck::cast_slice(&raw));
        Ok(())
    }

    fn frame_mm(&self, x: u64, w: &Weight, out: u64, t: usize) -> Result<(), String> {

        let d = self.dev_weight(w)?;
        if d.is_host() {
            return Err("frame_mm: 호스트 폴백 가중치 (W_CAP 초과)".into());
        }
        let (xh, xlen) = self.frame_get(x)?;
        let (oh, olen) = self.frame_get(out)?;
        let n_in = d.shape().0;
        let n_out = d.shape().1;
        if xlen != t * n_in || olen != t * n_out {
            return Err(format!(
                "frame_mm: 형상 불일치 x={xlen} (t·n_in={}) out={olen} (t·n_out={})",
                t * n_in,
                t * n_out
            ));
        }
        // SAFETY: launch_gemm과 동일 그리드/가드 — 입출력 핸들은 프레임 소유.
        unsafe { self.launch_gemm_into(&d, xh, t, oh)? };
        N_OP.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn frame_mm_group(&self, x: u64, ws: &[Weight], outs: &[u64], t: usize) -> Result<(), String> {

        if ws.len() != outs.len() {
            return Err(format!("frame_mm_group: ws({}) != outs({})", ws.len(), outs.len()));
        }
        for (w, &o) in ws.iter().zip(outs.iter()) {
            self.frame_mm(x, w, o, t)?;
        }
        Ok(())
    }

    fn frame_op(&self, op: &llm170_core::matmul::FrameOp) -> Result<(), String> {
        use llm170_core::matmul::FrameOp;
        let t_op0 = std::time::Instant::now();
        let r = self.frame_op_inner(op);
        acc(&T_FOP, t_op0.elapsed());
        N_FOP.fetch_add(1, Ordering::Relaxed);
        r
    }

    fn frame_quant_q8(&self, src: u64, xq: u64, xd: u64, n: usize) -> Result<(), String> {
        if n % 32 != 0 {
            return Err("frame_quant_q8: n%32 != 0".into());
        }
        let (sh, _) = self.frame_get(src)?;
        let (qh, qlen) = self.frame_get(xq)?;
        if qlen != n / 4 {
            return Err(format!("frame_quant_q8: xq 길이 {qlen} != {}", n / 4));
        }
        let (dh, dlen) = self.frame_get(xd)?;
        if dlen != n / 32 {
            return Err(format!("frame_quant_q8: xd 길이 {dlen} != {}", n / 32));
        }
        // SAFETY: 그리드 n/32 블록 커버·유닛당 순차 — 상한 내.
        unsafe {
            gemm2::quant_q8::launch_unchecked(
                &self.client,
                CubeCount::Static((n / 32).div_ceil(32) as u32, 1, 1),
                CubeDim::new_1d(32),
                TensorArg::from_raw_parts(sh, [1].into(), [n].into()),
                TensorArg::from_raw_parts(qh, [1].into(), [n / 4].into()),
                TensorArg::from_raw_parts(dh, [1].into(), [n / 32].into()),
                n,
                127.0f32,
            );
        }
        N_OP.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn frame_mm_q8(&self, xq: u64, xd: u64, w: &Weight, out: u64, n: usize) -> Result<(), String> {
        use llm170_gguf::GgmlType;
        if !llm170_core::matmul::w4a8_ty(w.ty) {
            return Err("frame_mm_q8: 미지원 타입".into());
        }
        let d = self.dev_weight(w)?;
        if d.is_host() {
            return Err("frame_mm_q8: 호스트 폴백 가중치".into());
        }
        let (xqh, xqlen) = self.frame_get(xq)?;
        let (xdh, xdlen) = self.frame_get(xd)?;
        let (oh, olen) = self.frame_get(out)?;
        let (n_in, n_out) = d.shape();
        if xqlen != n_in / 4 || xdlen != n_in / 32 || olen != n_out {
            return Err(format!(
                "frame_mm_q8: 형상 불일치 xq={xqlen}/{} xd={xdlen}/{} out={olen}/{n_out} n={n}",
                n_in / 4,
                n_in / 32
            ));
        }
        let pg = self.acquire_buf(n_out * 64 * 8)?;
        let gx = n_out.min(65535);
        let gz = n_out.div_ceil(gx);
        // SAFETY: gemm_q8i(_q3k)·reduce와 동일 그리드/가드.
        unsafe {
            if w.ty == GgmlType::Q3K {
                gemm2::gemm_q8i_q3k::launch_unchecked(
                    &self.client,
                    CubeCount::Static(gx as u32, 1, gz as u32),
                    CubeDim::new_1d(64),
                    TensorArg::from_raw_parts(xqh, [1].into(), [xqlen].into()),
                    TensorArg::from_raw_parts(xdh, [1].into(), [xdlen].into()),
                    TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                    n_in,
                    n_out,
                    gx,
                );
            } else if w.ty == GgmlType::Q6K {
                gemm2::gemm_q8i_q6k::launch_unchecked(
                    &self.client,
                    CubeCount::Static(gx as u32, 1, gz as u32),
                    CubeDim::new_1d(64),
                    TensorArg::from_raw_parts(xqh, [1].into(), [xqlen].into()),
                    TensorArg::from_raw_parts(xdh, [1].into(), [xdlen].into()),
                    TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                    n_in,
                    n_out,
                    gx,
                );
            } else if w.ty == GgmlType::Q5K || w.ty == GgmlType::Q4K || w.ty == GgmlType::Q8_0 {
                if w.ty == GgmlType::Q4K {
                    gemm2::gemm_q8i_q4k::launch_unchecked(
                        &self.client,
                        CubeCount::Static(gx as u32, 1, gz as u32),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(xqh, [1].into(), [xqlen].into()),
                        TensorArg::from_raw_parts(xdh, [1].into(), [xdlen].into()),
                        TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                        n_in,
                        n_out,
                        gx,
                    );
                } else if w.ty == GgmlType::Q8_0 {
                    gemm2::gemm_q8i_q8_0::launch_unchecked(
                        &self.client,
                        CubeCount::Static(gx as u32, 1, gz as u32),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(xqh, [1].into(), [xqlen].into()),
                        TensorArg::from_raw_parts(xdh, [1].into(), [xdlen].into()),
                        TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                        n_in,
                        n_out,
                        gx,
                    );
                } else {
                    gemm2::gemm_q8i_q5k::launch_unchecked(
                        &self.client,
                        CubeCount::Static(gx as u32, 1, gz as u32),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(xqh, [1].into(), [xqlen].into()),
                        TensorArg::from_raw_parts(xdh, [1].into(), [xdlen].into()),
                        TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                        n_in,
                        n_out,
                        gx,
                    );
                }
            } else if w.ty == GgmlType::Iq4Nl {
                gemm2::gemm_q8i_iq4nl::launch_unchecked(
                    &self.client,
                    CubeCount::Static(gx as u32, 1, gz as u32),
                    CubeDim::new_1d(64),
                    TensorArg::from_raw_parts(xqh, [1].into(), [xqlen].into()),
                    TensorArg::from_raw_parts(xdh, [1].into(), [xdlen].into()),
                    TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                    TensorArg::from_raw_parts(self.ktab2.clone(), [1].into(), [256].into()),
                    n_in,
                    n_out,
                    gx,
                );
            } else {
                gemm2::gemm_q8i::launch_unchecked(
                    &self.client,
                    CubeCount::Static(gx as u32, 1, gz as u32),
                    CubeDim::new_1d(64),
                    TensorArg::from_raw_parts(xqh, [1].into(), [xqlen].into()),
                    TensorArg::from_raw_parts(xdh, [1].into(), [xdlen].into()),
                    TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [d.words()].into()),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                    TensorArg::from_raw_parts(self.ktab2.clone(), [1].into(), [256].into()),
                    n_in,
                    n_out,
                    gx,
                );
            }
            gemm2::reduce_parts_f64::launch_unchecked(
                &self.client,
                CubeCount::Static(gx as u32, 1, gz as u32),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(pg.clone(), [1].into(), [n_out * 64].into()),
                TensorArg::from_raw_parts(oh, [1].into(), [n_out].into()),
                n_out,
                gx,
            );
        }
        self.release_bufs(&[(pg, n_out * 64 * 8)]);
        N_OP.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl<R: Runtime> GpuMatmul<R> {
    fn frame_op_inner(&self, op: &llm170_core::matmul::FrameOp) -> Result<(), String> {
        use llm170_core::matmul::FrameOp;

        let one = |h: u64| self.frame_get(h).map(|v| v.0);
        let aux = |v: &[f32]| -> Result<Handle, String> { Ok(self.client.create_from_slice(bytemuck::cast_slice(v))) };
        match op {
            FrameOp::SiluDiv { t, div, n } => {
                let (th, tl) = self.frame_get(*t)?;
                if tl != *n {
                    return Err("SiluDiv: 길이 불일치".into());
                }
                let pg = aux(&[*div])?;
                // SAFETY: ABSOLUTE_POS 가드 — 범위 내.
                unsafe {
                    ew::ew_silu_div::launch_unchecked(
                        &self.client,
                        CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(th, [1].into(), [*n].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                        *n,
                    );
                }
                self.release_bufs(&[(pg, 4)]);
            }
            FrameOp::SiluMul { g, u, out, n } => {
                let gh = one(*g)?;
                let uh = one(*u)?;
                let oh = one(*out)?;
                // SAFETY: ABSOLUTE_POS 가드.
                unsafe {
                    ew::ew_silu_mul::launch_unchecked(
                        &self.client,
                        CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(gh, [1].into(), [*n].into()),
                        TensorArg::from_raw_parts(uh, [1].into(), [*n].into()),
                        TensorArg::from_raw_parts(oh, [1].into(), [*n].into()),
                        *n,
                    );
                }
            }
            FrameOp::Sigmoid { t, n } => {
                let th = one(*t)?;
                // SAFETY: ABSOLUTE_POS 가드.
                unsafe {
                    ew::ew_sigmoid::launch_unchecked(
                        &self.client,
                        CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(th, [1].into(), [*n].into()),
                        *n,
                    );
                }
            }
            FrameOp::RmsRows { x, w, out, eps, n, w_reps } => {
                let xh = one(*x)?;
                let wh = one(*w)?;
                let oh = one(*out)?;
                let rows = self.frame_get(*x)?.1 / n;
                let pg = aux(&[*eps])?;
                // 2-커널 세그먼트 구조 (2026-09-02 P0): 32유닛 병렬 부분합
                // (f64) → 순차 결합. 단일 유닛 순차 체인이 447µs/행이던
                // 근원 해소 — CPU ops.rs sq_sum과 동일 순서(비트 계약).
                const SEG: usize = 32;
                let p64 = self.acquire_buf(rows * SEG * 8)?;
                let p64b = p64.clone();
                // SAFETY: part는 유닛별 세그먼트(경계 가드), finish는 유닛 0만.
                unsafe {
                    ew::rms_rows_part::launch_unchecked(
                        &self.client,
                        CubeCount::Static(rows as u32, 1, 1),
                        CubeDim::new_1d(SEG as u32),
                        TensorArg::from_raw_parts(xh.clone(), [1].into(), [rows * n].into()),
                        TensorArg::from_raw_parts(p64.clone(), [1].into(), [rows * SEG].into()),
                        *n,
                        SEG,
                    );
                    ew::rms_rows_finish::launch_unchecked(
                        &self.client,
                        CubeCount::Static(rows as u32, 1, 1),
                        CubeDim::new_1d(SEG as u32),
                        TensorArg::from_raw_parts(xh, [1].into(), [rows * n].into()),
                        TensorArg::from_raw_parts(wh, [1].into(), [w_reps * n].into()),
                        TensorArg::from_raw_parts(p64, [1].into(), [rows * SEG].into()),
                        TensorArg::from_raw_parts(oh, [1].into(), [rows * n].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                        *n,
                        *w_reps,
                        SEG,
                    );
                }
                self.release_bufs(&[(pg, 4), (p64b, rows * SEG * 8)]);
            }
            FrameOp::NormGated { o, z, w, out, eps, d, n_h } => {
                let ohh = one(*o)?;
                let zh = one(*z)?;
                let wh = one(*w)?;
                let outh = one(*out)?;
                let rows = self.frame_get(*o)?.1 / d;
                let pg = aux(&[*eps])?;
                // SAFETY: 큐브당 1행, 유닛 0만 실행.
                unsafe {
                    ew::norm_gated_rows::launch_unchecked(
                        &self.client,
                        CubeCount::Static(rows as u32, 1, 1),
                        CubeDim::new_1d(32),
                        TensorArg::from_raw_parts(ohh, [1].into(), [rows * d].into()),
                        TensorArg::from_raw_parts(zh, [1].into(), [rows * d].into()),
                        TensorArg::from_raw_parts(wh, [1].into(), [n_h * d].into()),
                        TensorArg::from_raw_parts(outh, [1].into(), [rows * d].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                        *d,
                        *n_h,
                    );
                }
                self.release_bufs(&[(pg, 4)]);
            }
            FrameOp::NormGatedSilu { o, z, w, out, eps, d, n_h } => {
                let ohh = one(*o)?;
                let zh = one(*z)?;
                let wh = one(*w)?;
                let outh = one(*out)?;
                let rows = self.frame_get(*o)?.1 / d;
                let pg = aux(&[*eps])?;
                // SAFETY: 큐브당 1행, 유닛 0만 실행.
                unsafe {
                    ew::norm_gated_rows_silu::launch_unchecked(
                        &self.client,
                        CubeCount::Static(rows as u32, 1, 1),
                        CubeDim::new_1d(32),
                        TensorArg::from_raw_parts(ohh, [1].into(), [rows * d].into()),
                        TensorArg::from_raw_parts(zh, [1].into(), [rows * d].into()),
                        TensorArg::from_raw_parts(wh, [1].into(), [n_h * d].into()),
                        TensorArg::from_raw_parts(outh, [1].into(), [rows * d].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                        *d,
                        *n_h,
                    );
                }
                self.release_bufs(&[(pg, 4)]);
            }
            FrameOp::AttnQPrep { q, w, cs, out, eps, hd, pos, half } => {
                let qh = one(*q)?;
                let wh = one(*w)?;
                let ch = one(*cs)?;
                let oh = one(*out)?;
                let n_head = self.frame_get(*q)?.1 / (2 * hd);
                let pg = aux(&[*eps])?;
                // SAFETY: 큐브당 1헤드, 유닛 0만 실행 — 경계는 hd 내부 산술.
                unsafe {
                    ew::attn_q_prep::launch_unchecked(
                        &self.client,
                        CubeCount::Static(n_head as u32, 1, 1),
                        CubeDim::new_1d(32),
                        TensorArg::from_raw_parts(qh, [1].into(), [n_head * 2 * hd].into()),
                        TensorArg::from_raw_parts(wh, [1].into(), [hd].into()),
                        TensorArg::from_raw_parts(ch, [1].into(), [usize::MAX].into()),
                        TensorArg::from_raw_parts(oh, [1].into(), [n_head * 2 * hd].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                        *hd,
                        *pos,
                        *half,
                    );
                }
                self.release_bufs(&[(pg, 4)]);
            }
            FrameOp::AttnKPrep { k, w, cs, cache, eps, hd, pos, n_kv, half } => {
                let kh = one(*k)?;
                let wh = one(*w)?;
                let ch = one(*cs)?;
                let cah = one(*cache)?;
                let pg = aux(&[*eps])?;
                // SAFETY: 큐브당 1 kv-헤드, 유닛 0만 — 캐시 오프셋 산술 경계.
                unsafe {
                    ew::attn_k_prep::launch_unchecked(
                        &self.client,
                        CubeCount::Static(*n_kv as u32, 1, 1),
                        CubeDim::new_1d(32),
                        TensorArg::from_raw_parts(kh, [1].into(), [n_kv * hd].into()),
                        TensorArg::from_raw_parts(wh, [1].into(), [hd].into()),
                        TensorArg::from_raw_parts(ch, [1].into(), [usize::MAX].into()),
                        TensorArg::from_raw_parts(cah, [1].into(), [usize::MAX].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                        *hd,
                        *pos,
                        *n_kv,
                        *half,
                    );
                }
                self.release_bufs(&[(pg, 4)]);
            }
            FrameOp::L2Rows { x, eps, d } => {
                let xh = one(*x)?;
                let rows = self.frame_get(*x)?.1 / d;
                let pg = aux(&[*eps])?;
                // SAFETY: 큐브당 1행, 유닛 0만 실행.
                unsafe {
                    ew::l2_rows::launch_unchecked(
                        &self.client,
                        CubeCount::Static(rows as u32, 1, 1),
                        CubeDim::new_1d(32),
                        TensorArg::from_raw_parts(xh, [1].into(), [rows * d].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                        *d,
                    );
                }
                self.release_bufs(&[(pg, 4)]);
            }
            FrameOp::QKNormRope { q, k, qw, kw, cs, eps, kqs, pos, n_head, n_kv, hd, n_rot } => {
                let qh = one(*q)?;
                let kh = one(*k)?;
                let qwh = one(*qw)?;
                let kwh = one(*kw)?;
                let csh = one(*cs)?;
                let rows = n_head + n_kv;
                // SAFETY: 큐브당 1행, 유닛 0만 실행 — 범위 내.
                unsafe {
                    ew::qk_norm_rope::launch_unchecked(
                        &self.client,
                        CubeCount::Static(rows as u32, 1, 1),
                        CubeDim::new_1d(32),
                        TensorArg::from_raw_parts(qh, [1].into(), [n_head * 2 * hd].into()),
                        TensorArg::from_raw_parts(kh, [1].into(), [n_kv * hd].into()),
                        TensorArg::from_raw_parts(qwh, [1].into(), [n_head * hd].into()),
                        TensorArg::from_raw_parts(kwh, [1].into(), [n_kv * hd].into()),
                        TensorArg::from_raw_parts(csh, [1].into(), [2048 * (n_rot / 2) * 2].into()),
                        *eps,
                        *kqs,
                        *pos,
                        *n_head,
                        *n_kv,
                        *hd,
                        *n_rot,
                    );
                }
            }
            FrameOp::HcGateMean { xn, gate, out, hc, n } => {
                let xh = one(*xn)?;
                let gh = one(*gate)?;
                let oh = one(*out)?;
                let t = self.frame_get(*xn)?.1 / (hc * n);
                let pg = aux(&[*hc as f32])?;
                // SAFETY: 그리드 (n블록, t), 유닛 가드 i<n.
                unsafe {
                    ew::hc_gate_mean::launch_unchecked(
                        &self.client,
                        CubeCount::Static(n.div_ceil(64) as u32, t as u32, 1),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(xh, [1].into(), [t * hc * n].into()),
                        TensorArg::from_raw_parts(gh, [1].into(), [t * hc * n].into()),
                        TensorArg::from_raw_parts(oh, [1].into(), [t * n].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                        *n,
                        *hc,
                    );
                }
                self.release_bufs(&[(pg, 4)]);
            }
            FrameOp::HcCombine { res, out, inj, hc, n, total } => {
                let rh = one(*res)?;
                let oh = one(*out)?;
                let ih = one(*inj)?;
                let pg = aux(&[*hc as f32])?;
                // SAFETY: ABSOLUTE_POS 가드.
                unsafe {
                    ew::hc_combine::launch_unchecked(
                        &self.client,
                        CubeCount::Static(total.div_ceil(64) as u32, 1, 1),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(rh, [1].into(), [*total].into()),
                        TensorArg::from_raw_parts(oh, [1].into(), [total / hc].into()),
                        TensorArg::from_raw_parts(ih, [1].into(), [total / (hc * n)].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                        *n,
                        *hc,
                        *total,
                    );
                }
                self.release_bufs(&[(pg, 4)]);
            }
            FrameOp::GdnBetaG { b, a, dtb, sa, bg, n_h } => {
                let bh = one(*b)?;
                let ah = one(*a)?;
                let dh = one(*dtb)?;
                let sh = one(*sa)?;
                let gh = one(*bg)?;
                // SAFETY: ABSOLUTE_POS 가드 h<n_h.
                unsafe {
                    ew::gdn_beta_g::launch_unchecked(
                        &self.client,
                        CubeCount::Static(n_h.div_ceil(64) as u32, 1, 1),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(bh, [1].into(), [*n_h].into()),
                        TensorArg::from_raw_parts(ah, [1].into(), [*n_h].into()),
                        TensorArg::from_raw_parts(dh, [1].into(), [*n_h].into()),
                        TensorArg::from_raw_parts(sh, [1].into(), [*n_h].into()),
                        TensorArg::from_raw_parts(gh, [1].into(), [n_h * 2].into()),
                        *n_h,
                    );
                }
            }
            FrameOp::GdnConv { qkv, cw, state, out, ch, k, t_len } => {
                let qh = one(*qkv)?;
                let chh = one(*cw)?;
                let sh = one(*state)?;
                let oh = one(*out)?;
                // SAFETY: 큐브당 1채널, 유닛 0만 실행 — t 순차 갱신.
                unsafe {
                    ew::gdn_conv::launch_unchecked(
                        &self.client,
                        CubeCount::Static(*ch as u32, 1, 1),
                        CubeDim::new_1d(32),
                        TensorArg::from_raw_parts(qh, [1].into(), [t_len * ch].into()),
                        TensorArg::from_raw_parts(chh, [1].into(), [ch * k].into()),
                        TensorArg::from_raw_parts(sh, [1].into(), [(k - 1) * ch].into()),
                        TensorArg::from_raw_parts(oh, [1].into(), [t_len * ch].into()),
                        *ch,
                        *k,
                        *t_len,
                    );
                }
            }
            FrameOp::MoeTop10 { route, ids, wt, n_exp, k_sel } => {
                let rh = one(*route)?;
                let ih = one(*ids)?;
                let wh = one(*wt)?;
                let t = self.frame_get(*route)?.1 / n_exp;
                // SAFETY: 큐브당 1토큰, 유닛 0만 실행.
                unsafe {
                    ew::moe_top10::launch_unchecked(
                        &self.client,
                        CubeCount::Static(t as u32, 1, 1),
                        CubeDim::new_1d(32),
                        TensorArg::from_raw_parts(rh, [1].into(), [t * n_exp].into()),
                        TensorArg::from_raw_parts(ih, [1].into(), [t * k_sel].into()),
                        TensorArg::from_raw_parts(wh, [1].into(), [t * k_sel].into()),
                        *n_exp,
                        *k_sel,
                    );
                }
            }
            FrameOp::RopeApply { x, cs, pos_base, rows_per_tok, pos_mul, stride, half } => {
                let xh = one(*x)?;
                let csh = one(*cs)?;
                let rows = self.frame_get(*x)?.1 / stride;
                // SAFETY: 그리드 (half블록, rows), 유닛 가드 p<half.
                unsafe {
                    ew::rope_apply::launch_unchecked(
                        &self.client,
                        CubeCount::Static(half.div_ceil(64) as u32, rows as u32, 1),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(xh, [1].into(), [rows * stride].into()),
                        TensorArg::from_raw_parts(csh, [1].into(), [usize::MAX].into()),
                        *pos_base,
                        *rows_per_tok,
                        *pos_mul,
                        *stride,
                        *half,
                    );
                }
            }
            FrameOp::IdxPool { cache, out, first_block, dim, r } => {
                let chh = one(*cache)?;
                let oh = one(*out)?;
                let n_new = self.frame_get(*out)?.1 / dim;
                // SAFETY: 큐브당 1블록, 유닛 가드 i<dim.
                unsafe {
                    ew::idx_pool::launch_unchecked(
                        &self.client,
                        CubeCount::Static(n_new as u32, 1, 1),
                        CubeDim::new_1d(128),
                        TensorArg::from_raw_parts(chh, [1].into(), [usize::MAX].into()),
                        TensorArg::from_raw_parts(oh, [1].into(), [n_new * dim].into()),
                        *first_block,
                        *dim,
                        *r,
                    );
                }
            }
            FrameOp::IdxScores { qr, bk, scores, idx_heads, dim } => {
                let qh = one(*qr)?;
                let bh = one(*bk)?;
                let sh = one(*scores)?;
                let n_blocks = self.frame_get(*scores)?.1;
                // SAFETY: 큐브당 1블록, 유닛 0만 실행.
                unsafe {
                    ew::idx_scores::launch_unchecked(
                        &self.client,
                        CubeCount::Static(n_blocks as u32, 1, 1),
                        CubeDim::new_1d(32),
                        TensorArg::from_raw_parts(qh, [1].into(), [idx_heads * dim].into()),
                        TensorArg::from_raw_parts(bh, [1].into(), [n_blocks * dim].into()),
                        TensorArg::from_raw_parts(sh, [1].into(), [n_blocks].into()),
                        *idx_heads,
                        *dim,
                    );
                }
            }
            FrameOp::Scale { t, s, n } => {
                let th = one(*t)?;
                let pg = aux(&[*s])?;
                // SAFETY: ABSOLUTE_POS 가드.
                unsafe {
                    ew::ew_scale::launch_unchecked(
                        &self.client,
                        CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(th, [1].into(), [*n].into()),
                        TensorArg::from_raw_parts(pg.clone(), [1].into(), [1].into()),
                        *n,
                    );
                }
                self.release_bufs(&[(pg, 4)]);
            }
            FrameOp::CopyRows { src, dst, src_off, dst_off, n } => {
                let sh = one(*src)?;
                let dh = one(*dst)?;
                // SAFETY: ABSOLUTE_POS 가드.
                unsafe {
                    ew::copy_rows::launch_unchecked(
                        &self.client,
                        CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(sh, [1].into(), [usize::MAX].into()),
                        TensorArg::from_raw_parts(dh, [1].into(), [usize::MAX].into()),
                        *src_off,
                        *dst_off,
                        *n,
                    );
                }
            }
            FrameOp::MoeWeightedSum { ys, wt, out, k, n } => {
                let yh = one(*ys)?;
                let wh = one(*wt)?;
                let oh = one(*out)?;
                // SAFETY: 그리드 (n블록), 유닛 가드 i<n.
                unsafe {
                    ew::moe_weighted_sum::launch_unchecked(
                        &self.client,
                        CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(yh, [1].into(), [k * n].into()),
                        TensorArg::from_raw_parts(wh, [1].into(), [*k].into()),
                        TensorArg::from_raw_parts(oh, [1].into(), [*n].into()),
                        *k,
                        *n,
                    );
                }
            }
            FrameOp::AxpyScaled { y, x, s, n } => {
                let yh = one(*y)?;
                let xh = one(*x)?;
                let sh = one(*s)?;
                // SAFETY: ABSOLUTE_POS 가드.
                unsafe {
                    ew::axpy_scaled::launch_unchecked(
                        &self.client,
                        CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
                        CubeDim::new_1d(64),
                        TensorArg::from_raw_parts(yh, [1].into(), [*n].into()),
                        TensorArg::from_raw_parts(xh, [1].into(), [*n].into()),
                        TensorArg::from_raw_parts(sh, [1].into(), [1].into()),
                        *n,
                    );
                }
            }
        }
        Ok(())
    }
}

impl<R: Runtime> llm170_core::matmul::FrameState for GpuMatmul<R> {
    fn frame_gdn_ar(
        &self,
        q_scaled: u64,
        k: u64,
        v: u64,
        beta_ge: u64,
        states: u64,
        out: u64,
        n_seqs: usize,
        h_k: usize,
        h_v: usize,
        d: usize,
    ) -> Result<(), String> {
        let sg = self.frame_get(states)?.0;
        let qg = self.frame_get(q_scaled)?.0;
        let kg = self.frame_get(k)?.0;
        let vg = self.frame_get(v)?.0;
        let bg = self.frame_get(beta_ge)?.0;
        let og = self.frame_get(out)?.0;
        let n_pairs = n_seqs * h_v;
        // SAFETY: 그리드 (n_pairs,1,1) — gdn_ar_gpu와 동일, 입출력 프레임 소유.
        unsafe {
            gdn_kernel::gdn_ar::launch_unchecked(
                &self.client,
                CubeCount::Static(n_pairs as u32, 1, 1),
                CubeDim::new_1d(128),
                TensorArg::from_raw_parts(sg, [1].into(), [usize::MAX].into()),
                TensorArg::from_raw_parts(qg, [1].into(), [usize::MAX].into()),
                TensorArg::from_raw_parts(kg, [1].into(), [usize::MAX].into()),
                TensorArg::from_raw_parts(vg, [1].into(), [usize::MAX].into()),
                TensorArg::from_raw_parts(bg, [1].into(), [usize::MAX].into()),
                TensorArg::from_raw_parts(og, [1].into(), [usize::MAX].into()),
                d,
                h_k * d,
                h_v * d,
                h_v,
                h_k,
            );
        }
        N_OP.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// ck는 호출부가 kq_scale 사전곱 상태로 유지 (append 시점 스케일 —
    /// 값 경로의 사용 시점 곱과 동일 수치).
    fn frame_qsa_attention(
        &self,
        q: u64,
        ck: u64,
        cv: u64,
        mask: u64,
        out: u64,
        _kq_scale: f32,
        n_past: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<(), String> {
        let qg = self.frame_get(q)?.0;
        let ckg = self.frame_get(ck)?.0;
        let cvg = self.frame_get(cv)?.0;
        let mg = self.frame_get(mask)?.0;
        let og = self.frame_get(out)?.0;
        let sg = self.acquire_buf(t * n_head * n_past * 4)?;
        // SAFETY: qsa_attention_inner와 동일 그리드/가드.
        unsafe {
            attn::qsa_score::launch_unchecked(
                &self.client,
                CubeCount::Static(n_past as u32, n_head as u32, t as u32),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(qg.clone(), [1].into(), [usize::MAX].into()),
                TensorArg::from_raw_parts(ckg.clone(), [1].into(), [usize::MAX].into()),
                TensorArg::from_raw_parts(mg.clone(), [1].into(), [usize::MAX].into()),
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
                TensorArg::from_raw_parts(qg, [1].into(), [usize::MAX].into()),
                TensorArg::from_raw_parts(sg.clone(), [1].into(), [t * n_head * n_past].into()),
                TensorArg::from_raw_parts(cvg, [1].into(), [usize::MAX].into()),
                TensorArg::from_raw_parts(og, [1].into(), [t * n_head * hd].into()),
                n_past,
                n_head,
                n_kv,
                hd,
                t,
            );
        }
        self.release_bufs(&[(sg, t * n_head * n_past * 4)]);
        N_OP.fetch_add(2, Ordering::Relaxed);
        Ok(())
    }

    fn frame_moe_gemm(
        &self,
        x: u64,
        ws: &Weight,
        ids: u64,
        out: u64,
        n_expert_stack: usize,
    ) -> Result<(), String> {
        let d = self.dev_weight(ws)?;
        if d.is_host() {
            return Err("frame_moe_gemm: 스택 가중치 호스트 폴백 (예산 초과)".into());
        }
        let (n_in, n_out_full) = d.shape();
        let n_out = n_out_full / n_expert_stack;
        let wtype = d.ty() as u32 as usize;
        let wwords = d.words();
        let exp_bytes = wwords * 4 / n_expert_stack;
        let xh = self.frame_get(x)?.0;
        let eg = self.frame_get(ids)?.0;
        let oh = self.frame_get(out)?.0;
        let k = self.frame_get(x)?.1 / n_in;
        if self.frame_get(out)?.1 != k * n_out {
            return Err("frame_moe_gemm: out 형상 불일치".into());
        }
        let pg = self.acquire_buf(k * n_out * 64 * 4)?;
        let gx = n_out.min(65535);
        let gz = n_out.div_ceil(gx);
        // SAFETY: moe_down_gpu와 동일 그리드/가드 — ids를 GPU 핸들에서 직독.
        unsafe {
            gemm5::gemm_q5::launch_unchecked(
                &self.client,
                CubeCount::Static(gx as u32, k as u32, gz as u32),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(xh, [1].into(), [k * n_in].into()),
                TensorArg::from_raw_parts(d.gpu()?.clone(), [1].into(), [wwords].into()),
                TensorArg::from_raw_parts(pg.clone(), [1].into(), [k * n_out * 64].into()),
                TensorArg::from_raw_parts(eg, [1].into(), [usize::MAX].into()),
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
                TensorArg::from_raw_parts(oh, [1].into(), [k * n_out].into()),
                n_out,
                k,
                gx,
                64,
            );
        }
        self.release_bufs(&[(pg, k * n_out * 64 * 4)]);
        N_OP.fetch_add(2, Ordering::Relaxed);
        Ok(())
    }
}
