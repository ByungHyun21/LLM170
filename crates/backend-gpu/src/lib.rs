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
    let tlen = if t <= 8 { t.max(1).next_power_of_two().min(8) } else { t };
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

#[derive(Clone)]
struct DevWeight {
    h: Handle,
    ty: GgmlType,
    n_in: usize,
    n_out: usize,
    bytes: usize,
}

/// GPU matmul 가속기 — 런타임 제네릭(HIP 또는 wgpu/Vulkan).
/// 가중치는 데이터 포인터 키로 1회 업로드 후 상주. 커널 소스는 런타임 공유 (ADR-0009).
pub struct GpuMatmul<R: Runtime> {
    client: ComputeClient<R>,
    weights: Mutex<HashMap<usize, DevWeight>>,
    /// 크기별 재사용 스크래치/아웃 버퍼 풀(스택) — hipMalloc churn 방지.
    /// 인플라이트 중 재할당 방지: 획득은 pop(비면 alloc), read 동기화 후 반납.
    bufs: Mutex<HashMap<usize, Vec<Handle>>>,
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
        let ktab: Vec<f32> = llm170_core::KVALUES_IQ4NL
            .iter()
            .map(|&v| v as f32)
            .collect();
        let grid3 = llm170_core::IQ3S_GRID;
        let ktab = client.create_from_slice(bytemuck::cast_slice(&ktab));
        let grid3 = client.create_from_slice(bytemuck::cast_slice(&grid3));
        Ok(GpuMatmul {
            client,
            weights: Mutex::new(HashMap::new()),
            bufs: Mutex::new(HashMap::new()),
            ktab,
            grid3,
        })
    }

    /// 풀에서 버퍼 획득 (같은 크기 연속 획득 시 서로 다른 핸들).
    fn acquire_buf(&self, bytes: usize) -> Result<Handle, String> {
        let mut pool = self.bufs.lock().map_err(|_| "buf pool lock poisoned")?;
        if let Some(h) = pool.get_mut(&bytes).and_then(|v| v.pop()) {
            return Ok(h);
        }
        Ok(self.client.empty(bytes))
    }

    /// read 동기화 후 반납 (크기 명시 — Handle은 크기 비공개).
    fn release_bufs(&self, hs: &[(Handle, usize)]) {
        if let Ok(mut pool) = self.bufs.lock() {
            for (h, bytes) in hs {
                pool.entry(*bytes).or_default().push(h.clone());
            }
        }
    }

    fn dev_weight(&self, w: &Weight) -> Result<DevWeight, String> {
        let key = w.data.as_ptr() as usize;
        let mut map = self
            .weights
            .lock()
            .map_err(|_| "weights cache lock poisoned")?;
        if let Some(d) = map.get(&key) {
            return Ok(d.clone());
        }
        // WGSL에 u8이 없어 u32 워드로 운반 (byte() 헬퍼가 언팩). 4바이트 정렬 필수.
        if w.data.len() % 4 != 0 {
            return Err(format!("tensor bytes {} not 4-byte aligned", w.data.len()));
        }
        let h = self.client.create_from_slice(w.data);
        let d = DevWeight {
            h,
            ty: w.ty,
            n_in: w.n_in as usize,
            n_out: w.n_out as usize,
            bytes: w.data.len() / 4,
        };
        map.insert(key, d.clone());
        Ok(d)
    }

    /// W4A8 GPU GEMM — x를 q8로 양자화해 전송, gemm_q6 + reduce. [t][n_out] 플랫.
    pub fn matmul_w4a8_gpu(&self, x: &[f32], w: &Weight) -> Result<Vec<f32>, String> {
        let y = llm170_core::quant::quantize_row_q8_ref(x);
        let t = 1usize; // 단일 벡터(디코드) 변형
        let d = self.dev_weight(w)?;
        let n_in = d.n_in;
        let n_out = d.n_out;
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
                TensorArg::from_raw_parts(d.h.clone(), [1].into(), [d.bytes].into()),
                TensorArg::from_raw_parts(pg.clone(), [1].into(), [t * n_out * 64].into()),
                TensorArg::from_raw_parts(self.ktab.clone(), [1].into(), [16].into()),
                TensorArg::from_raw_parts(self.grid3.clone(), [1].into(), [512].into()),
                n_in,
                n_out,
                t,
                d.ty as u32 as usize,
            );
            gemm2::reduce_parts::launch_unchecked(
                &self.client,
                CubeCount::Static(n_out as u32, t as u32, 1),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(pg.clone(), [1].into(), [t * n_out * 64].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [t * n_out].into()),
                n_out,
                t,
                64,
            );
        }
        let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
        self.release_bufs(&[(og, t * n_out * 4), (pg, t * n_out * 64 * 4)]);
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }

    /// 디버그: 텐서 블록 0 요소별 디양자화 값 (gpu-de).
    pub fn debug_dequant_block(&self, w: &Weight) -> Result<Vec<f32>, String> {
        let d = self.dev_weight(w)?;
        let (blck, _) = d.ty.block_info();
        let n = blck as usize;
        let og = self.client.empty(n * 4);
        // SAFETY: 그리드가 정확히 요소 수만큼 — 범위 가드 내.
        unsafe {
            gemm2::debug_de::launch_unchecked(
                &self.client,
                CubeCount::Static(n as u32, 1, 1),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(d.h.clone(), [1].into(), [d.bytes].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [n].into()),
                TensorArg::from_raw_parts(self.ktab.clone(), [1].into(), [16].into()),
                TensorArg::from_raw_parts(self.grid3.clone(), [1].into(), [512].into()),
                d.ty as u32 as usize,
                n,
            );
        }
        let raw = self.client.read_one(og).map_err(|e| e.to_string())?;
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }

    /// 디버그: 블록 0 원시 바이트 (gpu-de-bytes).
    pub fn debug_block_bytes(&self, w: &Weight) -> Result<Vec<f32>, String> {
        let d = self.dev_weight(w)?;
        let n = 110.min(d.bytes * 4);
        let og = self.client.empty(n * 4);
        // SAFETY: 그리드 = 요소 수, 가드 내.
        unsafe {
            gemm2::debug_bytes::launch_unchecked(
                &self.client,
                CubeCount::Static(n as u32, 1, 1),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(d.h.clone(), [1].into(), [d.bytes].into()),
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
                TensorArg::from_raw_parts(d.h.clone(), [1].into(), [d.bytes].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [n].into()),
                n,
                mode,
            );
        }
        let raw = self.client.read_one(og).map_err(|e| e.to_string())?;
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }

    /// GEMM 런치 (비동기, v2 k-레인) — x는 외부에서 업로드해 전달. 반환: out 핸들.
    fn launch_gemm(&self, d: &DevWeight, xg: Handle, t: usize) -> Result<(Handle, Handle, usize, usize), String> {
        let (n_in, n_out) = (d.n_in, d.n_out);
        // 스크래치 예산 512MiB 내에서 최대 슬라이스 — comptime 특수화 {64,16,4}
        let slices: usize = if t * n_out * 64 * 4 <= 512 << 20 {
            64
        } else if t * n_out * 16 * 4 <= 512 << 20 {
            16
        } else {
            4
        };
        // 디코드(t ≤ 8, t = tlen 패딩됨) — 토큰 상각 q3, 아니면 q2
        let decode = t <= 8 && slices == 64;
        let og = self.acquire_buf(t * n_out * 4)?;
        let pg = self.acquire_buf(t * n_out * slices * 4)?;
        // SAFETY: 두 경로 모두 그리드가 (n_out, t)를 덮고 시작부 범위 가드 —
        // 인덱스 상한 내, 무한루프 없음.
        unsafe {
            if decode {
                gemm2::gemm_q3::launch_unchecked(
                    &self.client,
                    CubeCount::Static(n_out as u32, 1, 1),
                    CubeDim::new_1d(64),
                    TensorArg::from_raw_parts(xg, [1].into(), [t * n_in].into()),
                    TensorArg::from_raw_parts(d.h.clone(), [1].into(), [d.bytes].into()),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [t * n_out * 64].into()),
                    TensorArg::from_raw_parts(self.ktab.clone(), [1].into(), [16].into()),
                    TensorArg::from_raw_parts(self.grid3.clone(), [1].into(), [512].into()),
                    n_in,
                    n_out,
                    t,
                    d.ty as u32 as usize,
                );
            } else {
                let gy = t.div_ceil(4) as u32;
                gemm2::gemm_q2::launch_unchecked(
                    &self.client,
                    CubeCount::Static(n_out as u32, gy, 1),
                    CubeDim::new_2d(64, 4),
                    TensorArg::from_raw_parts(xg, [1].into(), [t * n_in].into()),
                    TensorArg::from_raw_parts(d.h.clone(), [1].into(), [d.bytes].into()),
                    TensorArg::from_raw_parts(pg.clone(), [1].into(), [t * n_out * slices].into()),
                    TensorArg::from_raw_parts(self.ktab.clone(), [1].into(), [16].into()),
                    TensorArg::from_raw_parts(self.grid3.clone(), [1].into(), [512].into()),
                    n_in,
                    n_out,
                    t,
                    d.ty as u32 as usize,
                    slices,
                );
            }
            gemm2::reduce_parts::launch_unchecked(
                &self.client,
                CubeCount::Static(n_out as u32, t as u32, 1),
                CubeDim::new_1d(64),
                TensorArg::from_raw_parts(pg.clone(), [1].into(), [t * n_out * slices].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [t * n_out].into()),
                n_out,
                t,
                slices,
            );
        }
        Ok((og, pg, t * n_out * 4, t * n_out * slices * 4))
    }

    /// GEMM 실행 → [t][n_out] 플랫 결과 (tlen 패딩은 호출부 책임).
    fn run_gemm(&self, d: &DevWeight, xf: &[f32], t: usize) -> Result<Vec<f32>, String> {
        let t0 = std::time::Instant::now();
        let xg = self.client.create_from_slice(bytemuck::cast_slice(xf));
        acc(&T_UP, t0.elapsed());
        let t1 = std::time::Instant::now();
        let (og, pg, ob, pb) = self.launch_gemm(d, xg, t)?;
        acc(&T_LAUNCH, t1.elapsed());
        let t2 = std::time::Instant::now();
        let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
        acc(&T_READ, t2.elapsed());
        self.release_bufs(&[(og, ob), (pg, pb)]);
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }
}

impl<R: Runtime> Accelerator for GpuMatmul<R> {
    fn matmul_batch(
        &self,
        xs: &[Vec<f32>],
        w: &Weight,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        profile_span!("gpu::matmulB");
        // prefill(t > 8) — q2 커널이 토큰 상각 없어 ALU 병목. 공유메모리 타일
        // 커널 전까지 CPU 폴백 (수치 동일: 같은 dequant·순차 누산).
        if xs.len() > 8 {
            llm170_core::matmul::matmul_batch(xs, w, outs);
            return Ok(());
        }
        let t = xs.len();
        let n_in = w.n_in as usize;
        if xs.iter().any(|r| r.len() != n_in) {
            return Err(format!("matmul_batch: x 행 길이 != n_in ({n_in})"));
        }
        let d = self.dev_weight(w)?;
        let mut xf = Vec::with_capacity(t * n_in);
        for r in xs {
            xf.extend_from_slice(r);
        }
        let res = self.run_gemm(&d, &xf, t)?;
        for (ti, out) in outs.iter_mut().enumerate() {
            out.copy_from_slice(&res[ti * d.n_out..(ti + 1) * d.n_out]);
        }
        Ok(())
    }

    fn matmul(&self, x: &[f32], w: &Weight, out: &mut [f32]) -> Result<(), String> {
        profile_span!("gpu::matmul1");
        let d = self.dev_weight(w)?;
        if x.len() != d.n_in {
            return Err(format!("matmul: x 길이 {} != n_in {}", x.len(), d.n_in));
        }
        let res = self.run_gemm(&d, x, 1)?;
        out.copy_from_slice(&res[..d.n_out]);
        Ok(())
    }

    fn matmul_group(
        &self,
        xs: &[Vec<f32>],
        ws: &[Weight],
        outs: &mut [Vec<Vec<f32>>],
    ) -> Result<(), String> {
        profile_span!("gpu::matmulG");
        // prefill(t > 8) — 위와 동일 사유로 CPU 폴백.
        if xs.len() > 8 {
            for (w, out) in ws.iter().zip(outs.iter_mut()) {
                llm170_core::matmul::matmul_batch(xs, w, out);
            }
            return Ok(());
        }
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
        let devs: Vec<DevWeight> =
            ws.iter().map(|w| self.dev_weight(w)).collect::<Result<_, _>>()?;
        for (d, out) in devs.iter().zip(outs.iter()) {
            if out.len() != t || out.iter().any(|r| r.len() != d.n_out) {
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
        let pairs: Vec<_> =
            devs.iter().map(|d| self.launch_gemm(d, xg.clone(), tlen)).collect::<Result<Vec<_>, _>>()?;
        acc(&T_LAUNCH, t1.elapsed());
        let t2 = std::time::Instant::now();
        let mut to_release: Vec<(Handle, usize)> = Vec::with_capacity(pairs.len() * 2);
        for (((og, pg, ob, pb), d), out_rows) in pairs.iter().zip(devs.iter()).zip(outs.iter_mut()) {
            let raw = self.client.read_one(og.clone()).map_err(|e| e.to_string())?;
            let res: &[f32] = bytemuck::cast_slice(&raw);
            for (ti, out) in out_rows.iter_mut().enumerate() {
                out.copy_from_slice(&res[ti * d.n_out..(ti + 1) * d.n_out]);
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
