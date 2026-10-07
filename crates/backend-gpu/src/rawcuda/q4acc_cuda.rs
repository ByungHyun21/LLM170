//! Q4 GGUF 값경로 가속기 — rawcuda Q4Cuda 위임(plans/cuda-port.md §1.3 S6,
//! 2026-10-08). VkAcc 선례(plans/84 B)의 값경로 전용 판: FrameHost 미구현
//! (frame_capable 기본 false) → Engine4는 값 경로로 동작하며 모든 GEMV를
//! 호스트 스테이징한다(Weight.data → add_q4k_bytes 등록 1회 → gemv/gemm_host).
//!
//! [수치 계약] q4 커널(q4_dequant_q4k·q4_gemv_q4k·q4_gemm_q4k_m)은 core
//! quant 미러와 비트일치 검증 완료(q4_cuda_probe — cuda_probe q4-*). 따라서
//! 이 가속기의 종단 계약은 **CPU 기준선과 토큰열 동일**이다(infer에서
//! --backend cpu와 같은 greedy 토큰). gemv(q4_gemv_q4k 64레인)와
//! gemm(q4_gemm_q4k_m 16×16 타일)이 서로 다른 커널이지만 둘 다 core
//! dot 순서 미러라 판이 바뀌어도 토큰은 유지된다.
//!
//! [키 등록] Weight 식별은 data 포인터(GGUF 텐서는 엔진 수명 동안 상주 —
//! VkAcc weight_bufs와 동일 판). 최초 matmul 호출에서 add_q4k_bytes로
//! 디바이스 적재, 이후 재사용. 비-Q4K 가중은 core CPU 폴백(VkAcc 선례 —
//! 조용한 폴백이 아니라 ty 게이트가 명시적 계약이다).
//!
//! [도메인] gemv_host는 grid-y 상한(n_out ≤ 65535)이 있어 lm_head 같은
//! 대형 n_out은 gemm_host t=1로 우회한다(gemm은 grid-x라 상한 없음).

use crate::rawcuda::q4_cuda::Q4Cuda;
use llm170_core::matmul::{Accelerator, MatmulHost, Weight};
use llm170_gguf::GgmlType;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Q4AccCuda {
    q4: Mutex<Q4Cuda>,
    /// Weight 식별(data 포인터) → 등록 키.
    keys: Mutex<HashMap<usize, String>>,
    seq: AtomicU64,
    device: String,
}

impl Q4AccCuda {
    pub fn new() -> Result<Self, String> {
        let q4 = Q4Cuda::new()?;
        let device = q4.device_name().to_string();
        Ok(Self {
            q4: Mutex::new(q4),
            keys: Mutex::new(HashMap::new()),
            seq: AtomicU64::new(0),
            device,
        })
    }

    /// Weight 등록(최초 1회) — Q4K만 호출한다(호출부 ty 게이트).
    fn register(&self, w: &Weight) -> Result<String, String> {
        let id = w.data.as_ptr() as usize;
        if let Some(k) = self.keys.lock().get(&id) {
            return Ok(k.clone());
        }
        let k = format!("w{}", self.seq.fetch_add(1, Ordering::Relaxed));
        self.q4
            .lock()
            .add_q4k_bytes(&k, w.data, w.n_in as usize, w.n_out as usize)?;
        self.keys.lock().insert(id, k.clone());
        Ok(k)
    }

    fn q4_supported(w: &Weight) -> bool {
        w.ty == GgmlType::Q4K
            && w.n_in > 0
            && (w.n_in as usize).is_multiple_of(256)
            && w.n_out > 0
            && w.data.len() == w.n_out as usize * (w.n_in as usize / 256) * 144
    }
}

impl MatmulHost for Q4AccCuda {
    fn matmul(&self, x: &[f32], w: &Weight, out: &mut [f32]) -> Result<(), String> {
        if !Self::q4_supported(w) {
            llm170_core::matmul::matmul(x, w, out);
            return Ok(());
        }
        let key = self.register(w)?;
        let n_out = w.n_out as usize;
        let y = {
            let mut q = self.q4.lock();
            // grid-y 상한 회피 — 대형 n_out(lm_head)은 gemm t=1(grid-x).
            if n_out > 65535 {
                q.gemm_host(&key, x)
            } else {
                q.gemv_host(&key, x)
            }
        }?;
        if y.len() != out.len() {
            return Err(format!(
                "q4acc matmul: out {} != {} (n_out={n_out})",
                out.len(),
                y.len()
            ));
        }
        out.copy_from_slice(&y);
        Ok(())
    }

    fn matmul_batch(
        &self,
        xs: &[Vec<f32>],
        w: &Weight,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        if !Self::q4_supported(w) {
            llm170_core::matmul::matmul_batch(xs, w, outs);
            return Ok(());
        }
        let key = self.register(w)?;
        let n_in = w.n_in as usize;
        let n_out = w.n_out as usize;
        if xs.is_empty() {
            for o in outs.iter_mut() {
                o.fill(0.0);
            }
            return Ok(());
        }
        // gemm_host는 [t][n_in] 연속 rows 계약 — 단일 업로드·단일 런치가
        // t회 gemv보다 유리하다(t=1도 동일 경로 — 커널 판만 다르다).
        let mut rows = Vec::with_capacity(xs.len() * n_in);
        for x in xs {
            if x.len() != n_in {
                return Err(format!("q4acc batch: x.len={} != n_in={n_in}", x.len()));
            }
            rows.extend_from_slice(x);
        }
        let y = self.q4.lock().gemm_host(&key, &rows)?;
        if y.len() != outs.len() * n_out {
            return Err(format!(
                "q4acc batch: out {} != {} (t={} n_out={n_out})",
                y.len(),
                outs.len() * n_out,
                xs.len()
            ));
        }
        for (t, o) in outs.iter_mut().enumerate() {
            o.copy_from_slice(&y[t * n_out..(t + 1) * n_out]);
        }
        Ok(())
    }
}

// ── 값경로 미구현 서브트레이트 — 전부 기본값 위임 ──
// EwOps·QsaOps 기본 메서드는 Err → 호출부 CPU 폴백(트레이트 계약 문구),
// FrameHost는 frame_capable 기본 false → Engine4 값 경로로 동작한다(VkAcc
// plans/84 B 판). GraphCapture pre_ready 기본 false, FrameState는 no-op
// 기본. 위임 본체가 없으면 오버헤드도 없다.

impl llm170_core::matmul::EwOps for Q4AccCuda {}
impl llm170_core::matmul::QsaOps for Q4AccCuda {}
impl llm170_core::matmul::FrameHost for Q4AccCuda {}
impl llm170_core::matmul::FrameState for Q4AccCuda {}
impl llm170_core::matmul::GraphCapture for Q4AccCuda {}

// SAFETY: CudaCtx는 프로세스당 1 컨텍스트 원용(rawcuda 전역 관례 —
// exl3_cuda 디코더도 스레드 간 이동한다). Q4AccCuda의 모든 진입은
// Mutex 직렬화라 동시 접근이 없고, 소유 스레드 종료 없이 이동만 한다.
unsafe impl Send for Q4AccCuda {}
unsafe impl Sync for Q4AccCuda {}

/// 값경로 가속기 팩토리 — attach_q4의 cuda 분기가 부른다.
pub fn new_q4_acc_cuda() -> Result<Arc<dyn Accelerator>, String> {
    let acc = Q4AccCuda::new()?;
    let dev = acc.device.clone();
    let arc: Arc<dyn Accelerator> = Arc::new(acc);
    eprintln!("# q4acc-cuda: {dev} (값경로 — Q4K만 오프로드, 나머지 CPU 폴백)");
    Ok(arc)
}
