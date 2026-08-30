//! f32 행렬-벡터/배치 곱 — 무게는 양자화 바이트에서 타일 단위로 디양자화.
//!
//! ggml 텐서 레이아웃: W [ne0=n_in, ne1=n_out] 행 우선 — out[o] = Σ_i x[i]·W[o,i].
//! ADR-0005: GPU 커널이 아닌 CPU 참조 경로. FMA 없는 mul+add (x86-64 기본 타깃은
//! auto-FMA가 없어 자동으로 성립; target-feature 변경 시 재검토 필요 — 주석 유지).

use llm170_gguf::GgmlType;
use llm170_profiler::profile_span;

/// mmap 상의 무게 텐서 참조.
#[derive(Clone, Copy)]
pub struct Weight<'a> {
    pub data: &'a [u8],
    pub ty: GgmlType,
    pub n_in: u64,
    pub n_out: u64,
}

impl<'a> Weight<'a> {
    /// 텐서 전체를 f32 벡터로 펼침 (ne0-빠른 행 우선: 요소 (i, j) @ j*n_in+i).
    pub fn dequant_f32_vec(&self) -> Vec<f32> {
        let n = self.n_in * self.n_out;
        let (blck, bsize) = self.ty.block_info();
        let rows = self.n_out;
        let mut v = vec![0.0f32; n as usize];
        for r in 0..rows {
            let s = r as usize * self.n_in as usize;
            crate::quant::dequant_row(
                self.ty,
                self.data,
                r,
                self.n_in,
                &mut v[s..s + self.n_in as usize],
            );
        }
        let _ = (blck, bsize);
        v
    }
}

/// 가속기(구현체는 backend-gpu) — 런타임 주입. 없으면 CPU 경로.
/// w 는 mmap 바이트 참조: 구현체는 첫 호출 시 데이터 포인터 키로 업로드 캐시.
pub trait Accelerator: Send + Sync {
    /// outs[t][o] = Σ_i xs[t][i]·W[o,i]
    fn matmul_batch(
        &self,
        xs: &[Vec<f32>],
        w: &Weight,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String>;
    /// out[o] = Σ_i x[i]·W[o,i]
    fn matmul(&self, x: &[f32], w: &Weight, out: &mut [f32]) -> Result<(), String>;
    /// 같은 입력 xs를 먹는 프로젝션 그룹: outs[i][t][o] = Σ xs[t]·W_i[o]. 기본 = 개별 실행.
    /// GPU 구현은 x 업로드 1회 + 런치 배치 + 단일 동기화로 파이프라이닝.
    fn matmul_group(
        &self,
        xs: &[Vec<f32>],
        ws: &[Weight],
        outs: &mut [Vec<Vec<f32>>],
    ) -> Result<(), String> {
        if ws.len() != outs.len() {
            return Err(format!("matmul_group: ws({}) != outs({})", ws.len(), outs.len()));
        }
        for (w, out) in ws.iter().zip(outs.iter_mut()) {
            self.matmul_batch(xs, w, out)?;
        }
        Ok(())
    }
}

/// matmul_group 디스패치 — 가속기 없으면 CPU 개별 배치.
pub fn mm_group(
    acc: &Acc,
    xs: &[Vec<f32>],
    ws: &[Weight],
    outs: &mut [Vec<Vec<f32>>],
) -> Result<(), crate::model::ModelError> {
    match acc.as_deref() {
        Some(a) => a.matmul_group(xs, ws, outs).map_err(crate::model::ModelError::Accel),
        None => {
            for (w, out) in ws.iter().zip(outs.iter_mut()) {
                matmul_batch(xs, w, out);
            }
            Ok(())
        }
    }
}

pub type Acc = Option<std::sync::Arc<dyn Accelerator>>;

/// matmul_batch 디스패치 — 가속기 없으면 CPU 스레드 경로.
pub fn mm_batch(
    acc: &Acc,
    xs: &[Vec<f32>],
    w: &Weight,
    outs: &mut [Vec<f32>],
) -> Result<(), crate::model::ModelError> {
    match acc.as_deref() {
        Some(a) => a
            .matmul_batch(xs, w, outs)
            .map_err(crate::model::ModelError::Accel),
        None => Ok(matmul_batch(xs, w, outs)),
    }
}

/// matmul 디스패치.
pub fn mm(
    acc: &Acc,
    x: &[f32],
    w: &Weight,
    out: &mut [f32],
) -> Result<(), crate::model::ModelError> {
    match acc.as_deref() {
        Some(a) => a.matmul(x, w, out).map_err(crate::model::ModelError::Accel),
        None => Ok(matmul(x, w, out)),
    }
}

pub fn n_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(32)
}

/// out[o] = Σ_i x[i]·W[o,i] (단일 토큰). 스레드별 행 슬라이스 소유.
pub fn matmul(x: &[f32], w: &Weight, out: &mut [f32]) {
    profile_span!("cpu::matmul1");
    let n_in = w.n_in as usize;
    let nt = n_threads().max(1).min(out.len());
    let rows_per = out.len().div_ceil(nt);
    let mut chunks: Vec<&mut [f32]> = out.chunks_mut(rows_per).collect();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (lo, ch) in chunks.iter_mut().enumerate() {
            let row0 = lo * rows_per;
            handles.push(scope.spawn(move || {
                let mut scratch = vec![0.0f32; n_in];
                for (r, o) in ch.iter_mut().enumerate() {
                    crate::quant::dequant_row(
                        w.ty,
                        w.data,
                        (row0 + r) as u64,
                        w.n_in,
                        &mut scratch,
                    );
                    let mut acc = 0.0f32;
                    for i in 0..n_in {
                        acc += x[i] * scratch[i];
                    }
                    *o = acc;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
}

/// 배치: outs[t][o] = Σ_i xs[t][i]·W[o,i].
/// 행(o)별로 한 번 디양자화해 B 토큰과 내적 — prefill에서 디양자화 비용 상각.
/// 스레드별 로컬 결과 [T][rows_per] → 조인 후 스캐터 (행 슬라이스 교차 차입 회피).
pub fn matmul_batch(xs: &[Vec<f32>], w: &Weight, outs: &mut [Vec<f32>]) {
    profile_span!("cpu::matmulB");
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let t = xs.len();
    assert_eq!(outs.len(), t);
    let nt = n_threads().max(1).min(n_out);
    let rows_per = n_out.div_ceil(nt);

    let mut locals: Vec<Vec<f32>> = vec![vec![0.0f32; t * rows_per]; nt];
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (g, local) in locals.iter_mut().enumerate() {
            let row0 = g * rows_per;
            handles.push(scope.spawn(move || {
                let mut scratch = vec![0.0f32; n_in];
                let rows = n_out.saturating_sub(row0).min(rows_per);
                for r in 0..rows {
                    crate::quant::dequant_row(
                        w.ty,
                        w.data,
                        (row0 + r) as u64,
                        w.n_in,
                        &mut scratch,
                    );
                    for (ti, x) in xs.iter().enumerate() {
                        let mut acc = 0.0f32;
                        for i in 0..n_in {
                            acc += x[i] * scratch[i];
                        }
                        local[ti * rows_per + r] = acc;
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
    for (g, local) in locals.iter().enumerate() {
        let row0 = g * rows_per;
        let rows = n_out.saturating_sub(row0).min(rows_per);
        for ti in 0..t {
            for r in 0..rows {
                outs[ti][row0 + r] = local[ti * rows_per + r];
            }
        }
    }
}

/// W4A8 변형 단일 벡터 matmul — x를 q8로 양자화해 타입별 정수 내적.
/// 성능 경로: 기준(f32) 대비 활성 양자화 오차 허용 전제.
pub fn matmul_w4a8(x: &[f32], w: &Weight, out: &mut [f32]) {
    profile_span!("cpu::matmul_w4a8");
    use crate::quant::{dot_row_w4a8, quantize_row_q8_ref};
    let n_in = w.n_in as usize;
    let y = quantize_row_q8_ref(x);
    let nt = n_threads().max(1).min(out.len());
    let rows_per = out.len().div_ceil(nt);
    let mut chunks: Vec<&mut [f32]> = out.chunks_mut(rows_per).collect();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (lo, ch) in chunks.iter_mut().enumerate() {
            let row0 = lo * rows_per;
            let y = &y;
            handles.push(scope.spawn(move || {
                for (r, o) in ch.iter_mut().enumerate() {
                    let row = row0 + r;
                    let base = row * (n_in / w.ty.blck_size() as usize) * w.ty.type_size() as usize;
                    *o = dot_row_w4a8(w.ty, &w.data[base..], w.n_in, y);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
}

/// w4a8 폴백용: 블록 1개 f32 디양자화 (미지원 타입).
fn dequant_row_f32(ty: GgmlType, blk: &[u8], out: &mut [f32], n: u64) {
    crate::quant::dequant_row(ty, blk, 0, n, out);
}
