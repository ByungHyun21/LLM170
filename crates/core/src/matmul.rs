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
    /// 큐 완결 동기화 — 풀 버퍼 재사용 전 비행 중 연산 종료 확정.
    /// read_one가 커널 완결을 보장하지 않는 결함(2026-09-01 실측) 대응.
    fn barrier(&self) {}

    /// outs[t][o] = Σ_i xs[t][i]·W[o,i]
    fn matmul_batch(
        &self,
        xs: &[Vec<f32>],
        w: &Weight,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String>;
    /// out[o] = Σ_i x[i]·W[o,i]
    fn matmul(&self, x: &[f32], w: &Weight, out: &mut [f32]) -> Result<(), String>;
    /// QSA 마스크드 밀집 GQA (GPU 전용 — 기본 미지원).
    #[allow(clippy::too_many_arguments)]
    fn qsa_attention(
        &self,
        _q: &[f32],
        _ck: &[f32],
        _cv: &[f32],
        _mask: &[u32],
        _kq_scale: f32,
        _n_past: usize,
        _n_head: usize,
        _n_kv: usize,
        _hd: usize,
        _t: usize,
    ) -> Result<Vec<f32>, String> {
        Err("qsa_attention: 이 가속기는 미지원".into())
    }

    /// 전문가 down처럼 입력이 가중치마다 다른 1행 짝: outs[i][o] = xs[i]·W_i[o].
    /// 기본 = 개별 실행. GPU 구현은 런치 배치 + 단일 동기화로 파이프라이닝.
    fn matmul_paired(
        &self,
        xs: &[Vec<f32>],
        ws: &[Weight],
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        if ws.len() != xs.len() || ws.len() != outs.len() {
            return Err(format!("matmul_paired: 형상 불일치 ws={} xs={} outs={}", ws.len(), xs.len(), outs.len()));
        }
        for ((x, w), o) in xs.iter().zip(ws.iter()).zip(outs.iter_mut()) {
            let mut tmp = vec![vec![0.0f32; w.n_out as usize]; 1];
            self.matmul_batch(std::slice::from_ref(x), w, &mut tmp)?;
            o.copy_from_slice(&tmp[0]);
        }
        Ok(())
    }

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

/// 단일 벡터 x에 대한 복수 가중치 내적 — thread::scope 1회로 스폰 오버헤드 제거.
/// qwen4exp 디코드: MoE 전문가(10×2+1)·HC(3)마다 개별 matmul 대신 사용.
/// outs[i][o] = Σ_j x[j]·W_i[o,j].
pub fn matmul_multi(x: &[f32], ws: &[Weight], outs: &mut [Vec<f32>]) {
    profile_span!("cpu::matmul_multi");
    debug_assert_eq!(ws.len(), outs.len());
    let offsets: Vec<usize> = ws
        .iter()
        .scan(0usize, |acc, w| {
            let o = *acc;
            *acc += w.n_out as usize;
            Some(o)
        })
        .collect();
    let total: usize = ws.iter().map(|w| w.n_out as usize).sum();
    let nt = n_threads().max(1).min(total.max(1));
    // 행 단위 워크 스틸링: AtomicU64 클레임 — 스레드 간 정적 분할 불필요,
    // 쓰기 경쟁 없음(각 행은 한 스레드만). outs 행 소유권은 unsafe 없이
    // split_at_mut 트리 대신 포인터 유사 안전 패턴: 각 (wi,row)는 유일.
    use std::sync::atomic::{AtomicU64, Ordering};
    let next = AtomicU64::new(0);
    let results: std::sync::Mutex<Vec<(usize, usize, f32)>> = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _t in 0..nt {
            let next_ref = &next;
            let offsets_ref = offsets.as_slice();
            // 각 스레드가 서로 다른 (wi,row)만 씀 — 쓰기 안전성은 클레임 유일성으로 보장.
            // 안전하게 만들기 위해 outs를 스레드 수로 열 우선 분할하는 대신,
            // 전역 행 인덱스 클레임 → 쓰기 대상 슬라이스를 unsafe 없이 얻기 위해
            // std::cell::UnsafeCell 회피: 쓰기는 메인 스레드가 결과 버퍼에 모아두고
            // 조인 후 분산. 간단·안전: 계산만 병렬, 기록은 직렬.
            let results_ref = &results;
            handles.push(scope.spawn(move || {
                let mut scratch: Vec<f32> = Vec::new();
                let mut local: Vec<(usize, usize, f32)> = Vec::new();
                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed) as usize;
                    if idx >= total {
                        break;
                    }
                    let mut wi = 0usize;
                    while wi < ws.len() && idx >= offsets_ref[wi] + ws[wi].n_out as usize {
                        wi += 1;
                    }
                    if wi >= ws.len() {
                        break;
                    }
                    let row = idx - offsets_ref[wi];
                    let w = &ws[wi];
                    let n_in = w.n_in as usize;
                    if scratch.len() != n_in {
                        scratch = vec![0.0f32; n_in];
                    }
                    let blocks = n_in / w.ty.blck_size() as usize;
                    let base = row * blocks * w.ty.type_size() as usize;
                    crate::quant::dequant_row(w.ty, &w.data[base..], 0, w.n_in, &mut scratch);
                    let mut acc = 0.0f32;
                    for i in 0..n_in {
                        acc += x[i] * scratch[i];
                    }
                    local.push((wi, row, acc));
                }
                results_ref.lock().unwrap().extend(local);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
    // 조인 후 기록 (클레임 유일성으로 중복 없음)
    let results = results.into_inner().unwrap();
    for (wi, row, v) in results {
        outs[wi][row] = v;
    }
}
