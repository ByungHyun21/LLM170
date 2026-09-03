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
pub trait Accelerator: FrameState + Send + Sync {
    /// rms_norm 오프로드 — 미구현 백엔드는 Err (호출부 CPU 폴백).
    fn rms_norm(
        &self,
        _xs: &[Vec<f32>],
        _w: &[f32],
        _eps: f32,
        _outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        Err("rms_norm: 미지원".into())
    }

    /// MoE 전문가 배치 down — K전문가 1런치. 미구현은 Err (호출부 폴백).
    /// xs 행 순서 = expert_ids 순 (스택 인덱스와 무관).
    #[allow(clippy::too_many_arguments)]
    fn moe_down(
        &self,
        _xs: &[Vec<f32>],
        _ws: &Weight,
        _expert_ids: &[u32],
        _n_expert_stack: usize,
        _outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        Err("moe_down: 미지원".into())
    }

    /// GDN AR 단일 토큰 상태 갱신 — 미구현 백엔드는 Err (호출부 CPU 폴백).
    #[allow(clippy::too_many_arguments)]
    fn gdn_ar(
        &self,
        _q_scaled: &[f32],
        _k: &[f32],
        _v: &[f32],
        _beta_ge: &[f32],
        _states: &mut [f32],
        _out: &mut [f32],
        _n_seqs: usize,
        _h_k: usize,
        _h_v: usize,
        _d: usize,
    ) -> Result<(), String> {
        Err("gdn_ar: 미지원".into())
    }

    /// GDN depthwise conv + ring (t토큰, 시퀀스 1) — 값 스타일 업/다운로드.
    /// qwen35 디코드 연결용 (02-2). 미지원 백엔드는 Err → 호출부 CPU 폴백.
    fn gdn_conv(
        &self,
        _qkv: &[f32],
        _conv_w: &[f32],
        _state: &mut [f32],
        _out: &mut [f32],
        _ch: usize,
        _k: usize,
    ) -> Result<(), String> {
        Err("gdn_conv: 미지원".into())
    }

    /// GDN β/e^g 사전 계산 → [h·2] 인터리브. 미지원은 Err.
    fn gdn_beta_g(
        &self,
        _b: &[f32],
        _a: &[f32],
        _dtb: &[f32],
        _sa: &[f32],
        _bg: &mut [f32],
    ) -> Result<(), String> {
        Err("gdn_beta_g: 미지원".into())
    }

    /// GDN norm_gated silu 게이트 (qwen35): rms(o)·silu(z)·w. w는 [n_h·d] 타일.
    fn gdn_norm_gated_silu(
        &self,
        _o: &[f32],
        _z: &[f32],
        _w: &[f32],
        _out: &mut [f32],
        _eps: f32,
        _d: usize,
    ) -> Result<(), String> {
        Err("gdn_norm_gated_silu: 미지원".into())
    }

    /// GDN 청크 프리필(t>1) — 값 스타일. q/k는 l2 완료·무스케일.
    #[allow(clippy::too_many_arguments)]
    fn gdn_chunk(
        &self,
        _q: &[f32],
        _k: &[f32],
        _v: &[f32],
        _beta: &[f32],
        _g: &[f32],
        _states: &mut [f32],
        _out: &mut [f32],
        _t_len: usize,
        _h_k: usize,
        _h_v: usize,
        _d: usize,
    ) -> Result<(), String> {
        Err("gdn_chunk: 미지원".into())
    }
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
    // ─── 프레임(활성화 GPU 상주) — 층 전체 상주 P2-4 (plans/gpu-frame.md) ───
    /// 기본 미지원(Err) — 프레임 경로는 구현 가속기에서만 사용하며, 값 반환
    /// 경로(위 matmul 계열)와 병행해 CPU golden 대조가 가능하다.

    /// 프레임 버퍼 할당 — u64는 가속기 레지스트리 토큰 (해제는 frame_free).
    fn frame_alloc(&self, _len: usize) -> Result<u64, String> {
        Err("frame_alloc: 미지원".into())
    }
    /// 프레임 버퍼 반납 (풀 재사용 — 해제 아님, ADR-0014).
    fn frame_free(&self, _h: u64) -> Result<(), String> {
        Err("frame_free: 미지원".into())
    }
    /// 호스트 → 프레임 버퍼 기록.
    /// u32 버퍼 기록 (qsa mask 등 — f32 프레임과 별도 원시 경로).
    fn frame_write_u32(&self, _h: u64, _data: &[u32]) -> Result<(), String> {
        Err("frame_write_u32: 미지원".into())
    }
    fn frame_write(&self, _h: u64, _data: &[f32]) -> Result<(), String> {
        Err("frame_write: 미지원".into())
    }
    /// 프레임 버퍼 → 호스트 판독 (동기 — forward 종료 1회가 설계상 목표).
    fn frame_read(&self, _h: u64, _out: &mut [f32]) -> Result<(), String> {
        Err("frame_read: 미지원".into())
    }
    /// 상주 GEMM: out[t·n_out..] = x[t·n_in..]·W — 업/다운로드 없음.
    fn frame_mm(&self, _x: u64, _w: &Weight, _out: u64, _t: usize) -> Result<(), String> {
        Err("frame_mm: 미지원".into())
    }
    /// 상주 GEMM 그룹 — 동일 입력 x, 가중치별 out.
    fn frame_mm_group(&self, _x: u64, _ws: &[Weight], _outs: &[u64], _t: usize) -> Result<(), String> {
        Err("frame_mm_group: 미지원".into())
    }
    /// 상주 elementwise/RoPE/인덱서 연산 — 커널 선택은 FrameOp 변형.
    fn frame_op(&self, _op: &FrameOp) -> Result<(), String> {
        Err("frame_op: 미지원".into())
    }
    /// 상주 q8 양자화: src(f32) → xq(u32 워드 n/8) + xd(f32 n/32) —
    /// quantize_row_q8_ref 비트 미러.
    fn frame_quant_q8(&self, _src: u64, _xq: u64, _xd: u64, _n: usize) -> Result<(), String> {
        Err("frame_quant_q8: 미지원".into())
    }
    /// 상주 W4A8 정수 GEMV (iq4_xs·q3_K, t=1) — (xq, xd) 소비.
    fn frame_mm_q8(&self, _xq: u64, _xd: u64, _w: &Weight, _out: u64, _n: usize) -> Result<(), String> {
        Err("frame_mm_q8: 미지원".into())
    }
}

/// 프레임 연산 식별 — 백엔드 커널 세트(backend-gpu/src/ew.rs)와 1:1.
/// u64는 전부 프레임 핸들. 수치 계약: CPU 참조(ops.rs·stages)와 동일 순서.
#[derive(Debug, Clone, Copy)]
pub enum FrameOp {
    /// in-place: v ← silu(v/div) — hc 저랭크 lo.
    SiluDiv { t: u64, div: f32, n: usize },
    /// GLU: out = silu(g)·u.
    SiluMul { g: u64, u: u64, out: u64, n: usize },
    /// in-place sigmoid.
    Sigmoid { t: u64, n: usize },
    /// 행별 RMSNorm (w는 w_reps 반복 — hc 그룹/헤드별).
    RmsRows { x: u64, w: u64, out: u64, eps: f32, n: usize, w_reps: usize },
    /// GDN norm_gated: out = rms(o)·σ(z), w 반복 = 헤드.
    NormGated { o: u64, z: u64, w: u64, out: u64, eps: f32, d: usize, n_h: usize },
    /// GDN norm_gated silu 변형 (qwen35): out = rms(o)·silu(z)·w.
    NormGatedSilu { o: u64, z: u64, w: u64, out: u64, eps: f32, d: usize, n_h: usize },
    /// GDN q/k 헤드별 in-place L2 norm.
    L2Rows { x: u64, eps: f32, d: usize },
    /// conv 출력 3분할 (q/k/v) — 카피 3런치 융합.
    Split3 { src: u64, d0: u64, d1: u64, d2: u64, n0: usize, n1: usize, n2: usize },
    /// L2 이중 행 + q 스케일 융합 (산술 l2_rows+scale 과 동일).
    L2Rows2Scale { q: u64, k: u64, eps: f32, scale: f32, d: usize, n_group: usize },
    /// 어텐션 q/k 헤드 rms+rope in-place (f64 중간 — 브리지 제거).
    QKNormRope {
        q: u64, k: u64, qw: u64, kw: u64, cs: u64,
        eps: f32, kqs: f32, pos: usize, n_head: usize, n_kv: usize,
        hd: usize, n_rot: usize,
    },
    /// hc 게이트 적용 + 스트림 평균 (hc는 나눗셈 피수로 사용).
    HcGateMean { xn: u64, gate: u64, out: u64, hc: usize, n: usize },
    /// hc combine: res += out·(2·σ(inj/hc)).
    HcCombine { res: u64, out: u64, inj: u64, hc: usize, n: usize, total: usize },
    /// GDN β/e^g 사전계산: bg[h·2]=σ(b), bg[h·2+1]=e^(softplus(a+dtb)·sa).
    GdnBetaG { b: u64, a: u64, dtb: u64, sa: u64, bg: u64, n_h: usize },
    /// GDN conv1d + ring shift + silu (state in-place).
    GdnConv { qkv: u64, cw: u64, state: u64, out: u64, ch: usize, k: usize, t_len: usize },
    /// MoE route top-k: ids/wt GPU 잔류.
    MoeTop10 { route: u64, ids: u64, wt: u64, n_exp: usize, k_sel: usize },
    /// NEOX RoPE (cs = [pos_max][half][2] cos,sin 인터리브 테이블).
    RopeApply { x: u64, cs: u64, pos_base: usize, rows_per_tok: usize, pos_mul: usize, stride: usize, half: usize },
    /// 인덱서 블록키 풀링 (mean of r rows).
    IdxPool { cache: u64, out: u64, first_block: usize, dim: usize, r: usize },
    /// 인덱서 스코어: Σ_h ReLU(qr·bk).
    IdxScores { qr: u64, bk: u64, scores: u64, idx_heads: usize, dim: usize },
    /// qwen35 어텐션 q 프리페어: 헤드 rms·rope·q‖gate 인터리브.
    AttnQPrep { q: u64, w: u64, cs: u64, out: u64, eps: f32, hd: usize, pos: usize, half: usize },
    /// qwen35 어텐션 k 프리페어: kv-헤드 rms·rope → 캐시 pos append.
    AttnKPrep { k: u64, w: u64, cs: u64, cache: u64, eps: f32, hd: usize, pos: usize, n_kv: usize, half: usize },
    /// in-place: v ← v·s (GDN q 사전 스케일).
    Scale { t: u64, s: f32, n: usize },
    /// 행 복사: dst[dst_off..+n] = src[src_off..+n] — 캐시 append 부품.
    CopyRows { src: u64, dst: u64, src_off: usize, dst_off: usize, n: usize },
    /// MoE shared 가산: y += x·s (s는 1원소 프레임 버퍼).
    AxpyScaled { y: u64, x: u64, s: u64, n: usize },
    /// MoE 전문가 가중 합: out = Σ_e wt[e]·ys[e].
    MoeWeightedSum { ys: u64, wt: u64, out: u64, k: usize, n: usize },
}

/// 프레임 상태 연산 — 상주 상태(kv/gdn/conv/blk)를 갱신하는 가속기 전용
/// 메서드. 값 경로 Accelerator 메서드와 대응하되 입출력이 전부 핸들.
pub trait FrameState {
    /// GDN AR 상태 갱신 (gdn_ar의 프레임 변형) — states·out 상주, 판독 없음.
    #[allow(clippy::too_many_arguments)]
    fn frame_gdn_ar(
        &self,
        _q_scaled: u64,
        _k: u64,
        _v: u64,
        _beta_ge: u64,
        _states: u64,
        _out: u64,
        _n_seqs: usize,
        _h_k: usize,
        _h_v: usize,
        _d: usize,
    ) -> Result<(), String> {
        Err("frame_gdn_ar: 미지원".into())
    }
    /// QSA 마스크드 밀집 GQA (qsa_attention 프레임 변형) — 캐시 상주.
    #[allow(clippy::too_many_arguments)]
    fn frame_qsa_attention(
        &self,
        _q: u64,
        _ck: u64,
        _cv: u64,
        _mask: u64,
        _out: u64,
        _kq_scale: f32,
        _n_past: usize,
        _n_head: usize,
        _n_kv: usize,
        _hd: usize,
        _t: usize,
    ) -> Result<(), String> {
        Err("frame_qsa_attention: 미지원".into())
    }
    /// MoE ids 구동 배치 GEMM — x 상주, ids 상주(GPU top10 출력 직결).
    /// stack은 전문가 스택 전체 뷰. outs는 [k_sel·n_out] 단일 프레임.
    fn frame_moe_gemm(
        &self,
        _x: u64,
        _ws: &Weight,
        _ids: u64,
        _out: u64,
        _n_expert_stack: usize,
    ) -> Result<(), String> {
        Err("frame_moe_gemm: 미지원".into())
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
/// 원시 HIP 디코드 (LLM170_RAWHIP=1) — 백엔드가 상주 DecodeState로
/// 토큰 1스텝 전체를 수행. 엔진은 임베딩 dequant·pos만 제공.
pub trait RawDecode: Send + Sync {
    /// 상태 초기화 (가중치·상수 업로드 1회) — wnames는 필요 텐서명.
    fn raw_init(
        &self,
        hp: &crate::model::hparams::Hparams,
        weights: &[(String, Weight<'_>)],
        consts: &[(String, Vec<f32>)],
        n_seqs: usize,
        ctx_len: usize,
        is_recr: Vec<bool>,
    ) -> Result<(), String>;
    /// 디코드 1스텝 — emb(임베딩 행) 기록 후 전체 층 수행, logits 반환.
    fn raw_step(&self, seq: usize, pos: usize, emb: &[f32]) -> Result<Vec<f32>, String>;
    /// greedy 스텝 — GPU argmax, 토큰만 (logits 전사 회피).
    fn raw_step_greedy(&self, seq: usize, pos: usize, emb: &[f32]) -> Result<u32, String> {
        Ok(greedy_from(&self.raw_step(seq, pos, emb)?))
    }
    /// 프리필 배치 — emb [t][n], 마지막 토큰 logits.
    fn raw_prefill(&self, seq: usize, pos0: usize, emb: &[f32]) -> Result<Vec<f32>, String> {
        let n = emb.len();
        let mut last = None;
        for ti in 0..(n / 512) {
            let _ = ti;
        }
        for ch in emb.chunks(512) {
            last = Some(self.raw_step(seq, pos0, ch)?);
        }
        Ok(last.unwrap_or_default())
    }
}

/// W4A8 정수 GEMV 경로 활성 (LLM170_W4A8=1) — iq4_xs·q3_K 디코드
/// matmul을 레인 f64 미러 정수 내적으로 전환. GPU frame/value 경로와
/// 동일 비트 (그룹핑 무관 설계). 프리필(t>1)은 무관.
pub fn w4a8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LLM170_W4A8").is_some())
}

/// W4A8 대상 타입 (정수 커널·미러 구현 완료분).
pub fn w4a8_ty(ty: llm170_gguf::GgmlType) -> bool {
    matches!(
        ty,
        llm170_gguf::GgmlType::Iq4Xs
            | llm170_gguf::GgmlType::Iq3S
            | llm170_gguf::GgmlType::Q3K
            | llm170_gguf::GgmlType::Q4K
            | llm170_gguf::GgmlType::Q5K
            | llm170_gguf::GgmlType::Q8_0
            | llm170_gguf::GgmlType::Iq4Nl
            | llm170_gguf::GgmlType::Q6K
    )
}

pub fn matmul(x: &[f32], w: &Weight, out: &mut [f32]) {
    profile_span!("cpu::matmul1");
    // W4A8 디코드 전환 — 활성 시 전 경로 동일 비트
    if w4a8_enabled() && w4a8_ty(w.ty) && x.len() == w.n_in as usize {
        let y = crate::quant::quantize_row_q8_ref(x);
        let blck = w.ty.blck_size() as usize;
        let bsize = w.ty.type_size() as usize;
        let row_bytes = (w.n_in as usize / blck) * bsize;
        for (o, out_o) in out.iter_mut().enumerate() {
            let row = &w.data[o * row_bytes..];
            *out_o = match w.ty {
                llm170_gguf::GgmlType::Q3K => {
                    crate::quant::dot_row_w4a8_q3k_lane(row, w.n_in, &y)
                }
                llm170_gguf::GgmlType::Iq3S => {
                    crate::quant::dot_row_w4a8_iq3s_lane(row, w.n_in, &y)
                }
                llm170_gguf::GgmlType::Q4K => {
                    crate::quant::dot_row_w4a8_q4k_lane(row, w.n_in, &y)
                }
                llm170_gguf::GgmlType::Q5K => {
                    crate::quant::dot_row_w4a8_q5k_lane(row, w.n_in, &y)
                }
                llm170_gguf::GgmlType::Q8_0 => {
                    crate::quant::dot_row_w4a8_q8_0_lane(row, w.n_in, &y)
                }
                llm170_gguf::GgmlType::Iq4Nl => {
                    crate::quant::dot_row_w4a8_iq4nl_lane(row, w.n_in, &y)
                }
                llm170_gguf::GgmlType::Q6K => {
                    crate::quant::dot_row_w4a8_q6k_lane(row, w.n_in, &y)
                }
                _ => crate::quant::dot_row_w4a8_iq4xs_lane(row, w.n_in, &y),
            };
        }
        return;
    }
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
    // W4A8 (지원 타입) — 행별 레인 미러 정수 내적 (GPU 배치 경로와 동일 비트)
    if w4a8_enabled() && w4a8_ty(w.ty) {
        let y_all: Vec<_> = xs.iter().map(|r| crate::quant::quantize_row_q8_ref(r)).collect();
        let blck = w.ty.blck_size() as usize;
        let bsize = w.ty.type_size() as usize;
        let row_bytes = (w.n_in as usize / blck) * bsize;
        for (ti, out) in outs.iter_mut().enumerate() {
            let y = &y_all[ti];
            for (o, out_o) in out.iter_mut().enumerate() {
                let row = &w.data[o * row_bytes..];
                *out_o = match w.ty {
                    llm170_gguf::GgmlType::Q3K => {
                        crate::quant::dot_row_w4a8_q3k_lane(row, w.n_in, y)
                    }
                    llm170_gguf::GgmlType::Iq3S => {
                        crate::quant::dot_row_w4a8_iq3s_lane(row, w.n_in, y)
                    }
                    llm170_gguf::GgmlType::Q4K => {
                        crate::quant::dot_row_w4a8_q4k_lane(row, w.n_in, y)
                    }
                    llm170_gguf::GgmlType::Q5K => {
                        crate::quant::dot_row_w4a8_q5k_lane(row, w.n_in, y)
                    }
                    llm170_gguf::GgmlType::Q8_0 => {
                        crate::quant::dot_row_w4a8_q8_0_lane(row, w.n_in, y)
                    }
                    llm170_gguf::GgmlType::Iq4Nl => {
                        crate::quant::dot_row_w4a8_iq4nl_lane(row, w.n_in, y)
                    }
                    llm170_gguf::GgmlType::Q6K => {
                        crate::quant::dot_row_w4a8_q6k_lane(row, w.n_in, y)
                    }
                    _ => crate::quant::dot_row_w4a8_iq4xs_lane(row, w.n_in, y),
                };
            }
        }
        return;
    }

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

/// logits → argmax (greedy와 동일 의미, 트레이트 기본구현용).
pub fn greedy_from(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}
