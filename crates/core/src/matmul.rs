//! f32 행렬-벡터/배치 곱 — 무게는 양자화 바이트에서 타일 단위로 디양자화.
//!
//! ggml 텐서 레이아웃: W [ne0=n_in, ne1=n_out] 행 우선 — out[o] = Σ_i x[i]·W[o,i].
//! ADR-0005: GPU 커널이 아닌 CPU 참조 경로. FMA 없는 mul+add (x86-64 기본 타깃은
//! auto-FMA가 없어 자동으로 성립; target-feature 변경 시 재검토 필요 — 주석 유지).
#![allow(dead_code)] // 프론트 정리(2026-09-14): 레거시·진단 경로 보존

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
    /// 그래프 캡처 시작/종료·재생 — 미지원 백엔드는 Err (드라이버가 폴백).
    fn graph_capture_begin(&self) -> Result<(), String> {
        Err("graph capture: 미지원".into())
    }
    fn graph_capture_end(&self) -> Result<(), String> {
        Err("graph capture: 미지원".into())
    }
    fn graph_replay(&self, _on: bool) -> Result<(), String> {
        Err("graph capture: 미지원".into())
    }

    /// 그래프 캡처 세그먼트 경계 — 스텝 내 호스트 왕복(d2h/h2d) 지점에서 호출된다.
    /// 캡처 구현체는 이 지점에서 현재 세그먼트를 닫고 다음을 연다(재생 시엔 순서대로 발사).
    /// 기본 no-op — 그래프를 지원하지 않는 백엔드는 그대로 둔다.
    fn capture_mark(&self, _tag: &str) -> Result<(), String> {
        Ok(())
    }

    /// KTRACE 덤프+재시작(q4acc 등 백엔드 훅) — 진단용 기본 no-op.
    fn ktrace_tick(&self) {}

    /// 시퀀스 상태 초기화(슬롯 반납) — 가속기가 들고 있는 시퀀스별 상주 상태
    /// (예: PLE n-gram 링)을 제거한다. 기본 no-op.
    fn acc_reset_seq(&self, _seq: usize) {}

    /// 프레임 logits [t][vocab]의 행별 argmax — GPU 판정 후 토큰만 회수
    /// (np greedy: vocab×t 플로트 전사 회피). 동률 시 최저 인덱스(CPU greedy와
    /// 동일 의미). 미구현 백엔드는 Err (호출부 폴백).
    fn frame_argmax_rows(
        &self,
        _logits: u64,
        _t: usize,
        _vocab: usize,
    ) -> Result<Vec<u32>, String> {
        Err("frame_argmax_rows: 미지원".into())
    }

    /// np 행별 conv — qkv/out은 [t][ch] 연속, states는 행(시퀀스)별 상태
    /// 핸들. gdn_conv(t=1) 산술 그대로 1런치 (plans/74 N2). 미구현은 Err.
    fn frame_gdn_conv_np(
        &self,
        _qkv: u64,
        _out: u64,
        _states: &[u64],
        _cw: u64,
        _ch: usize,
        _k: usize,
    ) -> Result<(), String> {
        Err("frame_gdn_conv_np: 미지원".into())
    }

    /// np 행별 AR — q/k/v/beta_ge/out은 [t][·] 연속, states는 행별 상태
    /// 핸들. gdn_ar_w_swap(t=1) 산술 그대로 1런치 (plans/74 N2). 미구현은 Err.
    #[allow(clippy::too_many_arguments)]
    fn frame_gdn_ar_np(
        &self,
        _q: u64,
        _k: u64,
        _v: u64,
        _beta_ge: u64,
        _out: u64,
        _states: &[u64],
        _h_k: usize,
        _h_v: usize,
        _d: usize,
    ) -> Result<(), String> {
        Err("frame_gdn_ar_np: 미지원".into())
    }

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

    /// FFN 상주 체인: xs → [gate,up] GEMV → silu·mul → down GEMV → xs 갱신.
    /// 미구현 백엔드는 Err (호출부 폴백 — 그룹+silu+down 개별).
    fn ffn_chain(
        &self,
        _xs: &[Vec<f32>],
        _gate_w: &Weight,
        _up_w: &Weight,
        _down_w: &Weight,
        _xs_out: &mut [Vec<f32>],
    ) -> Result<(), String> {
        Err("ffn_chain: 미지원".into())
    }

    /// silu(gate)·up 곱 배치 — 미구현 백엔드는 Err (호출부 CPU 폴백).
    fn silu_mul(
        &self,
        _gs: &[Vec<f32>],
        _us: &[Vec<f32>],
        _outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        Err("silu_mul: 미지원".into())
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
    /// 디바이스 총 메모리(바이트). 미지원/CPU면 0 — 적응형 버퍼 상한 결정에 쓴다.
    fn total_mem_bytes(&self) -> u64 {
        0
    }

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

    /// plans/67 2a: 프레임 버퍼의 q/k에 **RMS norm + rope**를 디바이스에서 적용
    /// (in-place). q는 [t][n_head*2*hd] (gate 절반은 그대로), k는 [t][n_kv*hd].
    /// `cs`는 cos/sin 로프 테이블(모델 상수)로 호출부가 넘긴다.
    #[allow(clippy::too_many_arguments)]
    /// plans/73(np): 프레임 버퍼 행 뷰 — base+off_elems 위치를 frames 테이블에
    /// 등록해 새 핸들을 반환한다. np 배치 디코드가 per-seq 상태 op(conv/AR/
    /// QSA 선택·rope·어텐션)에 행 슬라이스를 그대로 넘기기 위해서다.
    /// 기존 메서드·커널은 무변경(핸들 = 포인터이므로 그대로 소비된다).
    fn frame_slice(&self, _h: u64, _off_elems: usize, _len: usize) -> Result<u64, String> {
        Err("frame_slice: 이 가속기는 미지원".into())
    }

    #[allow(clippy::too_many_arguments)]
    fn frame_qk_norm_rope(
        &self,
        _q: u64,
        _k: u64,
        _q_norm: &[f32],
        _k_norm: &[f32],
        _cs: &[f32],
        _eps: f32,
        _pos0: usize,
        _n_head: usize,
        _n_kv: usize,
        _hd: usize,
        _n_rot: usize,
        _t: usize,
    ) -> Result<(), String> {
        Err("frame_qk_norm_rope: 이 가속기는 미지원".into())
    }

    /// plans/67 3단계: QSA KV 캐시 **디바이스 상주화** — (full_idx, seq) 풀에
    /// k/v 행(mm_group 출력 버퍼, 이미 norm·rope 완료)을 D2D append하고 풀
    /// 핸들을 반환한다. 어텐션이 이 풀을 직접 읽으면 매 층 매 스텝의 캐시
    /// 재업로드(8k 문맥 32MB)가 사라진다. 미지원이면 Err(호출부가 업로드 경로로).
    #[allow(clippy::too_many_arguments)]
    fn qsa_kv_dev(
        &self,
        _full_idx: usize,
        _seq: usize,
        _k: u64,
        _v: u64,
        _t: usize,
        _pos0: usize,
        _n_kv: usize,
        _hd: usize,
    ) -> Result<(u64, u64), String> {
        Err("qsa_kv_dev: 이 가속기는 미지원".into())
    }

    /// plans/72: 디코드(t=1) shared expert 융합 — gate+up+silu(1런치),
    /// down+sigmoid·axpy(1런치). 기존 8런치를 대체.
    fn shexp_gu(
        &self, _x: u64, _wg: &Weight, _wu: &Weight, _h: u64,
        _n_in: usize, _n_hidden: usize,
    ) -> Result<(), String> {
        Err("shexp_gu: 이 가속기는 미지원".into())
    }
    fn shexp_da(
        &self, _h: u64, _wd: &Weight, _s: u64, _mout: u64,
        _n_in: usize, _n_hidden: usize,
    ) -> Result<(), String> {
        Err("shexp_da: 이 가속기는 미지원".into())
    }

    /// 진단: 상주 풀 내용이 호스트 캐시와 비트一致하는지 검증(plans/67 3단계 디버그).
    fn qsa_kv_check(
        &self,
        _full_idx: usize,
        _seq: usize,
        _host_ck: &[f32],
        _host_cv: &[f32],
    ) -> Result<(), String> {
        Ok(())
    }

    /// 디바이스 상주 캐시판 어텐션 — ck/cv는 qsa_kv_dev가 반환한 핸들.
    /// 산술·커널은 qsa_attention_dev와 동일(t=1 분할 규약 포함).
    #[allow(clippy::too_many_arguments)]
    fn qsa_attention_dev_res(
        &self,
        _q: u64,
        _ck: u64,
        _cv: u64,
        _sel_idx: &[u32],
        _sel_off: &[u32],
        _kq_scale: f32,
        _n_head: usize,
        _n_kv: usize,
        _hd: usize,
        _t: usize,
        _out: u64,
    ) -> Result<(), String> {
        Err("qsa_attention_dev_res: 이 가속기는 미지원".into())
    }

    /// plans/73 SELCHECK 진단: qsa_sel_dev가 만든 디바이스 선택 목록을 호스트로
    /// 읽어 돌려준다(검증 전용 — 프로덕션 경로는 부르지 않는다).
    fn qsa_sel_readback(
        &self,
        _sel_idx: u64,
        _sel_off: u64,
        _list_len: usize,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        Err("qsa_sel_readback: 이 가속기는 미지원".into())
    }

    /// plans/73: PLE 수학의 디바이스판(디코드 t=1) — gate/conv/잔차 3커널.
    /// key/value 투영은 호출부가 frame_mm_group으로 수행한 뒤 이 메서드에
    /// 디바이스 버퍼를 넘긴다. ring은 (seq)별 상주 상태(워터마크 규약).
    #[allow(clippy::too_many_arguments)]
    fn ple_math_dev(
        &self,
        _res: u64,
        _key: u64,
        _value: u64,
        _nk: &[f32],
        _nq: &[f32],
        _nc: &[f32],
        _conv_w: &[f32],
        _gated: u64,
        _conv_out: u64,
        _gate_out: u64,
        _seq: usize,
        _t: usize,
        _eps: f32,
        _n_embd: usize,
        _hc: usize,
        _kern: usize,
        _dil: usize,
        _hist: usize,
        _host_ring: &[f32],
    ) -> Result<(), String> {
        Err("ple_math_dev: 이 가속기는 미지원".into())
    }

    /// plans/73: QSA 인덱서 선택의 **디바이스판** (디코드 t=1). iq/ik가 프레임
    /// 버퍼(디바이스)에 있을 때 호스트 왕복 없이 (1) ik를 idx 풀에 적립,
    /// (2) iq norm+rope, (3) 블록키 증분 갱신, (4) 점수·top-k·선택목록 전개까지
    /// 커널로 수행한다. 반환 = (sel_idx 디바이스 핸들, sel_off 핸들, 목록 길이).
    /// 산술은 stages::qsa_select와 동일 순서 — SELCHECK 프로브로 목록 일치 검증.
    #[allow(clippy::too_many_arguments)]
    fn qsa_sel_dev(
        &self,
        _full_idx: usize,
        _seq: usize,
        _iq: u64,
        _ik: u64,
        _t: usize,
        _pos0: usize,
        _idx_heads: usize,
        _idx_dim: usize,
        _r: usize,
        _idx_top_k: usize,
        _iqw: &[f32],
        _ikw: &[f32],
        _cs_idx: &[f32],
        _eps: f32,
    ) -> Result<(u64, u64, usize), String> {
        Err("qsa_sel_dev: 이 가속기는 미지원".into())
    }

    /// plans/73: 프리필(t>1)이 호스트 선택 후 **디바이스 idx 풀만** 갱신 — ik 청크
    /// h2d 적립 + 완성 블록의 블록키 재계산. 이후 디코드의 qsa_sel_dev가
    /// 풀을 이어 쓴다.
    #[allow(clippy::too_many_arguments)]
    fn qsa_idx_append_host(
        &self,
        _full_idx: usize,
        _seq: usize,
        _ik_host: &[f32],
        _t: usize,
        _pos0: usize,
        _idx_dim: usize,
        _r: usize,
        _ikw: &[f32],
        _cs_idx: &[f32],
        _eps: f32,
    ) -> Result<(), String> {
        Ok(())
    }

    /// plans/73: 디바이스 풀 → 호스트 캐시 재구축(디코드가 호스트 갱신을 건너뛴
    /// 뒤 프리필/폴백 진입 시 1회). kv_k/kv_v는 [pos*kv_row], idx_k는
    /// [pos*idx_dim], bk는 [(pos/r)*idx_dim]까지 채운다.
    #[allow(clippy::too_many_arguments)]
    fn qsa_host_rebuild(
        &self,
        _full_idx: usize,
        _seq: usize,
        _pos: usize,
        _kv_row: usize,
        _kv_k: &mut [f32],
        _kv_v: &mut [f32],
        _idx_k: &mut [f32],
        _bk: &mut [f32],
        _r: usize,
        _idx_dim: usize,
    ) -> Result<(), String> {
        Err("qsa_host_rebuild: 이 가속기는 미지원".into())
    }

    /// plans/73: sel 목록이 **디바이스 버퍼**에 이미 있는 상주 캐시판 어텐션 —
    /// 업로드 없이 qsa_attention_dev_res와 동일 커널(t=1 분할 우선).
    #[allow(clippy::too_many_arguments)]
    fn qsa_attention_dev_sel(
        &self,
        _q: u64,
        _ck: u64,
        _cv: u64,
        _sel_idx: u64,
        _sel_off: u64,
        _list_len: usize,
        _kq_scale: f32,
        _n_head: usize,
        _n_kv: usize,
        _hd: usize,
        _t: usize,
        _out: u64,
    ) -> Result<(), String> {
        Err("qsa_attention_dev_sel: 이 가속기는 미지원".into())
    }

    /// QSA 선택-목록 GQA의 **디바이스 q판** — q가 이미 디바이스 버퍼(wq의
    /// frame_mm_group 출력)에 있을 때 h2d 없이 어텐션을 돈다(plans/67 1단계).
    /// k/v는 기존처럼 캐시 업로드 경로(kv_sync)를 쓴다. 출력은 out 버퍼에.
    #[allow(clippy::too_many_arguments)]
    fn qsa_attention_dev(
        &self,
        _q: u64,
        _ck: &[f32],
        _cv: &[f32],
        _sel_idx: &[u32],
        _sel_off: &[u32],
        _kq_scale: f32,
        _n_head: usize,
        _n_kv: usize,
        _hd: usize,
        _t: usize,
        _out: u64,
    ) -> Result<(), String> {
        Err("qsa_attention_dev: 이 가속기는 미지원".into())
    }

    /// QSA 선택-목록 GQA — 마스크 대신 (t+1) 오프셋 + **오름차순** 위치 목록.
    /// 마스크 스캔과 산술 순서가 같다(선택 밖 키는 소프트맥스 상태를 바꾸지
    /// 않는다) — 프로브 `q4-qsa-check`가 비트 동일로 확인한다.
    #[allow(clippy::too_many_arguments)]
    fn qsa_attention_sel(
        &self,
        _q: &[f32],
        _ck: &[f32],
        _cv: &[f32],
        _sel_idx: &[u32],
        _sel_off: &[u32],
        _kq_scale: f32,
        _n_head: usize,
        _n_kv: usize,
        _hd: usize,
        _t: usize,
    ) -> Result<Vec<f32>, String> {
        Err("qsa_attention_sel: 이 가속기는 미지원".into())
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
    /// n = 처리할 원소 수(행 수 = n/d). 버퍼 길이가 아니라 토큰 수에서 온다.
    L2Rows { x: u64, eps: f32, d: usize, n: usize },
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
    /// k_sel행 브로드캐스트 — dst의 모든 행 = src 0행 (MoE t=1 gate/up, 1런치).
    BcastRows { src: u64, dst: u64, n: usize, rows: usize },
    /// MoE shared 가산: y += x·s (s는 1원소 프레임 버퍼).
    AxpyScaled { y: u64, x: u64, s: u64, n: usize },
    /// MoE 전문가 가중 합: out = Σ_e wt[e]·ys[e].
    MoeWeightedSum { ys: u64, wt: u64, out: u64, k: usize, n: usize },
}

/// 프레임 상태 연산 — 상주 상태(kv/gdn/conv/blk)를 갱신하는 가속기 전용
/// 메서드. 값 경로 Accelerator 메서드와 대응하되 입출력이 전부 핸들.
pub trait FrameState {
    /// 프레임 forward 시작 — 이번 스텝의 토큰 수. 프레임 버퍼는 t_max 크기로
    /// 잡히므로 "행 수 = 버퍼 길이/n" 유도가 t>1 청크에서 틀린다. op 커널이
    /// 토큰 수를 알아야 하는 지점(RmsRows/HcGateMean/NormGated/L2Rows/top-k/
    /// 가중합/split3)이 이 값을 쓴다.
    fn frame_begin(&self, _t: usize) {}

    /// 컨텍스트 길이 주입 — 가속기가 KV 등 **상한이 정해진 풀을 선할당**하는 데 쓴다
    /// (llama.cpp/vLLM처럼 "한 번 잡고 그 안에서만"). 기본은 무시.
    fn set_ctx_len(&self, _n: usize) {}

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
    /// MoE (t>1) — (토큰,전문가) 페어 행 gather:
    /// xsel[(tok·k_sel+e)·n + i] = mix[tok·n + i]. 기본 미지원.
    fn frame_moe_gather(
        &self,
        _mix: u64,
        _xsel: u64,
        _n: usize,
        _k_sel: usize,
        _t: usize,
    ) -> Result<(), String> {
        Err("frame_moe_gather: 미지원".into())
    }

    /// MoE (t>1) — 전문가 가중 합(토큰별):
    /// out[tok·n + i] = Σ_e wt[tok·k_sel+e]·ys[(tok·k_sel+e)·n + i]. 기본 미지원.
    fn frame_moe_scatter(
        &self,
        _ys: u64,
        _wt: u64,
        _out: u64,
        _k_sel: usize,
        _n: usize,
        _t: usize,
    ) -> Result<(), String> {
        Err("frame_moe_scatter: 미지원".into())
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
        _k_sel: usize,
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

    /// 128-행 타일 커널(j128/v4 CO) 로드 여부 — prefill 청크 128 게이트.
    /// 기본 false (무타일 백엔드).
    fn tile_big_chunk(&self) -> bool {
        false
    }

    /// raw_step + 최종 hidden 회수 (MTP 훅용). 기본 Err.
    fn raw_step_h(
        &self,
        _seq: usize,
        _pos: usize,
        _emb: &[f32],
        _h_out: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        Err("raw_step_h: 미지원".into())
    }

    /// 배치 검증 — 토큰 t개 처리 + 행별 argmax 반환 (MTP spec). 기본 Err.
    fn raw_verify(
        &self,
        _seq: usize,
        _pos0: usize,
        _emb: &[f32],
        _argmaxes: &mut Vec<u32>,
        _h_all: &mut Vec<f32>,
    ) -> Result<(), String> {
        Err("raw_verify: 미지원".into())
    }

    /// MTP 임베딩 선반입 (사이드 스트림) — 미지원 백엔드는 no-op.
    fn mtp_upload_tok_emb(&self, _tok_flat: &[f32]) -> Result<(), String> {
        Ok(())
    }

    /// raw_prefill + 마지막 행 hidden 회수 (MTP carry). 기본 Err.
    fn raw_prefill_h(
        &self,
        _seq: usize,
        _pos0: usize,
        _emb: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        Err("raw_prefill_h: 미지원".into())
    }

    /// np×spec 병합 verify — 행별 argmax. group_starts: seq별 그룹 첫 행. 기본 Err.
    fn verify_batch_ms(
        &self,
        _seqs: &[usize],
        _poss: &[usize],
        _group_starts: &[usize],
        _emb: &[f32],
        _argmaxes: &mut Vec<u32>,
        _h_all: &mut Vec<f32>,
    ) -> Result<(), String> {
        Err("verify_batch_ms: 미지원".into())
    }

    /// np 배치 디코드 — seq별 logits. 기본 Err.
    fn raw_step_multi(
        &self,
        _seqs: &[usize],
        _poss: &[u32],
        _emb: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        Err("raw_step_multi: 미지원".into())
    }

    /// MTP 체인 스텝 (h = 내부 mtp_cur): argmax. 기본 Err.
    fn mtp_step_chain(&self, _seq: usize, _tok_emb: &[f32], _pos: usize) -> Result<u32, String> {
        Err("mtp_step_chain: 미지원".into())
    }
    /// np 배치 디코드 greedy — 행별 토큰만 회수 (logits 전사·CPU 스캔 회피).
    /// 기본 구현은 raw_step_multi + CPU greedy 폴백.
    fn raw_step_multi_greedy(
        &self,
        seqs: &[usize],
        poss: &[u32],
        emb: &[f32],
    ) -> Result<Vec<u32>, String> {
        let ls = self.raw_step_multi(seqs, poss, emb)?;
        Ok(ls.iter().map(|l| greedy_from(l)).collect())
    }
    /// MTP 상태 진행 (trunk h, head 없음). 기본 Err.
    fn mtp_step_adv(
        &self,
        _seq: usize,
        _tok_emb: &[f32],
        _h: &[f32],
        _pos: usize,
    ) -> Result<(), String> {
        Err("mtp_step_adv: 미지원".into())
    }

    /// MTP 1스텝 GPU (blk.64): (argmax, h_next). 기본 Err.
    fn mtp_step_gpu(
        &self,
        _seq: usize,
        _tok_emb: &[f32],
        _h: &[f32],
        _pos: usize,
    ) -> Result<(u32, Vec<f32>), String> {
        Err("mtp_step_gpu: 미지원".into())
    }

    /// MTP 프리필 배치 (HIP 전용). carry_h = 이전 청크 마지막 행 hidden 1행 —
    /// 나머지 행은 디바이스의 본체 hidden(xs_t)에서 행 시프트로 조립한다.
    /// with_head=false면 KV 적립만(초안 없음). 다른 백엔드는 미지원.
    fn mtp_prefill_batch(
        &self,
        _seq: usize,
        _tok_embs: &[f32],
        _carry_h: &[f32],
        _t: usize,
        _pos0: usize,
        _with_head: bool,
    ) -> Result<u32, String> {
        Err("mtp_prefill_batch: 미지원(백엔드)".into())
    }

    /// MTP KV 적립 전용 스텝 — with_head=false면 전체 vocab 헤드(argmax)를 생략한다.
    /// 프롬프트 전 토큰의 KV를 쌓는 동안 헤드는 마지막 토큰만 필요하다.
    /// 기본 구현은 항상 헤드를 계산한다(미지원 백엔드 폴백).
    fn mtp_step_hidden(
        &self,
        seq: usize,
        tok_emb: &[f32],
        h: &[f32],
        pos: usize,
        with_head: bool,
    ) -> Result<Option<u32>, String> {
        let (am, _h) = self.mtp_step_gpu(seq, tok_emb, h, pos)?;
        Ok(if with_head { Some(am) } else { None })
    }

    /// GDN/conv 상태 스냅샷·복원 (spec 부분수용 롤백). 기본 Err.
    fn gdn_snapshot(&self) -> Result<(), String> {
        Err("gdn_snapshot: 미지원".into())
    }
    fn gdn_restore(&self) -> Result<(), String> {
        Err("gdn_restore: 미지원".into())
    }

    /// 시퀀스 상태 초기화 (서버 슬롯 반환 시) — GDN/conv 상주 상태 제로화.
    /// KV는 위치 색인이라 미제로 무해 (p ≤ pos만 판독). 기본 no-op.
    fn raw_reset(&self, _seq: usize) -> Result<(), String> {
        Ok(())
    }
    /// 정규화 h → head argmax (MTP draft). 기본 Err.
    fn mtp_head_argmax(&self, _h_normed: &[f32]) -> Result<u32, String> {
        Err("mtp_head_argmax: 미지원".into())
    }
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
            | llm170_gguf::GgmlType::Q5_1
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
                    llm170_gguf::GgmlType::Q5_1 => {
                        crate::quant::dot_row_w4a8_q5_1_lane(row, w.n_in, y)
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
