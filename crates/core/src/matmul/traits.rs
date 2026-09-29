//! f32 행렬-벡터/배치 곱 — 무게는 양자화 바이트에서 타일 단위로 디양자화.
//!
//! ggml 텐서 레이아웃: W [ne0=n_in, ne1=n_out] 행 우선 — out[o] = Σ_i x[i]·W[o,i].
//! ADR-0005: GPU 커널이 아닌 CPU 참조 경로. FMA 없는 mul+add (x86-64 기본 타깃은
//! auto-FMA가 없어 자동으로 성립; target-feature 변경 시 재검토 필요 — 주석 유지).

use super::weight::Weight;

pub trait GraphCapture: Send + Sync {
    /// 비동기 프리필용 스트림 페어 전환 — on 이면 이후 발행이 프리필 전용
    /// 스트림 쌍(메인+사이드)으로 간다. 미지원 백엔드는 no-op.
    fn pre_pair(&self, _on: bool) {}
    /// 프리필 완료 이벤트 기록 / 비블로킹 확인 / 메인 합류.
    fn pre_mark(&self) -> Result<(), String> {
        Err("pre_mark: 미지원".into())
    }
    fn pre_ready(&self) -> bool {
        false
    }
    fn pre_join(&self) -> Result<(), String> {
        Ok(())
    }

    /// 그래프 캡처 세그먼트 경계 — 스텝 내 호스트 왕복(d2h/h2d) 지점에서 호출된다.
    /// 캡처 구현체는 이 지점에서 현재 세그먼트를 닫고 다음을 연다(재생 시엔 순서대로 발사).
    /// 기본 no-op — 그래프를 지원하지 않는 백엔드는 그대로 둔다.
    fn capture_mark(&self, _tag: &str) -> Result<(), String> {
        Ok(())
    }
}

/// 양자화 matmul 계열과 동기 프리미티브.
///
/// plans/75 P1 — `Accelerator` 분해의 일부. 스테이지 코드는 필요한
/// capability 만 요구하도록 좁힐 수 있다(기본 구현은 종전과 동일).
pub trait MatmulHost: Send + Sync {
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

    /// 전문가 down처럼 입력이 가중치마다 다른 1행 짝: outs[i][o] = xs[i]·W_i[o].
    /// 기본 = 개별 실행. GPU 구현은 런치 배치 + 단일 동기화로 파이프라이닝.
    fn matmul_paired(
        &self,
        xs: &[Vec<f32>],
        ws: &[Weight],
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        if ws.len() != xs.len() || ws.len() != outs.len() {
            return Err(format!(
                "matmul_paired: 형상 불일치 ws={} xs={} outs={}",
                ws.len(),
                xs.len(),
                outs.len()
            ));
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
            return Err(format!(
                "matmul_group: ws({}) != outs({})",
                ws.len(),
                outs.len()
            ));
        }
        for (w, out) in ws.iter().zip(outs.iter_mut()) {
            self.matmul_batch(xs, w, out)?;
        }
        Ok(())
    }
}

/// 값 경로 elementwise/GDN/셰이프 op.
///
/// plans/75 P1 — `Accelerator` 분해의 일부. 스테이지 코드는 필요한
/// capability 만 요구하도록 좁힐 수 있다(기본 구현은 종전과 동일).
pub trait EwOps: Send + Sync {
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

    /// plans/72: 디코드(t=1) shared expert 융합 — gate+up+silu(1런치),
    /// down+sigmoid·axpy(1런치). 기존 8런치를 대체.
    fn shexp_gu(
        &self,
        _x: u64,
        _wg: &Weight,
        _wu: &Weight,
        _h: u64,
        _n_in: usize,
        _n_hidden: usize,
    ) -> Result<(), String> {
        Err("shexp_gu: 이 가속기는 미지원".into())
    }
    fn shexp_da(
        &self,
        _h: u64,
        _wd: &Weight,
        _s: u64,
        _mout: u64,
        _n_in: usize,
        _n_hidden: usize,
    ) -> Result<(), String> {
        Err("shexp_da: 이 가속기는 미지원".into())
    }

    /// plans/73: PLE 수학의 디바이스판(디코드 t=1) — gate/conv/잔차 3커널.
    /// key/value 투영은 호출부가 frame_mm_group으로 수행한 뒤 이 메서드에
    /// 디바이스 버퍼를 넘긴다. ring은 (seq)별 상주 상태(워터마크 규약).
    #[allow(clippy::too_many_arguments)]
    /// plans/97 — pos==0 상태의 GPU zero-fill(gdn+conv). 실패 시 CPU 업로드 폴백.
    fn frame_zero_states(&self, _gdn: &[u64], _conv: &[u64]) -> Result<(), String> {
        Err("frame_zero_states: 미지원".into())
    }

    /// plans/97 — token_embd(Q8_0) gather + hc 방송 GPU 오프로드.
    #[allow(clippy::too_many_arguments)]
    fn emb_q8_gather_dev(
        &self,
        _table_key: usize,
        _table: &[u8],
        _tokens: &[u32],
        _out: u64,
        _n: usize,
        _hc: usize,
    ) -> Result<(), String> {
        Err("emb_q8_gather_dev: 미지원".into())
    }

    /// plans/93 — PLE 임베딩 gather GPU 오프로드(IQ4_NL).
    fn ple_gather_dev(
        &self,
        _table_key: usize,
        _table: &[u8],
        _rows: &[u32],
        _out: u64,
        _hd: usize,
    ) -> Result<(), String> {
        Err("ple_gather_dev: 미지원".into())
    }

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
        // plans/93 P2: pos 기반 워터마크(역방향 감지). t>1 프리필 디바이스화의
        // 핵심 — t 기반 판정은 프리필(512)→디코드(1) 전환을 롤백으로 오판했다.
        _pos0: usize,
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

    /// plans/93 P2 — 디바이스 링 상태 판독(GPU 유휴 시점 호출 전제).
    /// 프리필 디바이스화 후 엔진 CPU 상태(seq_st.ple_conv) 재동기용.
    fn ple_ring_sync(&self, _seq: usize, _ring_out: &mut [f32]) -> Result<(), String> {
        Err("ple_ring_sync: 이 가속기는 미지원".into())
    }
}

/// QSA(인덱서·선택·KV 상주) 어텐션 계열.
///
/// plans/75 P1 — `Accelerator` 분해의 일부. 스테이지 코드는 필요한
/// capability 만 요구하도록 좁힐 수 있다(기본 구현은 종전과 동일).
pub trait QsaOps: Send + Sync {
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

    /// plans/89 재개: QSA 선택의 **프리필 다중 토큰 디바이스판** — iq/ik가
    /// 프레임 버퍼에 있을 때 (적립+블록키) → q_rope → 토큰별 점수 → 토큰별
    /// 비토닉 top-k → 평탄 목록+sel_off까지 전부 커널. 호스트 d2h 4회
    /// (배치 플러시)와 CPU 점수/정렬을 소거. 반환은 qsa_sel_dev 동일.
    /// nb > 4096(문맥 ~16k+)은 Err — 호스트 폴백.
    #[allow(clippy::too_many_arguments)]
    fn qsa_sel_dev_mt(
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
        Err("qsa_sel_dev_mt: 이 가속기는 미지원".into())
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
        // 계약 일관화(90 B4): 기본구현의 조용한 Ok(())는 "적립했다"고 거짓
        // 보고해 이후 디코드 풀 판돉이 구멍(워터마크 불일치)을 만든다 —
        // 미구현은 Err로 명시하고 호출부가 호스트 경로를 유지하게 한다.
        Err("qsa_idx_append_host: 미지원".into())
    }

    /// 인덱서 k 행을 **디바이스 버퍼에서 직접** 적립(블록 키 갱신 포함).
    /// t>1 프리필 단축 경로 전용 — 기본 미지원(호스트 경로 폴백).
    #[allow(clippy::too_many_arguments)]
    fn qsa_idx_append_dev(
        &self,
        _full_idx: usize,
        _seq: usize,
        _ik: u64,
        _t: usize,
        _pos0: usize,
        _idx_dim: usize,
        _r: usize,
        _ikw: &[f32],
        _cs_idx: &[f32],
        _eps: f32,
    ) -> Result<(), String> {
        Err("qsa_idx_append_dev 미지원".into())
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
}

/// 프레임 버퍼·프레임 op·프레임 GEMM — 상태는 FrameState 승계.
///
/// plans/75 P1 — `Accelerator` 분해의 일부. 스테이지 코드는 필요한
/// capability 만 요구하도록 좁힐 수 있다(기본 구현은 종전과 동일).
pub trait FrameHost: Send + Sync {
    /// 프레임 경로 완전성 — false면 엔진이 프레임 진입을 건너뛴다(값경로).
    /// 부분 구현 백엔드(plans/84 B vk)가 완성 전 기본 경로를 깨지 않게 한다.
    fn frame_capable(&self) -> bool {
        true
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
    /// 진단 판독 전 전체 동기 — buf_hash 계측이 커스텀 스트림 파이프라인의
    /// 미완결 쓰기를 읽는 경합을 없앤다(plans/84 E.2 mout 역설 교정).
    fn frame_sync(&self) {}
    fn frame_read(&self, _h: u64, _out: &mut [f32]) -> Result<(), String> {
        Err("frame_read: 미지원".into())
    }
    /// 상주 GEMM: out[t·n_out..] = x[t·n_in..]·W — 업/다운로드 없음.
    fn frame_mm(&self, _x: u64, _w: &Weight, _out: u64, _t: usize) -> Result<(), String> {
        Err("frame_mm: 미지원".into())
    }
    /// 상주 GEMM 그룹 — 동일 입력 x, 가중치별 out.
    fn frame_mm_group(
        &self,
        _x: u64,
        _ws: &[Weight],
        _outs: &[u64],
        _t: usize,
    ) -> Result<(), String> {
        Err("frame_mm_group: 미지원".into())
    }
    /// plans/104 — 격리 quant 버퍼 판(공유전문가 병렬 체인). 미지원 백엔드는
    /// 일반 그룹으로 폴백(산술 동일).
    fn frame_mm_group_sep(
        &self,
        x: u64,
        ws: &[Weight],
        outs: &[u64],
        t: usize,
    ) -> Result<(), String> {
        self.frame_mm_group(x, ws, outs, t)
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
    fn frame_mm_q8(
        &self,
        _xq: u64,
        _xd: u64,
        _w: &Weight,
        _out: u64,
        _n: usize,
    ) -> Result<(), String> {
        Err("frame_mm_q8: 미지원".into())
    }
}

/// GPU 가속기 합성 트레이트 — capability 서브트레이트의 합집합(plans/75 P1).
///
/// 호출부(`&dyn Accelerator`)는 종전 시그니처 그대로다. 구현체는 서브트레이트만
/// 구현하면 되므로(블랭킷) 백엔드가 필요한 capability 만 갖출 수 있다.
pub trait Accelerator:
    FrameState + GraphCapture + MatmulHost + EwOps + QsaOps + FrameHost + Send + Sync
{
}

impl<T> Accelerator for T where
    T: FrameState + GraphCapture + MatmulHost + EwOps + QsaOps + FrameHost + Send + Sync
{
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
    RmsRows {
        x: u64,
        w: u64,
        out: u64,
        eps: f32,
        n: usize,
        w_reps: usize,
    },
    /// GDN norm_gated: out = rms(o)·σ(z), w 반복 = 헤드.
    NormGated {
        o: u64,
        z: u64,
        w: u64,
        out: u64,
        eps: f32,
        d: usize,
        n_h: usize,
    },
    /// GDN norm_gated silu 변형 (qwen35): out = rms(o)·silu(z)·w.
    NormGatedSilu {
        o: u64,
        z: u64,
        w: u64,
        out: u64,
        eps: f32,
        d: usize,
        n_h: usize,
    },
    /// GDN q/k 헤드별 in-place L2 norm.
    /// n = 처리할 원소 수(행 수 = n/d). 버퍼 길이가 아니라 토큰 수에서 온다.
    L2Rows {
        x: u64,
        eps: f32,
        d: usize,
        n: usize,
    },
    /// conv 출력 3분할 (q/k/v) — 카피 3런치 융합.
    Split3 {
        src: u64,
        d0: u64,
        d1: u64,
        d2: u64,
        n0: usize,
        n1: usize,
        n2: usize,
    },
    /// L2 이중 행 + q 스케일 융합 (산술 l2_rows+scale 과 동일).
    L2Rows2Scale {
        q: u64,
        k: u64,
        eps: f32,
        scale: f32,
        d: usize,
        n_group: usize,
    },
    /// 어텐션 q/k 헤드 rms+rope in-place (f64 중간 — 브리지 제거).
    QKNormRope {
        q: u64,
        k: u64,
        qw: u64,
        kw: u64,
        cs: u64,
        eps: f32,
        kqs: f32,
        pos: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        n_rot: usize,
    },
    /// hc 게이트 적용 + 스트림 평균 (hc는 나눗셈 피수로 사용).
    HcGateMean {
        xn: u64,
        gate: u64,
        out: u64,
        hc: usize,
        n: usize,
    },
    /// hc combine: res += out·(2·σ(inj/hc)).
    HcCombine {
        res: u64,
        out: u64,
        inj: u64,
        hc: usize,
        n: usize,
        total: usize,
    },
    /// GDN β/e^g 사전계산: bg[h·2]=σ(b), bg[h·2+1]=e^(softplus(a+dtb)·sa).
    GdnBetaG {
        b: u64,
        a: u64,
        dtb: u64,
        sa: u64,
        bg: u64,
        n_h: usize,
    },
    /// GDN conv1d + ring shift + silu (state in-place).
    GdnConv {
        qkv: u64,
        cw: u64,
        state: u64,
        out: u64,
        ch: usize,
        k: usize,
        t_len: usize,
    },
    /// MoE route top-k: ids/wt GPU 잔류.
    MoeTop10 {
        route: u64,
        ids: u64,
        wt: u64,
        n_exp: usize,
        k_sel: usize,
    },
    /// NEOX RoPE (cs = [pos_max][half][2] cos,sin 인터리브 테이블).
    RopeApply {
        x: u64,
        cs: u64,
        pos_base: usize,
        rows_per_tok: usize,
        pos_mul: usize,
        stride: usize,
        half: usize,
    },
    /// 인덱서 블록키 풀링 (mean of r rows).
    IdxPool {
        cache: u64,
        out: u64,
        first_block: usize,
        dim: usize,
        r: usize,
    },
    /// 인덱서 스코어: Σ_h ReLU(qr·bk).
    IdxScores {
        qr: u64,
        bk: u64,
        scores: u64,
        idx_heads: usize,
        dim: usize,
    },
    /// qwen35 어텐션 q 프리페어: 헤드 rms·rope·q‖gate 인터리브.
    AttnQPrep {
        q: u64,
        w: u64,
        cs: u64,
        out: u64,
        eps: f32,
        hd: usize,
        pos: usize,
        half: usize,
    },
    /// qwen35 어텐션 k 프리페어: kv-헤드 rms·rope → 캐시 pos append.
    AttnKPrep {
        k: u64,
        w: u64,
        cs: u64,
        cache: u64,
        eps: f32,
        hd: usize,
        pos: usize,
        n_kv: usize,
        half: usize,
    },
    /// in-place: v ← v·s (GDN q 사전 스케일).
    Scale { t: u64, s: f32, n: usize },
    /// 행 복사: dst[dst_off..+n] = src[src_off..+n] — 캐시 append 부품.
    CopyRows {
        src: u64,
        dst: u64,
        src_off: usize,
        dst_off: usize,
        n: usize,
    },
    /// k_sel행 브로드캐스트 — dst의 모든 행 = src 0행 (MoE t=1 gate/up, 1런치).
    BcastRows {
        src: u64,
        dst: u64,
        n: usize,
        rows: usize,
    },
    /// MoE shared 가산: y += x·s (s는 1원소 프레임 버퍼).
    AxpyScaled { y: u64, x: u64, s: u64, n: usize },
    /// MoE 전문가 가중 합: out = Σ_e wt[e]·ys[e].
    MoeWeightedSum {
        ys: u64,
        wt: u64,
        out: u64,
        k: usize,
        n: usize,
    },
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

    /// plans/105(원장 80) — mxsel 생산 직후 1회 팩 정량(블록당 10워드
    /// [qs8][d][Σ]). 등록된 x는 llmmq가 팩 버퍼로 소비.
    /// 기본 Err: 팩을 소비하는 경로에서 미구현 백엔드가 조용히 Ok를
    /// 반환하면 llmmq가 스테일 버퍼를 읽는 무결 오염이 된다(107 P0-3,
    /// 저장소 계약 "조용한 Ok는 거짓 보고"). 팩 불필요 백엔드는 명시
    /// 오버라이드로 근거를 문서화할 것.
    fn frame_quant_pack(&self, _x: u64, _rows: usize, _n_in: usize) -> Result<(), String> {
        Err("frame_quant_pack: 백엔드 미구현 — 팩 버퍼 미생산".into())
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
