//! Flash-Next(Qwen4-Expert) CUDA 스캐폴드 공유 지원 — plans/124 G001(FNA),
//! 2026-10-05.
//!
//! 목표 FNA는 **스캐폴드 + 계약 지도**만 담는다 — 스테이지 산술 구현은
//! 후속 목표 FNB(QSA)·FNC(HC)·FND(GDN)·FNE(MoE)·FNG(PLE)·FNH(프레임
//! 디코드 체인)이 수행한다. 본 파일의 동작 코드는 (1) 픽스처 로더
//! (EXL3 5.05bpw 샤드 safetensors 오프셋 판독 + GGUF 4-샤드 헤더 파서),
//! (2) exl3_fn.fatbin 스모크 프로브, (3) 양 모델 세트 인벤토리 프로브뿉.
//!
//! ═══════════════════════════════════════════════════════════════════════
//! [계약 지도 — 스테이지별 REUSE vs NEW (FNA 판정, 2026-10-05)]
//! ═══════════════════════════════════════════════════════════════════════
//! 산술 진실의 계층(plans/124 §6): **커널 → crates/core CPU 참조(값 maxdiff
//! 판정의 유일 기준) → 수학 자체**. 아래 "원천" 열은 core 파일:행 — 모든
//! 신규 커널은 이 원천과 값 maxdiff로 판정한다(argmax 판정 금지).
//!
//! ┌─ 스테이지 ───┬─ 판정 ─┬─ 대상·원천 ────────────────────────────────┐
//! │ QSA 어텐션   │ NEW    │ 선택 위치 마스크 GQA+게이트 — exl3_attn_
//! │              │        │ fwd3s(G6)는 T≤8 비마스크 전체 KV 전용이라
//! │              │        │ 선택 리스트 스트리밍(kq_scale·sel_off)이
//! │              │        │ 없다. q행 [n_head·2hd] 게이트 인터리브
//! │              │        │ 레이아웃은 fwd3s와 동일(§3.4 q‖gate) —
//! │              │        │ 발사 규약만 차용. 원천: stages/qsa.rs
//! │              │        │ cpu_attn_row L13-64(dot→×kq_scale→max-sub-
//! │              │        │ exp→가중 V 누산→×sigmoid(q[gb+i])),
//! │              │        │ qsa_sel_list L323-358(블록 오름차순+테일
//! │              │        │ 평탄화 — 마스크 스캔과 산술 순서 동일 계약).
//! │ QSA 선택     │ NEW    │ 블록 점수(4-전개 dot·양수만 누적)·top-k
//! │              │        │ select_nth — 원천: stages/qsa.rs L209-247
//! │              │        │ (width=min(n_past, top_k+r−1)·n_sel_blocks
//! │              │        │ 공식 L229-231). G8 argmax 트리와는 다른
//! │              │        │ n_select+nth 계약.
//! │ QSA 인덱서   │ NEW    │ 블록 키 풀링(mean-pool r행→rms_norm→rope
//! │              │        │ pos=b·r) — 원천: stages/qsa.rs L165-196.
//! │              │        │ 인덱서 q norm+rope(n_rot=idx_dim=128 전체
//! │              │        │ 회전) — 원천: qsa_select L137-145.
//! │              │        │ exl3_attn_prep(G6)의 64차 부분회전과 폭이
//! │              │        │ 다름.
//! │ QSA q/k norm │ REUSE  │ 후보: exl3_attn_prep(G6) — rope base 1e7·
//! │ +rope        │        │ 64차 부분회전·q/k norm이 qwen4exp n_rot=64
//! │              │        │ (rope.dimension_count)·rope_base 1e7과 동일.
//! │              │        │ 단 norm 가중은 w−1 저장 규약(G6/§3.4) 대비
//! │              │        │ qwen4exp는 원값 저장(GGUF attn_q_norm.weight
//! │              │        │ f32) — FNB에서 등록 변환 규약 확정. 원천:
//! │              │        │ stages/qsa.rs L330-357(q rope)·L168-171
//! │              │        │ (k_prenormed — 이미 norm+rope된 k 재적용
//! │              │        │ 금지, frame_qk_norm_rope 계약).
//! │ QSA 투영     │ REUSE  │(조건부) q/k/v/iq/ik·o_proj 선형 — EXL3
//! │              │        │ 아카이브: exl3_had_in→gemv/gemm2→exl3_had_
//! │              │        │ out(G2/G4 자산, 결함 1·3·15호 가드).
//! │              │        │ **[FNA 실측 발견] 5.05bpw trellis tw=112
//! │              │        │ (krate=7)가 G2 커널 krate≤6 스테이징 상한
//! │              │        │ (stg[8·48])을 초과** — FNB는 krate 7 스테
//! │              │        │ 이징 확장(원장 갱신) 또는 분할 경로 필요.
//! │              │        │ GGUF: exl3_q4 MMQ(G8, Q4_K/Q5_K/Q8_0 블록).
//! │              │        │ KV 캐시·pos는 pp[0] 디바이스 판독 계약
//! │              │        │ (결함 4호 — G6 dkc/dvc/dpp 미러).
//! │ HC mix       │ NEW    │ grouped RMSNorm(hc=4스트림×2560 — 원천:
//! │              │        │ stages/hc.rs grouped_rms L14-21) + 저랭크
//! │              │        │ 게이트(silu(lo/hc) L60·g·sigmoid(gate) L69·
//! │              │        │ 스트림 평균 /=hc L78) + combine(s += out·
//! │              │        │ 2σ(inject/4) — layers.rs/forward 체인).
//! │              │        │ norm 코어 산술(트리 환원·정밀 sqrt)은
//! │              │        │ exl3_norm_resid(G3)·exl3_mtp_rms(G9) 미러
//! │              │        │ 계급 — FNC 커널의 적산 순서를 이들과 정렬.
//! │ HC 선형      │ REUSE  │(조건부) hc_{kind}_{down,up,inject}·output_
//! │              │        │ hc_{down,up}·nextn.hc_head_{down,up} — gemv/
//! │              │        │ gemm2(EXL3, krate 상동)·q4 MMQ(GGUF).
//! │ GDN conv     │ NEW    │ 채널별 conv_k=4탭 링 순차 처리 — 원천:
//! │              │        │ stages/gdn.rs L69-85. G5 exl3_gdn_conv는
//! │              │        │ 3탭 고정(링 [conv_k−1][conv_ch])이라 탭 수
//! │              │        │ 가 커널 계약. conv_ch=10240(2·16·128+48·128)
//! │              │        │ 은 27B와 동일. q/k 헤드 l2_norm(원천 L86-97,
//! │              │        │ eps=hp.eps)도 NEW — EXL3 l2perm은 lc 순열
//! │              │        │ scatter가 동반되어 그대로 쓸 수 없다(§3.3
//! │              │        │ 방향 계약·사고 1회).
//! │ GDN 사전     │ NEW    │(소형) β=sigmoid(b)·e^g=exp(softplus(a+
//! │              │        │ dt_bias)·ssm_a) — 원천: stages/gdn.rs
//! │              │        │ L52-60(호스트 사전 계산 — 커널 f32 exp
//! │              │        │ 계약은 G5 f64 트윈 원장 참조).
//! │ GDN scan     │ REUSE  │ 후보: exl3_gdn_scan(G5, -fmad=false) — 기하
//! │              │        │ 동일: d=128·value헤드 48(dt_rank=48·
//! │              │        │ d_state=128=6144=27B h_v·d)·상태 [48][128²].
//! │              │        │ 원천은 core/src/gdn.rs gdn_chunk_seq L22·
//! │              │        │ gdn_ar_batch L375 — **qwen35(27B)와
//! │              │        │ qwen4exp가 공유하는 동일 함수**라 G5 미러와
//! │              │        │ 원천이 같다. FND에서 β/e^g 사전 계약만 맞추
//! │              │        │ 면 재사용 확정. S0≠0 실입력 검증 의무
//! │              │        │ (§3.3 — 합성 S0=0은 상태 버그를 가린다).
//! │ GDN 게이트   │ NEW    │ z-게이트가 **sigmoid**(qwen35 silu와 유일
//! │              │        │ 차이) — 원천: stages/gdn.rs L156-168
//! │              │        │ gdn_norm_gated(GdnGate::Sigmoid) + core/
//! │              │        │ gdn_norm.rs L10·L26. G5 gate 커널은 silu
//! │              │        │ 게이트 class.
//! │ GDN 투영     │ REUSE  │(조건부) in_proj_qkv·in_proj_z·a/b 그룹+
//! │              │        │ out_proj — gemv/gemm2(EXL3, krate 7 이슈
//! │              │        │ 상동)·q4 MMQ(GGUF).
//! │ MoE 라우트   │ NEW    │ softmax(max-sub)→total_cmp 정렬→top-10→정규
//! │              │        │ 화(wsum.max(6.103_515_6e-5) 가드) — 원천:
//! │              │        │ stages/moe.rs L46-70. NaN 내성 total_cmp
//! │              │        │ 계약(107 W11)까지 미러.
//! │ MoE 전문가   │ REUSE  │ 후보: GGUF ffn_{gate,up,down}_exps 스택
//! │              │        │ [2560,640,512] Q4_K — G8 moe_down(ids 그룹
//! │              │        │ 런치) 산술 class(원천: stages/moe.rs
//! │              │        │ L204-216 토큰-메이저 (ti,e,w) 페어 스택
//! │              │        │ GEMM). EXL3는 전문가별 분산 텐서(mlp.
//! │              │        │ experts.{e}.* 512세트) — gemv/gemm2 퍼-
//! │              │        │ 전문가 경로(원천 L228-297 서브배치 class).
//! │              │        │ FNE에서 상주 예산 판정(스택 345MB/층 계약
//! │              │        │ L92-95·원장 18호 소형-T 점유).
//! │ MoE shared   │ REUSE  │ gate·up 활성화 silu·mul=exl3_ew(G7 그대로
//! │              │        │ — 원천: stages/moe.rs silu_rows L10-16·
//! │              │        │ L249-255)·down 선형 + sigmoid(sgate) 게이트
//! │              │        │ (L313). 라우터·shared 게이트 입력은 EXL3
//! │              │        │ mlp.gate.weight F16[512,2560] — 트렐리스
//! │              │        │ 아님(실측 2026-10-05).
//! │ PLE 해시     │ HOST   │ 호스트 u64 해시(mixed%vs+offs — 원천:
//! │              │        │ stages/ple.rs ple_hash_rows L276-335·
//! │              │        │ ple_hash L337-360 상태 진화 포함). GPU u64
//! │              │        │ 나머지 경제성 없음 + 값 경로와 동일 계약
//! │              │        │ 유지(청크 lookback hist0 스냅샷 규약 L296).
//! │ PLE 게더     │ NEW    │ n-gram 임베딩 테이블 행 오프셋 판독 스테이징.
//! │              │        │ EXL3: ngram_embedding.safetensors trellis
//! │              │        │ [320001536,61] I16(37GiB — 전량 상주 금지,
//! │              │        │ pread 스테이징: AGENTS plans/86 §6 mmap
//! │              │        │ fault-in 금지). GGUF: PLE 테이블 — 원천:
//! │              │        │ mod.rs ple_gather_parts L471(block_info 기반
//! │              │        │ 행 바이트 오프셋)·ple.rs L60-90(스레드 게더).
//! │ PLE 수학     │ NEW    │ key/query/conv grouped rms(원천: ple.rs
//! │              │        │ L119-135, hc.rs grouped_rms 재사용) →
//! │              │        │ sgn√|s| 게이트(dot/√n_embd·mag=max(|dot|,
//! │              │        │ 1e-6).sqrt()·sigmoid(±mag) — ple.rs
//! │              │        │ L148-150) → value 방송×게이트 → dilated
//! │              │        │ conv(kern=4·dil=3·hist=9, 시퀀스 상태 —
//! │              │        │ ple.rs L169-195; GDN conv 링과 동일한
//! │              │        │ "한 런치 체인" 계급) → silu → 잔차 2경로
//! │              │        │ (ple.rs L217-235: row += value·g + conv).
//! │ PLE 투영     │ REUSE  │(조건부) ple_key/ple_value 선형 — gemv/gemm2
//! │              │        │ ·q4 MMQ.
//! │ 프레임 디코드│ NEW    │ 체인(FNH): decode t=1(원천: frame/forward.rs
//! │ 체인         │        │ frame_forward_ex L65·decode_frame L669)·np
//! │              │        │ 배치(frame/np.rs L418)·멀티 프리필 row band
//! │              │        │ **한 번의 커널 호출 체인**(frame/multi.rs
//! │              │        │ L268 + PreViews frame/mod.rs L77-105 불변
//! │              │        │ 식)·MTP 드래프트(frame/mtp.rs L362)·검증
//! │              │        │ (frame/verify.rs). 스테이지 커널(FNB-FNG)
//! │              │        │ 산출물을 버퍼 배선으로 조립 — 원천은 버퍼
//! │              │        │ 명명(Frame4 frame/mod.rs L106)까지 포함.
//! │ ew/argmax/   │ REUSE  │ silu·mul=exl3_ew(G7) — MoE/HC/PLE 활성화.
//! │ norm 재사용  │        │ argmax=exl3_argmax(G7, n=로짓 길이 —
//! │              │        │ Flash-Next vocab 248320으로 27B와 동일,
//! │              │        │ 결함 8호). 노름 코어=exl3_norm_resid(G3)·
//! │              │        │ exl3_mtp_rms(G9).
//! └──────────────┴────────┴──────────────────────────────────────────────┘
//!
//! [프레임 게이팅 계약] env LLM170_FRAME*은 core 자체 프레임 경로의 게이트
//! (layers.rs frame_env_on L196-210) — rawcuda는 **산술만 미러**하며 게이팅
//! 을 재현하지 않는다(과제 계약·plans/124 §0 ENV 계산 경로 분기 금지).
//! layers.rs L1604: CPU 상태 제로화 직후 프레임 GPU 상태는 stale — pull 금지
//! (fn 모듈도 dirty 재동기 계약 필요). core의 "미지원" 에러 문자열
//! (stages/mod.rs Ctx mm 폴백 매칭)은 load-bearing — 번역·변경 금지.
//!
//! ═══════════════════════════════════════════════════════════════════════
//! [픽스처 적재 지도 — Hparams4/Model4 대응(양 모델 세트)]
//! ═══════════════════════════════════════════════════════════════════════
//! · GGUF 본체(4-샤드): 샤드1=메타 전용(nten=0·67kv — qwen4exp.* 하이퍼
//!   파라미터 전부), 샤드2-4=텐서 297/752/175(합 1224=split.tensors.count).
//!   kv→Hparams4 필드 대응은 core Model4::hparams(mod.rs L218-315)가
//!   단일 진실 — FnDims::from_gguf가 동일 키 세트를 읽는다. 타입 믹스
//!   (F32/Q4_K/Q5_K/Q5_1/Q8_0/IQ4_NL/BF16)는 FnGgufTensor.ty로 보존.
//! · GGUF MTP(mtp-*-shared-Q4_K_M|Q8_0): load_mtp 계약(mod.rs L347) —
//!   nextn_predict_layers==1·blk.{n_layer}(=48) 텐서만·compress_ratios에
//!   본체+1 원소(블록 48 ratio). GGUF 본체에는 nextn 텐서 없음.
//! · GGUF mmproj(mmproj-BF16.gguf): arch=clip 비전(clip.rs 소비 — FNA
//!   범위 밖, 인벤토리만).
//! · EXL3(Qwen3.8-Flash-Next-exl3-5.05bpw): 선형은 trellis 3중(suh F16/
//!   trellis I16/svh F16 + mul1 I32 스칼라) — krate=trellis ne[2]/16
//!   (exl3_cuda.rs load_linear_from_archive L950와 동일 산출). **krate
//!   최대 7 실측**(in_proj_qkv·shared_expert.down_proj tw=112) — G2
//!   커널 krate≤6 상한 초과, 위 지도 참조. 키 접두 model.language_model.
//!   layers.{il}.* ↔ GGUF blk.{il}.* 대응(linear_attn=GDN·attn_=QSA·
//!   mlp.{gate,experts,shared_expert}=MoE·ple.*=PLE·{attn,mlp}_
//!   hyper_connection=HC). MTP는 mtp.layers.0.*(샤드11 수록)·mtp_hyper_
//!   connection_mixer_patch.safetensors(13MB 별도). norm류는 F16/BF16
//!   원값(w−1 규약 없음 — GGUF f32와 동일 취급).
//!   ngram_embedding.safetensors(37GiB)는 index weight_map 밖 단독 파일
//!   — 헤더(920B)만 오프셋 판독, 행 게더는 FNG.
//!   PLE 층 인덱스 실측: EXL3 텐서 layers.1.ple.*·GGUF ple.layers=[1] 일치 —
//!   config.json ple_layer_ids=[2]는 HF측 표기 차이(FNG 이름 매핑 시 주의).
//!   quantization_config.json(92MB)은 스트림 파싱으로 상위 스칼라만
//!   추출(전량 적재 금지 — 과제 계약).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지 — scripts/
//! cuda_probe_shim.rs 단독 컴파일. safetensors 샤드 리더는 exl3_cuda.rs
//! StArchive(mtp_cuda와 동일 import 계약)를, GGUF 리더는 q4_cuda_probe.rs
//! GgufReader의 샤드 인지 확장판을 내장한다.

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::exl3_cuda::{JParser, JVal, StArchive};
use std::collections::HashMap;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// 기본 픽스처 경로(과제 지정 — 프로브 인자로 대체 가능).
pub const FN_EXL3_DIR: &str = "D:/models/Qwen3.8-Flash-Next-exl3-5.05bpw";
pub const FN_GGUF_MAIN: &str =
    "D:/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
pub const FN_GGUF_MMPROJ: &str = "D:/models/qwen3.8-Flash-Next/mmproj-BF16.gguf";
pub const FN_GGUF_MTP: &str =
    "D:/models/qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf";

// ── Flash-Next 형상(Hparams4 미러 — 값은 픽스처에서, 산술은 스테이지 목표) ──

/// qwen4exp 하이퍼파라미터(GGUF qwen4exp.* kv / EXL3 config.json text_config
/// 양원 — core Hparams4(mod.rs L51)의 CUDA층 필요 서브셋 미러).
/// 필드 대응은 Model4::hparams(mod.rs L218-315)가 단일 진실.
#[derive(Debug, Clone, PartialEq)]
pub struct FnDims {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub eps: f32,
    /// 본체 kv에 없음(0) — output.weight ne[1]에서 확정(인벤토리 실측).
    pub vocab: usize,
    /// GDN: d_inner(=ssm.inner_size).
    pub d_inner: usize,
    /// GDN: dt_rank(=ssm.time_step_rank — 값헤드 수와 동일 48).
    pub dt_rank: usize,
    /// GDN: d_state(=ssm.state_size — 128 고정 실측).
    pub d_state: usize,
    /// GDN: n_group(=ssm.group_count — k/q 헤드 수 16).
    pub n_group: usize,
    /// GDN: conv_k(=ssm.conv_kernel — **4**: 27B EXL3의 3과 상이).
    pub conv_k: usize,
    /// MoE: n_expert·n_expert_used·n_ff_exp·n_ff_shared.
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shared: usize,
    /// HC: hc(스트림 수 4)·hc_low_rank(320).
    pub hc: usize,
    pub hc_low_rank: usize,
    /// QSA 인덱서: idx_heads·idx_dim·idx_top_k.
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub idx_top_k: usize,
    /// compress[il] — 0=GDN·4=QSA(full_attention_interval 곱).
    pub compress: Vec<i32>,
    /// PLE 대상 층(ple.layers).
    pub ple_layers: Vec<usize>,
    pub ple_ngram: usize,
    pub ple_heads_per_ngram: usize,
    pub ple_conv_k: usize,
    pub ple_head_dim: usize,
}

impl FnDims {
    /// GDN conv 채널 수 — 원천: stages/gdn.rs L34(2·k_len+v_len).
    pub fn gdn_conv_ch(&self) -> usize {
        self.n_group * self.d_state * 2 + self.dt_rank * self.d_state
    }
    /// GDN v 폭 — dt_rank·d_state(=d_inner 실측 6144).
    pub fn gdn_v_len(&self) -> usize {
        self.dt_rank * self.d_state
    }
    /// QSA 블록 압축비 r = compress[il](QSA층 4).
    pub fn qsa_r(&self, il: usize) -> usize {
        self.compress[il] as usize
    }
    /// is_recr/is_ple/kq_scale 미러 — 원천: mod.rs Hparams4 L96-107.
    pub fn is_recr(&self, il: usize) -> bool {
        self.compress[il] == 0
    }
    pub fn is_ple(&self, il: usize) -> bool {
        self.ple_layers.contains(&il)
    }
    /// kq 스케일 1/√head_dim — 원천: mod.rs L104-106.
    pub fn kq_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// GGUF kv → FnDims — Model4::hparams(mod.rs L218-315)와 동일 키 세트.
    /// 키 누락은 Err(형상 추정 금지 — 결함 1호 정신).
    pub fn from_gguf(g: &FnGguf) -> Result<Self, String> {
        let need = |k: &str| g.kv_u64(k).ok_or_else(|| format!("gguf meta 누락: {k}"));
        let n_layer = need("qwen4exp.block_count")? as usize;
        let compress = g
            .kv_arr_i32("qwen4exp.attention.compress_ratios")
            .ok_or("gguf meta 누락: compress_ratios")?;
        if compress.len() != n_layer {
            return Err(format!(
                "compress_ratios 길이 {} ≠ block_count {n_layer}",
                compress.len()
            ));
        }
        let ple_layers = g
            .kv_arr_i32("qwen4exp.ple.layers")
            .ok_or("gguf meta 누락: ple.layers")?;
        Ok(FnDims {
            n_layer,
            n_embd: need("qwen4exp.embedding_length")? as usize,
            n_head: need("qwen4exp.attention.head_count")? as usize,
            n_kv: need("qwen4exp.attention.head_count_kv")? as usize,
            head_dim: need("qwen4exp.attention.key_length")? as usize,
            n_rot: need("qwen4exp.rope.dimension_count")? as usize,
            rope_base: g.kv_f64("qwen4exp.rope.freq_base").unwrap_or(1e7) as f32,
            eps: g
                .kv_f64("qwen4exp.attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-6) as f32,
            vocab: 0,
            d_inner: need("qwen4exp.ssm.inner_size")? as usize,
            dt_rank: need("qwen4exp.ssm.time_step_rank")? as usize,
            d_state: need("qwen4exp.ssm.state_size")? as usize,
            n_group: need("qwen4exp.ssm.group_count")? as usize,
            conv_k: need("qwen4exp.ssm.conv_kernel")? as usize,
            n_expert: need("qwen4exp.expert_count")? as usize,
            n_expert_used: need("qwen4exp.expert_used_count")? as usize,
            n_ff_exp: need("qwen4exp.expert_feed_forward_length")? as usize,
            n_ff_shared: need("qwen4exp.expert_shared_feed_forward_length")? as usize,
            hc: need("qwen4exp.hyper_connection.count")? as usize,
            hc_low_rank: need("qwen4exp.hyper_connection.low_rank")? as usize,
            idx_heads: need("qwen4exp.attention.indexer.head_count")? as usize,
            idx_dim: need("qwen4exp.attention.indexer.key_length")? as usize,
            idx_top_k: need("qwen4exp.attention.indexer.top_k")? as usize,
            compress: compress.to_vec(),
            ple_layers: ple_layers.iter().map(|&v| v as usize).collect(),
            ple_ngram: need("qwen4exp.ple.ngram_size")? as usize,
            ple_heads_per_ngram: need("qwen4exp.ple.heads_per_ngram")? as usize,
            ple_conv_k: need("qwen4exp.ple.conv_kernel")? as usize,
            ple_head_dim: need("qwen4exp.embedding_length_per_layer_input")? as usize,
        })
    }
}

// ── EXL3 픽스처: ngram_embedding.safetensors 헤더(오프셋 판독 전용) ──

/// ngram_embedding.safetensors(37GiB) 헤더 뷰 — 파일은 절대 전량 적재하지
/// 않는다(920B 헤더만 판독). 원천 산술(PLE 게더)은 FNG — 본 구조체는
/// 메타(I64 헤더 3종)와 head_bias(F16 16×160) 오프셋 리드만 제공.
pub struct FnNgramHead {
    pub path: PathBuf,
    pub data_base: u64,
    /// head_bias: F16 [16,160] — data_offsets.
    pub bias_off: (u64, u64),
    /// head_offsets·head_vocab_sizes·layer_multipliers(I64 소형 벡터).
    pub head_offsets: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
    pub layer_multipliers: Vec<u64>,
    /// trellis I16 [rows, 61] — 행 수·열 수와 data_offsets(FNG 참조용).
    pub trellis_shape: (u64, u64),
    pub trellis_off: (u64, u64),
}

impl FnNgramHead {
    /// 디렉터리에서 ngram_embedding.safetensors 헤더만 판독.
    pub fn open(dir: &Path) -> Result<Self, String> {
        let path = dir.join("ngram_embedding.safetensors");
        let mut f = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut lenb = [0u8; 8];
        f.read_exact(&mut lenb).map_err(|e| e.to_string())?;
        let hlen = u64::from_le_bytes(lenb);
        if hlen == 0 || hlen > (1 << 22) {
            return Err(format!("ngram 헤더 길이 {hlen} — 가드(실측 920)"));
        }
        let mut hb = vec![0u8; hlen as usize];
        f.read_exact(&mut hb).map_err(|e| e.to_string())?;
        let data_base = 8 + hlen;
        let v = JParser { b: &hb, p: 0 }.parse()?;
        let obj = v.as_obj().ok_or("ngram: 헤더가 객체 아님")?;
        let mut h = FnNgramHead {
            path,
            data_base,
            bias_off: (0, 0),
            head_offsets: Vec::new(),
            head_vocab_sizes: Vec::new(),
            layer_multipliers: Vec::new(),
            trellis_shape: (0, 0),
            trellis_off: (0, 0),
        };
        let mut seen = 0usize;
        for (name, tv) in obj {
            let offs = tv.get("data_offsets").and_then(JVal::as_arr);
            let shape = tv.get("shape").and_then(JVal::as_arr);
            let (Some(offs), Some(shape)) = (offs, shape) else {
                continue;
            };
            let g64 = |i: usize| offs.get(i).and_then(JVal::as_f64).map(|v| v as u64);
            if name.ends_with("head_bias") {
                h.bias_off = (g64(0).unwrap_or(0), g64(1).unwrap_or(0));
                seen += 1;
            } else if name.ends_with("head_offsets") {
                h.head_offsets = Self::read_i64s(&mut f, data_base, offs, shape)?;
                seen += 1;
            } else if name.ends_with("head_vocab_sizes") {
                h.head_vocab_sizes = Self::read_i64s(&mut f, data_base, offs, shape)?;
                seen += 1;
            } else if name.ends_with("layer_multipliers") {
                h.layer_multipliers = Self::read_i64s(&mut f, data_base, offs, shape)?;
                seen += 1;
            } else if name.ends_with("trellis") {
                h.trellis_shape = (
                    shape.first().and_then(JVal::as_f64).unwrap_or(0.0) as u64,
                    shape.get(1).and_then(JVal::as_f64).unwrap_or(0.0) as u64,
                );
                h.trellis_off = (g64(0).unwrap_or(0), g64(1).unwrap_or(0));
                seen += 1;
            }
        }
        if seen != 5 {
            return Err(format!("ngram: 인지 텐서 {seen}/5 — 레이아웃 변경 가드"));
        }
        Ok(h)
    }

    /// I64 소형 벡터(16·3원소) 직독 — 헤더 오프셋 판독 증명 겸용.
    fn read_i64s(
        f: &mut std::fs::File,
        data_base: u64,
        offs: &[JVal],
        shape: &[JVal],
    ) -> Result<Vec<u64>, String> {
        let b = offs
            .first()
            .and_then(JVal::as_f64)
            .ok_or("ngram: data_offsets[0]")? as u64;
        let e = offs
            .get(1)
            .and_then(JVal::as_f64)
            .ok_or("ngram: data_offsets[1]")? as u64;
        let n = shape
            .first()
            .and_then(JVal::as_f64)
            .ok_or("ngram: shape[0]")? as usize;
        if e - b != n as u64 * 8 || n > 4096 {
            return Err(format!("ngram: I64 벡터 {n}원소 — 가드"));
        }
        f.seek(SeekFrom::Start(data_base + b))
            .map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; n * 8];
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
        Ok(buf
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    /// head_bias(F16 [16,160] = 5120B) 직독 — PLE 상수 업로드·인벤토리 증명.
    pub fn read_head_bias(&self) -> Result<Vec<u8>, String> {
        let mut f = std::fs::File::open(&self.path).map_err(|e| e.to_string())?;
        f.seek(SeekFrom::Start(self.data_base + self.bias_off.0))
            .map_err(|e| e.to_string())?;
        let n = (self.bias_off.1 - self.bias_off.0) as usize;
        let mut buf = vec![0u8; n];
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
        Ok(buf)
    }
}

// ── EXL3 픽스처: quantization_config.json 스트림 파서(92MB — 전량 적재 금지) ──

/// quantization_config.json 상위 요약 — 청크 스트림 파싱 산출물.
#[derive(Debug, Clone, PartialEq)]
pub struct FnQuantSummary {
    pub quant_method: String,
    pub version: String,
    pub bits: f64,
    pub head_bits: f64,
    pub codebook: String,
    pub out_scales: String,
    pub vision_bits: f64,
    pub mtp_bits: f64,
    /// tensor_storage 객체의 모듈 항목 수(모듈별 양자 설정 수).
    pub tensor_storage_entries: usize,
    /// 스트림이 실제로 소비한 바이트(파일 크기 대조용).
    pub consumed_bytes: u64,
}

/// 상위(깊이1) 스칼라 키만 추출하는 스트림 파서 — 92MB를 JVal 트리로 만들지
/// 않는다(과제 계약). 상태: 깊이·문자열·키 위치(콜론 이전=키/이후=값) 추적.
/// tensor_storage(깊이1 객체) 내부 항목은 "키:{" 여닫이 계수로 집계.
pub fn fn_quant_config_stream(path: &Path) -> Result<FnQuantSummary, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut vals: HashMap<String, String> = HashMap::new();
    for w in [
        "quant_method",
        "version",
        "bits",
        "head_bits",
        "codebook",
        "out_scales",
        "vision_bits",
        "mtp_bits",
    ] {
        vals.insert(w.to_string(), String::new());
    }
    let mut entries = 0usize;
    let mut consumed = 0u64;
    let mut depth: usize = 0;
    let mut in_str = false;
    let mut esc = false;
    let mut cur_str = String::new();
    // 깊이별 상태: 콜론 이후(값 기대) 여부.
    let mut after_colon: Vec<bool> = vec![false];
    // 깊이별 대기 키(문자열이 닫힌 뒤 콜론 전).
    let mut pend_key: Vec<Option<String>> = vec![None];
    // 깊이1 스칼라값 캡처 중인 키.
    let mut cap_key: Option<String> = None;
    let mut val_buf = String::new();
    // tensor_storage 객체의 깊이(2 고정 — 열림 시 설정).
    let mut ts_depth: usize = 0;
    // tensor_storage 내부 "키:" 뒤 객체 값 시작 대기.
    let mut ts_pending = false;
    let mut buf = [0u8; 65536];
    'outer: loop {
        let n = r.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("quant config: 예기지 않은 EOF".into());
        }
        consumed += n as u64;
        for &c in &buf[..n] {
            if in_str {
                if esc {
                    esc = false;
                } else if c == b'\\' {
                    esc = true;
                } else if c == b'"' {
                    in_str = false;
                    if !after_colon[depth] && pend_key[depth].is_none() {
                        pend_key[depth] = Some(std::mem::take(&mut cur_str));
                    }
                } else if depth == 1 && after_colon[1] && cap_key.is_some() {
                    // 깊이1 스칼라 문자열값 — val_buf에 직접 캡처.
                    val_buf.push(c as char);
                } else if pend_key[depth].is_none() && !after_colon[depth] {
                    cur_str.push(c as char);
                }
                continue;
            }
            match c {
                b'"' => {
                    in_str = true;
                    cur_str.clear();
                }
                b'{' => {
                    // 값 시작이 객체: (a) 깊이1 캡처 대기 키였으면 스킵 대상 —
                    // 단 tensor_storage이면 깊이 추적 시작(키는 ':'에서
                    // cap_key로 이미 이동했으므로 여기서 판독),
                    // (b) tensor_storage 내부 "키:" 뒤 객체면 항목 계수.
                    if depth == 1 {
                        if let Some(k) = cap_key.take()
                            && k == "tensor_storage"
                        {
                            ts_depth = 2;
                        }
                    } else if depth == 2 && ts_pending {
                        entries += 1;
                        ts_pending = false;
                    }
                    depth += 1;
                    after_colon.push(false);
                    pend_key.push(None);
                }
                b'}' => {
                    if depth == 1 {
                        // 마지막 멤버(콜론 뒤 쉽표 없음)도 누락 없이 확정.
                        if let Some(k) = cap_key.take()
                            && !val_buf.trim().is_empty()
                        {
                            vals.insert(k, val_buf.trim().to_string());
                        }
                    }
                    depth -= 1;
                    after_colon.truncate(depth + 1);
                    pend_key.truncate(depth + 1);
                    if ts_depth > 0 && depth < ts_depth {
                        ts_depth = 0;
                        ts_pending = false;
                    }
                    if depth == 0 {
                        break 'outer;
                    }
                }
                b'[' => {
                    if depth == 1 {
                        pend_key[1] = None;
                        cap_key = None;
                    }
                    depth += 1;
                    after_colon.push(false);
                    pend_key.push(None);
                }
                b']' => {
                    depth -= 1;
                    after_colon.truncate(depth + 1);
                    pend_key.truncate(depth + 1);
                }
                b':' => {
                    after_colon[depth] = true;
                    if depth == 1 {
                        cap_key = pend_key[1].take();
                        val_buf.clear();
                    } else if depth == 2 && ts_depth == 2 {
                        pend_key[2] = None;
                        ts_pending = true;
                    }
                }
                b',' => {
                    if depth == 1
                        && let Some(k) = cap_key.take()
                        && !val_buf.trim().is_empty()
                    {
                        vals.insert(k, val_buf.trim().to_string());
                    }
                    after_colon[depth] = false;
                    ts_pending = false;
                }
                _ => {
                    if cap_key.is_some() {
                        val_buf.push(c as char);
                    }
                }
            }
        }
    }
    let gs = |k: &str| -> Result<String, String> {
        vals.get(k)
            .filter(|v| !v.is_empty())
            .cloned()
            .ok_or_else(|| format!("quant config: {k} 없음"))
    };
    let gn = |k: &str| -> Result<f64, String> {
        gs(k)?
            .trim_matches('"')
            .parse::<f64>()
            .map_err(|e| format!("quant config {k}: {e}"))
    };
    Ok(FnQuantSummary {
        quant_method: gs("quant_method")?.trim_matches('"').to_string(),
        version: gs("version")?.trim_matches('"').to_string(),
        bits: gn("bits")?,
        head_bits: gn("head_bits")?,
        codebook: gs("codebook")?.trim_matches('"').to_string(),
        out_scales: gs("out_scales")?.trim_matches('"').to_string(),
        vision_bits: gn("vision_bits")?,
        mtp_bits: gn("mtp_bits")?,
        tensor_storage_entries: entries,
        consumed_bytes: consumed,
    })
}

// ── GGUF 픽스처: 샤드 인지 v3 헤더 파서(q4_cuda_probe GgufReader 샤드 확장) ──

/// GGUF 텐서 항목 — shard·샤드 내 상대 오프셋 포함(절대 위치는
/// data_base[shard]+off로 산출 — read_rows 참조).
#[derive(Debug, Clone)]
pub struct FnGgufTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: u32,
    pub shard: usize,
    /// 해당 샤드 내 상대 데이터 오프셋(GGUF 규격 off).
    pub off: u64,
}

/// 캡처한 kv 스칼라/소형 배열(필요 키만 — 대형 tokenizer류는 스킵).
#[derive(Debug, Clone)]
pub enum FnGgufVal {
    F32(f32),
    U32(u32),
    U64(u64),
    Str(String),
    ArrI32(Vec<i32>),
    ArrU64(Vec<u64>),
}

/// GGUF v3 샤드 아카이브 뷰 — 헤더만 판독(데이터는 read_rows로 오프셋 직독).
pub struct FnGguf {
    pub shards: Vec<PathBuf>,
    /// 샤드별 데이터 시작 바이트(alignment 정렬 후).
    pub data_base: Vec<u64>,
    pub tensors: Vec<FnGgufTensor>,
    index: HashMap<String, usize>,
    kv: HashMap<String, FnGgufVal>,
    pub tensor_count_split: u64,
}

/// ggml 타입표 — crates/gguf/src/types.rs block_info(L134-168) 미러.
/// (blck_size, type_size) — 원본 표와 동일 값(계약: 이 표를 "정리" 금지).
fn gguf_block_info(ty: u32) -> Result<(u64, u64), String> {
    Ok(match ty {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        30 => (1, 2),     // BF16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        20 => (32, 18),   // IQ4_NL
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        t => return Err(format!("gguf 타입 {t} — 미정의(픽스처 범위 밖)")),
    })
}

/// 스트리밍 판독기 — q4_cuda_probe.rs GgufReader(L345) 미러(위치 추적).
struct FnGgufReader<R: Read> {
    r: R,
    pos: u64,
}

impl<R: Read> FnGgufReader<R> {
    fn new(r: R) -> Self {
        FnGgufReader { r, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<Vec<u8>, String> {
        let mut b = vec![0u8; n];
        self.r
            .read_exact(&mut b)
            .map_err(|e| format!("gguf 판독(pos={} n={n}): {e}", self.pos))?;
        self.pos += n as u64;
        Ok(b)
    }
    fn u32v(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64v(&mut self) -> Result<u64, String> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64v()? as usize;
        if n > (1 << 20) {
            return Err(format!("gguf 문자열 길이 {n} — 손상 파일 가드"));
        }
        let b = self.take(n)?;
        String::from_utf8(b).map_err(|e| format!("gguf 문자열 utf8: {e}"))
    }
    /// 값 1개 스킵(배열은 원소 재귀 — 문자열 배열 포함).
    fn skip_value(&mut self, ty: u32) -> Result<(), String> {
        const S: [u64; 13] = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8];
        match ty {
            8 => {
                self.string()?;
            }
            9 => {
                let et = self.u32v()?;
                let cnt = self.u64v()?;
                if S.get(et as usize).copied().unwrap_or(0) > 0 {
                    self.take((S[et as usize] * cnt) as usize)?;
                } else {
                    for _ in 0..cnt {
                        self.skip_value(et)?;
                    }
                }
            }
            t => {
                let s = S.get(t as usize).copied().unwrap_or(0);
                if s == 0 {
                    return Err(format!("gguf 값 타입 {t} — v3 규격 위반"));
                }
                self.take(s as usize)?;
            }
        }
        Ok(())
    }
}

impl FnGguf {
    /// 첫 샤드 경로로 개방 — split.count 규격에 따라 전 샤드 헤더 병합.
    /// 파일명 패턴 -0000N-of-0000M 교체는 core Model4::load(mod.rs L144-156)
    /// 와 동일 규칙(패턴 보존 필수).
    pub fn open(first: &Path) -> Result<Self, String> {
        let (kv, tensors0, data_base0, split_count, split_tensors) = Self::scan_shard(first, true)?;
        let mut shards = vec![first.to_path_buf()];
        let mut data_base = vec![data_base0];
        let mut tensors = tensors0;
        if split_count > 1 {
            let stem = first
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or("gguf: 경로 문자열화 실패")?;
            for i in 2..=split_count {
                let pat = format!("-{:05}-of-{:05}", 1, split_count);
                let rep = format!("-{:05}-of-{:05}", i, split_count);
                let name = stem.replace(&pat, &rep);
                let p = first.with_file_name(name);
                // 후속 샤드 kv는 split 3종뿐(실측) — kv 캡처 없이 헤더만.
                let (_, mut ts, db, _, _) = Self::scan_shard(&p, false)?;
                for t in ts.iter_mut() {
                    t.shard = (i - 1) as usize;
                }
                shards.push(p);
                data_base.push(db);
                tensors.extend(ts);
            }
        }
        let total = tensors.len() as u64;
        if split_count > 1 && split_tensors > 0 && total != split_tensors {
            return Err(format!(
                "gguf 샤드 병합 {total} ≠ split.tensors.count {split_tensors}"
            ));
        }
        let index = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        Ok(FnGguf {
            shards,
            data_base,
            tensors,
            index,
            kv,
            tensor_count_split: split_tensors,
        })
    }

    /// 샤드 1개 헤더 스캔. first=true면 kv 캡처(arch·qwen4exp.*·split.*)·
    /// alignment 획득, 아니면 kv 스킵(후속 샤드는 split 3kv 실측 —
    /// alignment는 기본 32 규격값 사용).
    fn scan_shard(
        path: &Path,
        first: bool,
    ) -> Result<(HashMap<String, FnGgufVal>, Vec<FnGgufTensor>, u64, u32, u64), String> {
        let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut g = FnGgufReader::new(BufReader::with_capacity(1 << 23, f));
        let magic = g.take(4)?;
        if magic != b"GGUF" {
            return Err("gguf: 매직 불일치 — GGUF 아님".into());
        }
        let ver = g.u32v()?;
        if ver != 3 {
            return Err(format!("gguf: 버전 {ver} — v3 전용 계약"));
        }
        let nten = g.u64v()?;
        let nkv = g.u64v()?;
        if nten > 1_000_000 || nkv > 100_000 {
            return Err(format!("gguf: 카운트 비정상({nten}/{nkv}) — 손상 가드"));
        }
        // GGUF v3 값 타입 크기: 0 U8·1 I8·2 U16·3 I16·4 U32·5 I32·6 F32·
        // 7 BOOL·8 STR·9 ARR·10 U64·11 I64·12 F64.
        const S: [u64; 13] = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8];
        let mut kv = HashMap::new();
        for _ in 0..nkv {
            let key = g.string()?;
            let ty = g.u32v()?;
            // 캡처 대상: 첫 샤드의 arch/alignment/split.*/qwen4exp.*.
            let want = first
                && (key == "general.architecture"
                    || key == "general.alignment"
                    || key.starts_with("split.")
                    || key.starts_with("qwen4exp."));
            if !want {
                g.skip_value(ty)?;
                continue;
            }
            let b4 = |g: &mut FnGgufReader<BufReader<std::fs::File>>| -> Result<[u8; 4], String> {
                let b = g.take(4)?;
                Ok([b[0], b[1], b[2], b[3]])
            };
            let b8 = |g: &mut FnGgufReader<BufReader<std::fs::File>>| -> Result<[u8; 8], String> {
                let b = g.take(8)?;
                Ok([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
            };
            match ty {
                8 => {
                    kv.insert(key, FnGgufVal::Str(g.string()?));
                }
                9 => {
                    let et = g.u32v()?;
                    let cnt = g.u64v()?;
                    // 소형 정수 배열만 캡처(하이퍼파라미터 — 그 외 스킵).
                    if cnt <= 4096 && matches!(et, 4 | 5 | 10 | 11) {
                        let b = g.take((S[et as usize] * cnt) as usize)?;
                        if et == 4 || et == 5 {
                            let v = b
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect();
                            kv.insert(key, FnGgufVal::ArrI32(v));
                        } else {
                            let v = b
                                .as_chunks::<8>()
                                .0
                                .iter()
                                .map(|c| {
                                    u64::from_le_bytes([
                                        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                                    ])
                                })
                                .collect();
                            kv.insert(key, FnGgufVal::ArrU64(v));
                        }
                    } else {
                        for _ in 0..cnt {
                            g.skip_value(et)?;
                        }
                    }
                }
                6 => {
                    let b = b4(&mut g)?;
                    kv.insert(key, FnGgufVal::F32(f32::from_le_bytes(b)));
                }
                4 => {
                    let b = b4(&mut g)?;
                    kv.insert(key, FnGgufVal::U32(u32::from_le_bytes(b)));
                }
                5 => {
                    let b = b4(&mut g)?;
                    kv.insert(key, FnGgufVal::U32(i32::from_le_bytes(b) as u32));
                }
                10 | 11 => {
                    let b = b8(&mut g)?;
                    kv.insert(key, FnGgufVal::U64(u64::from_le_bytes(b)));
                }
                0 | 7 => {
                    let b = g.take(1)?;
                    kv.insert(key, FnGgufVal::U64(b[0] as u64));
                }
                2 | 3 => {
                    let b = g.take(2)?;
                    kv.insert(key, FnGgufVal::U64(u16::from_le_bytes([b[0], b[1]]) as u64));
                }
                12 => {
                    return Err("gguf kv f64 — 픽스처 계약 밖(발견 시 추가)".into());
                }
                t => {
                    let s = S.get(t as usize).copied().unwrap_or(0);
                    if s == 0 {
                        return Err(format!("gguf kv 타입 {t} — v3 규격 위반"));
                    }
                    g.take(s as usize)?;
                }
            }
        }
        let mut tensors = Vec::with_capacity(nten as usize);
        for _ in 0..nten {
            let name = g.string()?;
            let nd = g.u32v()? as usize;
            if nd == 0 || nd > 4 {
                return Err(format!("gguf: 텐서 {name} n_dims={nd} — 가드"));
            }
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(g.u64v()?);
            }
            let ty = g.u32v()?;
            let off = g.u64v()?;
            tensors.push(FnGgufTensor {
                name,
                dims,
                ty,
                shard: 0,
                off,
            });
        }
        let mut align = 32u64;
        if first && let Some(FnGgufVal::U32(v)) = kv.get("general.alignment") {
            align = *v as u64;
        }
        let data_base = g.pos.div_ceil(align) * align;
        let split_count = match kv.get("split.count") {
            Some(FnGgufVal::U64(v)) => *v as u32,
            Some(FnGgufVal::U32(v)) => *v,
            _ => 1,
        };
        let split_tensors = match kv.get("split.tensors.count") {
            Some(FnGgufVal::U64(v)) => *v,
            Some(FnGgufVal::U32(v)) => *v as u64,
            _ => 0,
        };
        Ok((kv, tensors, data_base, split_count, split_tensors))
    }

    /// kv 접근자 — 숫자 계열은 U64/U32/F32 상호 강제 변환.
    pub fn kv_u64(&self, key: &str) -> Option<u64> {
        match self.kv.get(key) {
            Some(FnGgufVal::U64(v)) => Some(*v),
            Some(FnGgufVal::U32(v)) => Some(*v as u64),
            Some(FnGgufVal::F32(v)) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn kv_f64(&self, key: &str) -> Option<f64> {
        match self.kv.get(key) {
            Some(FnGgufVal::F32(v)) => Some(*v as f64),
            Some(FnGgufVal::U64(v)) => Some(*v as f64),
            Some(FnGgufVal::U32(v)) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn kv_str(&self, key: &str) -> Option<&str> {
        match self.kv.get(key) {
            Some(FnGgufVal::Str(v)) => Some(v),
            _ => None,
        }
    }
    pub fn kv_arr_i32(&self, key: &str) -> Option<&[i32]> {
        match self.kv.get(key) {
            Some(FnGgufVal::ArrI32(v)) => Some(v),
            _ => None,
        }
    }
    pub fn kv_arr_u64(&self, key: &str) -> Option<&[u64]> {
        match self.kv.get(key) {
            Some(FnGgufVal::ArrU64(v)) => Some(v),
            _ => None,
        }
    }

    /// 텐서 행 바이트 수 — ne[0] 원소의 (blck,bsize) 환산.
    /// 원천: ple_gather_parts(mod.rs L478-483)와 동일 산식.
    pub fn row_bytes(t: &FnGgufTensor) -> Result<u64, String> {
        let (blck, bsize) = gguf_block_info(t.ty)?;
        Ok(t.dims.first().copied().unwrap_or(0).div_ceil(blck) * bsize)
    }

    /// 텐서 조회.
    pub fn tensor(&self, name: &str) -> Option<&FnGgufTensor> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// 행 오프셋 직독(row0..row0+nrows의 원시 바이트) — 전량 적재 금지 계약.
    pub fn read_rows(&self, name: &str, row0: u64, nrows: u64) -> Result<Vec<u8>, String> {
        let t = self
            .tensor(name)
            .ok_or_else(|| format!("gguf 텐서 없음: {name}"))?;
        let rb = Self::row_bytes(t)?;
        let rows_total = t.dims.get(1).copied().unwrap_or(1);
        if row0 + nrows > rows_total {
            return Err(format!(
                "gguf 판독: row0={row0}+{nrows} > 행수 {rows_total} ({name})"
            ));
        }
        let mut f = std::fs::File::open(&self.shards[t.shard]).map_err(|e| e.to_string())?;
        let abs = self.data_base[t.shard] + t.off + row0 * rb;
        f.seek(SeekFrom::Start(abs)).map_err(|e| e.to_string())?;
        let n = (rb * nrows) as usize;
        let mut buf = vec![0u8; n];
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
        Ok(buf)
    }
}

// ── Flash-Next CUDA 모듈 공유 본체(스켈레톤) ──

/// Flash-Next CUDA 디코더 스켈레톤 — CudaCtx 단일 소유(단일 상주 원칙,
/// 2026-10-04 동결 사고 계약: 한 프로세스 모델 1개).
/// FNB-FNH가 스테이지 버퍼 필드를 추가한다(G2 "G4+ 필드 0/null 유지"
/// 계약과 동일 규율) — 산술 구현은 이 스캐폴드에 없다.
pub struct FnCuda {
    /// 디바이스 컨텍스트(디바이스·스트림·커널 레지스트리).
    pub cc: CudaCtx,
    /// 형상(Hparams4 미러 — 픽스처 적재 시 확정).
    pub dims: FnDims,
    /// 어휘(로짓 길이 — argmax n 계약, 결함 8호).
    pub vocab: usize,
    /// 시퀀스 위치 호스트 사본 — 진실은 FNB 소유 디바이스 pp(결함 4호).
    pub pos: u32,
}

impl FnCuda {
    /// 컨텍스트 개방 — dims는 FnGguf/EXL3 config에서 확정해 전달.
    pub fn new(dims: FnDims, vocab: usize) -> Result<Self, String> {
        Ok(FnCuda {
            cc: CudaCtx::new()?,
            dims,
            vocab,
            pos: 0,
        })
    }
}

// ── 오라클 스켈레톤(검증층 원천 인용 — 산술은 각 스테이지 목표가 구현) ──

/// Flash-Next 스테이지 오라클 스켈레톤 — 각 메서드는 대응 core 원천을
/// 그대로 재생하는 호스트 미러가 된다(값 maxdiff 판정의 참조측,
/// plans/124 §5). FNB-FNH 구현 전까지 Err(미구현).
pub struct FnOracle;

impl FnOracle {
    /// QSA 스테이지 오라클 — TODO(FNB): stages/qsa.rs qsa_layer L396-515
    /// (mm_group 5투영 → qsa_select L99 → q rope L330-357 → 어텐션 → wo)
    /// 전체를 값 경로 그대로. 음성대조 후보: sel 목록 누락·k 재 norm.
    pub fn qsa_layer(_il: usize, _xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        Err("미구현: FNB(fn_qsa_cuda) 목표".into())
    }
    /// HC mix 오라클 — TODO(FNC): stages/hc.rs hc_mix_ex L25-86(grouped
    /// rms·저랭크 silu(lo/hc)·게이트·스트림 평균) + combine(layers.rs).
    pub fn hc_mix(_il: usize, _kind: &str, _res_hc: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        Err("미구현: FNC(fn_hc_cuda) 목표".into())
    }
    /// GDN 스테이지 오라클 — TODO(FND): stages/gdn.rs gdn_layer L11-170
    /// (conv 링 L69-85·l2 L86-97·core/gdn.rs scan·norm_gated sigmoid).
    /// S0≠0 실입력 의무(합성 상태 가드 — plans/124 §3.3).
    pub fn gdn_layer(_il: usize, _xs: &[Vec<f32>], _t_len: usize) -> Result<Vec<Vec<f32>>, String> {
        Err("미구현: FND(fn_gdn_cuda) 목표".into())
    }
    /// MoE FFN 오라클 — TODO(FNE): stages/moe.rs moe_ffn L22-320(라우팅
    /// total_cmp·전문가 서브배치·shared sigmoid 게이트).
    pub fn moe_ffn(_il: usize, _xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        Err("미구현: FNE(fn_moe_cuda) 목표".into())
    }
    /// PLE 블록 오라클 — TODO(FNG): stages/ple.rs ple_block L25-235(게더·
    /// 게이트·dilated conv·잔차 2경로) + ple_hash L337(호스트 u64).
    pub fn ple_block(_il: usize, _tokens: &[u32]) -> Result<(), String> {
        Err("미구현: FNG(fn_ple_cuda) 목표".into())
    }
    /// 프레임 디코드 체인 오라클 — TODO(FNH): frame/forward.rs
    /// decode_frame L669·frame_forward_ex L65(스테이지 산출 배선 검증).
    pub fn decode_frame(_token: u32) -> Result<Vec<f32>, String> {
        Err("미구현: FNH(프레임 체인) 목표".into())
    }
}

// ── exl3_fn.fatbin 스모크 프로브(FNA — 배관 검증) ──

/// exl3_fn.fatbin 자산 해석 — LLM170_CUDA_FN_FATBIN_PATH 오버라이드 우선
/// (자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
fn fn_fatbin_bytes() -> Result<Vec<u8>, String> {
    const ENV: &str = "LLM170_CUDA_FN_FATBIN_PATH";
    const REL: &[&str] = &[
        "crates/backend-gpu/src/rawcuda/assets/exl3_fn.fatbin",
        "src/rawcuda/assets/exl3_fn.fatbin",
    ];
    if let Some(p) = std::env::var_os(ENV) {
        return std::fs::read(&p).map_err(|e| format!("{ENV}({p:?}) 읽기 실패: {e}"));
    }
    for r in REL {
        if let Ok(b) = std::fs::read(r) {
            return Ok(b);
        }
    }
    Err(format!(
        "exl3_fn.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
    ))
}

/// fn 스모크 — FNA 스캐폴드 배관 검증(fatbin 로드→런치→d2h→값).
/// 산술: out[i] = in[i]·hc_scale + i, in[i]=i·0.5, hc_scale=0.25(hc=4
/// 스트림 평균 계수 상징 — stages/hc.rs L78 /=hc). 곱·합 모두 f32 정확
/// 표현 범위라 -fmad=false 빌드와 무관하게 비트동일 판정(스모크는 배관
/// 검증 — 근사 허용 없음, smoke.cu와 동일 등급).
pub fn cuda_fn_smoke_check() -> Result<String, String> {
    let image = fn_fatbin_bytes()?;
    let mut cc = CudaCtx::new()?;
    let _g = cc.guard()?;
    cc.load_fatbin("exl3_fn", &image, &["llm170_fn_smoke"])?;
    let f = cc.function("llm170_fn_smoke")?;

    const N: usize = 4096;
    const BLOCK: u32 = 256;
    let hc_scale = 0.25f32;
    let input: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
    let din = cc.alloc(N * 4)?;
    let dout = cc.alloc(N * 4)?;
    // SAFETY: input은 길이 N*4 바이트의 f32 슬라이스 — 바이트 뷰 변환.
    let inb = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, N * 4) };
    cc.h2d(din, inb)?;

    let (mut a0, mut a1, mut a2, mut a3) = (din, dout, N as i32, hc_scale);
    let mut args: [*mut std::ffi::c_void; 4] = [
        (&mut a0) as *mut _ as *mut std::ffi::c_void,
        (&mut a1) as *mut _ as *mut std::ffi::c_void,
        (&mut a2) as *mut _ as *mut std::ffi::c_void,
        (&mut a3) as *mut _ as *mut std::ffi::c_void,
    ];
    cc.launch(f, (N as u32).div_ceil(BLOCK), 1, BLOCK, &mut args)?;
    cc.sync()?;

    let mut outb = vec![0u8; N * 4];
    cc.d2h(&mut outb, dout)?;
    // SAFETY: outb는 d2h가 채운 N*4 바이트 — f32 배열로 재해석(정렬·길이 일치).
    let out = unsafe { std::slice::from_raw_parts(outb.as_ptr() as *const f32, N) };
    let mut bad: Vec<usize> = Vec::new();
    for (i, &v) in out.iter().enumerate() {
        let want = input[i] * hc_scale + i as f32; // 커널과 동일 f32 연산
        if v.to_bits() != want.to_bits() && bad.len() < 4 {
            bad.push(i);
        }
    }
    cc.free(din)?;
    cc.free(dout)?;
    if !bad.is_empty() {
        return Err(format!(
            "fn 스모크 값 불일치 — {}/{}: {} (기대 비트동일)",
            bad.len().min(4),
            N,
            bad.iter()
                .map(|&i| format!("out[{i}]={:.6e}", out[i]))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    Ok(format!(
        "device: {} | flash-next scaffold PASS",
        cc.device_name
    ))
}

// ── 픽스처 인벤토리 프로브(양 로더 실증 — 오프셋 판독만) ──

/// EXL3 아카이브 인벤토리 — StArchive(index.json+11샤드 헤더) + ngram
/// 헤더 + quant 스트림. 전량 적재 없음.
fn fn_inv_exl3(dir: &Path) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let ar = StArchive::open(dir)?;
    // dtype 코드(StArchive 계약: 0 F32류·1 F16·2 BF16·3 기타2B(I16)).
    let mut dt = [0usize; 4];
    let mut trellis = 0usize;
    let mut krate_max: u64 = 0;
    let mut examples: Vec<String> = Vec::new();
    let ex_pat = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.trellis",
        "model.language_model.layers.0.mlp.experts.10.down_proj.trellis",
        "model.language_model.layers.1.ple.norm_key.weight",
        "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight",
        "model.language_model.layers.0.mlp.gate.weight",
    ];
    ar.each_tensor(|name, dtc, shape| {
        dt[dtc as usize] += 1;
        if name.ends_with(".trellis") {
            trellis += 1;
            if let Some(tw) = shape.get(2) {
                krate_max = krate_max.max(tw / 16);
            }
        }
        if ex_pat.contains(&name) {
            examples.push(format!(
                "  ex: {name} [{}]",
                shape
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
    });
    out.push(format!(
        "[fn-inv] EXL3 {} — tensors={} (F32류:{} F16:{} BF16:{} I16:{} · trellis선형 {} · krate≤{}) shards={}",
        dir.display(),
        ar.tensor_count(),
        dt[0],
        dt[1],
        dt[2],
        dt[3],
        trellis,
        krate_max,
        ar.shard_count(),
    ));
    examples.sort_by_key(|l| {
        ex_pat
            .iter()
            .position(|&p| l.contains(p.trim_start_matches("  ex: ")))
            .unwrap_or(99)
    });
    out.extend(examples);
    if krate_max > 6 {
        out.push(format!(
            "  [계약 지도] krate {krate_max} > 6 — G2 gemv/gemm2 스테이징 상한 초과(위 지도 REUSE(조건부) 항)"
        ));
    }
    // ngram(37GiB 단독 파일) — 헤더 920B만 + head_bias 오프셋 직독 증명.
    let ng = FnNgramHead::open(dir)?;
    let bias = ng.read_head_bias()?;
    let vocab_sum: u64 = ng.head_vocab_sizes.iter().sum();
    out.push(format!(
        "[fn-inv] ngram_embedding.safetensors — tensors=5 trellis I16 [{},{}] ({}B 전량적재금지) ngram={} heads={} vocab_sum={} · head_bias F16 {}B 직독 OK",
        ng.trellis_shape.0,
        ng.trellis_shape.1,
        ng.trellis_off.1 - ng.trellis_off.0,
        ng.layer_multipliers.len(),
        ng.head_offsets.len(),
        vocab_sum,
        bias.len(),
    ));
    // quantization_config.json(92MB) — 스트림 파싱.
    let q = fn_quant_config_stream(&dir.join("quantization_config.json"))?;
    out.push(format!(
        "[fn-inv] quantization_config.json (stream {}B consumed) — method={} bits={} head_bits={} codebook={} out_scales={} vision_bits={} mtp_bits={} tensor_storage_entries={}",
        q.consumed_bytes,
        q.quant_method,
        q.bits,
        q.head_bits,
        q.codebook,
        q.out_scales,
        q.vision_bits,
        q.mtp_bits,
        q.tensor_storage_entries,
    ));
    Ok(out)
}

/// GGUF 인벤토리 — 본체 4샤드 + mtp + mmproj, F32 행 오프셋 직독.
fn fn_inv_gguf(main: &Path, mtp: &Path, mmproj: &Path) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let g = FnGguf::open(main)?;
    let mut ty_cnt: HashMap<u32, usize> = HashMap::new();
    let mut shard_max = 0usize;
    for t in &g.tensors {
        *ty_cnt.entry(t.ty).or_default() += 1;
        shard_max = shard_max.max(t.shard);
    }
    let mut ty_s: Vec<String> = ty_cnt.iter().map(|(t, c)| format!("ty{t}:{c}")).collect();
    ty_s.sort();
    out.push(format!(
        "[fn-inv] GGUF main {} — arch={} shards={} tensors={} (split.tensors.count {}) types {}",
        main.display(),
        g.kv_str("general.architecture").unwrap_or("?"),
        g.shards.len(),
        g.tensors.len(),
        g.tensor_count_split,
        ty_s.join(" "),
    ));
    // 형상 실증(FnDims) + 샘플 텐서.
    let dims = FnDims::from_gguf(&g)?;
    let vocab = g
        .tensor("output.weight")
        .and_then(|t| t.dims.get(1))
        .copied()
        .unwrap_or(0);
    out.push(format!(
        "  dims: n_layer={} n_embd={} heads={}/{} hd={} n_rot={} vocab={} GDN(dt_rank={} d_state={} n_group={} conv_k={} conv_ch={}) MoE({}e top{} ffn{} shared{}) HC({}·{}) QSA-idx({}h·{}d·top{}) PLE(layers {:?} ngram{} hpng{} conv{} hd{})",
        dims.n_layer,
        dims.n_embd,
        dims.n_head,
        dims.n_kv,
        dims.head_dim,
        dims.n_rot,
        vocab,
        dims.dt_rank,
        dims.d_state,
        dims.n_group,
        dims.conv_k,
        dims.gdn_conv_ch(),
        dims.n_expert,
        dims.n_expert_used,
        dims.n_ff_exp,
        dims.n_ff_shared,
        dims.hc,
        dims.hc_low_rank,
        dims.idx_heads,
        dims.idx_dim,
        dims.idx_top_k,
        dims.ple_layers,
        dims.ple_ngram,
        dims.ple_heads_per_ngram,
        dims.ple_conv_k,
        dims.ple_head_dim,
    ));
    let mut shown = 0;
    for t in &g.tensors {
        if shown < 3
            && (t.name == "output.weight"
                || t.name.contains("blk.0.attn_qkv")
                || t.name.contains("ffn_gate_exps"))
        {
            out.push(format!(
                "  ex: {} ty{} [{:?}] shard{}",
                t.name,
                t.ty,
                t.dims,
                t.shard + 1
            ));
            shown += 1;
        }
    }
    // 샤드2 F32 노름 직독(1-D 텐서 — 행=텐서 전체 40960B) + 마지막 샤드
    // F32 직독(다중 샤드 오프셋 증명).
    let probe2 = g.read_rows("output_hc_norm.weight", 0, 1)?;
    let f32row = |b: &[u8]| -> f32 { f32::from_le_bytes([b[0], b[1], b[2], b[3]]) };
    let last = g
        .tensors
        .iter()
        .find(|t| t.shard == shard_max && t.ty == 0 && t.dims.len() == 1)
        .cloned();
    let mut last_line = String::new();
    if let Some(t) = last {
        let b = g.read_rows(&t.name, 0, 1)?;
        if b.len() >= 4 {
            last_line = format!(
                " · last-shard({}) F32 {}[{}] row0={} 직독 OK",
                shard_max + 1,
                t.name,
                t.dims[0],
                f32row(&b)
            );
        }
    }
    out.push(format!(
        "  offset-read: output_hc_norm.weight F32 shard2 row0 first={} ({}B){}",
        f32row(&probe2),
        probe2.len(),
        last_line,
    ));
    // MTP 모듈 — load_mtp 계약(mod.rs L347) 인자 검증.
    let m = FnGguf::open(mtp)?;
    out.push(format!(
        "[fn-inv] GGUF mtp {} — tensors={} arch={} nextn_predict_layers={:?}",
        mtp.display(),
        m.tensors.len(),
        m.kv_str("general.architecture").unwrap_or("?"),
        m.kv_u64("qwen4exp.nextn_predict_layers"),
    ));
    // mmproj(비전 — clip.rs 소비, FNA 범위 밖 인벤토리만).
    let v = FnGguf::open(mmproj)?;
    out.push(format!(
        "[fn-inv] GGUF mmproj {} — tensors={} arch={}",
        mmproj.display(),
        v.tensors.len(),
        v.kv_str("general.architecture").unwrap_or("?"),
    ));
    Ok(out)
}

/// fn-inv — 양 모델 세트 픽스처 인벤토리(로더 실증). 모델 파일 부재는
/// 비영 exit(계약: 실 파일 대상 증명이 목적).
pub fn cuda_fn_inventory_check() -> Result<String, String> {
    let mut lines = Vec::new();
    lines.extend(fn_inv_exl3(Path::new(FN_EXL3_DIR))?);
    lines.extend(fn_inv_gguf(
        Path::new(FN_GGUF_MAIN),
        Path::new(FN_GGUF_MTP),
        Path::new(FN_GGUF_MMPROJ),
    )?);
    Ok(lines.join("\n"))
}
