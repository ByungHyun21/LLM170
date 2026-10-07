//! Flash-Next PLE(n-gram 해시 임베딩) 모듈층 스켈레톤 — plans/124
//! G001(FNA)→FNG, 2026-10-05.
//!
//! [계약 요약 — 전체 판정표는 fn_support.rs 머리 계약 지도]
//! PLE는 blk.2 단일층(ple.layers=[2] 실측): n-gram 해시(호스트 u64) →
//! 16행×160 게더 → key/value 투영 → sgn√|s| 게이트 → 4스트림 방송 →
//! dilated depthwise conv(kern=4·dil=3·hist=9) → 잔차 2경로.
//! - HOST: n-gram 해시 — 원천: stages/ple.rs ple_hash_rows L276-335
//!   (mixed = ctx[0]·mult[0] ^ … 이후 mixed % vs[h] + offs[h], EOS 절단
//!   cut 전파·청크 lookback hist0 스냅샷 L296) + ple_hash L337-360
//!   (seq.ple_hist/ple_next_pos 진화). GPU u64 나머지 경제성 없음 +
//!   값 경로와 동일 계약 유지.
//! - NEW: 테이블 게더 스테이징 — EXL3 ngram_embedding.safetensors
//!   trellis I16 [320001536,61](37GiB — 절대 전량 상주/적재 금지,
//!   pread 스테이징: AGENTS plans/86 §6). GGUF PLE 테이블 행 오프셋
//!   산식은 mod.rs ple_gather_parts L471-505(block_info 기반).
//!   메타(head_offsets/vocabs/multipliers I64[16])·head_bias F16[16,160]
//!   는 FnNgramHead(fn_support)가 이미 오프셋 판독 제공.
//! - NEW: 수학 — grouped rms key/query/conv(원천: ple.rs L119-135 —
//!   hc.rs grouped_rms 재사용) → per-stream s = Σ key·query/√n_embd →
//!   sigmoid(sgn·√max(|s|,1e-6))(L148-150) → value 방송×게이트 →
//!   grouped norm → dilated conv(L169-195 — GDN conv 링과 동일한
//!   "한 런치 체인" 계급, 상태 [hist][hc·n]) → silu → 잔차
//!   row += value·g + conv(L217-235).
//! - REUSE(조건부): ple_key/ple_value 투영 — gemv/gemm2(EXL3, krate
//!   이슈 상동)·q4 MMQ(GGUF).
//! - 진단 계약: LLM170_PLE_DUMP 스테이지 해시(L237-260)·LLM170_PLE_
//!   VERIFY 프리페치 대조(L96-113)는 core 계측 — rawcuda 프로브는
//!   값 maxdiff로 동등 증명(계기 자체 검증, 원장 17호).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지.

use crate::rawcuda::fn_support::FnCuda;

/// PLE n-gram 크기(ple.ngram_size 실측 3 — bigram+trigram 2그룹).
pub const FN_PLE_NGRAM: usize = 3;
/// PLE n-gram당 헤드 수(ple.heads_per_ngram 실측 8 — bigram+trigram 16헤드).
pub const FN_PLE_HEADS_PER_NGRAM: usize = 8;
/// PLE 게이트 진폭 하한 — 원천: stages/ple.rs L148(max(|dot|,1e-6)).
pub const FN_PLE_MAG_FLOOR: f32 = 1e-6;

impl FnCuda {
    /// PLE 블록 진입 — TODO(FNG): stages/ple.rs ple_block L25-235 미러.
    /// 순서 계약: 해시 rows(호스트 ple_hash) → 게더 [t][16·160] →
    /// key/value 배치 투영 → 게이트·방송·norm → dilated conv(상태
    /// hist 9열 — padded = hist(상태)+t열, tail 갱신 L189-193) →
    /// 잔차 2경로(res_hc에 제자리 가산).
    pub fn ple_block(
        &mut self,
        _il: usize,
        _res_hc: &mut [Vec<f32>],
        _rows: &[u32],
    ) -> Result<(), String> {
        Err("미구현: FNG(fn_ple_cuda) 목표".into())
    }

    /// n-gram 해시(호스트) — TODO(FNG): stages/ple.rs ple_hash_rows
    /// L276-335 미러(순수 함수 — 프리페치 워커 공용 계약 plans/109 P7).
    /// 시퀀스 상태 진화(ple_hist/ple_next_pos)는 호출자 책임.
    pub fn ple_hash_rows(
        &self,
        _hist0: &[u32],
        _hist_valid: bool,
        _tokens: &[u32],
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        Err("미구현: FNG(fn_ple_cuda) 목표".into())
    }

    /// 테이블 게더 — TODO(FNG): EXL3 trellis 행 pread 스테이징(37GiB
    /// 오프로드 — 대형 업로드 mmap fault-in 금지, AGENTS/plans/86 §6)·
    /// GGUF ple_gather_parts(mod.rs L471) 오프셋 산식 미러.
    pub fn ple_gather(&mut self, _rows: &[u32], _out: &mut [f32]) -> Result<(), String> {
        Err("미구현: FNG(fn_ple_cuda) 목표".into())
    }
}
