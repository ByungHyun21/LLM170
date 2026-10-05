//! EXL3 직접 엔진 어댑터 (plans/121 A1) — TrellisResident + 슬롯별 SeqState.
//!
//! 배치 프리필(prefill_batch — coopmat GEMM 경로, pp512 60 t/s)과 순차
//! 디코드(decode_step, tg 4.69 t/s)를 Engine 열거형에 맞춘다. 상태는
//! 슬롯별 SeqState — 잡 완료 후 유지되므로 slot_loop의 완전 접두 재사용
//! (cached == 새 프롬프트 접두)가 자연 동작한다(부분 체크포인트 복원은
//! Q4 전용 경로라 자동 스킵).
//!
//! 메모리: 슬롯당 GDN 상태 ~200MB + KV 16층×2×ctx×1KB×4B (ctx 4096 ≈
//! 0.75GB/슬롯). 모델 상주 13.6GB와 합산해 슬롯 수·ctx는 가용 RAM 안에서.

use llm170_backend_gpu::rawvk::checks::{
    SeqState, TrellisResident, decode_step, new_seq_state, prefill_batch,
};

pub struct Exl3Engine {
    tr: TrellisResident,
    seqs: Vec<SeqState>,
    /// 정지 토큰(plans/130 F5) — tokenizer_config.json eos_token_id 파생,
    /// 실패 시 qwen 계열 기본 248044.
    pub eos: u32,
}

/// EXL3 모델 디렉터리의 정지 토큰 해석(plans/130 F5 — 공용, hip 어댑터도 사용).
/// tokenizer_config.json의 eos_token_id(단일 u64 또는 배열 선두)를 읽고
/// 실패 시 qwen 계열 기본 248044로 폴백(조용한 오염 금지 — 원장 89 계약상
/// 파싱 실패는 기본값 유지가 아니라 여기서는 허용: 토큰화기 없는 부분적재
/// 검증 디렉터리도 있어 경고 로그로만 남긴다).
pub(crate) fn exl3_eos_of(dir: &str) -> u32 {
    const DEFAULT_EOS: u32 = 248044;
    let Ok(txt) = std::fs::read_to_string(format!("{dir}/tokenizer_config.json")) else {
        return DEFAULT_EOS;
    };
    let Ok(j) = llm170_exl3::Json::parse(&txt) else {
        eprintln!("# [f5] tokenizer_config.json 파싱 실패 — EOS 기본 {DEFAULT_EOS}");
        return DEFAULT_EOS;
    };
    match j.get("eos_token_id") {
        Some(v) => {
            if let Some(n) = v.as_f64() {
                return n as u32;
            }
            if let Some(first) = v.as_num_array().and_then(|a| a.first().copied()) {
                return first as u32;
            }
            eprintln!("# [f5] eos_token_id 형태 미인식 — EOS 기본 {DEFAULT_EOS}");
            DEFAULT_EOS
        }
        None => {
            eprintln!("# [f5] eos_token_id 부재 — EOS 기본 {DEFAULT_EOS}");
            DEFAULT_EOS
        }
    }
}

// SAFETY: TrellisResident의 매핑 포인터(*mut u8)는 VkCtx 단일 소유로,
// 모든 GPU 접근은 slot_loop 스레드(이 구조체가 이동하는 유일 스레드)에서
// 직렬 실행된다 — decode_step/prefill_batch 내부 ar_pool 스레드는 VkCtx를
// 건드리지 않는다. 원장 120의 usize 래퍼 관례와 동일 근거.
unsafe impl Send for Exl3Engine {}

impl Exl3Engine {
    pub fn load(dir: &str, n_slots: usize, ctx_len: usize) -> Result<Self, String> {
        let eos = exl3_eos_of(dir);
        let tr = TrellisResident::load(dir)?;
        let n_layers = tr.n_layers;
        let seqs = (0..n_slots)
            .map(|_| new_seq_state(n_layers, ctx_len))
            .collect();
        Ok(Self { tr, seqs, eos })
    }

    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        prefill_batch(&mut self.tr, &mut self.seqs[seq], tokens)
    }

    pub fn decode1(&mut self, seq: usize, tok: u32) -> Result<Vec<f32>, String> {
        decode_step(&mut self.tr, &mut self.seqs[seq], tok)
    }

    pub fn reset_seq(&mut self, seq: usize) {
        // 제자리 클리어 — new_seq_state 재할당은 스택당 ~700MB memset을
        // 요청 전환마다 유발한다(plans/123 III-2). 상태/링/KV len만 0으로.
        let s = &mut self.seqs[seq];
        for g in s.gdn.iter_mut() {
            g.states.fill(0.0);
            g.conv.fill(0.0);
        }
        for k in s.kv.iter_mut() {
            k.len = 0;
        }
        for k in s.mtp_kv.iter_mut() {
            k.len = 0;
        }
        s.last_h.clear();
        s.last_logits.clear();
        s.last_tok = 0;
        s.pos = 0;
    }

    pub fn reset_states(&mut self) {
        for i in 0..self.seqs.len() {
            self.reset_seq(i);
        }
    }
}

#[cfg(test)]
mod f5_tests {
    use super::exl3_eos_of;
    use std::io::Write;

    fn tmpdir_with(cfg: &str) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!("f5t-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let mut f = std::fs::File::create(d.join("tokenizer_config.json")).unwrap();
        f.write_all(cfg.as_bytes()).unwrap();
        d.to_string_lossy().into_owned()
    }

    #[test]
    fn eos_single_id() {
        let d = tmpdir_with(r#"{"eos_token_id": 7}"#);
        assert_eq!(exl3_eos_of(&d), 7);
    }

    #[test]
    fn eos_id_array_takes_first() {
        let d = tmpdir_with(r#"{"eos_token_id": [11, 12]}"#);
        assert_eq!(exl3_eos_of(&d), 11);
    }

    #[test]
    fn eos_missing_falls_back_to_default() {
        let d = tmpdir_with(r#"{"model_type": "x"}"#);
        assert_eq!(exl3_eos_of(&d), 248044);
    }

    #[test]
    fn eos_no_dir_falls_back_to_default() {
        assert_eq!(exl3_eos_of("/nonexistent-dir-xyz"), 248044);
    }
}
