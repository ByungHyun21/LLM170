use super::cpu::greedy_from;
use super::weight::Weight;
pub trait RawDecode: Send + Sync {
    /// 상태 초기화 (가중치·상수 업로드 1회) — wnames는 필요 텐서명.
    fn raw_init(
        &self,
        hp: &crate::qwen35::hparams::Hparams,
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
    /// 선택적 per-seq 복원 — 부분수용 시 해당 seq만 되돌린다(plans/80 §C).
    fn gdn_restore_seq(&self, _seq: usize, _n_seqs: usize) -> Result<(), String> {
        Err("gdn_restore_seq: 미지원".into())
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
        let mut last = None;
        for ch in emb.chunks(512) {
            last = Some(self.raw_step(seq, pos0, ch)?);
        }
        Ok(last.unwrap_or_default())
    }
}
