//! frame/verify — MTP 스펙 배치 검증 포워드 (plans/110 W2).
//!
//! 단일 시퀀스의 제안 t행(≤7)을 **1회 포워드**로 검증한다. 구조는 np 배치
//! 디코드(frame/np.rs)의 단일 시퀀스 판: 공유 구간(hc_mix·투영·MoE·head)은
//! t행 배치, 상태 구간(PLE·GDN conv/AR·QSA 선택/KV)은 행별 t=1 순차 —
//! np가 유지하는 "배치 == 순차 decode1 비트 동일" 불변식을 그대로 계승한다.
//! 반환은 행별 GPU argmax(로짓 전사 없음).
//!
//! 롤백 계약: 이 포워드는 검증 **시도**다 — 호출부(mtp_spec_step)가 기각 시
//! GDN 디바이스 상태(st_gdn/st_conv) 스냅샷을 되돌리고 수용분만 재실행한다.
//! QSA KV/idx 풀은 pos 키 쓰기라 재실행에 멱등, PLE 링은 pos 기반
//! 워터마크 되감기(백엔드)로 호스트 링에서 리프레시된다.
use super::super::stages::{self, Ctx};
use super::forward::{hc_combine_frame, hc_mix_frame, moe_frame};
use super::np::{ensure_np_views, gdn_frame_np, qsa_frame_np};
use super::{Frame4, fs_begin, op};
use crate::matmul::{Accelerator, FrameOp};
use crate::qwen4exp::layers::SeqState4;
use crate::qwen4exp::{Hparams4, Model4, Q4Error};

/// 단일 시퀀스 t행 검증 포워드 — seq_sts[seq].pos부터 연속 t행.
/// 반환 y[i] = tokens[i] 처리 후 greedy 토큰(행별). pos 진행은 호출부가.
/// 부수효과: res_hc 행 export(mtp_h_export 시 f.last_res_hc_rows), QSA
/// 호스트 캐시 stale, PLE 링·GDN 상태 디바이스 전진.
/// 검증 배치 핀 가드 — 드롭 시 해제(조기 반환 포함).
struct RowPin<'a>(&'a dyn Accelerator);
impl<'a> RowPin<'a> {
    fn new(acc: &'a dyn Accelerator) -> Self {
        acc.frame_verify_rows(true);
        RowPin(acc)
    }
}
impl Drop for RowPin<'_> {
    fn drop(&mut self) {
        self.0.frame_verify_rows(false);
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn frame_forward_verify(
    acc: &dyn Accelerator,
    model: &Model4,
    ctx: &Ctx,
    seq_sts: &mut [SeqState4],
    seq: usize,
    f: &mut Frame4,
    tokens: &[u32],
) -> Result<Vec<u32>, Q4Error> {
    let hp: &Hparams4 = &model.hp;
    let (n, hc) = (hp.n_embd, hp.hc);
    let k_len = hp.n_group * hp.d_state;
    let v_len = hp.dt_rank * hp.d_state;
    let conv_ch = 2 * k_len + v_len;
    let eps = hp.eps;
    let t = tokens.len();
    if t == 0 || t > 7 {
        return Err(Q4Error::Io(format!("verify: t={t} (1..=7 필요)")));
    }
    // t=2..8 GEMV를 행별 t=1 커널로 — 순차 decode1 비트 동일(110 W2).
    let _row_pin = RowPin::new(acc);
    // 같은 seq 반복 = 행 체이닝(qsa_frame_np 발생수 pos·gdn_frame_np per-row
    // 폴백이 처리 — 1런치 경로는 distinct 조건으로 자동 배제).
    let seqs: Vec<usize> = vec![seq; t];
    fs_begin(acc, t);
    ensure_np_views(acc, f, t, conv_ch, k_len, v_len, n, hc, hp)?;

    // 0) 임베딩 — t행 → res_hc
    {
        let embd = model
            .w("token_embd.weight")
            .ok_or(Q4Error::MissingTensor("token_embd".into()))?;
        super::emb_broadcast_write(acc, &embd, tokens, f.res_hc, n, hc)?;
    }

    // PLE n-gram 행 — 행별 호스트 해시(같은 seq라 체인)
    let ple_rows: Vec<Vec<u32>> = if hp.is_ple(1) {
        // 110 W3 근원 수리: ple_hash는 pos 핸드셰이크(hist_valid =
        // next_pos == pos)로 체인한다 — 같은 seq의 t행을 한 번에 해시할 때는
        // 행마다 pos를 진행해야 순차 decode1과 동일한 행열이 나온다.
        // 종전엔 전 행이 같은 pos로 해시돼 행1의 hist_valid가 깨졌다(발산 근원).
        // 해시 후 pos 복원 — QSA/GDN 행별 처리는 base+occ로 자체 산출.
        let base = seq_sts[seq].pos;
        let mut v = Vec::with_capacity(t);
        for row in 0..t {
            seq_sts[seq].pos = base + row as u32;
            v.push(stages::ple_hash(
                ctx,
                &mut seq_sts[seq],
                &tokens[row..row + 1],
            ));
        }
        seq_sts[seq].pos = base;
        v
    } else {
        Vec::new()
    };

    let mut recr_idx = 0usize;
    let mut full_idx = 0usize;
    for il in 0..hp.n_layer {
        // 1) PLE — 배치 gather·투영 + 행별 math(링 체인) (plans/115 P0-4)
        // np.rs와 동일 구조: verify 행은 같은 seq의 드래프트 창이라 math의
        // 디바이스 링 체인(pos+row 순행)만 행별 t=1을 유지한다.
        if hp.is_ple(il) {
            let heads = hp.ple_heads_per_ngram * 2;
            let emb_w = heads * hp.ple_head_dim;
            let w_key = model.w4(&format!("blk.{il}.ple_key.weight"))?;
            let w_value = model.w4(&format!("blk.{il}.ple_value.weight"))?;
            let nk = model.f32_vec4(&format!("blk.{il}.ple_norm_key.weight"))?;
            let nq = model.f32_vec4(&format!("blk.{il}.ple_norm_query.weight"))?;
            let nc = model.f32_vec4(&format!("blk.{il}.ple_norm_conv.weight"))?;
            let cw = model.f32_vec4(&format!("blk.{il}.ple_conv1d.weight"))?;
            let vv = f.np_views.as_ref().unwrap();
            fs_begin(acc, t);
            let all_valid = ple_rows.iter().all(|r| r.len() == heads);
            if all_valid {
                let flat: Vec<u32> = ple_rows.concat();
                let gpu = (|| -> Result<(), Q4Error> {
                    let (tptr, tlen, _tty, thd) = ctx.model.ple_table_view()?;
                    // SAFETY (107 W8): ple_table_view 계약 — mmap 유효 범위(읽기 전용).
                    let tdata: &[u8] =
                        unsafe { std::slice::from_raw_parts(tptr as *const u8, tlen) };
                    acc.ple_gather_dev(tptr, tdata, &flat, f.ple_emb, thd)
                        .map_err(Q4Error::Io)
                })()
                .is_ok();
                if !gpu {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    crate::qwen4exp::frame::fb_incr(crate::qwen4exp::frame::FbId::PleGgpu);
                    ONCE.call_once(|| eprintln!("# verify-ple-ggpu: 실패 — 호스트 폴백"));
                    let mut emb = vec![0.0f32; emb_w * t];
                    for (row, r) in ple_rows.iter().enumerate() {
                        ctx.model
                            .ple_gather(r, &mut emb[row * emb_w..(row + 1) * emb_w])?;
                    }
                    acc.frame_write(f.ple_emb, &emb).map_err(Q4Error::Io)?;
                }
            } else {
                let mut emb = vec![0.0f32; emb_w * t];
                for (row, r) in ple_rows.iter().enumerate() {
                    if r.len() == heads {
                        ctx.model
                            .ple_gather(r, &mut emb[row * emb_w..(row + 1) * emb_w])?;
                    }
                }
                acc.frame_write(f.ple_emb, &emb).map_err(Q4Error::Io)?;
            }
            acc.frame_mm_group(f.ple_emb, &[w_key, w_value], &[f.ple_key, f.ple_value], t)
                .map_err(Q4Error::Io)?;
            for row in 0..t {
                fs_begin(acc, 1);
                acc.ple_math_dev(
                    vv.res_hc[row],
                    vv.ple_key[row],
                    vv.ple_value[row],
                    &nk,
                    &nq,
                    &nc,
                    &cw,
                    f.ple_gated,
                    f.ple_conv_out,
                    f.ple_gate,
                    seq,
                    seq_sts[seq].pos as usize + row,
                    1,
                    hp.eps,
                    n,
                    hc,
                    hp.ple_conv_k,
                    hp.ple_ngram,
                    (hp.ple_conv_k - 1) * hp.ple_ngram,
                    &seq_sts[seq].ple_conv,
                )
                .map_err(Q4Error::Io)?;
            }
            // W3 진단: PLE 링 합 — 다음 라운드 PLE 입력(발산 매개 후보).
            if llm170_diag::dump::opts().checksum {
                acc.frame_sync(); // 링 쓰기는 비동기 — ple_ring_sync 대기 후 계약
                let hist = (hp.ple_conv_k - 1) * hp.ple_ngram;
                let mut ring = vec![0.0f32; hist * hc * n];
                let rs = match acc.ple_ring_sync(seq, &mut ring) {
                    Ok(()) => format!(
                        "sum={:.6} v0={:.6} last={:.6}",
                        ring.iter().map(|&x| x as f64).sum::<f64>(),
                        ring[0],
                        ring[ring.len() - 1]
                    ),
                    Err(e) => format!("none({e})"),
                };
                eprintln!("[npck] V{il}.ring t={t} {rs}");
            }
        }

        fs_begin(acc, t); // 공유 구간
        // 2) hc attn mix
        hc_mix_frame(acc, model, f, il, "attn", eps, n, hc, t)?;

        // 3) GDN / QSA — 같은 seq 행 체이닝(np 스테이지 fn이 처리)
        if hp.is_recr(il) {
            gdn_frame_np(
                acc, model, f, il, &seqs, recr_idx, conv_ch, k_len, v_len, eps, t,
            )?;
            hc_combine_frame(acc, f, f.ffn_out, f.inj, n, hc, t)?;
            recr_idx += 1;
        } else {
            qsa_frame_np(acc, model, seq_sts, &seqs, f, il, t, full_idx)?;
            hc_combine_frame(acc, f, f.ffn_out, f.inj, n, hc, t)?;
            full_idx += 1;
        }

        // 4) hc ffn mix + MoE(t 배치) + combine — 110 W3: ple_hash pos
        // 핸드셰이크 수리 후 t=k-1 배치가 비스펙와 완전 등가(43/43)임을
        // 확인 — MoE t>1(gather/scatter) 포함 배치 경로 그대로.
        hc_mix_frame(acc, model, f, il, "ffn", eps, n, hc, t)?;
        moe_frame(acc, model, f, il, n, t)?;
        hc_combine_frame(acc, f, f.mout, f.inj, n, hc, t)?;
        // W3 진단: 스테이지 체크섬 — t≥2 공유구간과 t=1의 첫 발산 지점 특정.
        // il<4는 전 스테이지, 나머지 층은 잔차(res)만 8층 간격 표본(판독
        // 비용 상한). 태그 접두 V(verify)로 npck 스트림에서 분리.
        let ck_all4 = il < 4 && llm170_diag::dump::opts().checksum;
        let ck_res = il % 8 == 7 && llm170_diag::dump::opts().checksum;
        if ck_all4 {
            super::diag::frame_ck(acc, f.mix, n, t, &format!("V{il}.mix"));
            super::diag::frame_ck(acc, f.ffn_out, n, t, &format!("V{il}.attn"));
            super::diag::frame_ck(acc, f.mix, n, t, &format!("V{il}.mixf"));
            super::diag::frame_ck(acc, f.mout, n, t, &format!("V{il}.moe"));
        }
        if ck_all4 || ck_res || il + 1 == hp.n_layer {
            super::diag::frame_ck(acc, f.res_hc, hc * n, t, &format!("V{il}.res"));
        }
    }

    // 5) pre-mixer res_hc 행 export — 다음 라운드 드래프트 h 입력.
    if f.mtp_h_export {
        let mut rows = vec![0.0f32; t * hc * n];
        acc.frame_read(f.res_hc, &mut rows).map_err(Q4Error::Io)?;
        f.last_res_hc_rows = rows.chunks(hc * n).map(|c| c.to_vec()).collect();
    }

    // 6) head — 전 행 GEMM 1회 → [t][vocab] argmax(np 헤드와 동일 구조)
    let w_norm = f.consts["output_hc_norm"];
    op(
        acc,
        FrameOp::RmsRows {
            x: f.res_hc,
            w: w_norm,
            out: f.hxn,
            eps,
            n,
            w_reps: hc,
        },
    )?;
    let w_down = model.w4("output_hc_down.weight")?;
    acc.frame_mm(f.hxn, &w_down, f.hlo, t)
        .map_err(Q4Error::Io)?;
    op(
        acc,
        FrameOp::SiluDiv {
            t: f.hlo,
            div: hc as f32,
            n: f.hlo_len * t,
        },
    )?;
    let w_up = model.w4("output_hc_up.weight")?;
    acc.frame_mm(f.hlo, &w_up, f.hgate, t)
        .map_err(Q4Error::Io)?;
    op(
        acc,
        FrameOp::HcGateMean {
            xn: f.hxn,
            gate: f.hgate,
            out: f.hin,
            hc,
            n,
        },
    )?;
    let wout = model
        .w("output.weight")
        .ok_or(Q4Error::MissingTensor("output.weight".into()))?;
    acc.frame_mm(f.hin, &wout, f.logits_t, t)
        .map_err(Q4Error::Io)?;
    let toks = acc
        .frame_argmax_rows(f.logits_t, t, hp.vocab)
        .map_err(Q4Error::Io)?;
    Ok(toks)
}

/// GDN 디바이스 상태(st_gdn/st_conv) 원시 스냅샷 — 검증 배치 전 캡처,
/// 기각 시 복원(전치 없는 디바이스 레이아웃 그대로). 스냅샷 버퍼는 Frame4에
/// 상주 재사용(FN 기준 ~112MB/seq — 매 라운드 할당·0채움 회피).
pub(crate) fn verify_snap_capture(
    acc: &dyn Accelerator,
    f: &mut Frame4,
    seq: usize,
) -> Result<(), Q4Error> {
    if f.verify_snap_gdn.len() != f.st_gdn[seq].len() {
        f.verify_snap_gdn = vec![Vec::new(); f.st_gdn[seq].len()];
        f.verify_snap_conv = vec![Vec::new(); f.st_conv[seq].len()];
    }
    for (ri, &h) in f.st_gdn[seq].iter().enumerate() {
        if f.verify_snap_gdn[ri].len() != f.gdn_state_len {
            f.verify_snap_gdn[ri] = vec![0.0f32; f.gdn_state_len];
        }
        acc.frame_read(h, &mut f.verify_snap_gdn[ri])
            .map_err(Q4Error::Io)?;
    }
    for (ri, &h) in f.st_conv[seq].iter().enumerate() {
        if f.verify_snap_conv[ri].len() != f.conv_state_len {
            f.verify_snap_conv[ri] = vec![0.0f32; f.conv_state_len];
        }
        acc.frame_read(h, &mut f.verify_snap_conv[ri])
            .map_err(Q4Error::Io)?;
    }
    Ok(())
}

pub(crate) fn verify_snap_restore(
    acc: &dyn Accelerator,
    f: &Frame4,
    seq: usize,
) -> Result<(), Q4Error> {
    for (ri, &h) in f.st_gdn[seq].iter().enumerate() {
        acc.frame_write(h, &f.verify_snap_gdn[ri])
            .map_err(Q4Error::Io)?;
    }
    for (ri, &h) in f.st_conv[seq].iter().enumerate() {
        acc.frame_write(h, &f.verify_snap_conv[ri])
            .map_err(Q4Error::Io)?;
    }
    Ok(())
}
