//! qwen35 디코드 프레임 (t=1, 단일 시퀀스) — GDN층·FFN·head 전부 기기 상주,
//! 어텐션층만 값 브리지(판독→CPU attn_layer→기록, 층당 1 sync).
//! 게이트: LLM170_FRAME35=1 (기본 off — 검증 통과 후 전환).
//! 수치 계약: 값 경로(model/layers.rs)와 동일 순서 — 잔차는 AxpyScaled(s=1)로
//! xs += out·1.0 (x·1.0 ≡ x, f32 비트 불변).

use super::{Engine, ModelError};
use crate::matmul::{Accelerator, FrameOp, FrameState};
use crate::quant::dequant_row;
use std::collections::HashMap;

/// 프레임 상주 버퍼 세트 — 스텝마다 재사용, 해제 없음.
pub struct Frame35 {
    xs: u64, // 잔차 스트림 [n_embd]
    xn: u64, // norm 출력 [n_embd]
    // GDN 스테이지
    gqkv: u64,
    gz: u64,
    gb: u64,
    ga: u64,
    gbg: u64,
    gconv: u64,
    gq: u64,
    gk: u64,
    gv: u64,
    go: u64,
    ggated: u64,
    gout: u64,
    // FFN
    fgate: u64,
    fup: u64,
    fglu: u64,
    fdown: u64,
    logits: u64,
    one: u64, // AxpyScaled용 1.0 스케일 버퍼
    // 어텐션 프레임 (P1) — q/k 프리페어·인터리브·KV 캐시·마스크
    aq: u64,       // q+gate [n_head·2·hd]
    ak: u64,       // k [n_kv·hd]
    av: u64,       // v [n_kv·hd]
    aqrow: u64,    // q‖gate 인터리브 [n_head·2·hd]
    akstg: u64,    // k 행 스테이징 (CPU rms·rope 결과 → 캐시 append)
    aout: u64,     // 어텐션 출력 [n_head·hd]
    cs: u64,       // rope cos/sin [ctx][half][2]
    mask: u64,     // u32 [ctx] 전체 가시
    kv_k: Vec<Vec<u64>>, // [seq][full] k 캐시
    kv_v: Vec<Vec<u64>>, // [seq][full] v 캐시
    st_conv: Vec<Vec<u64>>, // [seq][recr]
    st_gdn: Vec<Vec<u64>>,  // [seq][recr]
    consts: HashMap<String, u64>,
}

impl Frame35 {
    pub fn new(acc: &dyn Accelerator, eng: &Engine) -> Result<Self, ModelError> {
        let hp = &eng.model.hp;
        let n = hp.n_embd;
        let k_len = hp.n_group * hp.d_state;
        let v_len = hp.dt_rank * hp.d_state;
        let conv_ch = hp.conv_ch();
        let d_inner = hp.d_inner;
        let n_ff = hp.n_ff;
        let n_recr = (0..hp.n_layer).filter(|&il| eng.model.is_recr(il)).count();
        let a = |len: usize| acc.frame_alloc(len).map_err(ModelError::Accel);
        // 실측 ctx — SeqState KV 버퍼 길이에서 유도 (tiny/실모델 공용).
        let ctx_frames = if eng.seqs.is_empty() || eng.seqs[0].kv_k_ref().is_empty() {
            2048
        } else {
            eng.seqs[0].kv_k_ref()[0].len() / (hp.n_kv * hp.head_dim)
        };
        let mut f = Frame35 {
            xs: a(n)?,
            xn: a(n)?,
            gqkv: a(conv_ch)?,
            gz: a(d_inner)?,
            gb: a(hp.dt_rank)?,
            ga: a(hp.dt_rank)?,
            gbg: a(hp.dt_rank * 2)?,
            gconv: a(conv_ch)?,
            gq: a(k_len)?,
            gk: a(k_len)?,
            gv: a(v_len)?,
            go: a(v_len)?,
            ggated: a(d_inner)?,
            gout: a(n)?,
            fgate: a(n_ff)?,
            fup: a(n_ff)?,
            fglu: a(n_ff)?,
            fdown: a(n)?,
            logits: a(eng.model.wchk("output.weight")?.n_out as usize)?,
            one: a(1)?,
            aq: a(hp.n_head * 2 * hp.head_dim)?,
            ak: a(hp.n_kv * hp.head_dim)?,
            av: a(hp.n_kv * hp.head_dim)?,
            aqrow: a(hp.n_head * 2 * hp.head_dim)?,
            akstg: a(hp.n_kv * hp.head_dim)?,
            aout: a(hp.n_head * hp.head_dim)?,
            cs: a(ctx_frames * (hp.n_rot / 2) * 2)?,
            mask: a(ctx_frames)?,
            st_conv: Vec::with_capacity(eng.seqs.len()),
            st_gdn: Vec::with_capacity(eng.seqs.len()),
            kv_k: Vec::with_capacity(eng.seqs.len()),
            kv_v: Vec::with_capacity(eng.seqs.len()),
            consts: HashMap::new(),
        };
        acc.frame_write(f.one, &[1.0f32]).map_err(ModelError::Accel)?;
        // 어텐션(P1): cs 테이블·마스크·KV 캐시 — CPU rope_head과 동일 값.
        {
            let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
            let half = n_rot / 2;
            let _ = n_head;
            let mut cs = vec![0.0f32; ctx_frames * half * 2];
            // ctx 상한 — hp에서 파생
            let ctx_n = ctx_frames;
            for pos in 0..ctx_n {
                for pp in 0..half {
                    let theta = hp.rope_base.powf(-(2.0 * pp as f32) / n_rot as f32);
                    let angle = pos as f32 * theta;
                    cs[pos * half * 2 + pp * 2] = angle.cos();
                    cs[pos * half * 2 + pp * 2 + 1] = angle.sin();
                }
            }
            acc.frame_write(f.cs, &cs).map_err(ModelError::Accel)?;
            let masks: Vec<u32> = vec![1u32; ctx_n];
            acc.frame_write_u32(f.mask, &masks).map_err(ModelError::Accel)?;
            let _ = (n_kv, hd);
        }
        for _ in 0..eng.seqs.len() {
            let mut gdn = Vec::with_capacity(n_recr);
            let mut conv = Vec::with_capacity(n_recr);
            for _ in 0..n_recr {
                gdn.push(a(hp.dt_rank * hp.d_state * hp.d_state)?);
                conv.push(a((hp.conv_k - 1) * conv_ch)?);
            }
            f.st_gdn.push(gdn);
            f.st_conv.push(conv);
            let n_full = (0..hp.n_layer).filter(|&il| !eng.model.is_recr(il)).count();
            let mut kk = Vec::with_capacity(n_full);
            let mut vv = Vec::with_capacity(n_full);
            for _ in 0..n_full {
                kk.push(a(ctx_frames * hp.n_kv * hp.head_dim)?);
                vv.push(a(ctx_frames * hp.n_kv * hp.head_dim)?);
            }
            f.kv_k.push(kk);
            f.kv_v.push(vv);
        }
        // 상수 가중치 — 층별 norm·GDN 계수 (qwen4exp Frame4 관례 동일).
        let mut put = |f: &mut Frame35, name: &str, v: &[f32]| -> Result<(), ModelError> {
            let h = acc.frame_alloc(v.len()).map_err(ModelError::Accel)?;
            acc.frame_write(h, v).map_err(ModelError::Accel)?;
            f.consts.insert(name.into(), h);
            Ok(())
        };
        for il in 0..hp.n_layer {
            put(&mut f, &format!("blk.{il}.attn_norm"), &eng.model.f32_vec(&format!("blk.{il}.attn_norm.weight"))?)?;
            put(&mut f, &format!("blk.{il}.post_norm"), &eng.model.f32_vec(&format!("blk.{il}.post_attention_norm.weight"))?)?;
            if !eng.model.is_recr(il) {
                put(&mut f, &format!("blk.{il}.attn_q_norm"), &eng.model.f32_vec(&format!("blk.{il}.attn_q_norm.weight"))?)?;
                put(&mut f, &format!("blk.{il}.attn_k_norm"), &eng.model.f32_vec(&format!("blk.{il}.attn_k_norm.weight"))?)?;
            }
            if eng.model.is_recr(il) {
                // ssm_norm [d_state] 전헤드 공유 — dt_rank 타일 업로드
                // (norm_gated_rows_silu가 헤드별 슬라이스 인덱싱, 2026-09-01 RCA).
                let sn = eng.model.f32_vec(&format!("blk.{il}.ssm_norm.weight"))?;
                let sn_tiled: Vec<f32> = sn.iter().copied().cycle().take(sn.len() * hp.dt_rank).collect();
                put(&mut f, &format!("blk.{il}.ssm_norm"), &sn_tiled)?;
                put(&mut f, &format!("blk.{il}.dt_bias"), &eng.model.f32_vec(&format!("blk.{il}.ssm_dt.bias"))?)?;
                put(&mut f, &format!("blk.{il}.ssm_a"), &eng.model.f32_vec(&format!("blk.{il}.ssm_a"))?)?;
                put(&mut f, &format!("blk.{il}.conv_w"), &eng.model.f32_vec(&format!("blk.{il}.ssm_conv1d.weight"))?)?;
            }
        }
        put(&mut f, "output_norm", &eng.model.f32_vec("output_norm.weight")?)?;
        Ok(f)
    }

    /// CPU SeqState의 GDN 상태를 GPU로 재동기 (prefill 직후) — 시퀀스 지정.
    pub fn sync_states(&mut self, acc: &dyn Accelerator, eng: &Engine, seq: usize) -> Result<(), ModelError> {
        for (ri, h) in self.st_gdn[seq].iter().enumerate() {
            acc.frame_write(*h, &eng.seqs[seq].gdn_s[ri]).map_err(ModelError::Accel)?;
        }
        for (ri, h) in self.st_conv[seq].iter().enumerate() {
            acc.frame_write(*h, &eng.seqs[seq].conv[ri]).map_err(ModelError::Accel)?;
        }
        // KV 캐시 — 프리필(값 경로)이 CPU 캐시에 기록한 분 동기 (2026-09-02
        // P1 RCA: 누락 시 첫 디코드부터 가비지 키 판독, 2토큰째 발산).
        // k 캐시는 kq_scale 사전 곱 상태로 상주 — append(krow·s)와 일관,
        // 값 경로 qsa_attention_inner의 업로드 곱과 동일 반올림.
        let kqs = eng.model.hp.kq_scale();
        for (fi, h) in self.kv_k[seq].iter().enumerate() {
            let scaled: Vec<f32> = eng.seqs[seq].kv_k_ref()[fi].iter().map(|v| v * kqs).collect();
            acc.frame_write(*h, &scaled).map_err(ModelError::Accel)?;
        }
        for (fi, h) in self.kv_v[seq].iter().enumerate() {
            acc.frame_write(*h, &eng.seqs[seq].kv_v_ref()[fi]).map_err(ModelError::Accel)?;
        }
        Ok(())
    }
}

fn op(acc: &dyn Accelerator, o: FrameOp) -> Result<(), ModelError> {
    acc.frame_op(&o).map_err(ModelError::Accel)
}

impl Engine {
    /// 프레임 디코드 1스텝 (t=1, seq 1개) — logits 반환.
    /// LLM170_FRAME35=1 게이트. 실패 시 Err (묵시 폴백 없음 — 명시적 옵트인).
    pub fn decode1_frame35(&mut self, seq: usize, token: u32) -> Result<Vec<f32>, ModelError> {
        let acc = self.acc.clone().ok_or(ModelError::Accel("frame35: 가속기 없음".into()))?;
        let hp = self.model.hp.clone();
        let n = hp.n_embd;
        let k_len = hp.n_group * hp.d_state;
        let v_len = hp.dt_rank * hp.d_state;
        let conv_ch = hp.conv_ch();
        let eps = hp.eps;

        if self.frame35.is_none() {
            let f0 = Frame35::new(acc.as_ref(), self)?;
            self.frame35 = Some(f0);
        }
        // take/put — 프레임 차입과 self 차입(가중치·attn 브리지) 분리.
        let mut f = self.frame35.take().expect("frame35");
        let r = Engine::frame35_step(&mut f, self, &acc, seq, token);
        self.frame35 = Some(f);
        r
    }

    /// 프레임 스텝 본체 — f와 eng 차입 분리 (take/put 패턴).
    fn frame35_step(
        f: &mut Frame35,
        eng: &mut Engine,
        acc: &std::sync::Arc<dyn Accelerator>,
        seq: usize,
        token: u32,
    ) -> Result<Vec<f32>, ModelError> {
        let hp = eng.model.hp.clone();
        let n = hp.n_embd;
        let k_len = hp.n_group * hp.d_state;
        let v_len = hp.dt_rank * hp.d_state;
        let conv_ch = hp.conv_ch();
        let eps = hp.eps;
        if !eng.frame35_clean[seq] {
            f.sync_states(acc.as_ref(), eng, seq)?;
            eng.frame35_clean[seq] = true;
        }

        // 0) 임베딩 — CPU dequant → 기록
        {
            let embd = eng.model.wchk("token_embd.weight")?;
            let mut row = vec![0.0f32; n];
            dequant_row(embd.ty, embd.data, token as u64, n as u64, &mut row);
            acc.frame_write(f.xs, &row).map_err(ModelError::Accel)?;
        }

        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        for il in 0..hp.n_layer {
            // 1) pre-norm
            let w_norm = f.consts[&format!("blk.{il}.attn_norm")];
            op(acc.as_ref(), FrameOp::RmsRows { x: f.xs, w: w_norm, out: f.xn, eps, n, w_reps: 1 })?;

            // 2) 층 본체 — GDN 프레임 / 어텐션 값 브리지
            if eng.model.is_recr(il) {
                let wqkv = eng.model.wchk(&format!("blk.{il}.attn_qkv.weight"))?;
                let wgate = eng.model.wchk(&format!("blk.{il}.attn_gate.weight"))?;
                let wb = eng.model.wchk(&format!("blk.{il}.ssm_beta.weight"))?;
                let wa = eng.model.wchk(&format!("blk.{il}.ssm_alpha.weight"))?;
                acc.frame_mm_group(f.xn, &[wqkv, wgate, wb, wa], &[f.gqkv, f.gz, f.gb, f.ga], 1)
                    .map_err(ModelError::Accel)?;
                // conv + ring
                let cw = f.consts[&format!("blk.{il}.conv_w")];
                op(acc.as_ref(), FrameOp::GdnConv { qkv: f.gqkv, cw, state: f.st_conv[seq][recr_idx], out: f.gconv, ch: conv_ch, k: hp.conv_k, t_len: 1 })?;
                // q/k/v 분할 + l2 + q·scale
                op(acc.as_ref(), FrameOp::CopyRows { src: f.gconv, dst: f.gq, src_off: 0, dst_off: 0, n: k_len })?;
                op(acc.as_ref(), FrameOp::CopyRows { src: f.gconv, dst: f.gk, src_off: k_len, dst_off: 0, n: k_len })?;
                op(acc.as_ref(), FrameOp::CopyRows { src: f.gconv, dst: f.gv, src_off: 2 * k_len, dst_off: 0, n: v_len })?;
                op(acc.as_ref(), FrameOp::L2Rows { x: f.gq, eps, d: hp.d_state })?;
                op(acc.as_ref(), FrameOp::L2Rows { x: f.gk, eps, d: hp.d_state })?;
                let scale = 1.0f32 / (hp.d_state as f32).sqrt();
                op(acc.as_ref(), FrameOp::Scale { t: f.gq, s: scale, n: k_len })?;
                // β/e^g
                let dtb = f.consts[&format!("blk.{il}.dt_bias")];
                let ssa = f.consts[&format!("blk.{il}.ssm_a")];
                op(acc.as_ref(), FrameOp::GdnBetaG { b: f.gb, a: f.ga, dtb, sa: ssa, bg: f.gbg, n_h: hp.dt_rank })?;
                // AR 갱신 — 상태 GPU 상주
                let fs: &dyn FrameState = acc.as_ref();
                fs.frame_gdn_ar(f.gq, f.gk, f.gv, f.gbg, f.st_gdn[seq][recr_idx], f.go, 1, hp.n_group, hp.dt_rank, hp.d_state)
                    .map_err(ModelError::Accel)?;
                // norm_gated(silu) + out proj
                let snorm = f.consts[&format!("blk.{il}.ssm_norm")];
                op(acc.as_ref(), FrameOp::NormGatedSilu { o: f.go, z: f.gz, w: snorm, out: f.ggated, eps, d: hp.d_state, n_h: hp.dt_rank })?;
                let wout = eng.model.wchk(&format!("blk.{il}.ssm_out.weight"))?;
                acc.frame_mm(f.ggated, &wout, f.gout, 1).map_err(ModelError::Accel)?;
                recr_idx += 1;
            } else {
                // 어텐션 프레임 (P1): q/k/v mm → 프리페어(rms·rope·인터리브·
                // 캐시 append) → qsa_attention(전가시 마스크) → wo.
                let pos = eng.seqs[seq].pos as usize;
                let (n_head, n_kv, hd) = (hp.n_head, hp.n_kv, hp.head_dim);
                if pos >= 2048 {
                    return Err(ModelError::Accel("frame35: ctx 2048 초과 (cs/캐시 상한)".into()));
                }
                let pos = eng.seqs[seq].pos as usize;
                let (n_head, n_kv, hd) = (hp.n_head, hp.n_kv, hp.head_dim);
                if pos >= 2048 {
                    return Err(ModelError::Accel("frame35: ctx 2048 초과".into()));
                }
                let pos = eng.seqs[seq].pos as usize;
                let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
                let wq = eng.model.wchk(&format!("blk.{il}.attn_q.weight"))?;
                let wk = eng.model.wchk(&format!("blk.{il}.attn_k.weight"))?;
                let wv = eng.model.wchk(&format!("blk.{il}.attn_v.weight"))?;
                acc.frame_mm_group(f.xn, &[wq, wk, wv], &[f.aq, f.ak, f.av], 1)
                    .map_err(ModelError::Accel)?;
                // rms·rope는 CPU — GPU 코드젠의 FMA 수축이 CPU strict FP와
                // 1-ulp 어긋나 토큰을 갈라놓음 (2026-09-02 P1 RCA).
                // q/k/v mm·attention·wo만 GPU: 층당 왕복 4회.
                let mut aqv = vec![0.0f32; n_head * 2 * hd];
                acc.frame_read(f.aq, &mut aqv).map_err(ModelError::Accel)?;
                let mut akv = vec![0.0f32; n_kv * hd];
                acc.frame_read(f.ak, &mut akv).map_err(ModelError::Accel)?;
                let qn = eng.model.f32_vec(&format!("blk.{il}.attn_q_norm.weight"))?;
                let kn = eng.model.f32_vec(&format!("blk.{il}.attn_k_norm.weight"))?;
                let mut qrow = vec![0.0f32; n_head * 2 * hd];
                for h in 0..n_head {
                    let src = aqv[h * 2 * hd..h * 2 * hd + hd].to_vec();
                    let mut qh = crate::ops::rms_norm(&src, &qn, eps);
                    crate::ops::rope_head(&mut qh, pos as u32, n_rot, hp.rope_base);
                    qrow[h * 2 * hd..h * 2 * hd + hd].copy_from_slice(&qh);
                    for i in 0..hd {
                        qrow[h * 2 * hd + hd + i] = aqv[h * 2 * hd + hd + i];
                    }
                }
                let mut krow = vec![0.0f32; n_kv * hd];
                for h in 0..n_kv {
                    let src = akv[h * hd..(h + 1) * hd].to_vec();
                    let mut kh = crate::ops::rms_norm(&src, &kn, eps);
                    crate::ops::rope_head(&mut kh, pos as u32, n_rot, hp.rope_base);
                    krow[h * hd..(h + 1) * hd].copy_from_slice(&kh);
                }
                // kq_scale 사전 곱 — 값 경로 qsa_attention_inner가 ck에
                // 곱하는 것과 동일 연산(비트 동일). frame_qsa_attention은
                // 스케일 인수를 무시(qwen4exp는 1.0).
                let kqs = eng.model.hp.kq_scale();
                let mut krow_scaled = krow;
                for v in krow_scaled.iter_mut() {
                    *v *= kqs;
                }
                acc.frame_write(f.aqrow, &qrow).map_err(ModelError::Accel)?;
                acc.frame_write(f.akstg, &krow_scaled).map_err(ModelError::Accel)?;
                op(acc.as_ref(), FrameOp::CopyRows { src: f.akstg, dst: f.kv_k[seq][full_idx], src_off: 0, dst_off: pos * n_kv * hd, n: n_kv * hd })?;
                op(acc.as_ref(), FrameOp::CopyRows { src: f.av, dst: f.kv_v[seq][full_idx], src_off: 0, dst_off: pos * n_kv * hd, n: n_kv * hd })?;
                let fs2: &dyn FrameState = acc.as_ref();
                fs2.frame_qsa_attention(f.aqrow, f.kv_k[seq][full_idx], f.kv_v[seq][full_idx], f.mask, f.aout, eng.model.hp.kq_scale(), pos + 1, n_head, n_kv, hd, 1)
                    .map_err(ModelError::Accel)?;
                let wo = eng.model.wchk(&format!("blk.{il}.attn_output.weight"))?;
                acc.frame_mm(f.aout, &wo, f.gout, 1).map_err(ModelError::Accel)?;
                full_idx += 1;
            }
            // 3) 잔차 가산: xs += gout·1.0
            op(acc.as_ref(), FrameOp::AxpyScaled { y: f.xs, x: f.gout, s: f.one, n })?;
            if std::env::var_os("LLM170_DEBUG_LAYERS").is_some() {
                let mut hv = vec![0.0f32; n];
                acc.frame_read(f.xs, &mut hv).map_err(ModelError::Accel)?;
                let m = hv.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                eprintln!("layer {il:>2} recr={} max|x|={m:.4} head={:.5},{:.5},{:.5},{:.5}",
                    eng.model.is_recr(il), hv[0], hv[1], hv[2], hv[3]);
            }

            // 4) FFN — post_norm → gate/up → silu·u → down → 잔차
            let pw = f.consts[&format!("blk.{il}.post_norm")];
            op(acc.as_ref(), FrameOp::RmsRows { x: f.xs, w: pw, out: f.xn, eps, n, w_reps: 1 })?;
            let gate_w = eng.model.wchk(&format!("blk.{il}.ffn_gate.weight"))?;
            let up_w = eng.model.wchk(&format!("blk.{il}.ffn_up.weight"))?;
            acc.frame_mm_group(f.xn, &[gate_w, up_w], &[f.fgate, f.fup], 1)
                .map_err(ModelError::Accel)?;
            op(acc.as_ref(), FrameOp::SiluMul { g: f.fgate, u: f.fup, out: f.fglu, n: hp.n_ff })?;
            let down_w = eng.model.wchk(&format!("blk.{il}.ffn_down.weight"))?;
            acc.frame_mm(f.fglu, &down_w, f.fdown, 1).map_err(ModelError::Accel)?;
            op(acc.as_ref(), FrameOp::AxpyScaled { y: f.xs, x: f.fdown, s: f.one, n })?;
        }

        // 5) head — output_norm + output GEMM + 판독 (최종 1 sync)
        {
            let wn = f.consts["output_norm"];
            op(acc.as_ref(), FrameOp::RmsRows { x: f.xs, w: wn, out: f.xn, eps, n, w_reps: 1 })?;
            let head = eng.model.wchk("output.weight")?;
            acc.frame_mm(f.xn, &head, f.logits, 1).map_err(ModelError::Accel)?;
            let mut logits = vec![0.0f32; head.n_out as usize];
            acc.frame_read(f.logits, &mut logits).map_err(ModelError::Accel)?;
            // MTP draft용 h_t 스냅샷 — 사용 중일 때만 추가 판독.
            if !eng.seqs[seq].mtp_h.is_empty() {
                let mut h = vec![0.0f32; n];
                acc.frame_read(f.xs, &mut h).map_err(ModelError::Accel)?;
                eng.seqs[seq].mtp_h.copy_from_slice(&h);
            }
            Ok(logits)
        }
    }
}
