use crate::rawhip::ctx::RawCtx as HipCtx;
use crate::rawvk::checks::TrellisResident;
use std::collections::HashMap;

// ── EXL3 hip 모듈층(3층 분리 원칙, 2026-10-04) ──
// [측정 원장 2026-10-04, 8060S] 모듈별 검증값:
//   GEMV 체인(lm_head k=5120 n=248320): 2.664e-4 · 117GB/s (vk 87 대비 +32%)
//   norm_resid: 2.861e-6 · 정밀 sqrt 계약
//   실선형(gate_proj·L5 혼합 krate z): 2.9-3.3e-4 · 전 krate 정상
//   배치 gemm2(전 T 16-512): 3.0-3.8e-4 · 2.8TF(스칼라 HFMA — 텐서코어화 과제)
//   GDN 체인(T=32): 1.724e-4 · rel>5% 0/196608 · 1.0ms(4커널)
//   어텐션(prep+fwd3s T=8): 1.639e-7
//   ew GPU화: 토큰 무결 유지
//   디코드(전 64층): greedy-4 완전 일치 · 로짓 1.9e-2 · tg 5.78(단일상주 안전)→sync 제거 판정 중
// 검증(exl3_hip_probe)과 메인(server)이 함께 쓰는 단일 진실:
// 가중치 상주 업로드·상태·활성 버퍼 수명·step(tok)→logits.
// 검증 자산(vk 참조·사다리 인자·덤프)은 이 층에 금지.

pub struct HipLin {
    pub k: usize,
    pub n: usize,
    pub krate: u32,
    pub suh: *mut u8,
    pub tre: *mut u8,
    pub svh: *mut u8,
}

pub struct Exl3HipDecoder {
    hc: HipCtx,
    lin: HashMap<String, HipLin>,
    hidden: usize,
    n_layers: usize,
    loaded_layers: usize,
    pos: u32,
    dx: *mut u8,
    dxn: *mut u8,
    dab: *mut u8,
    dzero: *mut u8,
    dah: *mut u8,
    dsb: *mut u8,
    dyb: *mut u8,
    dew: *mut u8,
    dqkv: *mut u8,
    dzv: *mut u8,
    dgq: *mut u8,
    dgk: *mut u8,
    dgv: *mut u8,
    dq2: *mut u8,
    dk2: *mut u8,
    dv2: *mut u8,
    dbg: *mut u8,
    dgo: *mut u8,
    dgate: *mut u8,
    dqh: *mut u8,
    dou: *mut u8,
    dring: *mut u8,
    dgst: *mut u8,
    dkc: *mut u8,
    dvc: *mut u8,
    dpp: *mut u8,
    dembed: *mut u8,
    dargmax: *mut u8,
    mtp_norms: Vec<Vec<f32>>,
    mtp_kv_k: Vec<f32>,
    mtp_kv_v: Vec<f32>,
    mtp_kv_len: usize,
    dmtpin: *mut u8,
    dnw: *mut u8,
    dqnw: *mut u8,
    dknw: *mut u8,
    dcw: *mut u8,
    dab_c: *mut u8,
    dal: *mut u8,
    ddt: *mut u8,
    dnw_g: *mut u8,
}

// SAFETY: RawCtx·할당 포인터 소유 — 단일 스레드 사용(서버 slot_loop와 동일 계약).
unsafe impl Send for Exl3HipDecoder {}

impl Exl3HipDecoder {
    /// 대형 pageable h2d는 페이지 미매핑 사례(47MB ab 내부 +2.6MB 폴트, 2026-10-04)
    /// — 4MB 청크로 나누어 모든 페이지를 확실히 커밋.
    #[allow(unused_mut)]
    fn h2d_chunked(hc: &HipCtx, mut dst: *mut u8, src: &[u8]) -> Result<(), String> {
        const CH: usize = 4 << 20;
        for off in (0..src.len()).step_by(CH) {
            let end = (off + CH).min(src.len());
            hc.h2d(unsafe { dst.add(off) }, &src[off..end])?;
        }
        Ok(())
    }
}

impl Exl3HipDecoder {
    /// lim_layers: 가중치 예산(해당 층까지만 업로드 — 메모리 절약 옵션, 검증 사다리가 악용).
    pub fn load(dir: &str, lim_layers: usize) -> Result<Self, String> {
        let hc = HipCtx::new()?;
        let mut tr = TrellisResident::load(dir)?;
        let hidden = tr.hidden;
        let n_layers = tr.n_layers;
        let n_gdn = n_layers - n_layers / 4;
        let nw = tr.norms_full_dump()?;
        eprintln!("  [hipl] nw row2[0..3]={:?}", &nw[2 * 5120..2 * 5120 + 3]);
        let (qnw, knw) = tr.attn_norms_dump()?;
        let (cw, ab_c, alog, dtb, nw_g) = tr.gdn_chain_consts()?;
        let keys = tr.linear_keys();
        // lim_layers=0: 본체 0층 + mtp 전체(모듈 격리 프로브) — 전체 키 사용.
        let need: Vec<String> = if lim_layers == 0 {
            keys.clone()
        } else if lim_layers < n_layers {
            let mut v = Vec::new();
            for il in 0..lim_layers {
                let lp = format!("model.language_model.layers.{il}");
                if il % 4 == 3 {
                    for nm in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                        v.push(format!("{lp}.self_attn.{nm}"));
                    }
                } else {
                    for nm in ["in_proj_qkv", "in_proj_z", "out_proj"] {
                        v.push(format!("{lp}.linear_attn.{nm}"));
                    }
                }
                for nm in ["gate_proj", "up_proj", "down_proj"] {
                    v.push(format!("{lp}.mlp.{nm}"));
                }
            }
            v.push("lm_head".to_string());
            v
        } else {
            keys.clone()
        };
        let f32b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let mut lin = HashMap::new();
        for key in &need {
            let (k, n, krate, suh, tre, svh) = tr.linear_raw(key)?;
            let dsuh = hc.alloc(suh.len())?;
            let dtre = hc.alloc(tre.len())?;
            let dsvh = hc.alloc(svh.len())?;
            Self::h2d_chunked(&hc, dsuh, &suh)?;
            Self::h2d_chunked(&hc, dtre, &tre)?;
            Self::h2d_chunked(&hc, dsvh, &svh)?;
            lin.insert(
                key.clone(),
                HipLin {
                    k,
                    n,
                    krate,
                    suh: dsuh,
                    tre: dtre,
                    svh: dsvh,
                },
            );
        }
        let dnw = hc.alloc(nw.len() * 4)?;
        Self::h2d_chunked(&hc, dnw, f32b(&nw))?;
        let dqnw = hc.alloc(qnw.len() * 4)?;
        Self::h2d_chunked(&hc, dqnw, f32b(&qnw))?;
        let dknw = hc.alloc(knw.len() * 4)?;
        Self::h2d_chunked(&hc, dknw, f32b(&knw))?;
        let dcw = hc.alloc(cw.len() * 4)?;
        Self::h2d_chunked(&hc, dcw, f32b(&cw))?;
        let dab_c = hc.alloc(ab_c.len() * 4)?;
        Self::h2d_chunked(&hc, dab_c, f32b(&ab_c))?;
        let dal = hc.alloc(alog.len() * 4)?;
        Self::h2d_chunked(&hc, dal, f32b(&alog))?;
        let ddt = hc.alloc(dtb.len() * 4)?;
        Self::h2d_chunked(&hc, ddt, f32b(&dtb))?;
        let dnw_g = hc.alloc(nw_g.len() * 4)?;
        Self::h2d_chunked(&hc, dnw_g, f32b(&nw_g))?;
        let dring = hc.alloc(n_gdn * 3 * 10240 * 4)?;
        let dgst = hc.alloc(n_gdn * 48 * 16384 * 4)?;
        let dkc = hc.alloc(16 * 1024 * 1024 * 4)?;
        let dvc = hc.alloc(16 * 1024 * 1024 * 4)?;
        Self::h2d_chunked(&hc, dring, &vec![0u8; n_gdn * 3 * 10240 * 4])?;
        Self::h2d_chunked(&hc, dgst, &vec![0u8; n_gdn * 48 * 16384 * 4])?;
        Self::h2d_chunked(&hc, dkc, &vec![0u8; 16 * 1024 * 1024 * 4])?;
        Self::h2d_chunked(&hc, dvc, &vec![0u8; 16 * 1024 * 1024 * 4])?;
        let dpp = hc.alloc(4)?;
        hc.h2d(dpp, &0u32.to_le_bytes())?;
        let embed_all: Vec<f32> = tr.embed.clone();
        let dembed = hc.alloc(embed_all.len() * 4)?;
        Self::h2d_chunked(&hc, dembed, f32b(&embed_all))?;
        let mtp_keys = [
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.norm.weight",
        ];
        let mut mtp_norms = Vec::with_capacity(5);
        for mk in mtp_keys {
            let w = tr.norm(mk).ok_or(format!("mtp norm {mk}"))?.to_vec();
            mtp_norms.push(w);
        }
        let (qn_w, kn_w) = (
            tr.norm("mtp.layers.0.self_attn.q_norm.weight")
                .ok_or("mtp qn")?
                .to_vec(),
            tr.norm("mtp.layers.0.self_attn.k_norm.weight")
                .ok_or("mtp kn")?
                .to_vec(),
        );
        mtp_norms.push(qn_w);
        mtp_norms.push(kn_w);
        let dmtpin = hc.alloc(128 * 1024 * 4)?; // FFN down 입력 17408 f32 상한
        drop(tr);
        let dargmax = hc.alloc(4)?;

        let tmax = 64usize;
        let dx = hc.alloc(hidden * 4)?;
        let dxn = hc.alloc(hidden * 4)?;
        let dab = hc.alloc(hidden * 4)?;
        let dzero = hc.alloc(hidden * 4)?;
        hc.h2d(dzero, &vec![0u8; hidden * 4])?;
        let dah = hc.alloc(17408 * 2)?;
        let dsb = hc.alloc(16 * 248320 * 4)?;
        let dyb = hc.alloc(248320 * 4)?;
        let dew = hc.alloc(tmax * 17408 * 4)?;
        let dqkv = hc.alloc(tmax * 10240 * 4)?;
        let dzv = hc.alloc(tmax * 6144 * 4)?;
        let dgq = hc.alloc(tmax * 2048 * 4)?;
        let dgk = hc.alloc(tmax * 2048 * 4)?;
        let dgv = hc.alloc(tmax * 6144 * 4)?;
        let dq2 = hc.alloc(tmax * 6144 * 4)?;
        let dk2 = hc.alloc(tmax * 2048 * 4)?;
        let dv2 = hc.alloc(tmax * 6144 * 4)?;
        let dbg = hc.alloc(tmax * 96 * 4)?;
        let dgo = hc.alloc(tmax * 6144 * 4)?;
        let dgate = hc.alloc(tmax * 6144 * 4)?;
        let dqh = hc.alloc(tmax * 12288 * 4)?;
        let dou = hc.alloc(tmax * 6144 * 4)?;
        Ok(Self {
            hc,
            lin,
            hidden,
            n_layers,
            loaded_layers: lim_layers,
            pos: 0,
            dx,
            dxn,
            dab,
            dzero,
            dah,
            dsb,
            dyb,
            dew,
            dqkv,
            dzv,
            dgq,
            dgk,
            dgv,
            dq2,
            dk2,
            dv2,
            dbg,
            dgo,
            dgate,
            dqh,
            dou,
            dembed,
            dargmax,
            mtp_norms,
            mtp_kv_k: vec![0f32; 4096 * 4 * 256],
            mtp_kv_v: vec![0f32; 4096 * 4 * 256],
            mtp_kv_len: 0,
            dmtpin,
            dring,
            dgst,
            dkc,
            dvc,
            dpp,
            dnw,
            dqnw,
            dknw,
            dcw,
            dab_c,
            dal,
            ddt,
            dnw_g,
        })
    }

    fn gemv_chain(&mut self, l: &HipLin, dx_in: *mut u8, dyb_out: *mut u8) -> Result<(), String> {
        let mut kc = (l.k / 128) as i32;
        let mut ks = l.k as i32;
        let (mut p0, mut p1, mut p2) = (dx_in, l.suh, self.dah);
        self.hc.launch(
            "exl3_had_in",
            (l.k / 128) as u32,
            1,
            128,
            &mut [
                &mut p0 as *mut *mut u8 as *mut _,
                &mut p1 as *mut *mut u8 as *mut _,
                &mut p2 as *mut *mut u8 as *mut _,
                &mut kc as *mut i32 as *mut _,
                &mut ks as *mut i32 as *mut _,
            ],
        )?;
        let (mut kt, mut nt, mut kk) = ((l.k / 16) as i32, (l.n / 16) as i32, l.krate as i32);
        let (mut g0, mut g1, mut g2) = (self.dah, l.tre, self.dsb);
        self.hc.launch3(
            "exl3_gemv",
            ((l.n / 16) / 8) as u32,
            16,
            1,
            128,
            &mut [
                &mut g0 as *mut *mut u8 as *mut _,
                &mut g1 as *mut *mut u8 as *mut _,
                &mut g2 as *mut *mut u8 as *mut _,
                &mut kt as *mut i32 as *mut _,
                &mut nt as *mut i32 as *mut _,
                &mut kk as *mut i32 as *mut _,
            ],
        )?;
        let (mut nch, mut nsg, mut nst) = ((l.n / 128) as i32, 16i32, l.n as i32);
        let (mut c0, mut c1, mut c2) = (self.dsb, l.svh, dyb_out);
        self.hc.launch(
            "exl3_had_out",
            (l.n / 128) as u32,
            1,
            128,
            &mut [
                &mut c0 as *mut *mut u8 as *mut _,
                &mut c1 as *mut *mut u8 as *mut _,
                &mut c2 as *mut *mut u8 as *mut _,
                &mut nch as *mut i32 as *mut _,
                &mut nsg as *mut i32 as *mut _,
                &mut nst as *mut i32 as *mut _,
            ],
        )?;
        Ok(())
    }

    fn norm(&mut self, w: usize, ab_in: *mut u8) -> Result<(), String> {
        let mut tl = 1i32;
        let mut wo = (w * 5120) as i32;
        let (mut a0, mut a1, mut a2, mut a3) = (self.dx, self.dnw, ab_in, self.dxn);
        self.hc.launch(
            "exl3_norm_resid",
            1,
            1,
            1024,
            &mut [
                &mut a0 as *mut *mut u8 as *mut _,
                &mut a1 as *mut *mut u8 as *mut _,
                &mut a2 as *mut *mut u8 as *mut _,
                &mut a3 as *mut *mut u8 as *mut _,
                &mut tl as *mut i32 as *mut _,
                &mut wo as *mut i32 as *mut _,
            ],
        )?;
        Ok(())
    }

    /// 토큰 ID 직접 forward(임베딩 행을 디바이스에서 판독) — 단일 모델 상주용.
    pub fn forward_tok(&mut self, tok: u32) -> Result<Vec<f32>, String> {
        let mut rb = vec![0u8; self.hidden * 4];
        self.hc.d2h(&mut rb, unsafe {
            self.dembed.add(tok as usize * self.hidden * 4)
        })?;
        self.hc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let row: &[f32] =
            unsafe { std::slice::from_raw_parts(rb.as_ptr() as *const f32, self.hidden) };
        let (lg, _) = self.forward(row)?;
        Ok(lg)
    }

    /// 임베딩 판독 + forward + GPU argmax — 로짓 전체 전송 없이 다음 토큰 ID만.
    pub fn step_tok(&mut self, tok: u32) -> Result<u32, String> {
        let mut rb = vec![0u8; self.hidden * 4];
        self.hc.d2h(&mut rb, unsafe {
            self.dembed.add(tok as usize * self.hidden * 4)
        })?;
        self.hc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let row: &[f32] =
            unsafe { std::slice::from_raw_parts(rb.as_ptr() as *const f32, self.hidden) };
        let _ = self.forward(row)?; // 로짓 d2h 포함(검증 경로 겸용) — 최적화 시 read 스킵 분리
        let mut an = 1i32;
        let (mut a0, mut a1) = (self.dyb, self.dargmax);
        self.hc.launch(
            "exl3_argmax",
            1,
            1,
            1024,
            &mut [
                &mut a0 as *mut *mut u8 as *mut _,
                &mut a1 as *mut *mut u8 as *mut _,
                &mut an as *mut i32 as *mut _,
            ],
        )?;
        self.hc.sync()?;
        let mut ob = vec![0u8; 4];
        self.hc.d2h(&mut ob, self.dargmax)?;
        self.hc.sync()?;
        Ok(u32::from_le_bytes([ob[0], ob[1], ob[2], ob[3]]))
    }

    /// MTP 드래프트 1스텝(plans/121 A2 수학 그대로, vk mtp_step 미러):
    /// enorm(e)‖hnorm(h) → mtp.fc → gated-attn(자체 KV, 호스트) → o+resid → FFN → resid → shared norm → lm_head.
    /// 선형은 전부 hip gemv(GPU), 노름·rope·ew·어텐션 가중합은 호스트(T=1 소형).
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_draft(&mut self, token: u32, h_in: &[f32], pos: u32) -> Result<Vec<f32>, String> {
        let h = self.hidden;
        let eps = 1e-6f32;
        let rms = |x: &[f32], w: &[f32]| -> Vec<f32> {
            let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let s = 1.0 / (ms + eps).sqrt();
            x.iter().zip(w).map(|(v, wv)| v * s * wv).collect()
        };
        // SAFETY: dembed 직접 판독은 d2h 경유가 원칙이나 여기선 h2d 직전 스텝 완료 동기 이후.
        let mut rb = vec![0u8; h * 4];
        self.hc
            .d2h(&mut rb, unsafe { self.dembed.add(token as usize * h * 4) })?;
        self.hc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let e: &[f32] = unsafe { std::slice::from_raw_parts(rb.as_ptr() as *const f32, h) };
        let e_n = rms(e, &self.mtp_norms[0]);
        let h_n = rms(h_in, &self.mtp_norms[1]);
        let mut cat = Vec::with_capacity(2 * h);
        cat.extend_from_slice(&e_n);
        cat.extend_from_slice(&h_n);
        // mtp.fc GEMV
        let mut cur = self.gemv_host("mtp.fc", &cat)?;
        // 어텐션
        let xn = rms(&cur, &self.mtp_norms[2]);
        let lp = "mtp.layers.0.self_attn";
        let q_gate = self.gemv_host(&format!("{lp}.q_proj"), &xn)?;
        let k = self.gemv_host(&format!("{lp}.k_proj"), &xn)?;
        let v = self.gemv_host(&format!("{lp}.v_proj"), &xn)?;
        let (n_head, n_kv, head_dim, n_rot) = (24usize, 4usize, 256usize, 64usize);
        let rope_base = 1e7f32;
        let rope1 = |hd: &mut [f32], pos: u32| {
            for i in 0..n_rot / 2 {
                let p = rope_base.powi(-(2 * i as i32) as i32 / n_rot as i32);
                let (a, b) = (hd[i], hd[i + n_rot / 2]);
                hd[i] = a * (pos as f32 * p).cos() - b * (pos as f32 * p).sin();
                hd[i + n_rot / 2] = a * (pos as f32 * p).sin() + b * (pos as f32 * p).sin();
            }
        };
        let mut q_heads = vec![0f32; n_head * head_dim];
        let mut gate_heads = vec![0f32; n_head * head_dim];
        for hh in 0..n_head {
            let src = hh * head_dim * 2;
            q_heads[hh * head_dim..(hh + 1) * head_dim]
                .copy_from_slice(&q_gate[src..src + head_dim]);
            gate_heads[hh * head_dim..(hh + 1) * head_dim]
                .copy_from_slice(&q_gate[src + head_dim..src + head_dim * 2]);
        }
        let kb0 = pos as usize * n_kv * head_dim;
        for hh in 0..n_head {
            let b0 = hh * head_dim;
            let mut head = q_heads[b0..b0 + head_dim].to_vec();
            head = rms(&head, &self.mtp_norms[5]);
            rope1(&mut head, pos);
            q_heads[b0..b0 + head_dim].copy_from_slice(&head);
        }
        for hh in 0..n_kv {
            let b0 = hh * head_dim;
            let mut head = k[b0..b0 + head_dim].to_vec();
            head = rms(&head, &self.mtp_norms[6]);
            rope1(&mut head, pos);
            self.mtp_kv_k[kb0 + hh * head_dim..kb0 + (hh + 1) * head_dim].copy_from_slice(&head);
            self.mtp_kv_v[kb0 + hh * head_dim..kb0 + (hh + 1) * head_dim]
                .copy_from_slice(&v[b0..b0 + head_dim]);
        }
        self.mtp_kv_len = (pos as usize + 1).max(self.mtp_kv_len);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_rep = n_head / n_kv;
        let kv_len = self.mtp_kv_len;
        let mut attn_out = vec![0f32; n_head * head_dim];
        for hh in 0..n_head {
            let kv_h = hh / n_rep;
            let b0 = hh * head_dim;
            let mut scores = vec![0f32; kv_len];
            for (tt, sc) in scores.iter_mut().enumerate() {
                let kb = tt * n_kv * head_dim + kv_h * head_dim;
                let mut d = 0f32;
                for i in 0..head_dim {
                    d += q_heads[b0 + i] * self.mtp_kv_k[kb + i];
                }
                *sc = d * scale;
            }
            let maxv = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f64;
            for sc in scores.iter_mut() {
                *sc = (*sc - maxv).exp();
                sum += *sc as f64;
            }
            for tt in 0..kv_len {
                let w = scores[tt] as f32 / sum as f32;
                let vb = tt * n_kv * head_dim + kv_h * head_dim;
                for i in 0..head_dim {
                    attn_out[b0 + i] += w * self.mtp_kv_v[vb + i];
                }
            }
            for i in 0..head_dim {
                let g = gate_heads[b0 + i];
                let sig = 1.0 / (1.0 + (-g).exp());
                attn_out[b0 + i] *= sig;
            }
        }
        let o = self.gemv_host(&format!("{lp}.o_proj"), &attn_out)?;
        for i in 0..h {
            cur[i] += o[i];
        }
        // FFN
        let xf = rms(&cur, &self.mtp_norms[3]);
        let g_ = self.gemv_host("mtp.layers.0.mlp.gate_proj", &xf)?;
        let u_ = self.gemv_host("mtp.layers.0.mlp.up_proj", &xf)?;
        let mut ewv = vec![0f32; g_.len()];
        for i in 0..g_.len() {
            let s = g_[i] / (1.0 + (-g_[i]).exp());
            ewv[i] = s * u_[i];
        }
        let d_ = self.gemv_host("mtp.layers.0.mlp.down_proj", &ewv)?;
        for i in 0..h {
            cur[i] += d_[i];
        }
        let hn = rms(&cur, &self.mtp_norms[4]);
        self.gemv_host("lm_head", &hn)
    }

    /// 호스트 벡터 → gemv 체인 1회 → 호스트 결과(드래프트 소형 전용 — 본체는 상주 경로).
    fn gemv_host(&mut self, key: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let l = self.lin.get(key).ok_or(format!("lin {key}"))?;
        let l = HipLin {
            k: l.k,
            n: l.n,
            krate: l.krate,
            suh: l.suh,
            tre: l.tre,
            svh: l.svh,
        };
        self.hc.h2d(self.dmtpin, unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4)
        })?;
        self.hc.sync()?;
        let out = self.dyb;
        self.gemv_chain(&l, self.dmtpin, out)?;
        let mut ob = vec![0u8; l.n * 4];
        self.hc.d2h(&mut ob, out)?;
        self.hc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let y: Vec<f32> =
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, l.n).to_vec() };
        Ok(y)
    }

    /// 1토큰 forward → 로짓. ew(silu·mul)는 호스트(정확성 우선 — 추후 커널화).
    pub fn forward(&mut self, embed_row: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        let n_layers = self.loaded_layers.min(self.n_layers);
        let f32b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.hc.h2d(self.dx, f32b(embed_row))?;
        let mut ab = self.dzero;
        for il in 0..n_layers {
            let lp = format!("model.language_model.layers.{il}");
            let gdn_il = (0..il).filter(|i| i % 4 != 3).count();
            self.norm(2 * il, ab)?;
            if il == 1 {
                let mut xb2 = vec![0u8; self.hidden * 4];
                self.hc.d2h(&mut xb2, self.dx)?;
                // SAFETY: d2h 완료 후 재해석.
                let xf2: &[f32] =
                    unsafe { std::slice::from_raw_parts(xb2.as_ptr() as *const f32, self.hidden) };
                let r2 = (xf2.iter().map(|v| v * v).sum::<f32>() / self.hidden as f32).sqrt();
                eprintln!("  [hipl] L1 진입 x rms={r2:.5} x0={:.6}", xf2[0]);
            }
            if il <= 1 {
                let mut xnb = vec![0u8; self.hidden * 4];
                self.hc.d2h(&mut xnb, self.dxn)?;
                // SAFETY: d2h 완료 후 재해석.
                let xnf: &[f32] =
                    unsafe { std::slice::from_raw_parts(xnb.as_ptr() as *const f32, self.hidden) };
                let r = (xnf.iter().map(|v| v * v).sum::<f32>() / self.hidden as f32).sqrt();
                eprintln!("  [hipl] L{il} xn rms={r:.5} xn0={:.6}", xnf[0]);
            }
            if il % 4 == 3 {
                let mut ai = (il / 4) as i32;
                let lq_key = format!("{lp}.self_attn.q_proj");
                let lk_key = format!("{lp}.self_attn.k_proj");
                let lv_key = format!("{lp}.self_attn.v_proj");
                let lo_key = format!("{lp}.self_attn.o_proj");
                let (dxn, dqh, dgq, dgv, dou, dab) =
                    (self.dxn, self.dqh, self.dgq, self.dgv, self.dou, self.dab);
                let lq = HipLin {
                    k: self.lin[&lq_key].k,
                    n: self.lin[&lq_key].n,
                    krate: self.lin[&lq_key].krate,
                    suh: self.lin[&lq_key].suh,
                    tre: self.lin[&lq_key].tre,
                    svh: self.lin[&lq_key].svh,
                };
                let lk = HipLin {
                    k: self.lin[&lk_key].k,
                    n: self.lin[&lk_key].n,
                    krate: self.lin[&lk_key].krate,
                    suh: self.lin[&lk_key].suh,
                    tre: self.lin[&lk_key].tre,
                    svh: self.lin[&lk_key].svh,
                };
                let lv = HipLin {
                    k: self.lin[&lv_key].k,
                    n: self.lin[&lv_key].n,
                    krate: self.lin[&lv_key].krate,
                    suh: self.lin[&lv_key].suh,
                    tre: self.lin[&lv_key].tre,
                    svh: self.lin[&lv_key].svh,
                };
                let lo = HipLin {
                    k: self.lin[&lo_key].k,
                    n: self.lin[&lo_key].n,
                    krate: self.lin[&lo_key].krate,
                    suh: self.lin[&lo_key].suh,
                    tre: self.lin[&lo_key].tre,
                    svh: self.lin[&lo_key].svh,
                };
                self.gemv_chain(&lq, dxn, dqh)?;
                self.gemv_chain(&lk, dxn, dgq)?;
                self.gemv_chain(&lv, dxn, dgv)?;
                let mut tl2 = 1i32;
                let mut p0v = self.pos as i32;
                let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) = (
                    dqh, dgq, dgv, self.dqnw, self.dknw, self.dq2, self.dkc, self.dvc, self.dpp,
                );
                self.hc.launch3(
                    "exl3_attn_prep",
                    1,
                    28,
                    1,
                    128,
                    &mut [
                        &mut a0 as *mut *mut u8 as *mut _,
                        &mut a1 as *mut *mut u8 as *mut _,
                        &mut a2 as *mut *mut u8 as *mut _,
                        &mut a3 as *mut *mut u8 as *mut _,
                        &mut a4 as *mut *mut u8 as *mut _,
                        &mut a5 as *mut *mut u8 as *mut _,
                        &mut a6 as *mut *mut u8 as *mut _,
                        &mut a7 as *mut *mut u8 as *mut _,
                        &mut a8 as *mut *mut u8 as *mut _,
                        &mut tl2 as *mut i32 as *mut _,
                        &mut p0v as *mut i32 as *mut _,
                        &mut ai as *mut i32 as *mut _,
                    ],
                )?;
                let (mut f0, mut f1, mut f2, mut f3, mut f4, mut f5) =
                    (self.dq2, self.dkc, self.dvc, dqh, dou, self.dpp);
                self.hc.launch3(
                    "exl3_attn_fwd3s",
                    1,
                    24,
                    1,
                    256,
                    &mut [
                        &mut f0 as *mut *mut u8 as *mut _,
                        &mut f1 as *mut *mut u8 as *mut _,
                        &mut f2 as *mut *mut u8 as *mut _,
                        &mut f3 as *mut *mut u8 as *mut _,
                        &mut f4 as *mut *mut u8 as *mut _,
                        &mut f5 as *mut *mut u8 as *mut _,
                        &mut tl2 as *mut i32 as *mut _,
                        &mut p0v as *mut i32 as *mut _,
                        &mut ai as *mut i32 as *mut _,
                    ],
                )?;
                self.gemv_chain(&lo, dou, dab)?;
            } else {
                let lq_key = format!("{lp}.linear_attn.in_proj_qkv");
                let lz_key = format!("{lp}.linear_attn.in_proj_z");
                let lo_key = format!("{lp}.linear_attn.out_proj");
                let lq = HipLin {
                    k: self.lin[&lq_key].k,
                    n: self.lin[&lq_key].n,
                    krate: self.lin[&lq_key].krate,
                    suh: self.lin[&lq_key].suh,
                    tre: self.lin[&lq_key].tre,
                    svh: self.lin[&lq_key].svh,
                };
                let lz = HipLin {
                    k: self.lin[&lz_key].k,
                    n: self.lin[&lz_key].n,
                    krate: self.lin[&lz_key].krate,
                    suh: self.lin[&lz_key].suh,
                    tre: self.lin[&lz_key].tre,
                    svh: self.lin[&lz_key].svh,
                };
                let lo = HipLin {
                    k: self.lin[&lo_key].k,
                    n: self.lin[&lo_key].n,
                    krate: self.lin[&lo_key].krate,
                    suh: self.lin[&lo_key].suh,
                    tre: self.lin[&lo_key].tre,
                    svh: self.lin[&lo_key].svh,
                };
                let (dxn, dqkv, dzv) = (self.dxn, self.dqkv, self.dzv);
                self.gemv_chain(&lq, dxn, dqkv)?;
                self.gemv_chain(&lz, dxn, dzv)?;
                let mut tl = 1i32;
                let mut lay = gdn_il as i32;
                let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) = (
                    self.dqkv, self.dcw, self.dring, self.dgq, self.dgk, self.dgv,
                );
                self.hc.launch(
                    "exl3_gdn_conv",
                    80,
                    1,
                    128,
                    &mut [
                        &mut a0 as *mut *mut u8 as *mut _,
                        &mut a1 as *mut *mut u8 as *mut _,
                        &mut a2 as *mut *mut u8 as *mut _,
                        &mut a3 as *mut *mut u8 as *mut _,
                        &mut a4 as *mut *mut u8 as *mut _,
                        &mut a5 as *mut *mut u8 as *mut _,
                        &mut tl as *mut i32 as *mut _,
                        &mut lay as *mut i32 as *mut _,
                    ],
                )?;
                let (mut hk16, mut hv48, mut dd128) = (16i32, 48i32, 128i32);
                let (
                    mut b0,
                    mut b1,
                    mut b2,
                    mut b3,
                    mut b4,
                    mut b5,
                    mut b6,
                    mut b7,
                    mut b8,
                    mut b9,
                    mut bb,
                ) = (
                    self.dgq, self.dgk, self.dgv, self.dxn, self.dab_c, self.dal, self.ddt,
                    self.dq2, self.dk2, self.dv2, self.dbg,
                );
                eprintln!(
                    "  [l2dbg] L{il} q={:p} k={:p} v={:p} xn={:p} ab={:p} al={:p} dt={:p} qo={:p} ko={:p} vo={:p}",
                    self.dgq,
                    self.dgk,
                    self.dgv,
                    self.dxn,
                    self.dab_c,
                    self.dal,
                    self.ddt,
                    self.dq2,
                    self.dk2,
                    self.dv2
                );
                self.hc.launch3(
                    "exl3_gdn_l2perm",
                    48,
                    1,
                    1,
                    128,
                    &mut [
                        &mut b0 as *mut *mut u8 as *mut _,
                        &mut b1 as *mut *mut u8 as *mut _,
                        &mut b2 as *mut *mut u8 as *mut _,
                        &mut b3 as *mut *mut u8 as *mut _,
                        &mut b4 as *mut *mut u8 as *mut _,
                        &mut b5 as *mut *mut u8 as *mut _,
                        &mut b6 as *mut *mut u8 as *mut _,
                        &mut b7 as *mut *mut u8 as *mut _,
                        &mut b8 as *mut *mut u8 as *mut _,
                        &mut b9 as *mut *mut u8 as *mut _,
                        &mut bb as *mut *mut u8 as *mut _,
                        &mut tl as *mut i32 as *mut _,
                        &mut lay as *mut i32 as *mut _,
                    ],
                )?;
                let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5) =
                    (self.dq2, self.dk2, self.dv2, self.dbg, self.dgst, self.dgo);
                self.hc.launch3(
                    "exl3_gdn_scan",
                    48,
                    1,
                    1,
                    128,
                    &mut [
                        &mut c0 as *mut *mut u8 as *mut _,
                        &mut c1 as *mut *mut u8 as *mut _,
                        &mut c2 as *mut *mut u8 as *mut _,
                        &mut c3 as *mut *mut u8 as *mut _,
                        &mut c4 as *mut *mut u8 as *mut _,
                        &mut c5 as *mut *mut u8 as *mut _,
                        &mut tl as *mut i32 as *mut _,
                        &mut hk16 as *mut i32 as *mut _,
                        &mut hv48 as *mut i32 as *mut _,
                        &mut dd128 as *mut i32 as *mut _,
                        &mut lay as *mut i32 as *mut _,
                    ],
                )?;
                let (mut e0, mut e1, mut e2, mut e3) = (self.dgo, self.dzv, self.dnw_g, self.dgate);
                self.hc.launch3(
                    "exl3_gdn_gate",
                    48,
                    1,
                    1,
                    128,
                    &mut [
                        &mut e0 as *mut *mut u8 as *mut _,
                        &mut e1 as *mut *mut u8 as *mut _,
                        &mut e2 as *mut *mut u8 as *mut _,
                        &mut e3 as *mut *mut u8 as *mut _,
                        &mut tl as *mut i32 as *mut _,
                        &mut lay as *mut i32 as *mut _,
                    ],
                )?;
                let dgate = self.dgate;
                self.gemv_chain(&lo, dgate, self.dab)?;
            }
            self.norm(2 * il + 1, self.dab)?;
            {
                let mut xb = vec![0u8; self.hidden * 4];
                self.hc.d2h(&mut xb, self.dx)?;
                // SAFETY: d2h 완료 후 재해석.
                let xf: &[f32] =
                    unsafe { std::slice::from_raw_parts(xb.as_ptr() as *const f32, self.hidden) };
                let r = (xf.iter().map(|v| v * v).sum::<f32>() / self.hidden as f32).sqrt();
                eprintln!("  [hipl] L{il} post-attn rms={r:.5}");
                if il == 1 {
                    let mut gb2 = vec![0u8; 8];
                    self.hc.d2h(&mut gb2, self.dgate)?;
                    // SAFETY: d2h 완료 후 재해석.
                    let gg: &[f32] =
                        unsafe { std::slice::from_raw_parts(gb2.as_ptr() as *const f32, 2) };
                    eprintln!("  [hipl] L1 gated[0..2]={gg:?}");
                }
            }
            let lg_key = format!("{lp}.mlp.gate_proj");
            let lu_key = format!("{lp}.mlp.up_proj");
            let ld_key = format!("{lp}.mlp.down_proj");
            let lg = HipLin {
                k: self.lin[&lg_key].k,
                n: self.lin[&lg_key].n,
                krate: self.lin[&lg_key].krate,
                suh: self.lin[&lg_key].suh,
                tre: self.lin[&lg_key].tre,
                svh: self.lin[&lg_key].svh,
            };
            let lu = HipLin {
                k: self.lin[&lu_key].k,
                n: self.lin[&lu_key].n,
                krate: self.lin[&lu_key].krate,
                suh: self.lin[&lu_key].suh,
                tre: self.lin[&lu_key].tre,
                svh: self.lin[&lu_key].svh,
            };
            let ld = HipLin {
                k: self.lin[&ld_key].k,
                n: self.lin[&ld_key].n,
                krate: self.lin[&ld_key].krate,
                suh: self.lin[&ld_key].suh,
                tre: self.lin[&ld_key].tre,
                svh: self.lin[&ld_key].svh,
            };
            let (dxn, dqh, dgo, dew, dab) = (self.dxn, self.dqh, self.dgo, self.dew, self.dab);
            self.gemv_chain(&lg, dxn, dqh)?;
            self.gemv_chain(&lu, dxn, dgo)?;
            let mut ewn = lg.n as i32;
            let (mut w0, mut w1, mut w2) = (dqh, dgo, dew);
            self.hc.launch3(
                "exl3_ew",
                (lg.n.div_ceil(128)) as u32,
                1,
                1,
                128,
                &mut [
                    &mut w0 as *mut *mut u8 as *mut _,
                    &mut w1 as *mut *mut u8 as *mut _,
                    &mut w2 as *mut *mut u8 as *mut _,
                    &mut ewn as *mut i32 as *mut _,
                ],
            )?;
            self.gemv_chain(&ld, dew, dab)?;
            ab = dab;
            {
                let mut xb = vec![0u8; self.hidden * 4];
                self.hc.d2h(&mut xb, self.dx)?;
                // SAFETY: d2h 완료 후 재해석.
                let xf: &[f32] =
                    unsafe { std::slice::from_raw_parts(xb.as_ptr() as *const f32, self.hidden) };
                let r = (xf.iter().map(|v| v * v).sum::<f32>() / self.hidden as f32).sqrt();
                eprintln!("  [hipl] L{il} post-ffn rms={r:.5}");
                if il == 0 {
                    let mut fb = vec![0u8; 8];
                    self.hc.d2h(&mut fb, self.dab)?;
                    // SAFETY: d2h 완료 후 재해석.
                    let ff: &[f32] =
                        unsafe { std::slice::from_raw_parts(fb.as_ptr() as *const f32, 2) };
                    eprintln!("  [hipl] L0 down[0..2]={ff:?}");
                }
            }
        }
        let mut hb = vec![0u8; self.hidden * 4];
        self.hc.d2h(&mut hb, self.dx)?;
        self.norm(128, self.dab)?;
        self.pos += 1;
        self.hc.h2d(self.dpp, &self.pos.to_le_bytes())?;
        let lh_key = "lm_head".to_string();
        let llh = HipLin {
            k: self.lin[&lh_key].k,
            n: self.lin[&lh_key].n,
            krate: self.lin[&lh_key].krate,
            suh: self.lin[&lh_key].suh,
            tre: self.lin[&lh_key].tre,
            svh: self.lin[&lh_key].svh,
        };
        self.gemv_chain(&llh, self.dxn, self.dyb)?;
        let mut lb = vec![0u8; llh.n * 4];
        self.hc.d2h(&mut lb, self.dyb)?;
        // SAFETY: d2h 완료 후 재해석.
        let logits =
            unsafe { std::slice::from_raw_parts(lb.as_ptr() as *const f32, llh.n).to_vec() };
        let hidden =
            unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const f32, self.hidden).to_vec() };
        Ok((logits, hidden))
    }
}
// 마커 mod1
// 마커 ll
// 마커 l2d
// 마커 chk
// 마커 hid
// 마커 rms1
// 마커 dcmp
// 마커 l1g
// 마커 xn1
// 마커 xn2
// 마커 lx
// 마커 rw2
// 마커 abfix
// 마커 posr
// 마커 final
// 마커 ew1
// 마커 sy1
// 마커 sr1
// 마커 fx649
// 마커 dropfx
// 마커 syn2
// 마커 am1
// 마커 mtp1
// 마커 mtp2
// 마커 mtp3
// 마커 mtp4
// 마커 mtph
// 마커 mtpi
