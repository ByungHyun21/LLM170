//! EXL3 hip MTP 드래프트(CPU·GPU) — 연속 분할(R6, plans/129, 순수 이동).
use super::Exl3HipDecoder;

impl Exl3HipDecoder {
    /// MTP 드래프트 1스텝(plans/121 A2 수학 그대로, vk mtp_step 미러):
    /// enorm(e)‖hnorm(h) → mtp.fc → gated-attn(자체 KV, 호스트) → o+resid → FFN → resid → shared norm → lm_head.
    /// 선형은 전부 hip gemv(GPU), 노름·rope·ew·어텐션 가중합은 호스트(T=1 소형).
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_draft(&mut self, token: u32, h_in: &[f32], pos: u32) -> Result<Vec<f32>, String> {
        // 호스트 MTP KV도 kvcap 상한(plans/128 P0) — kb0 인덱싱 경계 가드.
        if pos as usize >= self.kvcap as usize {
            return Err(format!(
                "mtp context overflow: pos={} >= kvcap={}",
                pos, self.kvcap
            ));
        }
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
        if self.dbg_layers {
            eprintln!("  [hfc] cur={:?}", &cur[..8]);
        }
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
                let p = rope_base.powi(-(2 * i as i32) / n_rot as i32);
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
        if self.dbg_layers {
            eprintln!("  [hat] attn={:?}", &attn_out[..8]);
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
        if self.dbg_layers {
            eprintln!("  [hff] cur2={:?}", &cur[..8]);
        }
        let hn = rms(&cur, &self.mtp_norms[4]);
        if self.dbg_layers {
            eprintln!("  [hxn] hn={:?}", &hn[..8]);
        }
        self.gemv_host("lm_head", &hn)
    }

    /// MTP 드래프트 GPU 체인(v2) — 중간 호스트 왕복 제거(라운드당 1 h2d + 종료 4B d2h).
    /// cat(enorm(e)‖hnorm(h))만 호스트 노름, 이후 전부 디바이스:
    /// fc → attn(prep/fwd3s 자체 KV) → o+resid+ffn_norm(융합) → FFN(ew) → resid+shared norm → lm_head → argmax.
    pub fn mtp_draft_gpu(&mut self, token: u32, h_in: &[f32], pos: u32) -> Result<u32, String> {
        // MTP 자체 KV(dmtpk)도 kvcap 상한 — pos가 상한이면 초과 행 기록 불가.
        if pos as usize >= self.kvcap as usize {
            return Err(format!(
                "mtp context overflow: pos={} >= kvcap={}",
                pos, self.kvcap
            ));
        }
        let h = self.hidden;
        let eps = 1e-6f32;
        let rms = |x: &[f32], w: &[f32]| -> Vec<f32> {
            let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let sc = 1.0 / (ms + eps).sqrt();
            x.iter().zip(w).map(|(v, wv)| v * sc * wv).collect()
        };
        let mut rb = vec![0u8; h * 4];
        // SAFETY: dembed 행 오프셋.
        let ep = unsafe { self.dembed.add(token as usize * h * 4) };
        self.hc.d2h(&mut rb, ep)?;
        self.hc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let e: &[f32] = unsafe { std::slice::from_raw_parts(rb.as_ptr() as *const f32, h) };
        let e_n = rms(e, &self.mtp_norms[0]);
        let h_n = rms(h_in, &self.mtp_norms[1]);
        let mut cat = Vec::with_capacity(2 * h);
        cat.extend_from_slice(&e_n);
        cat.extend_from_slice(&h_n);
        let f32b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.hc.h2d(self.dbx, f32b(&cat))?;

        let fc = self.lin["mtp.fc"].clone_shallow();
        let lq = self.lin["mtp.layers.0.self_attn.q_proj"].clone_shallow();
        let lk = self.lin["mtp.layers.0.self_attn.k_proj"].clone_shallow();
        let lv = self.lin["mtp.layers.0.self_attn.v_proj"].clone_shallow();
        let lo = self.lin["mtp.layers.0.self_attn.o_proj"].clone_shallow();
        let lg = self.lin["mtp.layers.0.mlp.gate_proj"].clone_shallow();
        let lu = self.lin["mtp.layers.0.mlp.up_proj"].clone_shallow();
        let ld = self.lin["mtp.layers.0.mlp.down_proj"].clone_shallow();
        let llh = self.lin["lm_head"].clone_shallow();
        // mtp 노름 행 포인터(dmtpnw: [0]enorm [1]hnorm [2]attn_ln [3]post_ln [4]shared [5]qn [6]kn)
        // SAFETY: dmtpnw 내 행 오프셋 — 소유 포인터 캡처(self 대여 회피).
        let dmtpnw = self.dmtpnw;
        let nrow = move |i: usize| unsafe { dmtpnw.add(i * 5120 * 4) };

        // fc → cur(dbab)
        self.had16_batch(self.dbx, fc.k, 1, fc.suh)?;
        self.gemm2_batch(&fc, 1, self.dbab)?;
        if self.dbg_layers {
            let d8 = self.dump8_dev(self.dbab);
            eprintln!("  [gfc] cur={d8:?}");
        }
        // attn_norm(순수): xn = norm(cur) — cur는 dbab 유지
        self.norm_ptr(self.dbab, nrow(2), self.dbzero, self.dbxn, 1)?;
        // q/k/v
        self.had16_batch(self.dbxn, lq.k, 1, lq.suh)?;
        self.gemm2_batch(&lq, 1, self.dsb)?;
        self.had16_batch(self.dbxn, lk.k, 1, lk.suh)?;
        self.gemm2_batch(&lk, 1, self.dsb2)?;
        self.had16_batch(self.dbxn, lv.k, 1, lv.suh)?;
        self.gemm2_batch(&lv, 1, self.dsb3)?;
        // prep+fwd3s(자체 KV, layer=0, pp=dmtpp)
        self.hc.h2d(self.dmtpp, &pos.to_le_bytes())?;
        {
            let mut kv = self.kvcap;
            let mut tl2 = 1i32;
            let mut p0v = pos as i32;
            let mut ai = 0i32;
            let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) = (
                self.dsb,
                self.dsb2,
                self.dsb3,
                nrow(5),
                nrow(6),
                self.dq2,
                self.dmtpk,
                self.dmtpv,
                self.dmtpp,
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
                    &mut kv as *mut i32 as *mut _,
                ],
            )?;
            let (mut f0, mut f1, mut f2, mut f3, mut f4, mut f5) = (
                self.dq2, self.dmtpk, self.dmtpv, self.dsb, self.dou, self.dmtpp,
            );
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
                    &mut kv as *mut i32 as *mut _,
                ],
            )?;
        }
        if self.dbg_layers {
            let d8 = self.dump8_dev(self.dou);
            eprintln!("  [gat] attn={d8:?}");
        }
        // o → resid+ffn_norm 융합: dbx(cur=cat? 아니 — cur=dbab)… 주의: resid 스트림은 cur.
        // cur을 dbx로 옮기고: norm_ptr(x=dbx, ab=o, nw=post_ln) → dbx+=o, dbxn=norm.
        // (fc 출력을 dbx에 복사하는 대신 — 위에서 dbx는 cat 입력으로 쓰였고 gemm2는 dah16 소진后 재사용 안전)
        // SAFETY 없음 — gemm2가 dbx를 더 안 읽음(입력은 dah16).
        // dbx ← cur 복사: gemm2 fc 출력 dbab을 dbx로 20KB 복사는 d2d 필요 — 대신 resid를 반대로:
        // norm_ptr(x=dbab(cur), ab=o_buf, nw=post_ln) → cur+=o in dbab, dbxn=norm ✓
        self.had16_batch(self.dou, lo.k, 1, lo.suh)?;
        self.gemm2_batch(&lo, 1, self.dsb3)?;
        self.norm_ptr(self.dbab, nrow(3), self.dsb3, self.dbxn, 1)?;
        // FFN
        self.had16_batch(self.dbxn, lg.k, 1, lg.suh)?;
        self.gemm2_batch(&lg, 1, self.dsb)?;
        self.had16_batch(self.dbxn, lu.k, 1, lu.suh)?;
        self.gemm2_batch(&lu, 1, self.dsb2)?;
        let mut ewn = lg.n as i32;
        let (mut w0, mut w1, mut w2) = (self.dsb, self.dsb2, self.dew);
        self.hc.launch3(
            "exl3_ew",
            lg.n.div_ceil(128) as u32,
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
        self.had16_batch(self.dew, ld.k, 1, ld.suh)?;
        self.gemm2_batch(&ld, 1, self.dsb3)?;
        if self.dbg_layers {
            let d8 = self.dump8_dev(self.dbab);
            eprintln!("  [gff] cur2={d8:?}");
        }
        // resid+shared norm: cur(dbab)+=ffn, xn=shared norm
        self.norm_ptr(self.dbab, nrow(4), self.dsb3, self.dbxn, 1)?;
        // lm_head → argmax
        self.had16_batch(self.dbxn, llh.k, 1, llh.suh)?;
        self.gemm2_batch(&llh, 1, self.dsb)?;
        let mut an = llh.n as i32;
        let (mut a0, mut a1) = (self.dsb, self.dargmax);
        self.hc.launch3(
            "exl3_argmax",
            1,
            1,
            1,
            1024,
            &mut [
                &mut a0 as *mut *mut u8 as *mut _,
                &mut a1 as *mut *mut u8 as *mut _,
                &mut an as *mut i32 as *mut _,
            ],
        )?;
        let mut ob = vec![0u8; 4];
        self.hc.d2h(&mut ob, self.dargmax)?;
        self.hc.sync()?;
        Ok(u32::from_le_bytes([ob[0], ob[1], ob[2], ob[3]]))
    }
}
