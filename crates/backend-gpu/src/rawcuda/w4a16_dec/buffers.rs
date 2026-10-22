use super::kernels::decode::sk_ksp;
use super::mem::realloc_fields;
use super::*;

impl W4a16Dec {
    pub(super) fn realloc_buf(
        &mut self,
        buf: Buf,
        need_bytes: usize,
        cap_elems: usize,
        what: &str,
    ) -> Result<Buf, String> {
        self.cc.capture_guard(what);
        self.graph_invalidate();
        self.cc.sync()?;
        let ptr = self.cc.alloc(need_bytes)?;
        if buf.ptr != 0 {
            self.cc.free(buf.ptr)?;
        }
        Ok(Buf {
            ptr,
            cap: cap_elems,
        })
    }

    pub(super) fn ensure_norm_bufs(&mut self, t_len: usize) -> Result<(), String> {
        let need = t_len * self.hidden;
        if need > self.norm.cap {
            self.cc.capture_guard("norm_cap");
            self.graph_invalidate();
            self.cc.sync()?;
            self.norm.cap = 0;
            realloc_fields(
                |p| self.cc.free(p),
                |n| self.cc.alloc(n),
                [&mut self.norm.dx, &mut self.norm.dab, &mut self.norm.dxn],
                [need * 4, need * 4, need * 4],
            )?;
            self.norm.cap = need;
        }
        Ok(())
    }

    pub(super) fn ensure_dx32(&mut self, n: usize) -> Result<CUdeviceptr, String> {
        if n > self.dx32.cap {
            self.dx32 = self.realloc_buf(self.dx32, n * 4, n, "dx32_cap")?;
        }
        Ok(self.dx32.ptr)
    }

    pub(super) fn ensure_dx16(&mut self, n: usize) -> Result<CUdeviceptr, String> {
        if n > self.dx16.cap {
            self.dx16 = self.realloc_buf(self.dx16, n * 2, n, "dx16_cap")?;
        }
        Ok(self.dx16.ptr)
    }

    pub(super) fn ensure_dxp(&mut self, n: usize) -> Result<CUdeviceptr, String> {
        if n > self.dxp.cap {
            self.dxp = self.realloc_buf(self.dxp, n * 4, n, "dxp_cap")?;
        }
        Ok(self.dxp.ptr)
    }

    pub(super) fn ensure_ew_bufs(&mut self, n: usize) -> Result<(), String> {
        if n > self.ew.cap {
            self.cc.capture_guard("ew_cap");
            self.graph_invalidate();
            self.cc.sync()?;
            self.ew.cap = 0;
            realloc_fields(
                |p| self.cc.free(p),
                |b| self.cc.alloc(b),
                [&mut self.ew.dewg, &mut self.ew.dewu, &mut self.ew.dew],
                [n * 4, n * 4, n * 4],
            )?;
            self.ew.cap = n;
        }
        Ok(())
    }

    pub(super) fn ensure_gdn_bufs(&mut self, t_len: usize) -> Result<(), String> {
        if t_len <= self.gdn.gdn_t_cap {
            return Ok(());
        }
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        let (hd, cch, kl, vl) = (dm.hidden, dm.conv_ch(), dm.k_len(), dm.v_len());
        self.graph_invalidate();
        self.cc.sync()?;
        self.gdn.gdn_t_cap = 0;
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [
                &mut self.gdn.dqkv,
                &mut self.gdn.dzv,
                &mut self.gdn.dgxn,
                &mut self.gdn.dgq,
                &mut self.gdn.dgk,
                &mut self.gdn.dgv,
                &mut self.gdn.dq2,
                &mut self.gdn.dk2,
                &mut self.gdn.dv2,
                &mut self.gdn.dbg,
                &mut self.gdn.dgo,
                &mut self.gdn.dgate,
                &mut self.gdn.dgpart,
                &mut self.gdn.dgdc,
                &mut self.gdn.dakq,
                &mut self.gdn.dhbuf,
                &mut self.gdn.dwu,
                &mut self.gdn.ddcbuf,
            ],
            [
                t_len * cch * 4,
                t_len * vl * 4,
                t_len * hd * 4,
                t_len * kl * 4,
                t_len * kl * 4,
                t_len * vl * 4,
                t_len * kl * 4,
                t_len * kl * 4,
                t_len * vl * 4,
                t_len * dm.bg_len() * 4,
                t_len * vl * 4,
                t_len * vl * 4,
                self.n_slots * dm.h_v * 8 * 256 * 4,
                self.n_slots * dm.h_v * 128 * 4,
                dm.h_v * (CHAIN_TMAX / GDN_CS) * 2 * GDN_CS * GDN_CS * 2,
                dm.h_v * (CHAIN_TMAX / GDN_CS) * dm.d * dm.d * 2,
                dm.h_v * (CHAIN_TMAX / GDN_CS) * GDN_CS * 2 * dm.d * 4,
                dm.h_v * (CHAIN_TMAX / GDN_CS) * GDN_CS * dm.d * 4,
            ],
        )?;
        self.gdn.gdn_t_cap = t_len;
        Ok(())
    }

    pub(super) fn ensure_attn_bufs(&mut self, t_len: usize) -> Result<(), String> {
        if t_len <= self.attn.attn_t_cap {
            return Ok(());
        }
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        self.graph_invalidate();
        self.cc.sync()?;
        self.attn.attn_t_cap = 0;
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [
                &mut self.attn.dqg_a,
                &mut self.attn.dkin_a,
                &mut self.attn.dvin_a,
                &mut self.attn.dqh_a,
                &mut self.attn.doutv_a,
                &mut self.attn.dattn_part,
            ],
            [
                t_len * dm.qg_dim() * 4,
                t_len * dm.kv_dim() * 4,
                t_len * dm.kv_dim() * 4,
                t_len * dm.q_dim() * 4,
                t_len * dm.q_dim() * 4,
                t_len
                    * dm.q_heads
                    * (if t_len <= 8 {
                        ATTN_SPLITS_D
                    } else {
                        ATTN_SPLITS
                    })
                    * 258
                    * 4,
            ],
        )?;
        self.attn.attn_t_cap = t_len;
        Ok(())
    }

    pub(super) fn ensure_chain_bufs(&mut self) -> Result<(), String> {
        if self.chain_bufs_ok {
            return Ok(());
        }
        let h = self.hidden;
        let ad = self.attn.attn.ok_or("attn: 형상 미등록")?;
        let gd = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        let probe = self.window.start;
        let (ff_gate, ff_up) = if self.plain_weights {
            let (_, g, _) = self.plain_spec(&format!("blk.{probe}.moe_shared_gate.weight"))?;
            let (_, u, _) = self.plain_spec(&format!("blk.{probe}.moe_shared_up.weight"))?;
            (g, u)
        } else {
            let (_, _, g, _) = self.lin_spec(&format!("blk.{probe}.ffn_gate.weight"))?;
            let (_, _, u, _) = self.lin_spec(&format!("blk.{probe}.ffn_up.weight"))?;
            (g, u)
        };
        let w0 = ad.qg_dim().max(gd.conv_ch()).max(ff_gate);
        let w1 = ad.kv_dim().max(gd.v_len()).max(ff_up);
        let tb = CHAIN_TMAX;
        self.graph_invalidate();
        self.cc.sync()?;
        let [c0, c1, c2, c3, c4] = &mut self.dchain;
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [&mut self.dres, &mut self.dab_dev, c0, c1, c2, c3, c4],
            [
                tb * h * 4,
                tb * h * 4,
                tb * w0 * 4,
                tb * w1 * 4,
                tb * w1 * 4,
                tb * ff_up * 4,
                tb * h * 4,
            ],
        )?;
        Self::zero_dev(&self.cc, self.dab_dev, tb * h * 4)?;
        self.stg_w0 = w0;
        self.stg_w1 = w1;
        self.stg_w2 = ff_up;
        if self.dtg == 0 {
            self.dtg = self.cc.alloc(tb * 4)?;
        }
        self.build_multi_tables()?;
        self.build_sk_tables()?;
        self.chain_bufs_ok = true;
        Ok(())
    }

    fn build_multi_tables(&mut self) -> Result<(), String> {
        let plain = self.plain_weights;
        let interval = self.attn.attn.ok_or("attn: 형상 미등록")?.interval;
        let [s0, s1, s1b, _s2, _s3] = self.dchain;
        let (w0, w1) = (self.stg_w0, self.stg_w1);
        let nl = self.n_layers;
        let span = if plain { 20usize } else { 25usize };
        let mut tab: Vec<u64> = vec![0; nl * span];
        let win = self.window;
        let mut meta: Vec<[u32; 8]> = vec![[0u32; 8]; nl];
        for il in win.start..win.end {
            let mut br: Vec<(String, u64, usize)> = Vec::new();
            if (il + 1) % interval == 0 {
                br.push((format!("blk.{il}.attn_q.weight"), s0, w0));
                br.push((format!("blk.{il}.attn_k.weight"), s1, w1));
                br.push((format!("blk.{il}.attn_v.weight"), s1b, w1));
            } else {
                br.push((format!("blk.{il}.attn_qkv.weight"), s0, w0));
                br.push((format!("blk.{il}.attn_gate.weight"), s1, w1));
            }
            let gu: Vec<(String, u64, usize)> = if plain {
                Vec::new()
            } else {
                vec![
                    (format!("blk.{il}.ffn_gate.weight"), s0, w0),
                    (format!("blk.{il}.ffn_up.weight"), s1, w1),
                ]
            };
            let mut wr = il * span;
            let mut gm = [[0u32; 4]; 2];
            for (gi, segs) in [&br, &gu].into_iter().enumerate() {
                let off = wr as u32;
                let mut total = 0usize;
                let mut tgrid = 0usize;
                let mut cnt = 0u32;
                for (name, out, wmax) in segs {
                    let (p0, p1, n, k) = if plain {
                        let (w, n, k) = self.plain_spec(name)?;
                        (w, 0u64, n, k)
                    } else {
                        self.lin_spec(name)?
                    };
                    if n > *wmax {
                        return Err(format!("gemv_multi({name}): n={n} > 스테이징 {wmax}"));
                    }
                    tab[wr] = p0;
                    wr += 1;
                    if !plain {
                        tab[wr] = p1;
                        wr += 1;
                    }
                    tab[wr] = *out;
                    wr += 1;
                    tab[wr] = n as u64;
                    wr += 1;
                    tab[wr] = k as u64;
                    wr += 1;
                    total += n;
                    tgrid += n.div_ceil(GEMV_TR);
                    cnt += 1;
                }
                gm[gi] = [off, cnt, total as u32, tgrid as u32];
            }
            meta[il] = [
                gm[0][0], gm[0][1], gm[0][2], gm[1][0], gm[1][1], gm[1][2], gm[0][3], gm[1][3],
            ];
        }
        let bytes = tab.len() * 8;
        let buf = unsafe { std::slice::from_raw_parts(tab.as_ptr() as *const u8, bytes) };
        if self.dmulti == 0 {
            self.dmulti = self.cc.alloc(bytes)?;
        }
        self.cc.h2d(self.dmulti, buf)?;
        self.cc.sync()?;
        self.dmulti_meta = meta;
        Ok(())
    }

    fn build_sk_tables(&mut self) -> Result<(), String> {
        let nl = self.n_layers;
        if self.plain_weights {
            self.dsk_meta = vec![[None, None]; nl];
            return Ok(());
        }
        let interval = self.attn.attn.ok_or("attn: 형상 미등록")?.interval;
        let win = self.window;
        let [s0, s1, s1b, _s2, _s3] = self.dchain;
        let (w0, w1) = (self.stg_w0, self.stg_w1);
        let mut tab: Vec<u64> = Vec::new();
        let mut meta: Vec<[Option<SkGroup>; 2]> = vec![[None, None]; nl];
        let mut need = 0usize;
        for il in win.start..win.end {
            let br: Vec<(String, u64, usize)> = if (il + 1) % interval == 0 {
                vec![
                    (format!("blk.{il}.attn_q.weight"), s0, w0),
                    (format!("blk.{il}.attn_k.weight"), s1, w1),
                    (format!("blk.{il}.attn_v.weight"), s1b, w1),
                ]
            } else {
                vec![
                    (format!("blk.{il}.attn_qkv.weight"), s0, w0),
                    (format!("blk.{il}.attn_gate.weight"), s1, w1),
                ]
            };
            let gu: Vec<(String, u64, usize)> = vec![
                (format!("blk.{il}.ffn_gate.weight"), s0, w0),
                (format!("blk.{il}.ffn_up.weight"), s1, w1),
            ];
            let mut groups = [None, None];
            for (gi, segs) in [&br, &gu].into_iter().enumerate() {
                if let Some(g) = sk_group_meta(self, &mut tab, segs) {
                    need = need.max(g.blocks as usize * g.ksp as usize);
                    groups[gi] = Some(g);
                }
            }
            meta[il] = groups;
        }
        for (_, n, k) in self.lins_mk.values() {
            let blocks = n.div_ceil(256);
            if let Some(ksp) = sk_ksp(blocks, *k) {
                need = need.max(blocks * ksp as usize);
            }
        }
        let bytes = tab.len() * 8;
        if bytes > 0 {
            let buf = unsafe { std::slice::from_raw_parts(tab.as_ptr() as *const u8, bytes) };
            if self.dsk == 0 {
                self.dsk = self.cc.alloc(bytes)?;
            }
            self.cc.h2d(self.dsk, buf)?;
        }
        if need > self.dsk_part.cap {
            self.dsk_part = self.realloc_buf(self.dsk_part, need * 256 * 4, need, "dsk_part")?;
        }
        self.cc.sync()?;
        self.dsk_meta = meta;
        Ok(())
    }

    pub(super) fn ensure_dyt(&mut self, t: usize) -> Result<CUdeviceptr, String> {
        let (mut mn, mut mk) = (0usize, 0usize);
        for &(_, _, n, k) in self.lins.values() {
            mn = mn.max(n);
            mk = mk.max(k);
        }
        for &(_, n, k) in self.plains.values() {
            mn = mn.max(n);
            mk = mk.max(k);
        }
        mn = mn.max(self.hidden).max(self.moe.n_experts);
        let need = t * mn;
        if need > self.dyt.cap {
            self.dyt = self.realloc_buf(self.dyt, need * 4, need, "dyt_cap")?;
        }
        if t * mk > self.xh.cap {
            self.xh = self.realloc_buf(self.xh, t * mk * 2, t * mk, "xh_cap")?;
        }
        Ok(self.dyt.ptr)
    }

    pub(super) fn ensure_dy(&mut self, n: usize) -> Result<CUdeviceptr, String> {
        if n > self.dy.cap {
            self.dy = self.realloc_buf(self.dy, n * 4, n, "y_cap")?;
        }
        Ok(self.dy.ptr)
    }

    pub fn ensure_moe_stage_chain(&mut self) -> Result<(), String> {
        if self.dchain[0] != 0 {
            return Ok(());
        }
        let w = self.hidden.max(self.moe.moe_ffn).max(1) * 4;
        self.graph_invalidate();
        self.cc.sync()?;
        for p in [0usize, 1, 3, 4] {
            self.dchain[p] = self.cc.alloc(w)?;
        }
        Ok(())
    }

    pub(super) fn ensure_moe_bufs(&mut self) -> Result<(), String> {
        if self.moe.moe_bufs_ok {
            return Ok(());
        }
        let h = self.hidden;
        let n_ff = self.moe.moe_ffn;
        let n_exp = self.moe.n_experts;
        if n_ff == 0 || n_exp == 0 || !self.moe.moe_group.is_multiple_of(32) {
            return Err("moe: 구성 미등록".into());
        }
        let tmax = CHAIN_TMAX;
        let np = self.moe.top_k.max(1) * 3;
        let pk = h * n_ff / 2 * np;
        let sk = h * n_ff / self.moe.moe_group * 2 * np;
        self.graph_invalidate();
        self.cc.sync()?;
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [
                &mut self.moe.stg.q,
                &mut self.moe.stg.s,
                &mut self.moe.drt.ptr,
                &mut self.moe.dmo.ptr,
            ],
            [pk, sk, tmax * n_exp * 4, tmax * h * 4],
        )?;
        self.moe.stg.cap = (pk, sk);
        self.moe.drt.cap = tmax * n_exp;
        self.moe.dmo.cap = tmax * h;
        let tk = tmax * self.moe.top_k.max(1);
        if self.moe.moe_gslot == 0 || self.moe.moe_gmax < tk {
            self.graph_invalidate();
            self.cc.sync()?;
            for p in [self.moe.moe_gslot, self.moe.moe_gcnt, self.moe.moe_goff] {
                if p != 0 {
                    self.cc.free(p)?;
                }
            }
            self.moe.moe_gslot = 0;
            self.moe.moe_gcnt = 0;
            self.moe.moe_goff = 0;
            self.moe.moe_gmax = 0;
            self.moe.moe_gslot = self.cc.alloc(n_exp * tk * 4)?;
            self.moe.moe_gcnt = self.cc.alloc(n_exp * 4)?;
            self.moe.moe_goff = self.cc.alloc(n_exp * 4)?;
            self.moe.moe_gmax = tk;
        }
        if self.moe.exp.cap < tk {
            for p in [
                self.moe.exp.gate,
                self.moe.exp.up,
                self.moe.exp.act,
                self.moe.exp.dn,
            ] {
                if p != 0 {
                    self.cc.free(p)?;
                }
            }
            self.moe.exp.gate = 0;
            self.moe.exp.up = 0;
            self.moe.exp.act = 0;
            self.moe.exp.dn = 0;
            self.moe.exp.cap = 0;
            self.moe.exp.gate = self.cc.alloc(tk * n_ff * 4)?;
            self.moe.exp.up = self.cc.alloc(tk * n_ff * 4)?;
            self.moe.exp.act = self.cc.alloc(tk * n_ff * 4)?;
            self.moe.exp.dn = self.cc.alloc(tk * h * 4)?;
            self.moe.exp.cap = tk;
        }
        for p in [self.moe.moe_idx, self.moe.moe_wt] {
            if p != 0 {
                self.cc.free(p)?;
            }
        }
        self.moe.moe_idx = 0;
        self.moe.moe_wt = 0;
        self.moe.moe_idx = self.cc.alloc(tk * 4)?;
        self.moe.moe_wt = self.cc.alloc(tk * 4)?;
        if self.cc.stream2.is_null() {
            self.cc.create_stream2()?;
        }
        if self.moe.dsg == 0 {
            self.moe.dsg = self.cc.alloc(8 * 4)?;
        }
        let dsh_need = tmax * self.moe.shared_ffn.max(1);
        if self.moe.dsh_cap < dsh_need {
            for p in self.moe.dsh {
                if p != 0 {
                    self.cc.free(p)?;
                }
            }
            self.moe.dsh = [0, 0];
            self.moe.dsh_cap = 0;
            let sz = dsh_need * 4;
            self.moe.dsh = [self.cc.alloc(sz)?, self.cc.alloc(sz)?];
            self.moe.dsh_cap = dsh_need;
        }
        if self.moe.ev_fork.is_empty() && self.n_layers > 0 {
            for _ in 0..self.n_layers * 2 {
                let ev = self.cc.event_create_nt()?;
                self.moe.ev_fork.push(ev);
            }
        }
        if !self.moe.moe_resident && self.moe.mcache_k == 0 && self.n_layers > 0 {
            let qsz = h * n_ff / 2;
            let ssz = h * n_ff / self.moe.moe_group.max(32) * 2;
            let per_slot = 3 * (qsz + ssz);
            let free = self.free_vram().map(|(f, _)| f).unwrap_or(0);
            let per_total = (self.n_layers * per_slot).max(1) as u64;
            let kmax = if self.moe.mcache_k_plan > 0 {
                (free / per_total) as usize
            } else {
                ((free / self.tuning_lru_frac.unwrap_or(2)) / per_total) as usize
            };
            let k = if self.moe.mcache_k_plan > 0 {
                self.moe.mcache_k_plan.min(96).min(kmax)
            } else {
                self.tuning_lru_cap.unwrap_or(96).min(kmax)
            };
            if k > 0 {
                let qb = self.n_layers * k * 3 * qsz;
                let sb = self.n_layers * k * 3 * ssz;
                let qr = self.cc.alloc(qb);
                let sr = self.cc.alloc(sb);
                if let (Ok(q), Ok(s)) = (qr, sr) {
                    self.moe.mcache_q = q;
                    self.moe.mcache_s = s;
                    self.moe.mcache_k = k;
                    self.moe.mcache_ids = vec![u16::MAX; self.n_layers * k];
                    self.moe.mcache_age = vec![0u32; self.n_layers * k];
                    eprintln!(
                        "[moe] 스트리밍 전문가 캐시 {k}슬롯/층 ({}MiB)",
                        (qb + sb) >> 20
                    );
                    if let Err(e) = self.ucache.allocate(
                        "moe:expert-lru",
                        CacheCategory::ExpertWeights,
                        Tier::Hbm,
                        (qb + sb) as u64,
                        base_util::EXPERT_LRU,
                    ) {
                        eprintln!("[ucache] moe:expert-lru 등록 실패(회계 전용): {e}");
                    } else {
                        eprintln!("{}", self.ucache.log_line("moe:expert-lru"));
                    }
                    if self.moe.moe_dev_tab == 0 && self.moe.n_experts > 0 {
                        let n = self.n_layers * self.moe.n_experts * 3 * 2;
                        let p = self.cc.alloc(n * 8)?;
                        let zeros = vec![0u8; n * 8];
                        Self::h2d_chunked(&self.cc, p, &zeros)?;
                        self.moe.moe_dev_tab = p;
                    }
                }
            }
        }
        if self.gdn.gdn.is_some() {
            self.ensure_chain_bufs()?;
            self.build_moe_multi_tab()?;
        }
        self.moe.moe_bufs_ok = true;
        self.ucache_sweep()?;
        Ok(())
    }

    pub fn set_lru_tuning(&mut self, cap: Option<usize>, frac: Option<u64>) -> Result<(), String> {
        self.tuning_lru_cap = cap;
        self.tuning_lru_frac = frac;
        if self.moe.mcache_k > 0 {
            self.graph_invalidate();
            self.cc.sync()?;
            self.ucache.release_if_present("moe:expert-lru");
            for p in [self.moe.mcache_q, self.moe.mcache_s] {
                if p != 0 {
                    self.cc.free(p)?;
                }
            }
            (self.moe.mcache_q, self.moe.mcache_s, self.moe.mcache_k) = (0, 0, 0);
            self.moe.mcache_ids.clear();
            self.moe.mcache_age.clear();
            self.moe.moe_bufs_ok = false;
        }
        Ok(())
    }

    pub(super) fn ucache_sweep(&mut self) -> Result<u64, String> {
        let evicted = self.ucache.take_evicted();
        if evicted.is_empty() {
            return Ok(0);
        }
        self.graph_invalidate();
        self.cc.sync()?;
        let mut freed = 0u64;
        for e in &evicted {
            match e.id.as_str() {
                "moe:expert-lru" => {
                    for p in [self.moe.mcache_q, self.moe.mcache_s] {
                        if p != 0 {
                            self.cc.free(p)?;
                        }
                    }
                    (self.moe.mcache_q, self.moe.mcache_s, self.moe.mcache_k) = (0, 0, 0);
                    self.moe.mcache_ids.clear();
                    self.moe.mcache_age.clear();
                    self.moe.moe_bufs_ok = false;
                }
                "spec:ckpt" => {
                    if !self.gdn.dchk.is_null() {
                        self.cc.pinned_free(self.gdn.dchk)?;
                    }
                    self.gdn.dchk = std::ptr::null_mut();
                    self.gdn.dchk_pos.clear();
                    self.gdn.dchk_saved.clear();
                    self.gdn.dchk_use.clear();
                }
                "spec:state" => {
                    for p in [
                        self.gdn.dwy_s0,
                        self.gdn.dwy_dc,
                        self.gdn.dwy_g,
                        self.gdn.dwy_k,
                        self.gdn.dsnap_ring,
                        self.gdn.dring0,
                        self.gdn.dsave,
                    ] {
                        if p != 0 {
                            self.cc.free(p)?;
                        }
                    }
                    (
                        self.gdn.dwy_s0,
                        self.gdn.dwy_dc,
                        self.gdn.dwy_g,
                        self.gdn.dwy_k,
                    ) = (0, 0, 0, 0);
                    (self.gdn.dsnap_ring, self.gdn.dring0, self.gdn.dsave) = (0, 0, 0);
                    self.spec_on = false;
                }
                id => {
                    return Err(format!(
                        "ucache: 퇴거 실행 불가 엔트리 \"{id}\" — Kv는 퇴거 불가 계약 위반"
                    ));
                }
            }
            freed += e.bytes;
        }
        if freed > 0 {
            eprintln!(
                "[ucache] 퇴거 실행: {}엔트리 {}MiB 실해제",
                evicted.len(),
                freed >> 20
            );
        }
        Ok(freed)
    }

    pub fn ucache_pressure(&mut self, hbm_budget: u64) -> Result<u64, String> {
        self.ucache.set_budget(Tier::Hbm, hbm_budget);
        self.ucache_sweep()
    }

    pub(super) fn warm_for_capture(&mut self) -> Result<(), String> {
        self.ensure_chain_bufs()?;
        if self.moe.n_experts > 0 {
            self.ensure_moe_bufs()?;
        }
        self.ensure_norm_bufs(CHAIN_TMAX)?;
        self.ensure_gdn_bufs(CHAIN_TMAX)?;
        self.ensure_attn_bufs(CHAIN_TMAX)?;
        self.ensure_dyt(CHAIN_TMAX)?;
        let (mut mk, mut mn) = (0usize, 0usize);
        for &(_, _, n, k) in self.lins.values() {
            mk = mk.max(k);
            mn = mn.max(n);
        }
        for &(_, n, k) in self.plains.values() {
            mk = mk.max(k);
            mn = mn.max(n);
        }
        mk = mk.max(CHAIN_TMAX * mk.max(self.hidden));
        mn = mn.max(self.hidden);
        if mk > self.dx32.cap {
            self.dx32 = self.realloc_buf(self.dx32, mk * 4, mk, "dx32_cap")?;
        }
        if mk > self.dx16.cap {
            self.dx16 = self.realloc_buf(self.dx16, mk * 2, mk, "dx16_cap")?;
        }
        if mn > self.dy.cap {
            self.dy = self.realloc_buf(self.dy, mn * 4, mn, "y_cap")?;
        }
        Ok(())
    }

    pub(super) fn ensure_batch_bufs(&mut self) -> Result<(), String> {
        if self.head_n == 0 {
            return Err("batch: head 미등록(업로드 선행)".into());
        }
        let tmax = self
            .n_slots
            .min(BATCH_DEC_MAX)
            .max(if self.spec_on { BATCH_DEC_MAX } else { 0 })
            .max(self.batch_tmax_min);
        if self.dbatch_lg == 0 {
            self.dbatch_lg = self.cc.alloc(tmax * self.head_n * 4)?;
            self.dbatch_am = self.cc.alloc(tmax * 4)?;
        }
        if self.graph.pin_batch_tok.is_null() {
            self.graph.pin_batch_tok = self.cc.pinned_alloc(tmax * 4)?;
            self.graph.pin_batch_in = self.cc.pinned_alloc(tmax * self.hidden * 4)?;
            self.graph.pin_batch_pos = self.cc.pinned_alloc(self.n_slots * 4)?;
            self.graph.pin_batch_lg = self.cc.pinned_alloc(tmax * self.head_n * 4)?;
            self.graph.pin_row_blk = self.cc.pinned_alloc(tmax * 8)?;
            self.graph.pin_row_pp = self.cc.pinned_alloc(tmax * 8)?;
            self.graph.pin_row_ring = self.cc.pinned_alloc(tmax * 8)?;
            self.graph.pin_row_st = self.cc.pinned_alloc(tmax * 8)?;
            self.graph.pin_row_sc = self.cc.pinned_alloc(tmax * 8)?;
        }
        if self.drow_blk == 0 {
            self.drow_blk = self.cc.alloc(tmax * 8)?;
            self.drow_pp = self.cc.alloc(tmax * 8)?;
            self.drow_ring = self.cc.alloc(tmax * 8)?;
            self.drow_st = self.cc.alloc(tmax * 8)?;
            self.drow_sc = self.cc.alloc(tmax * 8)?;
        }
        Ok(())
    }
}

fn sk_group_meta(
    me: &W4a16Dec,
    tab: &mut Vec<u64>,
    segs: &[(String, u64, usize)],
) -> Option<SkGroup> {
    let mut resolved: Vec<(u64, u64, u64, usize, usize)> = Vec::new();
    let mut blocks = 0usize;
    for (name, out, wmax) in segs {
        let (_, ds, n, k) = me.lin_spec(name).ok()?;
        let (dqm, _, _) = me.lin_mk(name)?;
        if !k.is_multiple_of(128) || n > *wmax {
            return None;
        }
        resolved.push((dqm, ds, *out, n, k));
        blocks += n.div_ceil(256);
    }
    let ksp = sk_ksp(blocks, resolved[0].4)?;
    let off = tab.len() as u32;
    for (qm, s, out, n, k) in &resolved {
        tab.extend_from_slice(&[*qm, *s, *out, *n as u64, *k as u64]);
    }
    Some(SkGroup {
        off,
        cnt: resolved.len() as u32,
        blocks: blocks as u32,
        ksp,
    })
}
