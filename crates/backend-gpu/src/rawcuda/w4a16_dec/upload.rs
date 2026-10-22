use super::*;

impl W4a16Dec {
    pub fn upload_lin(
        &mut self,
        name: &str,
        q: &[u8],
        s: &[u8],
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        self.upload_lin_g(name, q, s, n, k, 128, false)
    }

    pub fn upload_lin_g(
        &mut self,
        name: &str,
        q: &[u8],
        s: &[u8],
        n: usize,
        k: usize,
        group: usize,
        scale_bf16: bool,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        crate::rawcuda::gptq4::kernel_sym(false, group, scale_bf16)?;
        if q.len() != n * (k / 2) || s.len() != n * (k / group) * 2 {
            return Err(format!(
                "upload_lin_g {name}: q={} s={} != n{n} k{k} g{group} 계약",
                q.len(),
                s.len()
            ));
        }
        let dq = self.cc.alloc(q.len())?;
        let ds = match self.cc.alloc(s.len()) {
            Ok(p) => p,
            Err(e) => {
                let _ = self.cc.free(dq);
                return Err(e);
            }
        };
        let r = (|| {
            Self::h2d_chunked(&self.cc, dq, q)?;
            Self::h2d_chunked(&self.cc, ds, s)
        })();
        if let Err(e) = r {
            let _ = self.cc.free(dq);
            let _ = self.cc.free(ds);
            return Err(e);
        }
        self.weights_bytes += (q.len() + s.len()) as u64;
        if let Some((oq, os, _, _)) = self.lins.insert(name.to_string(), (dq, ds, n, k)) {
            let _ = self.cc.free(oq);
            let _ = self.cc.free(os);
        }
        if group == 128 && !scale_bf16 {
            self.lin_grp.remove(name);
        } else {
            self.lin_grp.insert(name.to_string(), (group, scale_bf16));
        }
        if (group == 128 && !scale_bf16) || (group == 32 && scale_bf16) {
            let _ = self.upload_lin_marlin(name, q, s, n, k, group, scale_bf16);
        }
        Ok(())
    }

    pub fn consolidate_marlin_dense(&mut self) {
        if self.moe.n_experts != 0 || self.lins.is_empty() || self.marlin_dense {
            return;
        }
        for (name, &(dq, _, _, k)) in self.lins.iter() {
            if dq == 0 {
                continue;
            }
            if !self.lins_mk.contains_key(name)
                || !self.lins_msc.contains_key(name)
                || !k.is_multiple_of(128)
                || self.lin_grp.get(name).copied().unwrap_or((128, false)) != (128, false)
            {
                return;
            }
        }
        let names: Vec<String> = self.lins.keys().cloned().collect();
        for name in names {
            let (dq, ds, n, k) = self.lins[&name];
            if dq == 0 {
                continue;
            }
            if self.cc.free(dq).is_ok() {
                self.lins.insert(name, (0, ds, n, k));
                let qb = (n * k / 2) as u64;
                self.weights_bytes = self.weights_bytes.saturating_sub(qb);
            }
        }
        self.marlin_dense = true;
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_lin_marlin(
        &mut self,
        name: &str,
        q: &[u8],
        s: &[u8],
        n: usize,
        k: usize,
        group: usize,
        scale_bf16: bool,
    ) -> bool {
        let repack = llm170_core::marlin::marlin_repack_q(
            unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u32, n * k / 8) },
            n,
            k,
        );
        let qm = match repack {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[marlin] {name}: 재패킹 스킵({e}) — GEMV 폴백");
                return false;
            }
        };
        let qb = unsafe { std::slice::from_raw_parts(qm.as_ptr() as *const u8, qm.len() * 4) };
        let dqm = match self.cc.alloc(qb.len()) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[marlin] {name}: 재패킹 VRAM 부족({e}) — GEMV 폴백");
                return false;
            }
        };
        if let Err(e) = Self::h2d_chunked(&self.cc, dqm, qb) {
            let _ = self.cc.free(dqm);
            eprintln!("[marlin] {name}: 재패킹 업로드 실패({e}) — GEMV 폴백");
            return false;
        }
        self.weights_bytes += qb.len() as u64;
        self.marlin_bytes += qb.len() as u64;
        if let Some((old, _, _)) = self.lins_mk.insert(name.to_string(), (dqm, n, k)) {
            let _ = self.cc.free(old);
            self.weights_bytes = self.weights_bytes.saturating_sub((n * k / 2) as u64);
            self.marlin_bytes = self.marlin_bytes.saturating_sub((n * k / 2) as u64);
        }
        let kg = k / group;
        if s.len() != n * kg * 2 {
            return false;
        }
        let sf: Vec<f32> = s
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| Self::scale_bits_to_f32(u16::from_le_bytes(*c), scale_bf16))
            .collect();
        let so: Vec<u8> = match llm170_core::marlin::marlin_permute_scales(&sf, n, k, group) {
            Ok(pf) => {
                let mut out = vec![0u8; pf.len() * 2];
                for (i, v) in pf.iter().enumerate() {
                    out[i * 2..i * 2 + 2].copy_from_slice(&Self::f32_to_f16_bits(*v).to_le_bytes());
                }
                out
            }
            Err(_) => return false,
        };
        if so.is_empty() {
            return false;
        }
        let Ok(dsc) = self.cc.alloc(so.len()) else {
            return false;
        };
        if Self::h2d_chunked(&self.cc, dsc, &so).is_err() {
            let _ = self.cc.free(dsc);
            return false;
        }
        self.weights_bytes += so.len() as u64;
        self.marlin_bytes += so.len() as u64;
        if let Some((old, _, _)) = self.lins_msc.insert(name.to_string(), (dsc, n, kg)) {
            let _ = self.cc.free(old);
            self.weights_bytes = self.weights_bytes.saturating_sub(so.len() as u64);
            self.marlin_bytes = self.marlin_bytes.saturating_sub(so.len() as u64);
        }
        true
    }

    pub fn set_norm_weights(&mut self, nw: &[f32], rows: usize) -> Result<(), String> {
        if self.hidden == 0 || self.hidden > 8192 {
            return Err(format!(
                "norm: hidden={} — (0, 8192] 계약(v[8] 레지스터 상한)",
                self.hidden
            ));
        }
        if rows == 0 || nw.len() != rows * self.hidden {
            return Err(format!(
                "norm: nw {} != rows {rows} × hidden {}",
                nw.len(),
                self.hidden
            ));
        }
        if self.dnw != 0 {
            self.cc.free(self.dnw)?;
        }
        self.dnw = 0;
        self.norm_w_rows = 0;
        let b = unsafe { std::slice::from_raw_parts(nw.as_ptr() as *const u8, nw.len() * 4) };
        let d = self.cc.alloc(nw.len() * 4)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, d, b) {
            let _ = self.cc.free(d);
            return Err(e);
        }
        self.dnw = d;
        self.norm_w_rows = rows;
        Ok(())
    }

    pub fn set_gdn(
        &mut self,
        dims: GdnDims,
        cw: &[f32],
        ab: &[f32],
        alog: &[f32],
        dtb: &[f32],
        nw: &[f32],
    ) -> Result<(), String> {
        let (n, cch, hv, hd) = (dims.n_gdn, dims.conv_ch(), dims.h_v, dims.hidden);
        if cw.len() != n * cch * 4
            || ab.len() != n * 2 * hv * hd
            || alog.len() != n * hv
            || dtb.len() != n * hv
            || nw.len() != n * 128
        {
            return Err("GDN: 상수 형상 계약 위반(cw/ab/alog/dtb/nw)".into());
        }
        for q in [
            self.gdn.dcw,
            self.gdn.dab_c,
            self.gdn.dalog,
            self.gdn.ddtb,
            self.gdn.dnwg,
            self.gdn.dring,
            self.gdn.dgst,
        ] {
            if q != 0 {
                self.cc.free(q)?;
            }
        }
        (
            self.gdn.dcw,
            self.gdn.dab_c,
            self.gdn.dalog,
            self.gdn.ddtb,
            self.gdn.dnwg,
        ) = (0, 0, 0, 0, 0);
        (self.gdn.dring, self.gdn.dgst) = (0, 0);
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let dcw = self.cc.alloc(cw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dcw, b(cw))?;
        let dab = self.cc.alloc(ab.len() * 4)?;
        Self::h2d_chunked(&self.cc, dab, b(ab))?;
        let dal = self.cc.alloc(alog.len() * 4)?;
        Self::h2d_chunked(&self.cc, dal, b(alog))?;
        let ddt = self.cc.alloc(dtb.len() * 4)?;
        Self::h2d_chunked(&self.cc, ddt, b(dtb))?;
        let dnw = self.cc.alloc(nw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dnw, b(nw))?;
        let slots = self.n_slots;
        let dring = self.cc.alloc(slots * n * 3 * cch * 4)?;
        Self::zero_dev(&self.cc, dring, slots * n * 3 * cch * 4)?;
        let dgst = self.cc.alloc(slots * n * hv * 128 * 128 * 2)?;
        Self::zero_dev(&self.cc, dgst, slots * n * hv * 128 * 128 * 2)?;
        let dsc = self.cc.alloc(slots * n * hv * 128 * 4)?;
        let ones = vec![1.0f32; slots * n * hv * 128];
        let onesb =
            unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 4) };
        Self::h2d_chunked(&self.cc, dsc, onesb)?;
        let dsc_ones = self.cc.alloc(n * hv * 128 * 4)?;
        Self::h2d_chunked(&self.cc, dsc_ones, &onesb[..n * hv * 128 * 4])?;
        (
            self.gdn.dcw,
            self.gdn.dab_c,
            self.gdn.dalog,
            self.gdn.ddtb,
            self.gdn.dnwg,
        ) = (dcw, dab, dal, ddt, dnw);
        self.gdn.dring = dring;
        self.gdn.dgst = dgst;
        self.gdn.dsc = dsc;
        self.gdn.dsc_ones = dsc_ones;
        self.gdn.gdn = Some(dims);
        Ok(())
    }

    pub fn set_kv_budget(&mut self, bytes: Option<u64>) {
        self.attn.kv_budget = bytes;
    }

    pub fn set_kv_vram_cap(&mut self, bytes: Option<u64>) {
        self.attn.kv_vram_cap = bytes;
    }

    pub(super) fn kv_block_addr(
        base_v: CUdeviceptr,
        base_h: CUdeviceptr,
        id: u64,
        stride: u64,
        vram_blocks: usize,
    ) -> u64 {
        if (id as usize) < vram_blocks {
            base_v + id * stride
        } else {
            base_h + (id - vram_blocks as u64) * stride
        }
    }

    pub(super) fn free_kv_host_arenas(&mut self) -> Result<(), String> {
        self.ucache.release_if_present("attn:kv:host");
        for p in [
            self.attn.hkc_h,
            self.attn.hvc_h,
            self.attn.hksc_h,
            self.attn.hvsc_h,
        ] {
            if !p.is_null() {
                self.cc.pinned_free(p)?;
            }
        }
        self.attn.hkc_h = std::ptr::null_mut();
        self.attn.hvc_h = std::ptr::null_mut();
        self.attn.hksc_h = std::ptr::null_mut();
        self.attn.hvsc_h = std::ptr::null_mut();
        (self.attn.dkc_h, self.attn.dvc_h) = (0, 0);
        (self.attn.dksc_h, self.attn.dvsc_h) = (0, 0);
        self.attn.kv_host_bytes = 0;
        self.attn.kv_vram_blocks = 0;
        Ok(())
    }

    pub fn kv_tier_free(&self) -> (usize, usize) {
        self.attn
            .kv_pool
            .as_ref()
            .map(|p| (p.free_vram_blocks(), p.free_host_blocks()))
            .unwrap_or((0, 0))
    }

    pub fn kv_tier_bytes(&self) -> (u64, u64, usize, usize) {
        let total = self
            .attn
            .kv_pool
            .as_ref()
            .map(|p| p.total_blocks())
            .unwrap_or(0);
        let hb = total.saturating_sub(self.attn.kv_vram_blocks);
        (self.attn.kv_arena_bytes, self.attn.kv_host_bytes, hb, total)
    }

    pub fn set_attn(
        &mut self,
        dims: AttnDims,
        layers: Vec<AttnLayer>,
        qnw: &[f32],
        knw: &[f32],
    ) -> Result<(), String> {
        let n = dims.n_attn;
        if !layers.is_empty() && layers.len() != n {
            return Err(format!("attn: layers {} != n_attn {n}", layers.len()));
        }
        let d_max = layers.iter().map(|l| l.d).max().unwrap_or(dims.d);
        if qnw.len() != n * d_max || knw.len() != n * d_max {
            return Err(format!("attn: qnw/knw != {n}x{d_max}"));
        }
        let slots = self.n_slots;
        let (kvd, kvh) = if layers.is_empty() {
            (dims.kv_dim(), dims.kv_heads)
        } else {
            (
                layers
                    .iter()
                    .map(|l| l.kv_heads * l.d)
                    .max()
                    .unwrap_or(dims.kv_dim()),
                layers
                    .iter()
                    .map(|l| l.kv_heads)
                    .max()
                    .unwrap_or(dims.kv_heads),
            )
        };
        let mb = dims.cap.div_ceil(512);
        let owned_attn = self.n_attn_owned(&dims);
        let need_blocks = slots * owned_attn * mb;
        let block_k = 512 * kvd;
        let block_s = 512 * kvh * 4;
        let per_block = 2 * (block_k + block_s) as u64;
        let total_blocks = match self.attn.kv_budget {
            Some(_) if need_blocks == 0 => 0,
            Some(b) => ((b / per_block.max(1)) as usize).clamp(1, need_blocks),
            None => need_blocks,
        };
        let diag_vram = llm170_diag::flag::val("LLM170_KV_VRAM_MB")
            .and_then(|v| v.parse::<u64>().ok())
            .map(|mb| mb << 20);
        let vram_cap = diag_vram.or(self.attn.kv_vram_cap);
        let vram_blocks = match vram_cap {
            Some(_) if total_blocks == 0 => 0,
            Some(b) => ((b / per_block.max(1)) as usize).clamp(1, total_blocks),
            None => total_blocks,
        };
        let host_blocks = total_blocks - vram_blocks;
        self.ucache.release_if_present("attn:kv");
        self.free_kv_host_arenas()?;
        for q in [
            self.attn.dqnw_a,
            self.attn.dknw_a,
            self.attn.dkc,
            self.attn.dvc,
            self.attn.dksc,
            self.attn.dvsc,
            self.attn.dblk_tab,
            self.attn.dpp,
            self.attn.dqg_a,
            self.attn.dkin_a,
            self.attn.dvin_a,
            self.attn.dqh_a,
            self.attn.doutv_a,
        ] {
            if q != 0 {
                self.cc.free(q)?;
            }
        }
        (
            self.attn.dqnw_a,
            self.attn.dknw_a,
            self.attn.dkc,
            self.attn.dvc,
            self.attn.dpp,
        ) = (0, 0, 0, 0, 0);
        (self.attn.dksc, self.attn.dvsc) = (0, 0);
        self.attn.dblk_tab = 0;
        self.attn.kv_pool = None;
        (
            self.attn.dqg_a,
            self.attn.dkin_a,
            self.attn.dvin_a,
            self.attn.dqh_a,
            self.attn.doutv_a,
        ) = (0, 0, 0, 0, 0);
        self.attn.attn_t_cap = 0;
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let dq = self.cc.alloc(qnw.len() * 4)?;
        self.attn.dqnw_a = dq;
        Self::h2d_chunked(&self.cc, dq, b(qnw))?;
        let dk = self.cc.alloc(knw.len() * 4)?;
        self.attn.dknw_a = dk;
        Self::h2d_chunked(&self.cc, dk, b(knw))?;
        let pool = super::blocktable::KvPool::new(slots, n, mb, total_blocks, vram_blocks)?;
        let (dkc, dvc) = if vram_blocks == 0 {
            (0, 0)
        } else {
            let kbytes = vram_blocks * block_k;
            let kc = self.cc.alloc(kbytes)?;
            Self::zero_dev(&self.cc, kc, kbytes)?;
            let vc = self.cc.alloc(kbytes)?;
            Self::zero_dev(&self.cc, vc, kbytes)?;
            let sbytes = vram_blocks * block_s;
            let ks = self.cc.alloc(sbytes)?;
            Self::zero_dev(&self.cc, ks, sbytes)?;
            let vs = self.cc.alloc(sbytes)?;
            Self::zero_dev(&self.cc, vs, sbytes)?;
            self.attn.dksc = ks;
            self.attn.dvsc = vs;
            (kc, vc)
        };
        if host_blocks > 0 {
            let hk = host_blocks * block_k;
            let hs = host_blocks * block_s;
            let alloc4 = || -> Result<_, String> {
                let (hkc, dkc_h) = self.cc.pinned_alloc_device(hk)?;
                let (hvc, dvc_h) = match self.cc.pinned_alloc_device(hk) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = self.cc.pinned_free(hkc);
                        return Err(e);
                    }
                };
                let (hksc, dksc_h) = match self.cc.pinned_alloc_device(hs) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = self.cc.pinned_free(hkc);
                        let _ = self.cc.pinned_free(hvc);
                        return Err(e);
                    }
                };
                let (hvsc, dvsc_h) = match self.cc.pinned_alloc_device(hs) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = self.cc.pinned_free(hkc);
                        let _ = self.cc.pinned_free(hvc);
                        let _ = self.cc.pinned_free(hksc);
                        return Err(e);
                    }
                };
                Ok(((hkc, dkc_h), (hvc, dvc_h), (hksc, dksc_h), (hvsc, dvsc_h)))
            };
            match alloc4() {
                Ok(((hkc, dkc_h), (hvc, dvc_h), (hksc, dksc_h), (hvsc, dvsc_h))) => {
                    unsafe {
                        std::ptr::write_bytes(hkc, 0, hk);
                        std::ptr::write_bytes(hvc, 0, hk);
                        std::ptr::write_bytes(hksc, 0, hs);
                        std::ptr::write_bytes(hvsc, 0, hs);
                    }
                    self.attn.hkc_h = hkc;
                    self.attn.hvc_h = hvc;
                    self.attn.hksc_h = hksc;
                    self.attn.hvsc_h = hvsc;
                    self.attn.dkc_h = dkc_h;
                    self.attn.dvc_h = dvc_h;
                    self.attn.dksc_h = dksc_h;
                    self.attn.dvsc_h = dvsc_h;
                    self.attn.kv_host_bytes = 2 * (hk + hs) as u64;
                }
                Err(e) => {
                    for q in [dkc, dvc, self.attn.dksc, self.attn.dvsc] {
                        if q != 0 {
                            let _ = self.cc.free(q);
                        }
                    }
                    (self.attn.dkc, self.attn.dvc) = (0, 0);
                    (self.attn.dksc, self.attn.dvsc) = (0, 0);
                    return Err(e);
                }
            }
        }
        self.attn.kv_vram_blocks = vram_blocks;
        let dpp = self.cc.alloc(slots * 4)?;
        Self::zero_dev(&self.cc, dpp, slots * 4)?;
        let tab = vec![0u64; slots * n * mb * 4];
        let dblk = self.cc.alloc(tab.len() * 8)?;
        let tb = unsafe { std::slice::from_raw_parts(tab.as_ptr() as *const u8, tab.len() * 8) };
        Self::h2d_chunked(&self.cc, dblk, tb)?;
        let kv_bytes = (vram_blocks * (block_k * 2 + block_s * 2)) as u64 + tab.len() as u64 * 8;
        self.attn.kv_arena_bytes = kv_bytes;
        if let Err(e) = self.ucache.allocate(
            "attn:kv",
            CacheCategory::Kv,
            Tier::Hbm,
            kv_bytes,
            base_util::KV,
        ) {
            eprintln!("[ucache] attn:kv 등록 실패(회계 전용): {e}");
        } else {
            eprintln!("{}", self.ucache.log_line("attn:kv"));
        }
        if self.attn.kv_host_bytes > 0 {
            self.ucache.release_if_present("attn:kv:host");
            if let Err(e) = self.ucache.allocate(
                "attn:kv:host",
                CacheCategory::Kv,
                Tier::HostRam,
                self.attn.kv_host_bytes,
                base_util::KV,
            ) {
                eprintln!("[ucache] attn:kv:host 등록 실패(회계 전용): {e}");
            } else {
                eprintln!("{}", self.ucache.log_line("attn:kv:host"));
            }
        }
        self.attn.dblk_tab = dblk;
        self.attn.kv_pool = Some(pool);
        self.attn.dkc = dkc;
        self.attn.dvc = dvc;
        self.attn.dpp = dpp;
        self.attn.layers = layers;
        self.attn.attn = Some(dims);
        self.ucache_sweep()?;
        Ok(())
    }

    pub fn kv_grow(&mut self, slot: usize, n_tokens: usize) -> Result<usize, String> {
        let _g = self.cc.guard()?;
        let dm = self.attn.attn.ok_or("kv_grow: attn 미등록")?;
        if slot >= self.n_slots {
            return Err(format!("kv_grow: 슬롯 {slot} ≥ {}", self.n_slots));
        }
        let mb = dm.cap.div_ceil(512);
        let nb = (n_tokens + 2).div_ceil(512).min(mb);
        let (kvd, kvh) = self.attn_max_kv()?;
        let (bk, bs) = ((512 * kvd) as u64, (512 * kvh * 4) as u64);
        let (kc, vc, ks, vs) = (self.attn.dkc, self.attn.dvc, self.attn.dksc, self.attn.dvsc);
        let (kc_h, vc_h, ks_h, vs_h) = (
            self.attn.dkc_h,
            self.attn.dvc_h,
            self.attn.dksc_h,
            self.attn.dvsc_h,
        );
        let vram_blocks = self.attn.kv_vram_blocks;
        let n_attn = dm.n_attn;
        let floor = self.kv_floor.get(slot).copied().unwrap_or(0) as usize;
        let mut new = 0usize;
        for l in 0..n_attn {
            if !self.owns_attn_layer(l, &dm) {
                continue;
            }
            let sliding = self.attn.layers.get(l).map(|x| x.sliding).unwrap_or(false);
            for bi in 0..nb {
                if sliding && bi * 512 + 512 <= floor {
                    continue;
                }
                let have = self
                    .attn
                    .kv_pool
                    .as_ref()
                    .ok_or("kv_grow: KvPool 미등록")?
                    .get(slot, l, bi)
                    .is_some();
                if have {
                    continue;
                }
                let id = self
                    .attn
                    .kv_pool
                    .as_mut()
                    .ok_or("kv_grow: KvPool 미등록")?
                    .ensure(slot, l, bi)? as u64;
                new += 1;
                let mut bytes = [0u8; 32];
                for (i2, v) in [
                    Self::kv_block_addr(kc, kc_h, id, bk, vram_blocks),
                    Self::kv_block_addr(vc, vc_h, id, bk, vram_blocks),
                    Self::kv_block_addr(ks, ks_h, id, bs, vram_blocks),
                    Self::kv_block_addr(vs, vs_h, id, bs, vram_blocks),
                ]
                .iter()
                .enumerate()
                {
                    bytes[i2 * 8..(i2 + 1) * 8].copy_from_slice(&v.to_le_bytes());
                }
                let off = (((slot * n_attn + l) * mb + bi) * 4) as u64 * 8;
                self.cc.h2d(self.attn.dblk_tab + off, &bytes)?;
            }
        }
        if slot < self.kv_assigned.len() {
            self.kv_assigned[slot] = self.kv_assigned[slot].max(n_tokens);
        }
        Ok(new)
    }

    pub(super) fn kv_ensure(&mut self, slot: usize, pos_end: usize) -> Result<(), String> {
        if slot >= self.n_slots {
            return Err(format!("kv_ensure: 슬롯 {slot} ≥ {}", self.n_slots));
        }
        if pos_end <= self.kv_assigned[slot] {
            return Ok(());
        }
        self.kv_grow(slot, pos_end)?;
        Ok(())
    }

    pub fn kv_release(&mut self, slot: usize) -> usize {
        if slot < self.kv_assigned.len() {
            self.kv_assigned[slot] = 0;
        }
        if slot < self.kv_floor.len() {
            self.kv_floor[slot] = 0;
        }
        self.attn
            .kv_pool
            .as_mut()
            .map(|p| p.release(slot))
            .unwrap_or(0)
    }

    pub fn kv_floor_of(&self, slot: usize) -> u32 {
        self.kv_floor.get(slot).copied().unwrap_or(0)
    }

    pub fn kv_release_below(&mut self, slot: usize, floor_rows: u32) -> Result<usize, String> {
        let _g = self.cc.guard()?;
        let dm = self.attn.attn.ok_or("kv_release_below: attn 미등록")?;
        if slot >= self.n_slots {
            return Err(format!("kv_release_below: 슬롯 {slot} ≥ {}", self.n_slots));
        }
        if floor_rows <= self.kv_floor_of(slot) {
            return Ok(0);
        }
        let mb = dm.cap.div_ceil(512);
        let n_attn = dm.n_attn;
        let floor_bi = (floor_rows as usize) / 512;
        let sliding: Vec<bool> = (0..n_attn)
            .map(|l| self.attn.layers.get(l).map(|x| x.sliding).unwrap_or(false))
            .collect();
        let rel = self
            .attn
            .kv_pool
            .as_mut()
            .ok_or("kv_release_below: KvPool 미등록")?
            .release_below(slot, floor_bi, &sliding);
        if rel.is_empty() {
            self.kv_floor[slot] = floor_rows;
            return Ok(0);
        }
        let zero = [0u8; 32];
        for (l, bi) in &rel {
            let off = (((slot * n_attn + l) * mb + bi) * 4) as u64 * 8;
            self.cc.h2d(self.attn.dblk_tab + off, &zero)?;
        }
        self.kv_floor[slot] = floor_rows;
        Ok(rel.len())
    }

    pub fn kv_pool_bytes(&self) -> u64 {
        self.attn.kv_arena_bytes
    }

    pub fn kv_total_blocks(&self) -> usize {
        self.attn
            .kv_pool
            .as_ref()
            .map(|p| p.total_blocks())
            .unwrap_or(0)
    }

    pub fn kv_free_blocks(&self) -> usize {
        self.attn
            .kv_pool
            .as_ref()
            .map(|p| p.free_blocks())
            .unwrap_or(0)
    }

    pub fn kv_blocks_needed(&self, n_tokens: usize) -> usize {
        match self.attn.attn {
            Some(dm) if dm.n_attn > 0 => {
                let owned = self.n_attn_owned(&dm);
                if owned == 0 {
                    return 0;
                }
                let mb = dm.cap.div_ceil(512);
                (n_tokens + 2).div_ceil(512).min(mb) * owned
            }
            _ => 0,
        }
    }

    pub fn kv_cap_tokens(&self) -> usize {
        match self.attn.attn {
            Some(dm) if dm.n_attn > 0 => {
                let owned = self.n_attn_owned(&dm);
                if owned == 0 {
                    return 0;
                }
                ((self.kv_total_blocks() / owned) * 512).saturating_sub(2)
            }
            _ => 0,
        }
    }

    pub fn kv_used_blocks(&self, slot: usize) -> usize {
        self.attn
            .kv_pool
            .as_ref()
            .map(|p| p.used_blocks(slot))
            .unwrap_or(0)
    }

    pub fn upload_plain(
        &mut self,
        name: &str,
        data: &[u8],
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || data.len() < need {
            return Err(format!(
                "upload_plain({name}): 형상 계약 위반 n={n} k={k} bytes={} < {need}",
                data.len()
            ));
        }
        let dw = self.cc.alloc(need)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, dw, &data[..need]) {
            let _ = self.cc.free(dw);
            return Err(e);
        }
        self.weights_bytes += need as u64;
        if let Some((old, _, _)) = self.plains.insert(name.to_string(), (dw, n, k)) {
            let _ = self.cc.free(old);
        }
        Ok(())
    }

    pub fn set_plain_mode(&mut self, on: bool) {
        self.plain_weights = on;
    }

    pub fn set_moe(
        &mut self,
        n_experts: usize,
        top_k: usize,
        moe_ffn: usize,
        shared_ffn: usize,
        group: usize,
        scale_bf16: bool,
        glu_gelu: bool,
    ) {
        self.moe.glu_gelu = glu_gelu;
        self.moe.n_experts = n_experts;
        self.moe.top_k = top_k;
        self.moe.moe_ffn = moe_ffn;
        self.moe.shared_ffn = shared_ffn;
        self.moe.moe_group = group;
        self.moe.moe_scale_bf16 = scale_bf16;
    }

    pub fn set_expert_table(&mut self, tab: Vec<(u64, u64, u64, u64)>) {
        self.set_expert_table_hybrid_opts(tab, 0, None)
    }

    pub fn set_expert_table_nostage(&mut self, tab: Vec<(u64, u64, u64, u64)>) {
        self.set_expert_table_hybrid_opts(tab, 0, Some(false))
    }

    pub fn set_expert_table_hybrid(&mut self, tab: Vec<(u64, u64, u64, u64)>, hot_per: usize) {
        self.set_expert_table_hybrid_opts(tab, hot_per, None)
    }

    pub fn set_expert_table_hybrid_opts(
        &mut self,
        mut tab: Vec<(u64, u64, u64, u64)>,
        hot_per: usize,
        stage: Option<bool>,
    ) {
        if self.moe.host_stage_reg > 0 {
            let _ = unsafe {
                self.cc
                    .host_unregister(self.moe.host_stage.as_ptr() as *mut std::ffi::c_void)
            };
            self.moe.host_stage_reg = 0;
        }
        let do_stage = stage.unwrap_or_else(|| llm170_diag::flag::ne0("LLM170_MOE_HOST_STAGE"));
        if do_stage {
            let n_exp = self.moe.n_experts;
            let rows: Vec<usize> = (0..tab.len())
                .filter(|&i| n_exp == 0 || (i / 3) % n_exp >= hot_per)
                .collect();
            let total: u64 = rows.iter().map(|&i| tab[i].1 + tab[i].3).sum();
            let mut buf: Vec<u8> = Vec::new();
            if total > 0 && usize::try_from(total).is_ok_and(|n| buf.try_reserve_exact(n).is_ok()) {
                let mut offs: Vec<(usize, usize, usize)> = Vec::with_capacity(rows.len());
                for &i in &rows {
                    let (qp, ql, sp, sl) = tab[i];
                    let qo = buf.len();
                    buf.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(qp as *const u8, ql as usize)
                    });
                    let so = buf.len();
                    buf.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(sp as *const u8, sl as usize)
                    });
                    offs.push((i, qo, so));
                }
                let base = buf.as_ptr() as u64;
                for &(i, qo, so) in &offs {
                    tab[i].0 = base + qo as u64;
                    tab[i].2 = base + so as u64;
                }
                eprintln!(
                    "[moe] 호스트 RAM 스테이징 {:.2}GiB — 업로드 mmap 의존 제거(C2)",
                    total as f64 / (1u64 << 30) as f64
                );
                self.moe.host_stage = buf;
                let budget_mb = llm170_diag::flag::val("LLM170_MOE_PIN_MB")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(4096);
                let budget = budget_mb << 20;
                if budget > 0 && !self.moe.host_stage.is_empty() {
                    let mut bytes = self.moe.host_stage.len().min(budget);
                    if let Some(edge) = offs
                        .iter()
                        .flat_map(|&(_, qo, so)| [qo, so])
                        .filter(|&b| b <= bytes)
                        .max()
                    {
                        bytes = edge;
                    }
                    loop {
                        match unsafe {
                            self.cc.host_register(
                                self.moe.host_stage.as_ptr() as *mut std::ffi::c_void,
                                bytes,
                            )
                        } {
                            Ok(()) => {
                                self.moe.host_stage_reg = bytes;
                                eprintln!(
                                    "# moe: 스테이징 핀드 등록 {:.2}/{}GiB — 구간 DMA 직행(P7-fix)",
                                    bytes as f64 / (1u64 << 30) as f64,
                                    self.moe.host_stage.len() >> 30
                                );
                                break;
                            }
                            Err(e) => {
                                if bytes <= (4 << 20) {
                                    eprintln!(
                                        "# moe: 핀드 등록 실패({e}) — 드라이버 스테이징 유지(P7-fix)"
                                    );
                                    break;
                                }
                                bytes = (bytes / 4).max(4 << 20);
                            }
                        }
                    }
                }
            } else {
                eprintln!("# moe: 호스트 스테이징 {total}B 할당 실패 — mmap 스트리밍 유지");
            }
        }
        self.experts_bytes = tab.iter().map(|e| e.1 + e.3).sum();
        self.moe.moe_tab = tab;
    }

    pub fn upload_experts_resident(
        &mut self,
        host_tab: &[(u64, u64, u64, u64)],
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        if self.moe.n_experts == 0 || host_tab.len() != self.n_layers * self.moe.n_experts * 3 {
            return Err("upload_experts_resident: 테이블 형상 위반".into());
        }
        let mut pk = [0usize; 3];
        let mut sk = [0usize; 3];
        for (i, e) in host_tab.iter().enumerate() {
            pk[i % 3] += e.1 as usize;
            sk[i % 3] += e.3 as usize;
        }
        let (dpk, dsk, allocd) = self.alloc_expert_arenas(pk, sk)?;
        let mut dev_tab = Vec::with_capacity(host_tab.len());
        let (mut opk, mut osk) = ([0usize; 3], [0usize; 3]);
        for (i, e) in host_tab.iter().enumerate() {
            let p = i % 3;
            dev_tab.push((dpk[p] + opk[p] as u64, e.1, dsk[p] + osk[p] as u64, e.3));
            opk[p] += e.1 as usize;
            osk[p] += e.3 as usize;
        }
        let (mut opk2, mut osk2) = ([0usize; 3], [0usize; 3]);
        let r = (|| -> Result<(), String> {
            for (i, e) in host_tab.iter().enumerate() {
                let p = i % 3;
                let qb = unsafe { std::slice::from_raw_parts(e.0 as *const u8, e.1 as usize) };
                let sb = unsafe { std::slice::from_raw_parts(e.2 as *const u8, e.3 as usize) };
                Self::h2d_chunked(&self.cc, dpk[p] + opk2[p] as u64, qb)?;
                Self::h2d_chunked(&self.cc, dsk[p] + osk2[p] as u64, sb)?;
                opk2[p] += e.1 as usize;
                osk2[p] += e.3 as usize;
            }
            Ok(())
        })();
        if let Err(err) = r {
            for p in allocd {
                let _ = self.cc.free(p);
            }
            return Err(err);
        }
        let experts_bytes: u64 = host_tab.iter().map(|e| e.1 + e.3).sum();
        self.register_moe_resident(dev_tab, experts_bytes)
    }

    fn alloc_expert_arenas(
        &mut self,
        pk: [usize; 3],
        sk: [usize; 3],
    ) -> Result<([CUdeviceptr; 3], [CUdeviceptr; 3], Vec<CUdeviceptr>), String> {
        let mut dpk = [0 as CUdeviceptr; 3];
        let mut dsk = [0 as CUdeviceptr; 3];
        let mut allocd = Vec::new();
        let setup = (|| -> Result<(), String> {
            for p in 0..3 {
                dpk[p] = self.cc.alloc(pk[p])?;
                allocd.push(dpk[p]);
                dsk[p] = self.cc.alloc(sk[p])?;
                allocd.push(dsk[p]);
            }
            Ok(())
        })();
        if let Err(e) = setup {
            for p in allocd {
                let _ = self.cc.free(p);
            }
            return Err(e);
        }
        Ok((dpk, dsk, allocd))
    }

    fn register_moe_resident(
        &mut self,
        dev_tab: Vec<(u64, u64, u64, u64)>,
        experts_bytes: u64,
    ) -> Result<(), String> {
        let mut flat: Vec<u64> = Vec::with_capacity(dev_tab.len() * 2);
        for e in &dev_tab {
            flat.push(e.0);
            flat.push(e.2);
        }
        let fb = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 8) };
        if self.moe.moe_dev_tab != 0 {
            let _ = self.cc.free(self.moe.moe_dev_tab);
            self.moe.moe_dev_tab = 0;
        }
        let dtab = self.cc.alloc(flat.len() * 8)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, dtab, fb) {
            let _ = self.cc.free(dtab);
            return Err(e);
        }
        self.moe.moe_dev_tab = dtab;
        self.moe.moe_tab = dev_tab;
        self.experts_bytes = experts_bytes;
        self.weights_bytes += experts_bytes;
        self.moe.moe_resident = true;
        Ok(())
    }

    pub fn upload_experts_resident_streamed(
        &mut self,
        lens: &[(usize, usize)],
        fill: &mut dyn FnMut(usize, usize, u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        if self.moe.n_experts == 0 || lens.len() != self.n_layers * self.moe.n_experts * 3 {
            return Err("upload_experts_resident_streamed: 테이블 형상 위반".into());
        }
        let mut pk = [0usize; 3];
        let mut sk = [0usize; 3];
        for (i, &(ql, sl)) in lens.iter().enumerate() {
            pk[i % 3] += ql;
            sk[i % 3] += sl;
        }
        let (dpk, dsk, allocd) = self.alloc_expert_arenas(pk, sk)?;
        let mut dev_tab = Vec::with_capacity(lens.len());
        let (mut opk, mut osk) = ([0usize; 3], [0usize; 3]);
        for (i, &(ql, sl)) in lens.iter().enumerate() {
            let p = i % 3;
            dev_tab.push((
                dpk[p] + opk[p] as u64,
                ql as u64,
                dsk[p] + osk[p] as u64,
                sl as u64,
            ));
            opk[p] += ql;
            osk[p] += sl;
        }
        let r = (|| -> Result<(), String> {
            let mut ring = super::mem::PinnedRing::new(&self.cc, STREAM_CHUNK)?;
            let (mut cpk, mut csk) = ([0usize; 3], [0usize; 3]);
            for (i, &(ql, sl)) in lens.iter().enumerate() {
                let p = i % 3;
                if ql > 0 {
                    ring.send(dpk[p] + cpk[p] as u64, ql, &mut |off, buf| {
                        fill(i, 0, off, buf)
                    })?;
                }
                if sl > 0 {
                    ring.send(dsk[p] + csk[p] as u64, sl, &mut |off, buf| {
                        fill(i, 1, off, buf)
                    })?;
                }
                cpk[p] += ql;
                csk[p] += sl;
                llm170_diag::watchdog::bump();
            }
            ring.sync()
        })();
        if let Err(err) = r {
            for p in allocd {
                let _ = self.cc.free(p);
            }
            return Err(err);
        }
        let experts_bytes: u64 = lens.iter().map(|e| (e.0 + e.1) as u64).sum();
        if let Err(e) = self.build_marlin_moe_arenas(lens, fill) {
            eprintln!("[marlin_moe] 재패킹 스킵({e}) — 레거시 커널 폴백");
        }
        self.register_moe_resident(dev_tab, experts_bytes)
    }

    fn scale_bits_to_f32(b: u16, scale_bf16: bool) -> f32 {
        if scale_bf16 {
            llm170_core::quant::deq::bf16_to_f32(b)
        } else {
            llm170_core::quant::deq::half_to_f32(b)
        }
    }

    fn f32_to_f16_bits(v: f32) -> u16 {
        let x = v.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
        let man = x & 0x7F_FFFF;
        if exp >= 0x1F {
            return sign | 0x7C00;
        }
        if exp <= 0 {
            if exp < -10 {
                return sign;
            }
            let m = (man | 0x80_0000) >> (14 - exp);
            return sign | (m as u16);
        }
        let m = man >> 13;
        let mut r = sign | ((exp as u16) << 10) | (m as u16);
        let rem = man & 0x1FFF;
        if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
            r = r.wrapping_add(1);
        }
        r
    }

    fn build_marlin_moe_arenas(
        &mut self,
        lens: &[(usize, usize)],
        fill: &mut dyn FnMut(usize, usize, u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let (nl, ne, nff, h) = (
            self.n_layers,
            self.moe.n_experts,
            self.moe.moe_ffn,
            self.hidden,
        );
        let (k13, n13) = (h, 2 * nff);
        let (k2, n2) = (nff, h);
        if !k13.is_multiple_of(128)
            || !n13.is_multiple_of(64)
            || !k2.is_multiple_of(16)
            || !n2.is_multiple_of(64)
        {
            return Err(format!(
                "marlin_moe 계약: k13={k13} n13={n13} k2={k2} n2={n2}"
            ));
        }
        let g13 = k13 / 32;
        let g2 = k2 / 32;
        let st13 = n13 * k13 / 2;
        let st2 = n2 * k2 / 2;
        let ss13 = g13 * n13 * 2;
        let ss2 = g2 * n2 * 2;
        let (a13, a2, s13, s2) = (
            self.cc.alloc(nl * ne * st13)?,
            self.cc.alloc(nl * ne * st2)?,
            self.cc.alloc(nl * ne * ss13)?,
            self.cc.alloc(nl * ne * ss2)?,
        );
        let r = (|| -> Result<(), String> {
            const RCHUNK: usize = 8 << 20;
            let mat = |fill: &mut dyn FnMut(usize, usize, u64, &mut [u8]) -> Result<(), String>,
                       i: usize,
                       which: usize,
                       len: usize|
             -> Result<Vec<u8>, String> {
                let mut b = vec![0u8; len];
                for off in (0..len).step_by(RCHUNK) {
                    let end = (off + RCHUNK).min(len);
                    fill(i, which, off as u64, &mut b[off..end])?;
                }
                Ok(b)
            };
            let perm16 = |sb: &[u8], n: usize, k: usize, gs: usize| -> Result<Vec<u8>, String> {
                let g = k / gs;
                let sf: Vec<f32> = sb
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect();
                let pf = llm170_core::marlin::marlin_permute_scales(&sf, n, k, gs)?;
                let mut out = vec![0u8; g * n * 2];
                for (i, v) in pf.iter().enumerate() {
                    out[i * 2..i * 2 + 2].copy_from_slice(&Self::f32_to_f16_bits(*v).to_le_bytes());
                }
                Ok(out)
            };
            for layer in 0..nl {
                for e in 0..ne {
                    let i0 = (layer * ne + e) * 3;
                    let (ql0, sl0) = lens[i0];
                    let (ql1, sl1) = lens[i0 + 1];
                    let (qb0, sb0) = (mat(fill, i0, 0, ql0)?, mat(fill, i0, 1, sl0)?);
                    let (qb1, sb1) = (mat(fill, i0 + 1, 0, ql1)?, mat(fill, i0 + 1, 1, sl1)?);
                    let mut qcat = qb0;
                    qcat.extend_from_slice(&qb1);
                    let q32: Vec<u32> = qcat
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let rp = llm170_core::marlin::marlin_repack_q(&q32, n13, k13)?;
                    let rb: Vec<u8> = rp.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let off = (layer * ne + e) as u64 * st13 as u64;
                    Self::h2d_chunked(&self.cc, a13 + off, &rb)?;
                    let mut scat = sb0;
                    scat.extend_from_slice(&sb1);
                    let sp = perm16(&scat, n13, k13, 32)?;
                    Self::h2d_chunked(&self.cc, s13 + (layer * ne + e) as u64 * ss13 as u64, &sp)?;
                    let (ql2, sl2) = lens[i0 + 2];
                    let qb2 = mat(fill, i0 + 2, 0, ql2)?;
                    let sb2 = mat(fill, i0 + 2, 1, sl2)?;
                    let q32b: Vec<u32> = qb2
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let rp2 = llm170_core::marlin::marlin_repack_q(&q32b, n2, k2)?;
                    let rb2: Vec<u8> = rp2.iter().flat_map(|v| v.to_le_bytes()).collect();
                    Self::h2d_chunked(&self.cc, a2 + (layer * ne + e) as u64 * st2 as u64, &rb2)?;
                    let sp2 = perm16(&sb2, n2, k2, 32)?;
                    Self::h2d_chunked(&self.cc, s2 + (layer * ne + e) as u64 * ss2 as u64, &sp2)?;
                    llm170_diag::watchdog::bump();
                }
            }
            self.cc.sync()
        })();
        if let Err(e) = r {
            for p in [a13, a2, s13, s2] {
                let _ = self.cc.free(p);
            }
            return Err(e);
        }
        self.moe_m13 = a13;
        self.moe_m2 = a2;
        self.moe_s13 = s13;
        self.moe_s2 = s2;
        let bytes = (nl * ne * (st13 + st2 + ss13 + ss2)) as u64;
        self.weights_bytes += bytes;
        self.marlin_bytes += bytes;
        Ok(())
    }

    pub fn upload_experts_hybrid_streamed(
        &mut self,
        lens: &[(usize, usize)],
        fill: &mut dyn FnMut(usize, usize, u64, &mut [u8]) -> Result<(), String>,
        hot_per: usize,
        mcache_k_plan: usize,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let n_exp = self.moe.n_experts;
        if n_exp == 0 || lens.len() != self.n_layers * n_exp * 3 || hot_per == 0 || hot_per >= n_exp
        {
            return Err("upload_experts_hybrid_streamed: 인자 위반".into());
        }
        let hot_cell = |i: usize| (i / 3) % n_exp < hot_per;
        let (mut pk, mut sk) = ([0usize; 3], [0usize; 3]);
        for (i, &(ql, sl)) in lens.iter().enumerate() {
            if hot_cell(i) {
                pk[i % 3] += ql;
                sk[i % 3] += sl;
            }
        }
        let (dpk, dsk, allocd) = self.alloc_expert_arenas(pk, sk)?;
        let mut dev_tab = Vec::with_capacity(lens.len());
        let (mut opk, mut osk) = ([0usize; 3], [0usize; 3]);
        for (i, &(ql, sl)) in lens.iter().enumerate() {
            let p = i % 3;
            if hot_cell(i) {
                dev_tab.push((
                    dpk[p] + opk[p] as u64,
                    ql as u64,
                    dsk[p] + osk[p] as u64,
                    sl as u64,
                ));
                opk[p] += ql;
                osk[p] += sl;
            } else {
                dev_tab.push((0, ql as u64, 0, sl as u64));
            }
        }
        let r = (|| -> Result<(), String> {
            let mut ring = super::mem::PinnedRing::new(&self.cc, STREAM_CHUNK)?;
            let (mut cpk, mut csk) = ([0usize; 3], [0usize; 3]);
            for (i, &(ql, sl)) in lens.iter().enumerate() {
                let p = i % 3;
                if !hot_cell(i) {
                    continue;
                }
                if ql > 0 {
                    ring.send(dpk[p] + cpk[p] as u64, ql, &mut |off, buf| {
                        fill(i, 0, off, buf)
                    })?;
                }
                if sl > 0 {
                    ring.send(dsk[p] + csk[p] as u64, sl, &mut |off, buf| {
                        fill(i, 1, off, buf)
                    })?;
                }
                cpk[p] += ql;
                csk[p] += sl;
                llm170_diag::watchdog::bump();
            }
            ring.sync()
        })();
        if let Err(err) = r {
            for p in allocd {
                let _ = self.cc.free(p);
            }
            return Err(err);
        }
        let mut flat: Vec<u64> = Vec::with_capacity(dev_tab.len() * 2);
        for e in &dev_tab {
            flat.push(e.0);
            flat.push(e.2);
        }
        let fb = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 8) };
        if self.moe.moe_dev_tab != 0 {
            let _ = self.cc.free(self.moe.moe_dev_tab);
            self.moe.moe_dev_tab = 0;
        }
        let dtab = self.cc.alloc(flat.len() * 8)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, dtab, fb) {
            let _ = self.cc.free(dtab);
            return Err(e);
        }
        self.moe.moe_dev_tab = dtab;
        let hot_bytes: u64 = lens
            .iter()
            .enumerate()
            .filter(|&(i, _)| hot_cell(i))
            .map(|(_, &(ql, sl))| (ql + sl) as u64)
            .sum();
        self.weights_bytes += hot_bytes;
        self.moe.hot_tab = dev_tab;
        self.moe.hot_per_layer = hot_per;
        self.moe.mcache_k_plan = mcache_k_plan;
        Ok(())
    }

    pub fn upload_experts_marlin_only_streamed(
        &mut self,
        lens_win: &[(usize, usize)],
        fill: &mut dyn FnMut(usize, usize, u64, &mut [u8]) -> Result<(), String>,
        layer_lo: usize,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let (nl_all, ne, nff, h) = (
            self.n_layers,
            self.moe.n_experts,
            self.moe.moe_ffn,
            self.hidden,
        );
        let n_win = lens_win.len().checked_div(3 * ne.max(1)).unwrap_or(0);
        if ne == 0 || n_win == 0 || layer_lo + n_win > nl_all {
            return Err(format!(
                "marlin_only: lens {} → n_win {n_win} lo {layer_lo} nl {nl_all} 위반",
                lens_win.len()
            ));
        }
        let (k13, n13) = (h, 2 * nff);
        let (k2, n2) = (nff, h);
        if !k13.is_multiple_of(128)
            || !n13.is_multiple_of(64)
            || !k2.is_multiple_of(16)
            || !n2.is_multiple_of(64)
        {
            return Err(format!(
                "marlin_only 계약: k13={k13} n13={n13} k2={k2} n2={n2}"
            ));
        }
        let (g13, g2) = (k13 / 32, k2 / 32);
        let (st13, st2) = (n13 * k13 / 2, n2 * k2 / 2);
        let (ss13, ss2) = (g13 * n13 * 2, g2 * n2 * 2);
        let (a13, a2, s13, s2) = (
            self.cc.alloc(n_win * ne * st13)?,
            self.cc.alloc(n_win * ne * st2)?,
            self.cc.alloc(n_win * ne * ss13)?,
            self.cc.alloc(n_win * ne * ss2)?,
        );
        const RCHUNK: usize = 8 << 20;
        let mat = |fill: &mut dyn FnMut(usize, usize, u64, &mut [u8]) -> Result<(), String>,
                   i: usize,
                   which: usize,
                   len: usize|
         -> Result<Vec<u8>, String> {
            let mut b = vec![0u8; len];
            for off in (0..len).step_by(RCHUNK) {
                let end = (off + RCHUNK).min(len);
                fill(i, which, off as u64, &mut b[off..end])?;
            }
            Ok(b)
        };
        let scale_bf16 = self.moe.moe_scale_bf16;
        let perm16 = |sb: &[u8], n: usize, k: usize, gs: usize| -> Result<Vec<u8>, String> {
            let g = k / gs;
            let sf: Vec<f32> = sb
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| Self::scale_bits_to_f32(u16::from_le_bytes(*c), scale_bf16))
                .collect();
            let pf = llm170_core::marlin::marlin_permute_scales(&sf, n, k, gs)?;
            let mut out = vec![0u8; g * n * 2];
            for (i, v) in pf.iter().enumerate() {
                out[i * 2..i * 2 + 2].copy_from_slice(&Self::f32_to_f16_bits(*v).to_le_bytes());
            }
            Ok(out)
        };
        let r = (|| -> Result<(), String> {
            for layer in 0..n_win {
                for e in 0..ne {
                    let i0 = (layer * ne + e) * 3;
                    let (ql0, sl0) = lens_win[i0];
                    let (ql1, sl1) = lens_win[i0 + 1];
                    let (qb0, sb0) = (mat(fill, i0, 0, ql0)?, mat(fill, i0, 1, sl0)?);
                    let (qb1, sb1) = (mat(fill, i0 + 1, 0, ql1)?, mat(fill, i0 + 1, 1, sl1)?);
                    let mut qcat = qb0;
                    qcat.extend_from_slice(&qb1);
                    let q32: Vec<u32> = qcat
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let rp = llm170_core::marlin::marlin_repack_q(&q32, n13, k13)?;
                    let rb: Vec<u8> = rp.iter().flat_map(|v| v.to_le_bytes()).collect();
                    Self::h2d_chunked(&self.cc, a13 + (layer * ne + e) as u64 * st13 as u64, &rb)?;
                    let mut scat = sb0;
                    scat.extend_from_slice(&sb1);
                    let sp = perm16(&scat, n13, k13, 32)?;
                    Self::h2d_chunked(&self.cc, s13 + (layer * ne + e) as u64 * ss13 as u64, &sp)?;
                    let (ql2, sl2) = lens_win[i0 + 2];
                    let qb2 = mat(fill, i0 + 2, 0, ql2)?;
                    let sb2 = mat(fill, i0 + 2, 1, sl2)?;
                    let q32b: Vec<u32> = qb2
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let rp2 = llm170_core::marlin::marlin_repack_q(&q32b, n2, k2)?;
                    let rb2: Vec<u8> = rp2.iter().flat_map(|v| v.to_le_bytes()).collect();
                    Self::h2d_chunked(&self.cc, a2 + (layer * ne + e) as u64 * st2 as u64, &rb2)?;
                    let sp2 = perm16(&sb2, n2, k2, 32)?;
                    Self::h2d_chunked(&self.cc, s2 + (layer * ne + e) as u64 * ss2 as u64, &sp2)?;
                    llm170_diag::watchdog::bump();
                }
            }
            self.cc.sync()
        })();
        if let Err(e) = r {
            for p in [a13, a2, s13, s2] {
                let _ = self.cc.free(p);
            }
            return Err(e);
        }
        let (off13, off2) = (
            (layer_lo as u64 * ne as u64) * st13 as u64,
            (layer_lo as u64 * ne as u64) * st2 as u64,
        );
        let (os13, os2) = (
            (layer_lo as u64 * ne as u64) * ss13 as u64,
            (layer_lo as u64 * ne as u64) * ss2 as u64,
        );
        self.moe_m13 = a13.wrapping_sub(off13);
        self.moe_m2 = a2.wrapping_sub(off2);
        self.moe_s13 = s13.wrapping_sub(os13);
        self.moe_s2 = s2.wrapping_sub(os2);
        let bytes = (n_win * ne * (st13 + st2 + ss13 + ss2)) as u64;
        self.weights_bytes += bytes;
        self.set_expert_table(vec![(0u64, 0u64, 0u64, 0u64); nl_all * ne * 3]);
        self.moe.moe_resident = true;
        self.experts_bytes = bytes;
        Ok(())
    }

    pub fn upload_head(&mut self, data: &[u8], n: usize, k: usize) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || data.len() < need || !k.is_multiple_of(4) {
            return Err(format!(
                "upload_head: 형상 계약 위반 n={n} k={k} bytes={} < {need}",
                data.len()
            ));
        }
        let mut q8 = vec![0i8; n * k];
        let mut scv = vec![0f32; n];
        let src = &data[..need];
        let nt = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(4)
            .min(16);
        let rows_per = n.div_ceil(nt).max(1);
        {
            let qchunks: Vec<_> = q8.chunks_mut(rows_per * k).collect();
            let schunks: Vec<_> = scv.chunks_mut(rows_per).collect();
            std::thread::scope(|sp| {
                let mut r0 = 0usize;
                for (qc, sc_c) in qchunks.into_iter().zip(schunks) {
                    let base = r0;
                    sp.spawn(move || {
                        let nrows = qc.len() / k;
                        for r in 0..nrows {
                            let row = &src[(base + r) * k * 2..(base + r + 1) * k * 2];
                            let mut amax = 0.0f32;
                            for i in 0..k {
                                let v = f32::from_bits(
                                    u32::from(u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]))
                                        << 16,
                                );
                                amax = amax.max(v.abs());
                            }
                            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                            sc_c[r] = s;
                            for i in 0..k {
                                let v = f32::from_bits(
                                    u32::from(u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]))
                                        << 16,
                                );
                                qc[r * k + i] = (v / s).round().clamp(-127.0, 127.0) as i8;
                            }
                        }
                    });
                    r0 += rows_per;
                }
            });
        }
        for p in [self.head_w, self.head_sc, self.head_out, self.argmax_out] {
            if p != 0 {
                self.cc.free(p)?;
            }
        }
        self.head_w = 0;
        self.head_sc = 0;
        self.head_out = 0;
        self.argmax_out = 0;
        self.head_n = 0;
        self.head_k = 0;
        let dw = self.cc.alloc(n * k)?;
        let qb = unsafe { std::slice::from_raw_parts(q8.as_ptr() as *const u8, n * k) };
        if let Err(e) = Self::h2d_chunked(&self.cc, dw, qb) {
            let _ = self.cc.free(dw);
            return Err(e);
        }
        let dsc = self.cc.alloc(n * 4)?;
        let sb = unsafe { std::slice::from_raw_parts(scv.as_ptr() as *const u8, n * 4) };
        if let Err(e) = Self::h2d_chunked(&self.cc, dsc, sb) {
            let _ = self.cc.free(dw);
            let _ = self.cc.free(dsc);
            return Err(e);
        }
        let dout = match self.cc.alloc(n * 4) {
            Ok(p) => match self.cc.alloc(4) {
                Ok(a) => {
                    self.argmax_out = a;
                    p
                }
                Err(e) => {
                    let _ = self.cc.free(p);
                    let _ = self.cc.free(dw);
                    let _ = self.cc.free(dsc);
                    return Err(e);
                }
            },
            Err(e) => {
                let _ = self.cc.free(dw);
                let _ = self.cc.free(dsc);
                return Err(e);
            }
        };
        self.head_w = dw;
        self.head_sc = dsc;
        self.head_out = dout;
        self.head_n = n;
        self.head_k = k;
        self.weights_bytes += (n * k + n * 4 + n * 4) as u64;
        Ok(())
    }

    pub fn upload_embed(
        &mut self,
        data: &[u8],
        n: usize,
        k: usize,
        bf16: bool,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || data.len() < need {
            return Err(format!(
                "upload_embed: 형상 계약 위반 n={n} k={k} bytes={} < {need}",
                data.len()
            ));
        }
        if self.emb_dev != 0 {
            self.cc.free(self.emb_dev)?;
        }
        self.emb_dev = 0;
        self.emb_n = 0;
        self.emb_k = 0;
        let dw = self.cc.alloc(need)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, dw, &data[..need]) {
            let _ = self.cc.free(dw);
            return Err(e);
        }
        self.emb_dev = dw;
        self.emb_n = n;
        self.emb_k = k;
        self.emb_bf16 = bf16;
        self.weights_bytes += need as u64;
        Ok(())
    }

    pub fn embed_ready(&self) -> bool {
        self.emb_dev != 0 && self.emb_k == self.hidden
    }

    pub(super) fn embed_gather_launch(
        &self,
        toks: CUdeviceptr,
        out: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_embed_gather")?;
        let (mut p_e, mut p_tk, mut p_o) = (self.emb_dev, toks, out);
        let (mut p_t, mut p_k, mut p_n, mut p_b) = (
            t as i32,
            self.emb_k as i32,
            self.emb_n as i32,
            self.emb_bf16 as i32,
        );
        let elems = t * self.emb_k;
        self.cc.launch(
            f,
            elems.div_ceil(256) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l7(
                &mut p_e, &mut p_tk, &mut p_o, &mut p_t, &mut p_k, &mut p_n, &mut p_b,
            ),
        )
    }
}

const STREAM_CHUNK: usize = 8 << 20;

impl W4a16Dec {
    pub fn upload_lin_g_stream(
        &mut self,
        name: &str,
        n: usize,
        k: usize,
        group: usize,
        scale_bf16: bool,
        q_len: usize,
        s_len: usize,
        fill: &mut dyn FnMut(usize, u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        crate::rawcuda::gptq4::kernel_sym(false, group, scale_bf16)?;
        if q_len != n * (k / 2) || s_len != n * (k / group) * 2 {
            return Err(format!(
                "upload_lin_g_stream {name}: q={q_len} s={s_len} != n{n} k{k} g{group} 계약"
            ));
        }
        let dq = self.cc.alloc(q_len)?;
        let ds = match self.cc.alloc(s_len) {
            Ok(p) => p,
            Err(e) => {
                let _ = self.cc.free(dq);
                return Err(e);
            }
        };
        let r = (|| -> Result<(), String> {
            let mut ring = super::mem::PinnedRing::new(&self.cc, STREAM_CHUNK)?;
            ring.send(dq, q_len, &mut |off, buf| fill(0, off, buf))?;
            ring.send(ds, s_len, &mut |off, buf| fill(1, off, buf))?;
            ring.sync()
        })();
        if let Err(e) = r {
            let _ = self.cc.free(dq);
            let _ = self.cc.free(ds);
            return Err(e);
        }
        self.weights_bytes += (q_len + s_len) as u64;
        if let Some((oq, os, _, _)) = self.lins.insert(name.to_string(), (dq, ds, n, k)) {
            let _ = self.cc.free(oq);
            let _ = self.cc.free(os);
        }
        if group == 128 && !scale_bf16 {
            self.lin_grp.remove(name);
        } else {
            self.lin_grp.insert(name.to_string(), (group, scale_bf16));
        }
        if (group == 128 && !scale_bf16) || (group == 32 && scale_bf16) {
            const RCHUNK: usize = 8 << 20;
            let mut qbuf = vec![0u8; q_len];
            let mut ok = true;
            for off in (0..q_len).step_by(RCHUNK) {
                let end = (off + RCHUNK).min(q_len);
                if let Err(e) = fill(0, off as u64, &mut qbuf[off..end]) {
                    eprintln!("[marlin] {name}: q 재료화 실패({e}) — GEMV 폴백");
                    ok = false;
                    break;
                }
            }
            let mut sbuf = vec![0u8; s_len];
            let mut ok_s = true;
            for off in (0..s_len).step_by(RCHUNK) {
                let end = (off + RCHUNK).min(s_len);
                if let Err(e) = fill(1, off as u64, &mut sbuf[off..end]) {
                    eprintln!("[marlin] {name}: s 재료화 실패({e}) — GEMV 폴백");
                    ok_s = false;
                    break;
                }
            }
            if ok && ok_s {
                let _ = self.upload_lin_marlin(name, &qbuf, &sbuf, n, k, group, scale_bf16);
            }
        }
        Ok(())
    }

    pub fn upload_plain_stream(
        &mut self,
        name: &str,
        n: usize,
        k: usize,
        len: usize,
        fill: &mut dyn FnMut(u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || len < need {
            return Err(format!(
                "upload_plain_stream({name}): 형상 계약 위반 n={n} k={k} bytes={len} < {need}"
            ));
        }
        let dw = self.cc.alloc(need)?;
        let r = (|| -> Result<(), String> {
            let mut ring = super::mem::PinnedRing::new(&self.cc, STREAM_CHUNK)?;
            ring.send(dw, need, fill)?;
            ring.sync()
        })();
        if let Err(e) = r {
            let _ = self.cc.free(dw);
            return Err(e);
        }
        self.weights_bytes += need as u64;
        if let Some((old, _, _)) = self.plains.insert(name.to_string(), (dw, n, k)) {
            let _ = self.cc.free(old);
        }
        Ok(())
    }

    pub fn upload_embed_stream(
        &mut self,
        n: usize,
        k: usize,
        bf16: bool,
        len: usize,
        fill: &mut dyn FnMut(u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || len < need {
            return Err(format!(
                "upload_embed_stream: 형상 계약 위반 n={n} k={k} bytes={len} < {need}"
            ));
        }
        if self.emb_dev != 0 {
            self.cc.free(self.emb_dev)?;
        }
        self.emb_dev = 0;
        self.emb_n = 0;
        self.emb_k = 0;
        let dw = self.cc.alloc(need)?;
        let r = (|| -> Result<(), String> {
            let mut ring = super::mem::PinnedRing::new(&self.cc, STREAM_CHUNK)?;
            ring.send(dw, need, fill)?;
            ring.sync()
        })();
        if let Err(e) = r {
            let _ = self.cc.free(dw);
            return Err(e);
        }
        self.emb_dev = dw;
        self.emb_n = n;
        self.emb_k = k;
        self.emb_bf16 = bf16;
        self.weights_bytes += need as u64;
        Ok(())
    }

    pub fn upload_head_stream(
        &mut self,
        n: usize,
        k: usize,
        len: usize,
        fill: &mut dyn FnMut(u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || len < need || !k.is_multiple_of(4) {
            return Err(format!(
                "upload_head_stream: 형상 계약 위반 n={n} k={k} bytes={len} < {need}"
            ));
        }
        for p in [self.head_w, self.head_sc, self.head_out, self.argmax_out] {
            if p != 0 {
                self.cc.free(p)?;
            }
        }
        self.head_w = 0;
        self.head_sc = 0;
        self.head_out = 0;
        self.argmax_out = 0;
        self.head_n = 0;
        self.head_k = 0;
        let dw = self.cc.alloc(n * k)?;
        let dsc = match self.cc.alloc(n * 4) {
            Ok(p) => p,
            Err(e) => {
                let _ = self.cc.free(dw);
                return Err(e);
            }
        };
        let dout = match self.cc.alloc(n * 4) {
            Ok(p) => match self.cc.alloc(4) {
                Ok(a) => {
                    self.argmax_out = a;
                    p
                }
                Err(e) => {
                    let _ = self.cc.free(p);
                    let _ = self.cc.free(dw);
                    let _ = self.cc.free(dsc);
                    return Err(e);
                }
            },
            Err(e) => {
                let _ = self.cc.free(dw);
                let _ = self.cc.free(dsc);
                return Err(e);
            }
        };
        let nt = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(4)
            .min(16);
        let rows_blk = ((2 << 20) / (k * 2).max(1)).max(1);
        let mut raw = vec![0u8; rows_blk * k * 2];
        let mut q8 = vec![0i8; rows_blk * k];
        let mut scv = vec![0f32; rows_blk];
        let mut r0 = 0usize;
        while r0 < n {
            let rows = rows_blk.min(n - r0);
            let blen = rows * k * 2;
            fill((r0 * k * 2) as u64, &mut raw[..blen])?;
            {
                let rows_per = rows.div_ceil(nt).max(1);
                let qchunks: Vec<_> = q8[..rows * k].chunks_mut(rows_per * k).collect();
                let schunks: Vec<_> = scv[..rows].chunks_mut(rows_per).collect();
                std::thread::scope(|sp| {
                    let mut rr = 0usize;
                    for (qc, sc_c) in qchunks.into_iter().zip(schunks) {
                        let base = rr;
                        let raw_ref = &raw[..blen];
                        sp.spawn(move || {
                            let nrows = qc.len() / k;
                            for r in 0..nrows {
                                let row = &raw_ref[(base + r) * k * 2..(base + r + 1) * k * 2];
                                let mut amax = 0.0f32;
                                for i in 0..k {
                                    let v = f32::from_bits(
                                        u32::from(u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]))
                                            << 16,
                                    );
                                    amax = amax.max(v.abs());
                                }
                                let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                                sc_c[r] = s;
                                for i in 0..k {
                                    let v = f32::from_bits(
                                        u32::from(u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]))
                                            << 16,
                                    );
                                    qc[r * k + i] = (v / s).round().clamp(-127.0, 127.0) as i8;
                                }
                            }
                        });
                        rr += rows_per;
                    }
                });
            }
            let qb = unsafe { std::slice::from_raw_parts(q8.as_ptr() as *const u8, rows * k) };
            if let Err(e) = Self::h2d_chunked(&self.cc, dw + (r0 * k) as u64, qb) {
                let _ = self.cc.free(dw);
                let _ = self.cc.free(dsc);
                let _ = self.cc.free(dout);
                return Err(e);
            }
            let sb = unsafe { std::slice::from_raw_parts(scv.as_ptr() as *const u8, rows * 4) };
            if let Err(e) = Self::h2d_chunked(&self.cc, dsc + (r0 * 4) as u64, sb) {
                let _ = self.cc.free(dw);
                let _ = self.cc.free(dsc);
                let _ = self.cc.free(dout);
                return Err(e);
            }
            r0 += rows;
            llm170_diag::watchdog::bump();
        }
        self.head_w = dw;
        self.head_sc = dsc;
        self.head_out = dout;
        self.head_n = n;
        self.head_k = k;
        self.weights_bytes += (n * k + n * 4 + n * 4) as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::W4a16Dec;

    #[test]
    fn kv_block_addr_tiers() {
        let (bv, bh, stride) = (0x1_0000u64, 0x2_0000u64, 0x100u64);
        assert_eq!(W4a16Dec::kv_block_addr(bv, bh, 0, stride, 10), 0x1_0000);
        assert_eq!(
            W4a16Dec::kv_block_addr(bv, bh, 9, stride, 10),
            0x1_0000 + 9 * 0x100
        );
        assert_eq!(W4a16Dec::kv_block_addr(bv, bh, 10, stride, 10), 0x2_0000);
        assert_eq!(
            W4a16Dec::kv_block_addr(bv, bh, 12, stride, 10),
            0x2_0000 + 2 * 0x100
        );
        assert_eq!(
            W4a16Dec::kv_block_addr(bv, bh, 3, stride, 4),
            0x1_0000 + 3 * 0x100
        );
    }

    #[test]
    fn marlin_scale_dtype_branch() {
        assert_eq!(W4a16Dec::scale_bits_to_f32(0x3800, false), 0.5);
        assert_eq!(W4a16Dec::scale_bits_to_f32(0x3F00, true), 0.5);
        assert!((W4a16Dec::scale_bits_to_f32(0x3800, true) - 0.5).abs() > 1e-3);
        assert!((W4a16Dec::scale_bits_to_f32(0x3F00, false) - 0.5).abs() > 1e-3);
    }
}
