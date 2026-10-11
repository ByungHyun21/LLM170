//! [R1 2026-10-10] W4a16Dec 업로드·등록 — mod.rs 분해.
//! 선형/플레인/head 업로드, norm·GDN·attn·MoE 등록/테이블.

use super::*;

impl W4a16Dec {
    /// 선형 상주 업로드 — packed u32 [n][k/8] · scale f16 [n][k/128].
    pub fn upload_lin(
        &mut self,
        name: &str,
        q: &[u8],
        s: &[u8],
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        if q.len() != n * (k / 2) || s.len() != n * (k / 64) {
            return Err(format!(
                "upload_lin {name}: q={} s={} != n{n} k{k} 계약",
                q.len(),
                s.len()
            ));
        }
        // G2: alloc 전량 성공 → h2d 성공 시에만 상주 등록. 어느 단계든 실패하면
        // 성공분을 회수한다(부분 업로드 유실 금지 — 재시도가 처음부터).
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
        Ok(())
    }

    // ── norm ──

    /// 노름 가중 상주 등록 — nw [rows][hidden] f32(행 포인터 계약 w·hidden).
    pub fn set_norm_weights(&mut self, nw: &[f32], rows: usize) -> Result<(), String> {
        if self.hidden == 0 || !self.hidden.is_multiple_of(1024) || self.hidden > 8192 {
            return Err(format!(
                "norm: hidden={} — 1024 배수·8192 이하 계약",
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
        self.dnw = 0; // G1: 실패 시 재시도가 0을 해제하지 않게.
        self.norm_w_rows = 0;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let b = unsafe { std::slice::from_raw_parts(nw.as_ptr() as *const u8, nw.len() * 4) };
        let d = self.cc.alloc(nw.len() * 4)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, d, b) {
            let _ = self.cc.free(d); // G1: 부분 업로드 실패 시 유실 방지.
            return Err(e);
        }
        self.dnw = d;
        self.norm_w_rows = rows;
        Ok(())
    }

    // ── GDN ──

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
            self.dcw, self.dab_c, self.dalog, self.ddtb, self.dnwg, self.dring, self.dgst,
        ] {
            if q != 0 {
                self.cc.free(q)?;
            }
        }
        (self.dcw, self.dab_c, self.dalog, self.ddtb, self.dnwg) = (0, 0, 0, 0, 0);
        (self.dring, self.dgst) = (0, 0);
        let b =
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
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
        let dgst = self.cc.alloc(slots * n * hv * 128 * 128 * 4)?;
        Self::zero_dev(&self.cc, dgst, slots * n * hv * 128 * 128 * 4)?;
        (self.dcw, self.dab_c, self.dalog, self.ddtb, self.dnwg) = (dcw, dab, dal, ddt, dnw);
        self.dring = dring;
        self.dgst = dgst;
        self.gdn = Some(dims);
        Ok(())
    }

    pub fn set_attn(&mut self, dims: AttnDims, qnw: &[f32], knw: &[f32]) -> Result<(), String> {
        let n = dims.n_attn;
        if qnw.len() != n * dims.d || knw.len() != n * dims.d {
            return Err(format!("attn: qnw/knw != {n}x{}", dims.d));
        }
        let slots = self.n_slots;
        let kv_elems = slots * n * dims.cap * dims.kv_dim();
        // [KVQ 채택 2026-10-10] int8 KV 단일 경로(4× 절감) — 스케일은
        // 행×헤드 f32 1개(무시 가능).
        let kv_scales = slots * n * dims.cap * dims.kv_heads;
        for q in [
            self.dqnw_a,
            self.dknw_a,
            self.dkc,
            self.dvc,
            self.dksc,
            self.dvsc,
            self.dpp,
            self.dqg_a,
            self.dkin_a,
            self.dvin_a,
            self.dqh_a,
            self.doutv_a,
        ] {
            if q != 0 {
                self.cc.free(q)?;
            }
        }
        (self.dqnw_a, self.dknw_a, self.dkc, self.dvc, self.dpp) = (0, 0, 0, 0, 0);
        (self.dksc, self.dvsc) = (0, 0);
        (
            self.dqg_a,
            self.dkin_a,
            self.dvin_a,
            self.dqh_a,
            self.doutv_a,
        ) = (0, 0, 0, 0, 0);
        self.attn_t_cap = 0;
        let b =
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let dq = self.cc.alloc(qnw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dq, b(qnw))?;
        let dk = self.cc.alloc(knw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dk, b(knw))?;
        let (dkc, dvc) = {
            // int8 KV(1B/원소) + 스케일 2벌. 기록 attn_prep_q,
            // 판독 attn_fwd3s_part_q(병합은 f32 그대로).
            let kbytes = kv_elems;
            let kc = self.cc.alloc(kbytes)?;
            Self::zero_dev(&self.cc, kc, kbytes)?;
            let vc = self.cc.alloc(kbytes)?;
            Self::zero_dev(&self.cc, vc, kbytes)?;
            let ks = self.cc.alloc(kv_scales * 4)?;
            Self::zero_dev(&self.cc, ks, kv_scales * 4)?;
            let vs = self.cc.alloc(kv_scales * 4)?;
            Self::zero_dev(&self.cc, vs, kv_scales * 4)?;
            self.dksc = ks;
            self.dvsc = vs;
            (kc, vc)
        };
        let dpp = self.cc.alloc(slots * 4)?;
        Self::zero_dev(&self.cc, dpp, slots * 4)?;
        self.dqnw_a = dq;
        self.dknw_a = dk;
        self.dkc = dkc;
        self.dvc = dvc;
        self.dpp = dpp;
        self.attn = Some(dims);
        Ok(())
    }

    /// 플레인 bf16 [n][k] 상주 업로드 — [n][k] 원본(w4a16_gemv_bf16 계약).
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
        // [n][k] 원본 그대로 — w4a16_gemv_bf16(행=블록) 계약(전치 없음).
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

    /// 플레인 모드 전환(전문가 외 전부 bf16인 MoE 모델).
    pub fn set_plain_mode(&mut self, on: bool) {
        self.plain_weights = on;
    }

    /// MoE 구성 등록 — n_experts>0이면 체인은 플레인+MoE FFN 경로.
    pub fn set_moe(
        &mut self,
        n_experts: usize,
        top_k: usize,
        moe_ffn: usize,
        shared_ffn: usize,
        group: usize,
        scale_bf16: bool,
    ) {
        self.n_experts = n_experts;
        self.top_k = top_k;
        self.moe_ffn = moe_ffn;
        self.shared_ffn = shared_ffn;
        self.moe_group = group;
        self.moe_scale_bf16 = scale_bf16;
    }

    /// 전문가 슬라이스 테이블 등록 — (packed ptr/len, scale ptr/len) × (il,e,proj).
    /// 포인터는 서버 스토어 mmap 슬라이스 — 서버가 모델을 함께 보유하는 수명 계약.
    pub fn set_expert_table(&mut self, mut tab: Vec<(u64, u64, u64, u64)>) {
        // [P7-fix] 기존 스테이징이 등록돼 있으면 해제(버퍼 해제 전 필수).
        if self.host_stage_reg > 0 {
            // SAFETY: 등록 성공 상태(host_stage_reg>0)의 host_stage 선두 주소다.
            let _ = unsafe {
                self.cc
                    .host_unregister(self.host_stage.as_ptr() as *mut std::ffi::c_void)
            };
            self.host_stage_reg = 0;
        }
        // [C2 2026-10-09] 호스트 RAM 스테이징(사용자 결정) — 스트리밍 셋을
        // 로드 시 1회 호스트 RAM으로 복사한다(SSD mmap·페이지캐시 경로 의존
        // 제거). 탈출구 LLM170_MOE_HOST_STAGE=0, 할당 실패 시 mmap 유지.
        if llm170_diag::flag::ne0("LLM170_MOE_HOST_STAGE") {
            let total: u64 = tab.iter().map(|e| e.1 + e.3).sum();
            let mut buf: Vec<u8> = Vec::new();
            if total > 0 && usize::try_from(total).is_ok_and(|n| buf.try_reserve_exact(n).is_ok()) {
                let mut offs: Vec<(usize, usize)> = Vec::with_capacity(tab.len());
                for &(qp, ql, sp, sl) in &tab {
                    let qo = buf.len();
                    // SAFETY: tab 항목은 set_expert_table 수명 계약(mmap 슬라이스)
                    // — 호출 내 유효, len은 계약 검증 완료분.
                    buf.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(qp as *const u8, ql as usize)
                    });
                    let so = buf.len();
                    // SAFETY: 상동(scale 슬라이스).
                    buf.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(sp as *const u8, sl as usize)
                    });
                    offs.push((qo, so));
                }
                let base = buf.as_ptr() as u64;
                for (e, &(qo, so)) in tab.iter_mut().zip(offs.iter()) {
                    e.0 = base + qo as u64;
                    e.2 = base + so as u64;
                }
                eprintln!(
                    "[moe] 호스트 RAM 스테이징 {:.2}GiB — 업로드 mmap 의존 제거(C2)",
                    total as f64 / (1u64 << 30) as f64
                );
                self.host_stage = buf;
                // [P7-fix 2026-10-10] 핀드 DMA — 등록 가능한 선두 구간만
                // cuMemHostRegister(RLIMIT_MEMLOCK 한도). 등록 구간 소스 복사는
                // DMA 직행(실측 13.8GB/s), 나머지는 드라이버 스테이징 유지.
                // 예산 LLM170_MOE_PIN_MB(0=끔, 기본 4096).
                let budget_mb = llm170_diag::flag::val("LLM170_MOE_PIN_MB")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(4096);
                let budget = budget_mb << 20;
                if budget > 0 && !self.host_stage.is_empty() {
                    // memlock 한도가 작은 환경 대비 1/4씩 축소 재시도(하한 4MiB).
                    let mut bytes = self.host_stage.len().min(budget);
                    // [2026-10-12 결함수정] 등록 경계가 슬라이스(q/s)를 가르면 그
                    // 슬라이스의 async 복사가 실패한다(실측: 4GiB 경계가 L9
                    // 슬라이스를 관통 — cuMemcpyHtoDAsync INVALID_VALUE, 그
                    // 전문가가 선택된 프리필/디코드가 죽음). 등록 길이를 슬라이스
                    // 경계(선행 q·s 시작 오프셋)로 내린다.
                    if let Some(edge) = offs
                        .iter()
                        .flat_map(|&(qo, so)| [qo, so])
                        .filter(|&b| b <= bytes)
                        .max()
                    {
                        bytes = edge;
                    }
                    loop {
                        // SAFETY: host_stage는 구축 후 불변 Vec(boxed slice)라
                        // 해제 전까지 포인터 수명이 유지된다(해제는 상단 unregister).
                        match unsafe {
                            self.cc.host_register(
                                self.host_stage.as_ptr() as *mut std::ffi::c_void,
                                bytes,
                            )
                        } {
                            Ok(()) => {
                                self.host_stage_reg = bytes;
                                eprintln!(
                                    "# moe: 스테이징 핀드 등록 {:.2}/{}GiB — 구간 DMA 직행(P7-fix)",
                                    bytes as f64 / (1u64 << 30) as f64,
                                    self.host_stage.len() >> 30
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
        self.moe_tab = tab;
    }

    /// 전문가 상주 업로드 — 호스트 테이블(mmAP 슬라이스)을 VRAM 아레나
    /// (proj별 packed/scale 6개)로 올리고 테이블을 VRAM 포인터로 교체한다.
    /// 상주가 가능한 기기(170HX 64GB)에서 토큰당 PCIe 스트리밍을 제거한다.
    pub fn upload_experts_resident(
        &mut self,
        host_tab: &[(u64, u64, u64, u64)],
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        if self.n_experts == 0 || host_tab.len() != self.n_layers * self.n_experts * 3 {
            return Err("upload_experts_resident: 테이블 형상 위반".into());
        }
        let mut pk = [0usize; 3];
        let mut sk = [0usize; 3];
        for (i, e) in host_tab.iter().enumerate() {
            pk[i % 3] += e.1 as usize;
            sk[i % 3] += e.3 as usize;
        }
        let mut dpk = [0 as CUdeviceptr; 3];
        let mut dsk = [0 as CUdeviceptr; 3];
        let mut opk = [0usize; 3];
        let mut osk = [0usize; 3];
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
        let mut dev_tab = Vec::with_capacity(host_tab.len());
        // 1) 디바이스 테이블(아레나 오프셋) — 복사 없이 주소만 계산.
        for (i, e) in host_tab.iter().enumerate() {
            let p = i % 3;
            dev_tab.push((dpk[p] + opk[p] as u64, e.1, dsk[p] + osk[p] as u64, e.3));
            opk[p] += e.1 as usize;
            osk[p] += e.3 as usize;
        }
        let (mut opk2, mut osk2) = ([0usize; 3], [0usize; 3]);
        // 2) 전송 — h2d_chunked(페이지러블 비동기 + 드라이버 스테이징).
        // SAFETY: 항목은 서버 스토어 mmap 슬라이스(set_expert_table 수명 계약).
        let r = (|| -> Result<(), String> {
            for (i, e) in host_tab.iter().enumerate() {
                let p = i % 3;
                // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
                let qb = unsafe { std::slice::from_raw_parts(e.0 as *const u8, e.1 as usize) };
                // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
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
        // 디바이스 포인터 테이블 — (q,s) 쌍 평탄 배열(커널 간접 참조).
        let mut flat: Vec<u64> = Vec::with_capacity(dev_tab.len() * 2);
        for e in &dev_tab {
            flat.push(e.0);
            flat.push(e.2);
        }
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let fb = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 8) };
        if self.moe_dev_tab != 0 {
            let _ = self.cc.free(self.moe_dev_tab);
            self.moe_dev_tab = 0;
        }
        let dtab = self.cc.alloc(flat.len() * 8)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, dtab, fb) {
            let _ = self.cc.free(dtab);
            return Err(e);
        }
        self.moe_dev_tab = dtab;
        self.moe_tab = dev_tab;
        self.experts_bytes = host_tab.iter().map(|e| e.1 + e.3).sum();
        self.weights_bytes += self.experts_bytes;
        self.moe_resident = true;
        Ok(())
    }

    /// head(output.weight) 상주 업로드 — [I8head 2026-10-11] bf16 → int8
    /// (행별 amax/127 스케일) 양자화 후 업로드.
    /// 품질 판정(eval chunk4): en PPL +0.061%·argmax 99.139% / ko +0.0002%·
    /// 99.347% — KV int8 채택 밴드 · 27B 프로브 골든 유지. int4-g128 판은
    /// 기각(en +0.59%/92.7%). 대역: 2.54GB → 1.27GB+1MB(27B) — head 2.7ms
    /// (디코드의 14%)가 절반으로.
    pub fn upload_head(&mut self, data: &[u8], n: usize, k: usize) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || data.len() < need || !k.is_multiple_of(4) {
            return Err(format!(
                "upload_head: 형상 계약 위반 n={n} k={k} bytes={} < {need}",
                data.len()
            ));
        }
        // 호스트 양자화 — 행별 amax/127. 다중 스레드(1.27G원소 ~수백 ms).
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
        for p in [self.head_w, self.head_sc, self.head_out] {
            if p != 0 {
                self.cc.free(p)?;
            }
        }
        self.head_w = 0; // G1: 부분 상태(head_n=0)로 소비되지 않게 마지막에 대입.
        self.head_sc = 0;
        self.head_out = 0;
        self.head_n = 0;
        self.head_k = 0;
        // int8 본체 + f32 행스케일 업로드.
        let dw = self.cc.alloc(n * k)?;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수).
        let qb = unsafe { std::slice::from_raw_parts(q8.as_ptr() as *const u8, n * k) };
        if let Err(e) = Self::h2d_chunked(&self.cc, dw, qb) {
            let _ = self.cc.free(dw);
            return Err(e);
        }
        let dsc = self.cc.alloc(n * 4)?;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let sb = unsafe { std::slice::from_raw_parts(scv.as_ptr() as *const u8, n * 4) };
        if let Err(e) = Self::h2d_chunked(&self.cc, dsc, sb) {
            let _ = self.cc.free(dw);
            let _ = self.cc.free(dsc);
            return Err(e);
        }
        let dout = match self.cc.alloc(n * 4).and_then(|p| {
            self.cc.alloc(4).map(|a| {
                self.argmax_out = a;
                p
            })
        }) {
            Ok(p) => p,
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

    /// [S3-1 2026-10-11] token_embd 상주 업로드 — 프리필 임베딩 gather 원본.
    /// f16/bf16 원본 바이트 그대로(디코드는 커널 h2f/b2f = Rust 동일 함수 —
    /// embed_row와 값 비트동일).
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
        self.emb_dev = 0; // G1: 부분 상태로 소비되지 않게 마지막에 재대입.
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

    /// [S3-1] 임베딩 상주 여부 — 프리필 gather 경로 진입 조건.
    pub fn embed_ready(&self) -> bool {
        self.emb_dev != 0 && self.emb_k == self.hidden
    }

    /// [S3-1] 프리필(t≥2) — 토큰 id 배열 직접 진입. id h2d → gather(dres) →
    /// 체인. 값은 forward_prefill(행 경로)와 동일(골든 계약).
    pub fn forward_prefill_toks(
        &mut self,
        slot: usize,
        toks: &[u32],
        t: usize,
        head: bool,
    ) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if !self.embed_ready() {
            return Err("forward_prefill_toks: 임베딩 미등록".into());
        }
        if !(2..=CHAIN_TMAX).contains(&t) || toks.len() != t || slot >= self.n_slots {
            return Err(format!(
                "forward_prefill_toks: t={t} toks={} 계약 위반",
                toks.len()
            ));
        }
        if let Some(&bad) = toks.iter().find(|&&x| x as usize >= self.emb_n) {
            return Err(format!("forward_prefill_toks: 토큰 범위 위반 {bad}"));
        }
        self.ensure_chain_bufs()?;
        // SAFETY: toks 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let tb = unsafe { std::slice::from_raw_parts(toks.as_ptr() as *const u8, t * 4) };
        self.cc.h2d_async(self.dtg, tb)?;
        self.embed_gather_launch(self.dtg, self.dres, t)?;
        self.chain_t_sel(slot, None, t, head, false, false)
    }

    /// [S3-1] 임베딩 gather 런치 — toks[t] → out[t][k] f32.
    fn embed_gather_launch(
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
