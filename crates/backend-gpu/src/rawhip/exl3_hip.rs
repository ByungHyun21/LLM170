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
    pub pos: u32,
    n_gdn: usize,
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
    dbx: *mut u8,
    pub dbg_layers: bool,
    pub dbg_hcurve: bool,
    pub hcurve: Vec<(usize, Vec<f32>)>,
    dbxn: *mut u8,
    dbab: *mut u8,
    dbzero: *mut u8,
    dah16: *mut u8,
    dsb2: *mut u8,
    dsb3: *mut u8,
    dbat: *mut u8,
    dpos: *mut u8,
    pstage: *mut u8,
    pgout: *mut u8,
    pgh: *mut u8,
    pgall: *mut u8,
    gexec: Option<(usize, crate::rawhip::ctx::hipgraph::GraphExec)>,
    dmtpk: *mut u8,
    dmtpv: *mut u8,
    dmtpp: *mut u8,
    dmtpnw: *mut u8,
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

impl HipLin {
    fn clone_shallow(&self) -> HipLin {
        HipLin {
            k: self.k,
            n: self.n,
            krate: self.krate,
            suh: self.suh,
            tre: self.tre,
            svh: self.svh,
        }
    }
}

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
        // 배치(prefill/검증) 버퍼 — tmax=64행 상한(plans/121 hip prefill)
        let dbx = hc.alloc(64 * hidden * 4)?;
        let dbxn = hc.alloc(64 * hidden * 4)?;
        let dbab = hc.alloc(64 * hidden * 4)?;
        let dbzero = hc.alloc(64 * hidden * 4)?;
        hc.h2d(dbzero, &vec![0u8; 64 * hidden * 4])?;
        let dah16 = hc.alloc(64 * 34816)?;
        let dsb2 = hc.alloc(64 * 17408 * 4)?; // 층 선형 n 상한(lm_head는 dsb)
        let dsb3 = hc.alloc(64 * 17408 * 4)?;
        // MTP 자체 KV(메인 16개 어텐션층과 분리 — prep의 layer 인덱스 0으로 사용)
        let dbat = hc.alloc(8 * 8 * 17408 * 4)?;
        let dpos = hc.alloc(4)?;
        hc.h2d(dpos, &0u32.to_le_bytes())?;
        // 캡처 호환 스테이징 — 핀(고정) 호스트 버퍼(페이지 가능 memcpy는 캡처 무효화).
        // SAFETY: hipHostMalloc — 핀(고정) 호스트 버퍼(캡처 호환 memcpy).
        let pin = |bytes: usize| -> Result<*mut u8, String> {
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            let r = unsafe { crate::rawhip::ctx::hipgraph::hipHostMalloc(&mut p, bytes, 0) };
            if r != 0 {
                return Err(format!("hipHostMalloc {r}({bytes}B)"));
            }
            Ok(p as *mut u8)
        };
        let pstage = pin(64 * hidden * 4)?;
        let pgout = pin(64 * 248320 * 4)?;
        let pgh = pin(hidden * 4)?;
        let pgall = pin(64 * hidden * 4)?; // 전 행 pre-norm h(MTP 훅 일관성) // kseg 부분합 [T≤8][kseg≤8][n≤17408]
        let dmtpk = hc.alloc(1024 * 1024 * 4)?;
        let dmtpv = hc.alloc(1024 * 1024 * 4)?;
        let dmtpp = hc.alloc(4)?;
        // 행 간격 5120 고정 — qn/kn(256원소)은 5120 패딩(행 포인터 산술 계약).
        let mut nwflat: Vec<f32> = Vec::with_capacity(7 * hidden);
        for (i, v) in mtp_norms.iter().enumerate() {
            nwflat.extend_from_slice(v);
            if i >= 5 {
                nwflat.resize((i + 1) * hidden, 0.0);
            }
        }
        let dmtpnw = hc.alloc(nwflat.len() * 4)?;
        hc.h2d(dmtpnw, unsafe {
            std::slice::from_raw_parts(nwflat.as_ptr() as *const u8, nwflat.len() * 4)
        })?;
        drop(tr);
        let dargmax = hc.alloc(4)?;

        let tmax = 64usize;
        let dx = hc.alloc(hidden * 4)?;
        let dxn = hc.alloc(hidden * 4)?;
        let dab = hc.alloc(hidden * 4)?;
        let dzero = hc.alloc(hidden * 4)?;
        hc.h2d(dzero, &vec![0u8; hidden * 4])?;
        let dah = hc.alloc(17408 * 2)?;
        let dsb = hc.alloc(tmax * 248320 * 4)?; // [조사 2026-10-04] 16행 하드코딩 잔존 — tmax=64 전제 위반(청크>16에서 lm_head mma OOB 폴트)
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
            n_gdn,
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
            dbx,
            dbg_layers: false,
            dbg_hcurve: false,
            hcurve: Vec::new(),
            dbxn,
            dbab,
            dbzero,
            dah16,
            dsb2,
            dsb3,
            dbat,
            dpos,
            pstage,
            pgout,
            pgh,
            pgall,
            gexec: None,
            dmtpk,
            dmtpv,
            dmtpp,
            dmtpnw,
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

    /// 제자리 상태 리셋 — 링/스캔 상태 0화 + pos 0. KV는 pos 의미론으로
    /// 도달 시 자연 갱신(재할당 없음 — plans/123 III-2 관례).
    pub fn reset_state(&mut self) -> Result<(), String> {
        let zeros_ring = vec![0u8; self.n_gdn * 3 * 10240 * 4];
        self.hc.h2d(self.dring, &zeros_ring)?;
        let zeros_st = vec![0u8; self.n_gdn * 48 * 16384 * 4];
        self.hc.h2d(self.dgst, &zeros_st)?;
        self.pos = 0;
        self.hc.h2d(self.dpos, &0u32.to_le_bytes())?;
        self.hc.h2d(self.dpp, &0u32.to_le_bytes())?;
        self.hc.sync()
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

    /// 임베딩 행 호스트 판독(배치 준비용).
    pub fn embed_row_host(&mut self, tok: u32) -> Vec<f32> {
        let mut rb = vec![0u8; self.hidden * 4];
        // SAFETY: dembed 내 행 오프셋(어휘·hidden 경계 내).
        let p = unsafe { self.dembed.add(tok as usize * self.hidden * 4) };
        let _ = self.hc.d2h(&mut rb, p);
        let _ = self.hc.sync();
        // SAFETY: d2h 완료 후 재해석.
        unsafe { std::slice::from_raw_parts(rb.as_ptr() as *const f32, self.hidden).to_vec() }
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
        let mut an = 248320i32;
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
        if self.dbg_layers {
            let d8 = self.dump8_dev(self.dsb);
            eprintln!("  [glg] lg={d8:?}");
        }
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

    /// 검증 덤프: 버퍼 선두 8 f32(동기 포함 — 디버그 전용).
    fn dump8_dev(&mut self, ptr: *mut u8) -> Vec<f32> {
        let mut rb = vec![0u8; 32];
        let _ = self.hc.d2h(&mut rb, ptr);
        let _ = self.hc.sync();
        // SAFETY: d2h 완료 후 재해석.
        unsafe { std::slice::from_raw_parts(rb.as_ptr() as *const f32, 8).to_vec() }
    }

    /// 순수+잔차 노름(norm_resid_p 직접 포인터) — x += ab, xn = norm(x)·nw. 커널 인자 순서.
    fn norm_ptr(
        &mut self,
        x: *mut u8,
        nw: *mut u8,
        ab: *mut u8,
        xn: *mut u8,
        t_len: usize,
    ) -> Result<(), String> {
        let mut tl = t_len as i32;
        let (mut a0, mut a1, mut a2, mut a3) = (x, nw, ab, xn);
        self.hc.launch(
            "exl3_norm_resid_p",
            t_len as u32,
            1,
            1024,
            &mut [
                &mut a0 as *mut *mut u8 as *mut _,
                &mut a1 as *mut *mut u8 as *mut _,
                &mut a2 as *mut *mut u8 as *mut _,
                &mut a3 as *mut *mut u8 as *mut _,
                &mut tl as *mut i32 as *mut _,
            ],
        )?;
        Ok(())
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

    /// 배치 forward(프리필·MTP 검증 공용) — rows: [T][hidden] 임베딩 행.
    /// GDN·FFN·선형은 gemm2 배치, 어텐션은 행별 prep/fwd3s 루프(소형-T 전용).
    /// 상태: dring/dgst/dkc/dvc는 pos..pos+T-1 순차 기록(디코드와 동일 규약).
    pub fn forward_batch(
        &mut self,
        rows: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        let t = rows.len();
        if t == 0 || t > 64 {
            return Err(format!("forward_batch: T={t} 범위 외(1..64)"));
        }
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        // SAFETY: pstage 64×hidden 상한 내 — 핀 쓰기.
        unsafe {
            std::ptr::copy_nonoverlapping(flat.as_ptr() as *const u8, self.pstage, flat.len() * 4);
        }
        self.batch_core(t)?;
        self.hc.sync()?; // 판독 배리어 — 캡처 코어는 비동기, g_out 확정 대기
        self.pos += t as u32;
        // SAFETY: g_out 재해석.
        let n = self.llh_n();
        let out: Vec<Vec<f32>> = (0..t)
            .map(|r| {
                // SAFETY: pgout 행 오프셋.
                let b = unsafe {
                    std::slice::from_raw_parts(self.pgout.add(r * n * 4) as *const f32, n)
                };
                b.to_vec()
            })
            .collect();
        // SAFETY: pgh 재해석.
        let last_h =
            unsafe { std::slice::from_raw_parts(self.pgh as *const f32, self.hidden).to_vec() };
        Ok((out, last_h))
    }

    /// 그래프 코어 — 모든 입출력이 고정 포인터(stage_rows/g_out/g_h/dpos).
    /// 캡처·재생·비캡처 공용(캡처 호환: 내부 sync/h2d-from-stack 없음).
    fn batch_core(&mut self, t: usize) -> Result<(), String> {
        let n_layers = self.loaded_layers.min(self.n_layers);
        // 캡처 호환 업로드 — 원시 핀 h2d(h2d는 내부 sync 포함 — 캡처 무효화).
        self.hc
            .h2d_nosync(self.dbx, self.pstage, t * self.hidden * 4)?;
        self.hc.d2d(self.dpp, self.dpos, 4)?;
        let mut ab = self.dbzero;
        for il in 0..n_layers {
            let lp = format!("model.language_model.layers.{il}");
            let gdn_il = (0..il).filter(|i| i % 4 != 3).count();
            self.norm_p(2 * il, ab, t)?;
            // [C 계기 2026-10-04] hstage — 순차 경로의 [hipl] L{il} xn 덤프와
            // 배치 경로의 값을 직접 대조해 첫 발산 층 경계를 확정한다.
            if llm170_diag::dump::opts().key("hstage") && il <= 1 {
                let mut xnb = vec![0u8; self.hidden * 4];
                let _ = self.hc.d2h(&mut xnb, self.dbxn);
                let _ = self.hc.sync();
                // SAFETY: d2h 완료 후 재해석 — 행0(5120원소).
                let xnf: &[f32] =
                    unsafe { std::slice::from_raw_parts(xnb.as_ptr() as *const f32, self.hidden) };
                let r = (xnf.iter().map(|v| v * v).sum::<f32>() / self.hidden as f32).sqrt();
                let mut xb2 = vec![0u8; self.hidden * 4];
                let _ = self.hc.d2h(&mut xb2, self.dbx);
                let _ = self.hc.sync();
                // SAFETY: d2h 완료 후 재해석 — 행0 잔차.
                let xf2: &[f32] =
                    unsafe { std::slice::from_raw_parts(xb2.as_ptr() as *const f32, self.hidden) };
                let r2 = (xf2.iter().map(|v| v * v).sum::<f32>() / self.hidden as f32).sqrt();
                eprintln!(
                    "  [hstb] L{il} 진입 x rms={r2:.5} x0={:.6} · xn rms={r:.5} xn0={:.6}",
                    xf2[0], xnf[0]
                );
            }
            if il % 4 == 3 {
                // 어텐션층 — q/k/v gemm2 후 행별 prep+fwd3s
                let lq = self.lin[&format!("{lp}.self_attn.q_proj")].clone_shallow();
                let lk = self.lin[&format!("{lp}.self_attn.k_proj")].clone_shallow();
                let lv = self.lin[&format!("{lp}.self_attn.v_proj")].clone_shallow();
                let lo = self.lin[&format!("{lp}.self_attn.o_proj")].clone_shallow();
                self.gemm2_batch(&lq, t, self.dsb)?;
                self.gemm2_batch(&lk, t, self.dsb2)?;
                self.gemm2_batch(&lv, t, self.dsb3)?;
                let mut ai = (il / 4) as i32;
                for r in 0..t {
                    let mut tl2 = 1i32;
                    let mut p0v = (self.pos + r as u32) as i32;
                    // SAFETY: 배치 버퍼 내 행 오프셋 — t≤64 경계 내.
                    let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) = unsafe {
                        (
                            self.dsb.add(r * 12288 * 4),
                            self.dsb2.add(r * 1024 * 4),
                            self.dsb3.add(r * 1024 * 4),
                            self.dqnw,
                            self.dknw,
                            self.dq2,
                            self.dkc,
                            self.dvc,
                            self.dpp,
                        )
                    };
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
                    // SAFETY: 배치 버퍼 내 행 오프셋 — t≤64 경계 내.
                    let (mut f0, mut f1, mut f2, mut f3, mut f4, mut f5) = unsafe {
                        (
                            self.dq2,
                            self.dkc,
                            self.dvc,
                            self.dsb.add(r * 12288 * 4),
                            self.dou.add(r * 6144 * 4),
                            self.dpp,
                        )
                    };
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
                    let mut pb0 = self.dpp;
                    self.hc.launch3(
                        "exl3_pos_bump",
                        1,
                        1,
                        1,
                        32,
                        &mut [&mut pb0 as *mut *mut u8 as *mut _],
                    )?;
                }
                // o_proj gemm2 — 입력 dou [T][6144]
                self.had16_batch(self.dou, 6144, t, lo.suh)?;
                self.gemm2_batch(&lo, t, self.dbab)?;
            } else {
                let lq = self.lin[&format!("{lp}.linear_attn.in_proj_qkv")].clone_shallow();
                let lz = self.lin[&format!("{lp}.linear_attn.in_proj_z")].clone_shallow();
                let lo = self.lin[&format!("{lp}.linear_attn.out_proj")].clone_shallow();
                self.had16_batch(self.dbxn, lq.k, t, lq.suh)?;
                self.gemm2_batch(&lq, t, self.dsb)?;
                self.had16_batch(self.dbxn, lz.k, t, lz.suh)?;
                self.gemm2_batch(&lz, t, self.dsb2)?;

                let mut tl = t as i32;
                let mut lay = gdn_il as i32;
                let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) =
                    (self.dsb, self.dcw, self.dring, self.dgq, self.dgk, self.dgv);
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
                    self.dgq, self.dgk, self.dgv, self.dbxn, self.dab_c, self.dal, self.ddt,
                    self.dq2, self.dk2, self.dv2, self.dbg,
                );
                self.hc.launch3(
                    "exl3_gdn_l2perm",
                    48,
                    t as u32,
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

                let (mut hk16, mut hv48, mut dd128) = (16i32, 48i32, 128i32);
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
                let (mut e0, mut e1, mut e2, mut e3) =
                    (self.dgo, self.dsb2, self.dnw_g, self.dgate);
                self.hc.launch3(
                    "exl3_gdn_gate",
                    48,
                    t as u32,
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

                self.had16_batch(self.dgate, lo.k, t, lo.suh)?;
                self.gemm2_batch(&lo, t, self.dbab)?;
            }
            self.norm_p(2 * il + 1, self.dbab, t)?;
            let lg = self.lin[&format!("{lp}.mlp.gate_proj")].clone_shallow();
            let lu = self.lin[&format!("{lp}.mlp.up_proj")].clone_shallow();
            let ld = self.lin[&format!("{lp}.mlp.down_proj")].clone_shallow();
            self.had16_batch(self.dbxn, lg.k, t, lg.suh)?;
            self.gemm2_batch(&lg, t, self.dsb)?;
            self.had16_batch(self.dbxn, lu.k, t, lu.suh)?;
            self.gemm2_batch(&lu, t, self.dsb2)?;
            let mut ewn = (t * lg.n) as i32;
            let (mut w0, mut w1, mut w2) = (self.dsb, self.dsb2, self.dew);
            self.hc.launch3(
                "exl3_ew",
                (t * lg.n).div_ceil(128) as u32,
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
            self.had16_batch(self.dew, ld.k, t, ld.suh)?;
            self.gemm2_batch(&ld, t, self.dbab)?;
            ab = self.dbab;
            if self.dbg_hcurve && [0usize, 1, 8, 32, 63].contains(&il) {
                let mut cb = vec![0u8; self.hidden * 4];
                let _ = self.hc.d2h(&mut cb, self.dbx);
                let _ = self.hc.sync();
                // SAFETY: d2h 완료 후 재해석.
                let cv = unsafe {
                    std::slice::from_raw_parts(cb.as_ptr() as *const f32, self.hidden).to_vec()
                };
                self.hcurve.push((il, cv));
            }
        }
        // last_h 캡처 — 최종 노름(마지막 FFN 합산) 전 잔차(vk 12/12·a1 0.625가
        // 측정된 규약 = h_seq와 동일 시점. 결함 11호 과교정 정정: '전'이되 dbx).
        {
            // SAFETY: 전 행 pre-norm dbx → pgall(캡처 호환, 핀).
            for r in 0..t {
                let prow = unsafe { self.dbx.add(r * self.hidden * 4) };
                let dst = unsafe { self.pgall.add(r * self.hidden * 4) };
                self.hc.d2h_pin_async(dst, prow, self.hidden * 4)?;
            }
            // SAFETY: 마지막 행 → g_h(호환 유지).
            let plast = unsafe { self.dbx.add((t - 1) * self.hidden * 4) };
            self.hc.d2h_pin_async(self.pgh, plast, self.hidden * 4)?;
        }
        // 최종 노름 + lm_head 행별 로짓
        self.norm_p(128, self.dbab, t)?;
        let llh = self.lin["lm_head"].clone_shallow();
        self.had16_batch(self.dbxn, llh.k, t, llh.suh)?;
        self.gemm2_batch(&llh, t, self.dsb)?;
        // dpos += t — 디바이스 pos 전진(그래프 재생 시 다음 라운드 위치).
        for _ in 0..t {
            let mut pb1 = self.dpos;
            self.hc.launch3(
                "exl3_pos_bump",
                1,
                1,
                1,
                32,
                &mut [&mut pb1 as *mut *mut u8 as *mut _],
            )?;
        }
        // 출력 d2h — 고정 호스트 버퍼(캡처 노드).
        for r in 0..t {
            // SAFETY: dsb 내 lm_head 행 오프셋.
            let rowp = unsafe { self.dsb.add(r * llh.n * 4) };
            // SAFETY: pgout 행 오프셋 — 원시 핀 d2h(캡처 호환).
            let dstp = unsafe { self.pgout.add(r * llh.n * 4) };
            self.hc.d2h_pin_async(dstp, rowp, llh.n * 4)?;
        }
        Ok(())
    }

    /// 배치 코어를 hipGraph로 캡처(런치 오버헤드 제거 — plans/121 hip 스케줄링).
    /// 워밍 1회 후 캡처: 모든 입출력이 고정 포인터라 재생은 현재 내용을 읽는다.
    pub fn capture_batch(&mut self, t: usize) -> Result<(), String> {
        use crate::rawhip::ctx::hipgraph as hg;
        // 워밍(커널 자원 초기화 완료 후 캡처)
        self.batch_core(t)?;
        self.hc.sync()?;
        unsafe {
            let st = hg::hipStreamBeginCapture(self.hc.stream as *mut _, 2);
            eprintln!("  [gcap] BeginCapture={st}");
            if st != 0 {
                return Err(format!("BeginCapture {st}"));
            }
            let core = self.batch_core(t);
            let mut graph: hg::Graph = std::ptr::null_mut();
            let en = hg::hipStreamEndCapture(self.hc.stream as *mut _, &mut graph);
            eprintln!("  [gcap] EndCapture={en} graph={graph:?} core={core:?}");
            core?;
            if en != 0 {
                return Err(format!("EndCapture {en}"));
            }
            let mut exec: hg::GraphExec = std::ptr::null_mut();
            let ie = hg::hipGraphInstantiate(&mut exec, graph, 0);
            hg::hipGraphDestroy(graph);
            if ie != 0 {
                return Err(format!("Instantiate {ie}"));
            }
            self.gexec = Some((t, exec));
        }
        Ok(())
    }

    /// 그래프 재생 — stage_rows를 채우고 launch. 반환 = (로짓 행들, last_h).
    pub fn replay_batch(&mut self, rows: &[Vec<f32>]) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        let (t, exec) = self.gexec.ok_or("그래프 미캡처 — capture_batch 먼저")?;
        if rows.len() != t {
            return Err(format!("캡처 T={t}와 불일치 rows={}", rows.len()));
        }
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        // SAFETY: pstage 핀 쓰기.
        unsafe {
            std::ptr::copy_nonoverlapping(flat.as_ptr() as *const u8, self.pstage, flat.len() * 4);
        }
        use crate::rawhip::ctx::hipgraph as hg;
        unsafe {
            let le = hg::hipGraphLaunch(exec, self.hc.stream as *mut _);
            if le != 0 {
                return Err(format!("GraphLaunch {le}"));
            }
        }
        self.hc.sync()?;
        self.pos += t as u32;
        let n = self.llh_n();
        // SAFETY: 재생 완료 후 g_out/g_h 재해석.
        let out: Vec<Vec<f32>> = (0..t)
            .map(|r| {
                // SAFETY: pgout 행 오프셋.
                let b = unsafe {
                    std::slice::from_raw_parts(self.pgout.add(r * n * 4) as *const f32, n)
                };
                b.to_vec()
            })
            .collect();
        let last_h =
            unsafe { std::slice::from_raw_parts(self.pgh as *const f32, self.hidden).to_vec() };
        Ok((out, last_h))
    }

    /// lm_head 열수(래퍼 디코딩용).
    fn llh_n(&self) -> usize {
        self.lin["lm_head"].n
    }

    /// forward_batch + MTP KV 적립 훅(vk 패턴): 각 행의 타깃 hidden으로
    /// mtp 어텐션 KV[pos]를 채운다 — 이후 mtp_draft_gpu는 전체 문맥을 본다.
    pub fn forward_batch_with_mtp(
        &mut self,
        rows: &[Vec<f32>],
        toks: &[u32],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        let t = rows.len();
        let out = self.forward_batch(rows)?;
        // 종료 시점: dbab에 최종 잔차(전 노름 전) t행 존재 — 행별 적립.
        let fc = self.lin["mtp.fc"].clone_shallow();
        let lq = self.lin["mtp.layers.0.self_attn.q_proj"].clone_shallow();
        let lk = self.lin["mtp.layers.0.self_attn.k_proj"].clone_shallow();
        let lv = self.lin["mtp.layers.0.self_attn.v_proj"].clone_shallow();
        // SAFETY: dmtpnw 내 행.
        let dmtpnw = self.dmtpnw;
        let nrow = move |i: usize| unsafe { dmtpnw.add(i * 5120 * 4) };
        for r in 0..t {
            let pos = (self.pos as usize - t + r) as u32;
            // h 행 — g_h 규약('마지막 FFN 합산 전')과 동일 클래스를 쓰려면
            // norm_p(128) 전 dbx가 필요하지만 훅은 최종 노름 후 실행 — 마지막 행만
            // g_h(전)에서, 나머지 행은 dbx(후)에서 읽는다(전 행은 마지막 라운드 토큰만
            // 드래프트 입력이 됨). r<t-1 행의 h는 다음 라운드 검증에서만 사용.
            let mut hb = vec![0u8; self.hidden * 4];
            if r + 1 == t {
                // SAFETY: pgh(핀 last_h) 판독 — 상위 forward_batch 완료 동기 후.
                let ghs =
                    unsafe { std::slice::from_raw_parts(self.pgh as *const u8, self.hidden * 4) };
                hb.copy_from_slice(ghs);
            } else {
                // SAFETY: dbx 행(후 시점 — 검증 전용).
                let hp = unsafe { self.dbx.add(r * self.hidden * 4) };
                self.hc.d2h(&mut hb, hp)?;
                self.hc.sync()?;
            }
            // SAFETY: d2h 완료 후 재해석.
            let h: &[f32] =
                unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const f32, self.hidden) };
            let e = self.embed_row_host(toks[r]);
            let eps = 1e-6f32;
            let rms = |x: &[f32], w: &[f32]| -> Vec<f32> {
                let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
                let sc = 1.0 / (ms + eps).sqrt();
                x.iter().zip(w).map(|(v, wv)| v * sc * wv).collect()
            };
            let e_n = rms(&e, &self.mtp_norms[0]);
            let h_n = rms(h, &self.mtp_norms[1]);
            let mut cat = Vec::with_capacity(2 * self.hidden);
            cat.extend_from_slice(&e_n);
            cat.extend_from_slice(&h_n);
            let f32b = |v: &[f32]| unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
            };
            self.hc.h2d(self.dbx, f32b(&cat))?;
            // fc → attn_norm → q/k/v → prep(KV 적립만)
            self.had16_batch(self.dbx, fc.k, 1, fc.suh)?;
            self.gemm2_batch(&fc, 1, self.dbab)?;
            self.norm_ptr(self.dbab, nrow(2), self.dbzero, self.dbxn, 1)?;
            self.had16_batch(self.dbxn, lq.k, 1, lq.suh)?;
            self.gemm2_batch(&lq, 1, self.dsb)?;
            self.had16_batch(self.dbxn, lk.k, 1, lk.suh)?;
            self.gemm2_batch(&lk, 1, self.dsb2)?;
            self.had16_batch(self.dbxn, lv.k, 1, lv.suh)?;
            self.gemm2_batch(&lv, 1, self.dsb3)?;
            self.hc.h2d(self.dmtpp, &pos.to_le_bytes())?;
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
                ],
            )?;
        }
        Ok(out)
    }

    /// 배치 노름(norm_resid_p) — dbx += ab, dbxn = norm(dbx)·w. nw는 행 포인터.
    fn norm_p(&mut self, w: usize, ab_in: *mut u8, t_len: usize) -> Result<(), String> {
        let mut tl = t_len as i32;
        // SAFETY: dnw 내 행 오프셋(w<129 경계 내).
        let nw_row = unsafe { self.dnw.add(w * 5120 * 4) };
        let (mut a0, mut a1, mut a2, mut a3) = (self.dbx, nw_row, ab_in, self.dbxn);
        self.hc.launch(
            "exl3_norm_resid_p",
            t_len as u32,
            1,
            1024,
            &mut [
                &mut a0 as *mut *mut u8 as *mut _,
                &mut a1 as *mut *mut u8 as *mut _,
                &mut a2 as *mut *mut u8 as *mut _,
                &mut a3 as *mut *mut u8 as *mut _,
                &mut tl as *mut i32 as *mut _,
            ],
        )?;
        Ok(())
    }

    /// 배치 had_in(f32 [T][k] → dah16 [T][k/2] f16쌍) — suh는 호출 선형에서 명시 전달.
    fn had16_batch(
        &mut self,
        src: *mut u8,
        k: usize,
        t_len: usize,
        suh: *mut u8,
    ) -> Result<(), String> {
        let mut kc = (k / 128) as i32;
        let mut ks = k as i32;
        let (mut p0, mut p1, mut p2) = (src, suh, self.dah16);
        self.hc.launch(
            "exl3_had_in",
            (k / 128) as u32,
            t_len as u32,
            128,
            &mut [
                &mut p0 as *mut *mut u8 as *mut _,
                &mut p1 as *mut *mut u8 as *mut _,
                &mut p2 as *mut *mut u8 as *mut _,
                &mut kc as *mut i32 as *mut _,
                &mut ks as *mut i32 as *mut _,
            ],
        )?;
        Ok(())
    }

    /// 배치 GEMM — 소형-T(≤8)·층선형(n≤17408)은 k-분할(kseg=8)로 점유 확보,
    /// 부분합 dbat [T][8][n] → had_out(nseg=8) 합산. 대형은 기존 단일 경로.
    fn gemm2_batch(&mut self, l: &HipLin, t_len: usize, out: *mut u8) -> Result<(), String> {
        if t_len <= 8 && l.n <= 17408 {
            let (mut kt, mut nt, mut kk, mut tt, mut ks) = (
                (l.k / 16) as i32,
                (l.n / 16) as i32,
                l.krate as i32,
                t_len as i32,
                8i32,
            );
            let (mut g0, mut g1, mut g2) = (self.dah16, l.tre, self.dbat);
            self.hc.launch3(
                "exl3_gemm2_kseg",
                (l.n / 64) as u32,
                8,
                1,
                128,
                &mut [
                    &mut g0 as *mut *mut u8 as *mut _,
                    &mut g1 as *mut *mut u8 as *mut _,
                    &mut g2 as *mut *mut u8 as *mut _,
                    &mut kt as *mut i32 as *mut _,
                    &mut nt as *mut i32 as *mut _,
                    &mut kk as *mut i32 as *mut _,
                    &mut tt as *mut i32 as *mut _,
                    &mut ks as *mut i32 as *mut _,
                ],
            )?;
            // had_out nseg=8 합산 → out(kseg8 최적 — 16은 미세 역행 측정)
            let (mut nch, mut nsg, mut nst) = ((l.n / 128) as i32, 8i32, l.n as i32);
            let (mut c0, mut c1, mut c2) = (self.dbat, l.svh, out);
            self.hc.launch3(
                "exl3_had_out",
                (l.n / 128) as u32,
                t_len as u32,
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
            return Ok(());
        }
        self.gemm2_batch_plain(l, t_len, out)
    }

    /// 기존 단일 gemm2(대형-T·lm_head) — [plans/127 B] exl3_gemm2_mma로 승격.
    /// 스칼라 1.5-2.8TF → mma 실측 9.6-10.5TF(T 64-512, 3.6-6.4倍·정합 ≤3.8e-4,
    /// 부분 T 16/21 포함 — exl3-hip-gemm T-sweep 원장). 출력 레이아웃[T][n]·
    /// had_out nseg=1 제자리 후처리 계약은 스칼라팧과 동일 — 교체 전용.
    fn gemm2_batch_plain(&mut self, l: &HipLin, t_len: usize, out: *mut u8) -> Result<(), String> {
        let (mut kt, mut nt, mut kk, mut tt) = (
            (l.k / 16) as i32,
            (l.n / 16) as i32,
            l.krate as i32,
            t_len as i32,
        );
        let (mut g0, mut g1, mut g2) = (self.dah16, l.tre, out);
        self.hc.launch3(
            "exl3_gemm2_mma",
            (l.n / 64) as u32,
            t_len.div_ceil(64) as u32,
            1,
            256,
            &mut [
                &mut g0 as *mut *mut u8 as *mut _,
                &mut g1 as *mut *mut u8 as *mut _,
                &mut g2 as *mut *mut u8 as *mut _,
                &mut kt as *mut i32 as *mut _,
                &mut nt as *mut i32 as *mut _,
                &mut kk as *mut i32 as *mut _,
                &mut tt as *mut i32 as *mut _,
            ],
        )?;
        self.hadout_batch(out, l.svh, l.n, t_len)
    }

    /// gemm2 출력 후처리 — H⁻¹⊙svh(nseg=1 제자리, 청크별 sm 스테이징이라 안전).
    fn hadout_batch(
        &mut self,
        buf: *mut u8,
        svh: *mut u8,
        n: usize,
        t_len: usize,
    ) -> Result<(), String> {
        let (mut nch, mut nsg, mut nst) = ((n / 128) as i32, 1i32, n as i32);
        let (mut c0, mut c1, mut c2) = (buf, svh, buf);
        self.hc.launch3(
            "exl3_had_out",
            (n / 128) as u32,
            t_len as u32,
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

    /// 배치 GEMM(dah16 → out [T][n]) — gemm2 커널.
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
            if self.dbg_hcurve && [0usize, 1, 8, 32, 63].contains(&il) {
                let mut cb = vec![0u8; self.hidden * 4];
                let _ = self.hc.d2h(&mut cb, self.dx);
                let _ = self.hc.sync();
                // SAFETY: d2h 완료 후 재해석.
                let cv = unsafe {
                    std::slice::from_raw_parts(cb.as_ptr() as *const f32, self.hidden).to_vec()
                };
                self.hcurve.push((il, cv));
            }

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
// 마커 cl3
// 마커 fb1
// 마커 fb2
// 마커 fb3
// 마커 fb5
// 마커 fb6
// 마커 fbfx
// 마커 ho1
// 마커 ho2
// 마커 dpp
// 마커 bld
// 마커 blf
// 마커 bd2
// 마커 gy2
// 마커 dcl
// 마커 mr2
// 마커 mr3
// 마커 dg2
// 마커 dg3
// 마커 dg4
// 마커 npf
// 마커 amf
// 마커 nwp
// 마커 d3p
// 마커 d5p
// 마커 npz
// 마커 kvh
// 마커 lhf
// 마커 ks2
// 마커 ho3
// 마커 gcap
// 마커 gcr
// 마커 gcr2
// 마커 gm2
// 마커 pin1
// 마커 fx25
// 마커 gdb
// 마커 gdb2
// 마커 dpf
// 마커 ks16
// 마커 rb1
// 마커 hpt
// 마커 hpf
// 마커 pga
// 마커 hcv
// 마커 hcf
// 마커 rs1
