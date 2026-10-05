use crate::rawhip::ctx::RawCtx as HipCtx;
use crate::rawvk::checks::TrellisResident;
use std::collections::HashMap;

mod batch;
mod forward;
mod gemm;
mod mtp;

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
    /// KV 위치 상한(plans/128 P0) — dkc/dvc·dmtpk/dmtpv 할당 크기와
    /// exl3_attn_prep/fwd3s kvcap 인자가 이 값으로 일치한다. 과거 1024
    /// 리터럴이 서빙 ctx를 1023으로 가둬 attn_prep 폴트→디코드 실패를 냈다.
    kvcap: i32,
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
    /// htrace(plans/128 P1 선행): 층 진입 잔차 플랫 목록 — 배치는 [il][t행],
    /// 순차는 [호출×64+il][1행]. LLM170_DUMP=htrace 게이트 — d2h+sync 포함해
    /// 그래프 캡처 경로와는 양립 불가(프로브 전용).
    pub htrace: Vec<Vec<Vec<f32>>>,
    /// atrace(plans/128 P1 선행): 첫 어텐션층(il==3)의 fwd3s 출력(dou)·o_proj 출력(dab)
    /// 행 플랫 목록 — 배치·순차 양경로. LLM170_DUMP=atrace 게이트.
    pub atrace_dou: Vec<Vec<f32>>,
    pub atrace_dab: Vec<Vec<f32>>,
    /// atrace 부속: fwd3s 입력 4종(qh·k·v·gate) 행 플랫 — 발산 입력 판별용.
    pub atrace_qh: Vec<Vec<f32>>,
    pub atrace_k: Vec<Vec<f32>>,
    pub atrace_v: Vec<Vec<f32>>,
    pub atrace_g: Vec<Vec<f32>>,
    pub atrace_xn: Vec<Vec<f32>>,
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
    /// kvcap: KV 위치 상한(plans/128 P0) — dkc/dvc 16층×kvcap×1024×4B×2,
    /// dmtpk/dmtpv kvcap×1024×4B씩. pos+t가 kvcap에 도달하면 Err로 우아하게
    /// 거절(폴트 아님). 최소 1024 권장(그 이하는 산술은 유효하나 경제성 없음).
    pub fn load(dir: &str, lim_layers: usize, kvcap: usize) -> Result<Self, String> {
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
        // KV RAM 예산 가드(plans/128 P0) — kvcap 32768이면 KV+MTP ≈4.5GiB.
        // 이 UMA 기기의 GTT 과다 할당은 시스템 동결 사고류(2026-10-04 원장)라
        // 할당 전 가용량 확인 후 명확한 에러로 거절(가드 상시화 정책 준용).
        let kv_bytes: u64 = (16 * 2 + 2) as u64 * kvcap as u64 * 1024 * 4;
        {
            let gib = |b: u64| format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64);
            let mut free: u64 = 0;
            if let Some((vf, _)) = crate::gpu_mem_free() {
                free = free.saturating_add(vf);
            }
            if let Ok(s) = std::fs::read_to_string("/proc/meminfo")
                && let Some(rest) = s.lines().find_map(|l| l.strip_prefix("MemAvailable:"))
                && let Ok(kb) = rest.trim().trim_end_matches(" kB").parse::<u64>()
            {
                // 회수 가능 캐시 이미 반영 — 여유 85%만 가용 산정(가드 정책 동일).
                free = free.saturating_add(kb * 1024 * 85 / 100);
            }
            // 마진 1GiB: 핀 버퍼·스테이징·런타임 변동.
            if kv_bytes + (1 << 30) > free {
                return Err(format!(
                    "KV 예산 부족: kvcap={kvcap}이 KV {}+마진을 요구하나 가용 {} — --ctx 하향",
                    gib(kv_bytes),
                    gib(free)
                ));
            }
        }
        // KV 캐시(plans/128 P0) — 16 어텐션층 × kvcap 위치 × 1024 f32.
        // kvcap 4096=512MiB · 8192=1GiB · 32768=4GiB(층당×2).
        // 제로 초기화 생략: 인과적 write-before-read — fwd3s는 [0, pos+t) 행만
        // 읽고 그 행은 항상 prep가 해당 pos 시점에 기록했다(reset 후에도
        // pos=0부터 재기록). 과거 64MiB 제로 h2d는 불필요 비용이었다.
        let dkc = hc.alloc(16 * kvcap * 1024 * 4)?;
        let dvc = hc.alloc(16 * kvcap * 1024 * 4)?;
        Self::h2d_chunked(&hc, dring, &vec![0u8; n_gdn * 3 * 10240 * 4])?;
        Self::h2d_chunked(&hc, dgst, &vec![0u8; n_gdn * 48 * 16384 * 4])?;
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
        // MTP 자체 KV — 메인 16층과 동일 kvcap 스케일(1층분, layer=0 인덱싱).
        let dmtpk = hc.alloc(kvcap * 1024 * 4)?;
        let dmtpv = hc.alloc(kvcap * 1024 * 4)?;
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
            mtp_kv_k: vec![0f32; kvcap * 4 * 256],
            mtp_kv_v: vec![0f32; kvcap * 4 * 256],
            mtp_kv_len: 0,
            dmtpin,
            dbx,
            dbg_layers: false,
            dbg_hcurve: false,
            htrace: Vec::new(),
            atrace_dou: Vec::new(),
            atrace_dab: Vec::new(),
            atrace_qh: Vec::new(),
            atrace_k: Vec::new(),
            atrace_v: Vec::new(),
            atrace_g: Vec::new(),
            atrace_xn: Vec::new(),
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
            kvcap: kvcap as i32,
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
}
