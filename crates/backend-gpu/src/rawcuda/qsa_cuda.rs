//! Flash-Next QSA(인덩서 top-k 게이트드 GQA) 모듈층 — plans/124 FND(fn4-qsa),
//! 2026-10-05. 산술 원천 = crates/core/src/qwen4exp/stages/qsa.rs(CPU 황금
//! 계약) — 본 모듈은 qsa_select(패스 A 캐시 적립·블록 키 풀링·패스 B top-k)
//! ·qsa_sel_list(호스트 평탄화 — 코어와 동일 정수 논리)·어텐션+게이트를
//! assets/exl3_fn_qsa.cu 커널로 미러한다. 검증층(qsa_cuda_probe.rs)이 오라클과
//! 값 maxdiff·선택 리스트 완전일치로 판정한다(3층 분리, plans/124 §5).
//!
//! [API 경계 — core qsa_layer 대비] core qsa_layer(L396-515)은 mm_group
//! 5투영(q/k/v/iq/ik·L433-452)과 mm_batch wo(L492)를 포함하나 이들은 전
//! 스테이지 공용 matmul(REUSE 트랙 — EXL3 gemv/gemm2 krate 7 이슈·Q4 MMQ,
//! FNA 계약 지도 "QSA 투영" 항)이라 본 모듈은 **투영 출력을 입·출력 경계로**
//! 한다: qsa_select는 kk/vv/iq/ik를 받고(코어와 동일 서명), qsa_stage가
//! q행(qg — norm+rope 전 wq 출력 원문)을 받아 L508-517 norm·rope + 선택
//! 리스트 어텐션(cpu_attn_row 산술) + 게이트까지 수행해 어텐션 출력
//! [t][n_head·hd]를 반환한다(wo 투영 직전 — 코어 attn_all과 동일 지점).
//!
//! [상태 규약] KV/idx/블록키 캐시·pp는 전부 디바이스 상주. pos의 진실은
//! pp[0](결함 4호) — set_pos(h2d 초기화)·pos_bump(커널 증분)로만 전진하고
//! 커널은 pp[0]을 판독해 캐시 기록 위치·n_past를 산출한다. 호스트 pos
//! 미러(pos_h)는 버퍼 크기 산정·경계 검사 전용(산술 비관여). 블록 키
//! 캐시 길이(bk_blocks)도 호스트 미러 — 코어 idx_bk.len()/dim(L200-202)과
//! 동일 증분 규약(청크 끝까지의 완전 블록만).
//!
//! [k_prenormed] kk가 이미 norm+rope 적용분이면 재적용 금지(qsa.rs L168-171,
//! layers.rs:170, AGENTS.md 안티패턴) — llm170_fn_qsa_k_rows의 prenormed
//! 분기가 코어 L139-148 분기와 동일하게 그대로 적립한다.
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지 — scripts/cuda_probe_shim.rs 단독 컴파일.

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use crate::rawcuda::ffi::CUdeviceptr;
use crate::rawcuda::fn_support::FnDims;

/// 선택 위치 상한(커널 QSA_SEL_CAP과 동일 값 — np = sel_cnt·r + tail ≤
/// min(n_past, top_k+r−1) = 2051 실측 상계).
pub const QSA_SEL_CAP: usize = 2052;
/// 블록 수 상한(커널 QSA_BK_MAX와 동일 값 — n_blocks = n_past/r ≤ cap/r).
pub const QSA_BK_MAX: usize = 2052;
/// 커널 목록(assets/exl3_fn_qsa.cu — load_fatbin 심볼 사전 등록).
const KERNELS: &[&str] = &[
    "llm170_fn_qsa_k_rows",
    "llm170_fn_qsa_v_rows",
    "llm170_fn_qsa_ik_rows",
    "llm170_fn_qsa_iq_rows",
    "llm170_fn_qsa_pool",
    "llm170_fn_qsa_select",
    "llm170_fn_qsa_q_rows",
    "llm170_fn_qsa_attn",
    "llm170_fn_qsa_pos_bump",
];

/// QSA CUDA 모듈 — 단일 디바이스 컨텍스트 소유(단일 상주 원칙, FNA 계약).
pub struct QsaCuda {
    /// 디바이스 컨텍스트(커널 레지스트리 포함).
    pub cc: CudaCtx,
    /// 형상(Hparams4 미러 — FnDims::from_gguf 산출).
    pub dims: FnDims,
    /// 캐시 위치 상한.
    cap: usize,
    /// QSA층 수(compress[il]≠0).
    n_full: usize,
    /// il → full_idx(layers.rs L1491-1543 증분 카운터와 동일 규칙).
    full_of_il: Vec<usize>,
    /// KV 캐시 [n_full·cap][n_kv·hd] ×2 · 인덱서 k [n_full·cap][idx_dim] ·
    /// 블록 키 [n_full·bk_cap][idx_dim].
    d_kv_k: CUdeviceptr,
    d_kv_v: CUdeviceptr,
    d_idx_k: CUdeviceptr,
    d_idx_bk: CUdeviceptr,
    bk_cap: usize,
    /// 노름 가중 [n_full][hd]·[n_full][idx_dim] ×2.
    d_knw: CUdeviceptr,
    d_qnw: CUdeviceptr,
    d_iqw: CUdeviceptr,
    d_ikw: CUdeviceptr,
    /// pos 디바이스 진실(결함 4호) + 호스트 미러(버퍼 산정 전용).
    d_pp: CUdeviceptr,
    pos_h: u32,
    /// per-fi 블록 키 캐시 길이(블록 수 — 코어 idx_bk.len()/idx_dim 미러).
    bk_blocks: Vec<usize>,
    /// 스테이징 [t_cap] 버퍼들.
    t_cap: usize,
    d_kk: CUdeviceptr,
    d_vv: CUdeviceptr,
    d_iq: CUdeviceptr,
    d_ik: CUdeviceptr,
    d_iqnr: CUdeviceptr,
    d_qg: CUdeviceptr,
    d_qbuf: CUdeviceptr,
    d_attn: CUdeviceptr,
    d_sel_blk: CUdeviceptr,
    d_sel_cnt: CUdeviceptr,
    d_sel_idx: CUdeviceptr,
    d_sel_off: CUdeviceptr,
}

/// 검증층 음성대조 모드(원장 17호 — 계기 자체 검증. 프로덕션 경로 사용 금지).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QsaNeg {
    /// 정상(프로덕션).
    Off,
    /// top-k off-by-one: n_sel+1 선택(잘못된 top-k 재현).
    TopkOff1,
    /// 잘못된 풀링: 블록 키 원천 행 +1 시프트(풀링 결함 재현).
    PoolShift1,
}

impl QsaCuda {
    /// 컨텍스트 개방 + 캐시 제로 할당 + fatbin 로드. cap은 pos0+t_len 상한.
    pub fn new(dims: FnDims, cap: usize) -> Result<Self, String> {
        if dims.n_rot != 64 || dims.head_dim != 256 || dims.idx_dim != 128 || dims.idx_heads != 4 {
            return Err(format!(
                "qsa: n_rot={} head_dim={} idx_dim={} idx_heads={} — 64/256/128/4 고정 계약(커널 상한)",
                dims.n_rot, dims.head_dim, dims.idx_dim, dims.idx_heads
            ));
        }
        if dims.rope_base != 1e7 {
            return Err(format!(
                "qsa: rope_base={} — 커널 qsa_theta 트윈은 ln(1e7) 고정 계약",
                dims.rope_base
            ));
        }
        if dims.compress.iter().any(|&c| c != 0 && c != 4) {
            return Err("qsa: compress[il]은 0(GDN) 또는 4(QSA)만 지원".into());
        }
        let n_full = dims.compress.iter().filter(|&&c| c != 0).count();
        if n_full == 0 {
            return Err("qsa: QSA층(compress≠0) 없음".into());
        }
        if cap == 0 || cap % 4 != 0 || cap / 4 + 2 > QSA_BK_MAX {
            return Err(format!(
                "qsa: cap={cap} — 4배수·n_blocks 상한 QSA_BK_MAX={QSA_BK_MAX} 계약"
            ));
        }
        let r = dims.compress.iter().find(|&&c| c != 0).copied().unwrap() as usize;
        if dims.compress.iter().any(|&c| c != 0 && c as usize != r) {
            return Err("qsa: QSA층 compress가 서로 다름 — 단일 r 계약 위반".into());
        }
        let mut full_of_il = vec![usize::MAX; dims.n_layer];
        let mut acc = 0usize;
        for il in 0..dims.n_layer {
            if dims.compress[il] != 0 {
                full_of_il[il] = acc;
                acc += 1;
            }
        }
        let image = qsa_fatbin_bytes()?;
        let mut cc = CudaCtx::new()?;
        let _g = cc.guard()?;
        cc.load_fatbin("exl3_fn_qsa", &image, KERNELS)?;
        let hd = dims.head_dim;
        let (kv, idx, bk_cap) = (
            n_full * cap * dims.n_kv * hd,
            n_full * cap * dims.idx_dim,
            cap / r + 2,
        );
        let z = |n: usize| -> Result<CUdeviceptr, String> {
            let d = cc.alloc(n * 4)?;
            // 제로 기록(코어 SeqState4 제로 초기화와 동일 — 히스토리 없음 상태).
            Exl3CudaDecoder::h2d_chunked(&cc, d, &vec![0u8; n * 4])?;
            Ok(d)
        };
        let d_pp = cc.alloc(4)?;
        cc.h2d(d_pp, &0u32.to_le_bytes())?;
        Ok(QsaCuda {
            d_kv_k: z(kv)?,
            d_kv_v: z(kv)?,
            d_idx_k: z(idx)?,
            d_idx_bk: z(n_full * bk_cap * dims.idx_dim)?,
            bk_cap,
            d_knw: z(n_full * hd)?,
            d_qnw: z(n_full * hd)?,
            d_iqw: z(n_full * dims.idx_dim)?,
            d_ikw: z(n_full * dims.idx_dim)?,
            d_pp,
            pos_h: 0,
            bk_blocks: vec![0; n_full],
            t_cap: 0,
            d_kk: 0,
            d_vv: 0,
            d_iq: 0,
            d_ik: 0,
            d_iqnr: 0,
            d_qg: 0,
            d_qbuf: 0,
            d_attn: 0,
            d_sel_blk: 0,
            d_sel_cnt: 0,
            d_sel_idx: 0,
            d_sel_off: 0,
            cc,
            dims,
            cap,
            n_full,
            full_of_il,
        })
    }

    /// il의 QSA 슬롯(full_idx) — layers.rs full_idx 증분 규칙 미러.
    pub fn full_idx(&self, il: usize) -> Result<usize, String> {
        if il >= self.dims.n_layer {
            return Err(format!("qsa: il={il} >= n_layer={}", self.dims.n_layer));
        }
        match self.full_of_il[il] {
            fi if fi != usize::MAX => Ok(fi),
            _ => Err(format!("qsa: il={il}은 QSA층 아님(compress==0 — GDN)")),
        }
    }

    /// QSA층 노름 가중 등록(호출층 단위 — GGUF blk.{il}.* 4종). knw/qnw는
    /// [hd], iqw/ikw는 [idx_dim] f32.
    pub fn set_norms(
        &mut self,
        il: usize,
        knw: &[f32],
        qnw: &[f32],
        iqw: &[f32],
        ikw: &[f32],
    ) -> Result<(), String> {
        let fi = self.full_idx(il)?;
        let (hd, id) = (self.dims.head_dim, self.dims.idx_dim);
        if knw.len() != hd || qnw.len() != hd || iqw.len() != id || ikw.len() != id {
            return Err(format!(
                "qsa norms: {}/{}/{}/{} — hd={hd} idx_dim={id} 계약 위반",
                knw.len(),
                qnw.len(),
                iqw.len(),
                ikw.len()
            ));
        }
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치).
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let _g = self.cc.guard()?;
        self.cc.h2d(self.d_knw + (fi * hd) as u64 * 4, b(knw))?;
        self.cc.h2d(self.d_qnw + (fi * hd) as u64 * 4, b(qnw))?;
        self.cc.h2d(self.d_iqw + (fi * id) as u64 * 4, b(iqw))?;
        self.cc.h2d(self.d_ikw + (fi * id) as u64 * 4, b(ikw))
    }

    /// pp[0] = pos(h2d — 최초 초기화/리셋 전용. 전진은 pos_bump).
    pub fn set_pos(&mut self, pos: u32) -> Result<(), String> {
        let _g = self.cc.guard()?;
        self.cc.h2d(self.d_pp, &pos.to_le_bytes())?;
        self.pos_h = pos;
        Ok(())
    }

    /// pp[0] += dt(커널 증분 — 결함 16호: 캡처 그래프 내 h2d 불가 대체).
    pub fn pos_bump(&mut self, dt: u32) -> Result<(), String> {
        let f = self.cc.function("llm170_fn_qsa_pos_bump")?;
        let mut p0 = self.d_pp;
        let mut d = dt;
        let mut args: [*mut std::ffi::c_void; 2] =
            [(&mut p0) as *mut _ as *mut _, (&mut d) as *mut _ as *mut _];
        self.cc.launch(f, 1, 1, 32, &mut args)?;
        self.pos_h = self.pos_h.wrapping_add(dt);
        Ok(())
    }

    /// pp[0] 판독(진실 원천 — 결함 4호).
    pub fn device_pos(&self) -> Result<u32, String> {
        let mut b = [0u8; 4];
        let _g = self.cc.guard()?;
        self.cc.d2h(&mut b, self.d_pp)?;
        Ok(u32::from_le_bytes(b))
    }

    /// 스테이징 버퍼 보장(t 상한 확장 시에만 재할당).
    fn ensure_t(&mut self, t_len: usize) -> Result<(), String> {
        if t_len <= self.t_cap {
            return Ok(());
        }
        let _g = self.cc.guard()?;
        for q in [
            self.d_kk,
            self.d_vv,
            self.d_iq,
            self.d_ik,
            self.d_iqnr,
            self.d_qg,
            self.d_qbuf,
            self.d_attn,
            self.d_sel_blk,
            self.d_sel_cnt,
            self.d_sel_idx,
            self.d_sel_off,
        ] {
            if q != 0 {
                self.cc.free(q)?;
            }
        }
        let d = &self.dims;
        let stride =
            d.idx_top_k / d.compress.iter().find(|&&c| c != 0).copied().unwrap() as usize + 2;
        let alloc = |n: usize| self.cc.alloc(n * 4);
        self.d_kk = alloc(t_len * d.n_kv * d.head_dim)?;
        self.d_vv = alloc(t_len * d.n_kv * d.head_dim)?;
        self.d_iq = alloc(t_len * d.idx_heads * d.idx_dim)?;
        self.d_ik = alloc(t_len * d.idx_dim)?;
        self.d_iqnr = alloc(t_len * d.idx_heads * d.idx_dim)?;
        self.d_qg = alloc(t_len * d.n_head * 2 * d.head_dim)?;
        self.d_qbuf = alloc(t_len * d.n_head * d.head_dim)?;
        self.d_attn = alloc(t_len * d.n_head * d.head_dim)?;
        self.d_sel_blk = alloc(t_len * stride)?;
        self.d_sel_cnt = alloc(t_len)?;
        self.d_sel_idx = alloc(t_len * (d.idx_top_k + d.idx_top_k / 4 + 8))?;
        self.d_sel_off = alloc(t_len + 1)?;
        self.t_cap = t_len;
        Ok(())
    }

    /// sel_stride — qsa.rs L237: idx_top_k/r + 2.
    pub fn sel_stride(&self) -> usize {
        let r = self
            .dims
            .compress
            .iter()
            .find(|&&c| c != 0)
            .copied()
            .unwrap() as usize;
        self.dims.idx_top_k / r + 2
    }

    /// 패스 A·풀링·패스 B 발사 + 선택 목록 판독 — 코어 qsa_select(qsa.rs
    /// L99-321)의 디바이스 미러. kk/vv: [t][n_kv·hd], iq: [t][idx_heads·idx_dim],
    /// ik: [t][idx_dim]. neg는 검증층 음성대조 전용(프로덕션 Off).
    pub fn qsa_select(
        &mut self,
        il: usize,
        kk: &[Vec<f32>],
        vv: &[Vec<f32>],
        iq: &[Vec<f32>],
        ik: &[Vec<f32>],
        t_len: usize,
        k_prenormed: bool,
        neg: QsaNeg,
    ) -> Result<(Vec<u32>, Vec<u32>, usize), String> {
        let fi = self.full_idx(il)?;
        let (n_kv, hd, id, ih) = (
            self.dims.n_kv,
            self.dims.head_dim,
            self.dims.idx_dim,
            self.dims.idx_heads,
        );
        let r = self.dims.compress[il] as usize;
        let (n_rot, eps, top_k) = (self.dims.n_rot, self.dims.eps, self.dims.idx_top_k);
        let stride = self.sel_stride();
        if t_len == 0 {
            return Err("qsa_select: t_len=0".into());
        }
        if kk.len() < t_len || vv.len() < t_len || iq.len() < t_len || ik.len() < t_len {
            return Err(format!(
                "qsa_select: 입력 행 부족 kk/vv/iq/ik {}/{}/{}/{} < t={t_len}",
                kk.len(),
                vv.len(),
                iq.len(),
                ik.len()
            ));
        }
        let pos0 = self.device_pos()? as usize;
        if pos0 + t_len > self.cap {
            return Err(format!(
                "qsa_select: pos={pos0}+t={t_len} > cap={}(KV 캐시 상한)",
                self.cap
            ));
        }
        if (pos0 + t_len) / r + 2 > QSA_BK_MAX {
            return Err(format!(
                "qsa_select: n_blocks 상한 QSA_BK_MAX={QSA_BK_MAX} 초과"
            ));
        }
        self.ensure_t(t_len)?;
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치 — 업로드 경로).
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let _g = self.cc.guard()?;
        let flat = |rows: &[Vec<f32>], want: usize| -> Result<Vec<f32>, String> {
            let mut o = Vec::with_capacity(rows.len() * want);
            for (t, row) in rows.iter().enumerate() {
                if t >= t_len {
                    break;
                }
                if row.len() != want {
                    return Err(format!("qsa_select: 행 {t} 폭 {} != {want}", row.len()));
                }
                o.extend_from_slice(row);
            }
            Ok(o)
        };
        let kkf = flat(kk, n_kv * hd)?;
        let vvf = flat(vv, n_kv * hd)?;
        let iqf = flat(iq, ih * id)?;
        let ikf = flat(ik, id)?;
        self.cc.h2d(self.d_kk, b(&kkf))?;
        self.cc.h2d(self.d_vv, b(&vvf))?;
        self.cc.h2d(self.d_iq, b(&iqf))?;
        self.cc.h2d(self.d_ik, b(&ikf))?;
        // ── 패스 A(qsa.rs L167-190) ──
        let (mut a_fi, mut a_cap, mut a_nkv, mut a_hd, mut a_rot, mut a_pre) = (
            fi as i32,
            self.cap as i32,
            n_kv as i32,
            hd as i32,
            n_rot as i32,
            k_prenormed as i32,
        );
        let mut p_pp = self.d_pp;
        {
            let f = self.cc.function("llm170_fn_qsa_k_rows")?;
            let (mut a0, mut a1, mut a3) = (self.d_kk, self.d_kv_k, self.d_knw);
            let mut a4 = p_pp;
            let mut a10 = eps;
            let mut args: [*mut std::ffi::c_void; 11] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a_fi) as *mut _ as *mut _,
                (&mut a_cap) as *mut _ as *mut _,
                (&mut a_nkv) as *mut _ as *mut _,
                (&mut a_hd) as *mut _ as *mut _,
                (&mut a_rot) as *mut _ as *mut _,
                (&mut a10) as *mut _ as *mut _,
                (&mut a_pre) as *mut _ as *mut _,
            ];
            self.cc.launch(f, t_len as u32, n_kv as u32, 1, &mut args)?;
        }
        {
            let f = self.cc.function("llm170_fn_qsa_v_rows")?;
            let (mut a0, mut a1, mut a3) = (self.d_vv, self.d_kv_v, p_pp);
            let mut args: [*mut std::ffi::c_void; 7] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a_fi) as *mut _ as *mut _,
                (&mut a_cap) as *mut _ as *mut _,
                (&mut a_nkv) as *mut _ as *mut _,
                (&mut a_hd) as *mut _ as *mut _,
            ];
            self.cc.launch(f, t_len as u32, 1, 256, &mut args)?;
        }
        {
            let f = self.cc.function("llm170_fn_qsa_ik_rows")?;
            let (mut a0, mut a1, mut a3) = (self.d_ik, self.d_idx_k, p_pp);
            let mut a_id = id as i32;
            let mut args: [*mut std::ffi::c_void; 6] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a_fi) as *mut _ as *mut _,
                (&mut a_cap) as *mut _ as *mut _,
                (&mut a_id) as *mut _ as *mut _,
            ];
            self.cc.launch(f, t_len as u32, 1, 128, &mut args)?;
        }
        {
            let f = self.cc.function("llm170_fn_qsa_iq_rows")?;
            let (mut a0, mut a1, mut a2) = (self.d_iq, self.d_iqnr, self.d_iqw);
            let (mut a_ih, mut a_id, mut a_rot) = (ih as i32, id as i32, id as i32);
            let mut a8 = eps;
            let mut args: [*mut std::ffi::c_void; 9] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut p_pp) as *mut _ as *mut _,
                (&mut a_fi) as *mut _ as *mut _,
                (&mut a_ih) as *mut _ as *mut _,
                (&mut a_id) as *mut _ as *mut _,
                (&mut a_rot) as *mut _ as *mut _,
                (&mut a8) as *mut _ as *mut _,
            ];
            self.cc.launch(f, t_len as u32, ih as u32, 1, &mut args)?;
        }
        // ── 블록 키 캐시 증분(qsa.rs L198-208) ──
        let n_blocks_max = (pos0 + t_len) / r;
        let b0 = self.bk_blocks[fi];
        let n_new = n_blocks_max.saturating_sub(b0);
        if n_new > 0 {
            let shift = if neg == QsaNeg::PoolShift1 { 1 } else { 0 };
            let f = self.cc.function("llm170_fn_qsa_pool")?;
            let (mut a0, mut a1, mut a2) = (self.d_idx_k, self.d_idx_bk, self.d_ikw);
            let (mut a_r, mut a_b0, mut a_nnew) = (r as i32, b0 as i32, n_new as i32);
            let (mut a_id, mut a_rot, mut a_shift) = (id as i32, id as i32, shift);
            let mut a_bc = self.bk_cap as i32;
            let mut a11 = eps;
            let mut args: [*mut std::ffi::c_void; 13] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a_fi) as *mut _ as *mut _,
                (&mut a_cap) as *mut _ as *mut _,
                (&mut a_bc) as *mut _ as *mut _,
                (&mut a_r) as *mut _ as *mut _,
                (&mut a_b0) as *mut _ as *mut _,
                (&mut a_nnew) as *mut _ as *mut _,
                (&mut a_id) as *mut _ as *mut _,
                (&mut a_rot) as *mut _ as *mut _,
                (&mut a11) as *mut _ as *mut _,
                (&mut a_shift) as *mut _ as *mut _,
            ];
            self.cc.launch(f, n_new as u32, 1, 1, &mut args)?;
            if neg == QsaNeg::Off {
                self.bk_blocks[fi] = n_blocks_max;
            }
        }
        // ── 패스 B(qsa.rs L210-307) ──
        let sel_delta = if neg == QsaNeg::TopkOff1 { 1 } else { 0 };
        {
            let f = self.cc.function("llm170_fn_qsa_select")?;
            let (mut a0, mut a1) = (self.d_iqnr, self.d_idx_bk);
            let (mut a4, mut a5) = (self.d_sel_blk, self.d_sel_cnt);
            let (mut a_ih, mut a_id, mut a_r, mut a_topk) =
                (ih as i32, id as i32, r as i32, top_k as i32);
            let mut a_st = stride as i32;
            let mut a_bc = self.bk_cap as i32;
            let mut a13 = sel_delta;
            let mut args: [*mut std::ffi::c_void; 13] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut p_pp) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a5) as *mut _ as *mut _,
                (&mut a_fi) as *mut _ as *mut _,
                (&mut a_bc) as *mut _ as *mut _,
                (&mut a_ih) as *mut _ as *mut _,
                (&mut a_id) as *mut _ as *mut _,
                (&mut a_r) as *mut _ as *mut _,
                (&mut a_topk) as *mut _ as *mut _,
                (&mut a_st) as *mut _ as *mut _,
                (&mut a13) as *mut _ as *mut _,
            ];
            self.cc.launch(f, t_len as u32, 1, 256, &mut args)?;
        }
        let mut blk_b = vec![0u8; t_len * stride * 4];
        let mut cnt_b = vec![0u8; t_len * 4];
        self.cc.d2h(&mut blk_b, self.d_sel_blk)?;
        self.cc.d2h(&mut cnt_b, self.d_sel_cnt)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
        let sel_blk =
            unsafe { std::slice::from_raw_parts(blk_b.as_ptr() as *const u32, t_len * stride) }
                .to_vec();
        let sel_cnt =
            unsafe { std::slice::from_raw_parts(cnt_b.as_ptr() as *const u32, t_len) }.to_vec();
        Ok((sel_blk, sel_cnt, stride))
    }

    /// 선택 목록 평탄화 — 코어 qsa_sel_list(qsa.rs L323-358)의 호스트 직접
    /// 미러(정수 논리 — 커널 불필요. 블록 오름차순 + 테일).
    pub fn qsa_sel_list(
        &self,
        sel_blk: &[u32],
        sel_cnt: &[u32],
        sel_stride: usize,
        pos0: usize,
        n_tok: usize,
        r: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut sel_off: Vec<u32> = vec![0u32; n_tok + 1];
        for t2 in 0..n_tok {
            let n_past = pos0 + t2 + 1;
            let tail_cnt = n_past - (n_past / r) * r;
            sel_off[t2 + 1] = sel_off[t2] + sel_cnt[t2] * r as u32 + tail_cnt as u32;
        }
        let mut sel_idx: Vec<u32> = vec![0u32; sel_off[n_tok] as usize];
        for t2 in 0..n_tok {
            let n_past = pos0 + t2 + 1;
            let tail_start = (n_past / r) * r;
            let mut o = sel_off[t2] as usize;
            for k2 in 0..sel_cnt[t2] as usize {
                let b = sel_blk[t2 * sel_stride + k2] as usize;
                for j in 0..r {
                    sel_idx[o] = (b * r + j) as u32;
                    o += 1;
                }
            }
            for j in tail_start..n_past {
                sel_idx[o] = j as u32;
                o += 1;
            }
        }
        (sel_idx, sel_off)
    }

    /// QSA 스테이지 본체 — 코어 qsa_layer(L396-515) 중 스테이지 산술:
    /// qsa_select → q행 norm+rope(L508-517) → 선택 리스트 어텐션+게이트
    /// (cpu_attn_row 산술). qg: [t][n_head·2hd](wq 출력 원문 — norm·rope는
    /// 본 함수가 q반에 적용, 게이트 반은 원값). 반환: [t][n_head·hd] attn_all.
    pub fn qsa_stage(
        &mut self,
        il: usize,
        qg: &[Vec<f32>],
        kk: &[Vec<f32>],
        vv: &[Vec<f32>],
        iq: &[Vec<f32>],
        ik: &[Vec<f32>],
        t_len: usize,
        k_prenormed: bool,
    ) -> Result<Vec<Vec<f32>>, String> {
        self.qsa_select(il, kk, vv, iq, ik, t_len, k_prenormed, QsaNeg::Off)?;
        let fi = self.full_idx(il)?;
        let d = &self.dims;
        let (n_head, hd) = (d.n_head, d.head_dim);
        let pos0 = self.device_pos()? as usize;
        if qg.len() < t_len {
            return Err(format!("qsa_stage: qg {} < t={t_len}", qg.len()));
        }
        let mut qgf = Vec::with_capacity(t_len * n_head * 2 * hd);
        for (t, row) in qg.iter().enumerate() {
            if t >= t_len {
                break;
            }
            if row.len() != n_head * 2 * hd {
                return Err(format!(
                    "qsa_stage: qg 행 {t} 폭 {} != {}",
                    row.len(),
                    n_head * 2 * hd
                ));
            }
            qgf.extend_from_slice(row);
        }
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치).
        let qb = unsafe { std::slice::from_raw_parts(qgf.as_ptr() as *const u8, qgf.len() * 4) };
        let _g = self.cc.guard()?;
        self.cc.h2d(self.d_qg, qb)?;
        {
            let f = self.cc.function("llm170_fn_qsa_q_rows")?;
            let (mut a0, mut a1, mut a2) = (self.d_qg, self.d_qbuf, self.d_qnw);
            let (mut a_nh, mut a_hd, mut a_rot) = (n_head as i32, hd as i32, d.n_rot as i32);
            let (mut a_fi, mut a_eps) = (fi as i32, d.eps);
            let mut p_pp = self.d_pp;
            let mut args: [*mut std::ffi::c_void; 9] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut p_pp) as *mut _ as *mut _,
                (&mut a_fi) as *mut _ as *mut _,
                (&mut a_nh) as *mut _ as *mut _,
                (&mut a_hd) as *mut _ as *mut _,
                (&mut a_rot) as *mut _ as *mut _,
                (&mut a_eps) as *mut _ as *mut _,
            ];
            self.cc
                .launch(f, t_len as u32, n_head as u32, 1, &mut args)?;
        }
        // sel_list 평탄화(호스트 — 코어와 동일) → 업로드.
        let r = d.compress[il] as usize;
        let stride = self.sel_stride();
        let mut bb = vec![0u8; t_len * stride * 4];
        let mut cb = vec![0u8; t_len * 4];
        self.cc.d2h(&mut bb, self.d_sel_blk)?;
        self.cc.d2h(&mut cb, self.d_sel_cnt)?;
        // SAFETY: d2h 완료 후 재해석(길이·정렬 일치).
        let blk = unsafe { std::slice::from_raw_parts(bb.as_ptr() as *const u32, t_len * stride) }
            .to_vec();
        let cnt = unsafe { std::slice::from_raw_parts(cb.as_ptr() as *const u32, t_len) }.to_vec();
        let (sel_idx, sel_off) = self.qsa_sel_list(&blk, &cnt, stride, pos0, t_len, r);
        for t in 0..t_len {
            let np = (sel_off[t + 1] - sel_off[t]) as usize;
            if np > QSA_SEL_CAP {
                return Err(format!(
                    "qsa_stage: t={t} 선택 위치 {np} > QSA_SEL_CAP={QSA_SEL_CAP}"
                ));
            }
        }
        // SAFETY: u32 슬라이스 → 바이트 뷰(길이·정렬 일치).
        let ub =
            |v: &[u32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.d_sel_idx, ub(&sel_idx))?;
        self.cc.h2d(self.d_sel_off, ub(&sel_off))?;
        {
            let f = self.cc.function("llm170_fn_qsa_attn")?;
            let (mut a0, mut a1, mut a2, mut a3) =
                (self.d_qbuf, self.d_kv_k, self.d_kv_v, self.d_qg);
            let (mut a4, mut a5, mut a6) = (self.d_sel_idx, self.d_sel_off, self.d_attn);
            let (mut a_nh, mut a_nkv, mut a_hd) = (n_head as i32, d.n_kv as i32, hd as i32);
            let (mut a_fi, mut a_cap, mut a_scale) = (fi as i32, self.cap as i32, d.kq_scale());
            let mut p_pp = self.d_pp;
            let mut args: [*mut std::ffi::c_void; 14] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a5) as *mut _ as *mut _,
                (&mut a6) as *mut _ as *mut _,
                (&mut p_pp) as *mut _ as *mut _,
                (&mut a_fi) as *mut _ as *mut _,
                (&mut a_cap) as *mut _ as *mut _,
                (&mut a_nh) as *mut _ as *mut _,
                (&mut a_nkv) as *mut _ as *mut _,
                (&mut a_hd) as *mut _ as *mut _,
                (&mut a_scale) as *mut _ as *mut _,
            ];
            self.cc
                .launch(f, t_len as u32, n_head as u32, 256, &mut args)?;
        }
        let mut ob = vec![0u8; t_len * n_head * hd * 4];
        self.cc.d2h(&mut ob, self.d_attn)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석(길이·정렬 일치).
        let flat =
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, t_len * n_head * hd) };
        Ok((0..t_len)
            .map(|t| flat[t * n_head * hd..(t + 1) * n_head * hd].to_vec())
            .collect())
    }

    // ── 검증층 판독 헬퍼(상태 대조용 — 프로덕션 비관여) ──

    /// kv_k/kv_v 상위 n행 판독 [n][n_kv·hd].
    pub fn read_kv(&self, fi: usize, n_rows: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let w = self.dims.n_kv * self.dims.head_dim;
        let take = |src: CUdeviceptr| -> Result<Vec<f32>, String> {
            let mut b = vec![0u8; n_rows * w * 4];
            let _g = self.cc.guard()?;
            self.cc.d2h(&mut b, src + (fi * self.cap * w) as u64 * 4)?;
            Ok(
                unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, n_rows * w) }
                    .to_vec(),
            )
        };
        Ok((take(self.d_kv_k)?, take(self.d_kv_v)?))
    }

    /// idx_k 상위 n행 판독 [n][idx_dim].
    pub fn read_idx_k(&self, fi: usize, n_rows: usize) -> Result<Vec<f32>, String> {
        let w = self.dims.idx_dim;
        let mut b = vec![0u8; n_rows * w * 4];
        let _g = self.cc.guard()?;
        self.cc
            .d2h(&mut b, self.d_idx_k + (fi * self.cap * w) as u64 * 4)?;
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, n_rows * w) }.to_vec())
    }

    /// idx_bk n_blocks 블록 판독 [n_blocks][idx_dim].
    pub fn read_idx_bk(&self, fi: usize, n_blocks: usize) -> Result<Vec<f32>, String> {
        let w = self.dims.idx_dim;
        let mut b = vec![0u8; n_blocks * w * 4];
        let _g = self.cc.guard()?;
        self.cc
            .d2h(&mut b, self.d_idx_bk + (fi * self.bk_cap * w) as u64 * 4)?;
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, n_blocks * w) }.to_vec())
    }
}

/// exl3_fn_qsa.fatbin 자산 해석 — 경로 오버라이드(자산 경로 대체일 뿐 계산
/// 경로 분기 아님 — smoke/fn 리졸버와 동일 규약).
fn qsa_fatbin_bytes() -> Result<Vec<u8>, String> {
    const ENV: &str = "LLM170_CUDA_FN_QSA_FATBIN_PATH";
    const REL: &[&str] = &[
        "crates/backend-gpu/src/rawcuda/assets/exl3_fn_qsa.fatbin",
        "src/rawcuda/assets/exl3_fn_qsa.fatbin",
    ];
    if let Some(p) = std::env::var_os(ENV) {
        return std::fs::read(&p).map_err(|e| format!("{ENV}({p:?}) 읽기 실패: {e}"));
    }
    for r in REL {
        if let Ok(b) = std::fs::read(r) {
            return Ok(b);
        }
    }
    Err(format!(
        "exl3_fn_qsa.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
    ))
}
