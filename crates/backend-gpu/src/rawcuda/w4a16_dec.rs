//! W4A16 CUDA 디코더 — 호스트 스테이징 순차 체인 (W3-2).
//!
//! 구 EXL3 CUDA 호스트(exl3_cuda*.rs, B2에서 삭제 — git 586758df^)의
//! 체인 구조를 W4A16 계약으로 개조한 것. 커널 자산(attn/gdn/norm/ew)과
//! 발사 시퀀스는 동일하고, 선형 GEMV만 exl3 trellis → gptq4 split으로
//! 교체한다. 임베딩 행·최종 head는 호출자(서버)가 소유한다 — 이 모듈은
//! 디바이스 체인만 안다.
//!
//! [활성 정밀도] 체인 중간은 f32(CPU 참조와 같은 계급). GEMV 입력만
//! f16으로 캐스팅(RN-even — 커널 계약: assets/gptq4.cu는 f16 비트 입력).
//! 따라서 종단 판정은 비트가 아니라 **토큰열**(골든 대조)이다 — 모듈
//! 비트 계약은 w4a16-gemv/gemm 게이트가 따로 담당.
//!
//! [슬롯] GDN 링/스캔 상태·KV 캐시·pp가 슬롯별(포인터 오프셋). 가중치는
//! 전 슬롯 공유 — 커널 layer 인덱스를 오염시키지 않는다.

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::ffi::CUdeviceptr;
use std::collections::HashMap;

/// fwd3s T 상한(assets/attn.cu ATTN_TMAX와 동일 값).
pub const ATTN_F3S_TMAX: usize = 8;
/// GDN scan 동적 공유메모리(assets/gdn.cu 계약 — 정적 48KB 초과).
pub const GDN_SCAN_SMEM: u32 = 61_828;

/// GDN 체인 형상(서버가 config에서 유도해 명시 등록 — 추정 금지).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GdnDims {
    pub n_gdn: usize,
    pub hidden: usize,
    pub h_k: usize,
    pub h_v: usize,
    pub d: usize,
}

impl GdnDims {
    pub fn conv_ch(&self) -> usize {
        (2 * self.h_k + self.h_v) * self.d
    }
    pub fn k_len(&self) -> usize {
        self.h_k * self.d
    }
    pub fn v_len(&self) -> usize {
        self.h_v * self.d
    }
    pub fn bg_len(&self) -> usize {
        2 * self.h_v
    }
}

/// 어텐션 형상(서버 등록 — d=256·rope 64차 고정 계약).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnDims {
    pub n_attn: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub d: usize,
    pub cap: usize,
}

impl AttnDims {
    pub fn q_dim(&self) -> usize {
        self.q_heads * self.d
    }
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.d
    }
    pub fn qg_dim(&self) -> usize {
        self.q_heads * 2 * self.d
    }
    pub fn kv_slot_elems(&self, slot: usize) -> usize {
        slot * self.n_attn * self.cap * self.kv_dim()
    }
}

/// f32 → f16 비트(RN-even, 서브노멀·inf/nan 처리) — GEMV 활성 캐스팅 계약.
/// 커널은 이 f16 비트를 dot_row_w4a16_lane과 동일 산술로 소비한다.
pub fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut sub = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        if rem > half || (rem == half && (sub & 1) == 1) {
            sub += 1;
        }
        return sign | (sub as u16);
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | (h as u16)
}

fn asset_bytes(env: &str, rel: &[&str]) -> Result<Vec<u8>, String> {
    if let Some(p) = llm170_diag::flag::val(env) {
        return std::fs::read(p).map_err(|e| format!("{env}({p}) 읽기 실패: {e}"));
    }
    for r in rel {
        if let Ok(b) = std::fs::read(r) {
            return Ok(b);
        }
    }
    Err(format!("자산 부재 — {rel:?} 또는 {env}"))
}

pub struct W4a16Dec {
    pub cc: CudaCtx,
    pub hidden: usize,
    pub n_layers: usize,
    pub n_slots: usize,
    pub slot_pos: Vec<u32>,
    /// 선형 상주 — 이름 → (packed, scale, n, k).
    lins: HashMap<String, (CUdeviceptr, CUdeviceptr, usize, usize)>,
    /// GEMV 스테이징 — x f16 [t][k], y f32 [t][n].
    dxh: CUdeviceptr,
    xh_cap: usize,
    dy: CUdeviceptr,
    y_cap: usize,
    // ── norm ──
    dnw: CUdeviceptr,
    norm_w_rows: usize,
    dx: CUdeviceptr,
    dab: CUdeviceptr,
    dxn: CUdeviceptr,
    norm_cap: usize,
    // ── ew ──
    dewg: CUdeviceptr,
    dewu: CUdeviceptr,
    dew: CUdeviceptr,
    ew_cap: usize,
    // ── GDN ──
    gdn: Option<GdnDims>,
    dcw: CUdeviceptr,
    dab_c: CUdeviceptr,
    dalog: CUdeviceptr,
    ddtb: CUdeviceptr,
    dnwg: CUdeviceptr,
    dring: CUdeviceptr,
    dgst: CUdeviceptr,
    dqkv: CUdeviceptr,
    dzv: CUdeviceptr,
    dgxn: CUdeviceptr,
    dgq: CUdeviceptr,
    dgk: CUdeviceptr,
    dgv: CUdeviceptr,
    dq2: CUdeviceptr,
    dk2: CUdeviceptr,
    dv2: CUdeviceptr,
    dbg: CUdeviceptr,
    dgo: CUdeviceptr,
    dgate: CUdeviceptr,
    gdn_t_cap: usize,
    // ── attn ──
    attn: Option<AttnDims>,
    dqnw_a: CUdeviceptr,
    dknw_a: CUdeviceptr,
    dkc: CUdeviceptr,
    dvc: CUdeviceptr,
    dpp: CUdeviceptr,
    dqg_a: CUdeviceptr,
    dkin_a: CUdeviceptr,
    dvin_a: CUdeviceptr,
    dqh_a: CUdeviceptr,
    doutv_a: CUdeviceptr,
    attn_t_cap: usize,
    // ── 디바이스 체인(S10 — 연산별 왕복 제거) ──
    dres: CUdeviceptr,
    dab_dev: CUdeviceptr,
    dchain: [CUdeviceptr; 5],
    stg_w0: usize,
    stg_w1: usize,
    stg_w2: usize,
    chain_bufs_ok: bool,
    /// t=1 GEMV 입력 x32(h2f 왕복 f32) 버퍼.
    dx32: CUdeviceptr,
    dx32_cap: usize,
    // ── GPU head(output.weight bf16) ──
    head_w: CUdeviceptr,
    head_n: usize,
    head_k: usize,
    head_out: CUdeviceptr,
    /// 층별 잔차 합 덤프(CPU LLM170_DUMP=debug_layers와 대조용).
    pub debug_layers: bool,
}

// SAFETY: CUDA 핸들(*mut c_void)은 Send가 아니지만, 이 디코더는 서버
// 슬롯 스레드 1개가 소유·사용한다(공유 없음). 컨텍스트 current 전환은
// 진입마다 cc.guard()가 수행한다 — 구 Exl3CudaDecoder의 동일 계약
// (586758df^ exl3_cuda.rs L234 unsafe impl Send) 미러.
unsafe impl Send for W4a16Dec {}

impl W4a16Dec {
    /// 디코더 생성 — 5개 fatbin(체인 커널) 로드. 가중치는 upload_*로 공급.
    pub fn new(n_slots: usize, hidden: usize, n_layers: usize) -> Result<Self, String> {
        let mut cc = CudaCtx::new()?;
        cc.load_fatbin(
            "gptq4",
            &asset_bytes(
                "LLM170_CUDA_GPTQ4_FATBIN_PATH",
                &[
                    "crates/backend-gpu/src/rawcuda/assets/gptq4.fatbin",
                    "src/rawcuda/assets/gptq4.fatbin",
                ],
            )?,
            &[
                "w4a16_gemm_g128",
                "w4a16_gemv_g128",
                "w4a16_cast_f16",
                "w4a16_cast_x32",
            ],
        )?;
        cc.load_fatbin(
            "norm",
            &asset_bytes(
                "LLM170_CUDA_NORM_FATBIN_PATH",
                &[
                    "crates/backend-gpu/src/rawcuda/assets/norm.fatbin",
                    "src/rawcuda/assets/norm.fatbin",
                ],
            )?,
            &["norm_resid"],
        )?;
        cc.load_fatbin(
            "gdn",
            &asset_bytes(
                "LLM170_CUDA_GDN_FATBIN_PATH",
                &[
                    "crates/backend-gpu/src/rawcuda/assets/gdn.fatbin",
                    "src/rawcuda/assets/gdn.fatbin",
                ],
            )?,
            &[
                "gdn_conv",
                "gdn_l2perm",
                "gdn_l2perm_gather",
                "gdn_scan",
                "gdn_gate",
            ],
        )?;
        cc.load_fatbin(
            "attn",
            &asset_bytes(
                "LLM170_CUDA_ATTN_FATBIN_PATH",
                &[
                    "crates/backend-gpu/src/rawcuda/assets/attn.fatbin",
                    "src/rawcuda/assets/attn.fatbin",
                ],
            )?,
            &[
                "attn_prep",
                "attn_prep_hostpos",
                "attn_fwd3s",
                "attn_pos_bump",
            ],
        )?;
        cc.load_fatbin(
            "head",
            &asset_bytes(
                "LLM170_CUDA_HEAD_FATBIN_PATH",
                &[
                    "crates/backend-gpu/src/rawcuda/assets/head.fatbin",
                    "src/rawcuda/assets/head.fatbin",
                ],
            )?,
            &["head_bf16"],
        )?;
        cc.load_fatbin(
            "ew",
            &asset_bytes(
                "LLM170_CUDA_EW_FATBIN_PATH",
                &[
                    "crates/backend-gpu/src/rawcuda/assets/ew.fatbin",
                    "src/rawcuda/assets/ew.fatbin",
                ],
            )?,
            &["ew", "ew_argmax"],
        )?;
        Ok(W4a16Dec {
            cc,
            hidden,
            n_layers,
            n_slots: n_slots.max(1),
            slot_pos: vec![0; n_slots.max(1)],
            lins: HashMap::new(),
            dxh: 0,
            xh_cap: 0,
            dy: 0,
            y_cap: 0,
            dnw: 0,
            norm_w_rows: 0,
            dx: 0,
            dab: 0,
            dxn: 0,
            norm_cap: 0,
            dewg: 0,
            dewu: 0,
            dew: 0,
            ew_cap: 0,
            gdn: None,
            dcw: 0,
            dab_c: 0,
            dalog: 0,
            ddtb: 0,
            dnwg: 0,
            dring: 0,
            dgst: 0,
            dqkv: 0,
            dzv: 0,
            dgxn: 0,
            dgq: 0,
            dgk: 0,
            dgv: 0,
            dq2: 0,
            dk2: 0,
            dv2: 0,
            dbg: 0,
            dgo: 0,
            dgate: 0,
            gdn_t_cap: 0,
            attn: None,
            dqnw_a: 0,
            dknw_a: 0,
            dkc: 0,
            dvc: 0,
            dpp: 0,
            dqg_a: 0,
            dkin_a: 0,
            dvin_a: 0,
            dqh_a: 0,
            doutv_a: 0,
            attn_t_cap: 0,
            dres: 0,
            dab_dev: 0,
            dchain: [0; 5],
            stg_w0: 0,
            stg_w1: 0,
            stg_w2: 0,
            chain_bufs_ok: false,
            dx32: 0,
            dx32_cap: 0,
            head_w: 0,
            head_n: 0,
            head_k: 0,
            head_out: 0,
            debug_layers: false,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.cc.device_name
    }

    pub(crate) fn h2d_chunked(cc: &CudaCtx, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        const CH: usize = 4 << 20;
        for off in (0..src.len()).step_by(CH) {
            let end = (off + CH).min(src.len());
            cc.h2d(dst + off as u64, &src[off..end])?;
            // 업로드는 수 초~수십 초 — 와치독이 로드를 스텔로 오판하지 않게 심박.
            llm170_diag::watchdog::bump();
        }
        Ok(())
    }

    fn zero_dev(cc: &CudaCtx, ptr: CUdeviceptr, len: usize) -> Result<(), String> {
        let zero = vec![0u8; (4 << 20).min(len)];
        for off in (0..len).step_by(zero.len()) {
            cc.h2d(ptr + off as u64, &zero[..zero.len().min(len - off)])?;
        }
        Ok(())
    }

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
        let dq = self.cc.alloc(q.len())?;
        Self::h2d_chunked(&self.cc, dq, q)?;
        let ds = self.cc.alloc(s.len())?;
        Self::h2d_chunked(&self.cc, ds, s)?;
        if let Some((oq, os, _, _)) = self.lins.insert(name.to_string(), (dq, ds, n, k)) {
            let _ = self.cc.free(oq);
            let _ = self.cc.free(os);
        }
        Ok(())
    }

    /// 상주 선형 GEMV — x f32 → (RN-even) f16 → 커널 → y f32 [n].
    pub fn gemv_host(&mut self, name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        let Some(&(dq, ds, n, k)) = self.lins.get(name) else {
            return Err(format!("gemv: 상주 선형 없음: {name}"));
        };
        if x.len() != k {
            return Err(format!("gemv {name}: x={} != k={k}", x.len()));
        }
        if k > self.xh_cap {
            if self.dxh != 0 {
                self.cc.free(self.dxh)?;
            }
            self.dxh = self.cc.alloc(k * 2)?;
            self.xh_cap = k;
        }
        if n > self.y_cap {
            if self.dy != 0 {
                self.cc.free(self.dy)?;
            }
            self.dy = self.cc.alloc(n * 4)?;
            self.y_cap = n;
        }
        let xh: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
        let xb = unsafe { std::slice::from_raw_parts(xh.as_ptr() as *const u8, xh.len() * 2) };
        self.cc.h2d(self.dxh, xb)?;
        let f = self.cc.function("w4a16_gemm_g128")?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, self.dxh, self.dy);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, 1i32);
        let mut args: [*mut std::ffi::c_void; 7] = [
            (&mut p_q) as *mut _ as *mut _,
            (&mut p_s) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_y) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
            (&mut p_t) as *mut _ as *mut _,
        ];
        self.cc.launch(f, n.div_ceil(8) as u32, 1, 64, &mut args)?;
        let mut ob = vec![0u8; n * 4];
        self.cc.d2h(&mut ob, self.dy)?;
        self.cc.sync()?;
        Ok(ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
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
        let b = unsafe { std::slice::from_raw_parts(nw.as_ptr() as *const u8, nw.len() * 4) };
        let d = self.cc.alloc(nw.len() * 4)?;
        Self::h2d_chunked(&self.cc, d, b)?;
        self.dnw = d;
        self.norm_w_rows = rows;
        Ok(())
    }

    fn ensure_norm_bufs(&mut self, t_len: usize) -> Result<(), String> {
        let need = t_len * self.hidden;
        if need > self.norm_cap {
            for q in [self.dx, self.dab, self.dxn] {
                if q != 0 {
                    self.cc.free(q)?;
                }
            }
            self.dx = self.cc.alloc(need * 4)?;
            self.dab = self.cc.alloc(need * 4)?;
            self.dxn = self.cc.alloc(need * 4)?;
            self.norm_cap = need;
        }
        Ok(())
    }

    fn norm_resid_at(
        &mut self,
        w: usize,
        x_dev: CUdeviceptr,
        ab_dev: CUdeviceptr,
        t_len: usize,
    ) -> Result<CUdeviceptr, String> {
        if self.dnw == 0 {
            return Err("norm: 노름 가중 미등록".into());
        }
        if w >= self.norm_w_rows {
            return Err(format!("norm: w={w} >= rows={}", self.norm_w_rows));
        }
        let f = self.cc.function("norm_resid")?;
        let mut tl = t_len as i32;
        let mut wo = (w * self.hidden) as i32;
        let mut hd = self.hidden as i32;
        let (mut a0, mut a1, mut a2, mut a3) = (x_dev, self.dnw, ab_dev, self.dxn);
        let mut args: [*mut std::ffi::c_void; 7] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut wo) as *mut _ as *mut _,
            (&mut hd) as *mut _ as *mut _,
        ];
        self.cc.launch(f, t_len as u32, 1, 1024, &mut args)?;
        Ok(self.dxn)
    }

    /// 잔차 x_dev에 ab(호스트) 가산 + 노름 xn 판독(스테이징 1행).
    fn norm_resid_staged(
        &mut self,
        w: usize,
        x_dev: CUdeviceptr,
        ab: &[f32],
    ) -> Result<Vec<f32>, String> {
        if self.hidden == 0 || ab.len() != self.hidden || x_dev == 0 {
            return Err("norm: 순차 잔차/분기 폭 계약 위반".into());
        }
        self.ensure_norm_bufs(1)?;
        let abb = unsafe { std::slice::from_raw_parts(ab.as_ptr() as *const u8, ab.len() * 4) };
        self.cc.h2d(self.dab, abb)?;
        let xn = self.norm_resid_at(w, x_dev, self.dab, 1)?;
        let mut bytes = vec![0u8; self.hidden * 4];
        self.cc.d2h(&mut bytes, xn)?;
        self.cc.sync()?;
        Ok(
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.hidden) }
                .to_vec(),
        )
    }

    // ── ew(silu·mul) ──

    fn ensure_ew_bufs(&mut self, n: usize) -> Result<(), String> {
        if n > self.ew_cap {
            for q in [self.dewg, self.dewu, self.dew] {
                if q != 0 {
                    self.cc.free(q)?;
                }
            }
            self.dewg = self.cc.alloc(n * 4)?;
            self.dewu = self.cc.alloc(n * 4)?;
            self.dew = self.cc.alloc(n * 4)?;
            self.ew_cap = n;
        }
        Ok(())
    }

    pub fn ew_host(&mut self, g: &[f32], u: &[f32]) -> Result<Vec<f32>, String> {
        if g.is_empty() || g.len() != u.len() {
            return Err(format!("ew: g={} u={} 계약 위반", g.len(), u.len()));
        }
        self.ensure_ew_bufs(g.len())?;
        let gb = unsafe { std::slice::from_raw_parts(g.as_ptr() as *const u8, g.len() * 4) };
        let ub = unsafe { std::slice::from_raw_parts(u.as_ptr() as *const u8, u.len() * 4) };
        self.cc.h2d(self.dewg, gb)?;
        self.cc.h2d(self.dewu, ub)?;
        let f = self.cc.function("ew")?;
        let mut nn = g.len() as i32;
        let (mut a0, mut a1, mut a2) = (self.dewg, self.dewu, self.dew);
        let mut args: [*mut std::ffi::c_void; 4] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, g.len().div_ceil(128) as u32, 1, 128, &mut args)?;
        let mut yb = vec![0u8; g.len() * 4];
        self.cc.d2h(&mut yb, self.dew)?;
        self.cc.sync()?;
        Ok(unsafe { std::slice::from_raw_parts(yb.as_ptr() as *const f32, g.len()) }.to_vec())
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

    fn ensure_gdn_bufs(&mut self, t_len: usize) -> Result<(), String> {
        if t_len <= self.gdn_t_cap {
            return Ok(());
        }
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        let (hd, cch, kl, vl) = (dm.hidden, dm.conv_ch(), dm.k_len(), dm.v_len());
        for q in [
            self.dqkv, self.dzv, self.dgxn, self.dgq, self.dgk, self.dgv, self.dq2, self.dk2,
            self.dv2, self.dbg, self.dgo, self.dgate,
        ] {
            if q != 0 {
                self.cc.free(q)?;
            }
        }
        self.dqkv = self.cc.alloc(t_len * cch * 4)?;
        self.dzv = self.cc.alloc(t_len * vl * 4)?;
        self.dgxn = self.cc.alloc(t_len * hd * 4)?;
        self.dgq = self.cc.alloc(t_len * kl * 4)?;
        self.dgk = self.cc.alloc(t_len * kl * 4)?;
        self.dgv = self.cc.alloc(t_len * vl * 4)?;
        self.dq2 = self.cc.alloc(t_len * kl * 4)?;
        self.dk2 = self.cc.alloc(t_len * kl * 4)?;
        self.dv2 = self.cc.alloc(t_len * vl * 4)?;
        self.dbg = self.cc.alloc(t_len * dm.bg_len() * 4)?;
        self.dgo = self.cc.alloc(t_len * vl * 4)?;
        self.dgate = self.cc.alloc(t_len * vl * 4)?;
        self.gdn_t_cap = t_len;
        Ok(())
    }

    fn gdn_chain_dev(&mut self, slot: usize, layer: usize, t_len: usize) -> Result<(), String> {
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        if layer >= dm.n_gdn || slot >= self.n_slots || t_len == 0 {
            return Err(format!(
                "GDN: 범위 위반 layer={layer} slot={slot} t={t_len}"
            ));
        }
        self.ensure_gdn_bufs(t_len)?;
        let ring_slot = slot * dm.n_gdn * 3 * dm.conv_ch();
        let st_slot = slot * dm.n_gdn * dm.h_v * 128 * 128;
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let (mut hk, mut hv, mut dd) = (dm.h_k as i32, dm.h_v as i32, dm.d as i32);
        let (mut hd, mut kl, mut vl, mut cch) = (
            dm.hidden as i32,
            dm.k_len() as i32,
            dm.v_len() as i32,
            dm.conv_ch() as i32,
        );

        let f = self.cc.function("gdn_conv")?;
        let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5) = (
            self.dqkv,
            self.dcw,
            self.dring + (ring_slot as u64) * 4,
            self.dgq,
            self.dgk,
            self.dgv,
        );
        let mut ac: [*mut std::ffi::c_void; 11] = [
            (&mut c0) as *mut _ as *mut _,
            (&mut c1) as *mut _ as *mut _,
            (&mut c2) as *mut _ as *mut _,
            (&mut c3) as *mut _ as *mut _,
            (&mut c4) as *mut _ as *mut _,
            (&mut c5) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
            (&mut kl) as *mut _ as *mut _,
            (&mut vl) as *mut _ as *mut _,
            (&mut cch) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, (dm.conv_ch() / 128) as u32, 1, 128, &mut ac)?;

        let f = self.cc.function("gdn_l2perm")?;
        let (
            mut l0,
            mut l1,
            mut l2,
            mut l3,
            mut l4,
            mut l5,
            mut l6,
            mut l7,
            mut l8,
            mut l9,
            mut l10,
        ) = (
            self.dgq, self.dgk, self.dgv, self.dgxn, self.dab_c, self.dalog, self.ddtb, self.dq2,
            self.dk2, self.dv2, self.dbg,
        );
        let mut al: [*mut std::ffi::c_void; 16] = [
            (&mut l0) as *mut _ as *mut _,
            (&mut l1) as *mut _ as *mut _,
            (&mut l2) as *mut _ as *mut _,
            (&mut l3) as *mut _ as *mut _,
            (&mut l4) as *mut _ as *mut _,
            (&mut l5) as *mut _ as *mut _,
            (&mut l6) as *mut _ as *mut _,
            (&mut l7) as *mut _ as *mut _,
            (&mut l8) as *mut _ as *mut _,
            (&mut l9) as *mut _ as *mut _,
            (&mut l10) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
            (&mut hk) as *mut _ as *mut _,
            (&mut hv) as *mut _ as *mut _,
            (&mut hd) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, dm.h_v as u32, t_len as u32, 128, &mut al)?;

        let f = self.cc.function("gdn_scan")?;
        self.cc.set_dynamic_smem(f, GDN_SCAN_SMEM)?;
        let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) = (
            self.dq2,
            self.dk2,
            self.dv2,
            self.dbg,
            self.dgst + (st_slot as u64) * 4,
            self.dgo,
        );
        let mut as_: [*mut std::ffi::c_void; 11] = [
            (&mut s0) as *mut _ as *mut _,
            (&mut s1) as *mut _ as *mut _,
            (&mut s2) as *mut _ as *mut _,
            (&mut s3) as *mut _ as *mut _,
            (&mut s4) as *mut _ as *mut _,
            (&mut s5) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut hk) as *mut _ as *mut _,
            (&mut hv) as *mut _ as *mut _,
            (&mut dd) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
        ];
        self.cc
            .launch_shared(f, dm.h_v as u32, 1, 128, GDN_SCAN_SMEM, &mut as_)?;

        let f = self.cc.function("gdn_gate")?;
        let (mut g0, mut g1, mut g2, mut g3) = (self.dgo, self.dzv, self.dnwg, self.dgate);
        let mut ag: [*mut std::ffi::c_void; 8] = [
            (&mut g0) as *mut _ as *mut _,
            (&mut g1) as *mut _ as *mut _,
            (&mut g2) as *mut _ as *mut _,
            (&mut g3) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
            (&mut hk) as *mut _ as *mut _,
            (&mut hv) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, dm.h_v as u32, t_len as u32, 128, &mut ag)?;
        Ok(())
    }

    /// GDN 체인 호스트 진입 — xn·qkv·z 업로드 → 4커널 → gated 판독.
    pub fn gdn_chain_host(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        xn: &[f32],
        qkv: &[f32],
        z: &[f32],
    ) -> Result<Vec<f32>, String> {
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        if xn.len() != t_len * dm.hidden
            || qkv.len() != t_len * dm.conv_ch()
            || z.len() != t_len * dm.v_len()
        {
            return Err("GDN: 입력 형상 계약 위반".into());
        }
        self.ensure_gdn_bufs(t_len)?;
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.dgxn, b(xn))?;
        self.cc.h2d(self.dqkv, b(qkv))?;
        self.cc.h2d(self.dzv, b(z))?;
        self.gdn_chain_dev(slot, layer, t_len)?;
        let mut ob = vec![0u8; t_len * dm.v_len() * 4];
        self.cc.d2h(&mut ob, self.dgate)?;
        self.cc.sync()?;
        Ok(
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, t_len * dm.v_len()) }
                .to_vec(),
        )
    }

    // ── 어텐션 ──

    pub fn set_attn(&mut self, dims: AttnDims, qnw: &[f32], knw: &[f32]) -> Result<(), String> {
        let n = dims.n_attn;
        if qnw.len() != n * dims.d || knw.len() != n * dims.d {
            return Err(format!("attn: qnw/knw != {n}x{}", dims.d));
        }
        let slots = self.n_slots;
        let kv_elems = slots * n * dims.cap * dims.kv_dim();
        for q in [
            self.dqnw_a,
            self.dknw_a,
            self.dkc,
            self.dvc,
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
        (
            self.dqg_a,
            self.dkin_a,
            self.dvin_a,
            self.dqh_a,
            self.doutv_a,
        ) = (0, 0, 0, 0, 0);
        self.attn_t_cap = 0;
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let dq = self.cc.alloc(qnw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dq, b(qnw))?;
        let dk = self.cc.alloc(knw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dk, b(knw))?;
        let dkc = self.cc.alloc(kv_elems * 4)?;
        Self::zero_dev(&self.cc, dkc, kv_elems * 4)?;
        let dvc = self.cc.alloc(kv_elems * 4)?;
        Self::zero_dev(&self.cc, dvc, kv_elems * 4)?;
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

    fn ensure_attn_bufs(&mut self, t_len: usize) -> Result<(), String> {
        if t_len <= self.attn_t_cap {
            return Ok(());
        }
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        for q in [
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
        self.dqg_a = self.cc.alloc(t_len * dm.qg_dim() * 4)?;
        self.dkin_a = self.cc.alloc(t_len * dm.kv_dim() * 4)?;
        self.dvin_a = self.cc.alloc(t_len * dm.kv_dim() * 4)?;
        self.dqh_a = self.cc.alloc(t_len * dm.q_dim() * 4)?;
        self.doutv_a = self.cc.alloc(t_len * dm.q_dim() * 4)?;
        self.attn_t_cap = t_len;
        Ok(())
    }

    pub fn attn_set_pos(&mut self, slot: usize, pos: u32) -> Result<(), String> {
        if self.dpp == 0 || slot >= self.n_slots {
            return Err("attn: pp 미할당/슬롯 범위".into());
        }
        // 비동기 — 층마다 동기 H2D를 걸면 스트림이 매번 배수된다(층당 8ms 실측).
        self.cc
            .h2d_async(self.dpp + (slot as u64) * 4, &pos.to_le_bytes())
    }

    fn attn_pp_ptr(&self, slot: usize) -> CUdeviceptr {
        self.dpp + (slot as u64) * 4
    }

    fn attn_kv_ptr(&self, slot: usize) -> CUdeviceptr {
        match self.attn {
            Some(dm) => self.dkc + (dm.kv_slot_elems(slot) as u64) * 4,
            None => self.dkc,
        }
    }

    fn attn_vc_ptr(&self, slot: usize) -> CUdeviceptr {
        match self.attn {
            Some(dm) => self.dvc + (dm.kv_slot_elems(slot) as u64) * 4,
            None => self.dvc,
        }
    }

    fn attn_prep_launch(&mut self, slot: usize, layer: usize, t_len: usize) -> Result<(), String> {
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        let kv = self.attn_kv_ptr(slot);
        let f = self.cc.function("attn_prep")?;
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) = (
            self.dqg_a,
            self.dkin_a,
            self.dvin_a,
            self.dqnw_a,
            self.dknw_a,
            self.dqh_a,
            kv,
            self.attn_vc_ptr(slot),
            self.attn_pp_ptr(slot),
        );
        let mut args: [*mut std::ffi::c_void; 14] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut a4) as *mut _ as *mut _,
            (&mut a5) as *mut _ as *mut _,
            (&mut a6) as *mut _ as *mut _,
            (&mut a7) as *mut _ as *mut _,
            (&mut a8) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
            (&mut qh) as *mut _ as *mut _,
            (&mut kvh) as *mut _ as *mut _,
            (&mut cp) as *mut _ as *mut _,
        ];
        self.cc.launch(
            f,
            t_len as u32,
            (dm.q_heads + dm.kv_heads) as u32,
            128,
            &mut args,
        )
    }

    fn attn_fwd3s_launch(&mut self, slot: usize, layer: usize, t_len: usize) -> Result<(), String> {
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        if t_len == 0 || t_len > ATTN_F3S_TMAX {
            return Err(format!("attn fwd3s: T={t_len} — 소형 전용 도메인 위반"));
        }
        let f = self.cc.function("attn_fwd3s")?;
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        let (mut f0, mut f1, mut f2, mut f3, mut f4, mut f5) = (
            self.dqh_a,
            self.attn_kv_ptr(slot),
            self.attn_vc_ptr(slot),
            self.dqg_a,
            self.doutv_a,
            self.attn_pp_ptr(slot),
        );
        let mut args: [*mut std::ffi::c_void; 11] = [
            (&mut f0) as *mut _ as *mut _,
            (&mut f1) as *mut _ as *mut _,
            (&mut f2) as *mut _ as *mut _,
            (&mut f3) as *mut _ as *mut _,
            (&mut f4) as *mut _ as *mut _,
            (&mut f5) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
            (&mut qh) as *mut _ as *mut _,
            (&mut kvh) as *mut _ as *mut _,
            (&mut cp) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, t_len as u32, dm.q_heads as u32, 256, &mut args)
    }

    /// 어텐션 체인 호스트 진입 — qg·kin·vin 업로드 → pp=pos0 → prep → fwd3s
    /// → outv 판독.
    pub fn attn_chain_host(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg: &[f32],
        kin: &[f32],
        vin: &[f32],
        pos0: u32,
    ) -> Result<Vec<f32>, String> {
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        if layer >= dm.n_attn || slot >= self.n_slots {
            return Err(format!("attn: 범위 위반 layer={layer} slot={slot}"));
        }
        if qg.len() != t_len * dm.qg_dim()
            || kin.len() != t_len * dm.kv_dim()
            || vin.len() != t_len * dm.kv_dim()
        {
            return Err("attn: 입력 형상 계약 위반".into());
        }
        if pos0 as usize + t_len > dm.cap {
            return Err(format!(
                "attn: pos0={pos0} + T={t_len} > cap={}(--ctx 상향)",
                dm.cap
            ));
        }
        self.ensure_attn_bufs(t_len)?;
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.dqg_a, b(qg))?;
        self.cc.h2d(self.dkin_a, b(kin))?;
        self.cc.h2d(self.dvin_a, b(vin))?;
        self.attn_set_pos(slot, pos0)?;
        self.attn_prep_launch(slot, layer, t_len)?;
        self.attn_fwd3s_launch(slot, layer, t_len)?;
        let mut buf = vec![0u8; t_len * dm.q_dim() * 4];
        self.cc.d2h(&mut buf, self.doutv_a)?;
        self.cc.sync()?;
        Ok(
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, t_len * dm.q_dim()) }
                .to_vec(),
        )
    }

    // ── 순차 forward ──

    /// 1토큰 forward(호스트 스테이징) — 최종 노름 입력(head 입력)을 반환.
    /// KV·GDN 상태를 슬롯 위치만큼 전진시킨다.
    pub fn forward(&mut self, slot: usize, embed_row: &[f32]) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if embed_row.len() != self.hidden || slot >= self.n_slots {
            return Err("forward: 임베딩 폭/슬롯 계약 위반".into());
        }
        let pos = self.slot_pos[slot];
        let cap = self.attn.ok_or("attn: 형상 미등록")?.cap;
        if pos as usize >= cap {
            return Err(format!(
                "context overflow: slot{slot} pos={pos} >= kvcap={cap}"
            ));
        }
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("forward: 디코더 상수 미등록".into());
        }
        let dres = self.cc.alloc(self.hidden * 4)?;
        let result = (|| {
            let row = unsafe {
                std::slice::from_raw_parts(embed_row.as_ptr() as *const u8, self.hidden * 4)
            };
            self.cc.h2d(dres, row)?;
            self.forward_resident(slot, pos, dres)
        })();
        let freed = self.cc.free(dres);
        match (result, freed) {
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e),
            (Ok(v), Ok(())) => Ok(v),
        }
    }

    fn forward_resident(
        &mut self,
        slot: usize,
        pos: u32,
        dres: CUdeviceptr,
    ) -> Result<Vec<f32>, String> {
        let mut ab = vec![0.0f32; self.hidden];
        let mut gi = 0usize;
        let interval = 4usize;
        for il in 0..self.n_layers {
            let xn = self
                .norm_resid_staged(2 * il, dres, &ab)
                .map_err(|e| format!("L{il} input norm: {e}"))?;
            if il == 0 && self.debug_layers {
                eprintln!("  G0 xn[0..4]={:?}", &xn[..4]);
            }
            ab = if (il + 1) % interval == 0 {
                let ai = il / interval;
                let q = self
                    .gemv_host(&format!("blk.{il}.attn_q.weight"), &xn)
                    .map_err(|e| format!("L{il} q: {e}"))?;
                let k = self
                    .gemv_host(&format!("blk.{il}.attn_k.weight"), &xn)
                    .map_err(|e| format!("L{il} k: {e}"))?;
                let v = self
                    .gemv_host(&format!("blk.{il}.attn_v.weight"), &xn)
                    .map_err(|e| format!("L{il} v: {e}"))?;
                let out = self
                    .attn_chain_host(slot, ai, 1, &q, &k, &v, pos)
                    .map_err(|e| format!("L{il} attn: {e}"))?;
                self.gemv_host(&format!("blk.{il}.attn_output.weight"), &out)
                    .map_err(|e| format!("L{il} o: {e}"))?
            } else {
                let qkv = self
                    .gemv_host(&format!("blk.{il}.attn_qkv.weight"), &xn)
                    .map_err(|e| format!("L{il} qkv: {e}"))?;
                let z = self
                    .gemv_host(&format!("blk.{il}.attn_gate.weight"), &xn)
                    .map_err(|e| format!("L{il} z: {e}"))?;
                if il == 0 && self.debug_layers {
                    eprintln!("  G0 qkv[0..4]={:?}", &qkv[..4]);
                    eprintln!("  G0 z[0..4]={:?}", &z[..4]);
                }
                let gated = self
                    .gdn_chain_host(slot, gi, 1, &xn, &qkv, &z)
                    .map_err(|e| format!("L{il} gdn: {e}"))?;
                gi += 1;
                if il == 0 && self.debug_layers {
                    eprintln!("  G0 gated[0..4]={:?}", &gated[..4]);
                }
                let o = self
                    .gemv_host(&format!("blk.{il}.ssm_out.weight"), &gated)
                    .map_err(|e| format!("L{il} ssm_out: {e}"))?;
                if il == 0 && self.debug_layers {
                    eprintln!("  G0 out[0..4]={:?}", &o[..4]);
                }
                o
            };
            let xn = self
                .norm_resid_staged(2 * il + 1, dres, &ab)
                .map_err(|e| format!("L{il} post norm: {e}"))?;
            let gate = self
                .gemv_host(&format!("blk.{il}.ffn_gate.weight"), &xn)
                .map_err(|e| format!("L{il} gate: {e}"))?;
            let up = self
                .gemv_host(&format!("blk.{il}.ffn_up.weight"), &xn)
                .map_err(|e| format!("L{il} up: {e}"))?;
            let act = self
                .ew_host(&gate, &up)
                .map_err(|e| format!("L{il} ew: {e}"))?;
            ab = self
                .gemv_host(&format!("blk.{il}.ffn_down.weight"), &act)
                .map_err(|e| format!("L{il} down: {e}"))?;
            if self.debug_layers {
                let mut db = vec![0u8; self.hidden * 4];
                self.cc.d2h(&mut db, dres)?;
                self.cc.sync()?;
                let d: f64 = db
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c) as f64)
                    .sum();
                let a: f64 = ab.iter().map(|&v| v as f64).sum();
                eprintln!(
                    "  G{il:>2} recr={} sum={:.6}",
                    (il + 1) % interval != 0,
                    d + a
                );
            }
        }
        let xn = self
            .norm_resid_staged(2 * self.n_layers, dres, &ab)
            .map_err(|e| format!("final norm: {e}"))?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        Ok(xn)
    }

    /// 슬롯 상태 0화(GDN 링/스캔 + pp + pos).
    pub fn reset_state(&mut self, slot: usize) -> Result<(), String> {
        let _g = self.cc.guard()?;
        if slot >= self.n_slots {
            return Err(format!("reset: slot={slot} >= {}", self.n_slots));
        }
        self.slot_pos[slot] = 0;
        self.attn_set_pos(slot, 0)?;
        if let Some(dims) = self.gdn {
            let ring_bytes = dims.n_gdn * 3 * dims.conv_ch() * 4;
            let st_bytes = dims.n_gdn * dims.h_v * 128 * 128 * 4;
            let ring_off = (slot * dims.n_gdn * 3 * dims.conv_ch()) as u64 * 4;
            let st_off = (slot * dims.n_gdn * dims.h_v * 128 * 128) as u64 * 4;
            Self::zero_dev(&self.cc, self.dring + ring_off, ring_bytes)?;
            Self::zero_dev(&self.cc, self.dgst + st_off, st_bytes)?;
        }
        Ok(())
    }

    // ── 디바이스 체인(S10 — 왕복 제거) ──

    fn lin_spec(&self, name: &str) -> Result<(CUdeviceptr, CUdeviceptr, usize, usize), String> {
        self.lins
            .get(name)
            .copied()
            .ok_or_else(|| format!("상주 선형 없음: {name}"))
    }

    /// 체인 작업 버퍼 보장(1회 — 폭은 선형 형상에서 산출).
    fn ensure_chain_bufs(&mut self) -> Result<(), String> {
        if self.chain_bufs_ok {
            return Ok(());
        }
        let h = self.hidden;
        let ad = self.attn.ok_or("attn: 형상 미등록")?;
        let gd = self.gdn.ok_or("GDN: 형상 미등록")?;
        let (_, _, ff_gate, _) = self.lin_spec("blk.0.ffn_gate.weight")?;
        let (_, _, ff_up, _) = self.lin_spec("blk.0.ffn_up.weight")?;
        // 슬롯 0 = qg·qkv·gate/up, 슬롯 1 = kin/vin·z·up. ew가 gate·up을
        // 동시에 읽으므로 둘 다 FFN 폭 확보(구 S10 ensure_chain_bufs 계약).
        let w0 = ad.qg_dim().max(gd.conv_ch()).max(ff_gate);
        let w1 = ad.kv_dim().max(gd.v_len()).max(ff_up);
        for p in [
            self.dres,
            self.dab_dev,
            self.dchain[0],
            self.dchain[1],
            self.dchain[2],
            self.dchain[3],
            self.dchain[4],
        ] {
            if p != 0 {
                self.cc.free(p)?;
            }
        }
        self.dres = self.cc.alloc(h * 4)?;
        self.dab_dev = self.cc.alloc(h * 4)?;
        Self::zero_dev(&self.cc, self.dab_dev, h * 4)?;
        self.dchain = [
            self.cc.alloc(w0 * 4)?,
            self.cc.alloc(w1 * 4)?,
            self.cc.alloc(w1 * 4)?,
            self.cc.alloc(ff_up * 4)?,
            self.cc.alloc(h * 4)?,
        ];
        self.stg_w0 = w0;
        self.stg_w1 = w1;
        self.stg_w2 = ff_up;
        self.chain_bufs_ok = true;
        Ok(())
    }

    /// 디바이스 GEMV(t=1) — x_dev f32 → x32(h2f 왕복) → gptq4 행=블록 커널
    /// → dy. 반환 포인터는 self.dy(다음 gemv가 덮는다 — 스트림 순서 계약).
    fn gemv_dev(&mut self, name: &str, x_dev: CUdeviceptr) -> Result<CUdeviceptr, String> {
        let (dq, ds, n, k) = self.lin_spec(name)?;
        if k > 128 * 256 {
            return Err(format!("gemv_dev({name}): k={k} > 32768(smem 계약)"));
        }
        if k > self.dx32_cap {
            if self.dx32 != 0 {
                self.cc.free(self.dx32)?;
            }
            self.dx32 = self.cc.alloc(k * 4)?;
            self.dx32_cap = k;
        }
        if n > self.y_cap {
            if self.dy != 0 {
                self.cc.free(self.dy)?;
            }
            self.dy = self.cc.alloc(n * 4)?;
            self.y_cap = n;
        }
        let f = self.cc.function("w4a16_cast_x32")?;
        let mut nn = k as i32;
        let (mut c0, mut c1) = (x_dev, self.dx32);
        let mut ca: [*mut std::ffi::c_void; 3] = [
            (&mut c0) as *mut _ as *mut _,
            (&mut c1) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        self.cc.launch(f, k.div_ceil(256) as u32, 1, 256, &mut ca)?;
        let f = self.cc.function("w4a16_gemv_g128")?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, self.dx32, self.dy);
        let (mut p_n, mut p_k) = (n as i32, k as i32);
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut p_q) as *mut _ as *mut _,
            (&mut p_s) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_y) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
        ];
        self.cc.launch(f, n as u32, 1, 64, &mut args)?;
        Ok(self.dy)
    }

    /// GEMV → 상주 스테이징 복사(단일 대여 — 중첩 빌림 회피). 복사량은
    /// 그 선형의 실제 출력폭 n(스테이징 공유 폭까지 복사하면 범위 초과).
    fn gemv_stage(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        dst: CUdeviceptr,
        w: usize,
    ) -> Result<(), String> {
        let (_, _, n, _) = self.lin_spec(name)?;
        if n > w {
            return Err(format!("gemv_stage({name}): n={n} > 스테이징 {w}"));
        }
        let p = self.gemv_dev(name, x_dev)?;
        self.cc.d2d(dst, p, n * 4)
    }

    /// 노름 1회(디바이스 x·ab) — xn은 self.dxn(다음 노름이 덮는다).
    fn norm_resid_dev(
        &mut self,
        w: usize,
        x_dev: CUdeviceptr,
        ab_dev: CUdeviceptr,
        t_len: usize,
    ) -> Result<CUdeviceptr, String> {
        self.ensure_norm_bufs(t_len)?;
        self.norm_resid_at(w, x_dev, ab_dev, t_len)
    }

    /// GDN 체인 디바이스 상주 — xn·qkv·z(디바이스) → dgate.
    fn gdn_chain_dev_run(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        xn_dev: CUdeviceptr,
        qkv_dev: CUdeviceptr,
        z_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        self.ensure_gdn_bufs(t_len)?;
        self.cc.d2d(self.dgxn, xn_dev, t_len * dm.hidden * 4)?;
        self.cc.d2d(self.dqkv, qkv_dev, t_len * dm.conv_ch() * 4)?;
        self.cc.d2d(self.dzv, z_dev, t_len * dm.v_len() * 4)?;
        self.gdn_chain_dev(slot, layer, t_len)?;
        Ok(self.dgate)
    }

    /// 어텐션 체인 디바이스 상주 — qg·kin·vin(디바이스) → doutv.
    fn attn_chain_dev_run(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
        kin_dev: CUdeviceptr,
        vin_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        if t_len == 0 || t_len > ATTN_F3S_TMAX || layer >= dm.n_attn || slot >= self.n_slots {
            return Err("attn dev: 도메인/범위 위반".into());
        }
        let pos = self.slot_pos[slot];
        if pos as usize + t_len > dm.cap {
            return Err(format!("attn dev: pos{pos}+T{t_len} > cap{}", dm.cap));
        }
        self.attn_set_pos(slot, pos)?;
        self.ensure_attn_bufs(t_len)?;
        self.cc.d2d(self.dqg_a, qg_dev, t_len * dm.qg_dim() * 4)?;
        self.cc.d2d(self.dkin_a, kin_dev, t_len * dm.kv_dim() * 4)?;
        self.cc.d2d(self.dvin_a, vin_dev, t_len * dm.kv_dim() * 4)?;
        self.attn_prep_launch(slot, layer, t_len)?;
        self.attn_fwd3s_launch(slot, layer, t_len)?;
        Ok(self.doutv_a)
    }

    /// ew(silu·mul) 디바이스 발사 — g·u → y.
    fn ew_dev(
        &mut self,
        g_dev: CUdeviceptr,
        u_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        n: usize,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("ew: n=0".into());
        }
        let f = self.cc.function("ew")?;
        let mut nn = n as i32;
        let (mut a0, mut a1, mut a2) = (g_dev, u_dev, y_dev);
        let mut args: [*mut std::ffi::c_void; 4] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        self.cc.launch(f, n.div_ceil(128) as u32, 1, 128, &mut args)
    }

    /// 1토큰 forward(디바이스 체인) — 왕복은 임베딩 업로드 1회 + 최종 xn
    /// 판독 1회뿐. 산술은 스테이징 경로와 같은 커널·같은 순서(층 4주기).
    fn chain_device(&mut self, slot: usize, embed_row: &[f32]) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        if embed_row.len() != self.hidden || slot >= self.n_slots {
            return Err("forward_device: 임베딩 폭/슬롯 계약 위반".into());
        }
        let pos = self.slot_pos[slot];
        let cap = self.attn.ok_or("attn: 형상 미등록")?.cap;
        if pos as usize >= cap {
            return Err(format!(
                "context overflow: slot{slot} pos={pos} >= kvcap={cap}"
            ));
        }
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("forward_device: 디코더 상수 미등록".into());
        }
        self.ensure_chain_bufs()?;
        let row =
            unsafe { std::slice::from_raw_parts(embed_row.as_ptr() as *const u8, self.hidden * 4) };
        self.cc.h2d_async(self.dres, row)?;
        let [s0, s1, s1b, s2, s3] = self.dchain;
        let (w0, w1, w2) = (self.stg_w0, self.stg_w1, self.stg_w2);
        let mut ab = self.dab_dev;
        let mut gi = 0usize;
        for il in 0..self.n_layers {
            let xn = self
                .norm_resid_dev(2 * il, self.dres, ab, 1)
                .map_err(|e| format!("L{il} input norm: {e}"))?;
            let branch = if (il + 1) % 4 == 0 {
                self.gemv_stage(&format!("blk.{il}.attn_q.weight"), xn, s0, w0)
                    .map_err(|e| format!("L{il} q: {e}"))?;
                self.gemv_stage(&format!("blk.{il}.attn_k.weight"), xn, s1, w1)
                    .map_err(|e| format!("L{il} k: {e}"))?;
                self.gemv_stage(&format!("blk.{il}.attn_v.weight"), xn, s1b, w1)
                    .map_err(|e| format!("L{il} v: {e}"))?;
                self.attn_chain_dev_run(slot, il / 4, 1, s0, s1, s1b)
                    .map_err(|e| format!("L{il} attn: {e}"))?
            } else {
                self.gemv_stage(&format!("blk.{il}.attn_qkv.weight"), xn, s0, w0)
                    .map_err(|e| format!("L{il} qkv: {e}"))?;
                self.gemv_stage(&format!("blk.{il}.attn_gate.weight"), xn, s1, w1)
                    .map_err(|e| format!("L{il} z: {e}"))?;
                let g = self
                    .gdn_chain_dev_run(slot, gi, 1, xn, s0, s1)
                    .map_err(|e| format!("L{il} gdn: {e}"))?;
                gi += 1;
                g
            };
            let lo = if (il + 1) % 4 == 0 {
                format!("blk.{il}.attn_output.weight")
            } else {
                format!("blk.{il}.ssm_out.weight")
            };
            let out = self
                .gemv_dev(&lo, branch)
                .map_err(|e| format!("L{il} {lo}: {e}"))?;
            let xn2 = self
                .norm_resid_dev(2 * il + 1, self.dres, out, 1)
                .map_err(|e| format!("L{il} post norm: {e}"))?;
            self.gemv_stage(&format!("blk.{il}.ffn_gate.weight"), xn2, s0, w0)
                .map_err(|e| format!("L{il} gate: {e}"))?;
            self.gemv_stage(&format!("blk.{il}.ffn_up.weight"), xn2, s1, w1)
                .map_err(|e| format!("L{il} up: {e}"))?;
            self.ew_dev(s0, s1, s2, w2)?;
            let down = self
                .gemv_dev(&format!("blk.{il}.ffn_down.weight"), s2)
                .map_err(|e| format!("L{il} down: {e}"))?;
            self.cc.d2d(s3, down, self.hidden * 4)?;
            ab = s3;
            if self.debug_layers {
                let mut db = vec![0u8; self.hidden * 4];
                let mut abv = vec![0u8; self.hidden * 4];
                self.cc.d2h(&mut db, self.dres)?;
                self.cc.d2h(&mut abv, ab)?;
                self.cc.sync()?;
                let d: f64 = db
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c) as f64)
                    .sum();
                let a: f64 = abv
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c) as f64)
                    .sum();
                eprintln!("  G{il:>2} recr={} sum={:.6}", (il + 1) % 4 != 0, d + a);
            }
        }
        let xn_final = self
            .norm_resid_dev(2 * self.n_layers, self.dres, ab, 1)
            .map_err(|e| format!("final norm: {e}"))?;
        Ok(xn_final)
    }

    /// 1토큰 forward(디바이스 체인) — 최종 xn까지 판독(CPU head용).
    /// guard는 래퍼 전체를 덮는다 — 체인 이후의 d2h/h2d도 같은 스레드
    /// 컨텍스트가 필요하다(CtxGuard는 드랍 시 이전 컨텍스트로 복원).
    pub fn forward_device(&mut self, slot: usize, embed_row: &[f32]) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        let pos = self.slot_pos[slot];
        let xn_final = self.chain_device(slot, embed_row)?;
        let mut ob = vec![0u8; self.hidden * 4];
        self.cc.d2h(&mut ob, xn_final)?;
        self.cc.sync()?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        Ok(unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, self.hidden) }.to_vec())
    }

    /// bf16 head(output.weight) 상주 업로드 — head_bf16 GEMV.
    pub fn upload_head(&mut self, data: &[u8], n: usize, k: usize) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || data.len() < need {
            return Err(format!(
                "upload_head: 형상 계약 위반 n={n} k={k} bytes={} < {need}",
                data.len()
            ));
        }
        for p in [self.head_w, self.head_out] {
            if p != 0 {
                self.cc.free(p)?;
            }
        }
        self.head_w = self.cc.alloc(need)?;
        Self::h2d_chunked(&self.cc, self.head_w, &data[..need])?;
        self.head_out = self.cc.alloc(n * 4)?;
        self.head_n = n;
        self.head_k = k;
        Ok(())
    }

    /// 1토큰 forward + GPU head — xn 판독 없이 로짓만 회수(왕복 1회).
    pub fn forward_device_head(
        &mut self,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<Vec<f32>, String> {
        if self.head_w == 0 {
            return Err("forward_device_head: head 미등록 — upload_head 선행".into());
        }
        let _g = self.cc.guard()?;
        let pos = self.slot_pos[slot];
        let xn = self.chain_device(slot, embed_row)?;
        let f = self.cc.function("head_bf16")?;
        let (mut p_w, mut p_x, mut p_o) = (self.head_w, xn, self.head_out);
        let (mut p_n, mut p_k) = (self.head_n as i32, self.head_k as i32);
        let mut args: [*mut std::ffi::c_void; 5] = [
            (&mut p_w) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_o) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, self.head_n.div_ceil(4 * 256) as u32, 1, 256, &mut args)?;
        let mut ob = vec![0u8; self.head_n * 4];
        self.cc.d2h(&mut ob, self.head_out)?;
        self.cc.sync()?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        Ok(ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }
}
