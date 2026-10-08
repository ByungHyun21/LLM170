//! W4A16 GPTQ4 CUDA 모듈 — split(packed u32 + scale f16) GEMV/GEMM.
//! 비트 계약: core `dot_row_w4a16_lane`(64레인 f32 누산 → f64 tree64) —
//! assets/gptq4.cu 헤더 [빌드 계약](-fmad=false 필수).
//!
//! W2 단계: 값경로(업로드 → 커널 → 판독). 가중치 상주·디바이스 체인은 W3.
//! env 오버라이드(LLM170_CUDA_GPTQ4_FATBIN_PATH)는 flag 계약 경유.

use crate::rawcuda::ctx::CudaCtx;
use std::ffi::c_void;

pub struct Gptq4 {
    cc: CudaCtx,
}

impl Gptq4 {
    /// gptq4.fatbin 자산 해석 — env 오버라이드 우선(계산 경로 분기 아님).
    fn fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_GPTQ4_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/gptq4.fatbin",
            "src/rawcuda/assets/gptq4.fatbin",
        ];
        if let Some(p) = llm170_diag::flag::val(ENV) {
            return std::fs::read(p).map_err(|e| format!("{ENV}({p}) 읽기 실패: {e}"));
        }
        for r in REL {
            if let Ok(b) = std::fs::read(r) {
                return Ok(b);
            }
        }
        Err(format!("gptq4.fatbin 부재 — {REL:?} 또는 {ENV}"))
    }

    pub fn new() -> Result<Self, String> {
        let mut cc = CudaCtx::new()?;
        cc.load_fatbin(
            "gptq4",
            &Self::fatbin_bytes()?,
            &["w4a16_gemm_g128", "w4a16_gemv_g128"],
        )?;
        Ok(Gptq4 { cc })
    }

    /// f16 비트 → f32(정확 — 커널 h2f와 동일 값). t=1 GEMV의 x32 사전변환용.
    fn h2f(h: u16) -> f32 {
        let sign = ((h >> 15) as u32) << 31;
        let e = ((h >> 10) & 0x1F) as u32;
        let m = (h & 0x3FF) as u32;
        let bits = if e == 0 {
            if m == 0 {
                sign
            } else {
                // 서브노멀 정규화.
                let mut ee = 127 - 15 + 1;
                let mut f = m;
                while f & 0x400 == 0 {
                    f <<= 1;
                    ee -= 1;
                }
                sign | ((ee as u32) << 23) | ((f & 0x3FF) << 13)
            }
        } else if e == 0x1F {
            sign | (0xFF << 23) | (m << 13)
        } else {
            sign | ((e + 112) << 23) | (m << 13)
        };
        f32::from_bits(bits)
    }

    /// GEMV(t=1) — x f16 [k] → out f32 [n].
    pub fn gemv(
        &self,
        x: &[u16],
        q: &[u32],
        s: &[u16],
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        self.gemm(x, 1, q, s, n, k)
    }

    /// GEMM — x f16 [t][k], q u32 [n][k/8], s f16 [n][k/128] → out f32 [t][n].
    pub fn gemm(
        &self,
        x: &[u16],
        t: usize,
        q: &[u32],
        s: &[u16],
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if !k.is_multiple_of(128)
            || t == 0
            || q.len() != n * (k / 8)
            || s.len() != n * (k / 128)
            || x.len() != t * k
        {
            return Err(format!(
                "gptq4: 형상 계약 위반 q={} s={} x={} n={n} k={k} t={t}",
                q.len(),
                s.len(),
                x.len()
            ));
        }
        let bytes_u32 = |v: &[u32]| unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
        };
        let bytes_u16 = |v: &[u16]| unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
        };
        let dq = self.cc.alloc(q.len() * 4)?;
        let ds = self.cc.alloc(s.len() * 2)?;
        let dx = self.cc.alloc(x.len() * 2)?;
        let dout = self.cc.alloc(t * n * 4)?;
        // t=1은 행=블록 GEMV(x32 사전변환), t≥2는 구 GEMM 커널.
        let dx32 = if t == 1 { self.cc.alloc(k * 4)? } else { 0 };
        let r = (|| {
            self.cc.h2d(dq, bytes_u32(q))?;
            self.cc.h2d(ds, bytes_u16(s))?;
            if t == 1 {
                let xf: Vec<f32> = x.iter().map(|&h| Self::h2f(h)).collect();
                let xb =
                    unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
                self.cc.h2d(dx32, xb)?;
                let f = self.cc.function("w4a16_gemv_g128")?;
                let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, dx32, dout);
                let (mut p_n, mut p_k) = (n as i32, k as i32);
                let mut args: [*mut c_void; 6] = [
                    &mut p_q as *mut _ as *mut c_void,
                    &mut p_s as *mut _ as *mut c_void,
                    &mut p_x as *mut _ as *mut c_void,
                    &mut p_y as *mut _ as *mut c_void,
                    &mut p_n as *mut _ as *mut c_void,
                    &mut p_k as *mut _ as *mut c_void,
                ];
                self.cc.launch(f, n as u32, 1, 64, &mut args)?;
                self.cc.sync()?;
                let mut ob = vec![0u8; n * 4];
                self.cc.d2h(&mut ob, dout)?;
                return Ok(ob
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect::<Vec<f32>>());
            }
            self.cc.h2d(dx, bytes_u16(x))?;
            let f = self.cc.function("w4a16_gemm_g128")?;
            let (mut p_q, mut p_s, mut p_x, mut p_out) = (dq, ds, dx, dout);
            let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
            let mut args: [*mut c_void; 7] = [
                &mut p_q as *mut _ as *mut c_void,
                &mut p_s as *mut _ as *mut c_void,
                &mut p_x as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut p_n as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
                &mut p_t as *mut _ as *mut c_void,
            ];
            let gx = n.div_ceil(8) as u32;
            self.cc.launch(f, gx, 1, 64, &mut args)?;
            self.cc.sync()?;
            let mut ob = vec![0u8; t * n * 4];
            self.cc.d2h(&mut ob, dout)?;
            Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect::<Vec<f32>>())
        })();
        let _ = self.cc.free(dq);
        let _ = self.cc.free(ds);
        let _ = self.cc.free(dx);
        let _ = self.cc.free(dout);
        if dx32 != 0 {
            let _ = self.cc.free(dx32);
        }
        r
    }
}
