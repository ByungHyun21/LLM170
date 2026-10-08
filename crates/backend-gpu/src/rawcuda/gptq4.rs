//! W4A16 GPTQ4 CUDA 모듈 — split(packed u32 + scale f16) GEMV/GEMM.
//! 비트 계약: core `dot_row_w4a16_lane`(64레인 f32 누산 → f64 tree64) —
//! assets/gptq4.cu 헤더 [빌드 계약](-fmad=false 필수).
//!
//! W2 단계: 값경로(업로드 → 커널 → 판독). 가중치 상주·디바이스 체인은 W3.
//! env 오버라이드(LLM170_CUDA_GPTQ4_FATBIN_PATH)는 flag 계약 경유.

use crate::rawcuda::ctx::CudaCtx;
use std::ffi::c_void;

/// f16 비트 → f32(정확 — 커널 h2f와 동일 값). t=1 GEMV의 x32 사전변환용.
/// 계약 표면(pub): 서버 단위 테스트가 core `half_to_f32`와 전수 대조한다.
pub fn h2f(h: u16) -> f32 {
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

/// 형상 계약 검사(순수 — GPU 불필요, 단위 테스트 대상).
/// k ≤ 128*G4_SCMAX(32768)는 t=1 행=블록 GEMV의 smem `sc[G4_SCMAX]` 계약
/// (assets/gptq4.cu) — 상한 초과는 smem 오버런(UB)이라 호스트에서 거부한다.
pub fn check_shapes(
    t: usize,
    qlen: usize,
    slen: usize,
    xlen: usize,
    n: usize,
    k: usize,
    group: usize,
) -> Result<(), String> {
    if (group != 128 && group != 32)
        || !k.is_multiple_of(group)
        || t == 0
        || qlen != n * (k / 8)
        || slen != n * (k / group)
        || xlen != t * k
    {
        return Err(format!(
            "gptq4: 형상 계약 위반 q={qlen} s={slen} x={xlen} n={n} k={k} t={t} group={group}"
        ));
    }
    if t == 1 && k > group * 256 {
        return Err(format!(
            "gptq4: t=1 GEMV k 상한 위반 — k={k} > {}(smem sc 계약, g{group})",
            group * 256
        ));
    }
    Ok(())
}

/// 커널 심볼 선택 — (그룹, 스케일 dtype) 조합.
pub fn kernel_sym(gemm: bool, group: usize, scale_bf16: bool) -> Result<&'static str, String> {
    match (gemm, group, scale_bf16) {
        (false, 128, false) => Ok("w4a16_gemv_g128"),
        (false, 32, true) => Ok("w4a16_gemv_g32_bf16"),
        (true, 128, false) => Ok("w4a16_gemm_g128"),
        (true, 32, true) => Ok("w4a16_gemm_g32_bf16"),
        _ => Err(format!(
            "gptq4: 미지원 조합 group={group} scale_bf16={scale_bf16}"
        )),
    }
}

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
            &[
                "w4a16_gemm_g128",
                "w4a16_gemv_g128",
                "w4a16_gemm_g32_bf16",
                "w4a16_gemv_g32_bf16",
            ],
        )?;
        Ok(Gptq4 { cc })
    }

    /// GEMV(t=1) — x f16 [k] → out f32 [n]. group/scale_bf16 = 커널 변형 선택.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv(
        &self,
        x: &[u16],
        q: &[u32],
        s: &[u16],
        n: usize,
        k: usize,
        group: usize,
        scale_bf16: bool,
    ) -> Result<Vec<f32>, String> {
        self.gemm(x, 1, q, s, n, k, group, scale_bf16)
    }

    /// GEMM — x f16 [t][k], q u32 [n][k/8], s [n][k/group](f16/bf16) → out [t][n].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        x: &[u16],
        t: usize,
        q: &[u32],
        s: &[u16],
        n: usize,
        k: usize,
        group: usize,
        scale_bf16: bool,
    ) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        check_shapes(t, q.len(), s.len(), x.len(), n, k, group)?;
        let sym_gemv = kernel_sym(false, group, scale_bf16)?;
        let sym_gemm = kernel_sym(true, group, scale_bf16)?;
        let bytes_u32 = |v: &[u32]| unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
        };
        let bytes_u16 = |v: &[u16]| unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
        };
        let dq = self.cc.alloc(q.len() * 4)?;
        let ds = self.cc.alloc(s.len() * 2)?;
        // t≥2 경로는 f32 x(4B/원소) 업로드 — 넉넉히 4배로 잡는다(t=1은 dx32 사용).
        let dx = self.cc.alloc(x.len() * 4)?;
        let dout = self.cc.alloc(t * n * 4)?;
        // t=1은 행=블록 GEMV(x32 사전변환), t≥2는 구 GEMM 커널.
        let dx32 = if t == 1 { self.cc.alloc(k * 4)? } else { 0 };
        let r = (|| {
            self.cc.h2d(dq, bytes_u32(q))?;
            self.cc.h2d(ds, bytes_u16(s))?;
            if t == 1 {
                let xf: Vec<f32> = x.iter().map(|&h| h2f(h)).collect();
                let xb =
                    unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
                self.cc.h2d(dx32, xb)?;
                let f = self.cc.function(sym_gemv)?;
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
            // P3-b: 커널이 f32 x를 받는다(cast_x32 계약) — 호스트 h2f로 동형.
            let xf: Vec<f32> = x.iter().map(|&h| h2f(h)).collect();
            let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
            self.cc.h2d(dx, xb)?;
            let f = self.cc.function(sym_gemm)?;
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
            // 8행/블록 커널(2026-10-08 재작성 2) — grid = ceil(n/8), block 512.
            self.cc.launch(f, n.div_ceil(8) as u32, 1, 512, &mut args)?;
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

#[cfg(test)]
mod tests {
    use super::{check_shapes, kernel_sym};

    #[test]
    fn shapes_accept_g128_contract() {
        // n=8, k=5120: q=8*640, s=8*40, x=5120 (t=1).
        assert!(check_shapes(1, 8 * 640, 8 * 40, 5120, 8, 5120, 128).is_ok());
        // t=8 배치도 같은 계약.
        assert!(check_shapes(8, 8 * 640, 8 * 40, 8 * 5120, 8, 5120, 128).is_ok());
    }

    #[test]
    fn shapes_accept_g32_contract() {
        // W4-1: 35B 전문가 — n=512, k=2048, s=512*64(bf16 비트 2B).
        assert!(check_shapes(1, 512 * 256, 512 * 64, 2048, 512, 2048, 32).is_ok());
        assert!(check_shapes(8, 512 * 256, 512 * 64, 8 * 2048, 512, 2048, 32).is_ok());
        // g32 smem 상한 = 32*256 = 8192.
        assert!(check_shapes(1, 8 * 1024, 8 * 256, 8192, 8, 8192, 32).is_ok());
        assert!(check_shapes(1, 8 * 1056, 8 * 264, 8448, 8, 8448, 32).is_err());
    }

    #[test]
    fn shapes_reject_bad_rank_and_misalignment() {
        assert!(check_shapes(1, 8 * 640, 8 * 40, 5120, 8, 5121, 128).is_err()); // k % 128
        assert!(check_shapes(0, 8 * 640, 8 * 40, 0, 8, 5120, 128).is_err()); // t=0
        assert!(check_shapes(1, 7 * 640, 8 * 40, 5120, 8, 5120, 128).is_err()); // q 크기
        assert!(check_shapes(1, 8 * 640, 8 * 40, 5120, 8, 5120, 64).is_err()); // 미지원 그룹
    }

    #[test]
    fn shapes_reject_k_over_smem_for_gemv_only() {
        // A5: t=1은 smem 상한(32768), t≥2는 구 GEMM 커널이라 상한 없음.
        // k=32768(=128*256) 경계 통과, k=32896(한 그룹 초과)은 t=1만 거부.
        assert!(check_shapes(1, 8 * 4096, 8 * 256, 32768, 8, 32768, 128).is_ok());
        assert!(check_shapes(1, 8 * 4112, 8 * 257, 32896, 8, 32896, 128).is_err());
        assert!(check_shapes(2, 8 * 4112, 8 * 257, 2 * 32896, 8, 32896, 128).is_ok());
    }

    #[test]
    fn kernel_sym_selects_variants() {
        assert_eq!(kernel_sym(false, 128, false).unwrap(), "w4a16_gemv_g128");
        assert_eq!(kernel_sym(true, 128, false).unwrap(), "w4a16_gemm_g128");
        assert_eq!(kernel_sym(false, 32, true).unwrap(), "w4a16_gemv_g32_bf16");
        assert_eq!(kernel_sym(true, 32, true).unwrap(), "w4a16_gemm_g32_bf16");
        assert!(kernel_sym(true, 32, false).is_err());
        assert!(kernel_sym(false, 128, true).is_err());
    }
}
