//! EXL3 배치 GEMM(gemm2/kseg) 모듈층(plans/124 G4, G10 파일 분할
//! 2026-10-04). Exl3CudaDecoder의 배치 GEMM 임플 블록 — 레지스트리·
//! 컨텍스트·fatbin 리졸버는 exl3_cuda.rs 공유 글루.
//!
//! [용도] 배치 T행 선형: had_in T행 → mma m16n8k16 gemm2(소형-T kseg=8
//! 부분합 [T≤8][kseg][n] / 대형 단일 [T][n]) → had_out 정확 1회(H도메인
//! 출력 — 결함 3·15·18호 가드, plans/124 §1 "승부처": CUDA에서
//! wmma/mma m16n8k16으로 19TF+ 지향).
//!
//! [정합 — plans/129-cuda C2 원장, sm_89 실측 2026-10-04] (i) 27B k=5120
//! n=17408 K=3 T=32[plain]: 2.190e-6 · (ii) T=1[kseg]: 2.235e-7 ·
//! (iii) T=4[kseg]: 2.831e-7 · (iv) 35B k=2048 n=8192 K=4 T=32[plain]:
//! 5.662e-7 — 전 항목 임계 4e-4 대비 180~1780배 여유(mma f32 누산 순서
//! 계급). had_in 비트일치: kseg T=4 0/5120 · plain T=32 0/40960(결함 20호
//! 수정 후). 음성대조 had_out 2회: 3.118e-1 > 4e-4 → NEG-DETECTED
//! (결함 15호 감지).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). hip 8060S
//! 참고치 2.6 TF(스칼라); sm_80 자원 증거: exl3_gemm2 REG:72 SHARED:6144 →
//! 7블록/SM(28와프) · exl3_gemm2_kseg REG:80 SHARED:6144 → 6블록/SM(24와프).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지 — scripts/cuda_probe_shim.rs 단독 컴파일.

use crate::rawcuda::exl3_cuda::{CudaLin, Exl3CudaDecoder};
use crate::rawcuda::ffi::CUdeviceptr;

/// 배치 GEMM k-분할 수(hip kseg 원장 — 결함 18호 소형-T 점유 레버).
pub const GEMM2_KSEG: usize = 8;
/// kseg 경로 T 상한(hip gemm2_batch 게이트와 동일).
pub const GEMM2_KSEG_MAX_T: usize = 8;
/// kseg 경로 n 상한(hip 게이트와 동일 — 부분합 버퍼 예산 원천).
pub const GEMM2_KSEG_MAX_N: usize = 17408;
/// plain gemm2 블록당 t-타일(mma m16 2개 — 커널 계약).
pub const GEMM2_T_TILE: usize = 32;

impl Exl3CudaDecoder {
    // ── 배치 GEMM(gemm2/kseg — plans/124 G4 §3.1·§4.18) ──

    /// 배치 GEMM 작업 버퍼 보장(daht [t][k]f16팩 · dyt [t][n] f32 ·
    /// dbat [T≤8][kseg][n] — 확장 시에만 재할당). dx는 [t][k] f32로 확보.
    fn ensure_gemm2_bufs(&mut self, k: usize, n: usize, t_len: usize) -> Result<(), String> {
        self.ensure_x(t_len * k)?;
        let ah_bytes = t_len * k * 2;
        if ah_bytes > self.gemm_ah_cap {
            if self.daht != 0 {
                self.cc.free(self.daht)?;
            }
            self.daht = self.cc.alloc(ah_bytes)?;
            self.gemm_ah_cap = ah_bytes;
        }
        let y_bytes = t_len * n * 4;
        if y_bytes > self.gemm_y_cap {
            if self.dyt != 0 {
                self.cc.free(self.dyt)?;
            }
            self.dyt = self.cc.alloc(y_bytes)?;
            self.gemm_y_cap = y_bytes;
        }
        if n > self.gemm_bat_n {
            if self.dbat != 0 {
                self.cc.free(self.dbat)?;
            }
            // kseg 경로 T 상한 8행 고정(hip dbat 예산과 동일 원천).
            self.dbat = self.cc.alloc(GEMM2_KSEG_MAX_T * GEMM2_KSEG * n * 4)?;
            self.gemm_bat_n = n;
        }
        Ok(())
    }

    /// 배치 had_in(dx [t][k] f32 → daht [t][k/2] f16쌍) — suh는 선형별
    /// (호출 선형에서 명시 전달, 결함 1호). exl3_had_in은 grid.y=T로
    /// 이미 다중행 지원(G2 커널 원형 그대로).
    fn had16_batch(&mut self, k: usize, t_len: usize, suh: CUdeviceptr) -> Result<(), String> {
        let f_hin = self.cc.function("exl3_had_in")?;
        let (mut kc, mut ks) = ((k / 128) as i32, k as i32);
        let (mut a0, mut a1, mut a2) = (self.dx, suh, self.daht);
        let mut args_hin: [*mut std::ffi::c_void; 5] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut kc) as *mut _ as *mut _,
            (&mut ks) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f_hin, (k / 128) as u32, t_len as u32, 128, &mut args_hin)
    }

    /// had_out 발사 범용(hip hadout_batch 미러): nseg 합산 + WHT⁻¹·R·svh
    /// → dst. 제자리(src=dst)는 청크별 공유메모리 스테이징이라 안전.
    /// had_out은 체인당 정확 1회(결함 3·15호).
    fn hadout_batch(
        &mut self,
        src: CUdeviceptr,
        svh: CUdeviceptr,
        dst: CUdeviceptr,
        n: usize,
        t_len: usize,
        nseg: usize,
    ) -> Result<(), String> {
        let f_ho = self.cc.function("exl3_had_out")?;
        let (mut nch, mut nsg, mut nst) = ((n / 128) as i32, nseg as i32, n as i32);
        let (mut c0, mut c1, mut c2) = (src, svh, dst);
        let mut args_ho: [*mut std::ffi::c_void; 6] = [
            (&mut c0) as *mut _ as *mut _,
            (&mut c1) as *mut _ as *mut _,
            (&mut c2) as *mut _ as *mut _,
            (&mut nch) as *mut _ as *mut _,
            (&mut nsg) as *mut _ as *mut _,
            (&mut nst) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f_ho, (n / 128) as u32, t_len as u32, 128, &mut args_ho)
    }

    /// kseg GEMM 발사 — grid (n/64, kseg=8), 블록 128. 부분합 dbat
    /// [t][kseg][n](had_out nseg=kseg이 합산 — 결함 18호 소형-T 점유).
    fn gemm2_launch_kseg(&mut self, l: &CudaLin, t_len: usize) -> Result<(), String> {
        let f = self.cc.function("exl3_gemm2_kseg")?;
        let (mut kt, mut nt, mut kk, mut tt, mut ks) = (
            (l.k / 16) as i32,
            (l.n / 16) as i32,
            l.krate as i32,
            t_len as i32,
            GEMM2_KSEG as i32,
        );
        let (mut g0, mut g1, mut g2) = (self.daht, l.tre, self.dbat);
        let mut args: [*mut std::ffi::c_void; 8] = [
            (&mut g0) as *mut _ as *mut _,
            (&mut g1) as *mut _ as *mut _,
            (&mut g2) as *mut _ as *mut _,
            (&mut kt) as *mut _ as *mut _,
            (&mut nt) as *mut _ as *mut _,
            (&mut kk) as *mut _ as *mut _,
            (&mut tt) as *mut _ as *mut _,
            (&mut ks) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, (l.n / 64) as u32, GEMM2_KSEG as u32, 128, &mut args)
    }

    /// plain GEMM 발사 — grid (n/64, ceil(T/32)), 블록 128. H도메인
    /// 단일 출력 dyt [t][n](had_out nseg=1 제자리 후처리).
    fn gemm2_launch_plain(&mut self, l: &CudaLin, t_len: usize) -> Result<(), String> {
        let f = self.cc.function("exl3_gemm2")?;
        let (mut kt, mut nt, mut kk, mut tt) = (
            (l.k / 16) as i32,
            (l.n / 16) as i32,
            l.krate as i32,
            t_len as i32,
        );
        let (mut g0, mut g1, mut g2) = (self.daht, l.tre, self.dyt);
        let mut args: [*mut std::ffi::c_void; 7] = [
            (&mut g0) as *mut _ as *mut _,
            (&mut g1) as *mut _ as *mut _,
            (&mut g2) as *mut _ as *mut _,
            (&mut kt) as *mut _ as *mut _,
            (&mut nt) as *mut _ as *mut _,
            (&mut kk) as *mut _ as *mut _,
            (&mut tt) as *mut _ as *mut _,
        ];
        self.cc.launch(
            f,
            (l.n / 64) as u32,
            t_len.div_ceil(GEMM2_T_TILE) as u32,
            128,
            &mut args,
        )
    }

    /// 배치 GEMM 체인 내부: dx [t][k] 가정 → had16_batch → gemm2(kseg
    /// 게이트: t≤8 && n≤17408 — hip과 동일) → had_out 정확 1회 → dyt.
    /// double_had_out=true는 검증층 음성대조 계기 전용(결함 15호 재현).
    fn gemm2_chain_dev(
        &mut self,
        l: &CudaLin,
        t_len: usize,
        double_had_out: bool,
    ) -> Result<(), String> {
        if t_len == 0 {
            return Err("gemm2: t_len=0".into());
        }
        self.ensure_gemm2_bufs(l.k, l.n, t_len)?;
        self.had16_batch(l.k, t_len, l.suh)?;
        if t_len <= GEMM2_KSEG_MAX_T && l.n <= GEMM2_KSEG_MAX_N {
            self.gemm2_launch_kseg(l, t_len)?;
            self.hadout_batch(self.dbat, l.svh, self.dyt, l.n, t_len, GEMM2_KSEG)?;
        } else {
            self.gemm2_launch_plain(l, t_len)?;
            self.hadout_batch(self.dyt, l.svh, self.dyt, l.n, t_len, 1)?;
        }
        if double_had_out {
            // 음성대조: had_out 2회 적용(결함 15호) — 정상 호출 금지.
            self.hadout_batch(self.dyt, l.svh, self.dyt, l.n, t_len, 1)?;
        }
        Ok(())
    }

    /// 배치 GEMM 호스트 진입(hip gemm2 소비 형상): rows [t][k] f32 →
    /// y [t][n] f32(had_out 포함 — 자연 도메인). t 상한은 버퍼 확장이
    /// 처리(plain 경로 grid.y가 ceil(T/32)로 분할).
    pub fn gemm2_host(&mut self, key: &str, rows: &[f32]) -> Result<Vec<f32>, String> {
        let l = self.lin_copy(key)?;
        if rows.is_empty() || rows.len() % l.k != 0 {
            return Err(format!(
                "gemm2: rows.len={} k={} — [t][k] 계약 위반",
                rows.len(),
                l.k
            ));
        }
        let t_len = rows.len() / l.k;
        self.ensure_gemm2_bufs(l.k, l.n, t_len)?;
        // SAFETY: rows는 f32 슬라이스 — 바이트 뷰 변환.
        let xb = unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, rows.len() * 4) };
        self.cc.h2d(self.dx, xb)?;
        self.gemm2_chain_dev(&l, t_len, false)?;
        let mut ob = vec![0u8; t_len * l.n * 4];
        self.cc.d2h(&mut ob, self.dyt)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석(길이·정렬 일치).
        let y: &[f32] =
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, t_len * l.n) };
        Ok(y.to_vec())
    }

    /// 음성대조 계기(had_out 2회 적용 — 결함 15호) — 검증층 전용 API.
    /// 정상 경로는 gemm2_host(had_out 항상 1회).
    pub fn gemm2_host_double_had_out(
        &mut self,
        key: &str,
        rows: &[f32],
    ) -> Result<Vec<f32>, String> {
        let l = self.lin_copy(key)?;
        if rows.is_empty() || rows.len() % l.k != 0 {
            return Err(format!(
                "gemm2: rows.len={} k={} — [t][k] 계약 위반",
                rows.len(),
                l.k
            ));
        }
        let t_len = rows.len() / l.k;
        self.ensure_gemm2_bufs(l.k, l.n, t_len)?;
        // SAFETY: rows는 f32 슬라이스 — 바이트 뷰 변환.
        let xb = unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, rows.len() * 4) };
        self.cc.h2d(self.dx, xb)?;
        self.gemm2_chain_dev(&l, t_len, true)?;
        let mut ob = vec![0u8; t_len * l.n * 4];
        self.cc.d2h(&mut ob, self.dyt)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석(길이·정렬 일치).
        let y: &[f32] =
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, t_len * l.n) };
        Ok(y.to_vec())
    }

    /// 검증층 진단 — 단계별 산출 판독(G2 debug_run_stages 미러):
    /// stages=1: had_in만(f16 [t][k] 바이트 → out_ah). stages=2:
    /// + gemm2 H도메인 s(kseg 부분합은 nseg 순서로 호스트 합산 —
    /// had_out 적용 전 값, 결함 국소화 계기·원장 17호).
    pub fn debug_gemm2_stages(
        &mut self,
        key: &str,
        rows: &[f32],
        stages: u32,
        out_ah: &mut Vec<u8>,
        out_s: &mut Vec<f32>,
    ) -> Result<(), String> {
        if stages == 0 || stages > 2 {
            return Err(format!("debug_gemm2_stages: stages={stages} — 1 또는 2"));
        }
        let l = self.lin_copy(key)?;
        if rows.is_empty() || rows.len() % l.k != 0 {
            return Err(format!(
                "gemm2: rows.len={} k={} — [t][k] 계약 위반",
                rows.len(),
                l.k
            ));
        }
        let t_len = rows.len() / l.k;
        self.ensure_gemm2_bufs(l.k, l.n, t_len)?;
        // SAFETY: rows는 f32 슬라이스 — 바이트 뷰 변환.
        let xb = unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, rows.len() * 4) };
        self.cc.h2d(self.dx, xb)?;
        self.had16_batch(l.k, t_len, l.suh)?;
        if stages == 1 {
            out_ah.clear();
            out_ah.resize(t_len * l.k * 2, 0);
            self.cc.d2h(out_ah, self.daht)?;
            self.cc.sync()?;
            return Ok(());
        }
        let kseg_path = t_len <= GEMM2_KSEG_MAX_T && l.n <= GEMM2_KSEG_MAX_N;
        if kseg_path {
            self.gemm2_launch_kseg(&l, t_len)?;
            let mut sb = vec![0u8; t_len * GEMM2_KSEG * l.n * 4];
            self.cc.d2h(&mut sb, self.dbat)?;
            self.cc.sync()?;
            // SAFETY: d2h 완료 후 재해석 — nseg 합산은 had_out과 동일 순서.
            let s: &[f32] = unsafe {
                std::slice::from_raw_parts(sb.as_ptr() as *const f32, t_len * GEMM2_KSEG * l.n)
            };
            out_s.clear();
            out_s.resize(t_len * l.n, 0.0);
            for t in 0..t_len {
                for g in 0..GEMM2_KSEG {
                    for j in 0..l.n {
                        out_s[t * l.n + j] += s[(t * GEMM2_KSEG + g) * l.n + j];
                    }
                }
            }
        } else {
            self.gemm2_launch_plain(&l, t_len)?;
            let mut ob = vec![0u8; t_len * l.n * 4];
            self.cc.d2h(&mut ob, self.dyt)?;
            self.cc.sync()?;
            // SAFETY: d2h 완료 후 재해석.
            let s: &[f32] =
                unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, t_len * l.n) };
            out_s.clear();
            out_s.extend_from_slice(s);
        }
        Ok(())
    }
}
