//! EXL3 GEMV 체인 모듈층 — had_in→gemv→had_out 순차 선형(plans/124 G2,
//! G10 파일 분할 2026-10-04). Exl3CudaDecoder의 GEMV 임플 블록 —
//! 레지스트리·컨텍스트·fatbin 리졸버는 exl3_cuda.rs 공유 글루.
//!
//! [용도] 트렐리스 선형 1행(t=1) 실행: had_in(x·suh의 WHT + f16팩) →
//! exl3_gemv(H도메인 [nseg=16][n] 분할 부분합) → had_out(nseg 합산 +
//! WHT⁻¹·0.0884·svh — 생략 시 T=1은 argmax가 우연히 맞아도 전체가
//! 붕괴, 결함 3호). sb는 [nseg=16][n] 분할 부분합 → had_out이 nseg 합산.
//!
//! [정합 — plans/129-cuda C2 원장, RTX 4070 SUPER(sm_89) 검증 호스트
//! 실측 2026-10-04] 합성 (i) 27B gate_proj 형상: 1.623e-4 · (ii) 35B qkv
//! 형상: 1.021e-4 · (iiia/iiib) 동일 k 이중 선형: 1.414e-4/1.511e-4 ·
//! 실가중 27B gate_proj(load_keys): 2.216e-4 — 전 항목 임계 3e-4 이내
//! (plans/124 §1, 결함 20호 수정 후 값). 음성대조 2.844e-1 →
//! NEG-DETECTED.
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). hip 8060S
//! 참고치 117 GB/s(트렐리스 디코드 ALU 병목); sm_80 실측은 도착 후 기입.
//!
//! [이식 목표치 — plans/124 §1, hip 8060S 측정 원장 기준 참고]
//!   GEMV 체인(had_in→gemv→had_out): maxdiff ≤3e-4 (hip lm_head 2.664e-4)
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지 — scripts/cuda_probe_shim.rs 단독 컴파일.

use crate::rawcuda::exl3_cuda::{CudaLin, Exl3CudaDecoder, GEMV_NSEG};
use crate::rawcuda::ffi::CUdeviceptr;

impl Exl3CudaDecoder {
    /// 체인 진행 **전** 모든 선형의 k·n으로 작업 버퍼를 한 번에 확보한다.
    ///
    /// [S10 필수] 이게 없으면 진행 중 `ensure_bufs`가 nmax/kmax를 키워
    /// `dyb`·`dah`·`dsb`를 해제·재할당한다. 이미 잡아 둔 이전 GEMV 결과
    /// 포인터(dyb)가 dangling이 되어 다음 층이 엉뚱한 값을 읽는다 —
    /// 증상은 "어떤 층부터 hidden이 크게 벌어진다"이고, 실제로 16층에서
    /// hidden maxdiff 73.5로 재현됐다(원장 S10).
    pub fn prewarm_chain_bufs(&mut self) -> Result<(), String> {
        let mut kmax = 0usize;
        let mut nmax = 0usize;
        for l in self.lin.values() {
            kmax = kmax.max(l.k);
            nmax = nmax.max(l.n);
        }
        self.ensure_bufs(kmax, nmax)
    }

    /// GEMV 체인 작업 버퍼 보장(kmax/nmax 확장 시에만 재할당).
    fn ensure_bufs(&mut self, k: usize, n: usize) -> Result<(), String> {
        self.ensure_x(k)?;
        if k > self.kmax {
            if self.dah != 0 {
                self.cc.free(self.dah)?;
            }
            self.dah = self.cc.alloc(k * 2)?;
            self.kmax = k;
        }
        if n > self.nmax {
            if self.dsb != 0 {
                self.cc.free(self.dsb)?;
                self.cc.free(self.dyb)?;
            }
            self.dsb = self.cc.alloc(GEMV_NSEG * n * 4)?;
            self.dyb = self.cc.alloc(n * 4)?;
            self.nmax = n;
        }
        Ok(())
    }

    /// GEMV 체인 1회(had_in → gemv → had_out) — Exl3HipDecoder::gemv_chain
    /// 미러. 입력은 디바이스 상주 포인터(x_dev), 출력은 self.dyb에 남는다.
    /// had_out은 항상 1회(결함 3·15호).
    ///
    /// [S10 디바이스 체인] x_dev가 이미 디바이스에 있으면 호스트 왕복 없이
    /// 그대로 had_in을 건다. had_out=false는 검증층 음성대조 계기 전용이며
    /// H도메인 부분합만 self.dsb에 남긴다(호스트 판독은 호출자 몫).
    fn gemv_chain_dev(
        &mut self,
        l: &CudaLin,
        x_dev: CUdeviceptr,
        had_out: bool,
    ) -> Result<(), String> {
        self.ensure_bufs(l.k, l.n)?;
        let f_hin = self.cc.function("exl3_had_in")?;
        let f_gv = self.cc.function("exl3_gemv")?;
        let f_hout = self.cc.function("exl3_had_out")?;

        // had_in: 그리드 (k/128, T=1), 블록 128.
        let (mut a0, mut a1, mut a2) = (x_dev, l.suh, self.dah);
        let (mut kc, mut ks) = ((l.k / 128) as i32, l.k as i32);
        let mut args_hin: [*mut std::ffi::c_void; 5] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut kc) as *mut _ as *mut _,
            (&mut ks) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f_hin, (l.k / 128) as u32, 1, 128, &mut args_hin)?;

        // gemv: 그리드 ((n/16)/8, nseg=16), 블록 128 — sb는 [nseg][n].
        let (mut b0, mut b1, mut b2) = (self.dah, l.tre, self.dsb);
        let (mut kt, mut nt, mut kk) = ((l.k / 16) as i32, (l.n / 16) as i32, l.krate as i32);
        let mut args_gv: [*mut std::ffi::c_void; 6] = [
            (&mut b0) as *mut _ as *mut _,
            (&mut b1) as *mut _ as *mut _,
            (&mut b2) as *mut _ as *mut _,
            (&mut kt) as *mut _ as *mut _,
            (&mut nt) as *mut _ as *mut _,
            (&mut kk) as *mut _ as *mut _,
        ];
        self.cc.launch(
            f_gv,
            ((l.n / 16) / 8) as u32,
            GEMV_NSEG as u32,
            128,
            &mut args_gv,
        )?;
        if !had_out {
            // 음성대조: H도메인 부분합만 남긴다(had_out 생략 — 결함 3호 재현).
            return Ok(());
        }
        // had_out: 그리드 (n/128, 1), 블록 128 — nseg 합산 + WHT⁻¹·R·svh.
        let (mut c0, mut c1, mut c2) = (self.dsb, l.svh, self.dyb);
        let (mut nch, mut nsg, mut nst) = ((l.n / 128) as i32, GEMV_NSEG as i32, l.n as i32);
        let mut args_hout: [*mut std::ffi::c_void; 6] = [
            (&mut c0) as *mut _ as *mut _,
            (&mut c1) as *mut _ as *mut _,
            (&mut c2) as *mut _ as *mut _,
            (&mut nch) as *mut _ as *mut _,
            (&mut nsg) as *mut _ as *mut _,
            (&mut nst) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f_hout, (l.n / 128) as u32, 1, 128, &mut args_hout)
    }

    /// 디바이스 상주 입력 → GEMV 체인 → 결과를 dyb에 남긴다(호스트 왕복 0).
    /// 반환 포인터는 self.dyb(다음 GEMV가 재사용하므로 즉시 소비할 것).
    /// S10 성능 캠페인 진입점 — 산술은 gemv_host와 동일 경로다.
    pub fn gemv_dev(&mut self, key: &str, x_dev: CUdeviceptr) -> Result<CUdeviceptr, String> {
        let l = self.lin_copy(key)?;
        if x_dev == 0 {
            return Err(format!("gemv_dev({key}): 입력 포인터 0"));
        }
        self.gemv_chain_dev(&l, x_dev, true)?;
        Ok(self.dyb)
    }

    /// 호스트 벡터 → GEMV 체인 1회 → 호스트 결과(hip gemv_host 미러).
    pub fn gemv_host(&mut self, key: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let l = self.lin_copy(key)?;
        if x.len() != l.k {
            return Err(format!("gemv: x.len={} != k={}", x.len(), l.k));
        }
        self.ensure_bufs(l.k, l.n)?;
        // SAFETY: x는 길이 k*4 바이트의 f32 슬라이스 — 바이트 뷰 변환.
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        self.cc.h2d(self.dx, xb)?;
        self.gemv_chain_dev(&l, self.dx, true)?;
        let mut ob = vec![0u8; l.n * 4];
        self.cc.d2h(&mut ob, self.dyb)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let y: &[f32] = unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, l.n) };
        Ok(y.to_vec())
    }

    /// 검증층 진단 — 단계별 산출 판독(3층 분리: 호출은 검증층, 상태는
    /// 모듈층 소유). stages=1: had_in만(f16쌍팩 [k/2]u32 → out_ah).
    /// stages=2: + gemv([nseg][n] 부분합 → out_sb). 계산 경로 자체는
    /// gemv_chain_opt와 동일(ENV 분기 아님).
    pub fn debug_run_stages(
        &mut self,
        key: &str,
        x: &[f32],
        stages: u32,
        out_ah: &mut Vec<u8>,
        out_sb: &mut Vec<f32>,
    ) -> Result<(), String> {
        if stages == 0 || stages > 2 {
            return Err(format!("debug_run_stages: stages={stages} — 1 또는 2"));
        }
        let l = self.lin_copy(key)?;
        if x.len() != l.k {
            return Err(format!("x.len={} != k={}", x.len(), l.k));
        }
        self.ensure_bufs(l.k, l.n)?;
        // SAFETY: x는 길이 k*4 바이트의 f32 슬라이스 — 바이트 뷰 변환.
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        self.cc.h2d(self.dx, xb)?;
        let f_hin = self.cc.function("exl3_had_in")?;
        let (mut a0, mut a1, mut a2) = (self.dx, l.suh, self.dah);
        let (mut kc, mut ks) = ((l.k / 128) as i32, l.k as i32);
        let mut args_hin: [*mut std::ffi::c_void; 5] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut kc) as *mut _ as *mut _,
            (&mut ks) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f_hin, (l.k / 128) as u32, 1, 128, &mut args_hin)?;
        if stages == 1 {
            out_ah.clear();
            out_ah.resize(l.k * 2, 0);
            self.cc.d2h(out_ah, self.dah)?;
            self.cc.sync()?;
            return Ok(());
        }
        let f_gv = self.cc.function("exl3_gemv")?;
        let (mut b0, mut b1, mut b2) = (self.dah, l.tre, self.dsb);
        let (mut kt, mut nt, mut kk) = ((l.k / 16) as i32, (l.n / 16) as i32, l.krate as i32);
        let mut args_gv: [*mut std::ffi::c_void; 6] = [
            (&mut b0) as *mut _ as *mut _,
            (&mut b1) as *mut _ as *mut _,
            (&mut b2) as *mut _ as *mut _,
            (&mut kt) as *mut _ as *mut _,
            (&mut nt) as *mut _ as *mut _,
            (&mut kk) as *mut _ as *mut _,
        ];
        self.cc.launch(
            f_gv,
            ((l.n / 16) / 8) as u32,
            GEMV_NSEG as u32,
            128,
            &mut args_gv,
        )?;
        let mut sb = vec![0u8; GEMV_NSEG * l.n * 4];
        self.cc.d2h(&mut sb, self.dsb)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let s: &[f32] =
            unsafe { std::slice::from_raw_parts(sb.as_ptr() as *const f32, GEMV_NSEG * l.n) };
        out_sb.clear();
        out_sb.extend_from_slice(s);
        Ok(())
    }

    /// 음성대조 계기(had_out 생략 경로) — 검증층 전용 API. 정상 경로는
    /// gemv_host(had_out 항상 1회).
    pub fn gemv_host_skip_had_out(&mut self, key: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let l = self.lin_copy(key)?;
        if x.len() != l.k {
            return Err(format!("x.len={} != k={}", x.len(), l.k));
        }
        self.ensure_bufs(l.k, l.n)?;
        // SAFETY: x는 길이 k*4 바이트의 f32 슬라이스 — 바이트 뷰 변환.
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        self.cc.h2d(self.dx, xb)?;
        self.gemv_chain_dev(&l, self.dx, false)?;
        let mut sb = vec![0u8; GEMV_NSEG * l.n * 4];
        self.cc.d2h(&mut sb, self.dsb)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석 — nseg 합산은 had_out과 동일 순서.
        let s: &[f32] =
            unsafe { std::slice::from_raw_parts(sb.as_ptr() as *const f32, GEMV_NSEG * l.n) };
        let mut out = vec![0.0f32; l.n];
        for g in 0..GEMV_NSEG {
            for j in 0..l.n {
                out[j] += s[g * l.n + j];
            }
        }
        Ok(out)
    }
}
