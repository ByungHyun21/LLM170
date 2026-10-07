//! EXL3 ew(silu·mul)·argmax 모듈층(plans/124 G7, G10 파일 분할
//! 2026-10-04). Exl3CudaDecoder의 ew·argmax 임플 블록 — 컨텍스트·fatbin
//! 리졸버는 exl3_cuda.rs 공유 글루.
//!
//! [용도] ew: y = silu(g)·u 원소별(hip FFN 발사 미러, exl3_hip.rs
//! L786-796 — 그리드 (ceil(n/128),1)·블록 128). argmax: [n] 로짓 → 최대
//! 토큰 1개(단일 블록 1024스레드, n은 로짓 길이 248320 — 결함 8호:
//! 행수 아님). 산술 계약은 assets/exl3_ew.cu(src_exl3.hip 1:1 직이식,
//! -fmad=false 빌드 — exp 트윈이 f64 DAG).
//!
//! [정합 — plans/129-cuda C2 원장, sm_89 실측 2026-10-04] ew (i) 27B
//! n=17408: maxdiff 0.000e0 nan=0(오라클과 비트동일 — 존 |g|≥10 5804개 ·
//! |g|≤1e-3 2904개) · (ii) 35B n=512: 0.000e0(존 172/85) — 임계 1e-6.
//! argmax (iii-a) 최대@17+근접타이: 17 exact · (iii-b) 최대@248318:
//! 248318 exact · (iii-c) 동일값 타이 1·1024: 1024(트리 규칙 — 낮은 tid
//! 클래스 내 첫 등장 우승). 음성대조(결함 8호): (a) n=5120 → token 4321 ·
//! (b) n=1 → token 0 → NEG-DETECTED(비영 exit + 마커).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). sm_80 자원
//! 증거: exl3_ew REG:16 SHARED:0(순수 스트리밍) · exl3_argmax REG:31
//! SHARED:8192(단일 블록 1024스레드=32와프).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지 — scripts/cuda_probe_shim.rs 단독 컴파일.

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use crate::rawcuda::ffi::CUdeviceptr;

impl Exl3CudaDecoder {
    // ── ew(silu·mul)·argmax(G7 — plans/124 §1, rawhip FFN 발사 미러) ──

    /// ew 작업 버퍼 보장(dewg·dewu·dew를 n원소로 — 확장 시에만 재할당).
    fn ensure_ew_bufs(&mut self, n: usize) -> Result<(), String> {
        if n > self.ew_cap {
            if self.ew_cap > 0 {
                self.cc.free(self.dewg)?;
                self.cc.free(self.dewu)?;
                self.cc.free(self.dew)?;
            }
            self.dewg = self.cc.alloc(n * 4)?;
            self.dewu = self.cc.alloc(n * 4)?;
            self.dew = self.cc.alloc(n * 4)?;
            self.ew_cap = n;
        }
        Ok(())
    }

    /// argmax 출력 버퍼 보장(dargmax 4B — 최초 1회).
    fn ensure_argmax_buf(&mut self) -> Result<(), String> {
        if self.dargmax == 0 {
            self.dargmax = self.cc.alloc(4)?;
        }
        Ok(())
    }

    /// ew 디바이스 상주 1회(hip FFN 발사 미러, exl3_hip.rs L786-796):
    /// y = silu(g)·u, 그리드 (ceil(n/128),1) · 블록 128. 버퍼는 호출자
    /// 소유(디코드 체인 조립은 dsb·dsb2·dew — 후속 목표).
    pub fn ew_dev(
        &mut self,
        g_dev: CUdeviceptr,
        u_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        n: usize,
    ) -> Result<(), String> {
        if n == 0 || n > i32::MAX as usize {
            return Err(format!("ew: n={n} — (0, 2^31) 도메인 계약 위반"));
        }
        let f = self.cc.function("exl3_ew")?;
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

    /// 호스트 래퍼(검증층 기본 진입 — norm_resid_host 노선): g·u 업로드
    /// → ew → y 판독.
    pub fn ew_host(&mut self, g: &[f32], u: &[f32]) -> Result<Vec<f32>, String> {
        if g.is_empty() || g.len() != u.len() {
            return Err(format!(
                "ew: g.len={} u.len={} — 동일 길이(비어있지 않은) 계약 위반",
                g.len(),
                u.len()
            ));
        }
        self.ensure_ew_bufs(g.len())?;
        // SAFETY: g·u는 f32 슬라이스 — 길이 일치 바이트 뷰 변환.
        let gb = unsafe { std::slice::from_raw_parts(g.as_ptr() as *const u8, g.len() * 4) };
        let ub = unsafe { std::slice::from_raw_parts(u.as_ptr() as *const u8, u.len() * 4) };
        self.cc.h2d(self.dewg, gb)?;
        self.cc.h2d(self.dewu, ub)?;
        self.ew_dev(self.dewg, self.dewu, self.dew, g.len())?;
        let mut yb = vec![0u8; g.len() * 4];
        self.cc.d2h(&mut yb, self.dew)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
        Ok(unsafe { std::slice::from_raw_parts(yb.as_ptr() as *const f32, g.len()) }.to_vec())
    }

    /// argmax 디바이스 상주 1회(hip step_tok L493-511·FFN 꼬리 L811-824
    /// 미러): [n] 로짓 → 최대 토큰 1개. n은 "로짓 길이"(248320 — 결함
    /// 8호: 행수 아님). 단일 블록 1024스레드(원본 계약, n≤1M).
    pub fn argmax_dev(&mut self, lg_dev: CUdeviceptr, n: usize) -> Result<u32, String> {
        if n == 0 || n > 1 << 20 {
            return Err(format!(
                "argmax: n={n} — (0, 1M] 도메인 계약(단일 블록 리덕션)"
            ));
        }
        self.ensure_argmax_buf()?;
        let f = self.cc.function("exl3_argmax")?;
        let mut nn = n as i32;
        let (mut a0, mut a1) = (lg_dev, self.dargmax);
        let mut args: [*mut std::ffi::c_void; 3] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        self.cc.launch(f, 1, 1, 1024, &mut args)?;
        let mut ob = vec![0u8; 4];
        self.cc.d2h(&mut ob, self.dargmax)?;
        self.cc.sync()?;
        Ok(u32::from_le_bytes([ob[0], ob[1], ob[2], ob[3]]))
    }

    /// 호스트 래퍼(검증층 진입 + 결함 8호 음성대조의 n 주입구): 로짓
    /// 업로드 → argmax(n) → 토큰. n은 스캔 길이(≤ logits.len()).
    /// 프로덕션 호출은 n = 로짓 길이(logits.len() = 248320) — 잘못된
    /// n(행수 혼동)의 발사는 프로브 음성대조뿐(원장 17호 계기 원칙).
    pub fn argmax_host_n(&mut self, logits: &[f32], n: usize) -> Result<u32, String> {
        if n == 0 || n > logits.len() {
            return Err(format!(
                "argmax: n={n} > logits.len={} — 스캔 길이 계약 위반",
                logits.len()
            ));
        }
        let need = logits.len() * 4;
        if self.dlgmax == 0 || logits.len() > self.lg_cap {
            if self.dlgmax != 0 {
                self.cc.free(self.dlgmax)?;
            }
            self.dlgmax = self.cc.alloc(need)?;
            self.lg_cap = logits.len();
        }
        // SAFETY: logits은 f32 슬라이스 — 바이트 뷰 변환. 대형 h2d는
        // 4MB 청크 분할(페이지 미매핑 가드 — hip 원장 2026-10-04).
        let lb = unsafe { std::slice::from_raw_parts(logits.as_ptr() as *const u8, need) };
        Self::h2d_chunked(&self.cc, self.dlgmax, lb)?;
        self.argmax_dev(self.dlgmax, n)
    }

    /// 호스트 래퍼(프로덕션 계약 진입 — n = 로짓 길이, 결함 8호).
    pub fn argmax_host(&mut self, logits: &[f32]) -> Result<u32, String> {
        self.argmax_host_n(logits, logits.len())
    }
}
