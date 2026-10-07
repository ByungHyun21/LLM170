//! EXL3 norm_resid 모듈층(plans/124 G3, G10 파일 분할 2026-10-04).
//! Exl3CudaDecoder의 노름 임플 블록 — 컨텍스트·fatbin 리졸버는
//! exl3_cuda.rs 공유 글루.
//!
//! [용도] rms_norm(x+ab)·nw[w행] eps=1e-6 + 제자리 잔차 가산 x = x + ab
//! (산술 계약 plans/124 §3.2, 결함 2·7호 가드 — 노름 w는 행 포인터:
//! nw 배열의 w·hidden 오프셋, 누락 시 전 노름 L0 행 판독).
//!
//! [정합 — plans/129-cuda C2 원장, sm_89 실측 2026-10-04] (i) 27B
//! hidden=5120 w=1/129: 4.768e-7 · (ii) 35B hidden=2048: 0.000e0 ·
//! (iii) 강분리 행 전 w + T=4: 0.000e0(코어 오라클과 비트일치) — 전 항목
//! 임계 3e-6 이내(plans/124 §1). 음성대조 (a) L0 판독: 6.613e0 ·
//! (b) eps 1e-5: 6.719e-4 → NEG-DETECTED(결함 2호·원장 17호 계기).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). sm_80 자원
//! 증거(cuobjdump, 커밋 fatbin): exl3_norm_resid REG:30 STACK:32
//! SHARED:4096(블록 1024스레드) → GA100 2블록/SM = 2048스레드 풀점유.
//!
//! [이식 목표치 — plans/124 §1] norm_resid: maxdiff ≤3e-6 (hip 2.861e-6)
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지 — scripts/cuda_probe_shim.rs 단독 컴파일.

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use crate::rawcuda::ffi::CUdeviceptr;

impl Exl3CudaDecoder {
    // ── norm_resid(G3 — plans/124 §3.2) ──

    /// 노름 가중 배열 상주 등록 — nw: f32 LE 바이트 [rows][hidden].
    /// hidden 사전 설정 필수(load 또는 검증층 지정). w 인덱스의 행
    /// 오프셋 계약(w·hidden)은 이 등록 형상에서 나온다(27B는 w·5120 —
    /// 오프셋 누락이 결함 2호: 전 노름 L0 행 판독).
    pub fn set_norm_weights(&mut self, nw: &[u8], rows: usize) -> Result<(), String> {
        if self.hidden == 0 || self.hidden % 1024 != 0 || self.hidden > 8192 {
            return Err(format!(
                "norm: hidden={} — 1024 배수·8192 이하 계약(커널 v[8] 상한)",
                self.hidden
            ));
        }
        if rows == 0 || nw.len() != rows * self.hidden * 4 {
            return Err(format!(
                "norm: nw {}B != rows {rows} × hidden {} × 4B",
                nw.len(),
                self.hidden
            ));
        }
        if self.dnw != 0 {
            self.cc.free(self.dnw)?;
        }
        let d = self.cc.alloc(nw.len())?;
        Self::h2d_chunked(&self.cc, d, nw)?;
        self.dnw = d;
        self.norm_w_rows = rows;
        Ok(())
    }

    /// 노름 작업 버퍼 보장(dx·dab·dxn을 [t_len][hidden]원소로 —
    /// 확장 시에만 재할당).
    fn ensure_norm_bufs(&mut self, t_len: usize) -> Result<(), String> {
        let need = t_len * self.hidden;
        self.ensure_x(need)?;
        if need > self.norm_cap {
            if self.norm_cap > 0 {
                self.cc.free(self.dab)?;
                self.cc.free(self.dxn)?;
            }
            self.dab = self.cc.alloc(need * 4)?;
            self.dxn = self.cc.alloc(need * 4)?;
            self.norm_cap = need;
        }
        Ok(())
    }

    /// norm_resid 디바이스 상주 1회(hip norm 미러) — xn = rms_norm(x+ab)·nw
    /// 의 w행, x = x + ab 제자리 기록(커널이 수행 — §3.2 잔차 스트림
    /// 계약). 그리드 (t_len,1), 블록 1024. w는 행 인덱스(경계 검사).
    /// 반환은 xn 포인터(판독은 호출자 — dx가 갱신된 잔차).
    pub fn norm_resid(
        &mut self,
        w: usize,
        ab_dev: CUdeviceptr,
        t_len: usize,
    ) -> Result<CUdeviceptr, String> {
        if self.dnw == 0 {
            return Err("norm: 노름 가중 미등록(set_norm_weights)".into());
        }
        if w >= self.norm_w_rows {
            return Err(format!("norm: w={w} >= rows={}", self.norm_w_rows));
        }
        self.ensure_norm_bufs(t_len)?;
        let f = self.cc.function("exl3_norm_resid")?;
        let mut tl = t_len as i32;
        let mut wo = (w * self.hidden) as i32; // 결함 2호: 행 오프셋 w·hidden
        let mut hd = self.hidden as i32;
        // 인자 순서 (x, nw, ab, xn) 고정 — 교차 시 잔차에 노름가중치가
        // 더해진다(결함 7호).
        let (mut a0, mut a1, mut a2, mut a3) = (self.dx, self.dnw, ab_dev, self.dxn);
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

    /// 호스트 래퍼(hip gemv_host 노선 — 검증층 기본 진입): x·ab 업로드 →
    /// norm_resid → (x' = x+ab, xn) 판독. t_len = x.len()/hidden.
    pub fn norm_resid_host(
        &mut self,
        w: usize,
        x: &[f32],
        ab: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if self.hidden == 0 {
            return Err("norm: hidden 미설정".into());
        }
        if x.len() != ab.len() || x.is_empty() || x.len() % self.hidden != 0 {
            return Err(format!(
                "norm: x.len={} ab.len={} hidden={} — [t][hidden] 계약 위반",
                x.len(),
                ab.len(),
                self.hidden
            ));
        }
        let t_len = x.len() / self.hidden;
        self.ensure_norm_bufs(t_len)?;
        // SAFETY: x·ab는 f32 슬라이스 — 길이 일치 바이트 뷰 변환.
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        let abb = unsafe { std::slice::from_raw_parts(ab.as_ptr() as *const u8, ab.len() * 4) };
        self.cc.h2d(self.dx, xb)?;
        self.cc.h2d(self.dab, abb)?;
        self.norm_resid(w, self.dab, t_len)?;
        let mut xob = vec![0u8; x.len() * 4];
        let mut xnb = vec![0u8; x.len() * 4];
        self.cc.d2h(&mut xob, self.dx)?;
        self.cc.d2h(&mut xnb, self.dxn)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
        let xo =
            unsafe { std::slice::from_raw_parts(xob.as_ptr() as *const f32, x.len()) }.to_vec();
        let xn =
            unsafe { std::slice::from_raw_parts(xnb.as_ptr() as *const f32, x.len()) }.to_vec();
        Ok((xo, xn))
    }
}
