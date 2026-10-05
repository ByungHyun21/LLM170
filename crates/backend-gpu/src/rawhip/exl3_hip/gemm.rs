//! EXL3 hip 배치 GEMM 프리미티브 — norm_p/had16/gemm2×2/hadout.
use super::{Exl3HipDecoder, HipLin};

impl Exl3HipDecoder {
    /// 배치 노름(norm_resid_p) — dbx += ab, dbxn = norm(dbx)·w. nw는 행 포인터.
    pub(super) fn norm_p(&mut self, w: usize, ab_in: *mut u8, t_len: usize) -> Result<(), String> {
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
    pub(super) fn had16_batch(
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

    /// 배치 GEMM — 소형-T(≤8)·층선형(n≤17408)은 exl3_gemv_m(m행 GEMV,
    /// plans/130): 가중치 디코드를 m행이 공유해 스트리밍이 GEMV급(단일행
    /// 117GB/s)으로 유지된다(kseg 스칼라팧은 T≤8에서 57GB/s 실측 — tbench 원장).
    /// 부분합 dbat [T][8][n] → had_out(nseg=8) 합산. 대형은 mma 경로.
    pub(super) fn gemm2_batch(
        &mut self,
        l: &HipLin,
        t_len: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        if t_len <= 8 && l.n <= 17408 {
            // 컴파일타임 T 특수화(m1..m8) — 런타임 T 판은 미사용 acc[8]로
            // 점유가 반토랑 나는 부정 판정(T=1 416ms vs kseg 286ms, plans/130).
            const GVM: [&str; 9] = [
                "",
                "exl3_gemv_m1",
                "exl3_gemv_m2",
                "exl3_gemv_m3",
                "exl3_gemv_m4",
                "exl3_gemv_m5",
                "exl3_gemv_m6",
                "exl3_gemv_m7",
                "exl3_gemv_m8",
            ];
            let (mut kt, mut nt, mut kk, mut tt) = (
                (l.k / 16) as i32,
                (l.n / 16) as i32,
                l.krate as i32,
                t_len as i32,
            );
            let (mut g0, mut g1, mut g2) = (self.dah16, l.tre, self.dbat);
            self.hc.launch3(
                GVM[t_len],
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
                    &mut tt as *mut i32 as *mut _,
                ],
            )?;
            // had_out nseg=16 합산(nseg=16 부분합 — 배포 gemv 체인 동일 구조).
            let (mut nch, mut nsg, mut nst) = ((l.n / 128) as i32, 16i32, l.n as i32);
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
    pub(super) fn gemm2_batch_plain(
        &mut self,
        l: &HipLin,
        t_len: usize,
        out: *mut u8,
    ) -> Result<(), String> {
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
    pub(super) fn hadout_batch(
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
}
