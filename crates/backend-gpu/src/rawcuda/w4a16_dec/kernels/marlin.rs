use super::super::*;

/// [2026-10-18] 마린 인스턴스 심볼 표 — assets/marlin.cu 채리픽 13종과 1:1
/// (threads, thread_m_blocks, thread_n_blocks(=N/16), thread_k_blocks(=K/16)).
/// 맹글명은 툴체인·시그니처 계약 고정 — 추출 = cuobjdump -symbols.
fn marlin_sym(threads: u32, m: usize, tn: usize, tk: usize) -> Option<&'static str> {
    Some(match (threads, m, tn, tk) {
        (128, 1, 4, 8) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi4ELi8ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (128, 1, 8, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi8ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (128, 2, 4, 8) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi2ELi4ELi8ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (128, 2, 8, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi2ELi8ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (128, 3, 4, 8) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi3ELi4ELi8ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (128, 3, 8, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi3ELi8ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (128, 4, 4, 8) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi4ELi4ELi8ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (128, 4, 8, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi4ELi8ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (256, 1, 16, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi1ELi16ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (256, 1, 8, 8) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi1ELi8ELi8ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (256, 2, 16, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi2ELi16ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (256, 3, 16, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi3ELi16ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (256, 4, 16, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi4ELi16ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        _ => return None,
    })
}

/// [2026-10-19] g32(group_blocks=2) 판 — assets/marlin_g32.cu 채리픽 13종과 1:1.
/// gemma4 dense(g32·bf16) t>16 경로 소비(맹글명은 툴체인 계약 — 재추출 절차
/// = marlin_g32.cu 빌드 후 cuobjdump -symbols).
fn marlin_g32_sym(threads: u32, m: usize, tn: usize, tk: usize) -> Option<&'static str> {
    Some(match (threads, m, tn, tk) {
        (128, 1, 4, 8) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (128, 1, 8, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (128, 2, 4, 8) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi2ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (128, 2, 8, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi2ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (128, 3, 4, 8) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi3ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (128, 3, 8, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi3ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (128, 4, 4, 8) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi4ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (128, 4, 8, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi4ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (256, 1, 16, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi1ELi16ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (256, 1, 8, 8) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi1ELi8ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (256, 2, 16, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi2ELi16ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (256, 3, 16, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi3ELi16ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        (256, 4, 16, 4) => {
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi4ELi16ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
        }
        _ => return None,
    })
}

impl W4a16Dec {
    /// [2026-10-18] vLLM 마린 GEMM 시도(27B dense t≥16 g128 · [2026-10-19]
    /// gemma4 dense t>16 g32) — 벤더 assets/marlin/(Apache-2.0). 재패킹
    /// (lins_mk)·스케일 순열(lins_msc) 사본이 있을 때만 동작 — 부재 시
    /// Ok(false)로 기존 경로 폴백(운영 계약: 사본 부재 = 용량/형상 계약
    /// 미충족일 때만). C = f16 스크래치 → `w4a16_h2f`로 f32(y_out).
    /// M 분할·타일 픽 = vLLM marlin_mm 휴리스틱 이식(max_par 16/128, m블록 4캡).
    /// group = 128(f16·27B, marlin.fatbin) | 32(bf16·gemma4, marlin_g32.fatbin).
    pub(in crate::rawcuda::w4a16_dec) fn marlin_gemm_try(
        &mut self,
        name: &str,
        t: usize,
        n: usize,
        k: usize,
        x16: CUdeviceptr,
        y_out: CUdeviceptr,
        group: usize,
    ) -> Result<bool, String> {
        let Some(&(bq, _, _)) = self.lins_mk.get(name) else {
            return Ok(false);
        };
        let Some(&(dsc, _, _)) = self.lins_msc.get(name) else {
            return Ok(false);
        };
        if t == 0 || !k.is_multiple_of(128) || !n.is_multiple_of(64) {
            return Ok(false);
        }
        let smem = self.cc.smem_optin_max().min(224 * 1024);
        if smem < 64 * 1024 {
            return Ok(false);
        }
        let need = t * n * 2;
        if need > self.dmc.cap {
            self.dmc = self.realloc_buf(self.dmc, need, need, "dmc_cap")?;
        }
        let c16 = self.dmc.ptr;
        // locks = sms×4 int32(0 초기화) — 커널이 스스로 리셋(재사용 계약).
        if self.dmlock == 0 {
            let bytes = self.cc.sm_count().max(1) as usize * 16;
            let p = self.cc.alloc(bytes)?;
            let z = vec![0u8; bytes];
            if let Err(e) = Self::h2d_chunked(&self.cc, p, &z) {
                let _ = self.cc.free(p);
                return Err(e);
            }
            self.dmlock = p;
        }
        let max_par = if n <= 4096 { 128usize } else { 16 };
        let (mut a_off, mut c_off, mut rest) = (0usize, 0usize, t);
        while rest > 0 {
            let par = (rest / 64).min(max_par);
            let m_split = if par > 0 { par * 64 } else { rest };
            let thread_m = m_split.div_ceil(16).min(4);
            let cands: &[(u32, usize, usize)] = if thread_m > 1 {
                &[(256, 16, 4), (128, 8, 4), (128, 4, 8)]
            } else {
                &[(256, 8, 8), (128, 8, 4), (128, 4, 8)]
            };
            let mut pick = None;
            for &(th, tn, tk) in cands {
                if n.is_multiple_of(tn * 16)
                    && k.is_multiple_of(tk * 16)
                    && let Some(sym) = if group == 32 {
                        marlin_g32_sym(th, thread_m, tn, tk)
                    } else {
                        marlin_sym(th, thread_m, tn, tk)
                    }
                {
                    pick = Some((th, sym));
                    break;
                }
            }
            let Some((th, sym)) = pick else {
                return Ok(false);
            };
            let f = self.cc.function(sym)?;
            self.cc.set_dynamic_smem(f, smem)?;
            let (mut p_a, mut p_b) = (x16 + (a_off * 2) as u64, bq);
            let (mut p_c, mut p_ct) = (c16 + (c_off * 2) as u64, 0u64);
            let (mut p_bias, mut p_as) = (0u64, 0u64);
            let (mut p_s, mut p_gs, mut p_zp) = (dsc, 0u64, 0u64);
            let (mut p_m, mut p_n, mut p_k) = (m_split as i32, n as i32, k as i32);
            let (mut p_lda, mut p_locks) = (k as i32, self.dmlock);
            let (mut p_hb, mut p_ua, mut p_fr) = (0i32, 0i32, 0i32);
            let mut p_ms = smem as i32;
            self.cc.launch_shared(
                f,
                self.cc.sm_count().max(1),
                1,
                th,
                smem,
                &mut crate::rawcuda::args::l18(
                    &mut p_a,
                    &mut p_b,
                    &mut p_c,
                    &mut p_ct,
                    &mut p_bias,
                    &mut p_as,
                    &mut p_s,
                    &mut p_gs,
                    &mut p_zp,
                    &mut p_m,
                    &mut p_n,
                    &mut p_k,
                    &mut p_lda,
                    &mut p_locks,
                    &mut p_hb,
                    &mut p_ua,
                    &mut p_fr,
                    &mut p_ms,
                ),
            )?;
            a_off += m_split * k;
            c_off += m_split * n;
            rest -= m_split;
        }
        // f16 C → f32(y_out) — 엔진 스트림 dtype 계약.
        let fh = self.cc.function("w4a16_h2f")?;
        let nn2 = t * n / 2;
        let mut n2 = nn2 as i32;
        let (mut q_i, mut q_o) = (c16, y_out);
        self.cc.launch(
            fh,
            nn2.div_ceil(256) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l3(&mut q_i, &mut q_o, &mut n2),
        )?;
        Ok(true)
    }
}
