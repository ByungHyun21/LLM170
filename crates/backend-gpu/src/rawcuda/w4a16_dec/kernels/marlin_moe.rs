// [2026-10-19] 스테이지② = 런처·정렬 변환 준비 — 소비(디스패치 배선)는 스테이지④.
// 그때까지 dead_code 허용(args.rs 미사용 아리티 전례).
#![allow(dead_code)]

use super::super::*;

/// [2026-10-19] marlin MoE 호스트 — reference/vllm csrc/moe/marlin_moe_wna16/
/// ops.cu 이식(타일 표·캐시 크기·유효성·allow_count). 벤더 커널과 툴체인 계약.
#[derive(Clone, Copy)]
struct MoeTh {
    k: usize,
    n: usize,
    th: u32,
}

const MOE_SMALL: [MoeTh; 3] = [
    MoeTh {
        k: 128,
        n: 128,
        th: 256,
    },
    MoeTh {
        k: 64,
        n: 128,
        th: 128,
    },
    MoeTh {
        k: 128,
        n: 64,
        th: 128,
    },
];
const MOE_LARGE: [MoeTh; 3] = [
    MoeTh {
        k: 64,
        n: 256,
        th: 256,
    },
    MoeTh {
        k: 64,
        n: 128,
        th: 128,
    },
    MoeTh {
        k: 128,
        n: 64,
        th: 128,
    },
];
const MOE_STAGES: usize = 4;
const MOE_GROUP: usize = 32; // g32(group_blocks=2)

/// 인스턴스별 REG(cuobjdump -res-usage — 고정 툴체인 계약) — allow_count 산정.
fn moe_regs(th: u32, m: usize, tn: usize, tk: usize, m8: bool) -> usize {
    match (th, m, tn, tk, m8) {
        (128, 4, 4, 8, false) => 255,
        (128, 3, 4, 8, false) => 250,
        (128, 2, 4, 8, false) => 216,
        (128, 1, 4, 8, false) => 156,
        (128, 1, 4, 8, true) => 121,
        (128, 4, 8, 4, false) => 255,
        (128, 3, 8, 4, false) => 244,
        (128, 2, 8, 4, false) => 218,
        (128, 1, 8, 4, false) => 144,
        (128, 1, 8, 4, true) => 121,
        (256, 4, 16, 4, false) => 255,
        (256, 3, 16, 4, false) => 250,
        (256, 2, 16, 4, false) => 211,
        (256, 1, 8, 8, false) => 154,
        (256, 1, 8, 8, true) => 121,
        _ => 255,
    }
}

/// 인스턴스 맹글명 표 — assets/marlin_moe.cu 채리픽 15종과 1:1.
fn moe_sym(th: u32, m: usize, tn: usize, tk: usize, m8: bool) -> Option<&'static str> {
    Some(match (th, m, tn, tk, m8) {
        (128, 1, 4, 8, false) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        (128, 1, 4, 8, true) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi4ELi8ELb1ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        (128, 1, 8, 4, false) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        (128, 1, 8, 4, true) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi8ELi4ELb1ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        (128, 2, 4, 8, false) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi2ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        (128, 2, 8, 4, false) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi2ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        (128, 3, 4, 8, false) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi3ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        (128, 3, 8, 4, false) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi3ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        (128, 4, 4, 8, false) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi4ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        (128, 4, 8, 4, false) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi4ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        (256, 1, 8, 8, false) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi1ELi8ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        (256, 1, 8, 8, true) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi1ELi8ELi8ELb1ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        (256, 2, 16, 4, false) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi2ELi16ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        (256, 3, 16, 4, false) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi3ELi16ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        (256, 4, 16, 4, false) => {
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi4ELi16ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb"
        }
        _ => return None,
    })
}

fn moe_scales_cache(t: &MoeTh, group: usize, stages: usize) -> usize {
    let tb_groups = if group == 0 { 1 } else { t.k.div_ceil(group) };
    tb_groups * t.n * 2 * stages
}

fn moe_kernel_cache(t: &MoeTh, m_blocks: usize, group: usize, stages: usize) -> usize {
    let pack = 8usize;
    let tb_m = m_blocks * 16;
    let sh_block_meta = tb_m * 16;
    let sh_a = stages * (tb_m * t.k) * 2;
    let sh_b = stages * (t.k * t.n / pack) * 4;
    let sh_red = tb_m * (t.n + 8) * 2;
    let sh_bias = t.n * 2;
    let tmp = sh_b.max(sh_red).max(sh_b.min(sh_red) + sh_bias);
    let sh_s = moe_scales_cache(t, group, stages);
    tmp + sh_a + sh_s + sh_block_meta
}

fn moe_valid(t: &MoeTh, m_blocks: usize, n: usize, k: usize, group: usize, smem: usize) -> bool {
    n.is_multiple_of(t.n)
        && k.is_multiple_of(t.k)
        && t.n >= 64
        && t.k >= 64
        && t.th >= 128
        && moe_kernel_cache(t, m_blocks, group, MOE_STAGES) <= smem
}

#[allow(clippy::too_many_arguments)]
fn moe_determine(
    m_blocks: usize,
    m8: bool,
    prob_m: usize,
    prob_n: usize,
    prob_k: usize,
    top_k: usize,
    group: usize,
    smem: usize,
    sms: usize,
) -> Option<(usize, MoeTh)> {
    let cfgs: &[MoeTh] = if m_blocks > 1 { &MOE_LARGE } else { &MOE_SMALL };
    let mut best: Option<(usize, MoeTh)> = None;
    for &t in cfgs {
        if !moe_valid(
            &t,
            m_blocks,
            prob_n,
            prob_k,
            group,
            smem.saturating_sub(512),
        ) {
            continue;
        }
        let cache = moe_kernel_cache(&t, m_blocks, group, MOE_STAGES);
        let regs = moe_regs(t.th, m_blocks, t.n / 16, t.k / 16, m8);
        let reg_size = regs.max(1) * t.th as usize * 4;
        let mut allow = (255 * 1024 / reg_size).min(smem / (cache + 1536));
        allow = if m_blocks == 1 {
            allow.clamp(1, 4)
        } else {
            allow.clamp(1, 2)
        };
        let work = (prob_n / t.n) * prob_m * top_k * 4;
        if work < sms * allow {
            allow = (work / sms).max(1);
        }
        if best.map(|(c, _)| allow > c).unwrap_or(true) {
            best = Some((allow, t));
        }
    }
    best
}

impl W4a16Dec {
    /// [2026-10-19] 마린 MoE GEMM 런치(vLLM marlin_mm 이식). Ok(false) =
    /// 설정 불가 — 호출부가 기존 grp2/d64 경로로 폴백(운영 계약).
    /// C는 f16([prob_m*top_k][prob_n]) — 소비자가 h2f/합산으로 f32 전환.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::rawcuda::w4a16_dec) fn marlin_moe_gemm_dev(
        &mut self,
        a16: CUdeviceptr,
        c16: CUdeviceptr,
        c_tmp: CUdeviceptr,
        bq: CUdeviceptr,
        dsc: CUdeviceptr,
        sorted: CUdeviceptr,
        experts: CUdeviceptr,
        npost: CUdeviceptr,
        topw: CUdeviceptr,
        prob_m: usize,
        prob_n: usize,
        prob_k: usize,
        top_k: usize,
        mul_topk: bool,
        moe_block_size: usize,
    ) -> Result<bool, String> {
        let m_blocks = moe_block_size.div_ceil(16);
        let m8 = moe_block_size == 8;
        let smem = self.cc.smem_optin_max().min(224 * 1024) as usize;
        if smem < 64 * 1024 {
            return Ok(false);
        }
        let sms = self.cc.sm_count().max(1) as usize;
        let Some((bps, t)) = moe_determine(
            m_blocks, m8, prob_m, prob_n, prob_k, top_k, MOE_GROUP, smem, sms,
        ) else {
            return Ok(false);
        };
        let (tnb, tkb) = (t.n / 16, t.k / 16);
        let Some(sym) = moe_sym(t.th, m_blocks, tnb, tkb, m8) else {
            return Ok(false);
        };
        let mut smem_launch = smem;
        if bps > 1 {
            smem_launch = smem / bps - 1024;
        }
        if moe_kernel_cache(&t, m_blocks, MOE_GROUP, MOE_STAGES) > smem_launch {
            return Ok(false);
        }
        let f = self.cc.function(sym)?;
        self.cc.set_dynamic_smem(f, smem_launch as u32)?;
        if self.dmlock == 0 {
            let bytes = sms * 16;
            let p = self.cc.alloc(bytes)?;
            let z = vec![0u8; bytes];
            if let Err(e) = Self::h2d_chunked(&self.cc, p, &z) {
                let _ = self.cc.free(p);
                return Err(e);
            }
            self.dmlock = p;
        }
        let (mut p_a, mut p_b) = (a16, bq);
        let (mut p_c, mut p_ct) = (c16, c_tmp);
        let (mut p_bias, mut p_as, mut p_s, mut p_gs, mut p_zp) = (0u64, 0u64, dsc, 0u64, 0u64);
        let (mut p_sid, mut p_eid, mut p_npp, mut p_topw) = (sorted, experts, npost, topw);
        let (mut p_topk, mut p_mult) = (top_k as i32, mul_topk as i32);
        let (mut p_m, mut p_n, mut p_k) = (prob_m as i32, prob_n as i32, prob_k as i32);
        let (mut p_locks, mut p_hb, mut p_ua, mut p_fr) = (self.dmlock, 0i32, 0i32, 1i32);
        self.cc.launch_shared(
            f,
            (sms * bps) as u32,
            1,
            t.th,
            smem_launch as u32,
            &mut crate::rawcuda::args::l22(
                &mut p_a,
                &mut p_b,
                &mut p_c,
                &mut p_ct,
                &mut p_bias,
                &mut p_as,
                &mut p_s,
                &mut p_gs,
                &mut p_zp,
                &mut p_sid,
                &mut p_eid,
                &mut p_npp,
                &mut p_topw,
                &mut p_topk,
                &mut p_mult,
                &mut p_m,
                &mut p_n,
                &mut p_k,
                &mut p_locks,
                &mut p_hb,
                &mut p_ua,
                &mut p_fr,
            ),
        )?;
        Ok(true)
    }

    /// [2026-10-19] 마린 MoE 정렬 변환(gslot→sorted_ids·expert_ids·num_post).
    pub(in crate::rawcuda::w4a16_dec) fn moe_align_marlin_dev(
        &mut self,
        sorted: CUdeviceptr,
        experts: CUdeviceptr,
        npost: CUdeviceptr,
        block: usize,
        sentinel: i32,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_moe_align_marlin")?;
        let (mut p_g, mut p_c, mut p_o) =
            (self.moe.moe_gslot, self.moe.moe_gcnt, self.moe.moe_goff);
        let (mut p_s, mut p_e, mut p_n) = (sorted, experts, npost);
        let (mut p_ne, mut p_b, mut p_sen) = (self.moe.n_experts as i32, block as i32, sentinel);
        self.cc.launch(
            f,
            1,
            1,
            256,
            &mut crate::rawcuda::args::l9(
                &mut p_g, &mut p_c, &mut p_o, &mut p_s, &mut p_e, &mut p_n, &mut p_ne, &mut p_b,
                &mut p_sen,
            ),
        )
    }
}
