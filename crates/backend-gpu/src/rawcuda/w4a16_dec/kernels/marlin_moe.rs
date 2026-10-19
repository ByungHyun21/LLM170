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

    /// [2026-10-19] 마린 MoE FFN 시도(상주 전용) — w13→활성→w2→topk 합산을
    /// dmo(f32)에 직접 기록. Ok(false) = 미가용/설정 불가 — 호출부가 레거시
    /// 커널(grp2/d64/accum)로 폴백. 스트라이드 식은 스테이지③ 업로드와 동일 계약.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::rawcuda::w4a16_dec) fn marlin_moe_ffn_dev(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        t: usize,
        h: usize,
        n_ff: usize,
        top_k: usize,
        gelu: bool,
    ) -> Result<bool, String> {
        if self.moe_m13 == 0 || self.moe_m2 == 0 || self.moe_s13 == 0 || self.moe_s2 == 0 {
            return Ok(false);
        }
        let ne = self.moe.n_experts;
        let ns = t * top_k;
        if ne == 0 || ns == 0 {
            return Ok(false);
        }
        // 블록 크기 휴리스틱(vLLM marlin_moe.py): M*topk/E/b < 0.9인 첫 b.
        let mut block = 64usize;
        for b in [8usize, 16, 32, 48, 64] {
            if (ns as f64) / (ne as f64) / (b as f64) < 0.9 {
                block = b;
                break;
            }
        }
        // 아레나 스트라이드(스테이지③과 동일식).
        let (st13, st2) = (n_ff * h, h * n_ff / 2);
        let (ss13, ss2) = ((h / 32) * 2 * n_ff * 2, (n_ff / 32) * h * 2);
        // 스크래치(지연 확보).
        let sms = self.cc.sm_count().max(1) as usize;
        let sorted_n = ns + ne * block;
        let exp_n = ns.div_ceil(block) + ne + 8;
        let c13_n = ns * 2 * n_ff * 2;
        let act_n = ns * n_ff * 2;
        let c2_n = ns * h * 2;
        // C_tmp — vLLM ops.cu: min(size_n·sorted, sms·4·block·max_thread_n),
        // m8(block=8)은 ×2(실측 계약 — 커널의 m8 리듀스가 2배 사용).
        let ctmp_n = ((2 * n_ff).max(h) * sorted_n).min(sms * 4 * block * 256)
            * if block == 8 { 2 } else { 1 }
            * 4;
        macro_rules! ensure_buf {
            ($f:ident, $n:expr, $w:expr) => {
                if $n > self.$f.cap {
                    self.$f = self.realloc_buf(self.$f, $n, $n, $w)?;
                }
            };
        }
        ensure_buf!(moe_msorted, sorted_n * 4, "moe_msorted");
        ensure_buf!(moe_mexp, exp_n * 4, "moe_mexp");
        ensure_buf!(moe_mnpost, 4, "moe_mnpost");
        ensure_buf!(moe_mc13, c13_n, "moe_mc13");
        ensure_buf!(moe_mact, act_n, "moe_mact");
        ensure_buf!(moe_mc2, c2_n, "moe_mc2");
        ensure_buf!(moe_mctmp, ctmp_n, "moe_mctmp");
        let x16 = self.cast_f16_dev(xn, t * h)?;
        // 정렬 변환(gslot → 마린 형식).
        self.moe_align_marlin_dev(
            self.moe_msorted.ptr,
            self.moe_mexp.ptr,
            self.moe_mnpost.ptr,
            block,
            ns as i32,
        )?;
        let topw = self.moe.moe_wt;
        // w13: gate|up 융합(prob_m=t·top_k=topk·가중 미적용).
        let b13 = self.moe_m13 + (il * ne) as u64 * st13 as u64;
        let s13 = self.moe_s13 + (il * ne) as u64 * ss13 as u64;
        if !self.marlin_moe_gemm_dev(
            x16,
            self.moe_mc13.ptr,
            self.moe_mctmp.ptr,
            b13,
            s13,
            self.moe_msorted.ptr,
            self.moe_mexp.ptr,
            self.moe_mnpost.ptr,
            topw,
            t,
            2 * n_ff,
            h,
            top_k,
            false,
            block,
        )? {
            return Ok(false);
        }
        // 활성 → act f16 — qwen35 MoE = silu·mul · gemma4 = gelu_tanh.
        {
            let f = self.cc.function(if gelu {
                "w4a16_moe_act_gelu_f16"
            } else {
                "w4a16_moe_act_silu_f16"
            })?;
            let (mut p_c, mut p_a) = (self.moe_mc13.ptr, self.moe_mact.ptr);
            let (mut p_ns, mut p_nf) = (ns as i32, n_ff as i32);
            let n = ns * n_ff;
            self.cc.launch(
                f,
                n.div_ceil(4096) as u32,
                1,
                256,
                &mut crate::rawcuda::args::l4(&mut p_c, &mut p_a, &mut p_ns, &mut p_nf),
            )?;
        }
        // w2: down(prob_m=ns·top_k=1·가중 적용).
        let b2 = self.moe_m2 + (il * ne) as u64 * st2 as u64;
        let s2 = self.moe_s2 + (il * ne) as u64 * ss2 as u64;
        if !self.marlin_moe_gemm_dev(
            self.moe_mact.ptr,
            self.moe_mc2.ptr,
            self.moe_mctmp.ptr,
            b2,
            s2,
            self.moe_msorted.ptr,
            self.moe_mexp.ptr,
            self.moe_mnpost.ptr,
            topw,
            ns,
            h,
            n_ff,
            1,
            true,
            block,
        )? {
            return Ok(false);
        }
        // topk 합산 → dmo f32.
        {
            let f = self.cc.function("w4a16_moe_topk_sum_f16")?;
            let (mut p_c, mut p_o) = (self.moe_mc2.ptr, self.moe.dmo.ptr);
            let (mut p_t, mut p_h, mut p_k) = (t as i32, h as i32, top_k as i32);
            self.cc.launch(
                f,
                (t * h).div_ceil(4096) as u32,
                1,
                256,
                &mut crate::rawcuda::args::l5(&mut p_c, &mut p_o, &mut p_t, &mut p_h, &mut p_k),
            )?;
        }
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
