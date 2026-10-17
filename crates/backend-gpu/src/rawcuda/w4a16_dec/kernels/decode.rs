use super::super::*;

/// [split-k marlin 디코드] k분할 정책 — 측정 확정(27B 전 형상, 2026-10-17):
/// **블록당 최소 5그룹(k≥640)을 유지하는 최대 분할(상한 8)**이 전 형상에서
/// 최적 ksp 대비 ±1% 내(qkv 1.61×·z 1.29×·ssm_out 1.13×·attn_q 1.75×·
/// attn_output 1.18×·gate/up 1.95×·down 1.84× — GEMV 대비 warm). 종전
/// "70블록 도달 최소 ksp"는 과소분할(n=5120에서 ksp8 대비 최대 1.46× 열위
/// 실측)로 기각. ksp는 g128 그룹(k/128)을 균분((k/128)%ksp==0 — 분할 경계
/// = 그룹 경계). 소형 n(blocks×ksp < 70SM)은 행=블록 GEMV(grid=n)이 병렬도
/// 우위 → None(기존 GEMV 폴백).
pub(in crate::rawcuda::w4a16_dec) fn sk_ksp(blocks: usize, k: usize) -> Option<u8> {
    if !k.is_multiple_of(128) {
        return None;
    }
    let groups = k / 128;
    // 상한 = groups/5(블록당 ≥5그룹 — 그 이상 분할은 x 재판독·워밍업 지배
    // 실측: 20블록×48g에서 ksp16이 ksp8 대비 열위)과 8의 최소.
    let cap = (groups / 5).clamp(1, SK_KSP_CAP);
    let mut best = 1usize;
    for ksp in [1usize, 2, 4, 8] {
        if ksp <= cap && groups.is_multiple_of(ksp) {
            best = ksp;
        }
    }
    (blocks * best >= SK_SM_TARGET).then_some(best as u8)
}

/// SM 수(CMP 170HX = GA100 70SM — 폴백 판정 기준).
pub(in crate::rawcuda::w4a16_dec) const SK_SM_TARGET: usize = 70;

/// ksp 탐색 상한(블록당 최소 그룹 수 = 5와 세트).
pub(in crate::rawcuda::w4a16_dec) const SK_KSP_CAP: usize = 8;

impl W4a16Dec {
    /// [split-k marlin] 다중세그먼트 t=1 sk 런치(branch·gate/up) —
    /// w4a16_gemv_g128_multi 대체. x는 f16(dx16 미러 — 생산자 융합 기록).
    /// 반환 false = sk 미구성(기존 GEMV 폴백). ksp>1이면 reduce 뒤따름
    /// (고정 소스 2단 — 스트림 순서로 본체 완료 후 합산).
    pub(in crate::rawcuda::w4a16_dec) fn sk_multi_dev(
        &mut self,
        il: usize,
        group: usize,
        x16: CUdeviceptr,
    ) -> Result<bool, String> {
        let Some(m) = self.dsk_meta.get(il).and_then(|m| m[group]) else {
            return Ok(false);
        };
        if m.cnt == 0 || self.dsk == 0 {
            return Ok(false);
        }
        let (blocks, ksp) = (m.blocks as usize, m.ksp as usize);
        let f = self.cc.function("w4a16_gemv_sk_marlin")?;
        let (mut p_t, mut p_x, mut p_p) = (self.dsk + m.off as u64 * 8, x16, self.dsk_part.ptr);
        let (mut p_c, mut p_s) = (m.cnt as i32, ksp as i32);
        self.cc.launch(
            f,
            blocks as u32,
            ksp as u32,
            128,
            &mut crate::rawcuda::args::l5(&mut p_t, &mut p_x, &mut p_p, &mut p_c, &mut p_s),
        )?;
        if ksp > 1 {
            let f = self.cc.function("w4a16_gemv_sk_reduce")?;
            self.cc.launch(
                f,
                blocks as u32,
                1,
                256,
                &mut crate::rawcuda::args::l4(&mut p_t, &mut p_p, &mut p_c, &mut p_s),
            )?;
        }
        Ok(true)
    }

    /// [split-k marlin] 단일 세그먼트 t=1 sk 런치(attn_output·ssm_out·
    /// ffn_down) — w4a16_gemv_g128(행=블록) 대체. out 직접 기록(ksp==1) 또는
    /// 부분합+reduce. 반환 false = 폴백.
    pub(in crate::rawcuda::w4a16_dec) fn sk_single_dev(
        &mut self,
        name: &str,
        x16: CUdeviceptr,
        out: CUdeviceptr,
    ) -> Result<bool, String> {
        let (_, ds, n, k) = self.lin_spec(name)?; // 원본 q는 이 경로 미사용(이중 버퍼)
        let Some((dqm, _, _)) = self.lin_mk(name) else {
            return Ok(false);
        };
        let blocks = n.div_ceil(256);
        let Some(ksp) = sk_ksp(blocks, k) else {
            return Ok(false);
        };
        let f = self.cc.function("w4a16_gemv_sk1_marlin")?;
        let (mut p_q, mut p_s, mut p_x) = (dqm, ds, x16);
        let (mut p_o, mut p_p) = (out, self.dsk_part.ptr);
        let (mut p_n, mut p_k, mut p_sp) = (n as i32, k as i32, ksp as i32);
        self.cc.launch(
            f,
            blocks as u32,
            ksp as u32,
            128,
            &mut crate::rawcuda::args::l8(
                &mut p_q, &mut p_s, &mut p_x, &mut p_o, &mut p_p, &mut p_n, &mut p_k, &mut p_sp,
            ),
        )?;
        if ksp > 1 {
            let f = self.cc.function("w4a16_gemv_sk1_reduce")?;
            self.cc.launch(
                f,
                blocks as u32,
                1,
                256,
                &mut crate::rawcuda::args::l4(&mut p_p, &mut p_o, &mut p_n, &mut p_sp),
            )?;
        }
        Ok(true)
    }
}
