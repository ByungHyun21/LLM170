//! probes/wmma — f32→f16 비트 변환(프로브 전용 근사, gemm 프로브가 사용).

/// f32 → f16 비트(호스트측, 프로브 전용 근사).
pub(super) fn half_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = (x >> 13) & 0x3FF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((exp as u16) << 10) | frac as u16
}
