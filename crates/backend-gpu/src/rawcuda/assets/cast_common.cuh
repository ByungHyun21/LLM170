// ── f16 비트 변환 공용 계약(2026-10-08) ──
// h2f: f16 비트 → f32(정확 — 서브노멀 정규화 루프 포함, libdevice 미사용).
// f2h: f32 → f16 비트(RN-even, 서브노멀·inf/nan) — 호스트 w4a16_dec::f32_to_f16
// 과 비트동일 계약. 이 두 함수는 **단일 정의**이며 모든 커널이 이 헤더를
// include 한다(gptq4/norm/ew) — 4중 수동 복사 시절의 드리프트 방지.
// 반드시 동형으로 수정할 것(서버 cast_contract 테스트가 호스트 쪽을 전수 대조).

__device__ __forceinline__ float h2f(unsigned short h) {
    unsigned sign = (unsigned)(h >> 15) << 31;
    unsigned e = (unsigned)(h >> 10) & 0x1Fu;
    unsigned m = (unsigned)h & 0x3FFu;
    unsigned bits;
    if (e == 0) {
        if (m == 0) {
            bits = sign;
        } else {
            // 서브노멀 — Rust half_to_f32와 동일 정규화 루프.
            int ee = 127 - 15 + 1;
            unsigned f = m;
            while ((f & 0x400u) == 0) {
                f <<= 1;
                ee -= 1;
            }
            f &= 0x3FFu;
            bits = sign | ((unsigned)ee << 23) | (f << 13);
        }
    } else if (e == 0x1Fu) {
        bits = sign | (0xFFu << 23) | (m << 13);
    } else {
        bits = sign | ((e + 112u) << 23) | (m << 13);
    }
    return __uint_as_float(bits);
}

// f32 → f16 비트(RN-even, 서브노멀·inf/nan) — 호스트 w4a16_dec::f32_to_f16
// 과 비트동일 계약(디바이스 체인 캐스트). 반드시 동형으로 수정할 것.
__device__ __forceinline__ unsigned short f2h(float v) {
    unsigned b = __float_as_uint(v);
    unsigned sign = (unsigned short)((b >> 16) & 0x8000u);
    int exp = (int)((b >> 23) & 0xFFu);
    unsigned man = b & 0x7FFFFFu;
    if (exp == 0xFF) {
        return (unsigned short)(sign | 0x7C00u | (man != 0 ? 0x200u : 0u));
    }
    int e = exp - 127 + 15;
    if (e >= 31) {
        return (unsigned short)(sign | 0x7C00u);
    }
    if (e <= 0) {
        if (e < -10) {
            return (unsigned short)sign;
        }
        unsigned m = man | 0x800000u;
        unsigned shift = (unsigned)(14 - e);
        unsigned half = 1u << (shift - 1);
        unsigned sub = m >> shift;
        unsigned rem = m & ((1u << shift) - 1u);
        if (rem > half || (rem == half && (sub & 1u) == 1u)) {
            sub += 1u;
        }
        return (unsigned short)(sign | sub);
    }
    unsigned h = ((unsigned)e << 10) | (man >> 13);
    unsigned rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) == 1u)) {
        h += 1u;
    }
    return (unsigned short)(sign | h);
}

