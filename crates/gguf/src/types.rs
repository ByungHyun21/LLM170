//! GGML 텐서 타입: id/이름/블록 크기.
//!
//! 근거: `source/llama.cpp/ggml/include/ggml.h` `enum ggml_type` (2026-08-30 판),
//! `ggml/src/ggml.c` type_traits 테이블, `ggml/src/ggml-common.h` 블록 구조체.
//! 블록 크기는 구조체 static_assert 산식 그대로 유도.

/// GGUF 파일에 실릴 수 있는 타입만. (is_quantized 원문 의미는 `quantized()` 로)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
    Tq1_0 = 34,
    Tq2_0 = 35,
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
}

impl GgmlType {
    pub fn from_u32(id: u32) -> Option<Self> {
        use GgmlType::*;
        Some(match id {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            15 => Q8K,
            16 => Iq2Xxs,
            17 => Iq2Xs,
            18 => Iq3Xxs,
            19 => Iq1S,
            20 => Iq4Nl,
            21 => Iq3S,
            22 => Iq2S,
            23 => Iq4Xs,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => Iq1M,
            30 => Bf16,
            34 => Tq1_0,
            35 => Tq2_0,
            39 => Mxfp4,
            40 => Nvfp4,
            41 => Q1_0,
            42 => Q2_0,
            _ => return None,
        })
    }

    /// ggml_traits 의 type_name
    pub fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "f32",
            F16 => "f16",
            Q4_0 => "q4_0",
            Q4_1 => "q4_1",
            Q5_0 => "q5_0",
            Q5_1 => "q5_1",
            Q8_0 => "q8_0",
            Q8_1 => "q8_1",
            Q2K => "q2_K",
            Q3K => "q3_K",
            Q4K => "q4_K",
            Q5K => "q5_K",
            Q6K => "q6_K",
            Q8K => "q8_K",
            Iq2Xxs => "iq2_xxs",
            Iq2Xs => "iq2_xs",
            Iq3Xxs => "iq3_xxs",
            Iq1S => "iq1_s",
            Iq4Nl => "iq4_nl",
            Iq3S => "iq3_s",
            Iq2S => "iq2_s",
            Iq4Xs => "iq4_xs",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            F64 => "f64",
            Iq1M => "iq1_m",
            Bf16 => "bf16",
            Tq1_0 => "tq1_0",
            Tq2_0 => "tq2_0",
            Mxfp4 => "mxfp4",
            Nvfp4 => "nvfp4",
            Q1_0 => "q1_0",
            Q2_0 => "q2_0",
        }
    }

    /// (blck_size, type_size) — 블록당 원소 수, 블록 바이트 크기.
    pub fn block_info(self) -> (u64, u64) {
        use GgmlType::*;
        match self {
            F32 => (1, 4),
            F16 => (1, 2),
            Bf16 => (1, 2),
            F64 => (1, 8),
            I8 => (1, 1),
            I16 => (1, 2),
            I32 => (1, 4),
            I64 => (1, 8),
            Q1_0 => (128, 18), // 2 + 128/8
            Q2_0 => (64, 18),  // 2 + 64/4
            Q4_0 => (32, 18),
            Q4_1 => (32, 20),
            Q5_0 => (32, 22),
            Q5_1 => (32, 24),
            Q8_0 => (32, 34),
            Q8_1 => (32, 36),
            Mxfp4 => (32, 17),
            Nvfp4 => (64, 36),
            Q2K => (256, 84),
            Q3K => (256, 110),
            Q4K => (256, 144),
            Q5K => (256, 176),
            Q6K => (256, 210),
            Q8K => (256, 292),
            Iq2Xxs => (256, 66),
            Iq2Xs => (256, 74),
            Iq2S => (256, 82),
            Iq3Xxs => (256, 98),
            Iq3S => (256, 110),
            Iq1S => (256, 50),
            Iq1M => (256, 56),
            Iq4Nl => (32, 18),
            Iq4Xs => (256, 136),
            Tq1_0 => (256, 54),
            Tq2_0 => (256, 66),
        }
    }

    pub fn blck_size(self) -> u64 {
        self.block_info().0
    }

    pub fn type_size(self) -> u64 {
        self.block_info().1
    }

    /// 양자화 저장 타입 여부 (원문 is_quantized)
    pub fn quantized(self) -> bool {
        use GgmlType::*;
        !matches!(self, F32 | F16 | Bf16 | F64 | I8 | I16 | I32 | I64)
    }

    /// 평균 비트/가중치
    pub fn bits_per_weight(self) -> f64 {
        let (blck, size) = self.block_info();
        size as f64 * 8.0 / blck as f64
    }

    /// ggml_nbytes 계산: ne[0] 을 블록으로 나눈 크기 × 나머지 차원.
    /// 정상 GGUF는 ne[0] % blck == 0 — 아니면 None (손상 파일 감지).
    pub fn nbytes(self, ne: &[u64; 4]) -> Option<u64> {
        let (blck, size) = self.block_info();
        if ne[0] % blck != 0 {
            return None;
        }
        let blocks0 = ne[0] / blck;
        Some(
            blocks0
                .checked_mul(size)?
                .checked_mul(ne[1])?
                .checked_mul(ne[2])?
                .checked_mul(ne[3])?,
        )
    }
}
