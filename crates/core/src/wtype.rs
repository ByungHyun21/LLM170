//! 가중치 타입 태그 — `Weight.ty` 계약(2026-10-08 단일 트랙 정리).
//!
//! 실사용 4종: `F32`·`F16`·`Bf16`(플레인) + `W4a16Split`(int4 g128 분리
//! 버퍼). 분리 버퍼는 파일에 나타나지 않는 in-memory 표현 — `data`=packed
//! 행우선 [n][k/8 u32], `aux`=scale 행우선 [n][k/128 u16]. `dequant_row`
//! 비경유(matmul 전용 arm이 소비).

/// 가중치 타입.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum WType {
    F32 = 0,
    F16 = 1,
    Bf16 = 30,
    /// W4A16 g128 sym — 분리 버퍼(packed+scale) in-memory 표현.
    W4a16Split = 100,
}

impl WType {
    /// (blck_size, type_size) — 블록당 원소 수, 블록 바이트 크기.
    pub fn block_info(self) -> (u64, u64) {
        match self {
            WType::F32 => (1, 4),
            WType::F16 | WType::Bf16 => (1, 2),
            // 분리 버퍼 총량(오프셋 해석은 matmul arm 전용 — 스케일 행 길이가
            // 그룹 크기를 정한다: k/128).
            WType::W4a16Split => (128, 66),
        }
    }

    pub fn blck_size(self) -> u64 {
        self.block_info().0
    }

    pub fn type_size(self) -> u64 {
        self.block_info().1
    }
}
