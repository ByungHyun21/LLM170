
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum WType {
    F32 = 0,
    F16 = 1,
    Bf16 = 30,
    W4a16Split = 100,
}

impl WType {
    pub fn block_info(self) -> (u64, u64) {
        match self {
            WType::F32 => (1, 4),
            WType::F16 | WType::Bf16 => (1, 2),
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
