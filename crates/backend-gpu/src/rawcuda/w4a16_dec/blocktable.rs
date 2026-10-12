//! [§3.4/§4.1 K6 2026-10-12] 블록 해시 paged KV — 블록 테이블 관리(호스트).
//!
//! 설계 원칙:
//! - **블록 = 512 토큰**(K3 체크포인트·tree.rs 노드와 정합).
//! - 슬롯별 연속 KV(`dkc + slot*cap*kv_dim`)를 블록 단위 분할 할당으로
//!   교체 — 블록 테이블`(slot, block_idx) → device ptr`이 주소 체계.
//! - **간접 참조 계층만 제공**(호스트 관리) — 커널 수정은 다음 트랜셰.
//! - 블록 공유(동일 접두 = 동일 블록 참조·refcount)가 최종 목적(②).

use super::CUdeviceptr;

/// 블록 테이블 — 슬롯별 블록 포인터 목록.
/// [K6-2차 진행 중] attn.cu 간접 참조 통합 전까지 미사용 — 통합 시 활성화.
/// 블록 크기 = 512 토큰 × kv_dim 바이트(int8 KV 기준).
#[allow(dead_code)] // K6-2차 통합 대기
pub struct BlockTable {
    /// 슬롯별 블록 포인터(블록 순서 = 토큰 순서).
    pub blocks: Vec<Vec<CUdeviceptr>>,
    /// 전체 블록 풀(할당된 블록 — 회수/재사용 관리).
    pub pool: Vec<CUdeviceptr>,
    /// 블록 크기(바이트).
    pub block_bytes: usize,
    /// 슬롯 수.
    pub n_slots: usize,
}

#[allow(dead_code)] // K6-2차 통합 대기
impl BlockTable {
    pub fn new(n_slots: usize, kv_dim: usize) -> Self {
        BlockTable {
            blocks: vec![Vec::new(); n_slots],
            pool: Vec::new(),
            block_bytes: 512 * kv_dim,
            n_slots,
        }
    }

    /// 슬롯에 블록 추가 — pos 위치의 블록 포인터 등록.
    pub fn push(&mut self, slot: usize, ptr: CUdeviceptr) -> Result<(), String> {
        if slot >= self.n_slots {
            return Err(format!("BlockTable: 슬롯 {slot} ≥ {}", self.n_slots));
        }
        self.blocks[slot].push(ptr);
        Ok(())
    }

    /// 슬롯 블록 전체 해제 — 슬롯 반환 시.
    pub fn clear(&mut self, slot: usize) {
        if slot < self.n_slots {
            self.blocks[slot].clear();
        }
    }

    /// 위치 → 블록 인덱스·블록 내 오프셋.
    pub fn locate(&self, pos: usize) -> (usize, usize) {
        (pos / 512, pos % 512)
    }

    /// 슬롯·위치 → 장치 주소(블록 기반).
    pub fn addr(&self, slot: usize, pos: usize) -> Result<CUdeviceptr, String> {
        let (bi, off) = self.locate(pos);
        let blks = self.blocks.get(slot).ok_or("BlockTable: 슬롯 범위")?;
        let base = blks.get(bi).ok_or("BlockTable: 블록 범위")?;
        Ok(base + (off * self.block_bytes / 512) as u64)
    }

    /// 슬롯별 사용 중 블록 수.
    pub fn n_blocks(&self, slot: usize) -> usize {
        self.blocks.get(slot).map_or(0, |b| b.len())
    }

    /// 전체 사용 중 블록 수(VRAM 사용량 계측).
    pub fn total_blocks(&self) -> usize {
        self.blocks.iter().map(|b| b.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locate_512_boundary() {
        let bt = BlockTable::new(2, 128);
        assert_eq!(bt.locate(0), (0, 0));
        assert_eq!(bt.locate(511), (0, 511));
        assert_eq!(bt.locate(512), (1, 0));
        assert_eq!(bt.locate(1024), (2, 0));
    }

    #[test]
    fn push_clear_count() {
        let mut bt = BlockTable::new(2, 128);
        bt.push(0, 0x1000).unwrap();
        bt.push(0, 0x2000).unwrap();
        bt.push(1, 0x3000).unwrap();
        assert_eq!(bt.n_blocks(0), 2);
        assert_eq!(bt.n_blocks(1), 1);
        assert_eq!(bt.total_blocks(), 3);
        bt.clear(0);
        assert_eq!(bt.n_blocks(0), 0);
        assert_eq!(bt.total_blocks(), 1);
    }

    #[test]
    fn addr_block_offset() {
        let mut bt = BlockTable::new(1, 128);
        bt.push(0, 0x1000).unwrap();
        assert_eq!(bt.addr(0, 0).unwrap(), 0x1000);
        assert_eq!(bt.addr(0, 1).unwrap(), 0x1000 + 128);
    }
}
