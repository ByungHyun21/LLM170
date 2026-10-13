//! [K6 3차 2026-10-13] KV 블록 풀 — 호스트 측 물리 블록 할당자.
//!
//! 설계:
//! - **블록 = 512 토큰 × 1층**(K3 체크포인트·tree.rs 노드와 정합).
//! - 물리 = 평탄 아레나 4종(k/v/ksc/vsc, 같은 블록 수) — 블록 주소 =
//!   `arena + id × block_bytes`. 커널은 디바이스 블록 테이블(dblk_tab)만
//!   보므로, 호스트가 (slot, layer, block) → 물리 id를 바꾸면 그대로 반영된다.
//! - **현 단계(보존 커밋)**: 총 블록 수 = 종전 용량(slots × 층수 × mb)과 동일,
//!   set_attn이 전량 선할당(주소 체계는 종전과 동형 — 슬롯 메이저).
//! - **다음 단계(입장 보장)**: 예산 기반 풀 + 온디맨드 `ensure_upto`(성장)·
//!   `release`(반납) — 슬롯 정적 예약 제거.

/// 물리 블록 할당자 — (slot, layer, block) → 물리 블록 id.
pub struct KvPool {
    n_slots: usize,
    n_layers: usize,
    /// 층당 블록 수(테이블 크기 = cap/512).
    mb_max: usize,
    /// 자유 블록 id(스택 — 할당 O(1)).
    free: Vec<u32>,
    /// slot_map[slot][layer * mb_max + bi] = 물리 id (u32::MAX = 미할당).
    slot_map: Vec<Vec<u32>>,
    used: Vec<usize>,
}

impl KvPool {
    /// `total_blocks` ≤ u32::MAX-1(센티널) 계약.
    pub fn new(
        n_slots: usize,
        n_layers: usize,
        mb_max: usize,
        total_blocks: usize,
    ) -> Result<Self, String> {
        if n_slots == 0 || n_layers == 0 || mb_max == 0 {
            return Err("KvPool: 0 차원".into());
        }
        if total_blocks >= u32::MAX as usize {
            return Err(format!("KvPool: 블록 {total_blocks} ≥ u32 센티널"));
        }
        Ok(KvPool {
            n_slots,
            n_layers,
            mb_max,
            free: (0..total_blocks as u32).rev().collect(),
            slot_map: vec![vec![u32::MAX; n_layers * mb_max]; n_slots],
            used: vec![0; n_slots],
        })
    }

    /// (slot, layer, bi) 블록 확보 — 있으면 그 id, 없으면 할당.
    pub fn ensure(&mut self, slot: usize, layer: usize, bi: usize) -> Result<u32, String> {
        if slot >= self.n_slots || layer >= self.n_layers || bi >= self.mb_max {
            return Err(format!(
                "KvPool: 범위 밖 slot={slot} layer={layer} bi={bi} \
                 ({}/{}/{})",
                self.n_slots, self.n_layers, self.mb_max
            ));
        }
        let e = layer * self.mb_max + bi;
        let cur = self.slot_map[slot][e];
        if cur != u32::MAX {
            return Ok(cur);
        }
        let id = self
            .free
            .pop()
            .ok_or_else(|| format!("KvPool: 블록 소진(자유 0) — slot={slot}"))?;
        self.slot_map[slot][e] = id;
        self.used[slot] += 1;
        Ok(id)
    }

    /// n_tokens까지 필요한 모든 (층,블록)을 확보 — 새로 할당된 물리 id 목록
    /// (호출부가 디바이스 테이블 엔트리를 갱신). 멱등.
    #[allow(dead_code)] // [K6-3] 다음 단계(온디맨드 성장·입장 보장) 배선 전
    pub fn ensure_upto(&mut self, slot: usize, n_tokens: usize) -> Result<Vec<u32>, String> {
        let nb = n_tokens.div_ceil(512).min(self.mb_max);
        let mut new = Vec::new();
        for l in 0..self.n_layers {
            for bi in 0..nb {
                let e = l * self.mb_max + bi;
                if self.slot_map[slot][e] == u32::MAX {
                    new.push(self.ensure(slot, l, bi)?);
                }
            }
        }
        Ok(new)
    }

    pub fn get(&self, slot: usize, layer: usize, bi: usize) -> Option<u32> {
        let id = self.slot_map.get(slot)?.get(layer * self.mb_max + bi)?;
        (*id != u32::MAX).then_some(*id)
    }

    /// 슬롯 반납 — 사용 블록 수 반환.
    #[allow(dead_code)] // [K6-3] 다음 단계(입장 보장·실효 ctx) 배선 전
    pub fn release(&mut self, slot: usize) -> usize {
        let mut n = 0usize;
        if slot < self.n_slots {
            for e in 0..self.slot_map[slot].len() {
                let id = self.slot_map[slot][e];
                if id != u32::MAX {
                    self.slot_map[slot][e] = u32::MAX;
                    self.free.push(id);
                    n += 1;
                }
            }
            self.used[slot] = 0;
        }
        n
    }

    #[allow(dead_code)] // [K6-3] 다음 단계(입장 보장·실효 ctx) 배선 전
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    #[allow(dead_code)] // [K6-3] 다음 단계(입장 보장·실효 ctx) 배선 전
    pub fn used_blocks(&self, slot: usize) -> usize {
        self.used.get(slot).copied().unwrap_or(0)
    }

    #[allow(dead_code)] // [K6-3] 다음 단계(입장 보장·실효 ctx) 배선 전
    pub fn mb_max(&self) -> usize {
        self.mb_max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_idempotent_and_upto() {
        let mut p = KvPool::new(2, 3, 4, 100).unwrap();
        let a = p.ensure(0, 1, 1).unwrap();
        assert_eq!(p.ensure(0, 1, 1).unwrap(), a, "멱등");
        let new = p.ensure_upto(0, 1024).unwrap();
        // 층 3 × 블록 2 = 6칸 중 (1,1)은 이미 있음 → 신규 5.
        assert_eq!(new.len(), 5);
        assert_eq!(p.used_blocks(0), 6);
        // 재호출은 신규 0(멱등).
        assert_eq!(p.ensure_upto(0, 1024).unwrap().len(), 0);
    }

    #[test]
    fn release_and_reuse() {
        let mut p = KvPool::new(1, 2, 4, 8).unwrap();
        p.ensure_upto(0, 2048).unwrap();
        assert_eq!(p.free_blocks(), 0);
        assert_eq!(p.release(0), 8);
        assert_eq!(p.free_blocks(), 8);
        // 반납 뒤 재할당 — 같은 풀에서 다시 8칸.
        p.ensure_upto(0, 2048).unwrap();
        assert_eq!(p.free_blocks(), 0);
    }

    #[test]
    fn exhausted_is_error() {
        let mut p = KvPool::new(1, 2, 4, 3).unwrap();
        let e = p.ensure_upto(0, 2048).unwrap_err();
        assert!(e.contains("블록 소진"), "{e}");
    }

    #[test]
    fn upto_clamps_to_mb_max() {
        let mut p = KvPool::new(1, 1, 4, 4).unwrap();
        // 요청 토큰이 테이블 범위를 넘어도 mb_max로 클램프(테이블 밖 엔트리 금지).
        let new = p.ensure_upto(0, 99999).unwrap();
        assert_eq!(new.len(), 4);
    }
}
