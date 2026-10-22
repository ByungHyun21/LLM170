
pub struct KvPool {
    n_slots: usize,
    n_layers: usize,
    mb_max: usize,
    free_vram: Vec<u32>,
    free_host: Vec<u32>,
    vram_blocks: usize,
    total: usize,
    slot_map: Vec<Vec<u32>>,
    used: Vec<usize>,
}

impl KvPool {
    pub fn new(
        n_slots: usize,
        n_layers: usize,
        mb_max: usize,
        total_blocks: usize,
        vram_blocks: usize,
    ) -> Result<Self, String> {
        if n_slots == 0 || n_layers == 0 || mb_max == 0 {
            return Err("KvPool: 0 차원".into());
        }
        if total_blocks >= u32::MAX as usize {
            return Err(format!("KvPool: 블록 {total_blocks} ≥ u32 센티널"));
        }
        let vb = vram_blocks.min(total_blocks);
        Ok(KvPool {
            n_slots,
            n_layers,
            mb_max,
            free_vram: (0..vb as u32).rev().collect(),
            free_host: (vb as u32..total_blocks as u32).rev().collect(),
            vram_blocks: vb,
            total: total_blocks,
            slot_map: vec![vec![u32::MAX; n_layers * mb_max]; n_slots],
            used: vec![0; n_slots],
        })
    }

    fn push_free(&mut self, id: u32) {
        if (id as usize) < self.vram_blocks {
            self.free_vram.push(id);
        } else {
            self.free_host.push(id);
        }
    }

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
            .free_vram
            .pop()
            .or_else(|| self.free_host.pop())
            .ok_or_else(|| format!("KvPool: 블록 소진(자유 0) — slot={slot}"))?;
        self.slot_map[slot][e] = id;
        self.used[slot] += 1;
        Ok(id)
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn release(&mut self, slot: usize) -> usize {
        let mut n = 0usize;
        if slot < self.n_slots {
            for e in 0..self.slot_map[slot].len() {
                let id = self.slot_map[slot][e];
                if id != u32::MAX {
                    self.slot_map[slot][e] = u32::MAX;
                    self.push_free(id);
                    n += 1;
                }
            }
            self.used[slot] = 0;
        }
        n
    }

    pub fn release_below(
        &mut self,
        slot: usize,
        floor_bi: usize,
        sliding: &[bool],
    ) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if slot >= self.n_slots || floor_bi == 0 {
            return out;
        }
        for l in 0..self.n_layers {
            if !sliding.get(l).copied().unwrap_or(false) {
                continue;
            }
            for bi in 0..floor_bi.min(self.mb_max) {
                let e = l * self.mb_max + bi;
                let id = self.slot_map[slot][e];
                if id != u32::MAX {
                    self.slot_map[slot][e] = u32::MAX;
                    self.push_free(id);
                    self.used[slot] = self.used[slot].saturating_sub(1);
                    out.push((l, bi));
                }
            }
        }
        out
    }

    #[allow(dead_code)]
    pub fn free_blocks(&self) -> usize {
        self.free_vram.len() + self.free_host.len()
    }

    pub fn free_vram_blocks(&self) -> usize {
        self.free_vram.len()
    }

    pub fn free_host_blocks(&self) -> usize {
        self.free_host.len()
    }

    pub fn total_blocks(&self) -> usize {
        self.total
    }

    #[allow(dead_code)]
    pub fn used_blocks(&self, slot: usize) -> usize {
        self.used.get(slot).copied().unwrap_or(0)
    }

    #[allow(dead_code)]
    pub fn mb_max(&self) -> usize {
        self.mb_max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_idempotent_and_upto() {
        let mut p = KvPool::new(2, 3, 4, 100, 100).unwrap();
        let a = p.ensure(0, 1, 1).unwrap();
        assert_eq!(p.ensure(0, 1, 1).unwrap(), a, "멱등");
        let new = p.ensure_upto(0, 1024).unwrap();
        assert_eq!(new.len(), 5);
        assert_eq!(p.used_blocks(0), 6);
        assert_eq!(p.ensure_upto(0, 1024).unwrap().len(), 0);
    }

    #[test]
    fn release_and_reuse() {
        let mut p = KvPool::new(1, 2, 4, 8, 8).unwrap();
        p.ensure_upto(0, 2048).unwrap();
        assert_eq!(p.free_blocks(), 0);
        assert_eq!(p.release(0), 8);
        assert_eq!(p.free_blocks(), 8);
        p.ensure_upto(0, 2048).unwrap();
        assert_eq!(p.free_blocks(), 0);
    }

    #[test]
    fn exhausted_is_error() {
        let mut p = KvPool::new(1, 2, 4, 3, 3).unwrap();
        let e = p.ensure_upto(0, 2048).unwrap_err();
        assert!(e.contains("블록 소진"), "{e}");
    }

    #[test]
    fn tiers_prefer_vram() {
        let mut p = KvPool::new(1, 1, 8, 8, 4).unwrap();
        for bi in 0..4 {
            assert!(p.ensure(0, 0, bi).unwrap() < 4, "VRAM 선할당");
        }
        assert!(p.ensure(0, 0, 4).unwrap() >= 4, "소진 후 호스트");
        p.release(0);
        assert_eq!(p.free_vram_blocks(), 4, "VRAM 복귀 = VRAM 스택");
        assert_eq!(p.free_host_blocks(), 4, "호스트 = 미사용 3 + 복귀 1");
        assert!(p.ensure(0, 0, 0).unwrap() < 4, "해제 후에도 VRAM 우선");
        assert!(p.ensure(0, 0, 1).unwrap() < 4);
        assert!(p.ensure(0, 0, 2).unwrap() < 4);
        assert!(p.ensure(0, 0, 3).unwrap() < 4);
        assert!(p.ensure(0, 0, 4).unwrap() >= 4);
    }

    #[test]
    fn upto_clamps_to_mb_max() {
        let mut p = KvPool::new(1, 1, 4, 4, 4).unwrap();
        let new = p.ensure_upto(0, 99999).unwrap();
        assert_eq!(new.len(), 4);
    }

    #[test]
    fn release_below_sliding_only() {
        let mut p = KvPool::new(1, 2, 4, 8, 8).unwrap();
        p.ensure_upto(0, 2048).unwrap();
        assert_eq!(p.free_blocks(), 0);
        let rel = p.release_below(0, 2, &[false, true]);
        assert_eq!(rel, vec![(1, 0), (1, 1)]);
        assert_eq!(p.free_blocks(), 2);
        assert_eq!(p.used_blocks(0), 6);
        assert!(p.get(0, 0, 0).is_some());
        assert!(p.get(0, 1, 0).is_none());
        assert!(p.release_below(0, 2, &[false, true]).is_empty());
        assert!(p.release_below(0, 0, &[false, true]).is_empty());
        p.ensure(0, 1, 0).unwrap();
        assert_eq!(p.free_blocks(), 1);
    }
}
