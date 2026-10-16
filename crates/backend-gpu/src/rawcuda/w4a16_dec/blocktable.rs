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
/// [2026-10-16] 2계층: id < `vram_blocks` = VRAM 아레나, 이후 = 호스트 아레나.
/// 자유 목록을 티어별 2스택으로 분리 — **VRAM 스택 우선 pop**(해제 LIFO가
/// 호스트 블록을 스택 꼭대기에 올려 VRAM 여유가 있어도 호스트를 재사용하던
/// 오프로드 점착을 차단, 2026-10-16 검토 반영).
pub struct KvPool {
    n_slots: usize,
    n_layers: usize,
    /// 층당 블록 수(테이블 크기 = cap/512).
    mb_max: usize,
    /// VRAM 자유 블록 id(먼저 pop — 성능 우선).
    free_vram: Vec<u32>,
    /// 호스트 자유 블록 id(VRAM 소진 시).
    free_host: Vec<u32>,
    /// VRAM 아레나 블록 수(티어 경계).
    vram_blocks: usize,
    /// 총 블록 수(용량 — 입장 불가 판정).
    total: usize,
    /// slot_map[slot][layer * mb_max + bi] = 물리 id (u32::MAX = 미할당).
    slot_map: Vec<Vec<u32>>,
    used: Vec<usize>,
}

impl KvPool {
    /// `total_blocks` ≤ u32::MAX-1(센티널) 계약. `vram_blocks` = 앞쪽 VRAM 몫.
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

    /// id의 티어 복귀 — VRAM id는 VRAM 스택(우선 재사용).
    fn push_free(&mut self, id: u32) {
        if (id as usize) < self.vram_blocks {
            self.free_vram.push(id);
        } else {
            self.free_host.push(id);
        }
    }

    /// (slot, layer, bi) 블록 확보 — 있으면 그 id, 없으면 할당(VRAM 우선).
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
                    self.push_free(id);
                    n += 1;
                }
            }
            self.used[slot] = 0;
        }
        n
    }

    /// [2.N Stage 2] 플로어 미만 해제 — 슬라이딩 층(`sliding[l]==true`)에서
    /// `bi < floor_bi`인 확보 블록을 자유 풀로 되돌린다. 풀층은 손대지 않는다
    /// (장거리 검색은 풀층이 담당 — 고정 창 기각과 달리 네이티브 SW층만
    /// 윈도+여유로 유지라 출력 불변). 해제된 (층, 블록) 목록 반환(호출부가
    /// 디바이스 테이블 엔트리를 0으로 지운다). 멱등·단조(플로어 전진만).
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

    #[allow(dead_code)] // [K6-3] 다음 단계(입장 보장·실효 ctx) 배선 전
    pub fn free_blocks(&self) -> usize {
        self.free_vram.len() + self.free_host.len()
    }

    /// [2026-10-16] VRAM 자유 블록 수(진단 — 오프로드 점착 판정).
    pub fn free_vram_blocks(&self) -> usize {
        self.free_vram.len()
    }

    /// [2026-10-16] 호스트 자유 블록 수(진단).
    pub fn free_host_blocks(&self) -> usize {
        self.free_host.len()
    }

    /// 총 블록 수 — 예약이 용량 자체를 넘는 요청의 조기 거부(무한 대기 방지).
    pub fn total_blocks(&self) -> usize {
        self.total
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
        let mut p = KvPool::new(2, 3, 4, 100, 100).unwrap();
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
        let mut p = KvPool::new(1, 2, 4, 8, 8).unwrap();
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
        let mut p = KvPool::new(1, 2, 4, 3, 3).unwrap();
        let e = p.ensure_upto(0, 2048).unwrap_err();
        assert!(e.contains("블록 소진"), "{e}");
    }

    /// [2026-10-16] 2계층 자유 목록 — VRAM 우선 pop, 해제 후에도 호스트 점착
    /// 없음(구 단일 LIFO는 방금 해제한 호스트 id를 꼭대기에 올려 재사용했다).
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
        // 해제 직후 재할당 = VRAM 우선(호스트 점착 없음).
        assert!(p.ensure(0, 0, 0).unwrap() < 4, "해제 후에도 VRAM 우선");
        assert!(p.ensure(0, 0, 1).unwrap() < 4);
        // VRAM 소진 뒤에야 호스트.
        assert!(p.ensure(0, 0, 2).unwrap() < 4);
        assert!(p.ensure(0, 0, 3).unwrap() < 4);
        assert!(p.ensure(0, 0, 4).unwrap() >= 4);
    }

    #[test]
    fn upto_clamps_to_mb_max() {
        let mut p = KvPool::new(1, 1, 4, 4, 4).unwrap();
        // 요청 토큰이 테이블 범위를 넘어도 mb_max로 클램프(테이블 밖 엔트리 금지).
        let new = p.ensure_upto(0, 99999).unwrap();
        assert_eq!(new.len(), 4);
    }

    /// [2.N Stage 2] 플로어 미만 해제 — 슬라이딩층만, 풀층 보존, 멱등.
    #[test]
    fn release_below_sliding_only() {
        // 2층(0=풀·1=슬라이딩) × 4블록, 전량 확보(8칸).
        let mut p = KvPool::new(1, 2, 4, 8, 8).unwrap();
        p.ensure_upto(0, 2048).unwrap();
        assert_eq!(p.free_blocks(), 0);
        let rel = p.release_below(0, 2, &[false, true]);
        // 슬라이딩층 블록 0·1만 해제(2칸).
        assert_eq!(rel, vec![(1, 0), (1, 1)]);
        assert_eq!(p.free_blocks(), 2);
        assert_eq!(p.used_blocks(0), 6);
        // 풀층(0,0)은 그대로 배정 유지.
        assert!(p.get(0, 0, 0).is_some());
        assert!(p.get(0, 1, 0).is_none());
        // 멱등 — 재호출 신규 0.
        assert!(p.release_below(0, 2, &[false, true]).is_empty());
        // 플로어 0 = 무동작.
        assert!(p.release_below(0, 0, &[false, true]).is_empty());
        // 해제분 재할당 가능.
        p.ensure(0, 1, 0).unwrap();
        assert_eq!(p.free_blocks(), 1);
    }
}
