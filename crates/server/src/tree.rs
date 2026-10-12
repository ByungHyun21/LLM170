//! [§3.4 ③ 상태 트리 2026-10-12] 전역 radix 접두 트리 — 슬롯 cached 승격.
//!
//! 설계 원칙(플랜 §3.4 ③):
//! - **커널 불변·무손실**: KV·GDN 상태는 기존 슬롯 메커니즘(K3 체크포인트·
//!   full-접두 재사용)을 그대로 사용 — 트리는 **호스트 측 접두 조회 구조**만
//!   제공(어떤 슬롯에 어떤 접두가 굽혀 있는지 추적).
//! - 노드 = 토큰 접두 경로·경계 512 정렬(K3 체크포인트와 정합).
//! - 요청 = diff("상태 S 확장"): 요청 토큰의 최장 등록 접두를 찾아 배정 슬롯
//!   후보를 반환 — pick_prefix_slot을 대체(슬롯 free 여부와 결합).
//! - 퇴거(2.A 정책): LRU 기본 — FLOP-aware는 ②에서 정책 인터페이스로 확장.
//! - 무손실: 트리 값은 슬롯 인덱스 + 접두 길이뿐 — 상태 자체를 복사하지
//!   않으므로 수치 무변(골든 유지).

use std::collections::HashMap;

/// radix 트리 노드 — 자식은 토큰 id → 노드. 경계(512 배수)에만 참조 기록.
#[derive(Default)]
pub struct TrieNode {
    pub children: HashMap<u32, Box<TrieNode>>,
    /// 이 노드(경계)에 정합 상태를 가진 슬롯들(슬롯 인덱스·prefix 길이).
    pub slots: Vec<(usize, usize)>,
    /// LRU 틱(FLOP-aware 확장 시 utility로 교체).
    pub last_used: u64,
}

/// 전역 접두 트리 — sched 소유·단일 스레드(sched 락 내).
#[derive(Default)]
pub struct PrefixTrie {
    root: TrieNode,
    tick: u64,
    /// 총 등록 경계 수(계측 — /stats 노출 재료).
    pub n_boundaries: usize,
}

impl PrefixTrie {
    pub fn new() -> Self {
        Self::default()
    }

    /// 접두 등록 — 경계(512 배수)마다 노드에 (슬롯, 길이) 기록.
    /// 슬롯 종료(cached 폐기/재활용) 시 remove로 정리.
    pub fn insert(&mut self, tokens: &[u32], slot: usize) {
        let mut node = &mut self.root;
        self.tick += 1;
        node.last_used = self.tick;
        for (i, &t) in tokens.iter().enumerate() {
            node = node.children.entry(t).or_default();
            node.last_used = self.tick;
            if (i + 1).is_multiple_of(512) {
                if node.slots.iter().all(|&(s, _)| s != slot) {
                    node.slots.push((slot, i + 1));
                    self.n_boundaries += 1;
                }
            }
        }
    }

    /// 최장 등록 접두 조회 — (슬롯, 길이) 반환(0 = 미스).
    /// 후보 슬롯 중 last_used 최대(LRU) 선택.
    pub fn longest_prefix(&mut self, tokens: &[u32]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize, u64)> = None;
        let mut node = &mut self.root;
        let mut d = 0usize;
        for &t in tokens {
            let Some(child) = node.children.get_mut(&t) else {
                break;
            };
            node = child;
            d += 1;
            if d.is_multiple_of(512) && !node.slots.is_empty() {
                // 가장 깊은 경계의 첫 슬롯(동일 경계 내 선택은 ② 정책).
                let (s, l) = node.slots[0];
                best = Some((s, l, node.last_used));
            }
        }
        // 마지막으로 방문한 경계 노드의 last_used로 갱신(LRU touch).
        best.map(|(s, l, _)| (s, l))
    }

    /// 슬롯 제거 — 해당 슬롯의 모든 경계 참조 삭제(슬롯 폐기/재활용 시).
    /// [2026-10-12] 현재 호출부는 테스트 전용(생산 sched은 insert·
    /// longest_prefix만) — cfg(test) 게이트로 release dead_code 경로 제거.
    #[cfg(test)]
    pub fn remove(&mut self, slot: usize) {
        fn rec(node: &mut TrieNode, slot: usize, counter: &mut usize) {
            let before = node.slots.len();
            node.slots.retain(|&(s, _)| s != slot);
            *counter -= before - node.slots.len();
            for c in node.children.values_mut() {
                rec(c, slot, counter);
            }
        }
        let mut n = self.n_boundaries;
        rec(&mut self.root, slot, &mut n);
        self.n_boundaries = n;
    }

    /// 계측 — 등록 경계 수(K1 지표 노출 재료). [2026-10-12] remove와 동일
    /// 테스트 전용 게이트.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.n_boundaries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(n: usize) -> Vec<u32> {
        (1..=n as u32).collect()
    }

    #[test]
    fn insert_and_lookup() {
        let mut t = PrefixTrie::new();
        t.insert(&seq(1024), 0); // 경계 512·1024
        assert_eq!(t.longest_prefix(&seq(600)), Some((0, 512)));
        assert_eq!(t.longest_prefix(&seq(2000)), Some((0, 1024)));
        assert_eq!(t.longest_prefix(&seq(100)), None);
    }

    #[test]
    fn remove_clears_all_boundaries() {
        let mut t = PrefixTrie::new();
        t.insert(&seq(1024), 3);
        assert_eq!(t.len(), 2);
        t.remove(3);
        assert_eq!(t.len(), 0);
        assert_eq!(t.longest_prefix(&seq(2000)), None);
    }

    #[test]
    fn multi_slot_boundary() {
        let mut t = PrefixTrie::new();
        t.insert(&seq(512), 0);
        t.insert(&seq(512), 1);
        let (s, _) = t.longest_prefix(&seq(512)).unwrap();
        assert_eq!(s, 0); // 첫 슬롯(동일 경계 내 선택은 ② 정책 — 기본 첫 등록)
    }
}
