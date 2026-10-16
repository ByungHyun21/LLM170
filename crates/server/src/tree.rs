//! [§3.4 ③ 상태 트리 2026-10-12] 전역 radix 접두 트리 — 슬롯 cached 승격.
//!
//! 설계 원칙(플랜 §3.4 ③):
//! - **커널 불변·무손실**: KV·GDN 상태는 기존 슬롯 메커니즘(K3 체크포인트·
//!   full-접두 재사용)을 그대로 사용 — 트리는 **호스트 측 접두 조회 구조**만
//!   제공(어떤 슬롯에 어떤 접두가 굽혀 있는지 추적).
//! - 노드 = 토큰 접두 경로·경계 512 정렬(K3 체크포인트와 정합).
//! - 요청 = diff("상태 S 확장"): 요청 토큰의 최장 등록 접두를 찾아 배정 슬롯
//!   후보를 반환 — pick_prefix_slot을 대체(슬롯 free 여부와 결합).
//! - 상한(2026-10-16): 생산 경로는 슬롯 제거를 호출하지 않는다(비용상 매 틱
//!   재등록만) — 노드가 살아있는 cached 총량 대비 커지면 rebuild로 스테일 제거.
//! - 무손실: 트리 값은 슬롯 인덱스 + 접두 길이뿐 — 상태 자체를 복사하지
//!   않으므로 수치 무변(골든 유지).

use std::collections::HashMap;
use std::collections::hash_map::Entry;

/// radix 트리 노드 — 자식은 토큰 id → 노드. 경계(512 배수)에만 참조 기록.
#[derive(Default)]
pub struct TrieNode {
    pub children: HashMap<u32, Box<TrieNode>>,
    /// 이 노드(경계)에 정합 상태를 가진 슬롯들(슬롯 인덱스·prefix 길이).
    pub slots: Vec<(usize, usize)>,
}

/// 전역 접두 트리 — sched 소유·단일 스레드(sched 락 내).
#[derive(Default)]
pub struct PrefixTrie {
    root: TrieNode,
    /// 총 등록 경계 수(계측 — /stats 노출 재료).
    pub n_boundaries: usize,
    /// 총 노드 수 — 스테일 재구축 트리거(sched가 판정). 노드는 스테일
    /// 삭제 훅 없이 누적되므로(=생산 insert 전용) 이 값이 상한 판정 입력.
    pub n_nodes: usize,
}

impl PrefixTrie {
    pub fn new() -> Self {
        Self::default()
    }

    /// 접두 등록 — 경계(512 배수)마다 노드에 (슬롯, 길이) 기록.
    /// 같은 슬롯의 중복 경계는 무시(매 틱 재등록 안전). 스테일 정리는
    /// rebuild(슬롯 제거 훅을 생산 경로가 부르지 않는 비용 구조).
    pub fn insert(&mut self, tokens: &[u32], slot: usize) {
        let mut node = &mut self.root;
        for (i, &t) in tokens.iter().enumerate() {
            node = match node.children.entry(t) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => {
                    self.n_nodes += 1;
                    e.insert(Box::default())
                }
            };
            if (i + 1).is_multiple_of(512) && node.slots.iter().all(|&(s, _)| s != slot) {
                node.slots.push((slot, i + 1));
                self.n_boundaries += 1;
            }
        }
    }

    /// 전체 재구축 — 살아 있는 (슬롯, cached)만 재삽입해 스테일 노드·참조를
    /// 제거한다(정본 = 슬롯 cached — 소비부가 후보 정합을 토큰 재비교로 검증).
    pub fn rebuild<'a>(&mut self, live: impl Iterator<Item = (usize, &'a [u32])>) {
        self.root = TrieNode::default();
        self.n_boundaries = 0;
        self.n_nodes = 0;
        for (slot, toks) in live {
            self.insert(toks, slot);
        }
    }

    /// 최장 등록 접두 조회 — (슬롯, 길이) 반환(0 = 미스).
    /// 가장 깊은 경계의 **첫 등록 슬롯**을 후보로 반환(동일 경계 복수 슬롯의
    /// 선택 규칙 = 첫 등록 고정 — 후보 정합은 소비부 재비교가 지킨다).
    pub fn longest_prefix(&self, tokens: &[u32]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        let mut node = &self.root;
        let mut d = 0usize;
        for &t in tokens {
            let Some(child) = node.children.get(&t) else {
                break;
            };
            node = child;
            d += 1;
            if d.is_multiple_of(512) && !node.slots.is_empty() {
                best = Some(node.slots[0]);
            }
        }
        best
    }

    /// 슬롯 제거 — 해당 슬롯의 모든 경계 참조 삭제. [2026-10-16] 테스트 전용
    /// (생산 정리 = rebuild — 스테일이 임계 초과일 때 일괄).
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

    /// 계측 — 등록 경계 수. 테스트 전용 게이트(remove와 동일 계약).
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
        assert_eq!(s, 0); // 첫 등록 슬롯 고정(동일 경계 복수 슬롯)
    }

    /// 스테일 재구축 — live만 남기고 노드/참조가 재작성되는지(경계 수·노드
    /// 수 감소 + 스테일 경로 미스 + 동일 시퀀스 재삽입 무증가).
    #[test]
    fn rebuild_prunes_stale() {
        let mut t = PrefixTrie::new();
        let live = seq(1024);
        let stale: Vec<u32> = (1000..1600).collect();
        t.insert(&live, 0);
        t.insert(&stale, 1);
        let n_before = t.n_nodes;
        assert_eq!(t.longest_prefix(&stale), Some((1, 512)));
        t.rebuild(std::iter::once((0usize, live.as_slice())));
        assert!(t.n_nodes < n_before, "스테일 노드 제거");
        assert_eq!(t.longest_prefix(&live), Some((0, 1024)));
        assert_eq!(t.longest_prefix(&stale), None);
        assert_eq!(t.len(), 2); // live 경계 512·1024만
        let n_after = t.n_nodes;
        t.insert(&live, 0); // 재등록 = 중복 없음(노드 불변)
        assert_eq!(t.n_nodes, n_after);
    }
}
