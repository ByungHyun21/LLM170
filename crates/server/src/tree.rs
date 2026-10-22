
use std::collections::HashMap;
use std::collections::hash_map::Entry;

#[derive(Default)]
pub struct TrieNode {
    pub children: HashMap<u32, Box<TrieNode>>,
    pub slots: Vec<(usize, usize)>,
}

#[derive(Default)]
pub struct PrefixTrie {
    root: TrieNode,
    pub n_boundaries: usize,
    pub n_nodes: usize,
}

impl PrefixTrie {
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn rebuild<'a>(&mut self, live: impl Iterator<Item = (usize, &'a [u32])>) {
        self.root = TrieNode::default();
        self.n_boundaries = 0;
        self.n_nodes = 0;
        for (slot, toks) in live {
            self.insert(toks, slot);
        }
    }

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
        t.insert(&seq(1024), 0);
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
        assert_eq!(s, 0);
    }

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
        assert_eq!(t.len(), 2);
        let n_after = t.n_nodes;
        t.insert(&live, 0);
        assert_eq!(t.n_nodes, n_after);
    }
}
