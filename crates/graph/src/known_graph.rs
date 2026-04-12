//! KnownGraph: graph algorithms that assume the full ancestry is already loaded.
//!
//! Ported from `vcsgraph/known_graph.py`.

use crate::tsort::MergeSorter;
use crate::{Error, RevnoVec};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::hash::Hash;

/// A key that may either be a real node or the synthetic "origin" sentinel
/// (equivalent to `NULL_REVISION` in the Python implementation).
///
/// Only used by [`KnownGraph::heads`], which has special semantics for the
/// origin: it is only considered a head when it is the sole candidate.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key<K> {
    Origin,
    Node(K),
}

#[derive(Debug, Clone)]
struct KnownGraphNode<K> {
    parent_keys: Option<Vec<K>>,
    child_keys: Vec<K>,
    gdfo: Option<u64>,
}

/// Produce a Vec of `items` ordered by hash of each element. Used as a stable
/// (within one process) cache key for unordered sets when `K: Ord` is not
/// required.
fn sort_by_hash<K: Hash, I: IntoIterator<Item = K>>(items: I) -> Vec<K> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let hash_of = |k: &K| {
        let mut h = DefaultHasher::new();
        k.hash(&mut h);
        h.finish()
    };
    let mut v: Vec<K> = items.into_iter().collect();
    v.sort_by_key(hash_of);
    v
}

impl<K> KnownGraphNode<K> {
    fn new(parent_keys: Option<Vec<K>>) -> Self {
        KnownGraphNode {
            parent_keys,
            child_keys: Vec::new(),
            gdfo: None,
        }
    }

    fn is_ghost(&self) -> bool {
        self.parent_keys.is_none()
    }
}

/// Information about a node in a merge-sorted graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeSortNode<K> {
    pub key: K,
    pub merge_depth: usize,
    pub revno: RevnoVec,
    pub end_of_merge: bool,
}

/// A graph where the full ancestry is already known.
///
/// Supports gdfo-based queries like [`heads`](Self::heads), plus various
/// topological orderings.
#[derive(Debug, Clone)]
pub struct KnownGraph<K: Hash + Eq + Clone> {
    nodes: FxHashMap<K, KnownGraphNode<K>>,
    known_heads: FxHashMap<Vec<K>, FxHashSet<K>>,
    do_cache: bool,
}

impl<K: Hash + Eq + Clone> KnownGraph<K> {
    /// Build a new KnownGraph from a parent map.
    pub fn new<I>(parent_map: I, do_cache: bool) -> Self
    where
        I: IntoIterator<Item = (K, Vec<K>)>,
    {
        let mut g = KnownGraph {
            nodes: FxHashMap::default(),
            known_heads: FxHashMap::default(),
            do_cache,
        };
        g.initialize_nodes(parent_map);
        g.find_gdfo();
        g
    }

    fn initialize_nodes<I>(&mut self, parent_map: I)
    where
        I: IntoIterator<Item = (K, Vec<K>)>,
    {
        for (key, parent_keys) in parent_map {
            // Ensure all parent nodes exist and record the reverse edge.
            for parent_key in &parent_keys {
                self.nodes
                    .entry(parent_key.clone())
                    .or_insert_with(|| KnownGraphNode::new(None))
                    .child_keys
                    .push(key.clone());
            }
            // Insert or update the node itself.
            let node = self
                .nodes
                .entry(key)
                .or_insert_with(|| KnownGraphNode::new(None));
            node.parent_keys = Some(parent_keys);
        }
    }

    fn find_tails(&self) -> Vec<K> {
        // A "tail" has no parents — either a real root (Some(empty)) or a
        // ghost (None). Both kinds are treated as gdfo=1 starting points,
        // matching the Python `not node.parent_keys` check.
        self.nodes
            .iter()
            .filter_map(|(k, n)| match &n.parent_keys {
                Some(p) if p.is_empty() => Some(k.clone()),
                None => Some(k.clone()),
                _ => None,
            })
            .collect()
    }

    fn find_tips(&self) -> Vec<K> {
        self.nodes
            .iter()
            .filter_map(|(k, n)| {
                if n.child_keys.is_empty() {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    fn find_gdfo(&mut self) {
        let mut known_parent_gdfos: FxHashMap<K, usize> = FxHashMap::default();
        let mut pending: Vec<K> = Vec::new();

        for key in self.find_tails() {
            self.nodes.get_mut(&key).unwrap().gdfo = Some(1);
            pending.push(key);
        }

        while let Some(node_key) = pending.pop() {
            let node_gdfo = self.nodes[&node_key].gdfo.unwrap();
            let child_keys = self.nodes[&node_key].child_keys.clone();
            for child_key in child_keys {
                let (known_gdfo, present) = match known_parent_gdfos.get(&child_key) {
                    Some(v) => (*v + 1, true),
                    None => (1, false),
                };
                let child = self.nodes.get_mut(&child_key).unwrap();
                let new_gdfo = node_gdfo + 1;
                if child.gdfo.is_none_or(|g| new_gdfo > g) {
                    child.gdfo = Some(new_gdfo);
                }
                let parent_len = child.parent_keys.as_ref().map(|p| p.len()).unwrap_or(0);
                if known_gdfo == parent_len {
                    pending.push(child_key.clone());
                    if present {
                        known_parent_gdfos.remove(&child_key);
                    }
                } else {
                    known_parent_gdfos.insert(child_key, known_gdfo);
                }
            }
        }
    }

    /// Return the parent keys for `key`. Returns `None` if `key` is a ghost,
    /// and an error-equivalent `None` lookup via `contains_key` otherwise.
    ///
    /// Matches the Python semantics: `None` means ghost, missing key would
    /// raise `KeyError` in Python — here the caller should check
    /// [`contains`](Self::contains) if disambiguation is needed.
    pub fn get_parent_keys(&self, key: &K) -> Option<&[K]> {
        self.nodes.get(key)?.parent_keys.as_deref()
    }

    /// Return the child keys for `key`. Returns an empty slice for tips.
    pub fn get_child_keys(&self, key: &K) -> Option<&[K]> {
        self.nodes.get(key).map(|n| n.child_keys.as_slice())
    }

    /// Return whether `key` is present in the graph at all (including ghosts).
    pub fn contains(&self, key: &K) -> bool {
        self.nodes.contains_key(key)
    }

    /// Return the number of nodes in the graph (including ghosts).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Return whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterate over all node keys in the graph (including ghosts).
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.nodes.keys()
    }

    /// Return the gdfo (greatest distance from origin) of `key`, if known.
    pub fn gdfo(&self, key: &K) -> Option<u64> {
        self.nodes.get(key).and_then(|n| n.gdfo)
    }

    /// Add a new node to the graph, possibly filling in a ghost.
    pub fn add_node(&mut self, key: K, parent_keys: Vec<K>) -> Result<(), Error<K>> {
        // Validate against existing state, then ensure the node exists with
        // its parents recorded. We hold off on inserting parents into the
        // graph until after this match, so the borrow of `existing` ends.
        match self.nodes.get_mut(&key) {
            Some(existing) => match &existing.parent_keys {
                Some(existing_parents) if existing_parents == &parent_keys => return Ok(()),
                Some(existing_parents) => {
                    return Err(Error::ParentMismatch {
                        expected: existing_parents.clone(),
                        actual: parent_keys,
                        key,
                    });
                }
                None => {
                    // Filling in a ghost: the heads cache is no longer
                    // trustworthy.
                    existing.parent_keys = Some(parent_keys.clone());
                    self.known_heads.clear();
                }
            },
            None => {
                self.nodes
                    .insert(key.clone(), KnownGraphNode::new(Some(parent_keys.clone())));
            }
        }

        let mut parent_gdfo: u64 = 0;
        for parent_key in &parent_keys {
            let parent_node = self.nodes.entry(parent_key.clone()).or_insert_with(|| {
                let mut n = KnownGraphNode::new(None);
                // Ghosts and roots have gdfo 1.
                n.gdfo = Some(1);
                n
            });
            if let Some(g) = parent_node.gdfo {
                parent_gdfo = parent_gdfo.max(g);
            }
            parent_node.child_keys.push(key.clone());
        }
        self.nodes.get_mut(&key).unwrap().gdfo = Some(parent_gdfo + 1);

        // Propagate gdfo updates to descendants (BFS).
        let mut pending: VecDeque<K> = VecDeque::new();
        pending.push_back(key);
        while let Some(node_key) = pending.pop_front() {
            let next_gdfo = self.nodes[&node_key].gdfo.unwrap() + 1;
            let child_keys = self.nodes[&node_key].child_keys.clone();
            for child_key in child_keys {
                let child = self.nodes.get_mut(&child_key).unwrap();
                if child.gdfo.is_none_or(|g| g < next_gdfo) {
                    child.gdfo = Some(next_gdfo);
                    pending.push_back(child_key);
                }
            }
        }
        Ok(())
    }

    /// Return the heads from amongst `keys`.
    ///
    /// Any key reachable from another key is filtered out. This method is
    /// sentinel-free on the core; the caller handles origin semantics by
    /// wrapping `K` in [`Key<K>`] and calling [`heads_with_origin`].
    ///
    /// All keys in `candidates` must be present in the graph (not ghosts).
    pub fn heads<I>(&mut self, candidates: I) -> FxHashSet<K>
    where
        I: IntoIterator<Item = K>,
    {
        let candidates: FxHashSet<K> = candidates.into_iter().collect();
        if candidates.len() < 2 {
            return candidates;
        }

        // Build a process-stable cache key by sorting candidates by their
        // hash. Hash collisions in the comparator just produce non-unique
        // orderings; the resulting Vec still uniquely identifies the input
        // set within a single process (different sets differ in length or
        // contents). We can't use BTreeSet here because K is not required
        // to be Ord.
        let heads_cache_key = sort_by_hash(candidates.iter().cloned());
        if let Some(cached) = self.known_heads.get(&heads_cache_key) {
            return cached.clone();
        }

        let mut seen: FxHashSet<K> = FxHashSet::default();
        let mut pending: Vec<K> = Vec::new();
        let mut min_gdfo: Option<u64> = None;
        for key in &candidates {
            let node = &self.nodes[key];
            if let Some(parents) = &node.parent_keys {
                pending.extend(parents.iter().cloned());
            }
            if let Some(g) = node.gdfo {
                min_gdfo = Some(min_gdfo.map_or(g, |m| m.min(g)));
            }
        }
        let min_gdfo = min_gdfo.unwrap_or(0);
        while let Some(node_key) = pending.pop() {
            if !seen.insert(node_key.clone()) {
                continue;
            }
            let node = &self.nodes[&node_key];
            if node.gdfo.is_some_and(|g| g <= min_gdfo) {
                continue;
            }
            if let Some(parents) = &node.parent_keys {
                pending.extend(parents.iter().cloned());
            }
        }
        let heads: FxHashSet<K> = candidates.difference(&seen).cloned().collect();
        if self.do_cache {
            self.known_heads.insert(heads_cache_key, heads.clone());
        }
        heads
    }

    /// Return the nodes of the graph in topological order (parents first).
    ///
    /// Errors with [`Error::Cycle`] if the graph is not fully connected via
    /// gdfo (i.e. contains a cycle).
    pub fn topo_sort(&self) -> Result<Vec<K>, Error<K>> {
        let unreachable: Vec<K> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.gdfo.is_none())
            .map(|(k, _)| k.clone())
            .collect();
        if !unreachable.is_empty() {
            return Err(Error::Cycle(unreachable));
        }
        let mut pending = self.find_tails();
        let mut num_seen_parents: FxHashMap<K, usize> =
            self.nodes.keys().map(|k| (k.clone(), 0)).collect();
        let mut topo_order: Vec<K> = Vec::new();
        while let Some(node_key) = pending.pop() {
            let node = &self.nodes[&node_key];
            // Skip ghosts in the output (matches Python behavior).
            if !node.is_ghost() {
                topo_order.push(node_key.clone());
            }
            let child_keys = node.child_keys.clone();
            for child_key in child_keys {
                let child = &self.nodes[&child_key];
                let seen_parents = num_seen_parents[&child_key] + 1;
                let parent_len = child.parent_keys.as_ref().map(|p| p.len()).unwrap_or(0);
                if seen_parents == parent_len {
                    pending.push(child_key.clone());
                    num_seen_parents.remove(&child_key);
                } else {
                    num_seen_parents.insert(child_key, seen_parents);
                }
            }
        }
        Ok(topo_order)
    }

    /// Return a reverse topological ordering grouped by prefix.
    ///
    /// `prefix_of` maps each key to its prefix bucket. Within each bucket the
    /// ordering is lexicographic (by `K: Ord`), which mirrors Python's use of
    /// tuple/bytes ordering there. Ghost nodes are skipped in the output.
    pub fn gc_sort<P, PFX>(&self, mut prefix_of: P) -> Vec<K>
    where
        K: Ord,
        P: FnMut(&K) -> PFX,
        PFX: Ord + Hash,
    {
        let mut prefix_tips: FxHashMap<PFX, Vec<K>> = FxHashMap::default();
        for key in self.find_tips() {
            prefix_tips.entry(prefix_of(&key)).or_default().push(key);
        }
        let mut num_seen_children: FxHashMap<K, usize> =
            self.nodes.keys().map(|k| (k.clone(), 0)).collect();

        let mut prefix_list: Vec<(PFX, Vec<K>)> = prefix_tips.into_iter().collect();
        prefix_list.sort_by(|a, b| a.0.cmp(&b.0));

        let mut result: Vec<K> = Vec::with_capacity(self.nodes.len());
        for (_prefix, tips) in prefix_list {
            // A min-heap (via Reverse) keeps the next-smallest key at the top
            // in O(log n), instead of re-sorting the pending vector after
            // every parent insertion.
            let mut pending: BinaryHeap<Reverse<K>> = tips.into_iter().map(Reverse).collect();
            while let Some(Reverse(node_key)) = pending.pop() {
                let node = &self.nodes[&node_key];
                if node.is_ghost() {
                    continue;
                }
                let parent_keys = node.parent_keys.as_deref().unwrap_or(&[]);
                for parent_key in parent_keys {
                    let parent_node = &self.nodes[parent_key];
                    let seen_children = num_seen_children[parent_key] + 1;
                    if seen_children == parent_node.child_keys.len() {
                        pending.push(Reverse(parent_key.clone()));
                        num_seen_children.remove(parent_key);
                    } else {
                        num_seen_children.insert(parent_key.clone(), seen_children);
                    }
                }
                result.push(node_key);
            }
        }
        result
    }
}

impl<K: Hash + Eq + Clone + std::fmt::Debug> KnownGraph<K> {
    /// Merge-sort the graph starting from `tip_key`.
    ///
    /// Requires `K: Debug` because the underlying [`MergeSorter`] does.
    pub fn merge_sort(&self, tip_key: K) -> Result<Vec<MergeSortNode<K>>, Error<K>> {
        let as_parent_map: HashMap<K, Vec<K>> = self
            .nodes
            .iter()
            .filter_map(|(k, n)| n.parent_keys.as_ref().map(|p| (k.clone(), p.clone())))
            .collect();
        MergeSorter::new(as_parent_map, Some(tip_key), None, true)
            .map(|item| {
                item.map(|(_, key, merge_depth, revno, end_of_merge)| MergeSortNode {
                    key,
                    merge_depth,
                    revno: revno.unwrap_or_default(),
                    end_of_merge,
                })
            })
            .collect()
    }
}

impl<K: Hash + Eq + Clone> KnownGraph<Key<K>> {
    /// `heads()` variant that implements the Python `NULL_REVISION` filter:
    /// [`Key::Origin`] is only a head if it is the sole candidate.
    pub fn heads_with_origin<I>(&mut self, candidates: I) -> FxHashSet<Key<K>>
    where
        I: IntoIterator<Item = Key<K>>,
    {
        let mut candidates: FxHashSet<Key<K>> = candidates.into_iter().collect();
        if candidates.contains(&Key::Origin) {
            candidates.remove(&Key::Origin);
            if candidates.is_empty() {
                let mut r = FxHashSet::default();
                r.insert(Key::Origin);
                return r;
            }
        }
        self.heads(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(edges: &[(&'static str, &[&'static str])]) -> KnownGraph<&'static str> {
        let pm = edges.iter().map(|(k, ps)| (*k, ps.to_vec()));
        KnownGraph::new(pm, true)
    }

    #[test]
    fn gdfo_linear() {
        // a -> b -> c
        let g = make(&[("a", &[]), ("b", &["a"]), ("c", &["b"])]);
        assert_eq!(g.gdfo(&"a"), Some(1));
        assert_eq!(g.gdfo(&"b"), Some(2));
        assert_eq!(g.gdfo(&"c"), Some(3));
    }

    #[test]
    fn gdfo_diamond() {
        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        let g = make(&[("a", &[]), ("b", &["a"]), ("c", &["a"]), ("d", &["b", "c"])]);
        assert_eq!(g.gdfo(&"a"), Some(1));
        assert_eq!(g.gdfo(&"b"), Some(2));
        assert_eq!(g.gdfo(&"c"), Some(2));
        assert_eq!(g.gdfo(&"d"), Some(3));
    }

    #[test]
    fn heads_trivial() {
        let mut g = make(&[("a", &[]), ("b", &["a"])]);
        let h = g.heads(vec!["a", "b"]);
        let expected: FxHashSet<_> = ["b"].iter().copied().collect();
        assert_eq!(h, expected);
    }

    #[test]
    fn heads_diamond() {
        let mut g = make(&[("a", &[]), ("b", &["a"]), ("c", &["a"]), ("d", &["b", "c"])]);
        let h = g.heads(vec!["b", "c"]);
        let expected: FxHashSet<_> = ["b", "c"].iter().copied().collect();
        assert_eq!(h, expected);
        let h2 = g.heads(vec!["a", "d"]);
        let expected2: FxHashSet<_> = ["d"].iter().copied().collect();
        assert_eq!(h2, expected2);
    }

    #[test]
    fn heads_with_origin_only() {
        let mut g: KnownGraph<Key<&'static str>> =
            KnownGraph::new(vec![(Key::Node("a"), vec![Key::Origin])], true);
        let h = g.heads_with_origin(vec![Key::Origin]);
        assert_eq!(h.len(), 1);
        assert!(h.contains(&Key::Origin));
    }

    #[test]
    fn heads_with_origin_ignored() {
        let mut g: KnownGraph<Key<&'static str>> =
            KnownGraph::new(vec![(Key::Node("a"), vec![Key::Origin])], true);
        let h = g.heads_with_origin(vec![Key::Origin, Key::Node("a")]);
        let expected: FxHashSet<_> = [Key::Node("a")].iter().cloned().collect();
        assert_eq!(h, expected);
    }

    #[test]
    fn topo_sort_basic() {
        let g = make(&[("a", &[]), ("b", &["a"]), ("c", &["a"]), ("d", &["b", "c"])]);
        let order = g.topo_sort().unwrap();
        // a must come before b, c; b, c must come before d.
        let pos = |x: &&str| order.iter().position(|n| n == x).unwrap();
        assert!(pos(&"a") < pos(&"b"));
        assert!(pos(&"a") < pos(&"c"));
        assert!(pos(&"b") < pos(&"d"));
        assert!(pos(&"c") < pos(&"d"));
    }

    #[test]
    fn add_node_fills_ghost() {
        // Start with b having ghost parent a.
        let mut g = make(&[("b", &["a"])]);
        // a is a ghost: present with None parents.
        assert!(g.get_parent_keys(&"a").is_none());
        g.add_node("a", vec![]).unwrap();
        assert_eq!(g.get_parent_keys(&"a"), Some(&[][..]));
        assert_eq!(g.gdfo(&"a"), Some(1));
        assert_eq!(g.gdfo(&"b"), Some(2));
    }

    #[test]
    fn add_node_duplicate_ok() {
        let mut g = make(&[("a", &[]), ("b", &["a"])]);
        g.add_node("b", vec!["a"]).unwrap();
    }

    #[test]
    fn add_node_mismatch_errors() {
        let mut g = make(&[("a", &[]), ("b", &["a"])]);
        let r = g.add_node("b", vec![]);
        assert!(matches!(r, Err(Error::ParentMismatch { .. })));
    }

    #[test]
    fn merge_sort_simple() {
        // a -> b -> c, linear
        let g = make(&[("a", &[]), ("b", &["a"]), ("c", &["b"])]);
        let ms = g.merge_sort("c").unwrap();
        let keys: Vec<_> = ms.iter().map(|n| n.key).collect();
        assert_eq!(keys, vec!["c", "b", "a"]);
    }
}
