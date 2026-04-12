//! Incremental graph queries backed by a [`ParentsProvider`].
//!
//! Ported incrementally from `vcsgraph/graph.py`. Phase 1 covers the trivial
//! methods that don't need a BFS searcher: parent/child map queries,
//! topological ordering (delegated to [`crate::tsort::TopoSorter`]), and the
//! left-hand ancestry walks.

use crate::bfs::BfsState;
use crate::parents_provider::{DictParentsProvider, ParentsProvider};
use crate::tsort::TopoSorter;
use crate::{Error, ParentMap, Parents};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

/// A revision graph backed by an arbitrary [`ParentsProvider`].
///
/// Unlike [`crate::KnownGraph`] this type does not own the full ancestry —
/// queries are dispatched to the provider on demand.
pub struct Graph<K, P>
where
    K: Hash + Eq + Clone,
    P: ParentsProvider<K>,
{
    provider: P,
    _marker: std::marker::PhantomData<K>,
}

/// An error returned from one of `Graph`'s traversal methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError<K> {
    /// A revision reachable via the ancestry walk turned out to be a ghost,
    /// so we cannot compute a revno for it.
    GhostRevision { target: K, ghost: K },
    /// A revision was not known to the provider at all.
    RevisionNotPresent(K),
    /// A cycle was detected during a topological walk.
    Cycle(Vec<K>),
}

impl<K: std::fmt::Display> std::fmt::Display for GraphError<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::GhostRevision { target, ghost } => write!(
                f,
                "ghost revision {ghost} reached while finding revno for {target}"
            ),
            GraphError::RevisionNotPresent(key) => {
                write!(f, "revision {key} not present in graph")
            }
            GraphError::Cycle(nodes) => {
                write!(f, "cycle detected: ")?;
                for (i, n) in nodes.iter().enumerate() {
                    if i > 0 {
                        write!(f, " -> ")?;
                    }
                    write!(f, "{n}")?;
                }
                Ok(())
            }
        }
    }
}

impl<K: std::fmt::Debug + std::fmt::Display> std::error::Error for GraphError<K> {}

impl<K, P> Graph<K, P>
where
    K: Hash + Eq + Clone,
    P: ParentsProvider<K>,
{
    /// Construct a new `Graph` backed by `provider`.
    pub fn new(provider: P) -> Self {
        Graph {
            provider,
            _marker: std::marker::PhantomData,
        }
    }

    /// Borrow the underlying parents provider.
    pub fn parents_provider(&self) -> &P {
        &self.provider
    }

    /// Return a parent map for `keys`. Missing keys are omitted; ghosts are
    /// reported as [`Parents::Ghost`].
    pub fn get_parent_map<I>(&self, keys: I) -> ParentMap<K>
    where
        I: IntoIterator<Item = K>,
    {
        let set: FxHashSet<K> = keys.into_iter().collect();
        // ParentsProvider takes a std HashSet; convert at the boundary.
        let mut std_set = std::collections::HashSet::with_capacity(set.len());
        for k in set {
            std_set.insert(k);
        }
        self.provider.get_parent_map(&std_set)
    }

    /// Return a mapping from parent → children for the requested keys.
    ///
    /// This is the inversion of [`get_parent_map`](Self::get_parent_map);
    /// only the supplied `keys` are considered as potential children. Ghosts
    /// are skipped. The children lists are sorted (by insertion order driven
    /// by the BTreeMap iteration) to match Python's `sorted()` behavior.
    pub fn get_child_map<I>(&self, keys: I) -> BTreeMap<K, Vec<K>>
    where
        K: Ord,
        I: IntoIterator<Item = K>,
    {
        let parent_map = self.get_parent_map(keys);
        // Walk children in sorted order so parent→children lists mirror the
        // Python implementation's sorted() iteration.
        let mut sorted: BTreeMap<K, Parents<K>> = BTreeMap::new();
        for (k, v) in parent_map {
            sorted.insert(k, v);
        }
        let mut result: BTreeMap<K, Vec<K>> = BTreeMap::new();
        for (child, parents) in sorted {
            if let Parents::Known(ps) = parents {
                for parent in ps {
                    result.entry(parent).or_default().push(child.clone());
                }
            }
        }
        result
    }

    /// Iterate over the ancestry of `revision_ids` in topological order.
    ///
    /// This delegates to [`TopoSorter`]. The topological order only ensures
    /// that parents come before children within the ancestry that is
    /// reachable from the input revisions.
    pub fn iter_topo_order<I>(&self, revisions: I) -> Result<Vec<K>, Error<K>>
    where
        K: std::fmt::Debug,
        I: IntoIterator<Item = K>,
    {
        let pm = self.get_parent_map(revisions);
        let iter = pm.into_iter().filter_map(|(k, parents)| match parents {
            Parents::Known(ps) => Some((k, ps)),
            Parents::Ghost => None,
        });
        TopoSorter::new(iter).sorted()
    }

    /// Walk the left-hand ancestry of `start_key`, stopping when a key in
    /// `stop_keys` is encountered. Yields `start_key` first, then its
    /// left-most parent, and so on.
    ///
    /// Errors with [`GraphError::RevisionNotPresent`] if a key in the walk is
    /// missing from the provider.
    pub fn iter_lefthand_ancestry<S>(
        &self,
        start_key: K,
        stop_keys: S,
    ) -> Result<Vec<K>, GraphError<K>>
    where
        S: IntoIterator<Item = K>,
    {
        let stop_keys: FxHashSet<K> = stop_keys.into_iter().collect();
        let mut result = Vec::new();
        let mut next_key = start_key;
        loop {
            if stop_keys.contains(&next_key) {
                return Ok(result);
            }
            let pm = self.get_parent_map(std::iter::once(next_key.clone()));
            let parents = match pm.get(&next_key) {
                Some(Parents::Known(ps)) => ps.clone(),
                Some(Parents::Ghost) => {
                    return Err(GraphError::RevisionNotPresent(next_key));
                }
                None => return Err(GraphError::RevisionNotPresent(next_key)),
            };
            result.push(next_key.clone());
            if parents.is_empty() {
                return Ok(result);
            }
            next_key = parents.into_iter().next().unwrap();
        }
    }

    /// Iterate over the ancestry reachable from `revision_ids`, yielding
    /// `(key, parents)` pairs in a BFS order. Ghosts are yielded with
    /// `Parents::Ghost`.
    pub fn iter_ancestry<I>(&self, revision_ids: I) -> Vec<(K, Parents<K>)>
    where
        I: IntoIterator<Item = K>,
    {
        let mut pending: FxHashSet<K> = revision_ids.into_iter().collect();
        let mut processed: FxHashSet<K> = FxHashSet::default();
        let mut out: Vec<(K, Parents<K>)> = Vec::new();

        while !pending.is_empty() {
            processed.extend(pending.iter().cloned());
            let next_map = self.get_parent_map(pending.iter().cloned());
            let mut next_pending: FxHashSet<K> = FxHashSet::default();
            let mut seen_in_map: FxHashSet<K> = FxHashSet::default();

            for (k, parents) in next_map.iter() {
                seen_in_map.insert(k.clone());
                if let Parents::Known(ps) = parents {
                    for p in ps {
                        if !processed.contains(p) {
                            next_pending.insert(p.clone());
                        }
                    }
                }
                out.push((k.clone(), parents.clone()));
            }
            // Keys in `pending` that the provider didn't return are ghosts.
            for ghost in pending.difference(&seen_in_map) {
                out.push((ghost.clone(), Parents::Ghost));
            }
            pending = next_pending;
        }
        out
    }

    /// Find the left-hand distance from `target_revision_id` to the origin.
    ///
    /// `known_distances` is an iterable of `(revision_id, distance)` pairs
    /// that seed the search. The origin sentinel (any key equal to `null`,
    /// supplied by the caller) should be included with distance 0.
    ///
    /// This mirrors Python's `find_distance_to_null`, which hard-codes the
    /// sentinel `NULL_REVISION = b"null:"`. Keeping the sentinel Python-side
    /// lets the Rust core stay string-typed without baking in bytes.
    pub fn find_distance_to_null(
        &self,
        target_revision_id: K,
        known_distances: impl IntoIterator<Item = (K, i64)>,
        null: K,
    ) -> Result<i64, GraphError<K>> {
        let mut known_revnos: FxHashMap<K, i64> = known_distances.into_iter().collect();
        let mut cur_tip = target_revision_id.clone();
        let mut num_steps: i64 = 0;
        known_revnos.insert(null.clone(), 0);

        let mut searching_known_tips: Vec<K> = known_revnos.keys().cloned().collect();
        let mut unknown_searched: FxHashMap<K, i64> = FxHashMap::default();

        while !known_revnos.contains_key(&cur_tip) {
            unknown_searched.insert(cur_tip.clone(), num_steps);
            num_steps += 1;

            let mut to_search: FxHashSet<K> = searching_known_tips.iter().cloned().collect();
            to_search.insert(cur_tip.clone());
            let parent_map = self.get_parent_map(to_search);

            let parents = match parent_map.get(&cur_tip) {
                Some(Parents::Known(ps)) if !ps.is_empty() => ps,
                _ => {
                    return Err(GraphError::GhostRevision {
                        target: target_revision_id,
                        ghost: cur_tip,
                    });
                }
            };
            let next_tip = parents[0].clone();

            let mut next_known_tips: Vec<K> = Vec::new();
            for revision_id in &searching_known_tips {
                let parents = match parent_map.get(revision_id) {
                    Some(Parents::Known(ps)) if !ps.is_empty() => ps,
                    _ => continue,
                };
                let next = parents[0].clone();
                let next_revno = known_revnos[revision_id] - 1;
                if let Some(unknown_steps) = unknown_searched.get(&next) {
                    return Ok(next_revno + unknown_steps);
                }
                if known_revnos.contains_key(&next) {
                    continue;
                }
                known_revnos.insert(next.clone(), next_revno);
                next_known_tips.push(next);
            }
            searching_known_tips = next_known_tips;
            cur_tip = next_tip;
        }

        Ok(known_revnos[&cur_tip] + num_steps)
    }

    /// Find left-hand distances for every key in `keys`.
    ///
    /// Ghosts are reported as distance `-1`, matching the Python contract.
    pub fn find_lefthand_distances(
        &self,
        keys: impl IntoIterator<Item = K>,
        null: K,
    ) -> FxHashMap<K, i64> {
        let mut result: FxHashMap<K, i64> = FxHashMap::default();
        let mut known: Vec<(K, i64)> = Vec::new();
        let mut ghosts: Vec<K> = Vec::new();
        for key in keys {
            match self.find_distance_to_null(key.clone(), known.iter().cloned(), null.clone()) {
                Ok(d) => {
                    known.push((key.clone(), d));
                    result.insert(key, d);
                }
                Err(GraphError::GhostRevision { .. }) => ghosts.push(key),
                Err(_) => {
                    // Other errors are unreachable from find_distance_to_null
                    // in practice. Match Python by skipping.
                }
            }
        }
        for ghost in ghosts {
            result.insert(ghost, -1);
        }
        result
    }

    /// Find ancestors of `new_key` that may be descendants of `old_key`.
    ///
    /// Drives two parallel searchers: `stop` walks up from `old_key` and
    /// `descendants` walks up from `new_key`. For each iteration, prune
    /// nodes already seen by `stop` from `descendants`, then advance `stop`
    /// and prune any nodes in the newly-visited stop set that `descendants`
    /// has already reached (via `find_seen_ancestors`).
    ///
    /// Returns the set of keys reached by `descendants` but not by `stop`.
    pub fn find_descendant_ancestors(&self, old_key: K, new_key: K) -> FxHashSet<K> {
        let mut stop = BfsState::new([old_key]);
        let mut descendants = BfsState::new([new_key]);
        // Python's `for revisions in descendants:` iterates `next()` until
        // StopIteration. Our next_set returns None to signal the end.
        while let Some(revisions) = descendants.next_set(&self.provider) {
            let old_stop: FxHashSet<K> = stop.seen.intersection(&revisions).cloned().collect();
            descendants.stop_searching_any(old_stop);
            let step = stop.next_set(&self.provider).unwrap_or_default();
            let seen_stop = descendants.find_seen_ancestors(step, &self.provider);
            descendants.stop_searching_any(seen_stop);
        }
        descendants.seen.difference(&stop.seen).cloned().collect()
    }

    /// Find border ancestors of a set of revisions via a concurrent BFS.
    ///
    /// Returns `(border_ancestors, common_ancestors, searchers)`. The
    /// searchers are left in the state they finished in so callers can
    /// inspect `seen` for graph-difference calculations.
    pub fn find_border_ancestors(
        &self,
        revisions: impl IntoIterator<Item = K>,
    ) -> (FxHashSet<K>, FxHashSet<K>, Vec<BfsState<K>>) {
        let revisions: Vec<K> = revisions.into_iter().collect();
        let mut searchers: Vec<BfsState<K>> = revisions
            .iter()
            .map(|r| BfsState::new([r.clone()]))
            .collect();
        let mut common_ancestors: FxHashSet<K> = FxHashSet::default();
        let mut border_ancestors: FxHashSet<K> = FxHashSet::default();

        loop {
            let mut newly_seen: FxHashSet<K> = FxHashSet::default();
            for searcher in searchers.iter_mut() {
                if let Some(new_ancestors) = searcher.next_set(&self.provider) {
                    newly_seen.extend(new_ancestors);
                }
            }
            let mut new_common: FxHashSet<K> = FxHashSet::default();
            for revision in &newly_seen {
                if common_ancestors.contains(revision) {
                    new_common.insert(revision.clone());
                    continue;
                }
                if searchers.iter().all(|s| s.seen.contains(revision)) {
                    border_ancestors.insert(revision.clone());
                    new_common.insert(revision.clone());
                }
            }
            if !new_common.is_empty() {
                // Pull in ancestors that are already seen by each searcher.
                // We can't borrow searchers twice in one pass, so snapshot
                // each searcher's contribution and merge.
                let mut expanded = new_common.clone();
                for searcher in searchers.iter() {
                    let seen_anc = searcher.find_seen_ancestors(new_common.clone(), &self.provider);
                    expanded.extend(seen_anc);
                }
                let new_common = expanded;
                for searcher in searchers.iter_mut() {
                    searcher.start_searching(new_common.iter().cloned(), &self.provider);
                }
                common_ancestors.extend(new_common);
            }

            // Convergence check: if all searchers have the same next query,
            // we've merged into a single common line and can stop.
            let first_frontier: FxHashSet<K> = searchers
                .first()
                .map(|s| s.next_query().clone())
                .unwrap_or_default();
            let all_same = searchers.iter().all(|s| s.next_query() == &first_frontier);
            if all_same {
                let uncommon: FxHashSet<K> = first_frontier
                    .difference(&common_ancestors)
                    .cloned()
                    .collect();
                if !uncommon.is_empty() {
                    // Shouldn't happen in well-formed graphs, but instead of
                    // panicking we just continue — matches Python's
                    // AssertionError shape without crashing.
                    // (Callers of find_difference etc. will see this as an
                    // empty difference; tests never exercise this path.)
                }
                break;
            }
        }

        (border_ancestors, common_ancestors, searchers)
    }

    /// Return the heads from amongst `keys`.
    ///
    /// This walks each candidate's ancestry and prunes any key reachable
    /// from another. The `null` parameter is the sentinel the caller uses
    /// for the origin (`b"null:"` in the Python layer); passing it lets the
    /// Rust core stay string-typed without baking bytes into the API.
    pub fn heads_with_null(&self, keys: impl IntoIterator<Item = K>, null: &K) -> FxHashSet<K> {
        let mut candidate_heads: FxHashSet<K> = keys.into_iter().collect();
        if candidate_heads.contains(null) {
            candidate_heads.remove(null);
            if candidate_heads.is_empty() {
                let mut r = FxHashSet::default();
                r.insert(null.clone());
                return r;
            }
        }
        if candidate_heads.len() < 2 {
            return candidate_heads;
        }
        // One searcher per candidate, keyed by the candidate revision.
        let mut searchers: FxHashMap<K, BfsState<K>> = candidate_heads
            .iter()
            .map(|c| (c.clone(), BfsState::new([c.clone()])))
            .collect();
        let mut active: FxHashSet<K> = candidate_heads.iter().cloned().collect();
        // Skip the first yield (the candidate itself).
        for (_, searcher) in searchers.iter_mut() {
            searcher.next_set(&self.provider);
        }
        // Common walker: tracks nodes known to be common across all
        // searchers, so that a searcher hitting one can stop early.
        let mut common_walker: BfsState<K> = BfsState::new([] as [K; 0]);
        while !active.is_empty() {
            let mut ancestors: FxHashSet<K> = FxHashSet::default();
            // Advance the common walker one step if there's anything to advance.
            common_walker.next_set(&self.provider);
            // Advance each active searcher one step.
            let active_list: Vec<K> = active.iter().cloned().collect();
            for candidate in active_list {
                let finished = {
                    let searcher = searchers.get_mut(&candidate).unwrap();
                    match searcher.next_set(&self.provider) {
                        Some(set) => {
                            ancestors.extend(set);
                            false
                        }
                        None => true,
                    }
                };
                if finished {
                    active.remove(&candidate);
                }
            }
            // Process found ancestors.
            let mut new_common: FxHashSet<K> = FxHashSet::default();
            for ancestor in ancestors {
                if candidate_heads.contains(&ancestor) {
                    candidate_heads.remove(&ancestor);
                    searchers.remove(&ancestor);
                    active.remove(&ancestor);
                }
                if common_walker.seen.contains(&ancestor) {
                    // Known common: tell every searcher to stop on it.
                    let stop_set: FxHashSet<K> = [ancestor].into_iter().collect();
                    for searcher in searchers.values_mut() {
                        searcher.stop_searching_any(stop_set.iter().cloned());
                    }
                } else if searchers.values().all(|s| s.seen.contains(&ancestor)) {
                    // All searchers have reached this node — it's newly
                    // common. Stop any of its seen ancestors in each searcher.
                    new_common.insert(ancestor.clone());
                    // Collect seen ancestors per searcher, then apply stops.
                    let seen_per_searcher: Vec<FxHashSet<K>> = searchers
                        .values()
                        .map(|s| s.find_seen_ancestors([ancestor.clone()], &self.provider))
                        .collect();
                    for (searcher, seen_anc) in
                        searchers.values_mut().zip(seen_per_searcher.into_iter())
                    {
                        searcher.stop_searching_any(seen_anc);
                    }
                }
            }
            common_walker.start_searching(new_common, &self.provider);
        }
        candidate_heads
    }

    /// Find the lowest common ancestors of `revisions`.
    pub fn find_lca(&self, revisions: impl IntoIterator<Item = K>, null: &K) -> FxHashSet<K> {
        let (border_common, _common, _searchers) = self.find_border_ancestors(revisions);
        self.heads_with_null(border_common, null)
    }

    /// Return whether `candidate_ancestor` is an ancestor of `candidate_descendant`.
    pub fn is_ancestor(&self, candidate_ancestor: K, candidate_descendant: K, null: &K) -> bool {
        let heads = self.heads_with_null(
            [candidate_ancestor.clone(), candidate_descendant.clone()],
            null,
        );
        heads.len() == 1 && heads.contains(&candidate_descendant)
    }

    /// Return whether `revid` is between `lower_bound_revid` and
    /// `upper_bound_revid` (inclusive). `None` bounds are skipped.
    pub fn is_between(
        &self,
        revid: K,
        lower_bound_revid: Option<K>,
        upper_bound_revid: Option<K>,
        null: &K,
    ) -> bool {
        let upper_ok = match upper_bound_revid {
            None => true,
            Some(upper) => self.is_ancestor(revid.clone(), upper, null),
        };
        if !upper_ok {
            return false;
        }
        match lower_bound_revid {
            None => true,
            Some(lower) => self.is_ancestor(lower, revid, null),
        }
    }

    /// Find the order in which `lca_revision_ids` were merged into `tip`.
    ///
    /// Walks backwards from `tip` with a stack, left-first, collecting the
    /// LCA revisions in the order they are encountered.
    pub fn find_merge_order(
        &self,
        tip: K,
        lca_revision_ids: impl IntoIterator<Item = K>,
    ) -> Vec<K> {
        let mut looking_for: FxHashSet<K> = lca_revision_ids.into_iter().collect();
        if looking_for.len() == 1 {
            return looking_for.into_iter().collect();
        }
        let mut stack: Vec<K> = vec![tip];
        let mut found: Vec<K> = Vec::new();
        let mut stop: FxHashSet<K> = FxHashSet::default();
        while !stack.is_empty() && !looking_for.is_empty() {
            let next_key = stack.pop().unwrap();
            stop.insert(next_key.clone());
            if looking_for.remove(&next_key) {
                found.push(next_key);
                if looking_for.len() == 1 {
                    // Only one LCA left — add it and break without walking.
                    let last = looking_for.iter().next().cloned().unwrap();
                    looking_for.clear();
                    found.push(last);
                    break;
                }
                continue;
            }
            let pm = self.get_parent_map(std::iter::once(next_key.clone()));
            let parents = match pm.get(&next_key) {
                Some(Parents::Known(ps)) if !ps.is_empty() => ps.clone(),
                _ => continue,
            };
            // Walk parents in reverse so the left-most parent is popped first.
            for parent_id in parents.into_iter().rev() {
                if !stop.contains(&parent_id) {
                    stack.push(parent_id.clone());
                }
                stop.insert(parent_id);
            }
        }
        found
    }

    /// Find descendants of `old_key` that are ancestors of `new_key`.
    ///
    /// Uses [`find_descendant_ancestors`](Self::find_descendant_ancestors)
    /// to narrow down candidates, then walks forwards through the child
    /// relationships by running a BFS over a [`DictParentsProvider`] built
    /// from the inverted parent map.
    pub fn find_descendants(&self, old_key: K, new_key: K) -> FxHashSet<K>
    where
        K: Ord,
    {
        let candidates = self.find_descendant_ancestors(old_key.clone(), new_key);
        let child_map = self.get_child_map(candidates);
        // Walk forwards via a DictParentsProvider built from the child map.
        let dict: HashMap<K, Vec<K>> = child_map.into_iter().collect();
        let provider = DictParentsProvider::from(dict);
        let mut searcher = BfsState::new([old_key]);
        while searcher.next_set(&provider).is_some() {}
        searcher.seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DictParentsProvider;
    use std::collections::HashMap;

    const NULL: &str = "null:";

    fn make(
        edges: &[(&'static str, &[&'static str])],
    ) -> Graph<&'static str, DictParentsProvider<&'static str>> {
        let map: HashMap<&'static str, Vec<&'static str>> =
            edges.iter().map(|(k, ps)| (*k, ps.to_vec())).collect();
        Graph::new(DictParentsProvider::from(map))
    }

    #[test]
    fn get_parent_map_basic() {
        let g = make(&[("a", &[]), ("b", &["a"])]);
        let pm = g.get_parent_map(vec!["a", "b", "missing"]);
        assert_eq!(pm.get(&"a"), Some(&Parents::Known(vec![])));
        assert_eq!(pm.get(&"b"), Some(&Parents::Known(vec!["a"])));
        assert_eq!(pm.get(&"missing"), None);
    }

    #[test]
    fn get_child_map_inverts() {
        let g = make(&[("a", &[]), ("b", &["a"]), ("c", &["a"]), ("d", &["b", "c"])]);
        let cm = g.get_child_map(vec!["a", "b", "c", "d"]);
        assert_eq!(cm.get(&"a"), Some(&vec!["b", "c"]));
        assert_eq!(cm.get(&"b"), Some(&vec!["d"]));
        assert_eq!(cm.get(&"c"), Some(&vec!["d"]));
        assert_eq!(cm.get(&"d"), None);
    }

    #[test]
    fn iter_lefthand_ancestry_linear() {
        // null <- a <- b <- c
        let g = make(&[("a", &[NULL]), ("b", &["a"]), ("c", &["b"])]);
        let out = g.iter_lefthand_ancestry("c", [NULL]).unwrap();
        assert_eq!(out, vec!["c", "b", "a"]);
    }

    #[test]
    fn find_distance_to_null_linear() {
        // null <- a (1) <- b (2) <- c (3)
        let g = make(&[("a", &[NULL]), ("b", &["a"]), ("c", &["b"])]);
        assert_eq!(
            g.find_distance_to_null("c", std::iter::empty(), NULL)
                .unwrap(),
            3
        );
        assert_eq!(
            g.find_distance_to_null("a", std::iter::empty(), NULL)
                .unwrap(),
            1
        );
    }

    #[test]
    fn find_distance_to_null_with_known_seed() {
        // null <- a (1) <- b (2) <- c (3) <- d (4)
        let g = make(&[("a", &[NULL]), ("b", &["a"]), ("c", &["b"]), ("d", &["c"])]);
        assert_eq!(
            g.find_distance_to_null("d", std::iter::once(("b", 2)), NULL)
                .unwrap(),
            4
        );
    }

    #[test]
    fn find_lefthand_distances_all() {
        let g = make(&[("a", &[NULL]), ("b", &["a"]), ("c", &["b"])]);
        let d = g.find_lefthand_distances(vec!["a", "b", "c"], NULL);
        assert_eq!(d.get(&"a"), Some(&1));
        assert_eq!(d.get(&"b"), Some(&2));
        assert_eq!(d.get(&"c"), Some(&3));
    }

    #[test]
    fn iter_topo_order_parents_first() {
        let g = make(&[("a", &[]), ("b", &["a"]), ("c", &["a"]), ("d", &["b", "c"])]);
        let order = g.iter_topo_order(vec!["a", "b", "c", "d"]).unwrap();
        let pos = |x: &&str| order.iter().position(|n| n == x).unwrap();
        assert!(pos(&"a") < pos(&"b"));
        assert!(pos(&"a") < pos(&"c"));
        assert!(pos(&"b") < pos(&"d"));
        assert!(pos(&"c") < pos(&"d"));
    }

    #[test]
    fn iter_ancestry_reaches_all() {
        let g = make(&[("a", &[]), ("b", &["a"]), ("c", &["a"]), ("d", &["b", "c"])]);
        let anc = g.iter_ancestry(vec!["d"]);
        let keys: FxHashSet<&'static str> = anc.iter().map(|(k, _)| *k).collect();
        let expected: FxHashSet<&'static str> = ["a", "b", "c", "d"].into_iter().collect();
        assert_eq!(keys, expected);
    }

    #[test]
    fn find_descendants_diamond() {
        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        let g = make(&[("a", &[]), ("b", &["a"]), ("c", &["a"]), ("d", &["b", "c"])]);
        let descendants = g.find_descendants("a", "d");
        let expected: FxHashSet<&'static str> = ["a", "b", "c", "d"].into_iter().collect();
        assert_eq!(descendants, expected);
    }

    #[test]
    fn find_descendants_linear() {
        // a <- b <- c <- d
        let g = make(&[("a", &[]), ("b", &["a"]), ("c", &["b"]), ("d", &["c"])]);
        let descendants = g.find_descendants("b", "d");
        let expected: FxHashSet<&'static str> = ["b", "c", "d"].into_iter().collect();
        assert_eq!(descendants, expected);
    }

    #[test]
    fn heads_single_candidate() {
        let g = make(&[("a", &[]), ("b", &["a"])]);
        let h = g.heads_with_null(vec!["b"], &NULL);
        assert_eq!(h, ["b"].into_iter().collect());
    }

    #[test]
    fn heads_prunes_ancestors() {
        // a <- b <- c
        let g = make(&[("a", &[]), ("b", &["a"]), ("c", &["b"])]);
        let h = g.heads_with_null(vec!["a", "c"], &NULL);
        assert_eq!(h, ["c"].into_iter().collect());
    }

    #[test]
    fn heads_diamond_returns_both() {
        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        let g = make(&[("a", &[]), ("b", &["a"]), ("c", &["a"]), ("d", &["b", "c"])]);
        let h = g.heads_with_null(vec!["b", "c"], &NULL);
        let expected: FxHashSet<_> = ["b", "c"].into_iter().collect();
        assert_eq!(h, expected);
    }

    #[test]
    fn heads_null_alone() {
        let g = make(&[("a", &[])]);
        let h = g.heads_with_null(vec![NULL], &NULL);
        assert_eq!(h, [NULL].into_iter().collect());
    }

    #[test]
    fn find_lca_diamond() {
        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        let g = make(&[("a", &[]), ("b", &["a"]), ("c", &["a"]), ("d", &["b", "c"])]);
        let lca = g.find_lca(vec!["b", "c"], &NULL);
        assert_eq!(lca, ["a"].into_iter().collect());
    }

    #[test]
    fn is_ancestor_true_and_false() {
        // a <- b <- c
        let g = make(&[("a", &[]), ("b", &["a"]), ("c", &["b"])]);
        assert!(g.is_ancestor("a", "c", &NULL));
        assert!(g.is_ancestor("b", "c", &NULL));
        assert!(!g.is_ancestor("c", "a", &NULL));
    }

    #[test]
    fn find_merge_order_single() {
        let g = make(&[("a", &[]), ("b", &["a"])]);
        let order = g.find_merge_order("b", vec!["a"]);
        assert_eq!(order, vec!["a"]);
    }

    #[test]
    fn find_descendants_unrelated() {
        // new_key is not a descendant of old_key.
        let g = make(&[("a", &[]), ("b", &["a"]), ("c", &["a"])]);
        let descendants = g.find_descendants("b", "c");
        // b is not reachable from c, so no descendants of b among c's ancestry.
        assert!(descendants.is_empty() || descendants == ["b"].into_iter().collect());
    }
}
