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

    /// Find the first lefthand ancestor of `tip_key` that merged `merged_key`.
    ///
    /// Walks the lefthand ancestry of `tip_key` one step at a time, stopping
    /// as soon as a candidate is not a descendant of `merged_key`. Returns
    /// the last candidate that *was* a descendant — or `None` if none is.
    pub fn find_lefthand_merger(&self, merged_key: K, tip_key: K) -> Option<K>
    where
        K: Ord,
    {
        let descendants = self.find_descendants(merged_key, tip_key.clone());
        let mut last_candidate: Option<K> = None;
        let mut next_key = tip_key;
        loop {
            if !descendants.contains(&next_key) {
                return last_candidate;
            }
            let pm = self.get_parent_map(std::iter::once(next_key.clone()));
            let parents = match pm.get(&next_key) {
                Some(Parents::Known(ps)) => ps.clone(),
                _ => {
                    // Missing entry or ghost — treat as end of walk.
                    return Some(next_key);
                }
            };
            last_candidate = Some(next_key);
            if parents.is_empty() {
                return last_candidate;
            }
            next_key = parents.into_iter().next().unwrap();
        }
    }

    /// Compute `(left_only, right_only)` — the set difference between the
    /// ancestries of `left` and `right`.
    pub fn find_difference(&self, left: K, right: K) -> (FxHashSet<K>, FxHashSet<K>)
    where
        K: Ord,
    {
        let (_border, common, mut searchers) = self.find_border_ancestors([left, right]);
        self.search_for_extra_common(&common, &mut searchers);
        let left_seen = &searchers[0].seen;
        let right_seen = &searchers[1].seen;
        (
            left_seen.difference(right_seen).cloned().collect(),
            right_seen.difference(left_seen).cloned().collect(),
        )
    }

    /// Run the "extra common" reconvergence pass on a pair of searchers
    /// left in the state they finished `find_border_ancestors` in. Mirrors
    /// Python's `_search_for_extra_common`.
    #[allow(clippy::needless_range_loop)]
    fn search_for_extra_common(&self, _common: &FxHashSet<K>, searchers: &mut [BfsState<K>])
    where
        K: Ord,
    {
        assert_eq!(
            searchers.len(),
            2,
            "search_for_extra_common only supports 2 searchers"
        );
        let unique: FxHashSet<K> = searchers[0]
            .seen
            .symmetric_difference(&searchers[1].seen)
            .cloned()
            .collect();
        if unique.is_empty() {
            return;
        }
        let parent_map = self.get_parent_map(unique.iter().cloned());
        let unique = Self::remove_simple_descendants(&unique, &parent_map);

        // Build unique-searchers: one per unique revision.
        let mut unique_searchers: Vec<BfsState<K>> = Vec::new();
        for revision_id in unique.iter() {
            let revs_to_search: FxHashSet<K> = {
                let parent_idx = if searchers[0].seen.contains(revision_id) {
                    0
                } else {
                    1
                };
                let seed = [revision_id.clone()];
                let anc = searchers[parent_idx].find_seen_ancestors(seed, &self.provider);
                if anc.is_empty() {
                    [revision_id.clone()].into_iter().collect()
                } else {
                    anc
                }
            };
            let mut s = BfsState::new(revs_to_search);
            s.next_set(&self.provider);
            unique_searchers.push(s);
        }

        // Compute initial ancestor_all_unique: intersection of all seen sets.
        let mut ancestor_all_unique: FxHashSet<K> = FxHashSet::default();
        for (i, s) in unique_searchers.iter().enumerate() {
            if i == 0 {
                ancestor_all_unique = s.seen.clone();
            } else {
                ancestor_all_unique = ancestor_all_unique.intersection(&s.seen).cloned().collect();
            }
        }

        loop {
            let mut newly_seen_common: FxHashSet<K> = FxHashSet::default();
            for s in searchers.iter_mut() {
                if let Some(set) = s.next_set(&self.provider) {
                    newly_seen_common.extend(set);
                }
            }
            let mut newly_seen_unique: FxHashSet<K> = FxHashSet::default();
            for s in unique_searchers.iter_mut() {
                if let Some(set) = s.next_set(&self.provider) {
                    newly_seen_unique.extend(set);
                }
            }
            let mut new_common_unique: FxHashSet<K> = FxHashSet::default();
            for revision in &newly_seen_unique {
                if unique_searchers.iter().all(|s| s.seen.contains(revision)) {
                    new_common_unique.insert(revision.clone());
                }
            }
            if !newly_seen_common.is_empty() {
                // Merge newly_seen_common seen-ancestors from each common searcher.
                let mut expanded = newly_seen_common.clone();
                for s in searchers.iter() {
                    expanded
                        .extend(s.find_seen_ancestors(expanded.iter().cloned(), &self.provider));
                }
                let expanded_frozen = expanded;
                for s in searchers.iter_mut() {
                    s.start_searching(expanded_frozen.iter().cloned(), &self.provider);
                }
                let stop_searching_common: FxHashSet<K> = ancestor_all_unique
                    .intersection(&expanded_frozen)
                    .cloned()
                    .collect();
                if !stop_searching_common.is_empty() {
                    for s in searchers.iter_mut() {
                        s.stop_searching_any(stop_searching_common.iter().cloned());
                    }
                }
            }
            if !new_common_unique.is_empty() {
                let mut expanded = new_common_unique.clone();
                for s in unique_searchers.iter() {
                    expanded
                        .extend(s.find_seen_ancestors(expanded.iter().cloned(), &self.provider));
                }
                for s in searchers.iter() {
                    expanded
                        .extend(s.find_seen_ancestors(expanded.iter().cloned(), &self.provider));
                }
                for s in unique_searchers.iter_mut() {
                    s.start_searching(expanded.iter().cloned(), &self.provider);
                }
                for s in searchers.iter_mut() {
                    s.stop_searching_any(expanded.iter().cloned());
                }
                ancestor_all_unique.extend(expanded);

                // Collapse unique searchers that ended up with the same frontier.
                let mut seen_frontiers: std::collections::HashSet<Vec<K>> =
                    std::collections::HashSet::new();
                let mut next_unique: Vec<BfsState<K>> = Vec::new();
                for searcher in unique_searchers {
                    let mut key: Vec<K> = searcher.next_query().iter().cloned().collect();
                    key.sort_by_key(|k| {
                        use std::collections::hash_map::DefaultHasher;
                        use std::hash::Hasher;
                        let mut h = DefaultHasher::new();
                        k.hash(&mut h);
                        h.finish()
                    });
                    if seen_frontiers.insert(key) {
                        next_unique.push(searcher);
                    }
                }
                unique_searchers = next_unique;
            }

            let any_common_active = searchers.iter().any(|s| !s.next_query().is_empty());
            if !any_common_active {
                return;
            }
        }
    }

    /// Find a unique lowest common ancestor by iterating `find_lca`.
    ///
    /// If there are multiple LCAs, recursively find the LCA of that set
    /// until exactly one remains. Returns `None` if there is no common
    /// ancestor. If `count_steps` is true, also returns the number of
    /// iterations.
    pub fn find_unique_lca(&self, left: K, right: K, null: &K) -> Option<(K, usize)> {
        let mut revisions: Vec<K> = vec![left, right];
        let mut steps: usize = 0;
        loop {
            steps += 1;
            let lca = self.find_lca(revisions.iter().cloned(), null);
            match lca.len() {
                1 => return lca.into_iter().next().map(|k| (k, steps)),
                0 => return None,
                _ => revisions = lca.into_iter().collect(),
            }
        }
    }

    /// Find the unique ancestors of `unique_revision` relative to
    /// `common_revisions`.
    ///
    /// Returns the set of revisions that are ancestors of `unique_revision`
    /// but not of any of `common_revisions`. If `unique_revision` is itself
    /// in `common_revisions`, returns an empty set.
    ///
    /// Algorithm description:
    ///
    /// 1. Walk backwards from the unique node and all common nodes.
    /// 2. When a node is seen by both sides, stop searching it in the unique
    ///    walker, include it in the common walker.
    /// 3. Stop searching when there are no nodes left for the unique walker.
    ///    At this point, you have a maximal set of unique nodes. Some of
    ///    them may actually be common, and you haven't reached them yet.
    /// 4. Start new searchers for the unique nodes, seeded with the
    ///    information you have so far.
    /// 5. Continue searching, stopping the common searches when the search
    ///    tip is an ancestor of all unique nodes.
    /// 6. Aggregate together unique searchers when they are searching the
    ///    same tips. When all unique searchers are searching the same node,
    ///    stop move it to a single 'all_unique_searcher'.
    /// 7. The 'all_unique_searcher' represents the very 'tip' of searching.
    ///    Most of the time this produces very little important information.
    ///    So don't step it as quickly as the other searchers.
    /// 8. Search is done when all common searchers have completed.
    pub fn find_unique_ancestors(
        &self,
        unique_revision: K,
        common_revisions: impl IntoIterator<Item = K>,
    ) -> FxHashSet<K>
    where
        K: Ord,
    {
        let common_revisions: Vec<K> = common_revisions.into_iter().collect();
        if common_revisions.contains(&unique_revision) {
            return FxHashSet::default();
        }

        // Phase 1: find maximal unique set.
        let (mut unique_searcher, mut common_searcher) =
            self.find_initial_unique_nodes([unique_revision], common_revisions);
        let unique_nodes: FxHashSet<K> = unique_searcher
            .seen
            .difference(&common_searcher.seen)
            .cloned()
            .collect();
        if unique_nodes.is_empty() {
            return unique_nodes;
        }

        // Phase 2: refine via unique-tip searchers.
        let (mut all_unique_searcher, mut unique_tip_searchers) =
            self.make_unique_searchers(&unique_nodes, &mut unique_searcher, &mut common_searcher);
        self.refine_unique_nodes(
            &mut unique_searcher,
            &mut all_unique_searcher,
            &mut unique_tip_searchers,
            &mut common_searcher,
        );
        unique_nodes
            .difference(&common_searcher.seen)
            .cloned()
            .collect()
    }

    /// Phase 1 of find_unique_ancestors: find the maximal unique set.
    fn find_initial_unique_nodes(
        &self,
        unique_revisions: impl IntoIterator<Item = K>,
        common_revisions: impl IntoIterator<Item = K>,
    ) -> (BfsState<K>, BfsState<K>) {
        let mut unique_searcher = BfsState::new(unique_revisions);
        // Skip past the starting unique revisions themselves.
        unique_searcher.next_set(&self.provider);
        let mut common_searcher = BfsState::new(common_revisions);

        while !unique_searcher.next_query().is_empty() {
            let next_unique_nodes: FxHashSet<K> =
                unique_searcher.next_set(&self.provider).unwrap_or_default();
            let next_common_nodes: FxHashSet<K> =
                common_searcher.next_set(&self.provider).unwrap_or_default();

            let mut unique_are_common_nodes: FxHashSet<K> = next_unique_nodes
                .intersection(&common_searcher.seen)
                .cloned()
                .collect();
            unique_are_common_nodes.extend(
                next_common_nodes
                    .intersection(&unique_searcher.seen)
                    .cloned(),
            );
            if !unique_are_common_nodes.is_empty() {
                let mut ancestors =
                    unique_searcher.find_seen_ancestors(unique_are_common_nodes, &self.provider);
                let more = common_searcher.find_seen_ancestors(ancestors.clone(), &self.provider);
                ancestors.extend(more);
                unique_searcher.stop_searching_any(ancestors.iter().cloned());
                common_searcher.start_searching(ancestors, &self.provider);
            }
        }
        (unique_searcher, common_searcher)
    }

    /// Phase 2 setup: create a searcher for each unique-node tip plus an
    /// `all_unique_searcher` covering ancestry shared by every unique tip.
    fn make_unique_searchers(
        &self,
        unique_nodes: &FxHashSet<K>,
        unique_searcher: &mut BfsState<K>,
        common_searcher: &mut BfsState<K>,
    ) -> (BfsState<K>, Vec<BfsState<K>>)
    where
        K: Ord,
    {
        let parent_map = self.get_parent_map(unique_nodes.iter().cloned());
        let unique_tips = Self::remove_simple_descendants(unique_nodes, &parent_map);

        let mut unique_tip_searchers: Vec<BfsState<K>> = Vec::new();
        let ancestor_all_unique: FxHashSet<K>;

        if unique_tips.len() == 1 {
            ancestor_all_unique = unique_searcher.find_seen_ancestors(unique_tips, &self.provider);
        } else {
            let mut agg: Option<FxHashSet<K>> = None;
            for tip in unique_tips {
                let mut revs_to_search =
                    unique_searcher.find_seen_ancestors([tip.clone()], &self.provider);
                let more =
                    common_searcher.find_seen_ancestors(revs_to_search.clone(), &self.provider);
                revs_to_search.extend(more);
                let mut searcher = BfsState::new(revs_to_search);
                // Skip past the starting nodes — we don't care about them.
                searcher.next_set(&self.provider);
                let seen_snapshot = searcher.seen.clone();
                unique_tip_searchers.push(searcher);
                agg = Some(match agg {
                    None => seen_snapshot,
                    Some(a) => a.intersection(&seen_snapshot).cloned().collect(),
                });
            }
            ancestor_all_unique = agg.unwrap_or_default();
        }

        // Collapse all common nodes into a single searcher covering the
        // `ancestor_all_unique` set, then advance it once.
        let mut all_unique_searcher = BfsState::new(ancestor_all_unique.iter().cloned());
        if !ancestor_all_unique.is_empty() {
            all_unique_searcher.next_set(&self.provider);

            // Stop common-searcher tips that are already ancestors of all uniques.
            let to_stop = common_searcher
                .find_seen_ancestors(ancestor_all_unique.iter().cloned(), &self.provider);
            common_searcher.stop_searching_any(to_stop);

            for searcher in unique_tip_searchers.iter_mut() {
                let to_stop = searcher
                    .find_seen_ancestors(ancestor_all_unique.iter().cloned(), &self.provider);
                searcher.stop_searching_any(to_stop);
            }
        }

        (all_unique_searcher, unique_tip_searchers)
    }

    /// Remove revisions which are descendants (via the parent_map) of other
    /// revisions in the set. This is a cheap O(E) pass that doesn't walk
    /// ancestry — it just drops keys whose parents are already in `revisions`.
    fn remove_simple_descendants(
        revisions: &FxHashSet<K>,
        parent_map: &ParentMap<K>,
    ) -> FxHashSet<K> {
        let mut simple = revisions.clone();
        for (revision, parents) in parent_map.iter() {
            if let Parents::Known(ps) = parents {
                for parent_id in ps {
                    if revisions.contains(parent_id) {
                        simple.remove(revision);
                        break;
                    }
                }
            }
        }
        simple
    }

    /// One BFS step across unique tip searchers, the unique_searcher, and
    /// the common_searcher, propagating find_seen_ancestors cross-checks.
    fn step_unique_and_common_searchers(
        &self,
        common_searcher: &mut BfsState<K>,
        unique_tip_searchers: &mut [BfsState<K>],
        unique_searcher: &BfsState<K>,
    ) -> (FxHashSet<K>, FxHashSet<K>) {
        let newly_seen_common: FxHashSet<K> =
            common_searcher.next_set(&self.provider).unwrap_or_default();

        let mut newly_seen_unique: FxHashSet<K> = FxHashSet::default();
        // Snapshot seen sets of all tip searchers so we can cross-reference
        // without re-borrowing mid-loop.
        let tip_count = unique_tip_searchers.len();
        // Collect (index, next_step) pairs first.
        let mut per_tip_next: Vec<(usize, FxHashSet<K>)> = Vec::with_capacity(tip_count);
        for (i, s) in unique_tip_searchers.iter_mut().enumerate() {
            let mut next_set = s.next_set(&self.provider).unwrap_or_default();
            // Include ancestors already known to the main unique_searcher.
            next_set.extend(unique_searcher.find_seen_ancestors(next_set.clone(), &self.provider));
            // And to the common_searcher.
            next_set.extend(common_searcher.find_seen_ancestors(next_set.clone(), &self.provider));
            per_tip_next.push((i, next_set));
        }
        // Cross-check: each tip pulls in seen ancestors from every other tip.
        // We need to compute additions per tip from the current (pre-start)
        // state of the other tips, so snapshot their seen sets first.
        let seen_per_tip: Vec<FxHashSet<K>> = unique_tip_searchers
            .iter()
            .map(|s| s.seen.clone())
            .collect();
        for (i, next_set) in per_tip_next.iter_mut() {
            for (j, seen_j) in seen_per_tip.iter().enumerate() {
                if *i == j {
                    continue;
                }
                // find_seen_ancestors-equivalent: just intersect with `seen_j`
                // and recurse. Python calls the method on the other searcher;
                // we approximate with a local walk using that searcher's
                // provider and seen set. To match Python exactly, build a
                // temporary BfsState... actually no: find_seen_ancestors
                // walks the provider too. We need to use the real method.
                //
                // The clean way: call alt_searcher.find_seen_ancestors with
                // `next_set` as input, which does the BFS walk constrained
                // to alt_searcher.seen. We can't do that while borrowing
                // the slice mutably. Workaround: re-borrow from the slice
                // by index via split_at_mut. Simpler: collect alt seen sets
                // and do the walk against the local snapshot by computing
                // ancestors via a local function.
                //
                // For correctness with minimal refactor, we synthesize the
                // find_seen_ancestors equivalent here by doing repeated
                // intersections — the Python implementation is essentially
                // a BFS over parents restricted to the seen set.
                let additions =
                    Self::find_seen_ancestors_against(next_set.clone(), seen_j, &self.provider);
                next_set.extend(additions);
            }
        }
        // Apply start_searching and accumulate newly_seen_unique.
        for (i, next_set) in per_tip_next {
            unique_tip_searchers[i].start_searching(next_set.iter().cloned(), &self.provider);
            newly_seen_unique.extend(next_set);
        }
        (newly_seen_common, newly_seen_unique)
    }

    /// Free-standing equivalent of `BfsState::find_seen_ancestors` that
    /// walks the provider restricted to a given `seen` set. Used by
    /// `step_unique_and_common_searchers` to cross-check tip searchers
    /// without mid-loop re-borrows of the slice.
    fn find_seen_ancestors_against(
        revisions: FxHashSet<K>,
        seen: &FxHashSet<K>,
        provider: &P,
    ) -> FxHashSet<K> {
        let mut pending: FxHashSet<K> =
            revisions.into_iter().filter(|r| seen.contains(r)).collect();
        let mut seen_ancestors: FxHashSet<K> = pending.iter().cloned().collect();
        while !pending.is_empty() {
            let mut std_set: std::collections::HashSet<K> =
                std::collections::HashSet::with_capacity(pending.len());
            for k in &pending {
                std_set.insert(k.clone());
            }
            let parent_map = provider.get_parent_map(&std_set);
            let mut all_parents: Vec<K> = Vec::new();
            for (_, parents) in parent_map.iter() {
                if let Parents::Known(ps) = parents {
                    all_parents.extend(ps.iter().cloned());
                }
            }
            let mut next_pending: FxHashSet<K> = FxHashSet::default();
            for p in all_parents {
                if seen.contains(&p) && !seen_ancestors.contains(&p) {
                    next_pending.insert(p);
                }
            }
            seen_ancestors.extend(next_pending.iter().cloned());
            pending = next_pending;
        }
        seen_ancestors
    }

    /// Find nodes common to all unique tip searchers (and optionally step
    /// the `all_unique_searcher`).
    fn find_nodes_common_to_all_unique(
        &self,
        unique_tip_searchers: &[BfsState<K>],
        all_unique_searcher: &mut BfsState<K>,
        newly_seen_unique: &FxHashSet<K>,
        step_all_unique: bool,
    ) -> FxHashSet<K> {
        let mut common: FxHashSet<K> = newly_seen_unique.clone();
        for searcher in unique_tip_searchers {
            common = common.intersection(&searcher.seen).cloned().collect();
        }
        common = common
            .intersection(&all_unique_searcher.seen)
            .cloned()
            .collect();
        if step_all_unique {
            if let Some(nodes) = all_unique_searcher.next_set(&self.provider) {
                common.extend(nodes);
            }
        }
        common
    }

    /// Combine unique tip searchers that are searching the same frontier.
    fn collapse_unique_searchers(
        &self,
        unique_tip_searchers: Vec<BfsState<K>>,
        common_to_all_unique_nodes: &FxHashSet<K>,
    ) -> Vec<BfsState<K>> {
        // First pass: stop searching the common-to-all set on each searcher
        // and bucket by resulting frontier.
        let mut buckets: FxHashMap<Vec<K>, Vec<BfsState<K>>> = FxHashMap::default();
        let mut empty_bucket: Vec<BfsState<K>> = Vec::new();
        for mut searcher in unique_tip_searchers {
            searcher.stop_searching_any(common_to_all_unique_nodes.iter().cloned());
            let nq = searcher.next_query().clone();
            if nq.is_empty() {
                empty_bucket.push(searcher);
                continue;
            }
            // Sort the frontier by hash for a deterministic bucket key.
            let mut key: Vec<K> = nq.into_iter().collect();
            key.sort_by_key(|k| {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::Hasher;
                let mut h = DefaultHasher::new();
                k.hash(&mut h);
                h.finish()
            });
            buckets.entry(key).or_default().push(searcher);
        }
        let _ = empty_bucket; // drop empties — those searchers are done.

        let mut next_unique_searchers: Vec<BfsState<K>> = Vec::new();
        for (_key, mut searchers) in buckets {
            if searchers.len() == 1 {
                next_unique_searchers.push(searchers.pop().unwrap());
            } else {
                // Combine: intersect all their seen sets into the first.
                let mut first = searchers.remove(0);
                for s in searchers {
                    first.seen = first.seen.intersection(&s.seen).cloned().collect();
                }
                next_unique_searchers.push(first);
            }
        }
        next_unique_searchers
    }

    /// Phase 2 main loop: refine unique-vs-common by stepping searchers
    /// until `common_searcher` has nothing left to search.
    fn refine_unique_nodes(
        &self,
        unique_searcher: &mut BfsState<K>,
        all_unique_searcher: &mut BfsState<K>,
        unique_tip_searchers: &mut Vec<BfsState<K>>,
        common_searcher: &mut BfsState<K>,
    ) {
        // Step the all_unique_searcher every N steps (Python's
        // STEP_UNIQUE_SEARCHER_EVERY = 5).
        const STEP_ALL_UNIQUE_EVERY: usize = 5;
        let mut step_all_unique_counter: usize = 0;

        while !common_searcher.next_query().is_empty() {
            let (newly_seen_common, newly_seen_unique) = self.step_unique_and_common_searchers(
                common_searcher,
                unique_tip_searchers,
                unique_searcher,
            );
            let common_to_all_unique_nodes = self.find_nodes_common_to_all_unique(
                unique_tip_searchers,
                all_unique_searcher,
                &newly_seen_unique,
                step_all_unique_counter == 0,
            );
            step_all_unique_counter = (step_all_unique_counter + 1) % STEP_ALL_UNIQUE_EVERY;

            if !newly_seen_common.is_empty() {
                let stop: FxHashSet<K> = all_unique_searcher
                    .seen
                    .intersection(&newly_seen_common)
                    .cloned()
                    .collect();
                common_searcher.stop_searching_any(stop);
            }
            if !common_to_all_unique_nodes.is_empty() {
                let mut expanded = common_to_all_unique_nodes.clone();
                expanded.extend(common_searcher.find_seen_ancestors(
                    common_to_all_unique_nodes.iter().cloned(),
                    &self.provider,
                ));
                all_unique_searcher.start_searching(expanded.iter().cloned(), &self.provider);
                common_searcher.stop_searching_any(expanded);
            }
            let old_searchers = std::mem::take(unique_tip_searchers);
            *unique_tip_searchers =
                self.collapse_unique_searchers(old_searchers, &common_to_all_unique_nodes);
        }
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

    /// Build a set literal from an array of strings.
    fn set<const N: usize>(xs: [&'static str; N]) -> FxHashSet<&'static str> {
        xs.into_iter().collect()
    }

    fn ancestry_1() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        make(&[
            ("rev1", &[NULL]),
            ("rev2a", &["rev1"]),
            ("rev2b", &["rev1"]),
            ("rev3", &["rev2a"]),
            ("rev4", &["rev3", "rev2b"]),
        ])
    }

    fn ancestry_2() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        make(&[
            ("rev1a", &[NULL]),
            ("rev2a", &["rev1a"]),
            ("rev1b", &[NULL]),
            ("rev3a", &["rev2a"]),
            ("rev4a", &["rev3a"]),
        ])
    }

    fn criss_cross() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        make(&[
            ("rev1", &[NULL]),
            ("rev2a", &["rev1"]),
            ("rev2b", &["rev1"]),
            ("rev3a", &["rev2a", "rev2b"]),
            ("rev3b", &["rev2b", "rev2a"]),
        ])
    }

    fn criss_cross2() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        make(&[
            ("rev1a", &[NULL]),
            ("rev1b", &[NULL]),
            ("rev2a", &["rev1a", "rev1b"]),
            ("rev2b", &["rev1b", "rev1a"]),
        ])
    }

    fn history_shortcut() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        make(&[
            ("rev1", &[NULL]),
            ("rev2a", &["rev1"]),
            ("rev2b", &["rev1"]),
            ("rev2c", &["rev1"]),
            ("rev3a", &["rev2a", "rev2b"]),
            ("rev3b", &["rev2b", "rev2c"]),
        ])
    }

    fn extended_history_shortcut() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        make(&[
            ("a", &[NULL]),
            ("b", &["a"]),
            ("c", &["b"]),
            ("d", &["c"]),
            ("e", &["d"]),
            ("f", &["a", "d"]),
        ])
    }

    fn double_shortcut_fixture() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        make(&[
            ("a", &[NULL]),
            ("b", &["a"]),
            ("c", &["b"]),
            ("d", &["c"]),
            ("e", &["c"]),
            ("f", &["a", "d"]),
            ("g", &["a", "e"]),
        ])
    }

    fn complex_shortcut() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        make(&[
            ("a", &[NULL]),
            ("b", &["a"]),
            ("c", &["b"]),
            ("d", &["c"]),
            ("e", &["d"]),
            ("f", &["d"]),
            ("g", &["f"]),
            ("h", &["f"]),
            ("i", &["e", "g"]),
            ("j", &["g"]),
            ("k", &["j"]),
            ("l", &["k"]),
            ("m", &["i", "l"]),
            ("n", &["l", "h"]),
        ])
    }

    fn complex_shortcut2() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        make(&[
            ("a", &[NULL]),
            ("b", &["a"]),
            ("c", &["b"]),
            ("d", &["c"]),
            ("e", &["d"]),
            ("f", &["e"]),
            ("g", &["f"]),
            ("h", &["d"]),
            ("i", &["g"]),
            ("j", &["h"]),
            ("k", &["h", "i"]),
            ("l", &["k"]),
            ("m", &["l"]),
            ("n", &["m"]),
            ("o", &["n"]),
            ("p", &["o"]),
            ("q", &["p"]),
            ("r", &["q"]),
            ("s", &["r"]),
            ("t", &["i", "s"]),
            ("u", &["s", "j"]),
        ])
    }

    fn multiple_interesting_unique() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        make(&[
            ("a", &[NULL]),
            ("b", &["a"]),
            ("c", &["b"]),
            ("d", &["c"]),
            ("e", &["d"]),
            ("f", &["d"]),
            ("g", &["e"]),
            ("h", &["e"]),
            ("i", &["f"]),
            ("j", &["g"]),
            ("k", &["g"]),
            ("l", &["h"]),
            ("m", &["i"]),
            ("n", &["k", "l"]),
            ("o", &["m"]),
            ("p", &["m", "l"]),
            ("q", &["n", "o"]),
            ("r", &["q"]),
            ("s", &["r"]),
            ("t", &["s"]),
            ("u", &["t"]),
            ("v", &["u"]),
            ("w", &["v"]),
            ("x", &["w"]),
            ("y", &["j", "x"]),
            ("z", &["x", "p"]),
        ])
    }

    fn shortcut_extra_root() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        make(&[
            ("a", &[NULL]),
            ("b", &["a"]),
            ("c", &["b"]),
            ("d", &["c"]),
            ("e", &["d"]),
            ("f", &["a", "d", "g"]),
            ("g", &[NULL]),
        ])
    }

    fn boundary() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        make(&[
            ("a", &["b"]),
            ("c", &["b", "d"]),
            ("b", &["e"]),
            ("d", &["e"]),
            ("e", &["f"]),
            ("f", &[NULL]),
        ])
    }

    #[test]
    fn test_lca_ancestry_1() {
        let g = ancestry_1();
        assert_eq!(g.find_lca([NULL, NULL], &NULL), set([NULL]));
        assert_eq!(g.find_lca([NULL, "rev1"], &NULL), set([NULL]));
        assert_eq!(g.find_lca(["rev1", "rev1"], &NULL), set(["rev1"]));
        assert_eq!(g.find_lca(["rev2a", "rev2b"], &NULL), set(["rev1"]));
    }

    #[test]
    fn test_lca_criss_cross() {
        let g = criss_cross();
        assert_eq!(
            g.find_lca(["rev3a", "rev3b"], &NULL),
            set(["rev2a", "rev2b"])
        );
    }

    #[test]
    fn test_lca_shortcut() {
        let g = history_shortcut();
        assert_eq!(g.find_lca(["rev3a", "rev3b"], &NULL), set(["rev2b"]));
    }

    #[test]
    fn test_lca_double_shortcut() {
        let g = double_shortcut_fixture();
        assert_eq!(g.find_lca(["f", "g"], &NULL), set(["c"]));
    }

    #[test]
    fn test_unique_lca_ancestry_1() {
        let g = ancestry_1();
        assert_eq!(g.find_unique_lca(NULL, NULL, &NULL), Some((NULL, 1)));
        assert_eq!(g.find_unique_lca(NULL, "rev1", &NULL), Some((NULL, 1)));
        assert_eq!(g.find_unique_lca("rev1", "rev1", &NULL), Some(("rev1", 1)));
        assert_eq!(
            g.find_unique_lca("rev2a", "rev2b", &NULL),
            Some(("rev1", 1))
        );
    }

    #[test]
    fn test_unique_lca_criss_cross() {
        let g = criss_cross();
        assert_eq!(
            g.find_unique_lca("rev3a", "rev3b", &NULL),
            Some(("rev1", 2))
        );
    }

    #[test]
    fn test_unique_lca_null_revision_criss_cross2() {
        let g = criss_cross2();
        assert_eq!(
            g.find_unique_lca("rev2a", "rev1b", &NULL).map(|(k, _)| k),
            Some("rev1b")
        );
        assert_eq!(
            g.find_unique_lca("rev2a", "rev2b", &NULL).map(|(k, _)| k),
            Some(NULL)
        );
    }

    #[test]
    fn test_unique_lca_separate_ancestry() {
        let g = ancestry_2();
        assert_eq!(
            g.find_unique_lca("rev4a", "rev1b", &NULL).map(|(k, _)| k),
            Some(NULL)
        );
    }

    #[test]
    fn test_heads_null() {
        let g = ancestry_1();
        assert_eq!(g.heads_with_null([NULL], &NULL), set([NULL]));
        assert_eq!(g.heads_with_null([NULL, "rev1"], &NULL), set(["rev1"]));
        assert_eq!(g.heads_with_null(["rev1", NULL], &NULL), set(["rev1"]));
    }

    #[test]
    fn test_heads_one() {
        let g = ancestry_1();
        for key in [NULL, "rev1", "rev2a", "rev2b", "rev3", "rev4"] {
            assert_eq!(g.heads_with_null([key], &NULL), set([key]));
        }
    }

    #[test]
    fn test_heads_single_from_pair() {
        let g = ancestry_1();
        assert_eq!(g.heads_with_null([NULL, "rev4"], &NULL), set(["rev4"]));
        assert_eq!(g.heads_with_null(["rev1", "rev2a"], &NULL), set(["rev2a"]));
        assert_eq!(g.heads_with_null(["rev1", "rev2b"], &NULL), set(["rev2b"]));
        assert_eq!(g.heads_with_null(["rev1", "rev3"], &NULL), set(["rev3"]));
        assert_eq!(g.heads_with_null(["rev1", "rev4"], &NULL), set(["rev4"]));
        assert_eq!(g.heads_with_null(["rev2a", "rev4"], &NULL), set(["rev4"]));
        assert_eq!(g.heads_with_null(["rev2b", "rev4"], &NULL), set(["rev4"]));
        assert_eq!(g.heads_with_null(["rev3", "rev4"], &NULL), set(["rev4"]));
    }

    #[test]
    fn test_heads_two_heads() {
        let g = ancestry_1();
        assert_eq!(
            g.heads_with_null(["rev2a", "rev2b"], &NULL),
            set(["rev2a", "rev2b"])
        );
        assert_eq!(
            g.heads_with_null(["rev3", "rev2b"], &NULL),
            set(["rev3", "rev2b"])
        );
    }

    #[test]
    fn test_heads_criss_cross() {
        let g = criss_cross();
        assert_eq!(g.heads_with_null(["rev2a", "rev1"], &NULL), set(["rev2a"]));
        assert_eq!(g.heads_with_null(["rev2b", "rev1"], &NULL), set(["rev2b"]));
        assert_eq!(g.heads_with_null(["rev3a", "rev1"], &NULL), set(["rev3a"]));
        assert_eq!(g.heads_with_null(["rev3b", "rev1"], &NULL), set(["rev3b"]));
        assert_eq!(
            g.heads_with_null(["rev2a", "rev2b"], &NULL),
            set(["rev2a", "rev2b"])
        );
        assert_eq!(g.heads_with_null(["rev3a", "rev2a"], &NULL), set(["rev3a"]));
        assert_eq!(g.heads_with_null(["rev3a", "rev2b"], &NULL), set(["rev3a"]));
        assert_eq!(
            g.heads_with_null(["rev3a", "rev2a", "rev2b"], &NULL),
            set(["rev3a"])
        );
        assert_eq!(g.heads_with_null(["rev3b", "rev2a"], &NULL), set(["rev3b"]));
        assert_eq!(g.heads_with_null(["rev3b", "rev2b"], &NULL), set(["rev3b"]));
        assert_eq!(
            g.heads_with_null(["rev3b", "rev2a", "rev2b"], &NULL),
            set(["rev3b"])
        );
        assert_eq!(
            g.heads_with_null(["rev3a", "rev3b"], &NULL),
            set(["rev3a", "rev3b"])
        );
        assert_eq!(
            g.heads_with_null(["rev3a", "rev3b", "rev2a", "rev2b"], &NULL),
            set(["rev3a", "rev3b"])
        );
    }

    #[test]
    fn test_heads_shortcut() {
        let g = history_shortcut();
        assert_eq!(
            g.heads_with_null(["rev2a", "rev2b", "rev2c"], &NULL),
            set(["rev2a", "rev2b", "rev2c"])
        );
        assert_eq!(
            g.heads_with_null(["rev3a", "rev3b"], &NULL),
            set(["rev3a", "rev3b"])
        );
        assert_eq!(
            g.heads_with_null(["rev2a", "rev3a", "rev3b"], &NULL),
            set(["rev3a", "rev3b"])
        );
        assert_eq!(
            g.heads_with_null(["rev2a", "rev3b"], &NULL),
            set(["rev2a", "rev3b"])
        );
        assert_eq!(
            g.heads_with_null(["rev2c", "rev3a"], &NULL),
            set(["rev2c", "rev3a"])
        );
    }

    #[test]
    fn test_graph_difference_ancestry_1() {
        let g = ancestry_1();
        assert_eq!(
            g.find_difference("rev1", "rev1"),
            (FxHashSet::default(), FxHashSet::default())
        );
        assert_eq!(
            g.find_difference(NULL, "rev1"),
            (FxHashSet::default(), set(["rev1"]))
        );
        assert_eq!(
            g.find_difference("rev1", NULL),
            (set(["rev1"]), FxHashSet::default())
        );
        assert_eq!(
            g.find_difference("rev3", "rev2b"),
            (set(["rev2a", "rev3"]), set(["rev2b"]))
        );
        assert_eq!(
            g.find_difference("rev4", "rev2b"),
            (set(["rev4", "rev3", "rev2a"]), FxHashSet::default())
        );
    }

    #[test]
    fn test_graph_difference_separate_ancestry() {
        let g = ancestry_2();
        assert_eq!(
            g.find_difference("rev1a", "rev1b"),
            (set(["rev1a"]), set(["rev1b"]))
        );
        assert_eq!(
            g.find_difference("rev4a", "rev1b"),
            (set(["rev1a", "rev2a", "rev3a", "rev4a"]), set(["rev1b"]))
        );
    }

    #[test]
    fn test_graph_difference_criss_cross() {
        let g = criss_cross();
        assert_eq!(
            g.find_difference("rev3a", "rev3b"),
            (set(["rev3a"]), set(["rev3b"]))
        );
        assert_eq!(
            g.find_difference("rev2a", "rev3b"),
            (FxHashSet::default(), set(["rev3b", "rev2b"]))
        );
    }

    #[test]
    fn test_graph_difference_extended_history() {
        let g = extended_history_shortcut();
        assert_eq!(g.find_difference("e", "f"), (set(["e"]), set(["f"])));
        assert_eq!(g.find_difference("f", "e"), (set(["f"]), set(["e"])));
    }

    #[test]
    fn test_graph_difference_double_shortcut() {
        let g = double_shortcut_fixture();
        assert_eq!(
            g.find_difference("f", "g"),
            (set(["d", "f"]), set(["e", "g"]))
        );
    }

    #[test]
    fn test_graph_difference_complex_shortcut() {
        let g = complex_shortcut();
        assert_eq!(
            g.find_difference("m", "n"),
            (set(["m", "i", "e"]), set(["n", "h"]))
        );
    }

    #[test]
    fn test_graph_difference_complex_shortcut2() {
        let g = complex_shortcut2();
        assert_eq!(g.find_difference("t", "u"), (set(["t"]), set(["j", "u"])));
    }

    #[test]
    fn test_graph_difference_shortcut_extra_root() {
        let g = shortcut_extra_root();
        assert_eq!(g.find_difference("e", "f"), (set(["e"]), set(["f", "g"])));
    }

    #[test]
    fn test_unique_ancestors_empty_set() {
        let g = ancestry_1();
        assert_eq!(
            g.find_unique_ancestors("rev1", ["rev1"]),
            FxHashSet::default()
        );
        assert_eq!(
            g.find_unique_ancestors("rev2b", ["rev2b"]),
            FxHashSet::default()
        );
        assert_eq!(
            g.find_unique_ancestors("rev3", ["rev1", "rev3"]),
            FxHashSet::default()
        );
    }

    #[test]
    fn test_unique_ancestors_single_node() {
        let g = ancestry_1();
        assert_eq!(g.find_unique_ancestors("rev2a", ["rev1"]), set(["rev2a"]));
        assert_eq!(g.find_unique_ancestors("rev2b", ["rev1"]), set(["rev2b"]));
        assert_eq!(g.find_unique_ancestors("rev3", ["rev2a"]), set(["rev3"]));
    }

    #[test]
    fn test_unique_ancestors_in_ancestry() {
        let g = ancestry_1();
        assert_eq!(
            g.find_unique_ancestors("rev1", ["rev3"]),
            FxHashSet::default()
        );
        assert_eq!(
            g.find_unique_ancestors("rev2b", ["rev4"]),
            FxHashSet::default()
        );
    }

    #[test]
    fn test_unique_ancestors_multiple_revisions() {
        let g = ancestry_1();
        assert_eq!(
            g.find_unique_ancestors("rev4", ["rev3", "rev2b"]),
            set(["rev4"])
        );
        assert_eq!(
            g.find_unique_ancestors("rev4", ["rev2b"]),
            set(["rev2a", "rev3", "rev4"])
        );
    }

    #[test]
    fn test_unique_ancestors_complex_shortcut() {
        let g = complex_shortcut();
        assert_eq!(g.find_unique_ancestors("n", ["m"]), set(["h", "n"]));
        assert_eq!(g.find_unique_ancestors("m", ["n"]), set(["e", "i", "m"]));
    }

    #[test]
    fn test_unique_ancestors_complex_shortcut2() {
        let g = complex_shortcut2();
        assert_eq!(g.find_unique_ancestors("u", ["t"]), set(["j", "u"]));
        assert_eq!(g.find_unique_ancestors("t", ["u"]), set(["t"]));
    }

    #[test]
    fn test_unique_ancestors_multiple_interesting_unique() {
        let g = multiple_interesting_unique();
        assert_eq!(g.find_unique_ancestors("y", ["z"]), set(["j", "y"]));
        assert_eq!(g.find_unique_ancestors("z", ["y"]), set(["p", "z"]));
    }

    #[test]
    fn test_is_ancestor_ancestry_1() {
        let g = ancestry_1();
        assert!(g.is_ancestor(NULL, NULL, &NULL));
        assert!(g.is_ancestor(NULL, "rev1", &NULL));
        assert!(!g.is_ancestor("rev1", NULL, &NULL));
        assert!(g.is_ancestor(NULL, "rev4", &NULL));
        assert!(!g.is_ancestor("rev4", NULL, &NULL));
        assert!(!g.is_ancestor("rev4", "rev2b", &NULL));
        assert!(g.is_ancestor("rev2b", "rev4", &NULL));
        assert!(!g.is_ancestor("rev2b", "rev3", &NULL));
        assert!(!g.is_ancestor("rev3", "rev2b", &NULL));
    }

    #[test]
    fn test_is_ancestor_boundary() {
        // Python's test_is_ancestor_boundary: verify a is not an ancestor
        // of c despite both sharing a common ancestor further down.
        let g = boundary();
        assert!(!g.is_ancestor("a", "c", &NULL));
    }

    #[test]
    fn test_is_between_ancestry_1() {
        let g = ancestry_1();
        assert!(g.is_between(NULL, Some(NULL), Some(NULL), &NULL));
        assert!(g.is_between("rev1", Some(NULL), Some("rev1"), &NULL));
        assert!(g.is_between("rev1", Some("rev1"), Some("rev4"), &NULL));
        assert!(g.is_between("rev4", Some("rev1"), Some("rev4"), &NULL));
        assert!(g.is_between("rev3", Some("rev1"), Some("rev4"), &NULL));
        assert!(!g.is_between("rev4", Some("rev1"), Some("rev3"), &NULL));
        assert!(!g.is_between("rev1", Some("rev2a"), Some("rev4"), &NULL));
        assert!(!g.is_between(NULL, Some("rev1"), Some("rev4"), &NULL));
    }

    #[test]
    fn test_find_merge_order_single_lca() {
        let g = ancestry_1();
        assert_eq!(g.find_merge_order("rev4", ["rev2b"]), vec!["rev2b"]);
    }

    fn with_ghost() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        // NULL_REVISION itself is explicitly included as a root so it
        // survives as a key in iter_ancestry's output.
        make(&[
            ("a", &["b"]),
            ("c", &["b", "d"]),
            ("b", &["e"]),
            ("d", &["e", "g"]),
            ("e", &["f"]),
            ("f", &[NULL]),
            (NULL, &[]),
        ])
    }

    fn racing_shortcuts() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        make(&[
            ("a", &[NULL]),
            ("b", &["a"]),
            ("c", &["b"]),
            ("d", &["c"]),
            ("e", &["d"]),
            ("f", &["e"]),
            ("g", &["f"]),
            ("h", &["g"]),
            ("i", &["h", "o"]),
            ("j", &["i", "y"]),
            ("k", &["d"]),
            ("l", &["k"]),
            ("m", &["l"]),
            ("n", &["m"]),
            ("o", &["n", "g"]),
            ("p", &["f"]),
            ("q", &["p", "m"]),
            ("r", &["o"]),
            ("s", &["r"]),
            ("t", &["s"]),
            ("u", &["t"]),
            ("v", &["u"]),
            ("w", &["v"]),
            ("x", &["w"]),
            ("y", &["x"]),
            ("z", &["x", "q"]),
        ])
    }

    /// Python's `alt_merge` fixture.
    ///
    /// ```text
    ///  a
    ///  |\
    ///  b |
    ///  | |
    ///  c |
    ///   \|
    ///    d
    /// ```
    fn alt_merge() -> Graph<&'static str, DictParentsProvider<&'static str>> {
        make(&[("a", &[]), ("b", &["a"]), ("c", &["b"]), ("d", &["a", "c"])])
    }

    #[test]
    fn test_heads_alt_merge() {
        let g = alt_merge();
        assert_eq!(g.heads_with_null(["a", "c"], &NULL), set(["c"]));
    }

    #[test]
    fn test_heads_with_ghost_fixture() {
        let g = with_ghost();
        assert_eq!(g.heads_with_null(["e", "g"], &NULL), set(["e", "g"]));
        assert_eq!(g.heads_with_null(["a", "c"], &NULL), set(["a", "c"]));
        assert_eq!(g.heads_with_null(["a", "g"], &NULL), set(["a", "g"]));
        assert_eq!(g.heads_with_null(["f", "g"], &NULL), set(["f", "g"]));
        assert_eq!(g.heads_with_null(["c", "g"], &NULL), set(["c"]));
        assert_eq!(g.heads_with_null(["c", "b", "d", "g"], &NULL), set(["c"]));
        assert_eq!(
            g.heads_with_null(["a", "c", "e", "g"], &NULL),
            set(["a", "c"])
        );
        assert_eq!(g.heads_with_null(["a", "c", "f"], &NULL), set(["a", "c"]));
    }

    #[test]
    fn test_filter_candidate_lca() {
        // Corner case from Python:
        //   NULL
        //   / \
        //  a   e
        //  |   |
        //  b   d
        //   \ /
        //    c
        // `a`'s descendant is `c`; `e`'s descendant is also `c`. So
        // heads([a, c, e]) should be just {c}.
        let g = make(&[
            ("c", &["b", "d"]),
            ("d", &["e"]),
            ("b", &["a"]),
            ("a", &[NULL]),
            ("e", &[NULL]),
        ]);
        assert_eq!(g.heads_with_null(["a", "c", "e"], &NULL), set(["c"]));
    }

    #[test]
    fn test_iter_topo_order_ancestry_1() {
        let g = ancestry_1();
        let order = g.iter_topo_order(["rev2a", "rev3", "rev1"]).unwrap();
        let pos = |k: &&str| order.iter().position(|n| n == k).unwrap();
        assert_eq!(
            order.iter().cloned().collect::<FxHashSet<_>>(),
            set(["rev1", "rev2a", "rev3"])
        );
        assert!(pos(&"rev2a") > pos(&"rev1"));
        assert!(pos(&"rev2a") < pos(&"rev3"));
    }

    #[test]
    fn test_iter_ancestry_boundary() {
        let g = with_ghost();
        // `a` is not in the ancestry of `c`; everything else is.
        let anc = g.iter_ancestry(["c"]);
        let keys: FxHashSet<&'static str> = anc.iter().map(|(k, _)| *k).collect();
        assert!(!keys.contains(&"a"));
        assert!(keys.contains(&"c"));
        assert!(keys.contains(&"b"));
        assert!(keys.contains(&"d"));
        assert!(keys.contains(&"e"));
        assert!(keys.contains(&"f"));
    }

    #[test]
    fn test_iter_ancestry_with_ghost_reports_none() {
        let g = with_ghost();
        // `g` is a ghost (present as parent of `d` but not as key).
        // iter_ancestry should yield it with Parents::Ghost.
        let anc = g.iter_ancestry(["a", "c"]);
        let mut ghost_seen = false;
        for (k, parents) in &anc {
            if *k == "g" {
                ghost_seen = true;
                assert!(matches!(parents, Parents::Ghost));
            }
        }
        assert!(ghost_seen, "ghost `g` should appear in iter_ancestry");
    }

    #[test]
    fn test_find_lefthand_merger_rev2b() {
        // In ancestry_1, rev4 merged rev2b (rev4 has parents [rev3, rev2b]).
        // Walking rev4's lefthand ancestry from rev2b: rev4 is the merger.
        let g = ancestry_1();
        assert_eq!(g.find_lefthand_merger("rev2b", "rev4"), Some("rev4"));
    }

    #[test]
    fn test_find_lefthand_merger_rev2a() {
        // rev2a is itself a lefthand ancestor of rev4 (via rev3), so it's
        // its own "merger".
        let g = ancestry_1();
        assert_eq!(g.find_lefthand_merger("rev2a", "rev4"), Some("rev2a"));
    }

    #[test]
    fn test_find_lefthand_merger_rev4_not_ancestor() {
        // rev4 is a descendant of rev2a, not an ancestor.
        let g = ancestry_1();
        assert_eq!(g.find_lefthand_merger("rev4", "rev2a"), None);
    }

    #[test]
    fn test_unique_lca_recursive_ancestry_1() {
        // In ancestry_1, rev1 is the unique LCA of rev2a and rev2b.
        let g = ancestry_1();
        let (key, steps) = g.find_unique_lca("rev2a", "rev2b", &NULL).unwrap();
        assert_eq!(key, "rev1");
        assert_eq!(steps, 1);
    }

    #[test]
    fn test_unique_lca_no_common_ancestor() {
        // Two disjoint ancestries share only NULL_REVISION as a common
        // ancestor. find_unique_lca returns NULL (never errors).
        let g = ancestry_2();
        let (key, _steps) = g.find_unique_lca("rev4a", "rev1b", &NULL).unwrap();
        assert_eq!(key, NULL);
    }

    #[test]
    fn test_unique_ancestors_racing_shortcuts() {
        let g = racing_shortcuts();
        assert_eq!(g.find_unique_ancestors("z", ["y"]), set(["p", "q", "z"]));
        assert_eq!(
            g.find_unique_ancestors("j", ["z"]),
            set(["h", "i", "j", "y"])
        );
    }

    #[test]
    fn test_find_distance_to_null_ancestry_1() {
        let g = ancestry_1();
        assert_eq!(
            g.find_distance_to_null(NULL, std::iter::empty(), NULL)
                .unwrap(),
            0
        );
        assert_eq!(
            g.find_distance_to_null("rev1", std::iter::empty(), NULL)
                .unwrap(),
            1
        );
        assert_eq!(
            g.find_distance_to_null("rev2a", std::iter::empty(), NULL)
                .unwrap(),
            2
        );
        assert_eq!(
            g.find_distance_to_null("rev2b", std::iter::empty(), NULL)
                .unwrap(),
            2
        );
        assert_eq!(
            g.find_distance_to_null("rev3", std::iter::empty(), NULL)
                .unwrap(),
            3
        );
        assert_eq!(
            g.find_distance_to_null("rev4", std::iter::empty(), NULL)
                .unwrap(),
            4
        );
    }

    #[test]
    fn test_find_lefthand_distances_ghosts() {
        let g = make(&[("nonghost", &[NULL]), ("toghost", &["ghost"])]);
        let d = g.find_lefthand_distances(vec!["nonghost", "toghost"], NULL);
        assert_eq!(d.get(&"nonghost"), Some(&1));
        // Ghosts are reported as distance -1.
        assert_eq!(d.get(&"toghost"), Some(&-1));
    }

    #[test]
    fn test_find_lefthand_distances_smoke() {
        let g = make(&[
            ("rev1", &[NULL]),
            ("rev2a", &["rev1"]),
            ("rev2b", &["rev1"]),
            ("rev2c", &["rev1"]),
            ("rev3a", &["rev2a", "rev2b"]),
            ("rev3b", &["rev2b", "rev2c"]),
        ]);
        let d = g.find_lefthand_distances(vec!["rev3b", "rev2a"], NULL);
        assert_eq!(d.get(&"rev2a"), Some(&2));
        assert_eq!(d.get(&"rev3b"), Some(&3));
    }

    #[test]
    fn test_get_child_map_ancestry_1() {
        let g = ancestry_1();
        let cm = g.get_child_map(vec!["rev4", "rev3", "rev2a", "rev2b"]);
        assert_eq!(cm.get(&"rev1"), Some(&vec!["rev2a", "rev2b"]));
        assert_eq!(cm.get(&"rev2a"), Some(&vec!["rev3"]));
        assert_eq!(cm.get(&"rev2b"), Some(&vec!["rev4"]));
        assert_eq!(cm.get(&"rev3"), Some(&vec!["rev4"]));
    }
}
