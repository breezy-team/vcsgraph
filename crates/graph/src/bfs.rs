//! Breadth-first ancestry search.
//!
//! Ported from the `_BreadthFirstSearcher` class in `vcsgraph/graph.py`.
//! The searcher walks the ancestry of a set of revisions, optionally with
//! ghosts split out, and supports mid-walk modifications via
//! [`BfsState::start_searching`] and [`BfsState::stop_searching_any`].
//!
//! The state is decoupled from the parents provider: each advance method
//! takes a `&impl ParentsProvider<K>` explicitly. This lets Python bindings
//! keep the provider adapter and the state as sibling fields in the same
//! pyclass without running into self-reference problems.

use crate::{ParentMap, Parents, ParentsProvider};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashSet;
use std::hash::Hash;

/// Which kind of result the searcher returned on its most recent call.
///
/// Callers can interleave `next` and `next_with_ghosts` calls; the searcher
/// transparently advances the underlying state when the mode flips.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReturnMode {
    /// Most recent return was a plain `next()` — revisions yielded before
    /// their parents were queried, so ghosts are mixed in with real nodes.
    Next,
    /// Most recent return was `next_with_ghosts()` — revisions yielded after
    /// their parents were queried, so ghosts are split out.
    NextWithGhosts,
}

/// Outcome of a `_do_query` step.
struct QueryResult<K> {
    /// Nodes present in the provider's response.
    found: FxHashSet<K>,
    /// Nodes not found (ghosts).
    ghosts: FxHashSet<K>,
    /// Parents of the found nodes that we haven't seen before.
    next: FxHashSet<K>,
    /// The full parent map returned by the provider for the queried keys.
    parents: FxHashMap<K, Vec<K>>,
}

/// Mutable state of a breadth-first ancestry search.
///
/// Constructed via [`BfsState::new`]; advanced with [`next`](Self::next) or
/// [`next_with_ghosts`](Self::next_with_ghosts), which both take a reference
/// to a parents provider. Mid-walk mutations go through
/// [`start_searching`](Self::start_searching) and
/// [`stop_searching_any`](Self::stop_searching_any).
pub struct BfsState<K: Hash + Eq + Clone> {
    /// All revisions the searcher has ever visited (seen or about to visit).
    pub seen: FxHashSet<K>,
    /// Revisions the caller originally asked to search from, plus any added
    /// via `start_searching`.
    pub started_keys: FxHashSet<K>,
    /// Revisions the caller explicitly asked to not descend through, plus
    /// any ghosts encountered. Ghosts are implicit stop points so the search
    /// can be repeated after ghosts are filled in.
    pub stopped_keys: FxHashSet<K>,

    next_query: FxHashSet<K>,
    current_present: FxHashSet<K>,
    current_ghosts: FxHashSet<K>,
    current_parents: FxHashMap<K, Vec<K>>,
    returning: ReturnMode,
    iterations: usize,
}

impl<K: Hash + Eq + Clone> BfsState<K> {
    /// Start a new search from `revisions`.
    pub fn new<I: IntoIterator<Item = K>>(revisions: I) -> Self {
        let next_query: FxHashSet<K> = revisions.into_iter().collect();
        let started_keys: FxHashSet<K> = next_query.iter().cloned().collect();
        BfsState {
            seen: FxHashSet::default(),
            started_keys,
            stopped_keys: FxHashSet::default(),
            next_query,
            current_present: FxHashSet::default(),
            current_ghosts: FxHashSet::default(),
            current_parents: FxHashMap::default(),
            returning: ReturnMode::NextWithGhosts,
            iterations: 0,
        }
    }

    /// Return the number of iterations performed so far.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Borrow the current frontier (next query set).
    ///
    /// Exposed so bindings can reflect Python's `_next_query` attribute,
    /// which existing callers in `graph.py` read (but do not mutate).
    pub fn next_query(&self) -> &FxHashSet<K> {
        &self.next_query
    }

    /// Snapshot of `(started_keys, excludes, included_keys)` describing what
    /// the searcher has reached. Matches Python's `get_state` return shape.
    ///
    /// This method intentionally calls the provider if the searcher is in
    /// `Next` mode, since we need the current query's children in order to
    /// list their parents as excludes. The subsequent iteration advances
    /// normally; the preview read is backed out of `seen`.
    pub fn get_state<P: ParentsProvider<K>>(
        &mut self,
        provider: &P,
    ) -> (FxHashSet<K>, FxHashSet<K>, FxHashSet<K>) {
        let next_query = if self.returning == ReturnMode::Next {
            let result = Self::do_query(&mut self.seen, &self.next_query, provider);
            // Undo the `seen` updates the preview made.
            for k in &result.next {
                self.seen.remove(k);
            }
            let mut nq = result.next;
            nq.extend(result.ghosts);
            nq
        } else {
            self.next_query.clone()
        };
        let mut excludes = self.stopped_keys.clone();
        excludes.extend(next_query);
        let included: FxHashSet<K> = self.seen.difference(&excludes).cloned().collect();
        (self.started_keys.clone(), excludes, included)
    }

    /// Advance the searcher and return the set yielded by Python's
    /// `__next__` / `next`.
    ///
    /// Each call yields the current query before its parents are queried,
    /// so ghosts are mixed in with present revisions.
    ///
    /// Returns `None` when there is nothing left to search.
    pub fn next_set<P: ParentsProvider<K>>(&mut self, provider: &P) -> Option<FxHashSet<K>> {
        if self.returning != ReturnMode::Next {
            self.returning = ReturnMode::Next;
            self.iterations += 1;
        } else {
            self.advance(provider);
        }
        if self.next_query.is_empty() {
            return None;
        }
        self.seen.extend(self.next_query.iter().cloned());
        Some(self.next_query.clone())
    }

    /// Advance the searcher and return `(present, ghosts)` the way Python's
    /// `next_with_ghosts` does.
    ///
    /// Returns `None` when there is nothing left to search.
    pub fn next_with_ghosts<P: ParentsProvider<K>>(
        &mut self,
        provider: &P,
    ) -> Option<(FxHashSet<K>, FxHashSet<K>)> {
        if self.returning != ReturnMode::NextWithGhosts {
            self.returning = ReturnMode::NextWithGhosts;
            self.advance(provider);
        }
        if self.next_query.is_empty() {
            return None;
        }
        self.advance(provider);
        Some((self.current_present.clone(), self.current_ghosts.clone()))
    }

    fn advance<P: ParentsProvider<K>>(&mut self, provider: &P) {
        self.iterations += 1;
        // Split borrow: `do_query` only needs to read `next_query` and
        // write `seen`, so we pass them as separate references and avoid
        // cloning `next_query` on every advance.
        let result = Self::do_query(&mut self.seen, &self.next_query, provider);
        self.current_present = result.found;
        self.current_ghosts = result.ghosts;
        self.next_query = result.next;
        self.current_parents = result.parents;
        // Ghosts become implicit stop points.
        self.stopped_keys
            .extend(self.current_ghosts.iter().cloned());
    }

    fn do_query<P: ParentsProvider<K>>(
        seen: &mut FxHashSet<K>,
        revisions: &FxHashSet<K>,
        provider: &P,
    ) -> QueryResult<K> {
        seen.extend(revisions.iter().cloned());

        // ParentsProvider takes a std HashSet by reference.
        let mut std_set: HashSet<K> = HashSet::with_capacity(revisions.len());
        for k in revisions {
            std_set.insert(k.clone());
        }
        let parent_map: ParentMap<K> = provider.get_parent_map(&std_set);

        let mut found: FxHashSet<K> = FxHashSet::default();
        let mut parents_of_found: FxHashSet<K> = FxHashSet::default();
        let mut parents_owned: FxHashMap<K, Vec<K>> = FxHashMap::default();
        for (rev_id, parents) in parent_map.iter() {
            found.insert(rev_id.clone());
            match parents {
                Parents::Known(ps) => {
                    parents_owned.insert(rev_id.clone(), ps.clone());
                    for p in ps {
                        if !seen.contains(p) {
                            parents_of_found.insert(p.clone());
                        }
                    }
                }
                // Python treats None parents as "continue" — no new parents
                // contributed.
                Parents::Ghost => {}
            }
        }
        let ghosts: FxHashSet<K> = revisions.difference(&found).cloned().collect();
        QueryResult {
            found,
            ghosts,
            next: parents_of_found,
            parents: parents_owned,
        }
    }

    /// Find already-seen ancestors of `revisions`.
    ///
    /// This walks backwards from `revisions` through `seen` keys only,
    /// querying the provider for parents. It matches the Python behavior:
    /// nodes not yet searched (in `next_query` when we're in `Next` mode)
    /// are skipped so we don't probe ahead of the search frontier.
    pub fn find_seen_ancestors<I, P>(&self, revisions: I, provider: &P) -> FxHashSet<K>
    where
        I: IntoIterator<Item = K>,
        P: ParentsProvider<K>,
    {
        let mut pending: FxHashSet<K> = revisions
            .into_iter()
            .filter(|r| self.seen.contains(r))
            .collect();
        let mut seen_ancestors: FxHashSet<K> = pending.iter().cloned().collect();

        // In `Next` mode `seen` contains nodes that have been *returned* but
        // whose parents haven't been queried yet. Skip those so we don't
        // probe ahead of the search frontier.
        let empty: FxHashSet<K> = FxHashSet::default();
        let not_searched_yet: &FxHashSet<K> = if self.returning == ReturnMode::Next {
            &self.next_query
        } else {
            &empty
        };

        pending.retain(|k| !not_searched_yet.contains(k));

        while !pending.is_empty() {
            let mut std_set: HashSet<K> = HashSet::with_capacity(pending.len());
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
                if self.seen.contains(&p) && !seen_ancestors.contains(&p) {
                    next_pending.insert(p);
                }
            }
            seen_ancestors.extend(next_pending.iter().cloned());
            next_pending.retain(|k| !not_searched_yet.contains(k));
            pending = next_pending;
        }
        seen_ancestors
    }

    /// Stop searching any of `revisions`. Returns the set of revisions
    /// actually removed from the current search frontier (not the ones that
    /// had already passed).
    pub fn stop_searching_any<I: IntoIterator<Item = K>>(&mut self, revisions: I) -> FxHashSet<K> {
        let revisions: FxHashSet<K> = revisions.into_iter().collect();
        let stopped: FxHashSet<K> = if self.returning == ReturnMode::Next {
            let stopped: FxHashSet<K> = self.next_query.intersection(&revisions).cloned().collect();
            self.next_query.retain(|k| !revisions.contains(k));
            stopped
        } else {
            let stopped_present: FxHashSet<K> = self
                .current_present
                .intersection(&revisions)
                .cloned()
                .collect();
            let stopped_ghosts: FxHashSet<K> = self
                .current_ghosts
                .intersection(&revisions)
                .cloned()
                .collect();
            let stopped: FxHashSet<K> = stopped_present.union(&stopped_ghosts).cloned().collect();
            self.current_present.retain(|k| !revisions.contains(k));
            self.current_ghosts.retain(|k| !revisions.contains(k));

            // Stopping X should stop returning parents of X — but only if no
            // other current node still references the same parent. Count
            // references to each parent from stopped_present, then decrement
            // for each non-stopped reference.
            let mut stop_rev_references: FxHashMap<K, i64> = FxHashMap::default();
            for rev in &stopped_present {
                if let Some(parents) = self.current_parents.get(rev) {
                    for parent_id in parents {
                        *stop_rev_references.entry(parent_id.clone()).or_insert(0) += 1;
                    }
                }
            }
            for parents in self.current_parents.values() {
                for parent_id in parents {
                    if let Some(count) = stop_rev_references.get_mut(parent_id) {
                        *count -= 1;
                    }
                }
            }
            let stop_parents: FxHashSet<K> = stop_rev_references
                .into_iter()
                .filter_map(|(k, refs)| if refs == 0 { Some(k) } else { None })
                .collect();
            self.next_query.retain(|k| !stop_parents.contains(k));
            stopped
        };
        self.stopped_keys.extend(stopped.iter().cloned());
        self.stopped_keys.extend(revisions);
        stopped
    }

    /// Add more revisions to the search.
    ///
    /// In `NextWithGhosts` mode this performs an immediate query on the new
    /// revisions and returns `Some((present, ghosts))`. In `Next` mode the
    /// new revisions join the current query without a provider call and the
    /// function returns `None`.
    pub fn start_searching<I, P>(
        &mut self,
        revisions: I,
        provider: &P,
    ) -> Option<(FxHashSet<K>, FxHashSet<K>)>
    where
        I: IntoIterator<Item = K>,
        P: ParentsProvider<K>,
    {
        let revisions: FxHashSet<K> = revisions.into_iter().collect();
        self.started_keys.extend(revisions.iter().cloned());
        let new_revisions: FxHashSet<K> = revisions.difference(&self.seen).cloned().collect();
        if self.returning == ReturnMode::Next {
            self.next_query.extend(new_revisions.iter().cloned());
            self.seen.extend(new_revisions);
            None
        } else {
            let result = Self::do_query(&mut self.seen, &revisions, provider);
            self.stopped_keys.extend(result.ghosts.iter().cloned());
            self.current_present.extend(result.found.iter().cloned());
            self.current_ghosts.extend(result.ghosts.iter().cloned());
            self.next_query.extend(result.next);
            for (k, v) in result.parents {
                self.current_parents.insert(k, v);
            }
            Some((result.found, result.ghosts))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DictParentsProvider;
    use std::collections::HashMap;

    fn provider(edges: &[(&'static str, &[&'static str])]) -> DictParentsProvider<&'static str> {
        let map: HashMap<&'static str, Vec<&'static str>> =
            edges.iter().map(|(k, ps)| (*k, ps.to_vec())).collect();
        DictParentsProvider::from(map)
    }

    fn as_set<const N: usize>(xs: [&'static str; N]) -> FxHashSet<&'static str> {
        xs.into_iter().collect()
    }

    #[test]
    fn next_walks_linear() {
        // a <- b <- c
        let p = provider(&[("a", &[]), ("b", &["a"]), ("c", &["b"])]);
        let mut s = BfsState::new(["c"]);
        assert_eq!(s.next_set(&p), Some(as_set(["c"])));
        assert_eq!(s.next_set(&p), Some(as_set(["b"])));
        assert_eq!(s.next_set(&p), Some(as_set(["a"])));
        assert_eq!(s.next_set(&p), None);
    }

    #[test]
    fn next_with_ghosts_splits() {
        // head -> present -> (child, ghost); child has no parents; ghost missing
        let p = provider(&[
            ("head", &["present"]),
            ("present", &["child", "ghost"]),
            ("child", &[]),
        ]);
        let mut s = BfsState::new(["head"]);
        assert_eq!(s.next_with_ghosts(&p), Some((as_set(["head"]), as_set([]))));
        assert_eq!(
            s.next_with_ghosts(&p),
            Some((as_set(["present"]), as_set([])))
        );
        assert_eq!(
            s.next_with_ghosts(&p),
            Some((as_set(["child"]), as_set(["ghost"])))
        );
        assert_eq!(s.next_with_ghosts(&p), None);
    }

    #[test]
    fn next_mode_mixes_ghosts_in_with_present() {
        // Same graph as above, but via next() — ghost should appear alongside child.
        let p = provider(&[
            ("head", &["present"]),
            ("present", &["child", "ghost"]),
            ("child", &[]),
        ]);
        let mut s = BfsState::new(["head"]);
        assert_eq!(s.next_set(&p), Some(as_set(["head"])));
        assert_eq!(s.next_set(&p), Some(as_set(["present"])));
        assert_eq!(s.next_set(&p), Some(as_set(["child", "ghost"])));
        assert_eq!(s.next_set(&p), None);
    }

    #[test]
    fn stop_searching_any_next_mode() {
        // In Next mode, `next_query` holds the set just returned by `next()`
        // (since the caller is given the query, not the results). So stopping
        // the set that was just yielded removes it from the frontier.
        let p = provider(&[
            ("head", &["present"]),
            ("present", &["stopped"]),
            ("stopped", &[]),
        ]);
        let mut s = BfsState::new(["head"]);
        assert_eq!(s.next_set(&p), Some(as_set(["head"])));
        assert_eq!(s.next_set(&p), Some(as_set(["present"])));
        let stopped = s.stop_searching_any(["present"]);
        assert_eq!(stopped, as_set(["present"]));
        // With `present` stopped before its parents are queried, the search
        // is now exhausted.
        assert_eq!(s.next_set(&p), None);
    }

    #[test]
    fn start_searching_next_with_ghosts_queries_immediately() {
        let p = provider(&[("new_root", &["its_parent"]), ("its_parent", &[])]);
        let mut s: BfsState<&'static str> = BfsState::new([] as [&'static str; 0]);
        let (found, ghosts) = s.start_searching(["new_root", "ghost"], &p).unwrap();
        assert!(found.contains(&"new_root"));
        assert!(ghosts.contains(&"ghost"));
    }

    /// Translated from `test_breadth_first_search_start_ghosts` in
    /// `vcsgraph/tests/test_graph.py`: starting with only a ghost, the first
    /// step yields just the ghost and then the search is exhausted.
    #[test]
    fn start_with_only_a_ghost() {
        let p = provider(&[("a-ghost", &[])]);
        let mut s = BfsState::new(["a-ghost"]);
        assert_eq!(s.next_set(&p), Some(as_set(["a-ghost"])));
        assert_eq!(s.next_set(&p), None);
    }

    /// Translated from `test_breadth_first_change_search`: stop the current
    /// frontier, start a new search from an unrelated revision, and
    /// verify the BFS picks up the new revision's ancestors.
    #[test]
    fn change_search_via_stop_and_start() {
        let p = provider(&[
            ("head", &["present"]),
            ("present", &["stopped"]),
            ("stopped", &[]),
            ("other", &["other_2"]),
            ("other_2", &[]),
        ]);
        let mut s = BfsState::new(["head"]);
        assert_eq!(s.next_with_ghosts(&p), Some((as_set(["head"]), as_set([]))));
        assert_eq!(
            s.next_with_ghosts(&p),
            Some((as_set(["present"]), as_set([])))
        );
        assert_eq!(s.stop_searching_any(["present"]), as_set(["present"]));
        let (present, ghosts) = s.start_searching(["other", "other_ghost"], &p).unwrap();
        assert_eq!(present, as_set(["other"]));
        assert_eq!(ghosts, as_set(["other_ghost"]));
        assert_eq!(
            s.next_with_ghosts(&p),
            Some((as_set(["other_2"]), as_set([])))
        );
        assert_eq!(s.next_with_ghosts(&p), None);
    }

    const NULL: &str = "null:";

    /// Mirrors `test_breadth_first_search_change_next_to_next_with_ghosts`:
    /// interleave `next()` and `next_with_ghosts()` on the same searcher
    /// and verify both modes produce sensible values.
    #[test]
    fn change_next_to_next_with_ghosts() {
        let p = provider(&[
            ("head", &["present"]),
            ("present", &["child", "ghost"]),
            ("child", &[]),
        ]);
        let mut s = BfsState::new(["head"]);
        assert_eq!(s.next_with_ghosts(&p), Some((as_set(["head"]), as_set([]))));
        assert_eq!(s.next_set(&p), Some(as_set(["present"])));
        assert_eq!(
            s.next_with_ghosts(&p),
            Some((as_set(["child"]), as_set(["ghost"])))
        );
        assert_eq!(s.next_set(&p), None);

        // Symmetric: start with next(), switch to next_with_ghosts().
        let mut s = BfsState::new(["head"]);
        assert_eq!(s.next_set(&p), Some(as_set(["head"])));
        assert_eq!(
            s.next_with_ghosts(&p),
            Some((as_set(["present"]), as_set([])))
        );
        assert_eq!(s.next_set(&p), Some(as_set(["child", "ghost"])));
        assert_eq!(s.next_with_ghosts(&p), None);
    }

    /// Mirrors `test_breadth_first_get_result_excludes_current_pending`:
    /// at the start, nothing is seen; after each advance, `get_state()`
    /// reports the started keys, the excluded set, and the included
    /// (fully explored) set.
    #[test]
    fn get_state_excludes_current_pending() {
        let p = provider(&[("head", &["child"]), ("child", &[NULL]), (NULL, &[])]);
        let mut s = BfsState::new(["head"]);
        let (started, excludes, included) = s.get_state(&p);
        assert_eq!(started, as_set(["head"]));
        assert_eq!(excludes, as_set(["head"]));
        assert_eq!(included, as_set([]));
        assert_eq!(s.seen, as_set([]));

        // After next: head is yielded, still excluded because child is
        // the next frontier.
        s.next_set(&p);
        let (_, excludes, included) = s.get_state(&p);
        assert_eq!(excludes, as_set(["child"]));
        assert_eq!(included, as_set(["head"]));
        assert_eq!(s.seen, as_set(["head"]));

        // After child: null is the next frontier.
        s.next_set(&p);
        let (_, excludes, included) = s.get_state(&p);
        assert_eq!(excludes, as_set([NULL]));
        assert_eq!(included, as_set(["head", "child"]));

        // After null: nothing left in the frontier.
        s.next_set(&p);
        let (_, excludes, included) = s.get_state(&p);
        assert_eq!(excludes, as_set([]));
        assert_eq!(included, as_set(["head", "child", NULL]));
    }

    /// Mirrors `test_breadth_first_stop_searching_not_queried`: a client
    /// may tell the searcher to stop a key, and stopped_keys records it
    /// for later exclusion from the result's included-set.
    #[test]
    fn stop_searching_records_stops() {
        let p = provider(&[
            ("head", &["child", "ghost1"]),
            ("child", &[NULL]),
            (NULL, &[]),
        ]);
        let mut s = BfsState::new(["head"]);
        s.next_set(&p); // yields head
        s.stop_searching_any([NULL, "ghost1"]);
        // The stopped keys are in stopped_keys regardless of whether
        // they've been visited yet.
        assert!(s.stopped_keys.contains(&NULL));
        assert!(s.stopped_keys.contains(&"ghost1"));
        // get_state() should exclude the stopped keys from the
        // "included" snapshot.
        let (_, excludes, included) = s.get_state(&p);
        assert!(excludes.contains(&NULL));
        assert!(excludes.contains(&"ghost1"));
        assert!(!included.contains(&NULL));
        assert!(!included.contains(&"ghost1"));
    }

    /// Mirrors `test_breadth_first_stop_searching_late`: stopping a key
    /// from an older iteration should still exclude it from the result.
    #[test]
    fn stop_searching_late() {
        let p = provider(&[
            ("head", &["middle"]),
            ("middle", &["child"]),
            ("child", &[NULL]),
            (NULL, &[]),
        ]);
        let mut s = BfsState::new(["head"]);
        s.next_set(&p); // yields head
        s.next_set(&p); // yields middle
        s.next_set(&p); // yields child
                        // Now stop both middle and child retroactively.
        s.stop_searching_any(["middle", "child"]);
        assert!(s.stopped_keys.contains(&"middle"));
        assert!(s.stopped_keys.contains(&"child"));
        // After the stop, the remaining state reflects that only the
        // original head is included.
        let (_, excludes, included) = s.get_state(&p);
        assert!(excludes.contains(&"middle"));
        assert!(excludes.contains(&"child"));
        assert_eq!(included, as_set(["head"]));
    }

    /// Mirrors `test_breadth_first_get_result_starting_a_ghost_ghost_is_excluded`:
    /// start_searching a ghost key mid-walk. The ghost is recorded in seen
    /// but gets filed under stopped_keys so it is excluded from included().
    #[test]
    fn start_searching_a_ghost_excludes_it() {
        let p = provider(&[("head", &["child"]), ("child", &[NULL]), (NULL, &[])]);
        let mut s = BfsState::new(["head"]);
        // Start-searching a ghost while in next_with_ghosts mode (the
        // default after construction). This returns (present, ghosts).
        let (present, ghosts) = s.start_searching(["ghost"], &p).unwrap();
        assert_eq!(present, as_set([]));
        assert_eq!(ghosts, as_set(["ghost"]));
        // ghost is now in stopped_keys so included() doesn't report it.
        assert!(s.stopped_keys.contains(&"ghost"));
    }

    /// Mirrors `test_breadth_first_revision_count_includes_NULL_REVISION`:
    /// walking to the sentinel should count it as part of `seen`.
    #[test]
    fn walk_includes_null_revision() {
        let p = provider(&[("head", &[NULL]), (NULL, &[])]);
        let mut s = BfsState::new(["head"]);
        s.next_set(&p); // yields head
        s.next_set(&p); // yields null
        assert_eq!(s.seen, as_set(["head", NULL]));
        assert_eq!(s.next_set(&p), None);
    }

    /// Mirrors `test_breadth_first_search_get_result_after_StopIteration`:
    /// hitting StopIteration should not invalidate the searcher; a
    /// subsequent get_state() still works.
    #[test]
    fn get_state_after_stop_iteration() {
        let p = provider(&[("head", &[NULL]), (NULL, &[])]);
        let mut s = BfsState::new(["head"]);
        while s.next_set(&p).is_some() {}
        // No more to yield.
        assert_eq!(s.next_set(&p), None);
        let (started, _excludes, included) = s.get_state(&p);
        assert_eq!(started, as_set(["head"]));
        assert!(included.contains(&"head"));
        assert!(included.contains(&NULL));
    }

    /// find_seen_ancestors should walk the parent chain and collect all
    /// ancestors already in `seen` — not new ones.
    #[test]
    fn find_seen_ancestors_walks_seen_chain() {
        let p = provider(&[
            ("head", &["middle"]),
            ("middle", &["child"]),
            ("child", &[NULL]),
            (NULL, &[]),
        ]);
        let mut s = BfsState::new(["head"]);
        // Walk the whole thing.
        while s.next_set(&p).is_some() {}
        // Ask for ancestors of "middle" — should find middle, child, null.
        let anc = s.find_seen_ancestors(["middle"], &p);
        assert!(anc.contains(&"middle"));
        assert!(anc.contains(&"child"));
        assert!(anc.contains(&NULL));
    }

    /// find_seen_ancestors should filter out keys not in seen.
    #[test]
    fn find_seen_ancestors_skips_unseen() {
        let p = provider(&[("head", &[NULL]), (NULL, &[]), ("unrelated", &[])]);
        let mut s = BfsState::new(["head"]);
        s.next_set(&p); // yields head
        let anc = s.find_seen_ancestors(["unrelated"], &p);
        // "unrelated" isn't in seen, so find_seen_ancestors returns an
        // empty set for it.
        assert!(!anc.contains(&"unrelated"));
    }

    /// stop_searching_any should return only the keys that were actually
    /// removed from the current frontier (not keys that had already been
    /// processed).
    #[test]
    fn stop_searching_any_returns_only_effective_stops() {
        let p = provider(&[("head", &["a"]), ("a", &["b"]), ("b", &[])]);
        let mut s = BfsState::new(["head"]);
        s.next_set(&p); // yields head
        s.next_set(&p); // yields a
                        // `head` is already returned; stopping it should report it as
                        // stopped but a no longer in frontier means only keys that are in
                        // next_query at stop time get returned.
        let stopped = s.stop_searching_any(["a"]);
        assert_eq!(stopped, as_set(["a"]));
        // After stopping a, the search is exhausted.
        assert_eq!(s.next_set(&p), None);
    }

    /// Starting an already-seen key should be a no-op on `seen` and the
    /// key should not be re-queried.
    #[test]
    fn start_searching_already_seen_is_noop() {
        let p = provider(&[("head", &["child"]), ("child", &[])]);
        let mut s = BfsState::new(["head"]);
        s.next_set(&p); // yields head, frontier now contains "child"
        let pre_seen = s.seen.clone();
        s.start_searching(["head"], &p);
        // seen should be unchanged (head was already there).
        assert_eq!(s.seen, pre_seen);
    }
}
