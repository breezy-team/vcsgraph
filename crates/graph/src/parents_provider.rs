use crate::{ParentMap, Parents};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Mutex;

pub trait ParentsProvider<K: PartialEq + Eq + Clone + Hash> {
    fn get_parent_map(&self, keys: &HashSet<K>) -> ParentMap<K>;
}

pub struct StackedParentsProvider<K> {
    parent_providers: Vec<Box<dyn ParentsProvider<K>>>,
}

impl<K> StackedParentsProvider<K> {
    pub fn new(parent_providers: Vec<Box<dyn ParentsProvider<K>>>) -> Self {
        StackedParentsProvider { parent_providers }
    }
}

impl<K: Hash + Eq + Clone> ParentsProvider<K> for StackedParentsProvider<K> {
    fn get_parent_map(&self, keys: &HashSet<K>) -> ParentMap<K> {
        let mut found = ParentMap::new();
        let mut remaining = keys.clone();

        for parent_provider in self.parent_providers.iter() {
            if remaining.is_empty() {
                break;
            }

            let new_found = parent_provider.get_parent_map(&remaining);
            for k in new_found.keys() {
                remaining.remove(k);
            }
            found.extend(new_found);
        }

        found
    }
}

pub struct DictParentsProvider<K: Hash + Eq + Clone>(ParentMap<K>);

impl<K: Hash + Eq + Clone> From<ParentMap<K>> for DictParentsProvider<K> {
    fn from(parent_map: ParentMap<K>) -> Self {
        DictParentsProvider(parent_map)
    }
}

impl<K: Hash + Eq + Clone> From<HashMap<K, Vec<K>>> for DictParentsProvider<K> {
    fn from(parent_map: HashMap<K, Vec<K>>) -> Self {
        DictParentsProvider::new(ParentMap(
            parent_map
                .into_iter()
                .map(|(k, v)| (k, Parents::Known(v)))
                .collect(),
        ))
    }
}

impl<K: Hash + Eq + Clone> DictParentsProvider<K> {
    pub fn new(parent_map: ParentMap<K>) -> Self {
        DictParentsProvider(parent_map)
    }
}

impl<K: Hash + Eq + Clone> ParentsProvider<K> for DictParentsProvider<K> {
    fn get_parent_map(&self, keys: &HashSet<K>) -> ParentMap<K> {
        ParentMap(
            keys.iter()
                .filter_map(|k| self.0.get_key_value(k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    }
}

/// A parents provider which caches its lookups.
///
/// Wraps an inner `ParentsProvider` and memoizes every `(key, parents)`
/// pair it returns. When cache-misses-tracking is enabled, keys that were
/// requested but not present in the inner provider are also remembered so
/// we don't re-request them.
///
/// The cache can be disabled and re-enabled at runtime; disabling clears
/// the cache entirely.
pub struct CachingParentsProvider<K: Hash + Eq + Clone, P: ParentsProvider<K>> {
    inner: P,
    // Interior mutability so `get_parent_map(&self, ...)` can populate the
    // cache. The whole provider is still Sync (via Mutex) which matches the
    // base trait's `&self` contract.
    state: Mutex<CacheState<K>>,
}

struct CacheState<K: Hash + Eq + Clone> {
    /// None when the cache is disabled, Some when enabled.
    cache: Option<FxHashMap<K, Parents<K>>>,
    /// Keys known to be missing from the inner provider. Only populated
    /// when `cache_misses` is true.
    missing_keys: FxHashSet<K>,
    /// Whether to remember keys that aren't in the inner provider.
    cache_misses: bool,
}

impl<K: Hash + Eq + Clone, P: ParentsProvider<K>> CachingParentsProvider<K, P> {
    /// Create a caching wrapper around `inner`. The cache is enabled by
    /// default with cache-misses tracking on.
    pub fn new(inner: P) -> Self {
        CachingParentsProvider {
            inner,
            state: Mutex::new(CacheState {
                cache: Some(FxHashMap::default()),
                missing_keys: FxHashSet::default(),
                cache_misses: true,
            }),
        }
    }

    /// Enable the cache. Matches Python's semantics: calling this when the
    /// cache is already enabled is an error. `cache_misses` controls
    /// whether missing keys are remembered between calls.
    pub fn enable_cache(&self, cache_misses: bool) -> Result<(), &'static str> {
        let mut state = self.state.lock().unwrap();
        if state.cache.is_some() {
            return Err("Cache enabled when already enabled.");
        }
        state.cache = Some(FxHashMap::default());
        state.cache_misses = cache_misses;
        state.missing_keys = FxHashSet::default();
        Ok(())
    }

    /// Disable and clear the cache.
    pub fn disable_cache(&self) {
        let mut state = self.state.lock().unwrap();
        state.cache = None;
        state.cache_misses = false;
        state.missing_keys = FxHashSet::default();
    }

    /// Return a snapshot of the current cache, or `None` if disabled.
    pub fn get_cached_map(&self) -> Option<FxHashMap<K, Parents<K>>> {
        let state = self.state.lock().unwrap();
        state.cache.clone()
    }

    /// Return entries from the cache without consulting the inner provider.
    pub fn get_cached_parent_map(&self, keys: &HashSet<K>) -> ParentMap<K> {
        let state = self.state.lock().unwrap();
        let Some(cache) = state.cache.as_ref() else {
            return ParentMap::new();
        };
        ParentMap(
            keys.iter()
                .filter_map(|k| cache.get_key_value(k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    }

    /// Note that `key` was missing from the inner provider.
    pub fn note_missing_key(&self, key: K) {
        let mut state = self.state.lock().unwrap();
        if state.cache_misses {
            state.missing_keys.insert(key);
        }
    }

    /// Snapshot of the missing-keys set.
    pub fn missing_keys(&self) -> FxHashSet<K> {
        self.state.lock().unwrap().missing_keys.clone()
    }

    /// Borrow the inner provider.
    pub fn inner(&self) -> &P {
        &self.inner
    }
}

impl<K: Hash + Eq + Clone, P: ParentsProvider<K>> ParentsProvider<K>
    for CachingParentsProvider<K, P>
{
    fn get_parent_map(&self, keys: &HashSet<K>) -> ParentMap<K> {
        // Fast path: cache disabled — delegate straight to inner.
        {
            let state = self.state.lock().unwrap();
            if state.cache.is_none() {
                drop(state);
                // Note: Python filters the response to only the requested
                // keys with non-None values; we do the same by filtering
                // known parents below.
                let pm = self.inner.get_parent_map(keys);
                let mut result = ParentMap::new();
                for k in keys {
                    if let Some(v) = pm.get(k) {
                        if matches!(v, Parents::Known(_)) {
                            result.insert(k.clone(), v.clone());
                        }
                    }
                }
                return result;
            }
        }

        // Determine which keys we still need to fetch from the inner
        // provider (not in cache and not known-missing).
        let needed: HashSet<K> = {
            let state = self.state.lock().unwrap();
            let cache = state.cache.as_ref().unwrap();
            keys.iter()
                .filter(|k| !cache.contains_key(*k) && !state.missing_keys.contains(*k))
                .cloned()
                .collect()
        };

        if !needed.is_empty() {
            let fetched = self.inner.get_parent_map(&needed);
            let mut state = self.state.lock().unwrap();
            let cache = state.cache.as_mut().unwrap();
            for (k, v) in fetched.iter() {
                cache.insert(k.clone(), v.clone());
            }
            if state.cache_misses {
                for k in &needed {
                    if !fetched.contains_key(k) {
                        state.missing_keys.insert(k.clone());
                    }
                }
            }
        }

        // Build the response from the cache, filtering out ghosts/None the
        // same way Python does.
        let state = self.state.lock().unwrap();
        let cache = state.cache.as_ref().unwrap();
        let mut result = ParentMap::new();
        for k in keys {
            if let Some(v) = cache.get(k) {
                if matches!(v, Parents::Known(_)) {
                    result.insert(k.clone(), v.clone());
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A ParentsProvider wrapper that counts how many distinct keys were
    /// requested across all calls to `get_parent_map`. Used to verify the
    /// caching wrapper avoids redundant lookups.
    struct CountingProvider<K: Hash + Eq + Clone, P: ParentsProvider<K>> {
        inner: P,
        requested: RefCell<Vec<K>>,
    }

    impl<K: Hash + Eq + Clone, P: ParentsProvider<K>> CountingProvider<K, P> {
        fn new(inner: P) -> Self {
            CountingProvider {
                inner,
                requested: RefCell::new(Vec::new()),
            }
        }
    }

    impl<K: Hash + Eq + Clone, P: ParentsProvider<K>> ParentsProvider<K> for CountingProvider<K, P> {
        fn get_parent_map(&self, keys: &HashSet<K>) -> ParentMap<K> {
            self.requested.borrow_mut().extend(keys.iter().cloned());
            self.inner.get_parent_map(keys)
        }
    }

    fn dict(edges: &[(&'static str, &[&'static str])]) -> DictParentsProvider<&'static str> {
        let map: HashMap<&'static str, Vec<&'static str>> =
            edges.iter().map(|(k, ps)| (*k, ps.to_vec())).collect();
        DictParentsProvider::from(map)
    }

    fn query(
        cp: &CachingParentsProvider<
            &'static str,
            CountingProvider<&'static str, DictParentsProvider<&'static str>>,
        >,
        keys: &[&'static str],
    ) -> ParentMap<&'static str> {
        let hs: HashSet<&'static str> = keys.iter().copied().collect();
        cp.get_parent_map(&hs)
    }

    #[test]
    fn caching_returns_known_parents() {
        let inner = CountingProvider::new(dict(&[("a", &[]), ("b", &["a"])]));
        let cp = CachingParentsProvider::new(inner);
        let pm = query(&cp, &["a", "b"]);
        assert_eq!(pm.get(&"a"), Some(&Parents::Known(vec![])));
        assert_eq!(pm.get(&"b"), Some(&Parents::Known(vec!["a"])));
    }

    #[test]
    fn caching_avoids_refetching_known_keys() {
        let inner = CountingProvider::new(dict(&[("a", &[]), ("b", &["a"])]));
        let cp = CachingParentsProvider::new(inner);
        query(&cp, &["a", "b"]);
        query(&cp, &["a", "b"]);
        // Only one round trip for each key.
        let requested = cp.inner().requested.borrow();
        let mut seen = FxHashSet::default();
        for k in requested.iter() {
            seen.insert(*k);
        }
        assert_eq!(seen, ["a", "b"].into_iter().collect::<FxHashSet<_>>());
        assert_eq!(requested.len(), 2);
    }

    #[test]
    fn caching_remembers_missing_keys() {
        let inner = CountingProvider::new(dict(&[("a", &[])]));
        let cp = CachingParentsProvider::new(inner);
        query(&cp, &["a", "missing"]);
        query(&cp, &["missing"]);
        // "missing" should have been requested exactly once.
        let requested = cp.inner().requested.borrow();
        let count = requested.iter().filter(|k| **k == "missing").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn disable_cache_clears_state() {
        let inner = CountingProvider::new(dict(&[("a", &[])]));
        let cp = CachingParentsProvider::new(inner);
        query(&cp, &["a"]);
        cp.disable_cache();
        query(&cp, &["a"]);
        // With the cache disabled, every call hits the inner provider.
        let requested = cp.inner().requested.borrow();
        let count = requested.iter().filter(|k| **k == "a").count();
        assert_eq!(count, 2);
    }

    #[test]
    fn enable_while_enabled_errors() {
        let cp = CachingParentsProvider::new(dict(&[("a", &[])]));
        assert!(cp.enable_cache(true).is_err());
    }

    #[test]
    fn reenabling_after_disable_works() {
        let cp = CachingParentsProvider::new(dict(&[("a", &[])]));
        cp.disable_cache();
        cp.enable_cache(true).unwrap();
        let hs: HashSet<&'static str> = ["a"].into_iter().collect();
        let pm = cp.get_parent_map(&hs);
        assert_eq!(pm.get(&"a"), Some(&Parents::Known(vec![])));
    }
}
