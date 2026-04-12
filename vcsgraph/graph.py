# Copyright (C) 2007-2011 Canonical Ltd
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

"""Graph algorithms for version control systems."""

__all__ = ["collapse_linear_regions", "invert_parent_map"]

from . import errors
from ._graph_rs import (
    _BreadthFirstSearcher as _RustBreadthFirstSearcher,
)
from ._graph_rs import (
    CachingParentsProvider,
    FrozenHeadsCache,
    GraphThunkIdsToKeys,
    HeadsCache,
    _RustGraph,
    collapse_linear_regions,
    invert_parent_map,
)

# NULL_REVISION constant
NULL_REVISION = b"null:"

# DIAGRAM of terminology
#       A
#       /\
#      B  C
#      |  |\
#      D  E F
#      |\/| |
#      |/\|/
#      G  H
#
# In this diagram, relative to G and H:
# A, B, C, D, E are common ancestors.
# C, D and E are border ancestors, because each has a non-common descendant.
# D and E are least common ancestors because none of their descendants are
# common ancestors.
# C is not a least common ancestor because its descendant, E, is a common
# ancestor.
#
# The find_unique_lca algorithm will pick A in two steps:
# 1. find_lca('G', 'H') => ['D', 'E']
# 2. Since len(['D', 'E']) > 1, find_lca('D', 'E') => ['A']


class DictParentsProvider:
    """A parents provider for Graph objects."""

    def __init__(self, ancestry):
        self.ancestry = ancestry

    def __repr__(self):
        return f"DictParentsProvider({self.ancestry!r})"

    # Note: DictParentsProvider does not implement get_cached_parent_map
    #       Arguably, the data is clearly cached in memory. However, this class
    #       is mostly used for testing, and it keeps the tests clean to not
    #       change it.

    def get_parent_map(self, keys):
        """See StackedParentsProvider.get_parent_map."""
        ancestry = self.ancestry
        return {k: tuple(ancestry[k]) for k in keys if k in ancestry}


class StackedParentsProvider:
    """A parents provider which stacks (or unions) multiple providers.

    The providers are queries in the order of the provided parent_providers.
    """

    def __init__(self, parent_providers):
        self._parent_providers = parent_providers

    def __repr__(self):
        return f"{self.__class__.__name__}({self._parent_providers!r})"

    def get_parent_map(self, keys):
        """Get a mapping of keys => parents.

        A dictionary is returned with an entry for each key present in this
        source. If this source doesn't have information about a key, it should
        not include an entry.

        [NULL_REVISION] is used as the parent of the first user-committed
        revision.  Its parent list is empty.

        :param keys: An iterable returning keys to check (eg revision_ids)
        :return: A dictionary mapping each key to its parents
        """
        found = {}
        remaining = set(keys)
        # This adds getattr() overhead to each get_parent_map call. However,
        # this is StackedParentsProvider, which means we're dealing with I/O
        # (either local indexes, or remote RPCs), so CPU overhead should be
        # minimal.
        for parents_provider in self._parent_providers:
            get_cached = getattr(parents_provider, "get_cached_parent_map", None)
            if get_cached is None:
                continue
            new_found = get_cached(remaining)
            found.update(new_found)
            remaining.difference_update(new_found)
            if not remaining:
                break
        if not remaining:
            return found
        for parents_provider in self._parent_providers:
            try:
                new_found = parents_provider.get_parent_map(remaining)
            except errors.UnsupportedOperation:
                continue
            found.update(new_found)
            remaining.difference_update(new_found)
            if not remaining:
                break
        return found


class CallableToParentsProviderAdapter:
    """A parents provider that adapts any callable to the parents provider API.

    i.e. it accepts calls to self.get_parent_map and relays them to the
    callable it was constructed with.
    """

    def __init__(self, a_callable):
        self.callable = a_callable

    def __repr__(self):
        return f"{self.__class__.__name__}({self.callable!r})"

    def get_parent_map(self, keys):
        return self.callable(keys)


class Graph:
    """Provide incremental access to revision graphs.

    This is the generic implementation; it is intended to be subclassed to
    specialize it for other repository types.
    """

    def __init__(self, parents_provider):
        """Construct a Graph that uses several graphs as its input.

        This should not normally be invoked directly, because there may be
        specialized implementations for particular repository types.  See
        Repository.get_graph().

        :param parents_provider: An object providing a get_parent_map call
            conforming to the behavior of
            StackedParentsProvider.get_parent_map.
        """
        if getattr(parents_provider, "get_parents", None) is not None:
            self.get_parents = parents_provider.get_parents
        if getattr(parents_provider, "get_parent_map", None) is not None:
            self.get_parent_map = parents_provider.get_parent_map
        self._parents_provider = parents_provider
        # Rust-backed helper for methods that have been ported. Uses the same
        # parents provider; the Rust side calls back into Python via a GIL
        # adapter when it needs a parent lookup.
        self._rs = _RustGraph(parents_provider)

    def __repr__(self):
        return f"Graph({self._parents_provider!r})"

    def find_lca(self, *revisions):
        """Determine the lowest common ancestors of the provided revisions.

        A lowest common ancestor is a common ancestor none of whose
        descendants are common ancestors.  In graphs, unlike trees, there may
        be multiple lowest common ancestors.

        This algorithm has two phases.  Phase 1 identifies border ancestors,
        and phase 2 filters border ancestors to determine lowest common
        ancestors.

        In phase 1, border ancestors are identified, using a breadth-first
        search starting at the bottom of the graph.  Searches are stopped
        whenever a node or one of its descendants is determined to be common

        In phase 2, the border ancestors are filtered to find the least
        common ancestors.  This is done by searching the ancestries of each
        border ancestor.

        Phase 2 is perfomed on the principle that a border ancestor that is
        not an ancestor of any other border ancestor is a least common
        ancestor.

        Searches are stopped when they find a node that is determined to be a
        common ancestor of all border ancestors, because this shows that it
        cannot be a descendant of any border ancestor.

        The scaling of this operation should be proportional to:

        1. The number of uncommon ancestors
        2. The number of border ancestors
        3. The length of the shortest path between a border ancestor and an
           ancestor of all border ancestors.
        """
        return self._rs.find_lca(revisions)

    def find_difference(self, left_revision, right_revision):
        """Determine the graph difference between two revisions."""
        return self._rs.find_difference(left_revision, right_revision)

    def find_descendants(self, old_key, new_key):
        """Find descendants of old_key that are ancestors of new_key."""
        return self._rs.find_descendants(old_key, new_key)

    def _find_descendant_ancestors(self, old_key, new_key):
        """Find ancestors of new_key that may be descendants of old_key."""
        return self._rs._find_descendant_ancestors(old_key, new_key)

    def _remove_simple_descendants(self, revisions, parent_map):
        """Remove revisions which are children of other ones in the set.

        This doesn't do any graph searching, it just checks the immediate
        parent_map to find if there are any children which can be removed.

        :param revisions: A set of revision_ids
        :return: A set of revision_ids with the children removed
        """
        simple_ancestors = set(revisions)
        for revision, parent_ids in parent_map.items():
            if parent_ids is None:
                continue
            for parent_id in parent_ids:
                if parent_id in revisions:
                    simple_ancestors.discard(revision)
                    break
        return simple_ancestors

    def get_child_map(self, keys):
        """Get a mapping from parents to children of the specified keys.

        This is simply the inversion of get_parent_map.  Only supplied keys
        will be discovered as children.
        :return: a dict of key:child_list for keys.
        """
        parent_map = self._parents_provider.get_parent_map(keys)
        parent_child = {}
        for child, parents in sorted(parent_map.items()):
            for parent in parents:
                parent_child.setdefault(parent, []).append(child)
        return parent_child

    def find_distance_to_null(self, target_revision_id, known_revision_ids):
        """Find the left-hand distance to the NULL_REVISION.

        (This can also be considered the revno of a branch at
        target_revision_id.)

        :param target_revision_id: A revision_id which we would like to know
            the revno for.
        :param known_revision_ids: [(revision_id, revno)] A list of known
            revno, revision_id tuples. We'll use this to seed the search.
        """
        return self._rs.find_distance_to_null(target_revision_id, known_revision_ids)

    def find_lefthand_distances(self, keys):
        """Find the distance to null for all the keys in keys.

        :param keys: keys to lookup.
        :return: A dict key->distance for all of keys.
        """
        return self._rs.find_lefthand_distances(keys)

    def find_unique_ancestors(self, unique_revision, common_revisions):
        """Find the unique ancestors for a revision versus others.

        This returns the ancestry of unique_revision, excluding all revisions
        in the ancestry of common_revisions. If unique_revision is in the
        ancestry, then the empty set will be returned.

        :param unique_revision: The revision_id whose ancestry we are
            interested in.
            (XXX: Would this API be better if we allowed multiple revisions on
            to be searched here?)
        :param common_revisions: Revision_ids of ancestries to exclude.
        :return: A set of revisions in the ancestry of unique_revision
        """
        return self._rs.find_unique_ancestors(unique_revision, common_revisions)

    def get_parent_map(self, revisions):  # type: ignore
        """Get a map of key:parent_list for revisions.

        This implementation delegates to get_parents, for old parent_providers
        that do not supply get_parent_map.
        """
        result = {}
        for rev, parents in self.get_parents(revisions):
            if parents is not None:
                result[rev] = parents
        return result

    def _make_breadth_first_searcher(self, revisions):
        return _RustBreadthFirstSearcher(revisions, self)

    def heads(self, keys):
        """Return the heads from amongst keys.

        This is done by searching the ancestries of each key.  Any key that is
        reachable from another key is not returned; all the others are.

        This operation scales with the relative depth between any two keys. If
        any two keys are completely disconnected all ancestry of both sides
        will be retrieved.

        :param keys: An iterable of keys.
        :return: A set of the heads. Note that as a set there is no ordering
            information. Callers will need to filter their input to create
            order if they need it.
        """
        return self._rs.heads(keys)

    def find_merge_order(self, tip_revision_id, lca_revision_ids):
        """Find the order that each revision was merged into tip.

        This basically just walks backwards with a stack, and walks left-first
        until it finds a node to stop.
        """
        return self._rs.find_merge_order(tip_revision_id, lca_revision_ids)

    def find_lefthand_merger(self, merged_key, tip_key):
        """Find the first lefthand ancestor of tip_key that merged merged_key.

        We do this by first finding the descendants of merged_key, then
        walking through the lefthand ancestry of tip_key until we find a key
        that doesn't descend from merged_key.  Its child is the key that
        merged merged_key.

        :return: The first lefthand ancestor of tip_key to merge merged_key.
            merged_key if it is a lefthand ancestor of tip_key.
            None if no ancestor of tip_key merged merged_key.
        """
        return self._rs.find_lefthand_merger(merged_key, tip_key)

    def find_unique_lca(self, left_revision, right_revision, count_steps=False):
        """Find a unique LCA.

        Find lowest common ancestors.  If there is no unique  common
        ancestor, find the lowest common ancestors of those ancestors.

        Iteration stops when a unique lowest common ancestor is found.
        The graph origin is necessarily a unique lowest common ancestor.

        Note that None is not an acceptable substitute for NULL_REVISION.
        in the input for this method.

        :param count_steps: If True, the return value will be a tuple of
            (unique_lca, steps) where steps is the number of times that
            find_lca was run.  If False, only unique_lca is returned.
        """
        return self._rs.find_unique_lca(left_revision, right_revision, count_steps)

    def iter_ancestry(self, revision_ids):
        """Iterate the ancestry of this revision.

        :param revision_ids: Nodes to start the search
        :return: Yield tuples mapping a revision_id to its parents for the
            ancestry of revision_id.
            Ghosts will be returned with None as their parents, and nodes
            with no parents will have NULL_REVISION as their only parent. (As
            defined by get_parent_map.)
            There will also be a node for (NULL_REVISION, ())
        """
        yield from self._rs.iter_ancestry(revision_ids)

    def iter_lefthand_ancestry(self, start_key, stop_keys=None):
        if stop_keys is None:
            stop_keys = ()
        next_key = start_key

        def get_parents(key):
            try:
                return self._parents_provider.get_parent_map([key])[key]
            except KeyError as err:
                raise errors.RevisionNotPresent(next_key, self) from err

        while True:
            if next_key in stop_keys:
                return
            parents = get_parents(next_key)
            yield next_key
            if len(parents) == 0:
                return
            else:
                next_key = parents[0]

    def iter_topo_order(self, revisions):
        """Iterate through the input revisions in topological order.

        This sorting only ensures that parents come before their children.
        An ancestor may sort after a descendant if the relationship is not
        visible in the supplied list of revisions.
        """
        return iter(self._rs.iter_topo_order(revisions))

    def is_ancestor(self, candidate_ancestor, candidate_descendant):
        """Determine whether a revision is an ancestor of another.

        We answer this using heads() as heads() has the logic to perform the
        smallest number of parent lookups to determine the ancestral
        relationship between N revisions.
        """
        return self._rs.is_ancestor(candidate_ancestor, candidate_descendant)

    def is_between(self, revid, lower_bound_revid, upper_bound_revid):
        """Determine whether a revision is between two others.

        returns true if and only if:
        lower_bound_revid <= revid <= upper_bound_revid
        """
        return self._rs.is_between(revid, lower_bound_revid, upper_bound_revid)


_counters = [0, 0, 0, 0, 0, 0, 0]

# Import KnownGraph to make it available through this module for compatibility
