# Copyright (C) 2005-2010 Canonical Ltd
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

"""Exceptions for vcsgraph operations."""


class Error(Exception):
    """Base class for vcsgraph exceptions."""

    pass


class UnsupportedOperation(Error):
    """Requested operation is not supported."""

    pass


class GhostRevisionsHaveNoRevno(Error):
    """When searching for revnos, we encounter a ghost."""

    def __init__(self, revision_id, ghost_revision_id):
        """Initialize GhostRevisionsHaveNoRevno exception.

        Args:
            revision_id: The revision ID being searched for.
            ghost_revision_id: The ghost revision ID encountered.
        """
        self.revision_id = revision_id
        self.ghost_revision_id = ghost_revision_id
        super().__init__(
            f"Ghost revision {ghost_revision_id!r} has no revno, "
            f"cannot determine revno for {revision_id!r}"
        )

    def __eq__(self, other):
        if not isinstance(other, GhostRevisionsHaveNoRevno):
            return False
        return (
            self.revision_id == other.revision_id
            and self.ghost_revision_id == other.ghost_revision_id
        )


class InvalidRevisionId(Error):
    """Invalid revision ID."""

    def __init__(self, revision_id, client):
        """Initialize InvalidRevisionId exception.

        Args:
            revision_id: The invalid revision ID.
            client: The client that encountered the invalid revision.
        """
        self.revision_id = revision_id
        self.client = client
        super().__init__(f"Invalid revision ID {revision_id!r}")

    def __eq__(self, other):
        if not isinstance(other, InvalidRevisionId):
            return False
        return self.revision_id == other.revision_id and self.client == other.client


class NoCommonAncestor(Error):
    """No common ancestor found between revisions."""

    def __init__(self, revision_a, revision_b):
        """Initialize NoCommonAncestor exception.

        Args:
            revision_a: First revision ID.
            revision_b: Second revision ID.
        """
        self.revision_a = revision_a
        self.revision_b = revision_b
        super().__init__(
            f"No common ancestor found between {revision_a!r} and {revision_b!r}"
        )

    def __eq__(self, other):
        if not isinstance(other, NoCommonAncestor):
            return False
        return (
            self.revision_a == other.revision_a and self.revision_b == other.revision_b
        )


class RevisionNotPresent(Error):
    """Revision not present in the graph."""

    def __init__(self, revision_id, graph):
        """Initialize RevisionNotPresent exception.

        Args:
            revision_id: The revision ID not present in the graph.
            graph: The graph object where the revision was not found.
        """
        self.revision_id = revision_id
        self.graph = graph
        super().__init__(f"Revision {revision_id!r} not present in graph")

    def __eq__(self, other):
        if not isinstance(other, RevisionNotPresent):
            return False
        return self.revision_id == other.revision_id and self.graph == other.graph


class GraphCycleError(Error):
    """Cycle detected in graph.

    Raised when a cycle is detected in a directed graph that should be acyclic.
    """

    def __init__(self, graph):
        """Initialize with the graph containing the cycle.

        Args:
            graph: The graph object that contains a cycle.
        """
        self.graph = graph
        super().__init__(f"Cycle in graph {graph!r}")

    def __eq__(self, other):
        if not isinstance(other, GraphCycleError):
            return False
        return self.graph == other.graph
