# vcsgraph

A Python library providing graph algorithms optimized for version control systems.

## Overview

`vcsgraph` is a high-performance graph algorithms library specifically designed for working with version control system (VCS) data structures. It provides efficient implementations of common graph operations needed by VCS tools, with both pure Python and Rust-accelerated implementations for performance-critical operations.

## Features

- **Topological Sorting**: Multiple algorithms for sorting commits/revisions in topological order
- **Graph Traversal**: Efficient algorithms for traversing revision graphs and finding common ancestors
- **Multi-parent Support**: Handle complex merge scenarios with multiple parent revisions
- **Known Graph Operations**: Optimized operations on graphs where the full structure is known in advance
- **Rust Acceleration**: Performance-critical algorithms implemented in Rust with Python bindings via PyO3

## Installation

```bash
pip install vcsgraph
```

## Key Components

### Graph Operations

The `Graph` class provides fundamental graph operations:
- Finding least common ancestors (LCA)
- Finding unique ancestors
- Computing differences between revision sets
- Finding merge bases between branches

### Topological Sorting

Multiple sorting implementations optimized for different use cases:
- `topo_sort()`: Fast sorting when the complete result is needed
- `TopoSorter`: Iterator-based sorting for processing partial results
- `MergeSorter`: Specialized sorting that preserves merge history

### Multi-parent Diffs

The `MultiParent` class handles complex diff scenarios with multiple parent revisions, essential for three-way merges and conflict resolution.

### Known Graph

The `KnownGraph` class provides optimized operations when the complete graph structure is known, enabling faster ancestor calculations and traversals.

## Usage Examples

### Basic Topological Sort

```python
from vcsgraph import topo_sort

# Define a graph as a list of (node, parents) tuples
graph = [
    (b'rev1', []),
    (b'rev2', [b'rev1']),
    (b'rev3', [b'rev1']),
    (b'rev4', [b'rev2', b'rev3']),
]

# Sort nodes topologically (parents before children)
sorted_nodes = topo_sort(graph)
```

### Using Graph for Ancestry Operations

```python
from vcsgraph import Graph, DictParentsProvider

# Create a parents provider from a dictionary
ancestry = {
    b'rev1': (b'null:',),
    b'rev2a': (b'rev1',),
    b'rev2b': (b'rev1',),
    b'rev3': (b'rev2a',),
    b'rev4': (b'rev3', b'rev2b'),
}
parents_provider = DictParentsProvider(ancestry)

# Create a graph and find merge bases
graph = Graph(parents_provider)
merge_base = graph.find_merge_base(b'rev2a', b'rev2b')
```

### Working with Known Graphs

```python
from vcsgraph import KnownGraph

# Create a known graph from parent relationships
parent_map = {
    b'rev1': (b'null:',),
    b'rev2': (b'rev1',),
    b'rev3': (b'rev2',),
}
kg = KnownGraph(parent_map)

# Get heads (revisions with no children)
heads = kg.heads([b'rev1', b'rev2', b'rev3'])
```

## Performance

The library uses Rust for performance-critical operations while maintaining a Python interface. Key optimizations include:
- Memory-efficient graph representations
- Optimized ancestor searching algorithms
- Lazy evaluation where possible
- Caching of frequently accessed data

## License

This project is licensed under the GNU General Public License v2 or later. See the COPYING.txt file for details.

## Origins

This library was originally part of the Breezy version control system and has been extracted as a standalone package for use in other VCS-related projects.