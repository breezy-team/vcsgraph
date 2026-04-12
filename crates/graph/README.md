# vcs-graph

Graph algorithms for version control systems.

`vcs-graph` provides building blocks used by version control tools: topological
sorting (including merge-aware sorting that preserves branch structure),
parent-map manipulation, least common ancestor queries, and related
operations. It is the Rust core behind the [`vcsgraph`][vcsgraph-py] Python
package, originally extracted from the Breezy version control system.

## Features

- `TopoSorter` — iterative topological sort of a parent graph.
- `MergeSorter` — merge-aware topological sort that assigns revision numbers
  and tracks merge depth, suitable for rendering commit history.
- `ParentMap` / `ChildMap` — parent/child map types with utilities like
  `invert_parent_map` and `collapse_linear_regions`.
- `ParentsProvider` trait with `DictParentsProvider` and
  `StackedParentsProvider` implementations.
- Optional `pyo3` feature for exposing these types to Python.

## Example

```rust
use std::collections::HashMap;
use vcs_graph::tsort::TopoSorter;

let graph: HashMap<&str, Vec<&str>> = HashMap::from([
    ("A", vec![]),
    ("B", vec!["A"]),
    ("C", vec!["A"]),
    ("D", vec!["B", "C"]),
]);

let sorted = TopoSorter::new(graph.into_iter()).sorted().unwrap();
// Parents always come before children.
```

## License

GPL-2.0-or-later. See `COPYING.txt` in the repository root.

[vcsgraph-py]: https://pypi.org/project/vcsgraph/
