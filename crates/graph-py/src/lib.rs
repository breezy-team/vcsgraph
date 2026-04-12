#![allow(non_snake_case)]

use vcs_graph::bfs::BfsState;
use vcs_graph::graph::{Graph as RsGraph, GraphError};
use vcs_graph::known_graph::KnownGraph as RsKnownGraph;
use vcs_graph::{ChildMap, ParentMap, Parents, ParentsProvider, RevnoVec};

use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::wrap_pyfunction;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

import_exception!(vcsgraph.errors, GraphCycleError);

struct PyNode(Py<PyAny>);

impl std::fmt::Debug for PyNode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        Python::attach(|py| {
            let repr = self.0.bind(py).repr();
            if PyErr::occurred(py) {
                return Err(std::fmt::Error);
            }
            if let Ok(repr) = repr {
                write!(f, "{}", repr)
            } else {
                write!(f, "???")
            }
        })
    }
}

impl std::fmt::Display for PyNode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        Python::attach(|py| {
            let repr = self.0.bind(py).repr();
            if PyErr::occurred(py) {
                return Err(std::fmt::Error);
            }
            if let Ok(repr) = repr {
                write!(f, "{}", repr)
            } else {
                write!(f, "???")
            }
        })
    }
}

impl Clone for PyNode {
    fn clone(&self) -> PyNode {
        Python::attach(|py| PyNode(self.0.clone_ref(py)))
    }
}

impl From<Py<PyAny>> for PyNode {
    fn from(obj: Py<PyAny>) -> PyNode {
        PyNode(obj)
    }
}

impl<'a> From<Bound<'a, PyAny>> for PyNode {
    fn from(obj: Bound<'a, PyAny>) -> PyNode {
        PyNode(obj.unbind())
    }
}

impl<'py> FromPyObject<'_, 'py> for PyNode {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(PyNode(obj.to_owned().unbind()))
    }
}

impl<'py> IntoPyObject<'py> for PyNode {
    type Target = PyAny;

    type Output = Bound<'py, Self::Target>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0.clone_ref(py).into_bound(py))
    }
}

impl Hash for PyNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Python::attach(|py| match self.0.bind(py).hash() {
            Err(err) => err.restore(py),
            Ok(hash) => state.write_isize(hash),
        });
    }
}

impl PartialEq for PyNode {
    fn eq(&self, other: &PyNode) -> bool {
        Python::attach(|py| match self.0.bind(py).eq(other.0.bind(py)) {
            Err(err) => {
                err.restore(py);
                false
            }
            Ok(b) => b,
        })
    }
}

impl std::cmp::Eq for PyNode {}

impl PartialOrd for PyNode {
    fn partial_cmp(&self, other: &PyNode) -> Option<std::cmp::Ordering> {
        self.cmp(other).into()
    }
}

impl Ord for PyNode {
    fn cmp(&self, other: &PyNode) -> std::cmp::Ordering {
        Python::attach(|py| match self.0.bind(py).lt(other.0.bind(py)) {
            Err(err) => {
                err.restore(py);
                std::cmp::Ordering::Equal
            }
            Ok(b) => {
                if b {
                    std::cmp::Ordering::Less
                } else {
                    match self.0.bind(py).gt(other.0.bind(py)) {
                        Err(err) => {
                            err.restore(py);
                            std::cmp::Ordering::Equal
                        }
                        Ok(b) => {
                            if b {
                                std::cmp::Ordering::Greater
                            } else {
                                std::cmp::Ordering::Equal
                            }
                        }
                    }
                }
            }
        })
    }
}

/// Given a map from child => parents, create a map of parent => children
#[pyfunction]
fn invert_parent_map(parent_map: ParentMap<PyNode>) -> ChildMap<PyNode> {
    vcs_graph::invert_parent_map::<PyNode>(&parent_map)
}

/// Collapse regions of the graph that are 'linear'.
///
/// For example::
///
///   A:[B], B:[C]
///
/// can be collapsed by removing B and getting::
///
///   A:[C]
///
/// :param parent_map: A dictionary mapping children to their parents
/// :return: Another dictionary with 'linear' chains collapsed
#[pyfunction]
fn collapse_linear_regions(parent_map: ParentMap<PyNode>) -> PyResult<ParentMap<PyNode>> {
    Ok(vcs_graph::collapse_linear_regions::<PyNode>(&parent_map))
}

#[pyclass]
struct PyParentsProvider {
    provider: Box<dyn vcs_graph::ParentsProvider<PyNode> + Send + Sync>,
}

#[pymethods]
impl PyParentsProvider {
    fn get_parent_map(&mut self, py: Python, keys: Py<PyAny>) -> PyResult<ParentMap<PyNode>> {
        let mut hash_key: HashSet<PyNode> = HashSet::new();
        for key in keys.bind(py).try_iter()? {
            hash_key.insert(key?.into());
        }
        Ok(self
            .provider
            .get_parent_map(&hash_key.into_iter().collect()))
    }
}

#[pyfunction]
fn DictParentsProvider(
    py: Python<'_>,
    parent_map: ParentMap<PyNode>,
) -> PyResult<Bound<'_, PyParentsProvider>> {
    let provider = PyParentsProvider {
        provider: Box::new(vcs_graph::DictParentsProvider::<PyNode>::new(parent_map)),
    };
    Bound::new(py, provider)
}

#[pyclass]
struct TopoSorter {
    sorter: vcs_graph::tsort::TopoSorter<PyNode>,
}

#[pymethods]
impl TopoSorter {
    #[new]
    fn new(py: Python, graph: Py<PyAny>) -> PyResult<TopoSorter> {
        let iter = if graph.bind(py).is_instance_of::<PyDict>() {
            graph
                .cast_bound::<PyDict>(py)?
                .call_method0("items")?
                .try_iter()?
        } else {
            graph.bind(py).try_iter()?
        };
        let graph = iter
            .map(|k| k?.extract::<(Py<PyAny>, Vec<Py<PyAny>>)>())
            .map(|k| k.map(|(k, vs)| (PyNode::from(k), vs.into_iter().map(PyNode::from).collect())))
            .collect::<PyResult<Vec<(PyNode, Vec<PyNode>)>>>()?;

        let sorter = vcs_graph::tsort::TopoSorter::<PyNode>::new(graph.into_iter());
        Ok(TopoSorter { sorter })
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        match self.sorter.next() {
            None => Ok(None),
            Some(Ok(node)) => Ok(Some(node.into_pyobject(py)?.unbind())),
            Some(Err(vcs_graph::Error::Cycle(e))) => Err(GraphCycleError::new_err(e)),
            Some(Err(e)) => panic!("Unexpected error: {:?}", e),
        }
    }

    fn __iter__(slf: PyRefMut<Self>) -> PyRefMut<Self> {
        slf
    }

    fn iter_topo_order(slf: PyRefMut<Self>) -> PyRefMut<Self> {
        slf
    }

    fn sorted(&mut self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let mut ret = Vec::new();
        while let Some(node) = self.__next__(py)? {
            ret.push(node);
        }
        Ok(ret)
    }
}

fn revno_vec_to_py(py: Python, revno: RevnoVec) -> Py<PyAny> {
    PyTuple::new(py, revno.into_iter().map(|v| v.into_pyobject(py).unwrap()))
        .unwrap()
        .into_any()
        .unbind()
}

#[pyclass]
struct MergeSorter {
    sorter: vcs_graph::tsort::MergeSorter<PyNode>,
}

fn branch_tip_is_null(py: Python, branch_tip: Py<PyAny>) -> bool {
    if let Ok(branch_tip) = branch_tip.extract::<&[u8]>(py) {
        branch_tip == b"null:"
    } else if let Ok((branch_tip,)) = branch_tip.extract::<(Vec<u8>,)>(py) {
        branch_tip.as_slice() == b"null:"
    } else {
        false
    }
}

#[pymethods]
impl MergeSorter {
    #[new]
    #[pyo3(signature = (graph, branch_tip=None, mainline_revisions=None, generate_revno=false))]
    fn new(
        py: Python,
        graph: Py<PyAny>,
        mut branch_tip: Option<Py<PyAny>>,
        mainline_revisions: Option<Py<PyAny>>,
        generate_revno: Option<bool>,
    ) -> PyResult<MergeSorter> {
        let iter = if graph.bind(py).is_instance_of::<PyDict>() {
            graph
                .cast_bound::<PyDict>(py)?
                .call_method0("items")?
                .try_iter()?
        } else {
            graph.bind(py).try_iter()?
        };
        let graph = iter
            .map(|k| k?.extract::<(Py<PyAny>, Vec<Py<PyAny>>)>())
            .map(|k| k.map(|(k, vs)| (PyNode::from(k), vs.into_iter().map(PyNode::from).collect())))
            .collect::<PyResult<HashMap<PyNode, Vec<PyNode>>>>()?;

        let mainline_revisions = if let Some(mainline_revisions) = mainline_revisions {
            let mainline_revisions = mainline_revisions
                .bind(py)
                .try_iter()?
                .map(|k| {
                    let item = k?;
                    Ok(item.extract::<Py<PyAny>>()?)
                })
                .collect::<PyResult<Vec<Py<PyAny>>>>()?;
            Some(mainline_revisions.into_iter().map(PyNode::from).collect())
        } else {
            None
        };

        // The null: revision doesn't exist in the graph, so don't attempt to remove it
        if let Some(ref mut tip_obj) = branch_tip {
            if branch_tip_is_null(py, tip_obj.clone_ref(py)) {
                branch_tip = None;
            }
        }

        let sorter = vcs_graph::tsort::MergeSorter::<PyNode>::new(
            graph,
            branch_tip.map(PyNode::from),
            mainline_revisions,
            generate_revno.unwrap_or(false),
        );
        Ok(MergeSorter { sorter })
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        match self.sorter.next() {
            None => Ok(None),
            Some(Ok((sequence_number, node, merge_depth, None, end_of_merge))) => Ok(Some(
                (
                    sequence_number,
                    node.into_pyobject(py)?.unbind(),
                    merge_depth,
                    end_of_merge,
                )
                    .into_pyobject(py)?
                    .unbind()
                    .into(),
            )),

            Some(Ok((sequence_number, node, merge_depth, Some(revno), end_of_merge))) => Ok(Some(
                (
                    sequence_number,
                    node.into_pyobject(py)?.unbind(),
                    merge_depth,
                    revno_vec_to_py(py, revno),
                    end_of_merge,
                )
                    .into_pyobject(py)?
                    .unbind()
                    .into(),
            )),
            Some(Err(vcs_graph::Error::Cycle(e))) => Err(GraphCycleError::new_err(e)),
            Some(Err(e)) => panic!("Unexpected error: {:?}", e),
        }
    }

    fn __iter__(slf: PyRefMut<Self>) -> PyRefMut<Self> {
        slf
    }

    fn iter_topo_order(slf: PyRefMut<Self>) -> PyRefMut<Self> {
        slf
    }

    fn sorted<'a>(&mut self, py: Python<'a>) -> PyResult<Bound<'a, PyList>> {
        let ret = PyList::empty(py);
        loop {
            let item = self.__next__(py)?;
            if let Some(item) = item {
                ret.append(item)?;
            } else {
                break;
            }
        }
        Ok(ret)
    }
}

/// Topological sort a graph which groups merges.
///
/// :param graph: sequence of pairs of node->parents_list.
/// :param branch_tip: the tip of the branch to graph. Revisions not
///                    reachable from branch_tip are not included in the
///                    output.
/// :param mainline_revisions: If not None this forces a mainline to be
///                            used rather than synthesised from the graph.
///                            This must be a valid path through some part
///                            of the graph. If the mainline does not cover all
///                            the revisions, output stops at the start of the
///                            old revision listed in the mainline revisions
///                            list.
///                            The order for this parameter is oldest-first.
/// :param generate_revno: Optional parameter controlling the generation of
///     revision number sequences in the output. See the output description of
///     the MergeSorter docstring for details.
/// :result: See the MergeSorter docstring for details.
///
/// Node identifiers can be any hashable object, and are typically strings.
#[pyfunction]
#[pyo3(signature = (graph, branch_tip=None, mainline_revisions=None, generate_revno=false))]
fn merge_sort(
    py: Python,
    graph: Py<PyAny>,
    branch_tip: Option<Py<PyAny>>,
    mainline_revisions: Option<Py<PyAny>>,
    generate_revno: Option<bool>,
) -> PyResult<Bound<PyList>> {
    let mut sorter = MergeSorter::new(py, graph, branch_tip, mainline_revisions, generate_revno)?;
    sorter.sorted(py)
}

const NULL_REVISION: &[u8] = b"null:";

fn is_null_revision(py: Python, obj: &Py<PyAny>) -> bool {
    if let Ok(b) = obj.extract::<&[u8]>(py) {
        b == NULL_REVISION
    } else {
        false
    }
}

#[pyclass(name = "KnownGraph")]
struct PyKnownGraph {
    inner: RsKnownGraph<PyNode>,
}

fn extract_parent_map(py: Python, parent_map: Py<PyAny>) -> PyResult<Vec<(PyNode, Vec<PyNode>)>> {
    let iter = if parent_map.bind(py).is_instance_of::<PyDict>() {
        parent_map
            .cast_bound::<PyDict>(py)?
            .call_method0("items")?
            .try_iter()?
    } else {
        parent_map.bind(py).try_iter()?
    };
    iter.map(|k| k?.extract::<(Py<PyAny>, Vec<Py<PyAny>>)>())
        .map(|k| k.map(|(k, vs)| (PyNode::from(k), vs.into_iter().map(PyNode::from).collect())))
        .collect()
}

#[pymethods]
impl PyKnownGraph {
    #[new]
    #[pyo3(signature = (parent_map, do_cache=true))]
    fn new(py: Python, parent_map: Py<PyAny>, do_cache: Option<bool>) -> PyResult<Self> {
        let pm = extract_parent_map(py, parent_map)?;
        Ok(PyKnownGraph {
            inner: RsKnownGraph::new(pm, do_cache.unwrap_or(true)),
        })
    }

    /// Add a new node to the graph. If `key` was a ghost, it is filled in.
    fn add_node(
        &mut self,
        py: Python,
        key: Py<PyAny>,
        parent_keys: Vec<Py<PyAny>>,
    ) -> PyResult<()> {
        let key = PyNode::from(key);
        let parents: Vec<PyNode> = parent_keys.into_iter().map(PyNode::from).collect();
        self.inner
            .add_node(key.clone(), parents)
            .map_err(|e| match e {
                vcs_graph::Error::ParentMismatch {
                    key,
                    expected,
                    actual,
                } => {
                    let key_repr = format!("{:?}", key);
                    let expected_py: Vec<Py<PyAny>> =
                        expected.into_iter().map(|n| n.0.clone_ref(py)).collect();
                    let actual_py: Vec<Py<PyAny>> =
                        actual.into_iter().map(|n| n.0.clone_ref(py)).collect();
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Parent key mismatch, existing node {} has parents of {:?} not {:?}",
                        key_repr, expected_py, actual_py
                    ))
                }
                other => pyo3::exceptions::PyValueError::new_err(format!("{:?}", other)),
            })
    }

    /// Return the parent keys for `key`. Returns `None` for ghosts; raises
    /// `KeyError` if `key` is not in the graph.
    fn get_parent_keys(&self, py: Python, key: Py<PyAny>) -> PyResult<Option<Vec<Py<PyAny>>>> {
        let node = PyNode::from(key);
        if !self.inner.contains(&node) {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!("{:?}", node)));
        }
        Ok(self
            .inner
            .get_parent_keys(&node)
            .map(|ps| ps.iter().map(|n| n.0.clone_ref(py)).collect()))
    }

    /// Return the child keys for `key`. Raises `KeyError` if `key` is not in
    /// the graph.
    fn get_child_keys(&self, py: Python, key: Py<PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        let node = PyNode::from(key);
        match self.inner.get_child_keys(&node) {
            Some(cs) => Ok(cs.iter().map(|n| n.0.clone_ref(py)).collect()),
            None => Err(pyo3::exceptions::PyKeyError::new_err(format!("{:?}", node))),
        }
    }

    /// Return the heads from amongst `keys`.
    fn heads<'py>(&mut self, py: Python<'py>, keys: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let mut candidates: Vec<Py<PyAny>> = Vec::new();
        let mut had_null = false;
        for k in keys.bind(py).try_iter()? {
            let item: Py<PyAny> = k?.extract()?;
            if is_null_revision(py, &item) {
                had_null = true;
            } else {
                candidates.push(item);
            }
        }
        if candidates.is_empty() && had_null {
            // NULL_REVISION is only a head if it's the only entry.
            let fs = py.import("builtins")?.getattr("frozenset")?;
            let null = pyo3::types::PyBytes::new(py, NULL_REVISION);
            return fs.call1((vec![null.into_any()],));
        }
        let nodes: Vec<PyNode> = candidates.into_iter().map(PyNode::from).collect();
        let heads = self.inner.heads(nodes);
        let py_set: Vec<Py<PyAny>> = heads.into_iter().map(|n| n.0).collect();
        let fs = py.import("builtins")?.getattr("frozenset")?;
        fs.call1((py_set,))
    }

    /// Return the nodes in topological order (parents first).
    fn topo_sort(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        match self.inner.topo_sort() {
            Ok(v) => Ok(v.into_iter().map(|n| n.0.clone_ref(py)).collect()),
            Err(vcs_graph::Error::Cycle(_)) => {
                Err(GraphCycleError::new_err(("cycle in known graph",)))
            }
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("{:?}", e))),
        }
    }

    /// Return a reverse topological ordering grouped by prefix.
    ///
    /// Mirrors the Python implementation: keys that are bytes use their first
    /// byte as the prefix bucket; single-element keys use an empty prefix.
    fn gc_sort(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let prefix_of = |k: &PyNode| -> Vec<u8> {
            Python::attach(|py| {
                let bound = k.0.bind(py);
                if let Ok(b) = bound.extract::<&[u8]>() {
                    if b.len() == 1 {
                        Vec::new()
                    } else {
                        vec![b[0]]
                    }
                } else if let Ok(s) = bound.extract::<&str>() {
                    if s.len() == 1 {
                        Vec::new()
                    } else {
                        vec![s.as_bytes()[0]]
                    }
                } else if let Ok(first) = bound.get_item(0) {
                    if let Ok(b) = first.extract::<&[u8]>() {
                        b.to_vec()
                    } else if let Ok(s) = first.extract::<&str>() {
                        s.as_bytes().to_vec()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            })
        };
        let v = self.inner.gc_sort(prefix_of);
        Ok(v.into_iter().map(|n| n.0.clone_ref(py)).collect())
    }

    /// Compute the merge-sorted graph output starting at `tip_key`.
    ///
    /// If `tip_key` is `None`, `b"null:"`, or `(b"null:",)`, returns an empty
    /// list (matches the Python null-tip semantics).
    fn merge_sort(
        &self,
        py: Python,
        tip_key: Py<PyAny>,
    ) -> PyResult<Vec<Py<PyKnownGraphMergeSortNode>>> {
        if tip_key.is_none(py) || branch_tip_is_null(py, tip_key.clone_ref(py)) {
            return Ok(Vec::new());
        }
        let tip = PyNode::from(tip_key);
        if !self.inner.contains(&tip) {
            return Ok(Vec::new());
        }
        let result = self.inner.merge_sort(tip).map_err(|e| match e {
            vcs_graph::Error::Cycle(_) => GraphCycleError::new_err(("cycle in known graph",)),
            other => pyo3::exceptions::PyValueError::new_err(format!("{:?}", other)),
        })?;
        result
            .into_iter()
            .map(|n| {
                let revno = revno_vec_to_py(py, n.revno);
                Py::new(
                    py,
                    PyKnownGraphMergeSortNode {
                        key: n.key.0,
                        merge_depth: n.merge_depth,
                        revno,
                        end_of_merge: n.end_of_merge,
                    },
                )
            })
            .collect()
    }

    /// Return a mapping-like view of all nodes in the graph, keyed by node
    /// key. Each value is a `_KnownGraphNode` exposing live `key`,
    /// `parent_keys`, `child_keys`, and `gdfo` attributes.
    #[getter]
    fn _nodes(slf: Py<Self>) -> PyKnownGraphNodesView {
        PyKnownGraphNodesView { graph: slf }
    }
}

#[pyclass(name = "_KnownGraphNodesView")]
struct PyKnownGraphNodesView {
    graph: Py<PyKnownGraph>,
}

#[pymethods]
impl PyKnownGraphNodesView {
    fn __getitem__(&self, py: Python, key: Py<PyAny>) -> PyResult<PyKnownGraphNode> {
        let g = self.graph.borrow(py);
        let node = PyNode::from(key.clone_ref(py));
        if !g.inner.contains(&node) {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!("{:?}", node)));
        }
        Ok(PyKnownGraphNode {
            graph: self.graph.clone_ref(py),
            key,
        })
    }

    fn __contains__(&self, py: Python, key: Py<PyAny>) -> bool {
        self.graph.borrow(py).inner.contains(&PyNode::from(key))
    }

    fn __len__(&self, py: Python) -> usize {
        self.graph.borrow(py).inner.len()
    }

    fn __iter__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let keys: Vec<Py<PyAny>> = self
            .graph
            .borrow(py)
            .inner
            .keys()
            .map(|n| n.0.clone_ref(py))
            .collect();
        Ok(pyo3::types::PyList::new(py, keys)?
            .call_method0("__iter__")?
            .unbind())
    }

    fn keys(&self, py: Python) -> Vec<Py<PyAny>> {
        self.graph
            .borrow(py)
            .inner
            .keys()
            .map(|n| n.0.clone_ref(py))
            .collect()
    }

    fn values(&self, py: Python) -> Vec<PyKnownGraphNode> {
        self.graph
            .borrow(py)
            .inner
            .keys()
            .map(|n| PyKnownGraphNode {
                graph: self.graph.clone_ref(py),
                key: n.0.clone_ref(py),
            })
            .collect()
    }
}

#[pyclass(name = "_KnownGraphNode")]
struct PyKnownGraphNode {
    graph: Py<PyKnownGraph>,
    key: Py<PyAny>,
}

#[pymethods]
impl PyKnownGraphNode {
    #[getter]
    fn key(&self, py: Python) -> Py<PyAny> {
        self.key.clone_ref(py)
    }

    #[getter]
    fn gdfo(&self, py: Python) -> Option<u64> {
        self.graph
            .borrow(py)
            .inner
            .gdfo(&PyNode::from(self.key.clone_ref(py)))
    }

    #[getter]
    fn parent_keys(&self, py: Python) -> Option<Vec<Py<PyAny>>> {
        self.graph
            .borrow(py)
            .inner
            .get_parent_keys(&PyNode::from(self.key.clone_ref(py)))
            .map(|ps| ps.iter().map(|n| n.0.clone_ref(py)).collect())
    }

    #[getter]
    fn child_keys(&self, py: Python) -> Vec<Py<PyAny>> {
        self.graph
            .borrow(py)
            .inner
            .get_child_keys(&PyNode::from(self.key.clone_ref(py)))
            .map(|cs| cs.iter().map(|n| n.0.clone_ref(py)).collect())
            .unwrap_or_default()
    }

    fn __repr__(&self, py: Python) -> String {
        format!(
            "_KnownGraphNode({:?}  gdfo:{:?} par:{:?} child:{:?})",
            self.key
                .bind(py)
                .repr()
                .map(|r| r.to_string())
                .unwrap_or_default(),
            self.gdfo(py),
            self.parent_keys(py),
            self.child_keys(py),
        )
    }
}

#[pyclass(name = "_MergeSortNode")]
struct PyKnownGraphMergeSortNode {
    #[pyo3(get, set)]
    key: Py<PyAny>,
    #[pyo3(get, set)]
    merge_depth: usize,
    #[pyo3(get, set)]
    revno: Py<PyAny>,
    #[pyo3(get, set)]
    end_of_merge: bool,
}

/// Adapter letting a Python parents provider satisfy Rust's `ParentsProvider`
/// trait. Holds a raw `Py<PyAny>` and dispatches `get_parent_map(keys)` via
/// the GIL.
///
/// The Python provider's `get_parent_map` must accept an iterable of keys
/// and return a dict-like `{key: parents_list}` (missing keys are treated
/// as ghosts). Any Python exception during the call is caught and converted
/// to an empty response — matching the Python Graph's behavior of treating
/// the provider as best-effort.
struct PyParentsProviderAdapter {
    provider: Py<PyAny>,
}

impl ParentsProvider<PyNode> for PyParentsProviderAdapter {
    fn get_parent_map(&self, keys: &HashSet<PyNode>) -> ParentMap<PyNode> {
        Python::attach(|py| {
            let key_list = pyo3::types::PyList::empty(py);
            for k in keys {
                if key_list.append(k.0.bind(py)).is_err() {
                    return ParentMap::new();
                }
            }
            let result = match self
                .provider
                .bind(py)
                .call_method1("get_parent_map", (key_list,))
            {
                Ok(r) => r,
                Err(err) => {
                    err.restore(py);
                    return ParentMap::new();
                }
            };
            result.extract::<ParentMap<PyNode>>().unwrap_or_default()
        })
    }
}

#[pyclass(name = "_RustGraph")]
struct PyGraph {
    inner: RsGraph<PyNode, PyParentsProviderAdapter>,
    provider_py: Py<PyAny>,
}

fn extract_iter_pynodes(py: Python, obj: &Py<PyAny>) -> PyResult<Vec<PyNode>> {
    let mut out = Vec::new();
    for item in obj.bind(py).try_iter()? {
        out.push(PyNode::from(item?));
    }
    Ok(out)
}

fn parent_map_to_pydict<'py>(
    py: Python<'py>,
    pm: ParentMap<PyNode>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in pm {
        match v {
            Parents::Known(ps) => {
                let list: Vec<Py<PyAny>> = ps.into_iter().map(|n| n.0).collect();
                d.set_item(k.0, list)?;
            }
            Parents::Ghost => {
                d.set_item(k.0, py.None())?;
            }
        }
    }
    Ok(d)
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new(py: Python, parents_provider: Py<PyAny>) -> PyResult<Self> {
        let adapter = PyParentsProviderAdapter {
            provider: parents_provider.clone_ref(py),
        };
        Ok(PyGraph {
            inner: RsGraph::new(adapter),
            provider_py: parents_provider,
        })
    }

    /// Return the wrapped parents provider as given at construction time.
    #[getter]
    fn parents_provider(&self, py: Python) -> Py<PyAny> {
        self.provider_py.clone_ref(py)
    }

    fn get_parent_map<'py>(
        &self,
        py: Python<'py>,
        keys: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let nodes = extract_iter_pynodes(py, &keys)?;
        let pm = self.inner.get_parent_map(nodes);
        parent_map_to_pydict(py, pm)
    }

    fn get_child_map<'py>(&self, py: Python<'py>, keys: Py<PyAny>) -> PyResult<Bound<'py, PyDict>> {
        let nodes = extract_iter_pynodes(py, &keys)?;
        let cm = self.inner.get_child_map(nodes);
        let d = PyDict::new(py);
        for (parent, children) in cm {
            let list: Vec<Py<PyAny>> = children.into_iter().map(|n| n.0).collect();
            d.set_item(parent.0, list)?;
        }
        Ok(d)
    }

    fn iter_topo_order(&self, py: Python, revisions: Py<PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        let nodes = extract_iter_pynodes(py, &revisions)?;
        self.inner
            .iter_topo_order(nodes)
            .map(|v| v.into_iter().map(|n| n.0).collect())
            .map_err(|e| match e {
                vcs_graph::Error::Cycle(_) => {
                    GraphCycleError::new_err(("cycle in graph while iter_topo_order",))
                }
                other => pyo3::exceptions::PyValueError::new_err(format!("{:?}", other)),
            })
    }

    /// Iterate the left-hand ancestry from `start_key` until a key in
    /// `stop_keys` is hit (or the origin is reached).
    #[pyo3(signature = (start_key, stop_keys=None))]
    fn iter_lefthand_ancestry(
        &self,
        py: Python,
        start_key: Py<PyAny>,
        stop_keys: Option<Py<PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let start = PyNode::from(start_key);
        let stop: Vec<PyNode> = match stop_keys {
            None => Vec::new(),
            Some(obj) => {
                let mut s = Vec::new();
                for item in obj.bind(py).try_iter()? {
                    s.push(PyNode::from(item?));
                }
                s
            }
        };
        self.inner
            .iter_lefthand_ancestry(start, stop)
            .map(|v| v.into_iter().map(|n| n.0).collect())
            .map_err(graph_error_to_py)
    }

    /// Iterate ancestry, yielding `(key, parents_list_or_None)` pairs.
    fn iter_ancestry<'py>(
        &self,
        py: Python<'py>,
        revision_ids: Py<PyAny>,
    ) -> PyResult<Vec<Bound<'py, pyo3::types::PyTuple>>> {
        let nodes = extract_iter_pynodes(py, &revision_ids)?;
        let pairs = self.inner.iter_ancestry(nodes);
        pairs
            .into_iter()
            .map(|(k, parents)| {
                let parents_obj: Py<PyAny> = match parents {
                    Parents::Known(ps) => {
                        let list: Vec<Py<PyAny>> = ps.into_iter().map(|n| n.0).collect();
                        pyo3::types::PyTuple::new(py, list)?.into_any().unbind()
                    }
                    Parents::Ghost => py.None(),
                };
                pyo3::types::PyTuple::new(py, [k.0, parents_obj])
            })
            .collect()
    }

    fn find_distance_to_null(
        &self,
        py: Python,
        target_revision_id: Py<PyAny>,
        known_revision_ids: Py<PyAny>,
    ) -> PyResult<i64> {
        let target = PyNode::from(target_revision_id);
        let mut known: Vec<(PyNode, i64)> = Vec::new();
        for item in known_revision_ids.bind(py).try_iter()? {
            let (k, d): (Py<PyAny>, i64) = item?.extract()?;
            known.push((PyNode::from(k), d));
        }
        let null_bytes = pyo3::types::PyBytes::new(py, NULL_REVISION);
        let null = PyNode::from(null_bytes.into_any().unbind());
        self.inner
            .find_distance_to_null(target, known, null)
            .map_err(graph_error_to_py)
    }

    fn find_lefthand_distances<'py>(
        &self,
        py: Python<'py>,
        keys: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let nodes = extract_iter_pynodes(py, &keys)?;
        let null_bytes = pyo3::types::PyBytes::new(py, NULL_REVISION);
        let null = PyNode::from(null_bytes.into_any().unbind());
        let result = self.inner.find_lefthand_distances(nodes, null);
        let d = PyDict::new(py);
        for (k, dist) in result {
            d.set_item(k.0, dist)?;
        }
        Ok(d)
    }

    /// Return the heads from amongst keys.
    ///
    /// This is done by searching the ancestries of each key.  Any key that is
    /// reachable from another key is not returned; all the others are.
    ///
    /// This operation scales with the relative depth between any two keys. If
    /// any two keys are completely disconnected all ancestry of both sides
    /// will be retrieved.
    ///
    /// :param keys: An iterable of keys.
    /// :return: A set of the heads. Note that as a set there is no ordering
    ///     information. Callers will need to filter their input to create
    ///     order if they need it.
    fn heads<'py>(
        &self,
        py: Python<'py>,
        keys: Py<PyAny>,
    ) -> PyResult<Bound<'py, pyo3::types::PySet>> {
        let nodes = extract_iter_pynodes(py, &keys)?;
        let null_bytes = pyo3::types::PyBytes::new(py, NULL_REVISION);
        let null = PyNode::from(null_bytes.into_any().unbind());
        let result = self.inner.heads_with_null(nodes, &null);
        pynodes_to_pyset(py, result)
    }

    /// Determine the lowest common ancestors of the provided revisions.
    ///
    /// A lowest common ancestor is a common ancestor none of whose
    /// descendants are common ancestors.  In graphs, unlike trees, there may
    /// be multiple lowest common ancestors.
    ///
    /// This algorithm has two phases.  Phase 1 identifies border ancestors,
    /// and phase 2 filters border ancestors to determine lowest common
    /// ancestors.
    ///
    /// In phase 1, border ancestors are identified, using a breadth-first
    /// search starting at the bottom of the graph.  Searches are stopped
    /// whenever a node or one of its descendants is determined to be common
    ///
    /// In phase 2, the border ancestors are filtered to find the least
    /// common ancestors.  This is done by searching the ancestries of each
    /// border ancestor.
    ///
    /// Phase 2 is perfomed on the principle that a border ancestor that is
    /// not an ancestor of any other border ancestor is a least common
    /// ancestor.
    ///
    /// Searches are stopped when they find a node that is determined to be a
    /// common ancestor of all border ancestors, because this shows that it
    /// cannot be a descendant of any border ancestor.
    ///
    /// The scaling of this operation should be proportional to:
    ///
    /// 1. The number of uncommon ancestors
    /// 2. The number of border ancestors
    /// 3. The length of the shortest path between a border ancestor and an
    ///    ancestor of all border ancestors.
    fn find_lca<'py>(
        &self,
        py: Python<'py>,
        revisions: Py<PyAny>,
    ) -> PyResult<Bound<'py, pyo3::types::PySet>> {
        let nodes = extract_iter_pynodes(py, &revisions)?;
        // Match Python's `_find_border_ancestors` precondition: None is
        // not a valid revision id. Raise InvalidRevisionId before running
        // the algorithm so callers see the same error they used to.
        for n in &nodes {
            if n.0.is_none(py) {
                return Err(InvalidRevisionId::new_err((
                    py.None(),
                    self.provider_py.clone_ref(py),
                )));
            }
        }
        let null_bytes = pyo3::types::PyBytes::new(py, NULL_REVISION);
        let null = PyNode::from(null_bytes.into_any().unbind());
        let result = self.inner.find_lca(nodes, &null);
        pynodes_to_pyset(py, result)
    }

    /// Determine whether a revision is an ancestor of another.
    ///
    /// We answer this using heads() as heads() has the logic to perform the
    /// smallest number of parent lookups to determine the ancestral
    /// relationship between N revisions.
    fn is_ancestor(
        &self,
        py: Python,
        candidate_ancestor: Py<PyAny>,
        candidate_descendant: Py<PyAny>,
    ) -> bool {
        let null_bytes = pyo3::types::PyBytes::new(py, NULL_REVISION);
        let null = PyNode::from(null_bytes.into_any().unbind());
        self.inner.is_ancestor(
            PyNode::from(candidate_ancestor),
            PyNode::from(candidate_descendant),
            &null,
        )
    }

    /// Determine whether a revision is between two others.
    ///
    /// returns true if and only if:
    /// lower_bound_revid <= revid <= upper_bound_revid
    #[pyo3(signature = (revid, lower_bound_revid, upper_bound_revid))]
    fn is_between(
        &self,
        py: Python,
        revid: Py<PyAny>,
        lower_bound_revid: Option<Py<PyAny>>,
        upper_bound_revid: Option<Py<PyAny>>,
    ) -> bool {
        let null_bytes = pyo3::types::PyBytes::new(py, NULL_REVISION);
        let null = PyNode::from(null_bytes.into_any().unbind());
        self.inner.is_between(
            PyNode::from(revid),
            lower_bound_revid.map(PyNode::from),
            upper_bound_revid.map(PyNode::from),
            &null,
        )
    }

    /// Find the order that each revision was merged into tip.
    ///
    /// This basically just walks backwards with a stack, and walks left-first
    /// until it finds a node to stop.
    fn find_merge_order(
        &self,
        py: Python,
        tip_revision_id: Py<PyAny>,
        lca_revision_ids: Py<PyAny>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let tip = PyNode::from(tip_revision_id);
        let lcas = extract_iter_pynodes(py, &lca_revision_ids)?;
        let result = self.inner.find_merge_order(tip, lcas);
        Ok(result.into_iter().map(|n| n.0).collect())
    }

    /// Find descendants of `old_key` that are ancestors of `new_key`.
    fn find_descendants<'py>(
        &self,
        py: Python<'py>,
        old_key: Py<PyAny>,
        new_key: Py<PyAny>,
    ) -> PyResult<Bound<'py, pyo3::types::PySet>> {
        let old = PyNode::from(old_key);
        let new = PyNode::from(new_key);
        let result = self.inner.find_descendants(old, new);
        pynodes_to_pyset(py, result)
    }

    /// Find ancestors of `new_key` that may be descendants of `old_key`.
    fn _find_descendant_ancestors<'py>(
        &self,
        py: Python<'py>,
        old_key: Py<PyAny>,
        new_key: Py<PyAny>,
    ) -> PyResult<Bound<'py, pyo3::types::PySet>> {
        let old = PyNode::from(old_key);
        let new = PyNode::from(new_key);
        let result = self.inner.find_descendant_ancestors(old, new);
        pynodes_to_pyset(py, result)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let r = self.provider_py.bind(py).repr()?;
        Ok(format!("Graph({})", r))
    }
}

import_exception!(vcsgraph.errors, GhostRevisionsHaveNoRevno);
import_exception!(vcsgraph.errors, InvalidRevisionId);
import_exception!(vcsgraph.errors, RevisionNotPresent);

fn graph_error_to_py(e: GraphError<PyNode>) -> PyErr {
    match e {
        GraphError::GhostRevision { target, ghost } => Python::attach(|py| {
            GhostRevisionsHaveNoRevno::new_err((target.0.clone_ref(py), ghost.0.clone_ref(py)))
        }),
        GraphError::RevisionNotPresent(key) => {
            Python::attach(|py| RevisionNotPresent::new_err((key.0.clone_ref(py),)))
        }
        GraphError::Cycle(_) => GraphCycleError::new_err(("cycle in graph",)),
    }
}

/// Helper: convert a set of PyNodes into a Python `set()`.
fn pynodes_to_pyset<'py>(
    py: Python<'py>,
    set: rustc_hash::FxHashSet<PyNode>,
) -> PyResult<Bound<'py, pyo3::types::PySet>> {
    let items: Vec<Py<PyAny>> = set.into_iter().map(|n| n.0).collect();
    pyo3::types::PySet::new(py, items)
}

/// Python binding for [`BfsState`]. Owns its own adapter over a Python
/// parents provider and holds the Rust BFS state as sibling fields.
///
/// `dict=true` gives the class a `__dict__`, which lets the existing Python
/// callers in `graph.py` stash a `_label` attribute on the instance for
/// debug logging. This can be removed once Phase 4 (heads / border
/// ancestors) no longer relies on it.
#[pyclass(name = "_BreadthFirstSearcher", dict)]
struct PyBFSearcher {
    state: BfsState<PyNode>,
    adapter: PyParentsProviderAdapter,
}

#[pymethods]
impl PyBFSearcher {
    #[new]
    fn new(py: Python, revisions: Py<PyAny>, parents_provider: Py<PyAny>) -> PyResult<Self> {
        let mut revs: Vec<PyNode> = Vec::new();
        for item in revisions.bind(py).try_iter()? {
            revs.push(PyNode::from(item?));
        }
        Ok(PyBFSearcher {
            state: BfsState::new(revs),
            adapter: PyParentsProviderAdapter {
                provider: parents_provider,
            },
        })
    }

    /// `seen` attribute — matches the Python API. Returns a set snapshot.
    #[getter]
    fn seen<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PySet>> {
        pynodes_to_pyset(py, self.state.seen.clone())
    }

    /// `_next_query` attribute — the current search frontier.
    ///
    /// Read-only snapshot. Existing callers in `graph.py` iterate and
    /// truth-test this attribute but do not mutate it.
    #[getter]
    fn _next_query<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PySet>> {
        pynodes_to_pyset(py, self.state.next_query().clone())
    }

    /// `_iterations` attribute — number of advance steps performed.
    #[getter]
    fn _iterations(&self) -> usize {
        self.state.iterations()
    }

    /// Python `step` returns the next query set, or `()` on StopIteration.
    fn step<'py>(&mut self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        match self.state.next_set(&self.adapter) {
            Some(set) => Ok(pynodes_to_pyset(py, set)?.into_any().unbind()),
            None => Ok(pyo3::types::PyTuple::empty(py).into_any().unbind()),
        }
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PySet>> {
        match self.state.next_set(&self.adapter) {
            Some(set) => pynodes_to_pyset(py, set),
            None => Err(pyo3::exceptions::PyStopIteration::new_err(())),
        }
    }

    fn next<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PySet>> {
        self.__next__(py)
    }

    fn next_with_ghosts<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, pyo3::types::PySet>,
        Bound<'py, pyo3::types::PySet>,
    )> {
        match self.state.next_with_ghosts(&self.adapter) {
            Some((present, ghosts)) => Ok((
                pynodes_to_pyset(py, present)?,
                pynodes_to_pyset(py, ghosts)?,
            )),
            None => Err(pyo3::exceptions::PyStopIteration::new_err(())),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn get_state<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, pyo3::types::PySet>,
        Bound<'py, pyo3::types::PySet>,
        Bound<'py, pyo3::types::PySet>,
    )> {
        let (started, excludes, included) = self.state.get_state(&self.adapter);
        Ok((
            pynodes_to_pyset(py, started)?,
            pynodes_to_pyset(py, excludes)?,
            pynodes_to_pyset(py, included)?,
        ))
    }

    fn find_seen_ancestors<'py>(
        &self,
        py: Python<'py>,
        revisions: Py<PyAny>,
    ) -> PyResult<Bound<'py, pyo3::types::PySet>> {
        let mut revs: Vec<PyNode> = Vec::new();
        for item in revisions.bind(py).try_iter()? {
            revs.push(PyNode::from(item?));
        }
        let result = self.state.find_seen_ancestors(revs, &self.adapter);
        pynodes_to_pyset(py, result)
    }

    fn stop_searching_any<'py>(
        &mut self,
        py: Python<'py>,
        revisions: Py<PyAny>,
    ) -> PyResult<Bound<'py, pyo3::types::PySet>> {
        let mut revs: Vec<PyNode> = Vec::new();
        for item in revisions.bind(py).try_iter()? {
            revs.push(PyNode::from(item?));
        }
        let stopped = self.state.stop_searching_any(revs);
        pynodes_to_pyset(py, stopped)
    }

    fn start_searching(&mut self, py: Python, revisions: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let mut revs: Vec<PyNode> = Vec::new();
        for item in revisions.bind(py).try_iter()? {
            revs.push(PyNode::from(item?));
        }
        match self.state.start_searching(revs, &self.adapter) {
            Some((present, ghosts)) => {
                let pres = pynodes_to_pyset(py, present)?;
                let gh = pynodes_to_pyset(py, ghosts)?;
                Ok(
                    pyo3::types::PyTuple::new(py, [pres.into_any(), gh.into_any()])?
                        .into_any()
                        .unbind(),
                )
            }
            // In Next mode Python returns None (the function has no explicit
            // return), which matches our None here.
            None => Ok(py.None()),
        }
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let prefix = if self.state.iterations() > 0 {
            "searching"
        } else {
            "starting"
        };
        let seen_repr =
            pyo3::types::PyList::new(py, self.state.seen.iter().map(|n| n.0.clone_ref(py)))?
                .repr()?;
        let next_repr = {
            let next_keys: Vec<Py<PyAny>> = self
                .state
                .started_keys
                .iter()
                .map(|n| n.0.clone_ref(py))
                .collect();
            pyo3::types::PyList::new(py, next_keys)?.repr()?
        };
        Ok(format!(
            "_BreadthFirstSearcher(iterations={}, {}={}, seen={})",
            self.state.iterations(),
            prefix,
            next_repr,
            seen_repr
        ))
    }
}

#[pymodule]
fn _graph_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(invert_parent_map))?;
    m.add_wrapped(wrap_pyfunction!(collapse_linear_regions))?;
    m.add_wrapped(wrap_pyfunction!(DictParentsProvider))?;
    m.add_wrapped(wrap_pyfunction!(merge_sort))?;
    m.add_class::<TopoSorter>()?;
    m.add_class::<MergeSorter>()?;
    m.add_class::<PyKnownGraph>()?;
    m.add_class::<PyKnownGraphMergeSortNode>()?;
    m.add_class::<PyKnownGraphNode>()?;
    m.add_class::<PyKnownGraphNodesView>()?;
    m.add_class::<PyGraph>()?;
    m.add_class::<PyBFSearcher>()?;
    Ok(())
}
