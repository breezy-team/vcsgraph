#![allow(non_snake_case)]

use vcs_graph::known_graph::KnownGraph as RsKnownGraph;
use vcs_graph::{ChildMap, ParentMap, RevnoVec};

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
    Ok(())
}
