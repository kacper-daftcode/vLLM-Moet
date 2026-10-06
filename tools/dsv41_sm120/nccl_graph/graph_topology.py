"""Inspect (and, for experiments, rewrite) the topology of a captured CUDA graph around its NCCL kernel nodes.

    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g): ...
    topo = Topology(g.raw_cuda_graph())
    print(topo.summary())          # node types, NCCL kernel nodes with their parents / children
    topo.drop_event_nodes()        # experiment only: remove event record / wait nodes, keep their ordering
    g.instantiate()

Uses cuda-python (cuda.bindings), which the serving image ships.
"""

from __future__ import annotations

from collections import Counter

import cuda.bindings.driver as drv
import cuda.bindings.runtime as rt

_T = rt.cudaGraphNodeType
_SHORT = {
    _T.cudaGraphNodeTypeKernel: "kernel",
    _T.cudaGraphNodeTypeMemcpy: "memcpy",
    _T.cudaGraphNodeTypeMemset: "memset",
    _T.cudaGraphNodeTypeHost: "host",
    _T.cudaGraphNodeTypeGraph: "child-graph",
    _T.cudaGraphNodeTypeEmpty: "empty",
    _T.cudaGraphNodeTypeWaitEvent: "event-wait",
    _T.cudaGraphNodeTypeEventRecord: "event-record",
    _T.cudaGraphNodeTypeMemAlloc: "mem-alloc",
    _T.cudaGraphNodeTypeMemFree: "mem-free",
}


def _ok(res):
    err = res[0]
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return res[1:] if len(res) > 2 else res[1]


def _kernel_name(node) -> str:
    try:
        params = _ok(drv.cuGraphKernelNodeGetParams(drv.CUgraphNode(int(node))))
    except RuntimeError:
        return "?"
    func = getattr(params, "func", None)
    if func is not None and int(func) != 0:
        try:
            return _ok(drv.cuFuncGetName(func)).decode()
        except RuntimeError:
            pass
    kern = getattr(params, "kern", None)
    if kern is not None and int(kern) != 0 and hasattr(drv, "cuKernelGetName"):
        try:
            return _ok(drv.cuKernelGetName(kern)).decode()
        except RuntimeError:
            pass
    return "?"


class Topology:
    def __init__(self, raw_graph: int):
        self.graph = rt.cudaGraph_t(init_value=int(raw_graph))
        self.refresh()

    def refresh(self) -> None:
        n = _ok(rt.cudaGraphGetNodes(self.graph, 0))[1]
        nodes = _ok(rt.cudaGraphGetNodes(self.graph, n))[0]
        self.nodes = list(nodes)
        self.idx = {int(nd): i for i, nd in enumerate(self.nodes)}
        self.type = [_ok(rt.cudaGraphNodeGetType(nd)) for nd in self.nodes]
        self.name = [_kernel_name(nd) if t == _T.cudaGraphNodeTypeKernel else _SHORT.get(t, str(t))
                     for nd, t in zip(self.nodes, self.type)]
        ne = _ok(rt.cudaGraphGetEdges(self.graph, 0))
        ne = ne[-1] if isinstance(ne, tuple) else ne
        res = _ok(rt.cudaGraphGetEdges(self.graph, ne))
        frm, to = res[0], res[1]
        self.edges = [(self.idx[int(a)], self.idx[int(b)]) for a, b in zip(frm, to)]
        self.parents = [[] for _ in self.nodes]
        self.children = [[] for _ in self.nodes]
        for a, b in self.edges:
            self.children[a].append(b)
            self.parents[b].append(a)

    def is_nccl(self, i: int) -> bool:
        return self.type[i] == _T.cudaGraphNodeTypeKernel and "nccl" in self.name[i].lower()

    def nccl_nodes(self) -> list[int]:
        return [i for i in range(len(self.nodes)) if self.is_nccl(i)]

    def _label(self, i: int) -> str:
        nm = self.name[i]
        return nm[:40] if self.type[i] == _T.cudaGraphNodeTypeKernel else f"<{nm}>"

    def summary(self, show: int = 3) -> str:
        counts = Counter(_SHORT.get(t, str(t)) for t in self.type)
        nccl = self.nccl_nodes()
        lines = [f"{len(self.nodes)} nodes, {len(self.edges)} edges: "
                 + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
                 f"{len(nccl)} NCCL kernel nodes; in/out degree of each: "
                 + " ".join(f"{len(self.parents[i])}/{len(self.children[i])}" for i in nccl[:12])
                 + (" ..." if len(nccl) > 12 else "")]
        for i in nccl[:show] + nccl[len(nccl) // 2: len(nccl) // 2 + 1]:
            lines.append(f"  NCCL node {i} {self.name[i][:48]}")
            lines.append("    parents : " + "; ".join(self._label(p) for p in self.parents[i]))
            lines.append("    children: " + "; ".join(self._label(c) for c in self.children[i]))
        return "\n".join(lines)

    def _reachable_without(self, src: int, dst: int) -> bool:
        """Is dst reachable from src without the direct edge src -> dst?"""
        stack = [c for c in self.children[src] if c != dst]
        seen = set(stack)
        while stack:
            u = stack.pop()
            if u == dst:
                return True
            for c in self.children[u]:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return False

    def _remove_edge(self, a: int, b: int) -> None:
        _ok((rt.cudaGraphRemoveDependencies(self.graph, [self.nodes[a]], [self.nodes[b]], None, 1)[0], None))

    def _add_edge(self, a: int, b: int) -> None:
        _ok((rt.cudaGraphAddDependencies(self.graph, [self.nodes[a]], [self.nodes[b]], None, 1)[0], None))

    def drop_event_nodes(self) -> int:
        """Experiment: destroy every event record / wait node, connecting its parents to its children."""
        ev = [i for i, t in enumerate(self.type)
              if t in (_T.cudaGraphNodeTypeWaitEvent, _T.cudaGraphNodeTypeEventRecord)]
        self._drop_nodes(ev)
        return len(ev)

    def topo_order(self) -> list[int]:
        indeg = [len(p) for p in self.parents]
        ready = [i for i, d in enumerate(indeg) if d == 0]
        order = []
        while ready:
            u = ready.pop()
            order.append(u)
            for c in self.children[u]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
        return order

    def _drop_nodes(self, drop: list[int]) -> None:
        for i in drop:
            for p in self.parents[i]:
                for c in self.children[i]:
                    if c not in self.children[p]:
                        self._add_edge(p, c)
                        self.children[p].append(c)
                        self.parents[c].append(p)
        for i in drop:
            _ok((rt.cudaGraphDestroyNode(self.nodes[i])[0], None))
        self.refresh()

    def drop_intermediate_records(self) -> int:
        """Keep, per recorded event, only the record node that runs last; the others become plain edges.

        With graph mixing on, NCCL records its strong stream's serialEvent after every NCCL kernel of the graph and
        makes the next NCCL kernel depend on that record node. Its NCCL kernels are chained either way, so for
        whoever waits on serialEvent (later graphs, eager NCCL calls) only the last record of the graph matters.
        """
        recs: dict[int, list[int]] = {}
        for i, t in enumerate(self.type):
            if t == _T.cudaGraphNodeTypeEventRecord:
                ev = _ok(rt.cudaGraphEventRecordNodeGetEvent(self.nodes[i]))
                recs.setdefault(int(ev), []).append(i)
        pos = {n: k for k, n in enumerate(self.topo_order())}
        drop = []
        for nodes in recs.values():
            if len(nodes) > 1:
                last = max(nodes, key=pos.get)
                drop += [i for i in nodes if i != last]
        self._drop_nodes(drop)
        return len(drop)

    def drop_redundant_nccl_edges(self) -> int:
        """Experiment: remove edges into NCCL nodes that are implied by another path (transitive)."""
        n = 0
        for b in self.nccl_nodes():
            for a in list(self.parents[b]):
                if self._reachable_without(a, b):
                    self._remove_edge(a, b)
                    self.children[a].remove(b)
                    self.parents[b].remove(a)
                    n += 1
        self.refresh()
        return n
