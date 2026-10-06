"""NCCL's per-collective serialEvent record nodes in a captured CUDA graph, reduced to the last one.

With graph mixing on (NCCL's default, graphUsageMode=2) NCCL serializes its device work across graphs and eager
calls through one event per communicator ("serialEvent" of the communicator's device strong stream): the first
NCCL launch captured in a graph waits on it (an external event-wait node), and every captured launch records it
afterwards (ncclStrongStreamRelease -> an event-record node behind the NCCL kernel), which the next NCCL kernel of
the graph then depends on. A graph with n collectives therefore carries n record nodes, each a second child of its
NCCL kernel and a second parent of the next one; on the RTX PRO 6000 host the graph executor pays ~2-3 us of idle
GPU in front of every NCCL kernel for it and puts the NCCL kernels on their own streams (nccl_gap_tp4.py measures it).

Only the last record of the graph matters to anyone waiting on the event (later graphs, eager NCCL calls: they wait
for the most recent record), and the order between the graph's own NCCL kernels is kept by direct edges. So the
record nodes are dropped except the topologically last one per event, each dropped node's parents connected to its
children. Only record nodes whose parents are all NCCL kernels are touched; a node is kept when one of the direct
edges it would be replaced by already exists with a non-default type (e.g. a programmatic edge).

    n_dropped, n_records = drop_intermediate_nccl_records(cudagraph.raw_cuda_graph())   # keep_graph=True capture

Every subset of the drops leaves a valid graph with the original ordering (edges are added before a node goes).
Installed into vLLM as vllm/compilation/moet_nccl_graph_records.py by patch_vllm_nccl_graph_records.py; the patched
capture sites call records_enabled() (VLLM_MOET_NCCL_GRAPH_RECORDS, default 1) and finish_capture().
"""

from __future__ import annotations

import os

import torch

_LOGGED = {"info": False, "warning": False}


def records_enabled() -> bool:
    return os.environ.get("VLLM_MOET_NCCL_GRAPH_RECORDS", "1") == "1"


def finish_capture(graph: torch.cuda.CUDAGraph) -> None:
    """After a keep_graph=True capture: drop the intermediate NCCL records, then instantiate the graph.

    A failing rewrite leaves the graph as captured (every partial rewrite is a valid graph) and is logged once.
    """
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    try:
        dropped, found = drop_intermediate_nccl_records(graph.raw_cuda_graph())
        if found and not _LOGGED["info"]:
            _LOGGED["info"] = True
            logger.info(
                "vllm-moet: NCCL serialEvent records in captured graphs reduced to the last one per event "
                "(first graph: %d of %d dropped; VLLM_MOET_NCCL_GRAPH_RECORDS=0 keeps them)", dropped, found)
    except Exception as e:
        if not _LOGGED["warning"]:
            _LOGGED["warning"] = True
            logger.warning("vllm-moet: NCCL graph records kept as captured: %s", e)
    graph.instantiate()


def _ok(res):
    if int(res[0]) != 0:
        raise RuntimeError(f"CUDA error {res[0]}")
    return res[1:] if len(res) > 2 else (res[1] if len(res) > 1 else None)


def drop_intermediate_nccl_records(raw_graph: int) -> tuple[int, int]:
    """Drop all but the last NCCL serialEvent record node per event; returns (dropped, record nodes found)."""
    import cuda.bindings.driver as drv
    import cuda.bindings.runtime as rt

    T = rt.cudaGraphNodeType
    graph = rt.cudaGraph_t(init_value=int(raw_graph))
    n = _ok(rt.cudaGraphGetNodes(graph, 0))[1]
    nodes = list(_ok(rt.cudaGraphGetNodes(graph, n))[0])
    types = [_ok(rt.cudaGraphNodeGetType(x)) for x in nodes]
    records = [i for i, t in enumerate(types) if t == T.cudaGraphNodeTypeEventRecord]
    if not records:
        return 0, 0
    idx = {int(x): i for i, x in enumerate(nodes)}
    ne = _ok(rt.cudaGraphGetEdges(graph, 0))[-1]
    frm, to, data, _ = _ok(rt.cudaGraphGetEdges(graph, ne))
    parents = [[] for _ in nodes]
    children = [[] for _ in nodes]
    edge_type = {}
    for k, (a, b) in enumerate(zip(frm, to)):
        a, b = idx[int(a)], idx[int(b)]
        children[a].append(b)
        parents[b].append(a)
        edge_type[(a, b)] = int(data[k].type) if data is not None and k < len(data) else 0

    names: dict[int, str] = {}

    def is_nccl_kernel(i: int) -> bool:
        if types[i] != T.cudaGraphNodeTypeKernel:
            return False
        if i not in names:
            name = ""
            params = _ok(drv.cuGraphKernelNodeGetParams(drv.CUgraphNode(int(nodes[i]))))
            func = getattr(params, "func", None)
            if func is not None and int(func) != 0:
                name = _ok(drv.cuFuncGetName(func)).decode()
            names[i] = name
        return "ncclDevKernel" in names[i]

    indeg = [len(p) for p in parents]
    ready = [i for i, d in enumerate(indeg) if d == 0]
    pos = {}
    while ready:
        u = ready.pop()
        pos[u] = len(pos)
        for c in children[u]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)

    by_event: dict[int, list[int]] = {}
    for r in records:
        if parents[r] and all(is_nccl_kernel(p) for p in parents[r]):
            by_event.setdefault(int(_ok(rt.cudaGraphEventRecordNodeGetEvent(nodes[r]))), []).append(r)
    drop = []
    for group in by_event.values():
        last = max(group, key=pos.__getitem__)
        for r in group:
            if r != last and all(edge_type.get((p, c), 0) == 0 for p in parents[r] for c in children[r]):
                drop.append(r)
    for r in drop:
        for p in parents[r]:
            for c in children[r]:
                if c not in children[p]:
                    _ok((rt.cudaGraphAddDependencies(graph, [nodes[p]], [nodes[c]], None, 1)[0],))
                    children[p].append(c)
                    parents[c].append(p)
        _ok((rt.cudaGraphDestroyNode(nodes[r])[0],))
        for p in parents[r]:
            children[p].remove(r)
        for c in children[r]:
            parents[c].remove(r)
    return len(drop), len(records)
