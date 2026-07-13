"""Standalone unit test for vllm/v1/attention/ops/dcp_sparse_topk.py.

Loads the module file directly with a stubbed GroupCoordinator so it runs on
a host without a full vLLM install. Pure CPU torch.
"""
import importlib.util
import sys
import types

import torch

# Stub vllm.distributed.parallel_state.GroupCoordinator (only used as a type
# annotation + duck-typed .world_size/.rank_in_group/.all_gather).
pkg_v = types.ModuleType("vllm")
pkg_d = types.ModuleType("vllm.distributed")
pkg_p = types.ModuleType("vllm.distributed.parallel_state")


class GroupCoordinator:  # noqa: D401
    pass


pkg_p.GroupCoordinator = GroupCoordinator
sys.modules.setdefault("vllm", pkg_v)
sys.modules["vllm.distributed"] = pkg_d
sys.modules["vllm.distributed.parallel_state"] = pkg_p

spec = importlib.util.spec_from_file_location(
    "dcp_sparse_topk",
    "/root/workspace/vllm-v0.24.0/vllm/v1/attention/ops/dcp_sparse_topk.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def brute_owner(pos: int, world: int, interleave: int) -> int:
    return (pos // interleave) % world


def brute_local_pos(pos: int, world: int, interleave: int) -> int:
    unit = pos // interleave
    return (unit // world) * interleave + pos % interleave


def test_counts_and_mapping():
    for world in (1, 2, 4):
        for interleave in (1, 4, 256):
            for L in (0, 1, 3, 255, 256, 257, 1000, 4096, 999_999):
                lens = torch.tensor([L], dtype=torch.int64)
                total = 0
                for r in range(world):
                    cnt = int(
                        m.dcp_local_count_from_global(lens, world, r, interleave)[0]
                    )
                    cnt_int = m.dcp_local_count_int(L, world, r, interleave)
                    assert cnt == cnt_int, (L, world, r, interleave, cnt, cnt_int)
                    # brute force
                    ref = sum(
                        1
                        for p in range(min(L, 5000))
                        if brute_owner(p, world, interleave) == r
                    )
                    if L <= 5000:
                        assert cnt == ref, (L, world, r, interleave, cnt, ref)
                    total += cnt
                    # local->global roundtrip on a sample
                    n_check = min(cnt, 2000)
                    if n_check > 0:
                        lp = torch.arange(n_check, dtype=torch.int32)
                        g = m.dcp_local_pos_to_global(lp, world, r, interleave)
                        for j in range(n_check):
                            gp = int(g[j])
                            assert brute_owner(gp, world, interleave) == r
                            assert brute_local_pos(gp, world, interleave) == j
                            assert gp < L or L > 5000
                assert total == L, (L, world, interleave, total)
    # -1 passthrough
    lp = torch.tensor([-1, 0, -1, 5], dtype=torch.int32)
    g = m.dcp_local_pos_to_global(lp, 4, 2, 16)
    assert g[0] == -1 and g[2] == -1
    print("counts/mapping OK")


class FakeGroup:
    """Emulates N-rank all_gather by collecting per-rank tensors."""

    def __init__(self, world, tensors_by_rank):
        self.world_size = world
        self.rank_in_group = 0
        self._t = tensors_by_rank

    def all_gather(self, t, dim=0):
        return torch.cat(self._t, dim=dim)


def test_merge():
    torch.manual_seed(0)
    world, interleave, topk = 4, 16, 64
    for L in (5, 40, 63, 64, 100, 999):
        rows = 3
        scores_global = torch.randn(rows, L)
        # reference: global topk ids per row (set equality; ties negligible w/ randn)
        ref_ids = []
        for row in range(rows):
            k = min(topk, L)
            ref = torch.topk(scores_global[row], k=k).indices.tolist()
            ref_ids.append(set(ref))

        packed_by_rank = []
        for r in range(world):
            owned = [
                p for p in range(L) if brute_owner(p, world, interleave) == r
            ]
            local_scores = torch.full((rows, topk), float("-inf"))
            local_ids = torch.full((rows, topk), -1, dtype=torch.int32)
            for row in range(rows):
                if owned:
                    s = scores_global[row, owned]
                    k = min(topk, len(owned))
                    top = torch.topk(s, k=k)
                    local_scores[row, :k] = top.values
                    for j, oi in enumerate(top.indices.tolist()):
                        local_ids[row, j] = owned[oi]
            packed = torch.empty((rows, 2, topk))
            packed[:, 0] = local_scores
            packed[:, 1] = local_ids.to(torch.float32)
            packed_by_rank.append(packed)

        group = FakeGroup(world, packed_by_rank)
        # feed rank 0's locals (merge output must not depend on which rank runs it)
        r0 = packed_by_rank[0]
        merged = m.dcp_merge_global_topk(
            r0[:, 1].to(torch.int32), r0[:, 0], topk, group
        )
        for row in range(rows):
            got = [int(x) for x in merged[row] if int(x) >= 0]
            assert len(got) == len(set(got)), "duplicates!"
            assert set(got) == ref_ids[row], (L, row, sorted(got), sorted(ref_ids[row]))
            # padding contract
            n_valid = len(ref_ids[row])
            assert (merged[row, n_valid:] == -1).all()
    print("merge OK")


def test_gather_scores():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    idx = torch.tensor([[2, 0, -1], [1, -1, 3]], dtype=torch.int32)
    s = m.dcp_gather_topk_scores(logits, idx)
    assert s[0, 0] == 3.0 and s[0, 1] == 1.0 and s[0, 2] == float("-inf")
    assert s[1, 0] == 20.0 and s[1, 1] == float("-inf") and s[1, 2] == 40.0
    # with col_offset (prefill workspace rebase)
    off = torch.tensor([1, 0], dtype=torch.int32)
    s2 = m.dcp_gather_topk_scores(logits, idx, col_offset=off)
    assert s2[0, 0] == 4.0 and s2[0, 1] == 2.0 and s2[0, 2] == float("-inf")
    print("gather OK")


test_counts_and_mapping()
test_gather_scores()
test_merge()
print("ALL-OK")
