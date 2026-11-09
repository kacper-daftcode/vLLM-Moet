#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests of moet_device_comm.py (MK-4): the order probe's decoding, the table, and the Python side of the protocol -
phase sequence, slot / arena parity and use counters, the arena bump allocator and its limits, the fallbacks, the verify
bookkeeping (mode 1's first-N agreement, mode verify's counters), step_done's watchdog record - with a fake extension that
reduces in NCCL's ring order on CPU tensors and a fake arena. The kernel itself is tested on 4 GPUs by MK/mk4/test_device_comm.py.

    python3 test_device_comm_unit.py
"""
from __future__ import annotations

import os
import sys
import types
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["VLLM_MOET_PREFILL_PIPE_DEVICE_COMM"] = "1"
os.environ["VLLM_MOET_PREFILL_PIPE_DEVICE_COMM_VERIFY_FIRST"] = "2"
os.environ["VLLM_MOET_PREFILL_PIPE_DEVICE_COMM_ARENA_MB"] = "1"
import moet_device_comm as dc  # noqa: E402

H = 128  # a narrow hidden for speed (row bytes a multiple of 16)


def ring_rs(parts: list[torch.Tensor], rank: int) -> torch.Tensor:
    """NCCL's reduce-scatter on CPU: rows of rank r summed as ((x[r+1] + x[r+2]) + x[r+3]) + x[r], bf16 after every add."""
    m = parts[0].shape[0]
    c = m // 4
    rows = slice(rank * c, (rank + 1) * c)
    acc = parts[(rank + 1) % 4][rows] + parts[(rank + 2) % 4][rows]
    acc = acc + parts[(rank + 3) % 4][rows]
    return acc + parts[rank][rows]


class FakeEvent:
    def record(self, stream=None):
        pass


class FakeStream:
    cuda_stream = 0

    def wait_event(self, ev):
        pass


class FakeExt:
    """Stands in for the CUDA extension: the four ranks' partials are known to the test (set_parts), the 'kernel' reduces this
    rank's rows in the ring order with the table (code per 4 elements), the gather lands in a CPU arena."""

    def __init__(self, arena_bytes: int) -> None:
        self.arena = [[torch.zeros(arena_bytes, dtype=torch.uint8) for _ in range(2)] for _ in range(2)]
        self.arena_bytes = arena_bytes
        self.parts: list[torch.Tensor] | None = None
        self.launches: list[tuple] = []
        self.err = [0] * 9
        self.rank = 0

    def rs_launch(self, stream, x, out, table, g, pseq, use, arrive_target, red_target, pieces, n_push, n_red, unroll, token, wd):
        assert arrive_target >= n_push and red_target >= pieces * n_red and 1 <= pieces <= 8, (arrive_target, red_target, pieces)
        self.launches.append(("rs", g, pseq, use))
        m, h = x.shape
        c = m // 4
        code = table[: c * h // 4].to(torch.int64).repeat_interleave(4).view(c, h)
        flat = torch.stack([p[self.rank * c:(self.rank + 1) * c] for p in self.parts]).reshape(4, -1)
        idx = [((code.reshape(-1) >> (2 * p)) & 3).unsqueeze(0) for p in range(4)]
        acc = torch.gather(flat, 0, idx[0]).squeeze(0)
        for p in range(1, 4):
            acc = acc + torch.gather(flat, 0, idx[p]).squeeze(0)
        out.copy_(acc.view(c, h))

    def ag_launch(self, stream, src, off, g, pseq, slot, use, arrive_target, red_target, pieces, n_push, n_red, unroll, token, wd):
        assert arrive_target >= n_push and red_target >= pieces * n_red and 1 <= pieces <= 8, (arrive_target, red_target, pieces)
        self.launches.append(("ag", g, pseq, slot, use, off))
        c, k = src.shape
        rb = k * src.element_size()
        for r in range(4):  # every rank's shard: the test's world holds identical shards rotated by rank
            if n_red == 0 and r == self.rank:
                continue  # split mode: the module copies the own shard itself
            rows = self.peer_shard(src, r)
            self.arena[g][pseq & 1][off + r * c * rb: off + (r + 1) * c * rb] = rows.contiguous().view(torch.uint8).reshape(-1)

    @staticmethod
    def peer_shard(src: torch.Tensor, r: int) -> torch.Tensor:
        return torch.roll(src, shifts=r, dims=0)

    def set_push_warps(self, w):
        self.push_warps = w

    def rs_wait_reduce(self, stream, x, out, table, g, pseq, use, red_target, pieces, grid, wd):
        self.launches.append(("rs_wait_reduce", g, pseq, use, grid))

    def ag_wait(self, stream, c, row_bytes, g, pseq, slot, use, pieces, wd):
        self.launches.append(("ag_wait", g, pseq, slot, use))

    def arena_view(self, g, par, off, rows, cols, like):
        n = rows * cols * like.element_size()
        return self.arena[g][par][off: off + n].view(like.dtype).view(rows, cols)

    # MK-5 (b): one instance for several tensors - the same landing as ag_launch per tensor, one launch record
    def ag_multi_launch(self, stream, srcs, offs, g, pseq, slot, use, arrive_target, n_push, unroll, wd):
        assert arrive_target >= n_push and 1 <= len(srcs) <= 3 and len(offs) == len(srcs)
        self.launches.append(("ag_multi", g, pseq, slot, use, tuple(offs), len(srcs)))
        for src, off in zip(srcs, offs):
            c, k = src.shape
            rb = k * src.element_size()
            for r in range(4):
                if r == self.rank:
                    continue
                rows = self.peer_shard(src, r)
                self.arena[g][pseq & 1][off + r * c * rb: off + (r + 1) * c * rb] = rows.contiguous().view(torch.uint8).reshape(-1)

    # MK-5 (c): the indexer arena [g][par] and its gather (the peers' padded rows = this rank's rotated by rank, as the test's world)
    IDX_SLOT = 5

    def idx_view(self, g, par, rows, cols, like):
        if not hasattr(self, "idx"):
            self.idx = [[torch.zeros(self.arena_bytes, dtype=torch.uint8) for _ in range(2)] for _ in range(2)]
        n = rows * cols * like.element_size()
        return self.idx[g][par][:n].view(like.dtype).view(rows, cols)

    def idx_launch(self, stream, src, g, par, reuse_min, use, arrive_target, n_push, unroll, wd):
        assert arrive_target >= n_push
        self.launches.append(("idx", g, par, reuse_min, use))
        c, k = src.shape
        rb = k * src.element_size()
        for r in range(4):
            if r == self.rank:
                continue
            self.idx[g][par][r * c * rb: (r + 1) * c * rb] = self.peer_shard(src, r).contiguous().view(torch.uint8).reshape(-1)

    def idx_wait(self, stream, g, par, use, wd):
        self.launches.append(("idx_wait", g, par, use))

    def set_push_form(self, stages, team, stamps, n_push):
        self.form = (stages, team, stamps, n_push)

    def err_check(self):
        return list(self.err)


def fake_state(ext: FakeExt) -> dc._State:
    st = dc._State()
    st.ext = ext
    st.rank, st.world = 0, 4
    st.device = torch.device("cpu")
    st.streams = [FakeStream(), FakeStream()]
    st.slot_bytes = 1024 * 5120 * 2
    st.arena_bytes = ext.arena_bytes
    st.v_n = torch.zeros((), dtype=torch.int64)
    st.v_bad = torch.zeros(3, dtype=torch.int64)
    st.idx_bytes = ext.arena_bytes
    return st


class DeviceCommUnit(unittest.TestCase):
    def setUp(self) -> None:
        dc._FORCE_GROUP = 0
        dc._ALLOW_CPU = True
        dc._DISABLED = None
        for k in ("rs_device", "ag_device", "rs_nccl_fallback", "ag_nccl_fallback", "verify_rs", "verify_ag", "steps"):
            dc._STATS[k] = 0
        dc._STATS["fallback_reasons"] = {}
        self.ext = FakeExt(1 << 20)
        self.st = fake_state(self.ext)
        dc._STATE = self.st
        dc._agree = lambda st, ok: bool(ok)  # one process stands in for the TP group
        self._cuda = (torch.cuda.Event, torch.cuda.current_stream)
        torch.cuda.Event = FakeEvent
        torch.cuda.current_stream = lambda *a, **k: FakeStream()
        self.m = 64
        torch.manual_seed(0)

    def tearDown(self) -> None:
        torch.cuda.Event, torch.cuda.current_stream = self._cuda
        dc._STATE = None
        dc._FORCE_GROUP = None
        dc._ALLOW_CPU = False

    # -- the probe
    def test_order_codes_decode_chain_and_reject_tree(self):
        m, h = 16, 32
        for rank in range(4):
            calls = [0]

            def rs_chain(x):
                r, s = calls[0] // 2, (2.0 ** -9, 3 * 2.0 ** -10)[calls[0] % 2]
                calls[0] += 1
                parts = [torch.full((m, h), 1.0 if rr == r else s, dtype=torch.bfloat16) for rr in range(4)]
                return ring_rs(parts, rank)

            code, ok = dc.rs_order_codes(rs_chain, rank, torch.device("cpu"), m, h)
            self.assertTrue(bool(ok.all()))
            r1, r2, r3 = (rank + 1) % 4, (rank + 2) % 4, (rank + 3) % 4
            lo, hi = min(r1, r2), max(r1, r2)
            self.assertTrue(bool((code == (lo | (hi << 2) | (r3 << 4) | (rank << 6))).all()), rank)
            table = dc.order_table(code, ok)
            self.assertIsNotNone(table)
            self.assertEqual(table.numel() % 64, 0)
            self.assertEqual(int(table[0]), lo | (hi << 2) | (r3 << 4) | (rank << 6))
            calls[0] = 0

            def rs_tree(x):
                r, s = calls[0] // 2, (2.0 ** -9, 3 * 2.0 ** -10)[calls[0] % 2]
                calls[0] += 1
                parts = [torch.full((m, h), 1.0 if rr == r else s, dtype=torch.bfloat16) for rr in range(4)]
                c = m // 4
                rows = slice(rank * c, (rank + 1) * c)
                return (parts[0][rows] + parts[1][rows]) + (parts[2][rows] + parts[3][rows])

            code_t, ok_t = dc.rs_order_codes(rs_tree, rank, torch.device("cpu"), m, h)
            self.assertFalse(bool(ok_t.any()))
            self.assertIsNone(dc.order_table(code_t, ok_t))

    def test_order_table_rejects_non_uniform_groups(self):
        code = torch.zeros(64, dtype=torch.int32)
        code[1] = 0x1B  # one element of the first 4 differs
        self.assertIsNone(dc.order_table(code, torch.ones(64, dtype=torch.bool)))

    # -- the protocol
    def _rs(self, seed: int, m: int | None = None):
        m = m or self.m
        g = torch.Generator().manual_seed(seed)
        parts = [(torch.randn(m, H, generator=g) * 3).to(torch.bfloat16) for _ in range(4)]
        self.ext.parts = parts
        ref = ring_rs(parts, 0)
        fallback_calls = []

        def fallback(x):
            fallback_calls.append(1)
            return ref.clone()

        return parts, ref, fallback, fallback_calls

    def _install_table(self, m: int = None):
        m = m or self.m
        c = m // 4
        code = torch.full((c * H,), 1 | (2 << 2) | (3 << 4) | (0 << 6), dtype=torch.int32)
        self.st.tables[(m, H, 0)] = dc.order_table(code, torch.ones(c * H, dtype=torch.bool))
        self.st.tables[(m, H, 1)] = self.st.tables[(m, H, 0)]

    def test_rs_mode1_first_n_then_device(self):
        self._install_table()
        for i in range(4):
            parts, ref, fb, calls = self._rs(100 + i)
            out = dc.reduce_scatter(parts[0], fb)
            self.assertTrue(torch.equal(out, ref), i)
            if i < 2:  # the first N = 2 collectives run NCCL too and return NCCL's
                self.assertEqual(calls, [1], i)
            else:
                self.assertEqual(calls, [], i)  # trusted: the device result alone
        self.assertTrue(self.st.trusted[("rs", self.m)])
        self.assertEqual(self.st.pseq[0], 4)
        rs = [l for l in self.ext.launches if l[0] == "rs"]
        self.assertEqual([l[:2] for l in rs], [("rs", 0)] * 4)
        self.assertEqual([l[3] for l in rs], [1, 1, 2, 2])  # use counts per parity
        self.assertEqual(dc._STATS["rs_device"], 4)
        self.assertEqual(dc._STATS["verify_rs"], 2)

    def test_rs_mode1_mismatch_keeps_nccl(self):
        self._install_table()
        parts, ref, fb, calls = self._rs(7)
        bad_ref = ref.clone()
        bad_ref[0, 0] = bad_ref[0, 0] + 1.0
        for i in range(3):
            out = dc.reduce_scatter(parts[0], lambda x: bad_ref.clone())
            if i < 2:
                self.assertTrue(torch.equal(out, bad_ref))  # NCCL's result consumed while untrusted
            else:
                self.assertIsNone(out)  # after the verdict: NCCL alone (the caller runs it), no device launch
        self.assertFalse(self.st.trusted[("rs", self.m)])
        self.assertEqual(int(self.st.v_bad[0]), 2)  # the first N comparisons recorded the mismatch
        self.assertEqual(len([l for l in self.ext.launches if l[0] == "rs"]), 2)
        self.assertEqual(dc._STATS["fallback_reasons"].get(f"device RS at M={self.m} differed from NCCL"), 1)

    def test_verify_mode_returns_nccl_and_counts(self):
        os.environ["VLLM_MOET_PREFILL_PIPE_DEVICE_COMM"] = "verify"
        try:
            self._install_table()
            for i in range(3):
                parts, ref, fb, calls = self._rs(200 + i)
                out = dc.reduce_scatter(parts[0], fb)
                self.assertEqual(calls, [1])
                self.assertTrue(torch.equal(out, ref))
                t = torch.randint(0, 256, (self.m // 4, 48), dtype=torch.uint8)
                got = dc.all_gather(t, lambda u: torch.cat([FakeExt.peer_shard(u, r) for r in range(4)]))
                self.assertTrue(torch.equal(got, torch.cat([FakeExt.peer_shard(t, r) for r in range(4)])))
            self.assertEqual(int(self.st.v_n), 6)
            self.assertEqual(int(self.st.v_bad.sum()), 0)
            dc.step_done()
            self.assertEqual(dc._STATS["verify_rs_mismatch"], 0)
        finally:
            os.environ["VLLM_MOET_PREFILL_PIPE_DEVICE_COMM"] = "1"

    def test_ag_arena_cursor_slots_and_limits(self):
        self._install_table()
        self.st.trusted[("rs", self.m)] = True
        parts, ref, fb, calls = self._rs(300)
        dc.reduce_scatter(parts[0], fb)
        c = self.m // 4
        nccl = lambda u: torch.cat([FakeExt.peer_shard(u, r) for r in range(4)])  # noqa: E731
        for k in range(4):  # four gathers fit the 1-MB arena: 4 x 16 rows x 4 x 64 B
            t = torch.randint(0, 256, (c, 64), dtype=torch.uint8)
            self.st.trusted[("ag", c)] = True
            got = dc.all_gather(t, nccl)
            self.assertTrue(torch.equal(got, nccl(t)))
        ag = [l for l in self.ext.launches if l[0] == "ag"]
        self.assertEqual([l[3] for l in ag], [1, 2, 3, 4])  # slots 1..4
        self.assertEqual([l[5] for l in ag], [0, 4096, 8192, 12288])  # the bump allocator, 256-B aligned
        t = torch.randint(0, 256, (c, 64), dtype=torch.uint8)
        got = dc.all_gather(t, nccl)  # the fifth: more than 4 gathers in a phase -> NCCL
        self.assertTrue(torch.equal(got, nccl(t)))
        self.assertEqual(dc._STATS["fallback_reasons"].get("more than 4 gathers in a phase"), 1)
        # a new phase resets the cursor and the slot index; a gather larger than the arena falls back
        parts, ref, fb, calls = self._rs(301)
        dc.reduce_scatter(parts[0], fb)
        big = torch.zeros((c, (self.ext.arena_bytes // (4 * c)) + 16), dtype=torch.uint8)
        got = dc.all_gather(big, nccl)
        self.assertEqual(dc._STATS["fallback_reasons"].get("arena full"), 1)
        t = torch.randint(0, 256, (c, 64), dtype=torch.uint8)
        dc.all_gather(t, nccl)
        ags = [l for l in self.ext.launches if l[0] == "ag"]
        self.assertEqual(ags[-1][3:6], (1, 1, 0))  # slot 1 again, offset 0, its first use on parity 1
        parts, ref, fb, calls = self._rs(302)
        dc.reduce_scatter(parts[0], fb)
        dc.all_gather(t, nccl)
        ags = [l for l in self.ext.launches if l[0] == "ag"]
        self.assertEqual(ags[-1][3:6], (1, 2, 0))  # parity 0 again: slot 1's second use
        # the slot / arena parity alternates with the phase sequence
        self.assertEqual([l[2] & 1 for l in self.ext.launches if l[0] == "rs"], [0, 1, 0])
        self.assertEqual(len([l for l in self.ext.launches if l[0] == "rs_wait_reduce"]), 3)  # split mode: a wait + reduce per RS

    def test_ag_many_one_instance_per_phase(self):
        # MK-5 (b): two tensors of one phase -> one ag_multi launch with two arena offsets, both gathered rows equal NCCL's; a tensor
        # the device path cannot take (odd row bytes) sends the whole call down the single-tensor path (fallback per tensor)
        os.environ["VLLM_MOET_PREFILL_PIPE_DEVICE_COMM_MULTI"] = "1"
        try:
            self._install_table()
            self.st.trusted[("rs", self.m)] = True
            parts, ref, fb, calls = self._rs(500)
            dc.reduce_scatter(parts[0], fb)
            c = self.m // 4
            nccl = lambda u: torch.cat([FakeExt.peer_shard(u, r) for r in range(4)])  # noqa: E731
            q = torch.randint(0, 256, (c, 64), dtype=torch.uint8)
            small = torch.randint(0, 256, (c, 16), dtype=torch.uint8)
            self.st.trusted[("ag", c)] = True
            got = dc.all_gather_many([q, small], nccl)
            self.assertTrue(torch.equal(got[0], nccl(q)))
            self.assertTrue(torch.equal(got[1], nccl(small)))
            multi = [l for l in self.ext.launches if l[0] == "ag_multi"]
            self.assertEqual(len(multi), 1)
            self.assertEqual(multi[0][3], 1)  # slot 1
            self.assertEqual(multi[0][5], (0, 4096))  # q's 4 x 16 x 64 B, then small (256-B aligned)
            self.assertEqual(multi[0][6], 2)
            self.assertEqual(len([l for l in self.ext.launches if l[0] == "ag_wait"]), 1)  # one wait for both
            self.assertEqual(self.st.ag_index[0], 1)
            self.assertEqual(dc._STATS["ag_multi_device"], 1)
            odd = torch.zeros((c, 3), dtype=torch.uint8)
            got = dc.all_gather_many([q, odd], nccl)
            self.assertTrue(torch.equal(got[0], nccl(q)) and torch.equal(got[1], nccl(odd)))
            self.assertEqual(dc._STATS["fallback_reasons"].get("row bytes 3"), 1)
            self.assertEqual(len([l for l in self.ext.launches if l[0] == "ag"]), 1)  # q alone went through the single path
        finally:
            os.environ.pop("VLLM_MOET_PREFILL_PIPE_DEVICE_COMM_MULTI", None)

    def test_all_gatherv_indexer_rows(self):
        # MK-5 (c): variable per-rank row counts, padded to the max, the valid rows of every rank land in `out` in rank order; the
        # parity alternates per call, the reuse threshold follows the group's phase sequence; verify mode compares and consumes NCCL's
        os.environ["VLLM_MOET_PREFILL_PIPE_DEVICE_COMM_IDX"] = "1"
        try:
            self._install_table()
            self.st.trusted[("rs", self.m)] = True
            sizes = [5, 3, 6, 2]
            width = 8  # 32-B rows (int32)
            dc._FORCE_GROUP = 0
            for i in range(3):
                local = torch.randint(0, 1 << 20, (sizes[0], width), dtype=torch.int32)
                # the test's world: rank r's rows = this rank's rotated by r (peer_shard), sizes differ -> the fake pads like the module
                def nccl(u):
                    pads = [FakeExt.peer_shard(torch.cat([u, torch.zeros(max(sizes) - sizes[0], width, dtype=torch.int32)]), r)[: sizes[r]] for r in range(4)]
                    return torch.cat(pads)
                out = torch.zeros((sum(sizes), width), dtype=torch.int32)
                self.assertTrue(dc.all_gatherv(local, sizes, out, nccl))
                self.assertTrue(torch.equal(out, nccl(local)), i)
            idx = [l for l in self.ext.launches if l[0] == "idx"]
            self.assertEqual([l[2] for l in idx], [0, 1, 0])  # the parity alternates
            self.assertEqual([l[3] for l in idx], [0, 0, 0])  # no phases yet: no reuse threshold
            self.assertEqual([l[4] for l in idx], [1, 1, 2])  # use counts per parity
            self.st.pseq[0] = 9
            local = torch.randint(0, 1 << 20, (sizes[0], width), dtype=torch.int32)
            out = torch.zeros((sum(sizes), width), dtype=torch.int32)
            self.assertTrue(dc.all_gatherv(local, sizes, out, nccl))
            self.assertEqual([l for l in self.ext.launches if l[0] == "idx"][-1][3], 5)  # reuse_min = pseq - 4
            self.assertEqual(dc._STATS["idx_device"], 4)
            bad = torch.zeros((sizes[0], 3), dtype=torch.int32)  # 12-B rows
            self.assertFalse(dc.all_gatherv(bad, sizes, out, nccl))
        finally:
            os.environ.pop("VLLM_MOET_PREFILL_PIPE_DEVICE_COMM_IDX", None)

    def test_ag_without_device_phase_falls_back(self):
        t = torch.randint(0, 256, (16, 64), dtype=torch.uint8)
        nccl = lambda u: torch.cat([FakeExt.peer_shard(u, r) for r in range(4)])  # noqa: E731
        self.assertTrue(torch.equal(dc.all_gather(t, nccl), nccl(t)))
        self.assertEqual(dc._STATS["fallback_reasons"].get("phase not on the device"), 1)
        self._install_table()
        parts, ref, fb, calls = self._rs(400)
        dc.reduce_scatter(parts[0], fb)
        odd = torch.zeros((16, 3), dtype=torch.uint8)  # 3 B rows: not a multiple of 16
        dc.all_gather(odd, nccl)
        self.assertEqual(dc._STATS["fallback_reasons"].get("row bytes 3"), 1)

    def test_rs_fallbacks(self):
        self.assertIsNone(dc.reduce_scatter(torch.zeros(63, H, dtype=torch.bfloat16), lambda x: x))  # rows not a multiple of 4
        self.assertIsNone(dc.reduce_scatter(torch.zeros(64, H, dtype=torch.float32), lambda x: x))  # dtype
        self.assertEqual(self.st.cur_pseq[0], -1)  # the phase is not on the device -> its gathers fall back too
        dc._FORCE_GROUP = None  # outside a pipelined phase (no pipeline module here)

        def boom():
            raise ImportError("no pipeline")

        dc._pipe = boom
        self.assertIsNone(dc.reduce_scatter(torch.zeros(64, H, dtype=torch.bfloat16), lambda x: x) if False else None)
        self.assertEqual(dc._STATS["fallback_reasons"].get("shape (63, 128) torch.bfloat16"), 1)

    def test_step_done_raises_on_watchdog_record(self):
        self.ext.err = [1, 1, 5, 3, 7, 0, 0, 2, 3_000_000_000]
        with self.assertRaises(RuntimeError):
            dc.step_done()
        self.assertEqual(dc._STATS["watchdog"]["what"], 1)
        self.assertAlmostEqual(dc._STATS["watchdog"]["waited_ms"], 3000.0)
        self.ext.err = [0] * 9


if __name__ == "__main__":
    unittest.main(verbosity=2)
