"""Focused fail-closed tests for W2 checkpoint/pack staging safety.

Run inside the serving image after the repository patch is applied:

    python3 /serve/tools/test_store_safety.py

No GPU or model checkpoint is required. The tests use tiny safetensors files
and mock memory readings; they never touch the production pack directory.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file
from vllm.model_executor.layers.quantization import mxfp4
from vllm.model_executor.layers.quantization.utils import moe_w2_cubit as cubit
from vllm.model_executor.layers.quantization.utils import moe_w2_store as store
from vllm.model_executor.layers.quantization.utils.moe_w2_cubit import (
    _StreamLoader,
)
from vllm.model_executor.model_loader import default_loader as default_loader_module
from vllm.model_executor.model_loader import get_model_loader, weight_utils
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

GIB = 1 << 30
SAFETY_ENV = {
    "VLLM_MOE_W2": "1",
    "VLLM_MOE_W2_STORE_DIR": "/tmp/test-w2-store-safety",
    "VLLM_MOE_W2_CACHE_CONTROL": "required",
    "VLLM_MOE_W2_MIN_MEM_AVAILABLE_GB": "16",
    "VLLM_MOE_W2_MIN_CGROUP_HEADROOM_GB": "4",
}


def cgroup_unlimited():
    return {
        "known": True,
        "version": 2,
        "limited": False,
        "max_available": None,
        "high_available": None,
        "path": "/test",
        "current": 1 * GIB,
        "events": {},
    }


def cgroup_limited(max_available: int, high_available: int | None = None):
    return {
        "known": True,
        "version": 2,
        "limited": True,
        "max_available": max_available,
        "high_available": high_available,
        "path": "/test",
        "current": 1 * GIB,
        "events": {},
    }


def clear_pending_drops():
    with store._pending_checkpoint_lock:
        store._pending_checkpoint_drops.clear()


class MemoryPreflightTest(unittest.TestCase):
    def test_host_floor_refuses_before_transient(self):
        with (
            mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
            mock.patch.object(store, "_mem_available_bytes", return_value=20 * GIB),
            mock.patch.object(
                store, "_cgroup_memory_status", side_effect=cgroup_unlimited
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "REFUSED unit-host"):
                store._memory_preflight("unit-host", 5 * GIB)

    def test_cgroup_floor_refuses_before_transient(self):
        with (
            mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
            mock.patch.object(store, "_mem_available_bytes", return_value=64 * GIB),
            mock.patch.object(
                store, "_cgroup_memory_status", return_value=cgroup_limited(6 * GIB)
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "memory.max headroom"):
                store._memory_preflight("unit-cgroup", 3 * GIB)

    def test_safe_operation_reports_budget(self):
        with (
            mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
            mock.patch.object(store, "_mem_available_bytes", return_value=64 * GIB),
            mock.patch.object(
                store, "_cgroup_memory_status", return_value=cgroup_limited(32 * GIB)
            ),
        ):
            report = store._memory_preflight("unit-pass", 2 * GIB)
        self.assertEqual(report["available"], 64 * GIB)
        self.assertEqual(report["transient"], 2 * GIB)

    def test_unknown_cgroup_refuses_instead_of_looking_unlimited(self):
        with (
            mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
            mock.patch.object(store, "_mem_available_bytes", return_value=64 * GIB),
            mock.patch.object(
                store,
                "_cgroup_memory_status",
                return_value={"known": False, "error": "unreadable"},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot determine.*cgroup"):
                store._memory_preflight("unit-unknown", 1 * GIB)

    def test_cgroup_v2_reports_crossed_soft_high_separately_from_hard_max(self):
        with tempfile.TemporaryDirectory() as root:
            files = {
                "memory.current": str(11 * GIB),
                "memory.high": str(10 * GIB),
                "memory.max": str(20 * GIB),
                "memory.stat": "anon 1\nfile 2\nfile_mapped 3\n",
                "memory.events": "high 0\nmax 0\noom 0\n",
                "memory.swap.current": "0",
                "memory.swap.max": "max",
            }
            for name, value in files.items():
                with open(os.path.join(root, name), "w") as f:
                    f.write(value)
            with mock.patch.object(
                store, "_active_cgroup_v2_dirs", return_value=[root]
            ):
                status = store._cgroup_memory_status()
        self.assertTrue(status["known"])
        self.assertTrue(status["limited"])
        self.assertEqual(status["max_available"], 9 * GIB)
        self.assertEqual(status["high_available"], -1 * GIB)
        self.assertEqual(status["file_mapped"], 3)

    def test_cgroup_preflight_does_not_gate_on_crossed_memory_high(self):
        with (
            mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
            mock.patch.object(store, "_mem_available_bytes", return_value=64 * GIB),
            mock.patch.object(
                store,
                "_cgroup_memory_status",
                return_value=cgroup_limited(8 * GIB, -1 * GIB),
            ),
        ):
            report = store._memory_preflight("unit-soft-high", 2 * GIB)
        self.assertEqual(report["cgroup_max_available"], 8 * GIB)
        self.assertEqual(report["cgroup_high_available"], -1 * GIB)

    def test_cgroup_soft_high_does_not_make_unlimited_max_look_limited(self):
        with tempfile.TemporaryDirectory() as root:
            files = {
                "memory.current": str(5 * GIB),
                "memory.high": str(10 * GIB),
                "memory.max": "max",
                "memory.stat": "",
                "memory.events": "high 0\nmax 0\noom 0\n",
            }
            for name, value in files.items():
                with open(os.path.join(root, name), "w") as f:
                    f.write(value)
            with mock.patch.object(
                store, "_active_cgroup_v2_dirs", return_value=[root]
            ):
                status = store._cgroup_memory_status()
        self.assertFalse(status["limited"])
        self.assertIsNone(status["max_available"])
        self.assertEqual(status["high_available"], 5 * GIB)

    def test_cgroup_v1_uses_tightest_process_ancestor(self):
        with (
            tempfile.TemporaryDirectory() as leaf,
            tempfile.TemporaryDirectory() as parent,
        ):
            files = {
                leaf: {
                    "memory.usage_in_bytes": str(2 * GIB),
                    "memory.limit_in_bytes": str(10 * GIB),
                    "memory.stat": "rss 1\ncache 2\nmapped_file 3\n",
                    "memory.failcnt": "0",
                },
                parent: {
                    "memory.usage_in_bytes": str(6 * GIB),
                    "memory.limit_in_bytes": str(12 * GIB),
                    "memory.stat": "",
                },
            }
            for directory, values in files.items():
                for name, value in values.items():
                    with open(os.path.join(directory, name), "w") as f:
                        f.write(value)
            with (
                mock.patch.object(store, "_active_cgroup_v2_dirs", return_value=[]),
                mock.patch.object(
                    store, "_active_cgroup_v1_dirs", return_value=[leaf, parent]
                ),
            ):
                status = store._cgroup_memory_status()
        self.assertTrue(status["known"])
        self.assertEqual(status["version"], 1)
        self.assertEqual(status["max_available"], 6 * GIB)
        self.assertIsNone(status["high_available"])
        self.assertEqual(status["anon"], 1)
        self.assertEqual(status["file"], 2)
        self.assertEqual(status["file_mapped"], 3)

    def test_stream_loader_guards_lazy_cpu_allocation(self):
        class Layer:
            _moe_w2_stream_shapes = {"w": (2, 3)}
            _moe_w2_pending = {"w": 2}

        param = torch.nn.Parameter(
            torch.empty(0, dtype=torch.float32), requires_grad=False
        )
        with (
            mock.patch.object(store, "allocation_preflight") as pre,
            mock.patch.object(store, "allocation_postflight") as post,
        ):

            def inner(*args, **kwargs):
                self.assertTrue(pre.called)
                self.assertFalse(post.called)

            loader = _StreamLoader(Layer(), "w", inner)
            loader(param, torch.empty(0))
        pre.assert_called_once_with(
            "lazy expert staging w (2, 3)", 2 * 3 * param.element_size()
        )
        post.assert_called_once_with("lazy expert staging w (2, 3)")

    def test_stream_loader_postflights_partial_copy_failure(self):
        class Layer:
            _moe_w2_stream_shapes = {"w": (2, 3)}
            _moe_w2_pending = {"w": 2}

        param = torch.nn.Parameter(
            torch.empty(0, dtype=torch.float32), requires_grad=False
        )

        def fail_after_touch(param, loaded_weight, *args, **kwargs):
            param.data.fill_(1)
            raise RuntimeError("copy failed")

        loader = _StreamLoader(Layer(), "w", fail_after_touch)
        with (
            mock.patch.object(store, "allocation_preflight"),
            mock.patch.object(store, "allocation_postflight") as post,
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                loader(param, torch.empty(0))
        post.assert_called_once_with("lazy expert staging w (2, 3)")


class Mxfp4StreamBuildTest(unittest.TestCase):
    @staticmethod
    def _inner_loader(param, loaded_weight, *args, **kwargs):
        return True if kwargs.get("return_success") else None

    @staticmethod
    def _layer(index=0):
        layer = torch.nn.Module()
        layer.layer_name = f"model.layers.{index}.mlp.experts"
        return layer

    @staticmethod
    def _method():
        method = mxfp4.Mxfp4MoEMethod.__new__(mxfp4.Mxfp4MoEMethod)
        method.moe = SimpleNamespace(has_bias=False)
        return method

    def test_mxfp4_create_retains_no_all_layer_expert_buffers(self):
        method = self._method()
        layers = []
        next_key = 0

        def pack_miss(layer):
            nonlocal next_key
            layer._moe_w2_create_key = next_key
            next_key += 1
            return False

        with (
            mock.patch.object(cubit, "is_w2_layer", return_value=True),
            mock.patch.object(cubit, "plan_pack_skip", side_effect=pack_miss),
            mock.patch.object(cubit, "enabled", return_value=True),
            mock.patch.object(cubit, "_STREAM", True),
        ):
            for index in range(24):
                layer = self._layer(index)
                method.create_weights(
                    layer,
                    num_experts=2,
                    hidden_size=64,
                    intermediate_size_per_partition=32,
                    params_dtype=torch.bfloat16,
                    weight_loader=self._inner_loader,
                )
                layers.append(layer)

        big = (
            "w13_weight",
            "w13_weight_scale",
            "w2_weight",
            "w2_weight_scale",
        )
        self.assertEqual(next_key, len(layers))
        self.assertTrue(
            all(getattr(layer, name).numel() == 0 for layer in layers for name in big)
        )
        self.assertTrue(
            all(
                layer._moe_w2_stream_shapes["w13_weight"] == (2, 64, 32)
                for layer in layers
            )
        )

    def test_mxfp4_stream_waits_for_all_expected_loads_then_builds_once(self):
        layer = self._layer()
        layer._moe_w2_create_key = 9
        specs = {
            "w13_weight": (2, 4, 2),
            "w13_weight_scale": (2, 4, 1),
            "w2_weight": (2, 4, 2),
            "w2_weight_scale": (2, 4, 1),
        }
        for name, shape in specs.items():
            param = torch.nn.Parameter(
                torch.zeros(shape, dtype=torch.uint8), requires_grad=False
            )
            param.weight_loader = self._inner_loader
            layer.register_parameter(name, param)

        with (
            mock.patch.object(cubit, "enabled", return_value=True),
            mock.patch.object(cubit, "_STREAM", True),
            mock.patch.object(cubit, "build_layer_planes") as build,
            mock.patch.object(store, "allocation_preflight"),
            mock.patch.object(store, "allocation_postflight"),
        ):
            self.assertTrue(cubit.arm_stream_build(layer, checkpoint_format="mxfp4"))
            expected = dict(layer._moe_w2_pending)
            self.assertEqual(
                expected,
                {
                    "w13_weight": 4,
                    "w13_weight_scale": 4,
                    "w2_weight": 2,
                    "w2_weight_scale": 2,
                },
            )
            originals = tuple(layer._moe_w2_stream_orig)
            for name, count in expected.items():
                param = getattr(layer, name)
                for load_index in range(count):
                    param.weight_loader(
                        param,
                        torch.empty(0),
                        return_success=True,
                    )
                    if (name, load_index) != ("w2_weight_scale", count - 1):
                        build.assert_not_called()

        build.assert_called_once_with(layer, 9)
        self.assertTrue(layer._moe_w2_stream_built)
        self.assertTrue(all(value == 0 for value in layer._moe_w2_pending.values()))
        self.assertTrue(all(param.numel() == 0 for param in originals))

    def test_unknown_stream_checkpoint_format_fails_before_mutation(self):
        layer = self._layer()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros(2, 4, 2, dtype=torch.uint8), requires_grad=False
        )
        with (
            mock.patch.object(cubit, "enabled", return_value=True),
            mock.patch.object(cubit, "_STREAM", True),
            self.assertRaisesRegex(ValueError, "unsupported checkpoint"),
        ):
            cubit.arm_stream_build(layer, checkpoint_format="typo")
        self.assertEqual(layer.w13_weight.numel(), 16)

    def test_mxfp4_pack_skip_uses_stashed_shapes_before_stub_shapes(self):
        from vllm.model_executor.layers.quantization.utils import moe_w2_delta

        layer = SimpleNamespace(
            _moe_w2_pack_skip=True,
            _moe_w2_shapes=(2, 64, 64, 32, 32),
            w13_weight=torch.nn.Parameter(
                torch.empty(0, dtype=torch.uint8), requires_grad=False
            ),
            w13_weight_scale=torch.nn.Parameter(
                torch.empty(0, dtype=torch.uint8), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                torch.empty(0, dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(
                torch.empty(0, dtype=torch.uint8), requires_grad=False
            ),
        )
        with (
            mock.patch.object(cubit, "_ensure_ready", return_value=True),
            mock.patch.object(cubit, "_require_kernels") as kernels,
            mock.patch.object(cubit, "_try_skip_requant", return_value=True) as skip,
            mock.patch.object(moe_w2_delta, "enabled", return_value=False),
            mock.patch.object(moe_w2_delta, "base_enabled", return_value=True),
        ):
            cubit.build_layer_planes(layer, 7)
        kernels.assert_called_once_with(64, 32, need_w4=False)
        skip.assert_called_once_with(
            layer,
            7,
            2,
            64,
            64,
            32,
            32,
            ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"),
        )

    def test_mxfp4_process_does_not_rebuild_stream_built_layer(self):
        method = self._method()
        layer = SimpleNamespace(
            layer_name="model.layers.0.mlp.experts",
            _moe_w2_create_key=11,
            _moe_w2_stream_built=True,
        )
        with (
            mock.patch.object(cubit, "is_w2_layer", return_value=True),
            mock.patch.object(cubit, "build_layer_planes") as build,
        ):
            method.process_weights_after_loading(layer)
        build.assert_not_called()
        self.assertEqual(layer._moe_w2_key, 11)


class CacheControlTest(unittest.TestCase):
    def setUp(self):
        clear_pending_drops()

    def test_checkpoint_done_issues_dontneed_for_entire_file(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"x" * 4096)
            f.flush()
            with (
                mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
                mock.patch.object(store, "_mem_available_bytes", return_value=64 * GIB),
                mock.patch.object(
                    store,
                    "_cgroup_memory_status",
                    return_value=cgroup_limited(32 * GIB),
                ),
                mock.patch.object(os, "posix_fadvise") as advise,
            ):
                store.checkpoint_file_preflight(f.name)
                store.checkpoint_file_done(f.name)
                store.checkpoint_cleanup_pending()
        self.assertEqual(advise.call_count, 2)
        _, offset, length, advice = advise.call_args.args
        self.assertEqual((offset, length), (0, 0))
        self.assertEqual(advice, os.POSIX_FADV_DONTNEED)

    def test_next_shard_retries_pending_drop_without_clearing_final_retry(self):
        with tempfile.TemporaryDirectory() as root:
            first = os.path.join(root, "first.safetensors")
            second = os.path.join(root, "second.safetensors")
            for path in (first, second):
                with open(path, "wb") as handle:
                    handle.write(b"x" * 4096)
            calls = []

            def drop(path, label):
                calls.append((path, label))
                return True

            with (
                mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
                mock.patch.object(store, "_drop_path_page_cache", side_effect=drop),
                mock.patch.object(store, "_memory_preflight"),
            ):
                store.checkpoint_file_done(first)
                self.assertIn(first, store._pending_checkpoint_drops)
                store.checkpoint_file_preflight(second)
                self.assertIn(first, store._pending_checkpoint_drops)
                store.checkpoint_cleanup_pending()

        self.assertEqual([path for path, _ in calls], [first, first, first])
        self.assertNotIn(first, store._pending_checkpoint_drops)

    def test_pending_retry_failure_refuses_next_shard_before_new_preflight(self):
        with tempfile.TemporaryDirectory() as root:
            first = os.path.join(root, "first.safetensors")
            second = os.path.join(root, "second.safetensors")
            for path in (first, second):
                with open(path, "wb") as handle:
                    handle.write(b"x" * 4096)
            with (
                mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
                mock.patch.object(store, "_memory_preflight") as preflight,
                mock.patch.object(
                    store,
                    "_drop_path_page_cache",
                    side_effect=[True, RuntimeError("retry failed")],
                ),
            ):
                store.checkpoint_file_done(first)
                preflight.reset_mock()
                with self.assertRaisesRegex(RuntimeError, "retry failed"):
                    store.checkpoint_file_preflight(second)

        preflight.assert_not_called()
        self.assertIn(first, store._pending_checkpoint_drops)

    def test_final_cleanup_drops_every_path_before_postflight_and_clear(self):
        paths = ("/test/first.safetensors", "/test/second.safetensors")
        with store._pending_checkpoint_lock:
            store._pending_checkpoint_drops.update(paths)
        with (
            mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
            mock.patch.object(store, "_drop_path_page_cache") as drop,
            mock.patch.object(
                store,
                "_memory_preflight",
                side_effect=RuntimeError("postflight floor"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "postflight floor"):
                store.checkpoint_cleanup_pending()

        self.assertEqual([call.args[0] for call in drop.call_args_list], sorted(paths))
        self.assertTrue(set(paths).issubset(store._pending_checkpoint_drops))

    def test_invalid_cache_mode_fails_closed(self):
        env = {**SAFETY_ENV, "VLLM_MOE_W2_CACHE_CONTROL": "typo"}
        with mock.patch.dict(os.environ, env, clear=False):
            with self.assertRaisesRegex(ValueError, "CACHE_CONTROL"):
                store._require_cache_control()

    def test_pack_presence_requires_exact_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as root:
            sidecar = os.path.join(root, "base.rank0of1.json")
            meta = {
                "version": store._PACK_VERSION,
                "tag": "base",
                "E": 2,
                "n_layers": 3,
                "slot_bytes": 1024,
                "stride": 4096,
                "build_identity": {
                    "operator_id": "checkpoint-a",
                    "zero_mode": "auto",
                },
                "layers": [0],
            }
            with open(sidecar, "w") as f:
                json.dump(meta, f)
            env = {
                **SAFETY_ENV,
                "VLLM_MOE_W2_STORE_DIR": root,
                "VLLM_MOE_W2_PACK_ID": "checkpoint-a",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(store, "_rank_suffix", return_value="rank0of1"),
            ):
                self.assertTrue(
                    store.pack_has_layer(
                        "base", 0, n_layers=3, n_experts=2, slot_bytes=1024
                    )
                )
                os.environ["VLLM_MOE_W2_PACK_ID"] = "checkpoint-b"
                self.assertFalse(
                    store.pack_has_layer(
                        "base", 0, n_layers=3, n_experts=2, slot_bytes=1024
                    )
                )

    def test_pack_identity_lookup_failure_is_fail_closed(self):
        from vllm.model_executor.layers.quantization.utils import (
            moe_w2_planes_cache,
        )

        env = {**SAFETY_ENV, "VLLM_MOE_W2_PACK_ID": ""}
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.object(
                moe_w2_planes_cache,
                "cache_identity",
                side_effect=RuntimeError("no config"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot identify.*checkpoint"):
                store._pack_build_identity()

    def test_required_fadvise_failure_raises_and_remains_retryable(self):
        with tempfile.NamedTemporaryFile() as f:
            with (
                mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
                mock.patch.object(store, "_mem_available_bytes", return_value=64 * GIB),
                mock.patch.object(
                    store,
                    "_cgroup_memory_status",
                    return_value=cgroup_limited(32 * GIB),
                ),
                mock.patch.object(
                    os, "posix_fadvise", side_effect=OSError("unsupported")
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "could not evict"):
                    store.checkpoint_file_done(f.name)
        self.assertIn(f.name, store._pending_checkpoint_drops)

    def test_preheat_postflight_failure_is_not_downgraded(self):
        with tempfile.TemporaryDirectory() as root:
            pack_path = os.path.join(root, "base.pack")
            heat_path = pack_path + ".heat.json"
            with open(pack_path, "wb") as f:
                f.write(b"x" * 4096)
            with open(heat_path, "w") as f:
                json.dump({"keys": [[0, 0]]}, f)

            tier = store.TieredPackStore.__new__(store.TieredPackStore)
            tier._heat_path = heat_path
            tier.path = pack_path
            tier._fd = os.open(pack_path, os.O_RDONLY)
            tier._present = {0}
            tier.E = 1
            tier.n_arena = 1
            tier.slot_bytes = 4096
            tier.stride = 4096
            tier._pool = mock.Mock()
            tier._pool.map.return_value = []
            tier._pos = {}
            tier._owner_pair = [None]
            tier._last = [0]
            tier._free = [0]
            tier._clock = 0
            try:
                with (
                    mock.patch.object(store, "_fadvise_dontneed", return_value=True),
                    mock.patch.object(
                        store,
                        "_memory_preflight",
                        side_effect=[{}, RuntimeError("postflight floor")],
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "postflight floor"):
                        tier._preheat()
            finally:
                os.close(tier._fd)

    def _bare_pack(self, root: str):
        pack = store.MmapPackStore.__new__(store.MmapPackStore)
        pack.slot_bytes = 4
        pack.E = 1
        pack.n_layers = 1
        pack.stride = 4096
        pack.path = os.path.join(root, "base.pack")
        pack._sidecar_path = os.path.join(root, "base.json")
        pack._fd = os.open(pack.path, os.O_RDWR | os.O_CREAT, 0o600)
        pack._wbuf = None
        pack._present = set()
        pack._meta = {
            "version": store._PACK_VERSION,
            "tag": "base",
            "E": 1,
            "n_layers": 1,
            "slot_bytes": 4,
            "stride": 4096,
            "layers": [],
        }
        pack._write_cache_drop_calls = 0
        pack._write_cache_drop_bytes = 0
        return pack

    def test_partial_pack_write_cleans_extent_without_publishing_layer(self):
        with tempfile.TemporaryDirectory() as root:
            pack = self._bare_pack(root)
            try:
                with (
                    mock.patch.object(store, "_memory_preflight"),
                    mock.patch.object(os, "pwrite", side_effect=[512, 0]),
                    mock.patch.object(os, "fdatasync") as sync,
                    mock.patch.object(
                        store, "_fadvise_dontneed", return_value=True
                    ) as drop,
                ):
                    with self.assertRaisesRegex(OSError, "short write"):
                        pack.add_layer(0, (torch.ones(1, 4, dtype=torch.uint8),))
                drop.assert_called_once_with(
                    pack._fd, 0, 4096, f"failed pack {pack.path} layer 0"
                )
                sync.assert_called_once_with(pack._fd)
                self.assertIsNone(pack._wbuf)
                self.assertNotIn(0, pack._present)
                self.assertFalse(os.path.exists(pack._sidecar_path))
            finally:
                os.close(pack._fd)

    def test_pack_fsync_failure_cleans_without_publishing_layer(self):
        with tempfile.TemporaryDirectory() as root:
            pack = self._bare_pack(root)
            try:
                with (
                    mock.patch.object(store, "_memory_preflight"),
                    mock.patch.object(
                        os, "pwrite", side_effect=lambda fd, data, off: len(data)
                    ),
                    mock.patch.object(
                        os, "fdatasync", side_effect=OSError("sync failed")
                    ),
                    mock.patch.object(
                        store, "_fadvise_dontneed", return_value=True
                    ) as drop,
                ):
                    with self.assertRaisesRegex(OSError, "sync failed"):
                        pack.add_layer(0, (torch.ones(1, 4, dtype=torch.uint8),))
                drop.assert_called_once_with(
                    pack._fd, 0, 4096, f"failed pack {pack.path} layer 0"
                )
                self.assertIsNone(pack._wbuf)
                self.assertNotIn(0, pack._present)
                self.assertFalse(os.path.exists(pack._sidecar_path))
            finally:
                os.close(pack._fd)

    def test_required_pack_fadvise_failure_never_publishes_layer(self):
        with tempfile.TemporaryDirectory() as root:
            pack = self._bare_pack(root)
            try:
                with (
                    mock.patch.object(store, "_memory_preflight"),
                    mock.patch.object(
                        os, "pwrite", side_effect=lambda fd, data, off: len(data)
                    ),
                    mock.patch.object(os, "fdatasync"),
                    mock.patch.object(
                        store,
                        "_fadvise_dontneed",
                        side_effect=RuntimeError("cache drop failed"),
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "cache drop failed"):
                        pack.add_layer(0, (torch.ones(1, 4, dtype=torch.uint8),))
                self.assertIsNone(pack._wbuf)
                self.assertNotIn(0, pack._present)
                self.assertFalse(os.path.exists(pack._sidecar_path))
            finally:
                os.close(pack._fd)


class WeightIteratorIntegrationTest(unittest.TestCase):
    def setUp(self):
        clear_pending_drops()

    def _checkpoint(self, root: str, name: str) -> str:
        path = os.path.join(root, name)
        save_file({"weight": torch.arange(8, dtype=torch.float32)}, path)
        return path

    def test_default_iterator_guards_and_releases_completed_shard(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._checkpoint(root, "one.safetensors")
            with (
                mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
                mock.patch.object(store, "checkpoint_file_preflight") as pre,
                mock.patch.object(store, "checkpoint_file_done") as done,
                mock.patch.object(
                    store,
                    "guarded_checkpoint_clone",
                    side_effect=lambda label, tensor: tensor,
                ) as clone,
            ):
                rows = list(
                    weight_utils.safetensors_weights_iterator(
                        [path], use_tqdm_on_load=False
                    )
                )
        self.assertEqual([name for name, _ in rows], ["weight"])
        pre.assert_called_once_with(path, 0)
        done.assert_called_once_with(path)
        self.assertEqual(clone.call_count, 1)

    def test_generator_close_runs_cache_release_finally(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._checkpoint(root, "close.safetensors")
            with (
                mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
                mock.patch.object(store, "checkpoint_file_preflight") as pre,
                mock.patch.object(store, "checkpoint_file_done") as done,
                mock.patch.object(
                    store,
                    "guarded_checkpoint_clone",
                    side_effect=lambda label, tensor: tensor,
                ) as clone,
            ):
                iterator = weight_utils.safetensors_weights_iterator(
                    [path], use_tqdm_on_load=False
                )
                item = next(iterator)
                iterator.close()
        pre.assert_called_once_with(path, 0)
        done.assert_called_once_with(path)
        self.assertEqual(clone.call_count, 1)
        self.assertEqual(item[0], "weight")

    def test_consumer_exception_runs_cache_release_finally(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._checkpoint(root, "throw.safetensors")
            with (
                mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
                mock.patch.object(store, "checkpoint_file_preflight"),
                mock.patch.object(store, "checkpoint_file_done") as done,
            ):

                def consume_and_fail():
                    iterator = weight_utils.safetensors_weights_iterator(
                        [path], use_tqdm_on_load=False
                    )
                    try:
                        for _name, _loaded_weight in iterator:
                            raise RuntimeError("consumer failed")
                    finally:
                        iterator.close()

                with self.assertRaisesRegex(RuntimeError, "consumer failed"):
                    consume_and_fail()
        done.assert_called_once_with(path)

    def test_explicit_prefetch_is_rejected_before_background_read(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._checkpoint(root, "prefetch.safetensors")
            with mock.patch.dict(os.environ, SAFETY_ENV, clear=False):
                iterator = weight_utils.safetensors_weights_iterator(
                    [path], use_tqdm_on_load=False, safetensors_load_strategy="prefetch"
                )
                with self.assertRaisesRegex(RuntimeError, "prefetch is unsafe"):
                    next(iterator)

    def test_multithread_request_becomes_guarded_sequential_load(self):
        with tempfile.TemporaryDirectory() as root:
            paths = [self._checkpoint(root, f"{i}.safetensors") for i in range(2)]
            with (
                mock.patch.dict(os.environ, SAFETY_ENV, clear=False),
                mock.patch.object(store, "checkpoint_file_preflight") as pre,
                mock.patch.object(store, "checkpoint_file_done") as done,
                mock.patch.object(
                    store,
                    "guarded_checkpoint_clone",
                    side_effect=lambda label, tensor: tensor,
                ) as clone,
            ):
                rows = list(
                    weight_utils.multi_thread_safetensors_weights_iterator(
                        paths, use_tqdm_on_load=False, max_workers=8
                    )
                )
        self.assertEqual(len(rows), 2)
        self.assertEqual(pre.call_count, 2)
        self.assertEqual(done.call_count, 2)
        self.assertEqual(clone.call_count, 2)

    def test_default_loader_retries_cleanup_after_consumer_unwinds(self):
        class ClosableIterator:
            def __init__(self):
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self):
                return "weight", torch.ones(1)

            def close(self):
                self.closed = True

        weights = ClosableIterator()
        loader = DefaultModelLoader.__new__(DefaultModelLoader)
        loader._init_ep_weight_filter = mock.Mock()
        loader.get_all_weights = mock.Mock(return_value=weights)
        model = mock.Mock()

        def consume_one_and_fail(iterator):
            next(iterator)
            raise RuntimeError("model consumer failed")

        model.load_weights.side_effect = consume_one_and_fail
        model_config = SimpleNamespace(quantization=None)
        with mock.patch.object(store, "checkpoint_cleanup_pending") as cleanup:
            with self.assertRaisesRegex(RuntimeError, "model consumer failed"):
                loader.load_weights(model, model_config)
        self.assertTrue(weights.closed)
        cleanup.assert_called_once_with()

    def test_default_loader_closes_underlying_iterator(self):
        class ClosableIterator:
            def __init__(self):
                self.done = False
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self):
                if self.done:
                    raise StopIteration
                self.done = True
                return "weight", torch.ones(1)

            def close(self):
                self.closed = True

        inner = ClosableIterator()
        loader = DefaultModelLoader.__new__(DefaultModelLoader)
        loader.load_config = SimpleNamespace(
            load_format="safetensors",
            model_loader_extra_config={},
            use_tqdm_on_load=False,
            safetensors_load_strategy="lazy",
            safetensors_prefetch_num_threads=1,
            safetensors_prefetch_block_size=1,
        )
        loader.local_expert_ids = None
        loader.counter_before_loading_weights = 0.0
        loader._prepare_weights = mock.Mock(
            return_value=("/tmp", ["test.safetensors"], True)
        )
        source = DefaultModelLoader.Source("/tmp/model", None, prefix="p.")
        with mock.patch.object(
            default_loader_module, "safetensors_weights_iterator", return_value=inner
        ):
            outer = loader._get_weights_iterator(source)
            self.assertEqual(next(outer)[0], "p.weight")
            outer.close()
        self.assertTrue(inner.closed)

    def test_default_loader_rejects_non_safetensors_checkpoint(self):
        loader = DefaultModelLoader.__new__(DefaultModelLoader)
        loader.load_config = SimpleNamespace(
            load_format="pt", model_loader_extra_config={}
        )
        loader.local_expert_ids = None
        loader.counter_before_loading_weights = 0.0
        loader._prepare_weights = mock.Mock(return_value=("/tmp", ["model.bin"], False))
        source = DefaultModelLoader.Source("/tmp/model", None)
        with mock.patch.dict(os.environ, SAFETY_ENV, clear=False):
            with self.assertRaisesRegex(RuntimeError, "non-safetensors"):
                loader._get_weights_iterator(source)

    def test_mutated_torchao_strategy_cannot_bypass_multithread_guard(self):
        loader = DefaultModelLoader.__new__(DefaultModelLoader)
        loader.load_config = SimpleNamespace(
            load_format="safetensors",
            model_loader_extra_config={"enable_multithread_load": True},
            safetensors_load_strategy="torchao",
        )
        loader.local_expert_ids = None
        loader.counter_before_loading_weights = 0.0
        loader._prepare_weights = mock.Mock(
            return_value=("/tmp", ["model.safetensors"], True)
        )
        source = DefaultModelLoader.Source("/tmp/model", None)
        with mock.patch.dict(os.environ, SAFETY_ENV, clear=False):
            with self.assertRaisesRegex(RuntimeError, "multi-thread.*torchao"):
                loader._get_weights_iterator(source)

    def test_loader_factory_rejects_unguarded_format(self):
        with mock.patch.dict(os.environ, SAFETY_ENV, clear=False):
            with self.assertRaisesRegex(RuntimeError, "bypasses.*cache-safety"):
                get_model_loader(SimpleNamespace(load_format="runai_streamer"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
