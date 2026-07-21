"""Focused tests for bounded MXFP4 W2 checkpoint staging.

Run inside the patched serving image; no GPU or checkpoint is required:

    python3 /serve/tools/test_mxfp4_stream_safety.py
"""

import ast
import inspect
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils import moe_w2_cubit as cubit
from vllm.model_executor.layers.quantization.utils import moe_w2_delta, moe_w2_store

BIG_PARAMS = (
    "w13_weight",
    "w13_weight_scale",
    "w2_weight",
    "w2_weight_scale",
)


def _loader(param, loaded_weight, *args, **kwargs):
    del loaded_weight, args
    return True if kwargs.get("return_success") else None


def _param(shape):
    param = torch.nn.Parameter(torch.empty(shape), requires_grad=False)
    param.weight_loader = _loader
    return param


class _Layer(torch.nn.Module):
    def __init__(self, layer_name="model.layers.3.mlp.experts"):
        super().__init__()
        self.layer_name = layer_name


class ArmStreamBuildTest(unittest.TestCase):
    def setUp(self):
        self.layer = _Layer()
        self.layer.register_parameter("w13_weight", _param((3, 4, 2)))
        self.layer.register_parameter("w13_weight_scale", _param((3, 4, 1)))
        self.layer.register_parameter("w2_weight", _param((3, 2, 2)))
        self.layer.register_parameter("w2_weight_scale", _param((3, 2, 1)))

    def test_mxfp4_arms_four_exact_counters_and_correct_builder(self):
        with (
            mock.patch.object(cubit, "_STREAM", True),
            mock.patch.object(cubit, "enabled", return_value=True),
        ):
            self.assertTrue(
                cubit.arm_stream_build(self.layer, checkpoint_format="mxfp4")
            )

        self.assertEqual(
            self.layer._moe_w2_pending,
            {
                "w13_weight": 6,
                "w13_weight_scale": 6,
                "w2_weight": 3,
                "w2_weight_scale": 3,
            },
        )
        self.assertIs(self.layer._moe_w2_stream_builder, cubit.build_layer_planes)
        self.assertEqual(self.layer._moe_w2_stream_format, "mxfp4")
        self.assertTrue(
            all(getattr(self.layer, name).numel() == 0 for name in BIG_PARAMS)
        )

    def test_nvfp4_default_keeps_scale2_completion_barrier(self):
        self.layer.register_parameter("w13_weight_scale_2", _param((3, 2)))
        self.layer.register_parameter("w2_weight_scale_2", _param((3,)))
        with (
            mock.patch.object(cubit, "_STREAM", True),
            mock.patch.object(cubit, "enabled", return_value=True),
        ):
            self.assertTrue(cubit.arm_stream_build(self.layer))
        self.assertEqual(self.layer._moe_w2_pending["w13_weight_scale_2"], 6)
        self.assertEqual(self.layer._moe_w2_pending["w2_weight_scale_2"], 3)
        self.assertIs(self.layer._moe_w2_stream_builder, cubit.build_layer_planes_nvfp4)

    def test_unknown_format_fails_before_mutating_layer(self):
        before = {name: getattr(self.layer, name).numel() for name in BIG_PARAMS}
        with (
            mock.patch.object(cubit, "_STREAM", True),
            mock.patch.object(cubit, "enabled", return_value=True),
            self.assertRaisesRegex(ValueError, "unsupported checkpoint format"),
        ):
            cubit.arm_stream_build(self.layer, checkpoint_format="typo")
        self.assertEqual(
            before,
            {name: getattr(self.layer, name).numel() for name in BIG_PARAMS},
        )
        self.assertFalse(hasattr(self.layer, "_moe_w2_pending"))

    def test_missing_loader_falls_back_without_partial_mutation(self):
        del self.layer.w2_weight_scale.weight_loader
        before = {name: getattr(self.layer, name).numel() for name in BIG_PARAMS}
        with (
            mock.patch.object(cubit, "_STREAM", True),
            mock.patch.object(cubit, "enabled", return_value=True),
        ):
            self.assertFalse(
                cubit.arm_stream_build(self.layer, checkpoint_format="mxfp4")
            )
        self.assertEqual(
            before,
            {name: getattr(self.layer, name).numel() for name in BIG_PARAMS},
        )
        self.assertFalse(hasattr(self.layer, "_moe_w2_pending"))

    def test_mxfp4_lazy_allocation_is_preflighted_before_copy(self):
        param = self.layer.w13_weight
        observed = []

        def guarded_loader(param, loaded_weight, *args, **kwargs):
            del param, loaded_weight, args
            observed.append((pre.called, post.called))
            return True if kwargs.get("return_success") else None

        param.weight_loader = guarded_loader
        with (
            mock.patch.object(cubit, "_STREAM", True),
            mock.patch.object(cubit, "enabled", return_value=True),
        ):
            cubit.arm_stream_build(self.layer, checkpoint_format="mxfp4")
        with (
            mock.patch.object(moe_w2_store, "allocation_preflight") as pre,
            mock.patch.object(moe_w2_store, "allocation_postflight") as post,
        ):
            param.weight_loader(param, torch.empty(0), return_success=True)
        self.assertEqual(observed, [(True, False)])
        pre.assert_called_once_with(
            "lazy expert staging w13_weight (3, 4, 2)",
            3 * 4 * 2 * param.element_size(),
        )
        post.assert_called_once_with("lazy expert staging w13_weight (3, 4, 2)")

    def test_last_mxfp4_tensor_dispatches_once_and_drops_originals(self):
        self.layer._moe_w2_create_key = 9
        with (
            mock.patch.object(cubit, "_STREAM", True),
            mock.patch.object(cubit, "enabled", return_value=True),
        ):
            cubit.arm_stream_build(self.layer, checkpoint_format="mxfp4")
        builder = mock.Mock()
        self.layer._moe_w2_stream_builder = builder
        originals = tuple(self.layer._moe_w2_stream_orig)
        with (
            mock.patch.object(moe_w2_store, "allocation_preflight") as pre,
            mock.patch.object(moe_w2_store, "allocation_postflight") as post,
        ):
            for name, count in tuple(self.layer._moe_w2_pending.items()):
                param = getattr(self.layer, name)
                for _ in range(count):
                    param.weight_loader(param, torch.empty(0), return_success=True)
        builder.assert_called_once_with(self.layer, 9)
        self.assertTrue(self.layer._moe_w2_stream_built)
        self.assertTrue(all(param.numel() == 0 for param in originals))
        self.assertEqual(pre.call_count, len(BIG_PARAMS))
        self.assertEqual(post.call_count, len(BIG_PARAMS))


class Mxfp4MethodIntegrationTest(unittest.TestCase):
    def test_many_constructed_layers_retain_zero_expert_bytes(self):
        method = object.__new__(Mxfp4MoEMethod)
        method.moe = SimpleNamespace(has_bias=False)
        layers = []
        next_key = 0

        def no_pack(layer):
            nonlocal next_key
            layer._moe_w2_create_key = next_key
            next_key += 1
            return False

        with (
            mock.patch.object(cubit, "is_w2_layer", return_value=True),
            mock.patch.object(cubit, "plan_pack_skip", side_effect=no_pack),
            mock.patch.object(cubit, "_STREAM", True),
            mock.patch.object(cubit, "enabled", return_value=True),
        ):
            for _ in range(32):
                layer = _Layer()
                method.create_weights(
                    layer,
                    num_experts=3,
                    hidden_size=32,
                    intermediate_size_per_partition=64,
                    params_dtype=torch.bfloat16,
                    weight_loader=_loader,
                )
                layers.append(layer)
                self.assertEqual(
                    sum(getattr(layer, name).numel() for name in BIG_PARAMS),
                    0,
                )

        retained = sum(
            getattr(layer, name).numel() * getattr(layer, name).element_size()
            for layer in layers
            for name in BIG_PARAMS
        )
        self.assertEqual(retained, 0)
        self.assertEqual(
            [layer._moe_w2_create_key for layer in layers], list(range(32))
        )

    def test_process_does_not_rebuild_stream_built_layer(self):
        method = object.__new__(Mxfp4MoEMethod)
        layer = _Layer()
        layer._moe_w2_create_key = 17
        layer._moe_w2_stream_built = True
        with (
            mock.patch.object(cubit, "is_w2_layer", return_value=True),
            mock.patch.object(cubit, "build_layer_planes") as build,
        ):
            method.process_weights_after_loading(layer)
        build.assert_not_called()
        self.assertEqual(layer._moe_w2_key, 17)


class PackSkipTest(unittest.TestCase):
    def test_mxfp4_pack_skip_uses_stashed_shapes_before_stub_data(self):
        layer = _Layer()
        layer._moe_w2_pack_skip = True
        layer._moe_w2_shapes = (3, 32, 16, 16, 16)
        for name in BIG_PARAMS:
            layer.register_parameter(name, _param((0,)))
        with (
            mock.patch.object(cubit, "_ensure_ready", return_value=True),
            mock.patch.object(cubit, "_require_kernels") as require,
            mock.patch.object(moe_w2_delta, "enabled", return_value=False),
            mock.patch.object(moe_w2_delta, "base_enabled", return_value=True),
            mock.patch.object(cubit, "_try_skip_requant", return_value=True) as skip,
        ):
            cubit.build_layer_planes(layer, 4)
        require.assert_called_once_with(16, 16, need_w4=False)
        skip.assert_called_once_with(layer, 4, 3, 32, 16, 16, 16, BIG_PARAMS)


class SourceWiringTest(unittest.TestCase):
    def test_mxfp4_create_wires_planner_before_format_specific_stream(self):
        source = Path(inspect.getsourcefile(Mxfp4MoEMethod)).read_text()
        tree = ast.parse(source)
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Mxfp4MoEMethod"
        )
        create = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "create_weights"
        )
        calls = [node for node in ast.walk(create) if isinstance(node, ast.Call)]

        def method_name(call):
            return call.func.attr if isinstance(call.func, ast.Attribute) else None

        planner = next(call for call in calls if method_name(call) == "plan_pack_skip")
        stream = next(call for call in calls if method_name(call) == "arm_stream_build")
        self.assertLess(planner.lineno, stream.lineno)
        fmt = next(kw.value for kw in stream.keywords if kw.arg == "checkpoint_format")
        self.assertIsInstance(fmt, ast.Constant)
        self.assertEqual(fmt.value, "mxfp4")


if __name__ == "__main__":
    os.environ.setdefault("VLLM_MOE_W2", "1")
    unittest.main(verbosity=2)
