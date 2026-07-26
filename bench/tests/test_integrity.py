import copy
import importlib.util
import json
import os
import random
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

RUNNER = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER))

import common  # noqa: E402
import integrity  # noqa: E402
import probes  # noqa: E402
from serverctl import Server, ServerFailed  # noqa: E402

SERVE_RECIPE_PATH = Path(__file__).resolve().parents[2] / "docker" / "serve_recipe.py"
_serve_spec = importlib.util.spec_from_file_location(
    "serve_recipe_for_test", SERVE_RECIPE_PATH)
serve_recipe = importlib.util.module_from_spec(_serve_spec)
_serve_spec.loader.exec_module(serve_recipe)


class IntegrityTests(unittest.TestCase):
    def test_canonical_hash_vector(self):
        value = {"b": [True, None, "x"], "a": 1}
        self.assertEqual(
            integrity.canonical_bytes(value),
            b'{"a":1,"b":[true,null,"x"]}',
        )
        self.assertEqual(
            integrity.canonical_sha256(value),
            "eca8cfb31ab74533e1eb2f4c74d2d55dfe3c79ac704787e54be8647ea7777eb1",
        )

    def test_effective_suite_normalizes_and_honors_disabled(self):
        suite = {
            "allow_recipe_overrides": True,
            "probes": [
                {"kind": "decode"},
                {"kind": "quality", "id": "think", "enabled": False},
            ],
        }
        recipe = {"id": "m/r", "suite_params": {"decode": {"runs": 3}}}
        got = common.merge_suite(suite, recipe)
        self.assertEqual(got[0]["enabled"], True)
        self.assertEqual(got[0]["runs"], 3)
        self.assertEqual(got[1]["enabled"], False)

    def test_non_boolean_enabled_override_is_rejected(self):
        suite = {
            "allow_recipe_overrides": True,
            "probes": [{"kind": "decode"}],
        }
        recipe = {
            "id": "m/r",
            "suite_params": {"decode": {"enabled": "false"}},
        }
        with self.assertRaisesRegex(ValueError, "enabled must be boolean"):
            common.merge_suite(suite, recipe)

    def test_probe_acceptance_rejects_failed_correctness_signals(self):
        self.assertFalse(probes.probe_succeeded(
            "needle", {"all_pass": False}))
        self.assertFalse(probes.probe_succeeded(
            "quality", {"runs": 2, "of": 2}))
        quality = {
            "runs": 2,
            "of": 2,
            "artifact": "raw.json",
            "artifact_sha256": "a" * 64,
            "vs_baseline": {
                "mcnemar_p": 0.25,
                "token_p50_delta_pct": -2.0,
                "token_p90_delta_pct": 2.5,
                "extra_truncated_no_answer": 1,
            },
        }
        self.assertTrue(probes.probe_succeeded("quality", quality))
        native_gate = {
            "min_mcnemar_p": 0.05,
            "max_abs_token_p50_delta_pct": 5,
            "max_abs_token_p90_delta_pct": 5,
            "max_extra_truncated_no_answer": 1,
        }
        self.assertTrue(probes.probe_succeeded(
            "quality", quality, native_gate))
        for field, value in (
            ("mcnemar_p", 0.01),
            ("token_p50_delta_pct", -5.1),
            ("token_p90_delta_pct", 5.1),
            ("extra_truncated_no_answer", 2),
        ):
            failed = copy.deepcopy(quality)
            failed["vs_baseline"][field] = value
            self.assertFalse(probes.probe_succeeded(
                "quality", failed, native_gate), field)

    def test_recipe_semantic_change_changes_binding(self):
        recipe = {
            "id": "m/r",
            "model": "M",
            "requires": {},
            "env": {"A": "1"},
        }
        registry = {
            "M": {
                "hf_repo": "org/model",
                "revision": "a" * 40,
            }
        }
        probes_cfg = [{"kind": "decode", "enabled": True}]
        before = integrity.result_bindings(
            recipe, "standard", probes_cfg, registry)
        changed = copy.deepcopy(recipe)
        changed["env"]["A"] = "2"
        after = integrity.result_bindings(
            changed, "standard", probes_cfg, registry)
        self.assertNotEqual(
            before["recipe_sha256"], after["recipe_sha256"])

    def test_binding_snapshot_is_self_verifying_and_release_scoped(self):
        recipe = {"id": "m/r", "model": "M", "requires": {}}
        registry = {
            "M": {"hf_repo": "org/model", "revision": "a" * 40}
        }
        box = {"id": "box", "runtime": "venv"}
        inputs = integrity.result_bindings(
            recipe, "standard", [{"kind": "decode", "enabled": True}],
            registry, release="r1", box=box)
        self.assertEqual(integrity.self_binding_errors(inputs), [])
        self.assertEqual(inputs["release"], "r1")
        tampered = copy.deepcopy(inputs)
        tampered["recipe_snapshot"]["id"] = "other"
        self.assertTrue(integrity.self_binding_errors(tampered))

    def test_schema2_result_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "result.json")
            common.reserve_result(path)
            common.write_result(path, {"schema": 2, "value": 1})
            with self.assertRaises(FileExistsError):
                common.reserve_result(path)
                common.write_result(path, {"schema": 2, "value": 2})
            self.assertEqual(json.loads(Path(path).read_text())["value"], 1)

    def test_artifact_path_cannot_escape_result_directory(self):
        with tempfile.TemporaryDirectory() as d:
            result = os.path.join(d, "box", "result.json")
            os.makedirs(os.path.dirname(result))
            with self.assertRaises(ValueError):
                integrity.artifact_path(result, "../../outside.json")

    def test_fresh_release_gate_rejects_imported_partial_and_drift(self):
        expected = {
            "recipe_sha256": "expected",
            "model_manifests": {"M": "manifest"},
        }
        base = {
            "schema": 2,
            "complete": True,
            "provenance": "live",
            "status": "ok",
            "suite": "standard",
            "inputs": expected,
            "env_fingerprint": {
                "runtime": "venv",
                "moet_dirty": False,
                "patch_sha256": "abc",
                "cubins": "ghi",
                "moet_sha": "jkl",
                "vllm_tree": {"dirty": False, "sha": "def"},
            },
        }
        runtime_manifest = {
            "patch_sha256": "abc",
            "cubins": "ghi",
            "moet_sha": "jkl",
            "vllm_sha": "def",
            "model_manifests": {"M": "manifest"},
        }
        self.assertEqual(
            integrity.strict_gate_errors(
                base, "standard", expected, runtime_manifest), [])
        for field, value in (
            ("provenance", "imported"),
            ("status", "partial"),
            ("inputs", {"recipe_sha256": "stale"}),
        ):
            bad = copy.deepcopy(base)
            bad[field] = value
            self.assertTrue(
                integrity.strict_gate_errors(
                    bad, "standard", expected, runtime_manifest))

    def test_occupied_port_fails_before_log_creation(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "nested", "server.log")
            server = Server(
                ["does-not-run"],
                {},
                log_path,
                port,
                [],
            )
            with self.assertRaisesRegex(ServerFailed, "already occupied"):
                server.start(1)
            self.assertFalse(os.path.exists(log_path))
        listener.close()

    def test_batch_probe_does_not_mutate_global_rng(self):
        prompts = []

        def fake_completion(_base, _model, prompt, _max_tokens, **_kwargs):
            prompts.append(prompt)
            return {"usage": {"completion_tokens": 10}}, 0.01

        state = random.getstate()
        with mock.patch.object(probes, "_completion", fake_completion):
            result = probes.batch_decode(
                "http://unused", "model", concurrency=(2,), runs=2,
                log=lambda _msg: None)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(len(prompts), 4)
        self.assertIn("2", result["levels"])

    def test_model_download_requires_revision_and_completion_marker(self):
        registry = {
            "M": {
                "hf_repo": "org/model",
                "revision": "a" * 40,
                "approx_gb": 1,
            }
        }
        calls = []

        def snapshot_download(*, repo_id, revision, local_dir):
            calls.append((repo_id, revision))
            os.makedirs(local_dir, exist_ok=True)
            Path(local_dir, "config.json").write_text("{}")
            Path(local_dir, "model.safetensors").write_bytes(b"weights")

        fake_hf = types.SimpleNamespace(snapshot_download=snapshot_download)
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(serve_recipe, "MODELS_DIR", d), \
                mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            # A config-only partial directory is not accepted.
            partial = Path(d, "M")
            partial.mkdir()
            Path(partial, "config.json").write_text("{}")
            path = serve_recipe.ensure_model("M", registry, do_print=False)
            self.assertEqual(calls, [("org/model", "a" * 40)])
            marker = json.loads(
                Path(path, ".vllm-moet-snapshot.json").read_text())
            self.assertEqual(marker["revision"], "a" * 40)
            serve_recipe.ensure_model("M", registry, do_print=False)
            self.assertEqual(len(calls), 1)
            Path(path, "model.safetensors").write_bytes(b"changed")
            serve_recipe.ensure_model("M", registry, do_print=False)
            self.assertEqual(len(calls), 2)
            Path(path, "model.safetensors").unlink()
            serve_recipe.ensure_model("M", registry, do_print=False)
            self.assertEqual(len(calls), 3)

    def test_quality_probe_rejects_historical_baseline_by_default(self):
        baseline = {
            "old": {
                "strict": False,
                "file": "unused.json",
            }
        }
        with mock.patch.object(
                common, "load_baseline_registry", return_value=baseline):
            with self.assertRaisesRegex(RuntimeError, "historical evidence"):
                probes.quality(
                    "http://127.0.0.1:1",
                    "served",
                    profile="gpqa-diamond",
                    baseline="old",
                    tool=__file__,
                    expected_tool_sha256=integrity.sha256_file(__file__),
                    candidate_model="M",
                    model_revision="a" * 40,
                    context=32768,
                )

    def test_quality_probe_rejects_serving_geometry_mismatch(self):
        baseline = {
            "strict": {
                "strict": True,
                "file": "unused.json",
                "model": "M",
                "model_revision": "a" * 40,
                "serve_args_sha256": "b" * 64,
                "context": 32768,
                "protocol": {
                    "profile": "gpqa-diamond",
                    "runs": 198,
                    "concurrency": 2,
                    "max_tokens": 30000,
                    "request_overrides": {},
                },
            }
        }
        with mock.patch.object(
                common, "load_baseline_registry", return_value=baseline):
            with self.assertRaisesRegex(
                    RuntimeError, "serve_args_sha256"):
                probes.quality(
                    "http://127.0.0.1:1",
                    "served",
                    profile="gpqa-diamond",
                    runs=198,
                    concurrency=2,
                    max_tokens=30000,
                    baseline="strict",
                    tool=__file__,
                    expected_tool_sha256=integrity.sha256_file(__file__),
                    candidate_model="M",
                    model_revision="a" * 40,
                    serve_args_sha256="c" * 64,
                    context=32768,
                )


if __name__ == "__main__":
    unittest.main()
