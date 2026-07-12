from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import (  # noqa: E402
    Completion,
    PoolGateError,
    PoolGatePolicy,
    ProvenanceError,
    WARMUP_CASES,
    WARMUP_SEED,
    WARMUP_SUITE_VERSION,
    WARMUP_TEMPERATURE,
    build_warmup_specs,
    evaluate_pool_gate,
    parse_pool_json,
    parse_pool_log,
    run_prewarm,
    validate_server_provenance,
    WarmupError,
)
from rescore_robust import (  # noqa: E402
    ScoreDataError,
    completion_token_count,
    score_file,
)
import eval_rig  # noqa: E402


LOCAL_FIXTURES = ROOT / "tests" / "fixtures"
SUPPLIED = LOCAL_FIXTURES


BASELINES = {
    "raw-rig-32k-base8-delta6-lru-s42.jsonl": {
        "sinks": 5,
        "clean": 27,
        "sink_ids": ["r07", "c10", "r14", "r15", "c18"],
    },
    "raw-rig-32k-b8d6-lru-s43.jsonl": {
        "sinks": 7,
        "clean": 22,
        "sink_ids": ["r04", "r06", "r07", "r12", "r14", "r15", "r16"],
    },
    "raw-rig-32k-b8d6-lru-s44.jsonl": {
        "sinks": 4,
        "clean": 27,
        "sink_ids": ["r01", "c03", "r10", "c10"],
    },
}


def valid_provenance():
    return {
        "target_host": "example-gpu-host",
        "host_boot_id": "44bd7f4c-4c61-4e68-a8ae-50b37d1ff7e2",
        "container_name": "ds4-w2-rig",
        "container_id": "sha256:" + "1" * 64,
        "container_inspect_sha256": "sha256:" + "5" * 64,
        "server_started_at_utc": "2026-07-11T12:00:00Z",
        "served_model": "deepseek-v4-flash-w2",
        "endpoint": "http://127.0.0.1:18001/v1/chat/completions",
        "source_commit": "a" * 40,
        "source_patch_sha256": "b" * 64,
        "image_ref": "vllm-moet-sm120:v024-test",
        "image_id": "sha256:" + "2" * 64,
        "checkpoint_fingerprint": "sha256:" + "3" * 64,
        "pack_fingerprint": "sha256:" + "4" * 64,
        "launcher_sha256": "sha256:" + "6" * 64,
        "runtime_argv": [
            "--model",
            "/model",
            "--served-model-name",
            "deepseek-v4-flash-w2",
            "--max-model-len",
            "32768",
            "--gpu-memory-utilization",
            "0.95",
            "--kv-cache-dtype",
            "fp8",
            "--speculative-config",
            '{"method":"deepseek_mtp","num_speculative_tokens":2}',
            "--port",
            "18001",
        ],
        "w2_environment": {
            "VLLM_MOE_W2": "1",
            "VLLM_MOE_W2_BASE_CACHE_GB": "8",
            "VLLM_MOE_W2_DELTA_GB": "6",
            "VLLM_MOE_W2_DELTA_POLICY": "lru",
            "VLLM_MOE_W2_GATE": "1",
            "VLLM_MOE_W2_GATE_TAU": "0.67",
        },
        "runtime": {
            "max_model_len": 32768,
            "base_cache_gb": 8,
            "delta_gb": 6,
            "delta_policy": "lru",
            "gate_tau": 0.67,
            "kv_cache_dtype": "fp8",
            "speculative_tokens": 2,
            "gpu_memory_utilization": 0.95,
        },
    }


class RobustScorerTests(unittest.TestCase):
    def test_corrected_baselines_from_supplied_jsonl(self):
        for filename, expected in BASELINES.items():
            with self.subTest(filename=filename):
                result = score_file(SUPPLIED / filename, expected_count=40)
                self.assertEqual(result["sinks"], expected["sinks"])
                self.assertEqual(result["clean"], expected["clean"])
                self.assertEqual(
                    [row["id"] for row in result["rows"] if row["sink"]],
                    expected["sink_ids"],
                )

    def test_ct_alias_is_used_for_max_token_detection(self):
        path = SUPPLIED / "raw-rig-32k-base8-delta6-lru-s42.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        c10 = next(row for row in rows if row["id"] == "c10")
        self.assertNotIn("completion_tokens", c10)
        self.assertEqual(completion_token_count(c10), 700)
        result = score_file(path, expected_count=40)
        scored = next(row for row in result["rows"] if row["id"] == "c10")
        self.assertTrue(scored["sink"])
        self.assertEqual(scored["why"], "MAXTOK-noncompletion")

    def test_conflicting_token_aliases_fail_closed(self):
        with self.assertRaises(ScoreDataError):
            completion_token_count({"ct": 700, "completion_tokens": 699})


class WarmupTests(unittest.TestCase):
    def test_specs_are_fixed_temp_zero_and_eval_seed_independent(self):
        first = build_warmup_specs("model-a")
        second = build_warmup_specs("model-a")
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(WARMUP_CASES))
        self.assertTrue(
            all(spec.temperature == WARMUP_TEMPERATURE == 0.0 for _, spec in first)
        )
        self.assertTrue(all(spec.seed == WARMUP_SEED for _, spec in first))

    def test_success_emits_a_receipt_for_every_case(self):
        answers = iter(case.expected for case in WARMUP_CASES)
        receipts = []

        def client(_spec):
            return Completion(
                ok=True,
                status_code=200,
                completion_tokens=5,
                finish_reason="stop",
                content=f"FINAL: {next(answers)}.",
            )

        result = run_prewarm(client, "model-a", receipts.append)
        self.assertEqual(len(result), len(WARMUP_CASES))
        self.assertEqual(receipts, result)
        self.assertTrue(all(receipt["accepted"] for receipt in result))
        self.assertEqual(WARMUP_SUITE_VERSION, "ds4-w2-prewarm-v4")
        self.assertTrue(
            all(receipt["suite_version"] == WARMUP_SUITE_VERSION for receipt in result)
        )
        self.assertTrue(all("<answer>" not in case.prompt for case in WARMUP_CASES))

    def test_native_answer_tag_is_accepted_exactly(self):
        answers = iter(case.expected for case in WARMUP_CASES)

        def client(_spec):
            return Completion(
                ok=True,
                status_code=200,
                completion_tokens=5,
                finish_reason="stop",
                content=f"<answer>{next(answers)}</answer>",
            )

        result = run_prewarm(client, "model-a", lambda _receipt: None)
        self.assertTrue(all(receipt["accepted"] for receipt in result))

    def test_deepseek_think_boundary_is_accepted_exactly(self):
        for formatter in (
            lambda answer: f"reasoning text</think>FINAL: {answer}",
            lambda answer: f"reasoning text</think><answer>{answer}</answer>",
            lambda answer: f"reasoning text\n<answer>{answer}</answer>",
        ):
            with self.subTest(formatter=formatter):
                answers = iter(case.expected for case in WARMUP_CASES)

                def client(_spec):
                    return Completion(
                        ok=True,
                        status_code=200,
                        completion_tokens=5,
                        finish_reason="stop",
                        content=formatter(next(answers)),
                    )

                result = run_prewarm(client, "model-a", lambda _receipt: None)
                self.assertTrue(all(receipt["accepted"] for receipt in result))

    def test_answer_markers_reject_trailing_or_ambiguous_content(self):
        bad_contents = (
            "FINAL: 4\ntrailing junk",
            "prefix FINAL: 4",
            "prefix <answer>4</answer>",
            "<answer>4</answer> suffix",
            "reasoning</think>FINAL: <answer> 4",
            "reasoning</think>FINAL: 4 trailing junk",
            "reasoning</think><answer>4</answer> trailing junk",
            "<answer>wrong</answer><answer>4</answer>",
            "<answer><answer>4</answer></answer>",
        )
        for content in bad_contents:
            with self.subTest(content=content):

                def client(_spec):
                    return Completion(
                        ok=True,
                        status_code=200,
                        completion_tokens=5,
                        finish_reason="stop",
                        content=content,
                    )

                with self.assertRaises(WarmupError):
                    run_prewarm(client, "model-a", lambda _receipt: None)

    def test_failure_is_receipted_and_stops_immediately(self):
        receipts = []
        calls = 0

        def client(_spec):
            nonlocal calls
            calls += 1
            if calls == 1:
                return Completion(
                    ok=True,
                    status_code=200,
                    completion_tokens=3,
                    finish_reason="stop",
                    content=f"FINAL: {WARMUP_CASES[0].expected}",
                )
            return Completion(ok=False, status_code=503, error="unavailable")

        with self.assertRaises(WarmupError) as context:
            run_prewarm(client, "model-a", receipts.append)
        self.assertEqual(calls, 2)
        self.assertEqual(len(receipts), 2)
        self.assertFalse(receipts[-1]["accepted"])
        self.assertEqual(context.exception.receipts, receipts)

    def test_client_exception_is_still_receipted(self):
        receipts = []

        def client(_spec):
            raise TimeoutError("timed out")

        with self.assertRaises(WarmupError):
            run_prewarm(client, "model-a", receipts.append)
        self.assertEqual(len(receipts), 1)
        self.assertIn("TimeoutError", receipts[0]["response"]["error"])


class ProvenanceTests(unittest.TestCase):
    def test_complete_provenance_passes(self):
        self.assertEqual(
            validate_server_provenance(valid_provenance()), valid_provenance()
        )

    def test_missing_exact_identity_fails(self):
        value = valid_provenance()
        del value["image_id"]
        with self.assertRaises(ProvenanceError):
            validate_server_provenance(value)

    def test_placeholder_identity_fails(self):
        value = valid_provenance()
        value["pack_fingerprint"] = "<collect me>"
        with self.assertRaises(ProvenanceError):
            validate_server_provenance(value)

    def test_source_patch_requires_full_sha256(self):
        value = valid_provenance()
        value["source_patch_sha256"] = "abc123"
        with self.assertRaisesRegex(ProvenanceError, "source_patch_sha256"):
            validate_server_provenance(value)

    def test_required_w2_environment_cannot_be_omitted(self):
        value = valid_provenance()
        del value["w2_environment"]["VLLM_MOE_W2_DELTA_POLICY"]
        with self.assertRaisesRegex(ProvenanceError, "missing w2_environment"):
            validate_server_provenance(value)

    def test_w2_environment_must_match_structured_runtime(self):
        value = valid_provenance()
        value["w2_environment"]["VLLM_MOE_W2_DELTA_GB"] = "4"
        with self.assertRaisesRegex(ProvenanceError, "VLLM_MOE_W2_DELTA_GB"):
            validate_server_provenance(value)

    def test_runtime_argv_must_match_structured_runtime(self):
        value = valid_provenance()
        index = value["runtime_argv"].index("--max-model-len")
        value["runtime_argv"][index + 1] = "131072"
        with self.assertRaisesRegex(ProvenanceError, "--max-model-len"):
            validate_server_provenance(value)

    def test_runtime_argv_speculative_config_must_match(self):
        value = valid_provenance()
        index = value["runtime_argv"].index("--speculative-config")
        value["runtime_argv"][index + 1] = (
            '{"method":"deepseek_mtp","num_speculative_tokens":4}'
        )
        with self.assertRaisesRegex(ProvenanceError, "num_speculative_tokens"):
            validate_server_provenance(value)


class PoolGateTests(unittest.TestCase):
    LOG = (
        "INFO [fp4] tick 128: 470/481 slots, covering 470/11008 experts (4.3%); "
        "hit-rate 71.2% tokens / 48.0% experts; window +40/-31, cumulative +900/-430\n"
        "INFO [base] KPI: replay 8.0% of last 64 steps (avg 2.0 missing pairs/step; "
        "cumulative 9.0% of 1024) — pool 688 slots = 6.2% of experts; "
        "UNRESTORED experts: 0; fp-residue: 0 steps\n"
    )

    def test_existing_log_kpis_parse_and_gate(self):
        snapshot = parse_pool_log(self.LOG)
        self.assertEqual(snapshot.fp4_tick, 128)
        self.assertEqual(snapshot.fp4_cached, 470)
        self.assertEqual(snapshot.fp4_slots, 481)
        self.assertEqual(snapshot.fp4_total_evicted, 430)
        self.assertEqual(snapshot.base_replay_pct, 8.0)
        self.assertEqual(snapshot.base_unrestored_experts, 0)
        policy = PoolGatePolicy(
            min_fp4_tick=1,
            min_fp4_occupancy=0.95,
            min_fp4_total_evicted=1,
            max_base_replay_pct=10.0,
            max_base_unrestored_experts=0,
            max_base_fp_residue_steps=0,
        )
        gate = evaluate_pool_gate(snapshot, policy)
        self.assertTrue(gate["passed"], gate["checks"])

    def test_missing_metric_fails_a_configured_check(self):
        snapshot = parse_pool_json({"tick": 5, "cached": 10, "n_slots": 10})
        gate = evaluate_pool_gate(snapshot, PoolGatePolicy(max_base_replay_pct=10.0))
        self.assertFalse(gate["passed"])
        self.assertIsNone(gate["checks"][0]["observed"])

    def test_current_flat_delta_dump_is_supported(self):
        snapshot = parse_pool_json(
            {
                "tick": 9,
                "n_slots": 481,
                "cached": 480,
                "promoted_total": 700,
                "evicted_total": 219,
            }
        )
        self.assertEqual(snapshot.fp4_tick, 9)
        self.assertEqual(snapshot.fp4_slots, 481)
        self.assertAlmostEqual(snapshot.fp4_occupancy, 480 / 481)

    def test_future_combined_gate_stats_are_supported(self):
        snapshot = parse_pool_json(
            {
                "fp4": {"tick": 9, "slots": 481, "cached": 480},
                "gate": {"steps": 100, "fired": 30},
            }
        )
        gate = evaluate_pool_gate(
            snapshot,
            PoolGatePolicy(
                min_gate_steps=64, min_gate_fire_rate=0.2, max_gate_fire_rate=0.4
            ),
        )
        self.assertTrue(gate["passed"], gate["checks"])
        self.assertEqual(snapshot.gate_fire_rate, 0.3)

    def test_post_eval_eviction_delta_must_show_live_churn(self):
        policy = PoolGatePolicy(
            min_fp4_occupancy=0.95,
            min_fp4_total_evicted_delta=16,
        )
        before = parse_pool_json(
            {"tick": 100, "n_slots": 481, "cached": 481, "evicted_total": 20}
        )
        frozen = parse_pool_json(
            {"tick": 200, "n_slots": 481, "cached": 481, "evicted_total": 20}
        )
        healthy = parse_pool_json(
            {"tick": 200, "n_slots": 481, "cached": 481, "evicted_total": 36}
        )

        pre_gate = evaluate_pool_gate(before, policy)
        self.assertTrue(pre_gate["passed"], pre_gate["checks"])
        self.assertEqual(pre_gate["deferred_checks"], ["fp4_total_evicted_delta"])

        frozen_gate = evaluate_pool_gate(frozen, policy, baseline=before)
        self.assertFalse(frozen_gate["passed"])
        delta = next(
            check
            for check in frozen_gate["checks"]
            if check["metric"] == "fp4_total_evicted_delta"
        )
        self.assertEqual(delta["observed"], 0)

        healthy_gate = evaluate_pool_gate(healthy, policy, baseline=before)
        self.assertTrue(healthy_gate["passed"], healthy_gate["checks"])

    def test_post_eval_eviction_delta_fails_when_counter_is_missing(self):
        policy = PoolGatePolicy(
            min_fp4_tick=1,
            min_fp4_total_evicted_delta=16,
        )
        before = parse_pool_json({"tick": 100})
        after = parse_pool_json({"tick": 200, "evicted_total": 36})
        gate = evaluate_pool_gate(after, policy, baseline=before)
        self.assertFalse(gate["passed"])
        delta = next(
            check
            for check in gate["checks"]
            if check["metric"] == "fp4_total_evicted_delta"
        )
        self.assertIsNone(delta["observed"])

    def test_nonsensical_thresholds_fail_closed(self):
        with self.assertRaisesRegex(PoolGateError, "non-negative"):
            PoolGatePolicy.from_mapping({"min_fp4_total_evicted_delta": -1})
        with self.assertRaisesRegex(PoolGateError, r"\[0, 100\]"):
            PoolGatePolicy.from_mapping({"max_base_replay_pct": 1000})


class EvalRigIntegrationTests(unittest.TestCase):
    def test_warm_run_writes_manifest_receipts_and_canonical_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items_path = root / "items.json"
            provenance_path = root / "server.json"
            policy_path = root / "policy.json"
            log_path = root / "pool.log"
            output = root / "out"
            items_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "r01",
                            "cat": "reasoning",
                            "prompt": "Compute 1+1",
                            "answer": "2",
                        },
                        {
                            "id": "c01",
                            "cat": "coding",
                            "prompt": "Compute 2+2",
                            "answer": "4",
                        },
                    ]
                )
            )
            provenance_path.write_text(json.dumps(valid_provenance()))
            policy_path.write_text(
                json.dumps(
                    {
                        "min_fp4_tick": 1,
                        "min_fp4_occupancy": 0.95,
                        "min_fp4_total_evicted_delta": 16,
                        "max_base_replay_pct": 10.0,
                    }
                )
            )
            log_path.write_text(PoolGateTests.LOG)
            args = argparse.Namespace(
                items=str(items_path),
                server_provenance=str(provenance_path),
                output_dir=str(output),
                run_label="unit-run",
                url=valid_provenance()["endpoint"],
                model="deepseek-v4-flash-w2",
                mode="warm",
                eval_seed=43,
                eval_temperature=0.6,
                eval_top_p=0.95,
                eval_max_tokens=700,
                eval_timeout=300,
                expected_count=2,
                pool_gate_policy=str(policy_path),
                pool_log_file=str(log_path),
                pool_json_file=None,
                pool_kpi_url=None,
                pool_command_json=None,
                pool_kpi_timeout=30,
            )
            warm_answers = iter(case.expected for case in WARMUP_CASES)
            calls = 0

            def client(spec):
                nonlocal calls
                calls += 1
                if calls <= len(WARMUP_CASES):
                    answer = next(warm_answers)
                    self.assertEqual(spec.temperature, 0.0)
                    self.assertEqual(spec.seed, WARMUP_SEED)
                else:
                    answer = "2" if "1+1" in spec.prompt else "4"
                    self.assertEqual(spec.temperature, 0.6)
                    self.assertEqual(spec.seed, 43)
                return Completion(
                    ok=True,
                    status_code=200,
                    completion_tokens=4,
                    finish_reason="stop",
                    content=f"FINAL: {answer}",
                )

            with (
                mock.patch.object(eval_rig, "parse_args", return_value=args),
                mock.patch.object(
                    eval_rig, "HttpCompletionClient", return_value=client
                ),
                mock.patch.object(
                    eval_rig,
                    "_load_pool_snapshot",
                    side_effect=[
                        parse_pool_log(PoolGateTests.LOG),
                        parse_pool_log(
                            PoolGateTests.LOG.replace(
                                "cumulative +900/-430", "cumulative +916/-446"
                            )
                        ),
                    ],
                ),
            ):
                self.assertEqual(eval_rig.main(), 0)

            manifest = json.loads((output / "unit-run.manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["quality_comparable"])
            self.assertTrue(manifest["pool_gate"]["passed"])
            self.assertTrue(manifest["pool_gate"]["post_eval"]["passed"])
            post_delta = next(
                check
                for check in manifest["pool_gate"]["post_eval"]["checks"]
                if check["metric"] == "fp4_total_evicted_delta"
            )
            self.assertEqual(post_delta["observed"], 16)
            receipts = (output / "unit-run.warmup.jsonl").read_text().splitlines()
            self.assertEqual(len(receipts), len(WARMUP_CASES))
            rows = [
                json.loads(line)
                for line in (output / "unit-run.raw.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["completion_tokens"] for row in rows], [4, 4])
            self.assertNotIn("ct", rows[0])
            self.assertTrue(all(row["exact"] for row in rows))


if __name__ == "__main__":
    unittest.main()
