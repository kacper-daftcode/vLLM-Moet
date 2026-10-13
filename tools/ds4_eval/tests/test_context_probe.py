from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from context_probe import (  # noqa: E402
    CaseSpec,
    ContextProbeError,
    ProbeConfig,
    _extract_exact_answer,
    run,
)


def valid_provenance(max_num_seqs: int = 1) -> dict:
    return {
        "target_host": "example-gpu-host",
        "host_boot_id": "44bd7f4c-4c61-4e68-a8ae-50b37d1ff7e2",
        "container_name": "ds4-w2-context-unit",
        "container_id": "1" * 64,
        "container_inspect_sha256": "5" * 64,
        "server_started_at_utc": "2026-07-11T12:00:00Z",
        "served_model": "deepseek-v4-flash-w2",
        "endpoint": "http://127.0.0.1:18001/v1/chat/completions",
        "source_commit": "a" * 40,
        "source_patch_sha256": "b" * 64,
        "image_ref": "vllm-moet-sm120:v024-test",
        "image_id": "sha256:" + "2" * 64,
        "checkpoint_fingerprint": "sha256:" + "3" * 64,
        "pack_fingerprint": "sha256:" + "4" * 64,
        "launcher_sha256": "6" * 64,
        "runtime_argv": [
            "--model",
            "/model",
            "--served-model-name",
            "deepseek-v4-flash-w2",
            "--max-model-len",
            "256",
            "--gpu-memory-utilization",
            "0.95",
            "--kv-cache-dtype",
            "fp8",
            "--max-num-seqs",
            str(max_num_seqs),
            "--port",
            "18001",
        ],
        "w2_environment": {
            "VLLM_MOE_W2": "1",
            "VLLM_MOE_W2_BASE_CACHE_GB": "8",
            "VLLM_MOE_W2_DELTA_GB": "6",
            "VLLM_MOE_W2_DELTA_POLICY": "lru",
            "VLLM_MOE_W2_GATE": "1",
            "VLLM_MOE_W2_GATE_TAU": "0.75",
        },
        "runtime": {
            "max_model_len": 256,
            "base_cache_gb": 8,
            "delta_gb": 6,
            "delta_policy": "lru",
            "gate_tau": 0.75,
            "kv_cache_dtype": "fp8",
            "speculative_tokens": 0,
            "gpu_memory_utilization": 0.95,
        },
    }


class FakeTransport:
    def __init__(
        self,
        *,
        finish_reason: str = "stop",
        answer_mode: str = "exact",
        usage_offset: int = 0,
        corrupt_tokens: bool = False,
    ):
        self.finish_reason = finish_reason
        self.answer_mode = answer_mode
        self.usage_offset = usage_offset
        self.corrupt_tokens = corrupt_tokens
        self.calls = []

    @staticmethod
    def _count(body: dict) -> int:
        # A deterministic stand-in for the chat tokenizer: one token per word
        # plus a fixed four-token template boundary.
        return len(body["messages"][0]["content"].split()) + 4

    def __call__(self, url: str, body: dict, _timeout: int):
        self.calls.append((url, body))
        count = self._count(body)
        if url.endswith("/tokenize"):
            token_count = count - 1 if self.corrupt_tokens else count
            return (
                200,
                {
                    "count": count,
                    "max_model_len": 256,
                    "tokens": list(range(token_count)),
                },
                0.01,
            )
        secret = re.search(
            r"the vault passphrase is ([A-Z0-9-]+)\.",
            body["messages"][0]["content"],
        ).group(1)
        if self.answer_mode == "exact":
            content = f"<answer>{secret}</answer>"
        else:
            content = f"The answer is {secret}."
        return (
            200,
            {
                "id": "cmpl-unit",
                "choices": [
                    {
                        "finish_reason": self.finish_reason,
                        "message": {"content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": count + self.usage_offset,
                    "completion_tokens": 5,
                },
            },
            0.02,
        )


class ContextProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.provenance = self.root / "server.json"
        self.provenance.write_text(json.dumps(valid_provenance()))
        self.config = ProbeConfig(
            server_provenance=str(self.provenance),
            output_dir=str(self.root / "out"),
            run_label="p2-context-unit",
            url="http://127.0.0.1:18001/v1/chat/completions",
            tokenize_url="http://127.0.0.1:18001/tokenize",
            model="deepseek-v4-flash-w2",
            expected_window=256,
            expected_kv_dtype="fp8",
            expected_base_gb=8,
            expected_delta_gb=6,
            expected_policy="lru",
            expected_tau=0.75,
            cases=(CaseSpec(100, 0.1), CaseSpec(120, 0.9)),
            max_tokens=16,
            prompt_token_tolerance=0,
            timeout_seconds=10,
        )

    def test_complete_run_writes_immutable_manifest_and_receipts(self):
        transport = FakeTransport()
        self.assertEqual(run(self.config, transport), 0)

        output = Path(self.config.output_dir)
        manifest_path = output / "p2-context-unit.manifest.json"
        raw_path = output / "p2-context-unit.context.jsonl"
        manifest = json.loads(manifest_path.read_text())
        receipts = [json.loads(line) for line in raw_path.read_text().splitlines()]

        self.assertEqual(manifest["status"], "complete")
        self.assertTrue(manifest["context_validated"])
        self.assertEqual(manifest["summary"]["accepted"], 2)
        self.assertEqual(len(receipts), 2)
        self.assertTrue(all(row["accepted"] for row in receipts))
        self.assertEqual(
            [row["calibration"]["observed_prompt_tokens"] for row in receipts],
            [100, 120],
        )
        self.assertEqual(
            [row["response"]["prompt_tokens"] for row in receipts], [100, 120]
        )
        self.assertEqual(len({row["expected_answer"] for row in receipts}), 2)
        self.assertTrue(all(row["request"]["thinking"] is False for row in receipts))

        with self.assertRaisesRegex(ContextProbeError, "refusing to overwrite"):
            run(self.config, transport)

    def test_wrong_exact_answer_fails_and_leaves_receipt(self):
        self.assertEqual(run(self.config, FakeTransport(answer_mode="junk")), 1)
        output = Path(self.config.output_dir)
        manifest = json.loads((output / "p2-context-unit.manifest.json").read_text())
        receipt = json.loads(
            (output / "p2-context-unit.context.jsonl").read_text().splitlines()[0]
        )
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse(manifest["context_validated"])
        self.assertFalse(receipt["accepted"])
        self.assertIn("expected exact terminal answer", receipt["error"])

    def test_finish_and_usage_must_match_fail_closed(self):
        for name, transport in (
            ("finish", FakeTransport(finish_reason="length")),
            ("usage", FakeTransport(usage_offset=1)),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                config = replace(
                    self.config,
                    output_dir=directory,
                    run_label=f"p2-context-{name}",
                )
                self.assertEqual(run(config, transport), 1)
                receipt = json.loads(
                    (Path(directory) / f"p2-context-{name}.context.jsonl").read_text()
                )
                self.assertFalse(receipt["accepted"])

    def test_tokenizer_count_and_token_array_must_agree(self):
        self.assertEqual(run(self.config, FakeTransport(corrupt_tokens=True)), 1)
        receipt = json.loads(
            (Path(self.config.output_dir) / "p2-context-unit.context.jsonl").read_text()
        )
        self.assertIn("tokens length does not match count", receipt["error"])

    def test_transport_error_is_receipted_and_fails_closed(self):
        def broken_transport(_url, _body, _timeout):
            raise ContextProbeError("HTTP 500 unit failure")

        self.assertEqual(run(self.config, broken_transport), 1)
        receipt = json.loads(
            (Path(self.config.output_dir) / "p2-context-unit.context.jsonl").read_text()
        )
        self.assertFalse(receipt["accepted"])
        self.assertIn("HTTP 500 unit failure", receipt["error"])

    def test_runtime_must_be_no_mtp_single_sequence(self):
        self.provenance.write_text(json.dumps(valid_provenance(max_num_seqs=2)))
        with self.assertRaisesRegex(ContextProbeError, "max-num-seqs 1"):
            run(self.config, FakeTransport())

        value = valid_provenance()
        value["runtime"]["speculative_tokens"] = 2
        value["runtime_argv"].extend(
            [
                "--speculative-config",
                '{"method":"deepseek_mtp","num_speculative_tokens":2}',
            ]
        )
        self.provenance.write_text(json.dumps(value))
        with self.assertRaisesRegex(ContextProbeError, "speculative_tokens=0"):
            run(self.config, FakeTransport())

    def test_exact_answer_accepts_only_terminal_native_wrappers(self):
        self.assertEqual(_extract_exact_answer("DS4-ABC"), "DS4-ABC")
        self.assertEqual(
            _extract_exact_answer("thinking</think>\n<answer>DS4-ABC</answer>"),
            "DS4-ABC",
        )
        self.assertEqual(_extract_exact_answer("FINAL: DS4-ABC"), "DS4-ABC")
        self.assertEqual(_extract_exact_answer("answer: DS4-ABC"), "answer: DS4-ABC")


if __name__ == "__main__":
    unittest.main()
