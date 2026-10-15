#!/usr/bin/env python3
"""Does the served DeepSeek-V4.1 configuration validate with --enable-dbo? (no model load, no GPU memory)

Builds the engine config docker/sm120/run-dsv41.sh would build with PREFILL_DBO=1 and runs vLLM's config
validation: on an unpatched tree it stops at "DeepSeek V4.1 Engram does not support DBO" (or the DSpark /
multimodal / all2all-backend gates); on the patched one it prints the resolved microbatch settings.
usage (serving image, the checkpoint at /model): python3 check_dbo_config.py [--model /model] [--no-dbo]"""

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model")
    ap.add_argument("--no-dbo", action="store_true", help="the served configuration as is")
    ap.add_argument("--prefill-threshold", type=int, default=1024)
    ap.add_argument("--decode-threshold", type=int, default=4096)
    args = ap.parse_args()

    from vllm.engine.arg_utils import EngineArgs

    kwargs = dict(
        model=args.model,
        served_model_name="deepseek-ai/DeepSeek-V4.1-Flash",
        tensor_parallel_size=4,
        tokenizer_mode="deepseek_v41",
        reasoning_parser="deepseek_v41",
        kv_cache_dtype="nvfp4_ds_mla",
        gpu_memory_utilization=0.94,
        max_model_len=524288,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        max_cudagraph_capture_size=64,
        speculative_config={
            "method": "dspark",
            "num_speculative_tokens": 5,
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "block",
            "enable_adaptive_verification": False,
        },
        attention_config={"indexer_kv_dtype": "mxfp4"},
    )
    if not args.no_dbo:
        kwargs.update(
            enable_dbo=True,
            dbo_prefill_token_threshold=args.prefill_threshold,
            dbo_decode_token_threshold=args.decode_threshold,
        )
    engine_args = EngineArgs(**kwargs)
    try:
        config = engine_args.create_engine_config()
    except Exception as exc:  # noqa: BLE001
        print(f"config REJECTED: {type(exc).__name__}: {exc}")
        return 1
    pc = config.parallel_config
    out = {
        "use_ubatching": pc.use_ubatching,
        "num_ubatches": pc.num_ubatches,
        "dbo_prefill_token_threshold": pc.dbo_prefill_token_threshold,
        "dbo_decode_token_threshold": pc.dbo_decode_token_threshold,
        "all2all_backend": pc.all2all_backend,
        "enable_expert_parallel": pc.enable_expert_parallel,
        "data_parallel_size": pc.data_parallel_size,
        "use_v2_model_runner": config.use_v2_model_runner,
        "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
        "swa_bounded_replay": getattr(config.cache_config, "swa_bounded_replay", None),
        "is_multimodal_model": config.model_config.is_multimodal_model,
        "speculative_method": config.speculative_config.method if config.speculative_config else None,
        "disable_cascade_attn": config.model_config.disable_cascade_attn,
    }
    print("config OK: " + json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
