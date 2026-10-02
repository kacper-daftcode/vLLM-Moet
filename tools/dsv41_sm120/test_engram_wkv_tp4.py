#!/usr/bin/env python3
"""Engram with the split wkv (patch_vllm_engram_wkv_tp.py) on 4 real TP ranks, against the replicated projection.

One `Engram` per rank (the real layout's 24 hash heads x 256 over a small random table, prime-sized buckets from a
tiny vocab; wkv synthetic or the checkpoint's, --checkpoint DIR, layer 1, loaded through the parameter loaders).
Every rank runs prepare_embeddings + forward - the full path: lookup, TP all-gather of the head rows, wkv, the
post-wkv kernel - with the split off (the reference), forced at every size (1..4096 tokens: the slices + NCCL
all-gather of the output) and at the default threshold, eagerly and replayed from CUDA graphs (NCCL inside the
capture); the outputs must be identical bit for bit. Then timings on this host: the output all-gather alone and
the forward with and without the split, in a CUDA graph at decode sizes and eagerly at prefill sizes.

usage (in the serving image with step 12, 4 GPUs):
    torchrun --nproc_per_node=4 test_engram_wkv_tp4.py [--checkpoint DIR] [--no-bench]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

K, N, TP = 6144, 25600, 4
DECODE_TS = list(range(1, 65))
PREFILL_TS = [65, 128, 256, 512, 1024, 2048, 4096]
OFF, FORCED, DEFAULT = (False, 64), (True, max(PREFILL_TS)), (True, 64)
CFG = SimpleNamespace(
    hidden_size=5120, hc_mult=4, rms_norm_eps=1e-20, engram_layer_ids=[1, 14], engram_num_embeddings=[20000] * 2,
    engram_max_ngram_size=4, engram_vocab_size=64, engram_n_heads=8, engram_head_dim=256,
    engram_pad_token_id=2, engram_compressed_vocab_size=99,
)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def wkv_tensors(args, dev, g) -> tuple[torch.Tensor, torch.Tensor]:
    if args.checkpoint:
        from safetensors import safe_open

        root = Path(args.checkpoint)
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        out = []
        for leaf in ("weight", "scale"):
            name = f"layers.1.engram.wkv.{leaf}"
            with safe_open(str(root / index[name]), "pt", device=str(dev)) as f:
                out.append(f.get_tensor(name))
        return out[0], out[1].view(torch.float8_e8m0fnu)
    w = torch.randint(0, 256, (N, K), dtype=torch.uint8, device=dev, generator=g)
    w[(w & 0x7F) == 0x7F] = 0
    s = torch.randint(127 - 14, 127 - 3, (N // 32, K // 32), dtype=torch.uint8, device=dev, generator=g)
    return w.view(torch.float8_e4m3fn), s.view(torch.float8_e8m0fnu)


def inputs(layout, t: int, dev, g):
    sizes = torch.tensor([p for per_ngram in layout.primes[0] for p in per_ngram], device=dev)
    offsets = layout.offsets[0].to(dev)
    ids = (torch.rand(t, sizes.numel(), device=dev, generator=g) * sizes).long() + offsets
    hidden = (torch.randn(t, CFG.hc_mult, CFG.hidden_size, device=dev, generator=g) * 2).to(torch.bfloat16)
    return ids.to(torch.int32), hidden


def graph_us(fn, per_graph: int = 10, reps: int = 15) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            for _ in range(per_graph):
                fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        dist.barrier()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        graph.replay()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) * 1000 / per_graph)
    return statistics.median(ts)


def eager_us(fn, iters: int = 10) -> float:
    fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == TP, "run with torchrun --nproc_per_node=4"
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
    )
    from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config
    from vllm.utils.torch_utils import set_default_torch_dtype

    import vllm.models.deepseek_v41.common.engram as E

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    if not hasattr(E, "_moet_engram_wkv"):
        log("engram.py is not patched (patch_vllm_engram_wkv_tp.py)")
        return 1
    qc = DeepseekV4FP8Config.from_config({"quant_method": "fp8", "activation_scheme": "dynamic",
                                          "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"})
    ok = True
    vllm_config = VllmConfig()
    vllm_config.scheduler_config.max_num_batched_tokens = max(PREFILL_TS)  # Engram's staging rows, as served
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.bfloat16), torch.device(dev):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method="env://",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=TP)
        layout = E.EngramLayout(CFG)
        engram = E.Engram(CFG, qc, layout, 0, use_sequence_parallel=False, prefix="model.layers.1.engram")
        g_rank = torch.Generator(device=dev).manual_seed(100 + rank)  # this rank's table shard
        g_all = torch.Generator(device=dev).manual_seed(7)  # identical on every rank
        emb = engram.embed_tokens
        table = torch.randint(0, 256, tuple(emb.weight.shape), dtype=torch.uint8, device=dev, generator=g_rank)
        table[(table & 0x7F) == 0x7F] = 0
        emb.weight.data.copy_(table.view(torch.float8_e4m3fn))
        emb.weight_scale_inv.data.copy_(torch.randint(127 - 8, 127 - 4, tuple(emb.weight_scale_inv.shape),
                                                      dtype=torch.uint8, device=dev, generator=g_rank))
        engram.q_weight.data.copy_((torch.randn(CFG.hc_mult, CFG.hidden_size, device=dev, generator=g_all)
                                    * 0.05).to(torch.bfloat16))
        engram.k_weight.data.copy_((torch.randn(CFG.hc_mult, CFG.hidden_size, device=dev, generator=g_all)
                                    * 0.05).to(torch.bfloat16))
        w8, s8 = wkv_tensors(args, dev, g_all)
        for param, tensor in ((engram.wkv.weight, w8), (engram.wkv.weight_scale, s8)):
            param.weight_loader(param, tensor)
        engram.wkv.quant_method.process_weights_after_loading(engram.wkv)
        del w8, s8, table
        torch.cuda.empty_cache()
        log(f"wkv {type(engram.wkv).__name__} {tuple(engram.wkv.weight.shape)}, "
            f"{type(engram.wkv.quant_method.kernel).__name__}")

        def run(mode, hidden, ids):
            E._MOET_WKV_TP = mode
            engram.prepare_embeddings(ids)
            return engram(hidden, ids)

        with torch.inference_mode():
            bad = []
            for t in DECODE_TS + PREFILL_TS:
                ids, hidden = inputs(layout, t, dev, g_all)
                ref = run(OFF, hidden, ids).clone()
                for mode in (FORCED, DEFAULT):
                    if not torch.equal(bits(run(mode, hidden, ids)), bits(ref)):
                        bad.append((t, mode))
            n_bad = torch.tensor([len(bad)], device=dev)
            dist.all_reduce(n_bad)
            split = getattr(engram.wkv, "_moet_wkv_part", (None, None))[1]
            log(f"eager, 1..4096 tokens, split forced and at the default threshold: split != replicated in "
                f"{n_bad.item()} (rank 0: {bad[:6]}); rank slice {None if split is None else tuple(split.weight.shape)}")
            ok &= n_bad.item() == 0 and split is not None

            gbad = []
            for t in (1, 6, 12, 16, 48, 64):
                ids, hidden = inputs(layout, t, dev, g_all)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    run(DEFAULT, hidden, ids)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=side):
                        out = run(DEFAULT, hidden, ids)
                torch.cuda.synchronize()
                for _ in range(3):
                    new_ids, new_hidden = inputs(layout, t, dev, g_all)
                    ids.copy_(new_ids)
                    hidden.copy_(new_hidden)
                    graph.replay()
                    torch.cuda.synchronize()
                    if not torch.equal(bits(out), bits(run(OFF, hidden, ids))):
                        gbad.append(t)
                del graph
            n_bad = torch.tensor([len(gbad)], device=dev)
            dist.all_reduce(n_bad)
            log(f"CUDA graph replays (NCCL all-gather captured), 1..64 tokens: split != replicated in {n_bad.item()}")
            ok &= n_bad.item() == 0

            if not args.no_bench:
                log(f"timings on {torch.cuda.get_device_name(dev)} x {TP}, us per call (graph: 10 calls per replay, "
                    f"median of 15; eager: wall time of 10 calls)")
                for t in (1, 6, 12, 24, 48, 64):
                    ids, hidden = inputs(layout, t, dev, g_all)
                    part = torch.randn(t, N // TP, device=dev, generator=g_all).to(torch.bfloat16)
                    t_ag = graph_us(lambda: tensor_model_parallel_all_gather(part))
                    t_rep = graph_us(lambda: run(OFF, hidden, ids))
                    t_split = graph_us(lambda: run(DEFAULT, hidden, ids))
                    log(f"  graph tokens={t:4d}: all-gather [{t}, {N // TP}] {t_ag:6.1f}   Engram forward "
                        f"replicated {t_rep:7.1f} -> split {t_split:7.1f} ({t_split - t_rep:+.1f})")
                for t in (65, 96, 128, 256, 1024, 4096):
                    ids, hidden = inputs(layout, t, dev, g_all)
                    part = torch.randn(t, N // TP, device=dev, generator=g_all).to(torch.bfloat16)
                    t_ag = eager_us(lambda: tensor_model_parallel_all_gather(part))
                    t_rep = eager_us(lambda: run(OFF, hidden, ids))
                    t_forced = eager_us(lambda: run(FORCED, hidden, ids))
                    log(f"  eager tokens={t:4d}: all-gather [{t}, {N // TP}] {t_ag:8.1f}   Engram forward "
                        f"replicated {t_rep:8.1f}, split forced {t_forced:8.1f} ({t_forced - t_rep:+.1f})")
    ok_t = torch.tensor([0 if ok else 1], device=dev)
    dist.all_reduce(ok_t)
    log("ALL OK" if ok_t.item() == 0 else "FAILED")
    return 0 if ok_t.item() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
