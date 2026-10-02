#!/usr/bin/env python3
"""Engram wkv split by output columns (patch_vllm_engram_wkv_tp.py) against the replicated projection, bit for bit.

wkv is MXFP8 6144 -> 25600 in the checkpoint's layout: e4m3 values [25600, 6144] and ue8m0 scales per 32 x 32 block
[800, 192], which vLLM's loader expands to one scale row per weight row before the FlashInfer kernel swizzles them
(F8_128x4). Synthetic tensors by default, or the checkpoint's own (--checkpoint DIR, layers 1 and 14).

op part (no patch needed): the full weight against its TP row slices [6400, 6144] on the served kernels - the
vLLM-Moet GEMV for 1..16 tokens (v3 up to 8, v1 above) and FlashInfer's SM120 CUTLASS GEMM in every tactic for
1..64 tokens and prefill sizes up to 4096 - on wide-range activations (x 2^-6..2^6 per 32-block); cat(slices) must
equal the full result bit for bit, and the CUTLASS tactics must agree with each other (FlashInfer's autotuner picks
one per shape at startup, so the full and the sliced GEMM may run different ones).

integration part (patched engram.py): the real ReplicatedLinear with DeepseekV4FP8Config (32 x 32 blocks, ue8m0),
loaded through its parameter loaders from the same checkpoint tensors; _moet_engram_wkv for each of 4 simulated TP
ranks (the all-gather is the identity in this one-rank process, so the rank outputs are concatenated here) must
equal wkv(x) bit for bit, eagerly for 1..4096 tokens (threshold lifted) and replayed from CUDA graphs, with the rank
slices views into the layer's weight and scales. Then the dispatch: the split above VLLM_MOET_ENGRAM_WKV_TP_MAX_TOKENS,
for TP = 1, sequence parallelism, an unquantized layer and VLLM_MOET_ENGRAM_WKV_TP=0 is the replicated layer, and
the two switches as read from the environment.

--bench: GPU time per call in a CUDA graph with cold weights (the full weight vs the 4 slices in rotation).

usage (in the serving image, engram.py patched for the integration part):
    python3 test_engram_wkv_tp.py [--part op|integration|all] [--checkpoint DIR] [--bench]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import statistics
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sm120_gemv"))

K, N, TP = 6144, 25600, 4
NS = N // TP
GEMV_MAX_M = 16  # patch_vllm_mxfp8_gemv.py: the GEMV above 16 tokens is CUTLASS's
DECODE_MS = list(range(1, 65))
PREFILL_MS = [65, 96, 128, 129, 255, 256, 384, 512, 777, 1024, 1536, 2048, 3000, 4096]
dev = torch.device("cuda")


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def weights(args, layer_id: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """(e4m3 [N, K], e8m0 [N/32, K/32]) as stored in the checkpoint."""
    if args.checkpoint:
        from safetensors import safe_open

        root = Path(args.checkpoint)
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        out = []
        for leaf in ("weight", "scale"):
            name = f"layers.{layer_id}.engram.wkv.{leaf}"
            with safe_open(str(root / index[name]), "pt", device="cuda") as f:
                out.append(f.get_tensor(name))
        w, s = out
        assert w.dtype == torch.float8_e4m3fn and tuple(w.shape) == (N, K), (w.dtype, w.shape)
        assert tuple(s.shape) == (N // 32, K // 32), s.shape
        return w, s.view(torch.float8_e8m0fnu)
    w = torch.randint(0, 256, (N, K), dtype=torch.uint8, device=dev, generator=g)
    w[(w & 0x7F) == 0x7F] = 0  # no e4m3 NaN codes
    s = torch.randint(127 - 14, 127 - 3, (N // 32, K // 32), dtype=torch.uint8, device=dev, generator=g)
    return w.view(torch.float8_e4m3fn), s.view(torch.float8_e8m0fnu)


def activations(m: int, g: torch.Generator) -> torch.Tensor:
    x = torch.randn(m, K, device=dev, generator=g)
    e = torch.randint(-6, 7, (m, K // 32), device=dev, generator=g).float()
    return (x * torch.exp2(e).repeat_interleave(32, dim=1)).to(torch.bfloat16)


def op_part(args, w8: torch.Tensor, s8: torch.Tensor, g: torch.Generator, tag: str) -> bool:
    import flashinfer.gemm.gemm_base as gb
    from mxfp8_gemv_sm120 import mxfp8_gemv

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize, swizzle_mxfp8_scale
    from vllm.utils import flashinfer as vllm_flashinfer

    rows = s8.view(torch.uint8).repeat_interleave(32, dim=0)  # KMxfp8Static's loader for 32 x 32 blocks
    w_sf = swizzle_mxfp8_scale(rows, M=N, K=K)
    sl = [(w8[r * NS : (r + 1) * NS], swizzle_mxfp8_scale(rows[r * NS : (r + 1) * NS].contiguous(), M=NS, K=K))
          for r in range(TP)]
    chunk = w_sf.numel() // TP
    ok = all(torch.equal(sl[r][1], w_sf[r * chunk : (r + 1) * chunk]) for r in range(TP))
    print(f"[{tag}] swizzled scales of the slices = byte ranges of the full layout: {ok}")

    mod = gb._load_gemm_sm120_mxfp8_module()
    ws = gb._get_cache_buf("mm_mxfp8_workspace", gb.DEFAULT_WORKSPACE_SIZE, dev)
    n_tac = mod.mxfp8_gemm_tactic_num()

    def cutlass(a, a_sf, w, sf, tactic):
        out = torch.empty(a.shape[0], w.shape[0], dtype=torch.bfloat16, device=dev)
        mod.mxfp8_gemm(a, w, a_sf, sf, out, ws, tactic)
        return out

    def served_cutlass(a, a_sf, w, sf):
        return vllm_flashinfer.mm_mxfp8(a, w.t(), a_sf, sf, out_dtype=torch.bfloat16, backend="cutlass")

    bad_split, bad_tactic, n_cmp = [], [], 0
    with torch.inference_mode():
        for m in DECODE_MS + PREFILL_MS:
            a, a_sf = mxfp8_e4m3_quantize(activations(m, g), is_sf_swizzled_layout=True)
            paths = {f"cutlass{t}": (lambda w, sf, t=t: cutlass(a, a_sf, w, sf, t)) for t in range(n_tac)}
            paths["mm_mxfp8"] = lambda w, sf: served_cutlass(a, a_sf, w, sf)
            if m <= GEMV_MAX_M:
                paths["gemv"] = lambda w, sf: mxfp8_gemv(a, a_sf, w, sf)
            fulls = {}
            for name, fn in paths.items():
                full = fn(w8, w_sf)
                split = torch.cat([fn(w, sf) for w, sf in sl], dim=1)
                n_cmp += 1
                if not torch.equal(bits(full), bits(split)):
                    bad_split.append((m, name, (bits(full) != bits(split)).sum().item()))
                fulls[name] = full
            ref = fulls["cutlass0"]
            for name, full in fulls.items():
                if name != "gemv" and not torch.equal(bits(full), bits(ref)):
                    bad_tactic.append((m, name))
    print(f"[{tag}] {n_cmp} (tokens, kernel) cases, {n_tac} CUTLASS tactics, GEMV for 1..{GEMV_MAX_M} tokens: "
          f"split != full in {len(bad_split)}, CUTLASS tactics disagree in {len(bad_tactic)}")
    for case in bad_split[:10]:
        print(f"    split differs: tokens={case[0]} kernel={case[1]} outputs={case[2]}")
    for case in bad_tactic[:10]:
        print(f"    tactic differs from tactic 0: tokens={case[0]} kernel={case[1]}")
    ok &= not bad_split and not bad_tactic
    if args.bench:
        bench(quant=mxfp8_e4m3_quantize, gemv=mxfp8_gemv, served_cutlass=served_cutlass, w8=w8, w_sf=w_sf, sl=sl, g=g)
    return ok


def graph_time(calls, per_graph: int, reps: int = 15) -> float:
    """us per call: `per_graph` calls (cycled from `calls`) captured in one CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls:
            c()
        s.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            for i in range(per_graph):
                calls[i % len(calls)]()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        graph.replay()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) * 1000 / per_graph)
    return statistics.median(ts)


def bench(quant, gemv, served_cutlass, w8, w_sf, sl, g) -> None:
    print(f"  bench on {torch.cuda.get_device_name(0)}: us per call, 20 calls per graph, cold weights "
          f"(full {N * K / 2**20:.0f} MiB; the {TP} slices in rotation)")
    with torch.inference_mode():
        for m in (1, 6, 8, 12, 16, 48, 4096):
            a, a_sf = quant(activations(m, g), is_sf_swizzled_layout=True)
            fn = gemv if m <= GEMV_MAX_M else served_cutlass
            kernel = "GEMV" if m <= GEMV_MAX_M else "CUTLASS"
            t_full = graph_time([lambda: fn(a, a_sf, w8, w_sf)], 20)
            t_split = graph_time([lambda w=w, sf=sf: fn(a, a_sf, w, sf) for w, sf in sl], 20)
            print(f"    tokens={m:4d} {kernel:7s} {N} cols {t_full:7.1f} us   {NS} cols {t_split:7.1f} us   "
                  f"({N * K / t_full / 1e6:.2f} / {NS * K / t_split / 1e6:.2f} TB/s)")


@contextlib.contextmanager
def simulated_rank(rank: int, size: int):
    """TP rank / world size as read by engram.py and the layer code; the all-gather is the identity."""
    import vllm.model_executor.layers.linear as L
    import vllm.model_executor.parameter as P
    import vllm.models.deepseek_v41.common.engram as E

    saved = []
    for mod in (E, L, P):
        for name, value in (("get_tensor_model_parallel_rank", rank), ("get_tensor_model_parallel_world_size", size)):
            if hasattr(mod, name):
                saved.append((mod, name, getattr(mod, name)))
                setattr(mod, name, lambda v=value: v)
    saved.append((E, "tensor_model_parallel_all_gather", E.tensor_model_parallel_all_gather))
    E.tensor_model_parallel_all_gather = lambda t, dim=-1: t
    try:
        yield
    finally:
        for mod, name, fn in saved:
            setattr(mod, name, fn)


def split_wkv(E, wkv, x, size: int = TP, sp: bool = False) -> list[torch.Tensor]:
    """_moet_engram_wkv on each simulated rank (fresh part per rank: the cache is per process in serving)."""
    outs = []
    for r in range(size):
        wkv.__dict__.pop("_moet_wkv_part", None)
        with simulated_rank(r, size):
            outs.append(E._moet_engram_wkv(wkv, x, sp))
    wkv.__dict__.pop("_moet_wkv_part", None)
    return outs


def integration_part(args, w8: torch.Tensor, s8: torch.Tensor, g: torch.Generator, tag: str, qc) -> bool:
    import vllm.models.deepseek_v41.common.engram as E
    from vllm.model_executor.layers.linear import ReplicatedLinear

    rep = ReplicatedLinear(K, N, bias=False, quant_config=qc, return_bias=False, prefix="model.layers.1.engram.wkv")
    for param, tensor in ((rep.weight, w8), (rep.weight_scale, s8)):
        param.weight_loader(param, tensor)  # as DeepseekV4Model.load_weights (".scale" -> ".weight_scale")
    rep.quant_method.process_weights_after_loading(rep)
    qm = rep.quant_method
    print(f"[{tag}] {type(qm).__name__} / {type(qm.fmt).__name__} / {type(qm.kernel).__name__}")
    ok = True
    chunk = rep.weight_scale.numel() // TP
    for r in range(TP):
        with simulated_rank(r, TP):
            part = E._moet_wkv_part(rep, TP)
        ok &= (part is not None
               and part.weight.data_ptr() == rep.weight.data_ptr() + r * NS * K
               and tuple(part.weight.shape) == (NS, K)
               and part.weight_scale.data_ptr() == rep.weight_scale.data_ptr() + r * chunk
               and part.weight_scale.numel() == chunk)
    print(f"[{tag}] rank slices are views of the layer's weight rows and scale byte ranges: {ok}")

    E._MOET_WKV_TP = (True, max(PREFILL_MS))  # split at every size for the bit check
    bad = []
    with torch.inference_mode():
        for m in DECODE_MS + PREFILL_MS:
            for _ in range(2 if m <= 64 else 1):
                x = activations(m, g)
                if not torch.equal(bits(rep(x)), bits(torch.cat(split_wkv(E, rep, x), dim=1))):
                    bad.append(m)
        print(f"[{tag}] eager: {len(DECODE_MS) * 2 + len(PREFILL_MS)} inputs, 1..4096 tokens, "
              f"split != replicated in {len(bad)} {sorted(set(bad))[:10]}")
        ok &= not bad
        E._MOET_WKV_TP = (True, 64)
        gbad = []
        for m in (1, 6, 8, 12, 16, 24, 48, 64):
            xs = activations(m, g)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                rep(xs), split_wkv(E, rep, xs)  # warm up outside capture (FlashInfer tactic lookup)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=side):
                    out_split = torch.cat(split_wkv(E, rep, xs), dim=1)
            torch.cuda.synchronize()
            for _ in range(3):
                xs.copy_(activations(m, g))
                graph.replay()
                torch.cuda.synchronize()
                if not torch.equal(bits(out_split), bits(rep(xs))):
                    gbad.append(m)
        print(f"[{tag}] CUDA graph replays (1..64 tokens): split != replicated in {len(gbad)} {gbad[:10]}")
        ok &= not gbad
    del rep
    torch.cuda.empty_cache()
    return ok


def dispatch_part(qc) -> bool:
    import vllm.models.deepseek_v41.common.engram as E
    from vllm.model_executor.layers.linear import ReplicatedLinear

    if not hasattr(E, "_moet_engram_wkv"):
        print("engram.py is not patched (patch_vllm_engram_wkv_tp.py)")
        return False
    g = torch.Generator(device=dev).manual_seed(3)
    w8, s8 = weights(SimpleArgs(), 0, g)
    mx = ReplicatedLinear(K, N, bias=False, quant_config=qc, return_bias=False, prefix="model.layers.1.engram.wkv")
    for param, tensor in ((mx.weight, w8), (mx.weight_scale, s8)):
        param.weight_loader(param, tensor)
    mx.quant_method.process_weights_after_loading(mx)
    bf = ReplicatedLinear(K, N, bias=False, quant_config=None, return_bias=False, prefix="model.layers.1.engram.wkv")
    bf.weight.data.normal_(0, 0.02, generator=g)
    ok = True
    with torch.inference_mode():
        for label, layer, cfg, tokens, size, sp, want in (
            ("TP=4, 6 tokens", mx, (True, 64), 6, TP, False, NS),
            ("TP=4, 64 tokens", mx, (True, 64), 64, TP, False, NS),
            ("TP=4, 65 tokens (above the threshold)", mx, (True, 64), 65, TP, False, N),
            ("TP=4, MAX_TOKENS=16, 17 tokens", mx, (True, 16), 17, TP, False, N),
            ("TP=1", mx, (True, 64), 6, 1, False, N),
            ("TP=4, sequence parallel", mx, (True, 64), 6, TP, True, N),
            ("TP=4, VLLM_MOET_ENGRAM_WKV_TP=0", mx, (False, 64), 6, TP, False, N),
            ("TP=4, unquantized wkv", bf, (True, 64), 6, TP, False, N),
        ):
            E._MOET_WKV_TP = cfg
            x = activations(tokens, g)
            out = split_wkv(E, layer, x, size, sp)[0]
            good = out.shape[-1] == want and (want == NS or torch.equal(bits(out), bits(layer(x))))
            print(f"  {label}: {'split' if out.shape[-1] == NS else 'replicated'} {'OK' if good else 'UNEXPECTED'}")
            ok &= good
    del mx, bf
    code = "import vllm.models.deepseek_v41.common.engram as E\nprint(E._moet_wkv_tp())\n"
    for env, expect in (({}, "(True, 64)"), ({"VLLM_MOET_ENGRAM_WKV_TP": "0"}, "(False, 64)"),
                        ({"VLLM_MOET_ENGRAM_WKV_TP_MAX_TOKENS": "16"}, "(True, 16)")):
        base = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_MOET_ENGRAM_WKV_TP")}
        r = subprocess.run([sys.executable, "-c", code], env={**base, **env}, capture_output=True, text=True)
        last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
        good = r.returncode == 0 and last == expect
        print(f"  environment {env or '(default)'}: {last} {'OK' if good else 'UNEXPECTED'}")
        ok &= good
    E._MOET_WKV_TP = None
    return ok


class SimpleArgs:
    checkpoint = ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("op", "integration", "all"), default="all")
    ap.add_argument("--checkpoint", default="", help="checkpoint directory: test layers 1 and 14's own wkv")
    ap.add_argument("--layers", default="1,14")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    print(f"device={torch.cuda.get_device_name(0)} VLLM_MOET_GEMV_IMPL={os.environ.get('VLLM_MOET_GEMV_IMPL', 'v3')}")
    g = torch.Generator(device=dev).manual_seed(41)
    sources = [(f"layer {i}", int(i)) for i in args.layers.split(",")] if args.checkpoint else [("synthetic", 0)]
    ok = True
    if args.part in ("op", "all"):
        for tag, layer_id in sources:
            w8, s8 = weights(args, layer_id, g)
            ok &= op_part(args, w8, s8, g, tag)
            del w8, s8
    if args.part in ("integration", "all"):
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import init_distributed_environment, initialize_model_parallel
        from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config
        from vllm.utils.torch_utils import set_default_torch_dtype

        import vllm.models.deepseek_v41.common.engram as E

        if not hasattr(E, "_moet_engram_wkv"):
            print("engram.py is not patched (patch_vllm_engram_wkv_tp.py)")
            return 1
        qc = DeepseekV4FP8Config.from_config({"quant_method": "fp8", "activation_scheme": "dynamic",
                                              "weight_block_size": [32, 32], "scale_fmt": "ue8m0",
                                              "expert_dtype": "fp4"})
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with set_current_vllm_config(VllmConfig()), set_default_torch_dtype(torch.bfloat16), torch.device(dev):
            init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}",
                                         local_rank=0, backend="nccl")
            initialize_model_parallel(tensor_model_parallel_size=1)
            for tag, layer_id in sources:
                w8, s8 = weights(args, layer_id, g)
                ok &= integration_part(args, w8, s8, g, tag, qc)
                del w8, s8
            print("[dispatch] _moet_engram_wkv under a simulated TP:")
            ok &= dispatch_part(qc)
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
