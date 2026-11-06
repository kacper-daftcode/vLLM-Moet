#!/usr/bin/env python3
"""Decode-shaped GEMM microbench for Ripple-65B TP4 on one RTX PRO 6000 (run inside the ripple image).

NVFP4 (ModelOpt W4A4) layers per rank: gate_up [20480 x 5120], down [5120 x 10240] - every vLLM NvFp4LinearKernel that
claims sm_120 (flashinfer_cutlass = served, cutlass, flashinfer_b12x, b12x, flashinfer_trtllm, flashinfer_cudnn, marlin
W4A16), through the kernel class on a fake layer exactly as vLLM runs it (activation quant included), timed as a CUDA graph
of ITERS back-to-back calls; outputs compared bit for bit with flashinfer_cutlass.
BF16 layers per rank: qkg [5376 x 5120], v [256 x 5120], o [5120 x 2560], drafter gate_up [20480 x 5120], lm_head
[25088 x 5120] - torch.matmul with cuBLAS (default) vs cuBLASLt preferred, bits compared.

usage (inside the ripple image, one free GPU): python3 decode_gemm_bench.py [--ms 16,96,256] [--iters 50] [--out /out/gemm_bench.json]
"""
import argparse
import json
import time

import torch
import torch.nn as nn

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig

BW = 2.11e12  # bytes/s: 512-bit GDDR7 at the 16.5 GHz memory clock these cards run (stock 14 GHz = 1.79 TB/s)


L2_DEFEAT = 300 << 20  # rotate weight copies so the graph loop streams > 300 MB (L2 is 128 MB)


def ncopies(wbytes):
    return max(2, -(-L2_DEFEAT // wbytes))


def time_graph(fns, iters, reps=5, autotune=False):
    """CUDA graph of `iters` calls cycling over `fns` (one per weight copy), replayed `reps` times: min µs per call."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        if autotune:
            from flashinfer.autotuner import autotune as fi_autotune
            with fi_autotune(True):
                for fn in fns:
                    fn()
        for fn in fns:
            for _ in range(2):
                fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(iters):
            fns[i % len(fns)]()
    torch.cuda.synchronize()
    best = None
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        dt = e0.elapsed_time(e1) * 1e3 / iters
        best = dt if best is None else min(best, dt)
    return best


def make_nvfp4_layer(n, k, gen):
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=gen),
                                requires_grad=False)
    ws = (torch.rand(n, k // 16, device="cuda", generator=gen) * 0.5 + 0.25).to(torch.float8_e4m3fn)
    layer.weight_scale = nn.Parameter(ws, requires_grad=False)
    layer.weight_global_scale = nn.Parameter(torch.tensor(0.0123, device="cuda"), requires_grad=False)
    layer.input_global_scale = nn.Parameter(torch.tensor(0.0456, device="cuda"), requires_grad=False)
    layer.input_global_scale_inv = nn.Parameter(1.0 / layer.input_global_scale.data, requires_grad=False)
    layer.alpha = nn.Parameter(layer.input_global_scale.data * layer.weight_global_scale.data, requires_grad=False)
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.params_dtype = torch.bfloat16
    layer.weight_scale_2 = nn.Parameter(layer.weight_global_scale.data.clone(), requires_grad=False)
    layer.input_scale = nn.Parameter(layer.input_global_scale.data.clone(), requires_grad=False)
    return layer


def nvfp4_kernels():
    from vllm.model_executor.kernels.linear.nvfp4 import flashinfer as fi
    from vllm.model_executor.kernels.linear.nvfp4 import cutlass as cu
    from vllm.model_executor.kernels.linear.nvfp4 import b12x as bx
    from vllm.model_executor.kernels.linear.nvfp4 import marlin as ma
    return [
        ("flashinfer_cutlass", fi.FlashInferCutlassNvFp4LinearKernel),
        ("cutlass", cu.CutlassNvFp4LinearKernel),
        ("flashinfer_b12x", fi.FlashInferB12xNvFp4LinearKernel),
        ("b12x", bx.B12xNvFp4LinearKernel),
        ("flashinfer_trtllm", fi.FlashInferTrtllmNvFp4LinearKernel),
        ("flashinfer_cudnn", fi.FlashInferCudnnNvFp4LinearKernel),
        ("marlin_w4a16", ma.MarlinNvFp4LinearKernel),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="16,96,256")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="/out/gemm_bench.json")
    a = ap.parse_args()
    ms = [int(x) for x in a.ms.split(",")]
    dev = torch.cuda.get_device_name(0)
    print("device", dev, torch.cuda.get_device_capability(0), "torch", torch.__version__)
    res = {"device": dev, "nvfp4": [], "bf16": []}

    # ---------------- NVFP4 ----------------
    shapes = [("gate_up", 20480, 5120), ("down", 5120, 10240)]
    for name, n, k in shapes:
        gen = torch.Generator(device="cuda"); gen.manual_seed(1234)
        base = make_nvfp4_layer(n, k, gen)
        wbytes = n * k // 2 + n * k // 16
        ref = {}
        for bname, kcls in nvfp4_kernels():
            ok, why = kcls.is_supported(120)
            if not ok:
                print(f"  {name:8s} {bname:18s} unsupported: {why}"); continue
            try:
                kern = kcls(NvFp4LinearLayerConfig())
                layers = []
                for c in range(ncopies(wbytes)):
                    layer = make_nvfp4_layer(n, k, torch.Generator(device="cuda").manual_seed(1234 + c))
                    kern.process_weights_after_loading(layer)
                    layers.append(layer)
                layer = layers[0]
            except Exception as e:  # noqa: BLE001
                print(f"  {name:8s} {bname:18s} prepare failed: {type(e).__name__}: {str(e)[:160]}"); continue
            for m in ms:
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=torch.Generator(device="cuda").manual_seed(7 + m))
                try:
                    y = kern.apply_weights(layer, x)
                    torch.cuda.synchronize()
                    us = time_graph([(lambda L=L: kern.apply_weights(L, x)) for L in layers], a.iters,
                                    autotune=bname.startswith("flashinfer"))
                    y = kern.apply_weights(layer, x)  # after autotune: the tactic the server would use
                except Exception as e:  # noqa: BLE001
                    print(f"  {name:8s} {bname:18s} M={m:4d} failed: {type(e).__name__}: {str(e)[:160]}"); continue
                if bname == "flashinfer_cutlass":
                    ref[m] = y.clone(); bits = "ref"
                elif m in ref:
                    same = torch.equal(y.view(torch.int16), ref[m].view(torch.int16))
                    d = (y.float() - ref[m].float()).abs().max().item()
                    bits = "IDENTICAL" if same else f"differs max|d| {d:.3e}"
                else:
                    bits = "-"
                eff = wbytes / (us * 1e-6) / BW * 100
                print(f"  {name:8s} {bname:18s} M={m:4d} {us:8.2f} us  {eff:5.1f} % of {BW/1e12:.2f} TB/s  bits vs fi_cutlass: {bits}", flush=True)
                res["nvfp4"].append(dict(shape=name, n=n, k=k, backend=bname, m=m, us=us, bw_eff=eff, bits=bits))
            del layer, layers, kern
            torch.cuda.empty_cache()

    # ---------------- BF16 ----------------
    shapes = [("qkg", 5376, 5120), ("v", 256, 5120), ("o", 5120, 2560), ("draft_gate_up", 20480, 5120), ("lm_head", 25088, 5120)]
    for name, n, k in shapes:
        wbytes = n * k * 2
        ws_ = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=torch.Generator(device="cuda").manual_seed(99 + c))
               for c in range(ncopies(wbytes))]
        w = ws_[0]
        for m in ms:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=torch.Generator(device="cuda").manual_seed(7 + m))
            ref = None
            for lib in ("cublas", "cublaslt"):
                torch.backends.cuda.preferred_blas_library(lib)
                y = torch.nn.functional.linear(x, w)
                torch.cuda.synchronize()
                us = time_graph([(lambda W=W: torch.nn.functional.linear(x, W)) for W in ws_], a.iters)
                if ref is None:
                    ref = y.clone(); bits = "ref"
                else:
                    same = torch.equal(y.view(torch.int16), ref.view(torch.int16))
                    d = (y.float() - ref.float()).abs().max().item()
                    bits = "IDENTICAL" if same else f"differs max|d| {d:.3e}"
                eff = wbytes / (us * 1e-6) / BW * 100
                print(f"  {name:14s} {lib:9s} M={m:4d} {us:8.2f} us  {eff:5.1f} %  bits vs cublas: {bits}", flush=True)
                res["bf16"].append(dict(shape=name, n=n, k=k, lib=lib, m=m, us=us, bw_eff=eff, bits=bits))
            torch.backends.cuda.preferred_blas_library("cublas")
        del ws_, w
        torch.cuda.empty_cache()
    # ---------------- MXFP8 (candidate for the attention projections) ----------------
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
    from vllm.model_executor.kernels.linear.mxfp8 import flashinfer as mfi, b12x as mbx, marlin as mma, emulation as mem
    mx_kernels = [("fi_cutlass_mxfp8", mfi.FlashInferCutlassMxfp8LinearKernel), ("b12x_mxfp8", mbx.B12xMxfp8LinearKernel),
                  ("marlin_mxfp8_w8a16", mma.MarlinMxfp8LinearKernel)]
    res["mxfp8"] = []
    shapes = [("qkg", 5376, 5120), ("v", 256, 5120), ("o", 5120, 2560), ("qkgv", 5632, 5120)]
    for name, n, k in shapes:
        wbytes = n * k + n * k // 32
        bf16_us = None
        for r in res["bf16"]:
            pass
        for bname, kcls in mx_kernels:
            ok, why = kcls.is_supported(120)
            if not ok:
                print(f"  {name:8s} {bname:20s} unsupported: {why}"); continue
            cfg = Mxfp8LinearLayerConfig(weight_shape=(n, k))
            ok, why = kcls.can_implement(cfg)
            if not ok:
                print(f"  {name:8s} {bname:20s} cannot implement: {why}"); continue
            try:
                kern = kcls(cfg)
                layers = []
                for c in range(ncopies(wbytes)):
                    g = torch.Generator(device="cuda").manual_seed(4321 + c)
                    layer = nn.Module()
                    w = torch.randn(n, k, device="cuda", generator=g) * 0.02
                    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import _mxfp8_e4m3_quantize_torch
                    wq, ws = _mxfp8_e4m3_quantize_torch(w)
                    layer.weight = nn.Parameter(wq, requires_grad=False)
                    layer.weight_scale = nn.Parameter(ws.view(n, k // 32).contiguous(), requires_grad=False)
                    layer.params_dtype = torch.bfloat16
                    layer.output_size_per_partition = n; layer.input_size_per_partition = k
                    layer.weight_block_size = [1, 32]
                    kern.process_weights_after_loading(layer)
                    layers.append(layer)
                layer = layers[0]
            except Exception as e:  # noqa: BLE001
                print(f"  {name:8s} {bname:20s} prepare failed: {type(e).__name__}: {str(e)[:200]}"); continue
            for m in ms:
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=torch.Generator(device="cuda").manual_seed(7 + m))
                try:
                    y = kern.apply_weights(layer, x); torch.cuda.synchronize()
                    us = time_graph([(lambda L=L: kern.apply_weights(L, x)) for L in layers], a.iters, autotune=bname.startswith("fi_"))
                except Exception as e:  # noqa: BLE001
                    print(f"  {name:8s} {bname:20s} M={m:4d} failed: {type(e).__name__}: {str(e)[:200]}"); continue
                eff = wbytes / (us * 1e-6) / BW * 100
                print(f"  {name:8s} {bname:20s} M={m:4d} {us:8.2f} us  {eff:5.1f} % of {BW/1e12:.2f} TB/s", flush=True)
                res["mxfp8"].append(dict(shape=name, n=n, k=k, backend=bname, m=m, us=us, bw_eff=eff))
            del layer, layers, kern
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("GEMM_BENCH_DONE")


if __name__ == "__main__":
    main()
