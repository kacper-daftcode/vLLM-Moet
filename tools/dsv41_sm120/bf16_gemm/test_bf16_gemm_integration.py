#!/usr/bin/env python3
"""The patched vLLM call sites (patch_vllm_bf16_gemm_sm120.py) against the ops they replace, inside the image.

Real layers - ReplicatedLinear 5120 -> 32 (indexer weights_proj), 512 -> 128 (wk), GateLinear 5120 -> 384 / 128
(router, drafter router), the compressor's torch.mm 5120 -> 1024 / 512 - go through the patched helpers the way the
model calls them: a profile-run-sized eager call first (256 tokens: cuBLAS, triggers the calibration; wk is
registered as at indexer construction and calibrated then), then every token count 1..16 eagerly and the decode
counts replayed from a CUDA graph captured with the sites on a side stream, each compared bit for bit with the
original op. Then the switches: VLLM_MOET_BF16_GEMM=0 (module never resolved) and an empty site list.

usage (in the serving image, patched): python3 test_bf16_gemm_integration.py
"""

from __future__ import annotations

import os
import subprocess
import sys

os.environ.setdefault("VLLM_MOET_BF16_GEMM_SITES", "indexer,wk,compressor,router")

import torch  # noqa: E402

dev = torch.device("cuda")


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def build_layers():
    import socket

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.linear import ReplicatedLinear

    g = torch.Generator(device=dev).manual_seed(5)
    layers = {}
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with set_current_vllm_config(VllmConfig()), torch.device(dev):
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", local_rank=0, backend="nccl"
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        for name, (k, n) in {"weights_proj": (5120, 32), "wk": (512, 128)}.items():
            lin = ReplicatedLinear(k, n, bias=False, quant_config=None, params_dtype=torch.bfloat16, prefix=name)
            lin.weight.data.copy_((torch.randn(n, k, device=dev, generator=g) * 0.05).to(torch.bfloat16))
            lin.quant_method.process_weights_after_loading(lin)
            layers[name] = lin
        for name, n in {"router": 384, "drafter_router": 128}.items():
            gate = GateLinear(5120, n, bias=False, out_dtype=torch.float32, params_dtype=torch.bfloat16, prefix=name)
            gate.weight.data.copy_((torch.randn(n, 5120, device=dev, generator=g) * 0.05).to(torch.bfloat16))
            layers[name] = gate
    comp = {n: (torch.randn(n, 5120, device=dev, generator=g) * 0.05).to(torch.bfloat16) for n in (1024, 512)}
    return layers, comp


def main() -> int:
    import vllm.model_executor.layers.fused_moe.router.gate_linear as G
    import vllm.models.deepseek_v41.attention as A

    assert hasattr(A, "_moet_bf16_linear") and hasattr(G, "_moet_bf16_mm"), "vLLM files not patched"
    mod = A._moet_bf16()
    assert mod is not None and G._moet_bf16() is mod, "bf16_gemm module did not load"
    print(f"module {mod.__file__}, sites {sorted(mod.enabled_sites())}")
    layers, comp = build_layers()
    assert layers["router"].allow_cublas_router_gemm, "GateLinear does not take tier 4 here"
    A._moet_bf16_register(128, 512, False, 512, "wk")  # as DeepseekV4Indexer.__init__ (owns_k)

    def patched(name, x):
        if name in ("weights_proj", "wk"):
            return A._moet_bf16_linear(layers[name], x, "indexer" if name == "weights_proj" else name)
        if name in ("router", "drafter_router"):
            return layers[name](x)[0]
        return A._moet_bf16_mm(x, comp[int(name.split("_")[1])], True, "compressor")

    def original(name, x):
        if name in ("weights_proj", "wk"):
            return layers[name].quant_method.apply(layers[name], x, None)
        w = layers[name].weight if name in ("router", "drafter_router") else comp[int(name.split("_")[1])]
        return torch.mm(x, w.T, out_dtype=torch.float32)

    names = ["weights_proj", "wk", "router", "drafter_router", "compressor_1024", "compressor_512"]
    kin = {nm: (512 if nm == "wk" else 5120) for nm in names}
    g = torch.Generator(device=dev).manual_seed(11)
    ok = True
    with torch.inference_mode():
        for nm in names:  # profile-run sized call: cuBLAS, calibrates the shape (and the registered wk)
            x = torch.randn(256, kin[nm], device=dev, generator=g).to(torch.bfloat16)
            ok &= torch.equal(bits(patched(nm, x)), bits(original(nm, x)))
        for key, cases in mod.table().items():
            print(f"  calibrated {key}: {len(cases)} token counts")
        for nm in names:
            for m in range(1, 17):
                for _ in range(4):
                    x = torch.randn(m, kin[nm], device=dev, generator=g).to(torch.bfloat16)
                    if not torch.equal(bits(patched(nm, x)), bits(original(nm, x))):
                        print(f"  eager {nm} M={m}: differs from the original op")
                        ok = False
        n_kernel = sum(1 for cases in mod.table().values() for _ in cases)
        print(f"  {n_kernel} (shape, token count) cases on the kernels")
        # graph: the attention sites on a side stream (vLLM's aux streams), the router on the capture stream
        for m in (6, 12):
            xs = {nm: torch.randn(m, kin[nm], device=dev, generator=g).to(torch.bfloat16) for nm in names}
            outs = {}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fork = torch.cuda.Event()
                fork.record()
                side.wait_event(fork)
                with torch.cuda.stream(side):
                    for nm in ("weights_proj", "compressor_1024", "compressor_512"):
                        outs[nm] = patched(nm, xs[nm])
                for nm in ("wk", "router", "drafter_router"):
                    outs[nm] = patched(nm, xs[nm])
                join = torch.cuda.Event()
                join.record(side)
                torch.cuda.current_stream().wait_event(join)
            for rep in range(3):
                for nm in names:
                    xs[nm].copy_(torch.randn(m, kin[nm], device=dev, generator=g).to(torch.bfloat16))
                graph.replay()
                torch.cuda.synchronize()
                for nm in names:
                    if not torch.equal(bits(outs[nm]), bits(original(nm, xs[nm]))):
                        print(f"  graph M={m} replay {rep} {nm}: differs from the original op")
                        ok = False
    # switches, in fresh processes
    for env, expect in (({"VLLM_MOET_BF16_GEMM": "0"}, "None"), ({"VLLM_MOET_BF16_GEMM_SITES": ""}, "frozenset()")):
        code = (
            "import vllm.models.deepseek_v41.attention as A\n"
            "m = A._moet_bf16()\n"
            "print(None if m is None else m.enabled_sites())\n"
        )
        r = subprocess.run([sys.executable, "-c", code], env={**os.environ, **env}, capture_output=True, text=True)
        last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
        good = r.returncode == 0 and last == expect
        print(f"  {env}: {last} {'OK' if good else 'UNEXPECTED'}")
        ok &= good
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
