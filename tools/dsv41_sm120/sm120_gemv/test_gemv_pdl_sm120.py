#!/usr/bin/env python3
"""Launch modes of the sm_120 MXFP8 GEMVs (programmatic dependent launch, L2 evict-first weight
loads): bit-identity against the committed kernels and CUDA-graph timings with cold weights, per
shape and in chains of the served decode graph's neighbour kernels.

Every mode must return the committed kernels' bits (int16 patterns, torch.equal) - v3 for M <= 8
and K % 128 == 0, v1 for M 9-16 and for K % 128 != 0 (shared w2), the grouped wo_a kernel - eagerly,
replayed from CUDA graphs with the producing kernel in front, and against a slow producer that
lets the GEMV launch early (it must neither read the activations before the producer finished nor
write an output that aliases a buffer the producer still reads).

Run inside the serving image on one sm_120 GPU; the reference is this extension's plain launch
(pdl=0, evict=0 - SASS-identical to the kernels served until 2026-10-02) or, with --ref-cu, a
kernel source compiled as a second extension (e.g. the one served before the launch modes):
    git show 139c398:tools/dsv41_sm120/sm120_gemv/mxfp8_gemv_sm120.cu > /tmp/ref/mxfp8_gemv_sm120_ref.cu
    python3 test_gemv_pdl_sm120.py [--ref-cu /tmp/ref/mxfp8_gemv_sm120_ref.cu] [--bench] [--chain]
        [--trace 6] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import zlib
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mxfp8_gemv_sm120 as gemv  # noqa: E402

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)

# DeepSeek-V4.1-Flash dense shapes per TP4 rank (N, K), test_mxfp8_gemv_sm120.py's served set plus
# Engram's wkv slice (6400 of 25600 rows per rank since the -engram image).
DENSE = {
    "q_a+kv_a 1792<-5120": (1792, 5120),
    "q_b+idx 8192<-1280": (8192, 1280),
    "wo_b 5120<-2048": (5120, 2048),
    "shared w13 1152<-5120": (1152, 5120),
    "shared w2 5120<-576": (5120, 576),
    "idx wq_b 4096<-1280": (4096, 1280),
    "engram 6400<-6144": (6400, 6144),
}
GROUPED = "wo_a 2x[1024<-4096]"
N_GROUPS, HEADS_PER_GROUP, NOPE, ROPE, O_LORA = 2, 8, 448, 64, 1024
HEAD_DIM = NOPE + ROPE
WO_A_K = HEADS_PER_GROUP * HEAD_DIM
MAX_POS = 4096
COLD_BYTES = 384 << 20  # weight copies cycled per graph: 3-4x the L2 of GB202 (96 / 128 MB)

RACE_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cstdint>
// A producer that lets PDL dependents launch at once, then spins, then snapshots `watch` and only
// then writes `dst`: a dependent that reads `dst` or writes `watch` before griddepcontrol.wait
// returns gets caught.
__global__ void slow_copy_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, long long n,
                                 const uint8_t* __restrict__ watch, uint8_t* __restrict__ snap, long long nw,
                                 unsigned long long spin_ns) {
  asm volatile("griddepcontrol.launch_dependents;");
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while (t - t0 < spin_ns);
  const long long stride = (long long)gridDim.x * blockDim.x;
  const long long i0 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  for (long long i = i0; i < nw; i += stride) snap[i] = *((volatile const uint8_t*)watch + i);
  __syncthreads();
  for (long long i = i0; i < n; i += stride) dst[i] = src[i];
}
void slow_copy(torch::Tensor src, torch::Tensor dst, torch::Tensor watch, torch::Tensor snap, int64_t spin_ns) {
  auto stream = c10::cuda::getCurrentCUDAStream(src.get_device()).stream();
  slow_copy_kernel<<<1, 1024, 0, stream>>>(src.data_ptr<uint8_t>(), dst.data_ptr<uint8_t>(), src.numel(),
                                           watch.data_ptr<uint8_t>(), snap.data_ptr<uint8_t>(), watch.numel(),
                                           (unsigned long long)spin_ns);
}
// Negative control: PDL-launched, never waits - reads `in` and writes `out` while the producer spins.
__global__ void no_wait_copy_kernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, long long n) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    out[i] = in[i];
}
void no_wait_copy(torch::Tensor in, torch::Tensor out) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1);
  cfg.blockDim = dim3(256);
  cfg.stream = c10::cuda::getCurrentCUDAStream(in.get_device()).stream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, no_wait_copy_kernel, (const uint8_t*)in.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(),
                     (long long)std::min(in.numel(), out.numel()));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("slow_copy", &slow_copy);
  m.def("no_wait_copy", &no_wait_copy);
}
"""


def load_ref(path: str):
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VLLM_MOET_GEMV_ARCH", "12.0")
    return load(
        name="vllm_moet_mxfp8_gemv_sm120_ref",
        sources=[path],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
    )


def load_race():
    from torch.utils.cpp_extension import load

    d = Path(os.environ.get("TORCH_EXTENSIONS_DIR", tempfile.gettempdir())) / "vllm_moet_gemv_pdl_race_src"
    d.mkdir(parents=True, exist_ok=True)
    src = d / "race.cu"
    if not src.exists() or src.read_text() != RACE_SRC:
        src.write_text(RACE_SRC)
    return load(name="vllm_moet_gemv_pdl_race", sources=[str(src)], extra_cuda_cflags=["-O3", "-std=c++17"])


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(bits(a), bits(b))


def rand_wide(shape, dev, spread: int, base: float = 1.0) -> torch.Tensor:
    """Normal values with a per-32-block magnitude in 2^[-spread, spread] (varied block scales)."""
    x = torch.randn(*shape, device=dev, dtype=torch.float32) * base
    blocks = shape[-1] // MXFP8_BLOCK_SIZE
    e = torch.randint(-spread, spread + 1, (*shape[:-1], blocks, 1), device=dev).float()
    x = (x.view(*shape[:-1], blocks, MXFP8_BLOCK_SIZE) * torch.exp2(e)).view(*shape)
    return x.to(torch.bfloat16)


def quant_weight(w: torch.Tensor):
    """The dense layer's weight as FlashInferCutlassMxfp8LinearKernel stores it: e4m3 + F8_128x4."""
    N, K = w.shape
    q, sf = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    return q, swizzle_mxfp8_scale(sf.view(N, -1), M=N, K=K)


def quant_weight_rowmajor(w: torch.Tensor):
    N, K = w.shape
    q, sf = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    return q, sf.view(N, K // MXFP8_BLOCK_SIZE).contiguous()


def quant_act(x: torch.Tensor):
    return mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)


def rope_cache(dev) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device=dev, dtype=torch.float32) / ROPE))
    freqs = torch.outer(torch.arange(MAX_POS, device=dev, dtype=torch.float32), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def inv_rope_quant(o, positions, cos_sin):
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

    q, sf = fused_inv_rope_fp8_quant(
        o, positions, cos_sin, n_groups=N_GROUPS, heads_per_group=HEADS_PER_GROUP, nope_dim=NOPE,
        rope_dim=ROPE, quant_group_size=MXFP8_BLOCK_SIZE, tma_aligned_scales=True, quantize=True,
    )
    return q.transpose(0, 1), sf.transpose(0, 1)  # the buffers the grouped GEMV takes


def parse_modes(s: str) -> list[tuple[int, int]]:
    out = []
    for item in s.split(","):
        p, f = item.split(":")
        out.append((int(p), int(f)))
    return out


def mode_name(m) -> str:
    return "ref" if m is None else f"pdl{m[0]}/ev{m[1]}"


# ----------------------------------------------------------------------------------------------
# CUDA-graph timing


def graph_time(calls, iters: int):
    """Capture the callables in order into one CUDA graph on a side stream; return (median us per
    replay, graph, outputs). The outputs stay valid while the graph lives."""
    for c in calls[: min(3, len(calls))]:
        c()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls[: min(2, len(calls))]:
            c()
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            outs = [c() for c in calls]
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) * 1000.0)
    return statistics.median(ts), g, outs


class Kernels:
    """The reference and the new extension behind one call signature. The reference is the kernel
    source given by --ref-cu (the one served before the launch modes) or, without it, this
    extension's plain launch (pdl=0, evict=0: the same SASS as the kernels served until 2026-10-02)."""

    def __init__(self, ref_mod):
        self.new = gemv._ext()
        self.ref = ref_mod

    def dense(self, mode):
        if mode is None:
            if self.ref is None:
                return lambda a, s, w, ws: self.new.mxfp8_gemv(a, s, w, ws, 0, 0)
            return lambda a, s, w, ws: self.ref.mxfp8_gemv(a, s, w, ws)
        p, f = mode
        return lambda a, s, w, ws: self.new.mxfp8_gemv(a, s, w, ws, p, f)

    def grouped(self, mode):
        if mode is None:
            if self.ref is None:
                return lambda a, s, w, ws: self.new.mxfp8_gemv_grouped(a, s, w, ws, 0, 0)
            return lambda a, s, w, ws: self.ref.mxfp8_gemv_grouped(a, s, w, ws)
        p, f = mode
        return lambda a, s, w, ws: self.new.mxfp8_gemv_grouped(a, s, w, ws, p, f)


# ----------------------------------------------------------------------------------------------
# bit-identity


def check_dense(kern: Kernels, modes, ms, dev, kinds=("normal", "wide")) -> int:
    fails = 0
    for name, (N, K) in DENSE.items():
        for kind in kinds:
            torch.manual_seed(zlib.crc32(f"{name}/{kind}".encode()))
            wf = (torch.randn(N, K, device=dev) * 0.05).bfloat16() if kind == "normal" else rand_wide((N, K), dev, 10, 0.05)
            w, ws = quant_weight(wf)
            for M in ms:
                x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) if kind == "normal" else rand_wide((M, K), dev, 12)
                a, s = quant_act(x)
                ref = kern.dense(None)(a, s, w, ws)
                bad = []
                for m in modes:
                    out = kern.dense(m)(a, s, w, ws)
                    if not same(out, ref):
                        bad.append(mode_name(m))
                # graph: quantize -> GEMV (programmatic edge), then fresh activations in the same buffer
                xs = x.clone()
                fns = [kern.dense(None)] + [kern.dense(m) for m in modes]
                calls = [(lambda f=f: f(*quant_act(xs), w, ws)) for f in fns]
                _, g, outs = graph_time(calls, 1)
                for rep in range(2):
                    if rep:
                        xs.copy_(torch.randn_like(xs) if kind == "normal" else rand_wide((M, K), dev, 12))
                        g.replay()
                        torch.cuda.synchronize()
                    eager = kern.dense(None)(*quant_act(xs), w, ws)
                    if not same(outs[0], eager):
                        bad.append(f"graph-ref#{rep}")
                    for m, o in zip(modes, outs[1:]):
                        if not same(o, eager):
                            bad.append(f"graph-{mode_name(m)}#{rep}")
                del g
                ok = not bad
                fails += 0 if ok else 1
                path = "v3" if (M <= 8 and K % 128 == 0) else "v1"
                print(f"[{'ok ' if ok else 'BAD'}] {name:24s} {kind:6s} M={M:2d} ({path}) "
                      f"{len(modes)} modes eager + graph x2{'' if ok else ' DIFF: ' + ', '.join(bad)}")
    return fails


def grouped_inputs(T, dev, cos_sin, kind):
    o = (torch.randn(T, N_GROUPS * HEADS_PER_GROUP, HEAD_DIM, device=dev, dtype=torch.bfloat16)
         if kind == "normal" else rand_wide((T, N_GROUPS * HEADS_PER_GROUP, HEAD_DIM), dev, 10))
    pos = torch.randint(0, MAX_POS, (T,), device=dev, dtype=torch.int64)
    return o, pos


def check_grouped(kern: Kernels, modes, ts, dev, kinds=("normal", "wide")) -> int:
    fails = 0
    cos_sin = rope_cache(dev)
    N = N_GROUPS * O_LORA
    for kind in kinds:
        torch.manual_seed(11 if kind == "normal" else 12)
        wf = (torch.randn(N, WO_A_K, device=dev) * 0.05).bfloat16() if kind == "normal" else rand_wide((N, WO_A_K), dev, 10, 0.05)
        w, ws = quant_weight_rowmajor(wf)
        for T in ts:
            o, pos = grouped_inputs(T, dev, cos_sin, kind)
            a, s = inv_rope_quant(o, pos, cos_sin)
            ref = kern.grouped(None)(a, s, w, ws)
            bad = [mode_name(m) for m in modes if not same(kern.grouped(m)(a, s, w, ws), ref)]
            os_ = o.clone()
            fns = [kern.grouped(None)] + [kern.grouped(m) for m in modes]
            calls = [(lambda f=f: f(*inv_rope_quant(os_, pos, cos_sin), w, ws)) for f in fns]
            _, g, outs = graph_time(calls, 1)
            for rep in range(2):
                if rep:
                    os_.copy_(grouped_inputs(T, dev, cos_sin, kind)[0])
                    g.replay()
                    torch.cuda.synchronize()
                eager = kern.grouped(None)(*inv_rope_quant(os_, pos, cos_sin), w, ws)
                if not same(outs[0], eager):
                    bad.append(f"graph-ref#{rep}")
                for m, out in zip(modes, outs[1:]):
                    if not same(out, eager):
                        bad.append(f"graph-{mode_name(m)}#{rep}")
            del g
            ok = not bad
            fails += 0 if ok else 1
            print(f"[{'ok ' if ok else 'BAD'}] {GROUPED:24s} {kind:6s} T={T:2d} "
                  f"{len(modes)} modes eager + graph x2{'' if ok else ' DIFF: ' + ', '.join(bad)}")
    return fails


def check_graph_shapes(kern: Kernels, modes, dev) -> int:
    """GEMVs back to back (two instances of a PDL kernel overlap; with VLLM_MOET_GEMV_V3_SPLITK_MAX
    > 1 they share the per-stream split-K counters) and a GEMV whose only predecessor is on another
    stream (a cross-stream join, no programmatic edge), both captured in CUDA graphs."""
    fails = 0
    torch.manual_seed(21)
    for name, (N, K) in DENSE.items():
        for M in (6, 16):
            ws = [quant_weight((torch.randn(N, K, device=dev) * 0.05).bfloat16()) for _ in range(3)]
            acts = [quant_act(torch.randn(M, K, device=dev, dtype=torch.bfloat16)) for _ in range(3)]
            refs = [kern.dense(None)(a, s, w, wsf) for (a, s), (w, wsf) in zip(acts, ws)]
            bad = []
            for m in modes:
                f = kern.dense(m)
                g, outs = capture(lambda: [f(a, s, w, wsf) for (a, s), (w, wsf) in zip(acts, ws)])
                g.replay()
                torch.cuda.synchronize()
                if not all(same(o, r) for o, r in zip(outs, refs)):
                    bad.append(f"back-to-back:{mode_name(m)}")
                del g, outs
                x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
                expect = kern.dense(None)(*quant_act(x), *ws[0])
                side = torch.cuda.Stream()

                def joined():
                    ev = torch.cuda.Event()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        a, s = quant_act(x)
                        ev.record()
                    torch.cuda.current_stream().wait_event(ev)
                    return f(a, s, *ws[0])

                g, out = capture(joined)
                g.replay()
                torch.cuda.synchronize()
                if not same(out, expect):
                    bad.append(f"join:{mode_name(m)}")
                del g, out
            ok = not bad
            fails += 0 if ok else 1
            print(f"[{'ok ' if ok else 'BAD'}] graph shapes {name:24s} M={M:2d}: 3 GEMVs back to back, "
                  f"GEMV after a cross-stream join{'' if ok else ' DIFF: ' + ', '.join(bad)}")
    return fails


def check_vllm(kern: Kernels, dev) -> int:
    """The served call sites with this process' environment (VLLM_MOET_GEMV_PDL /
    VLLM_MOET_GEMV_EVICT_FIRST as set, default on): vLLM's FlashInferCutlassMxfp8LinearKernel
    (patched by patch_vllm_mxfp8_gemv.py) and the wo_a o-projection kernel (patch_vllm_wo_a_sm120.py),
    eagerly and from a CUDA graph, against the committed kernels on the same quantized operands."""
    from torch.nn.parameter import Parameter

    from vllm.model_executor.kernels.linear.mxfp8 import flashinfer as fi_mod
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferCutlassMxfp8LinearKernel
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig

    fails = 0
    if not hasattr(fi_mod, "_sm120_gemv_for"):
        print("[BAD] vLLM's MXFP8 linear kernel is not patched (patch_vllm_mxfp8_gemv.py)")
        return 1
    torch.manual_seed(31)
    for name, (N, K) in DENSE.items():
        w = (torch.randn(N, K, device=dev) * 0.05).bfloat16()
        wq, wsf = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
        layer = torch.nn.Module()
        layer.weight = Parameter(wq, requires_grad=False)
        layer.weight_scale = Parameter(wsf.view(N, -1), requires_grad=False)
        lk = FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearLayerConfig())
        lk.process_weights_after_loading(layer)
        bad = []
        for M in (1, 6, 8, 12, 16):
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            expect = kern.dense(None)(*quant_act(x), layer.weight, layer.weight_scale)
            if fi_mod._sm120_gemv_for(M, torch.bfloat16) is None:
                bad.append(f"M={M}:not-on-gemv")
            if not same(lk.apply_weights(layer, x), expect):
                bad.append(f"M={M}:eager")
            g, out = capture(lambda: lk.apply_weights(layer, x))
            g.replay()
            torch.cuda.synchronize()
            if not same(out, expect):
                bad.append(f"M={M}:graph")
            del g, out
        ok = not bad
        fails += 0 if ok else 1
        print(f"[{'ok ' if ok else 'BAD'}] vLLM FlashInferCutlassMxfp8LinearKernel {name:24s} M=1..16 eager + graph"
              f"{'' if ok else ' DIFF: ' + ', '.join(bad)}")
    from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel

    gk = init_mxfp8_linear_kernel(bmm_batch_size=N_GROUPS)
    if type(gk).__name__ != "Sm120GemvMxfp8BmmLinearKernel":
        print(f"[BAD] wo_a kernel not selected (patch_vllm_wo_a_sm120.py): {type(gk).__name__}")
        return fails + 1
    N = N_GROUPS * O_LORA
    wq, wsf = quant_weight_rowmajor((torch.randn(N, WO_A_K, device=dev) * 0.05).bfloat16())
    layer = torch.nn.Module()
    layer.weight = Parameter(wq, requires_grad=False)
    layer.weight_scale = Parameter(wsf, requires_grad=False)
    layer.bmm_batch_size = N_GROUPS
    gk.process_weights_after_loading(layer)
    cos_sin = rope_cache(dev)
    ident = torch.nn.Identity()
    bad = []
    for T in (1, 6, 16, 48, 64):
        o, pos = grouped_inputs(T, dev, cos_sin, "normal")
        expect = kern.grouped(None)(*inv_rope_quant(o, pos, cos_sin), layer.weight, layer.weight_scale).flatten(1)

        def run():
            return gk.o_proj(o, pos, cos_sin, layer, ident, n_groups=N_GROUPS, heads_per_group=HEADS_PER_GROUP,
                             nope_dim=NOPE, rope_dim=ROPE, o_lora_rank=O_LORA)

        if not same(run(), expect):
            bad.append(f"T={T}:eager")
        g, out = capture(run)
        g.replay()
        torch.cuda.synchronize()
        if not same(out, expect):
            bad.append(f"T={T}:graph")
        del g, out
    ok = not bad
    fails += 0 if ok else 1
    print(f"[{'ok ' if ok else 'BAD'}] vLLM Sm120GemvMxfp8BmmLinearKernel.o_proj (wo_a) T=1..64 eager + graph"
          f"{'' if ok else ' DIFF: ' + ', '.join(bad)}")
    return fails


def region(t: torch.Tensor) -> int:
    """Bytes spanned by a strided view from its first element."""
    return (1 + sum((d - 1) * s for d, s in zip(t.shape, t.stride()))) * t.element_size()


def strided_view(buf: torch.Tensor, off: int, like: torch.Tensor) -> torch.Tensor:
    raw = buf[off:off + region(like)].view(like.dtype)
    return torch.as_strided(raw, like.shape, like.stride())


def capture(fn):
    """Capture fn() into a CUDA graph on a side stream (no warm-up); returns (graph, fn's result)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            res = fn()
    torch.cuda.synchronize()
    return g, res


def check_race(kern: Kernels, modes, dev, spin_us: float) -> int:
    """Slow producer in front of a PDL GEMV, eagerly and replayed from a CUDA graph: the GEMV must
    see the producer's activations and must not write its output - allocated in the block of a
    buffer the producer still reads, as the caching allocator / graph pool hand it out - before the
    producer is done. A PDL consumer without griddepcontrol.wait must be caught (the first launch
    of a kernel does not overlap - module loading -, so every kernel is warmed up first)."""
    race = load_race()
    fails = 0
    pattern = 1234.0
    spin = int(spin_us * 1000)

    src = torch.randint(0, 255, (4096,), device=dev, dtype=torch.uint8)
    dst = torch.zeros_like(src)
    snap = torch.empty(4096, device=dev, dtype=torch.uint8)
    watch = torch.full((2048,), pattern, device=dev, dtype=torch.bfloat16)

    def control():
        race.slow_copy(src, dst, watch.view(torch.uint8), snap, spin)
        race.no_wait_copy(dst, watch.view(torch.uint8))

    def caught():
        torch.cuda.synchronize()
        return (not torch.equal(watch.view(torch.uint8), src), not bool((snap.view(torch.bfloat16) == pattern).all()))

    def reset():
        dst.zero_()
        watch.fill_(pattern)
        torch.cuda.synchronize()

    control()
    reset()
    control()
    eager_c = caught()
    g, _ = capture(control)
    reset()
    g.replay()
    graph_c = caught()
    del g
    ok = all(eager_c) and all(graph_c)
    fails += 0 if ok else 1
    print(f"[{'ok ' if ok else 'BAD'}] race detector on a PDL consumer without griddepcontrol.wait: "
          f"eager stale/early-write caught={eager_c}, graph caught={graph_c}")

    cases = [(n, NK, M) for n, NK in DENSE.items() for M in (6, 16)] + [(GROUPED, None, 6)]
    for name, NK, M in cases:
        torch.manual_seed(5)
        if NK is not None:
            N, K = NK
            w, ws = quant_weight((torch.randn(N, K, device=dev) * 0.05).bfloat16())
            a_new, s_new = quant_act(torch.randn(M, K, device=dev, dtype=torch.bfloat16))
            a_old, s_old = quant_act(torch.randn(M, K, device=dev, dtype=torch.bfloat16))
            out_shape = (M, N)
            run = kern.dense
        else:
            N = N_GROUPS * O_LORA
            w, ws = quant_weight_rowmajor((torch.randn(N, WO_A_K, device=dev) * 0.05).bfloat16())
            cos_sin = rope_cache(dev)
            a_new, s_new = inv_rope_quant(*grouped_inputs(M, dev, cos_sin, "normal"), cos_sin)
            a_old, s_old = inv_rope_quant(*grouped_inputs(M, dev, cos_sin, "normal"), cos_sin)
            out_shape = (M, N_GROUPS, O_LORA)
            run = kern.grouped
        # one byte buffer holds the activations and their scales, so one producer writes both
        off_s = (region(a_new) + 255) // 256 * 256
        src = torch.zeros(off_s + region(s_new), device=dev, dtype=torch.uint8)
        dst = torch.zeros_like(src)
        a_src, s_src = strided_view(src, 0, a_new), strided_view(src, off_s, s_new)
        a_dst, s_dst = strided_view(dst, 0, a_new), strided_view(dst, off_s, s_new)
        a_src.copy_(a_new)
        s_src.copy_(s_new)
        expect = run(None)(a_src, s_src, w, ws)
        numel = expect.numel()
        snap = torch.empty(numel * 2, device=dev, dtype=torch.uint8)
        torch.cuda.synchronize()
        bad = []

        def trial():
            # the output-sized block freed last is the one the GEMV's output gets (the caching
            # allocator's / the graph pool's best fit): watch takes the previous output's block,
            # the GEMV takes watch's
            watch = torch.full(out_shape, pattern, device=dev, dtype=torch.bfloat16)
            ptr = watch.data_ptr()
            race.slow_copy(src, dst, watch.view(torch.uint8).view(-1), snap, spin)
            del watch
            out = run(m)(a_dst, s_dst, w, ws)
            return out, out.data_ptr() == ptr

        def verdict(tag, out, aliased):
            torch.cuda.synchronize()
            if not same(out, expect):
                bad.append(f"{tag}{mode_name(m)}:stale-activations")
            if not bool((snap.view(torch.bfloat16) == pattern).all()):
                bad.append(f"{tag}{mode_name(m)}:early-write")
            if not aliased:
                bad.append(f"{tag}{mode_name(m)}:(output did not alias the watched buffer)")

        def stale_l1():
            # old activations in the producer's output buffer, read once by the same kernel so
            # that L1s hold their lines when the producer overwrites them (the first launch of a
            # kernel also does not overlap - module loading - so this is the warm-up too)
            a_dst.copy_(a_old)
            s_dst.copy_(s_old)
            run(m)(a_dst, s_dst, w, ws)
            torch.cuda.synchronize()

        for m in [None] + list(modes):
            stale_l1()
            out, aliased = trial()
            verdict("", out, aliased)
            del out
            g, (out, aliased) = capture(trial)
            stale_l1()
            g.replay()
            verdict("graph:", out, aliased)
            del g, out
        hard = [b for b in bad if "did not alias" not in b]
        ok = not hard
        fails += 0 if ok else 1
        print(f"[{'ok ' if ok else 'BAD'}] race {name:24s} M={M:2d} producer {spin_us:.0f} us, eager + graph: "
              f"{'clean' if not bad else ', '.join(bad)}")
    return fails


# ----------------------------------------------------------------------------------------------
# timings


def cold_copies(nbytes: int, cap: int = 128) -> int:
    return max(2, min(cap, COLD_BYTES // max(nbytes, 1) + 1))


def bench_dense(kern: Kernels, modes, ms, dev, iters, results):
    print("\nper shape, cold weights, us per call: (a) GEMVs back to back, (b) FlashInfer's MXFP8 quantize -> GEMV")
    hdr = "".join(f"{mode_name(m):>16s}" for m in [None] + modes)
    print(f"{'shape':24s} {'M':>3s} {'':3s}{hdr}")
    for name, (N, K) in DENSE.items():
        torch.manual_seed(1)
        w, ws = quant_weight((torch.randn(N, K, device=dev) * 0.05).bfloat16())
        n = cold_copies(N * K + ws.numel())
        copies = [(w.clone(), ws.clone()) for _ in range(n)]
        for M in ms:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            a, s = quant_act(x)
            row_a, row_b = [], []
            for m in [None] + modes:
                f = kern.dense(m)
                ta, g, _ = graph_time([(lambda c=c: f(a, s, c[0], c[1])) for c in copies], iters)
                del g
                tb, g, _ = graph_time([(lambda c=c: f(*quant_act(x), c[0], c[1])) for c in copies], iters)
                del g
                row_a.append(ta / n)
                row_b.append(tb / n)
                results.setdefault("dense", []).append(
                    {"shape": name, "N": N, "K": K, "M": M, "mode": mode_name(m), "b2b_us": ta / n, "quant_gemv_us": tb / n})
            print(f"{name:24s} {M:3d} (a){''.join(f'{t:16.2f}' for t in row_a)}")
            print(f"{'':24s} {'':3s} (b){''.join(f'{t:16.2f}' for t in row_b)}")
        del copies
        torch.cuda.empty_cache()


def bench_grouped(kern: Kernels, modes, ts, dev, iters, results):
    print(f"\n{GROUPED}, cold weights, us per call: (a) back to back, (b) fused_inv_rope_fp8_quant -> grouped GEMV")
    N = N_GROUPS * O_LORA
    torch.manual_seed(2)
    w, ws = quant_weight_rowmajor((torch.randn(N, WO_A_K, device=dev) * 0.05).bfloat16())
    n = cold_copies(N * WO_A_K)
    copies = [w.clone() for _ in range(n)]
    cos_sin = rope_cache(dev)
    for T in ts:
        o, pos = grouped_inputs(T, dev, cos_sin, "normal")
        a, s = inv_rope_quant(o, pos, cos_sin)
        row_a, row_b = [], []
        for m in [None] + modes:
            f = kern.grouped(m)
            ta, g, _ = graph_time([(lambda c=c: f(a, s, c, ws)) for c in copies], iters)
            del g
            tb, g, _ = graph_time([(lambda c=c: f(*inv_rope_quant(o, pos, cos_sin), c, ws)) for c in copies], iters)
            del g
            row_a.append(ta / n)
            row_b.append(tb / n)
            results.setdefault("grouped", []).append({"T": T, "mode": mode_name(m), "b2b_us": ta / n, "rope_gemv_us": tb / n})
        print(f"{'T=' + str(T):24s}     (a){''.join(f'{t:16.2f}' for t in row_a)}")
        print(f"{'':24s}     (b){''.join(f'{t:16.2f}' for t in row_b)}")


class Chains:
    """Chains of the served decode graph (rank 0, per layer) with the real neighbour kernels:
    attention: quantize -> q_a+kv_a GEMV -> fused q/kv RMSNorm + quantize (Triton) -> q_b+idx GEMV
               -> a plain elementwise kernel (stands in for the non-PDL q-norm/RoPE, top-k and
               sparse-MLA kernels between q_b and the o-projection in the served graph)
               -> inverse RoPE + quantize (Triton) -> grouped wo_a GEMV -> quantize
               -> wo_b GEMV -> residual add
    shared expert: quantize -> w13 GEMV -> silu_and_mul -> quantize -> w2 GEMV (v1) -> add
    engram: quantize -> 6400<-6144 GEMV -> copy;   indexer: quantize -> wq_b 4096<-1280 GEMV -> copy
    Each layer has its own weights, so the weights stream from DRAM as in the server."""

    def __init__(self, dev, M, layers):
        import vllm._custom_ops  # noqa: F401  (registers torch.ops._C.silu_and_mul)
        from vllm.models.deepseek_v41.common.ops.query_quant import fused_q_kv_rmsnorm_quant

        self.norm_quant = fused_q_kv_rmsnorm_quant
        self.dev, self.M = dev, M
        torch.manual_seed(9)
        mk = lambda N, K: quant_weight((torch.randn(N, K, device=dev) * 0.05).bfloat16())  # noqa: E731
        self.att = [dict(qa=mk(1792, 5120), qb=mk(8192, 1280),
                         woa=quant_weight_rowmajor((torch.randn(N_GROUPS * O_LORA, WO_A_K, device=dev) * 0.05).bfloat16()),
                         wob=mk(5120, 2048)) for _ in range(layers["attention"])]
        self.ffn = [dict(w13=mk(1152, 5120), w2=mk(5120, 576)) for _ in range(layers["shared"])]
        self.eng = [mk(6400, 6144) for _ in range(layers["engram"])]
        self.idx = [mk(4096, 1280) for _ in range(layers["indexer"])]
        self.h = torch.randn(M, 5120, device=dev, dtype=torch.bfloat16)
        self.e = torch.randn(M, 6144, device=dev, dtype=torch.bfloat16)
        self.xi = torch.randn(M, 1280, device=dev, dtype=torch.bfloat16)
        self.qw = (1 + 0.1 * torch.randn(1280, device=dev)).bfloat16()
        self.kvw = (1 + 0.1 * torch.randn(512, device=dev)).bfloat16()
        self.cos_sin = rope_cache(dev)
        self.pos = torch.randint(0, MAX_POS, (M,), device=dev, dtype=torch.int64)

    def attention(self, kern: Kernels, mode, L):
        dense, grouped = kern.dense(mode), kern.grouped(mode)
        h = self.h

        def layer():
            y = dense(*quant_act(h), *L["qa"])
            qa, kv = self.norm_quant(y[:, :1280], y[:, 1280:], self.qw, self.kvw, 1e-6)
            q = dense(qa.data.view(-1, 1280), qa.scale, *L["qb"])
            o = (q * 0.5).view(self.M, N_GROUPS * HEADS_PER_GROUP, HEAD_DIM)
            z = grouped(*inv_rope_quant(o, self.pos, self.cos_sin), *L["woa"])
            out = dense(*quant_act(z.view(self.M, -1)), *L["wob"])
            return out + h
        return layer

    def shared(self, kern: Kernels, mode, L):
        dense = kern.dense(mode)
        h = self.h

        def layer():
            y = dense(*quant_act(h), *L["w13"])
            act = torch.empty(self.M, 576, device=self.dev, dtype=torch.bfloat16)
            torch.ops._C.silu_and_mul(act, y)
            return dense(*quant_act(act), *L["w2"]) + h
        return layer

    def single(self, kern: Kernels, mode, W, x):
        dense = kern.dense(mode)
        return lambda: dense(*quant_act(x), *W).clone()

    def calls(self, kern, mode, which):
        if which == "attention":
            return [self.attention(kern, mode, L) for L in self.att]
        if which == "shared":
            return [self.shared(kern, mode, L) for L in self.ffn]
        if which == "engram":
            return [self.single(kern, mode, W, self.e) for W in self.eng]
        return [self.single(kern, mode, W, self.xi) for W in self.idx]


def bench_chains(kern: Kernels, modes, ms, dev, iters, results, layers, placements: int) -> int:
    """Absolute chain times move by up to ~7 % with where the weights land in memory (a second
    set of the same tensors at other addresses), so every placement is timed in all modes and
    the table shows medians over placements; the deltas to ref are taken per placement."""
    fails = 0
    print(f"\nchains in one CUDA graph, cold weights, us per layer (each layer its own weights), median of "
          f"{placements} weight placements [delta to ref per placement, min..max]; bits vs ref")
    hdr = "".join(f"{mode_name(m):>22s}" for m in [None] + modes)
    print(f"{'chain':12s} {'M':>3s} {hdr}")
    gen = torch.Generator().manual_seed(17)
    for M in ms:
        per = {}  # (which, mode) -> [us per layer per placement]
        for pl in range(placements):
            pad = torch.empty(int(torch.randint(1, 96, (1,), generator=gen)) << 20, device=dev, dtype=torch.uint8)
            ch = Chains(dev, M, layers)
            for which in ("attention", "shared", "engram", "indexer"):
                ref_outs = None
                nl = len(ch.calls(kern, None, which))
                for m in [None] + modes:
                    t, g, outs = graph_time(ch.calls(kern, m, which), iters)
                    torch.cuda.synchronize()
                    if m is None:
                        ref_outs = [o.clone() for o in outs]
                    elif not all(same(o, r) for o, r in zip(outs, ref_outs)):
                        fails += 1
                        print(f"  BAD: chain {which} M={M} {mode_name(m)} differs from ref")
                    del g, outs
                    per.setdefault((which, mode_name(m)), []).append(t / nl)
                    results.setdefault("chains", []).append(
                        {"chain": which, "M": M, "mode": mode_name(m), "placement": pl, "us_per_layer": t / nl})
            del ch, pad
            torch.cuda.empty_cache()
        for which in ("attention", "shared", "engram", "indexer"):
            ref = per[(which, "ref")]
            cells = [f"{statistics.median(ref):22.2f}"]
            for m in modes:
                v = per[(which, mode_name(m))]
                d = [a - b for a, b in zip(v, ref)]
                cells.append(f"{statistics.median(v):8.2f} [{min(d):+5.2f}..{max(d):+5.2f}]")
            print(f"{which:12s} {M:3d} {''.join(cells)}")
    return fails


def trace_chains(kern: Kernels, modes, M, dev, layers, out_dir) -> None:
    """Kernel timelines of the chain graphs (torch profiler, 5 replays): per edge between
    consecutive kernels of a layer the median gap from the predecessor's end to the successor's
    start (negative: the successor came up while the predecessor ran), and the time per layer in
    which no kernel ran (the decode trace's GPU-idle metric)."""
    import re
    from collections import defaultdict

    from torch.profiler import ProfilerActivity, profile

    def short(n):
        n = re.sub(r"\(.*", "", n).replace("void ", "").replace("vllm_moet_sm120::", "")
        n = re.sub(r"^kernel_cutlass_kernel_flashinferquantization.*mxfp8_quantize.*", "mxfp8_quantize(FlashInfer)", n)
        n = re.sub(r"^at::native::\S*elementwise_kernel.*", "torch elementwise", n)
        return n[:40]

    ch = Chains(dev, M, layers)
    print(f"\nkernel timeline of the chain graphs, M={M}: median gap predecessor end -> successor start (us), "
          "GPU idle per layer")
    for which in ("attention", "shared"):
        for m in [None] + modes:
            calls = ch.calls(kern, m, which)
            _, g, _ = graph_time(calls, 3)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    g.replay()
                torch.cuda.synchronize()
            path = Path(out_dir) / f"trace_{which}_M{M}_{mode_name(m).replace('/', '_')}.json"
            prof.export_chrome_trace(str(path))
            ev = json.load(open(path))["traceEvents"]
            kerns = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
            per = len(kerns) // 5 // len(calls)  # kernels per layer
            gaps, idle, durs = defaultdict(list), [], defaultdict(list)
            for r in range(5):
                rep = kerns[r * per * len(calls):(r + 1) * per * len(calls)]
                for li in range(1, len(calls) - 1):  # skip the first / last layer of a replay
                    lay = rep[li * per:(li + 1) * per + 1]  # + the next layer's first kernel
                    end, busy_idle = lay[0]["ts"] + lay[0]["dur"], 0.0
                    for a, b in zip(lay, lay[1:]):
                        gaps[(short(a["name"]), short(b["name"]))].append(b["ts"] - (a["ts"] + a["dur"]))
                        if b["ts"] > end:
                            busy_idle += b["ts"] - end
                        end = max(end, b["ts"] + b["dur"])
                    idle.append(busy_idle)
                    for i, k in enumerate(lay[:-1]):
                        durs[(i, short(k["name"]))].append(k["dur"])
            print(f"  {which} {mode_name(m)}: GPU idle {statistics.median(idle):5.2f} us per layer; kernel durations "
                  f"(us, under PDL they include the time a kernel waits for its predecessor): "
                  + ", ".join(f"{n.split('<')[0]} {statistics.median(v):.2f}" for (_, n), v in sorted(durs.items())))
            for (a, b), v in gaps.items():
                print(f"      {statistics.median(v):+6.2f}  {a:40s} -> {b}")
            del g
            path.unlink()
    del ch
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-cu", default=os.environ.get("VLLM_MOET_GEMV_REF_CU", ""),
                    help="kernel source of the reference (e.g. the .cu served before the launch modes); "
                         "default: this extension's plain launch (pdl=0, evict=0)")
    ap.add_argument("--modes", default="0:0,1:0,2:0,3:0,1:1,3:1", help="pdl:evict list compared against the ref kernels")
    ap.add_argument("--ms", default="1,6,8,12,16")
    ap.add_argument("--ts", default="1,6,16,48,64", help="token counts of the grouped wo_a GEMV")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--no-vllm", action="store_true", help="skip the checks through vLLM's patched call sites")
    ap.add_argument("--bench", action="store_true", help="per-shape cold-weight timings")
    ap.add_argument("--bench-ms", default="1,6,8,12,16")
    ap.add_argument("--bench-modes", default=None, help="modes timed (default: --modes)")
    ap.add_argument("--chain", action="store_true", help="chain timings (+ their bits)")
    ap.add_argument("--chain-ms", default="6")
    ap.add_argument("--layers", default="attention=8,shared=32,engram=8,indexer=48")
    ap.add_argument("--placements", type=int, default=3, help="weight placements per chain measurement")
    ap.add_argument("--trace", type=int, default=0, metavar="M", help="kernel timeline of the chains at M rows")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--race-us", type=float, default=30.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--mem-gb", type=float, default=float(os.environ.get("VLLM_MOET_TEST_MEM_GB", "0")),
                    help="cap the caching allocator (GB) on a shared GPU; 0 = no cap")
    args = ap.parse_args()
    dev = torch.device("cuda")
    if args.mem_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb * (1 << 30) / total))
    ref_desc = args.ref_cu if args.ref_cu else "this extension, pdl=0 evict=0"
    print(f"device={torch.cuda.get_device_name(0)} sms={torch.cuda.get_device_properties(0).multi_processor_count} "
          f"VLLM_MOET_GEMV_IMPL={os.environ.get('VLLM_MOET_GEMV_IMPL', '(v3)')} ref={ref_desc}")
    t0 = time.time()
    kern = Kernels(load_ref(args.ref_cu) if args.ref_cu else None)
    print(f"extensions ready in {time.time() - t0:.0f}s")
    modes = parse_modes(args.modes)
    bmodes = parse_modes(args.bench_modes) if args.bench_modes else modes
    results = {"device": torch.cuda.get_device_name(0), "modes": [mode_name(m) for m in modes]}
    fails = 0
    if not args.no_check:
        ms = [int(m) for m in args.ms.split(",")]
        ts = [int(t) for t in args.ts.split(",")]
        fails += check_dense(kern, modes, ms, dev)
        fails += check_grouped(kern, modes, ts, dev)
        fails += check_race(kern, [m for m in modes if m[0] != 0], dev, args.race_us)
        fails += check_graph_shapes(kern, modes, dev)
        if not args.no_vllm:
            fails += check_vllm(kern, dev)
    if args.bench:
        bms = [int(m) for m in args.bench_ms.split(",")]
        bench_dense(kern, bmodes, bms, dev, args.iters, results)
        bench_grouped(kern, bmodes, [t for t in (1, 6, 16, 48) if t <= 64], dev, args.iters, results)
    if args.chain:
        layers = {k: int(v) for k, v in (kv.split("=") for kv in args.layers.split(","))}
        fails += bench_chains(kern, bmodes, [int(m) for m in args.chain_ms.split(",")], dev, args.iters, results, layers,
                              args.placements)
    if args.trace:
        layers = {k: int(v) for k, v in (kv.split("=") for kv in args.layers.split(","))}
        trace_chains(kern, bmodes, args.trace, dev, layers, tempfile.gettempdir())
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    print(f"\n{'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
