#!/usr/bin/env python3
"""bf12 (VLLM_MOE_W2_BF12) unit tests: lossless 12-bit BF16 container.

1. encode/decode roundtrip is bit-exact on adversarial BF16 payloads
   (normals across the exponent range, denormals, +-0, inf/nan bit
   patterns, single-exponent tensors, >7 exponent classes -> escapes).
2. Bf12LinearMethod.apply == F.linear on the original weight, bitwise
   (same GEMM over identical bytes), CUDA graph capture included.
3. convert_model walk: converts eligible LinearBase layers in-place,
   frees the BF16 param, output stays bitwise identical.

Run inside the serving container (needs torch + the patched vllm tree):
  python3 tools/test_bf12.py
"""
import sys

import torch

sys.path.insert(0, ".")

from vllm.model_executor.layers.quantization.utils import (  # noqa: E402
    moe_w2_bf12 as bf12,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAILS = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILS.append(name)


def roundtrip(w: torch.Tensor) -> bool:
    nib, lo, esc_bp, esc_hi, lut16 = bf12.encode_bf12(w)
    probe = type("P", (), {})()
    probe.bf12_nib, probe.bf12_lo = nib, lo
    probe.bf12_esc_bytepos, probe.bf12_esc_hi = esc_bp, esc_hi
    probe.bf12_lut16 = lut16
    out = torch.empty(w.numel() * 2, dtype=torch.uint8, device=w.device)
    bf12.decode_bf12(probe, out)
    # byte-view compare: bitwise identity, NaN-safe (bf16 NaN != NaN)
    return torch.equal(out, w.reshape(-1).view(torch.uint8))


# ---- 1. roundtrip ----------------------------------------------------------
print(f"== roundtrip (dev={DEV})")
g = torch.Generator(device="cpu").manual_seed(0)

w = (torch.randn(512, 1024, generator=g) * 0.02).bfloat16().to(DEV)
check("gaussian 0.02 (trained-weight-like)", roundtrip(w))

w = (torch.randn(512, 1024, generator=g)
     * torch.logspace(-20, 20, 1024).unsqueeze(0)).bfloat16().to(DEV)
check("41 decades of magnitude (escape-heavy)", roundtrip(w))

w = torch.full((256, 512), 0.0, dtype=torch.bfloat16, device=DEV)
w[::2] = -0.0
check("mixed +0/-0", roundtrip(w))

raw = torch.randint(0, 1 << 16, (512, 512), generator=g, dtype=torch.int32)
w = (raw.to(torch.uint16)).view(torch.bfloat16).to(DEV)
check("uniform random bit patterns (incl. inf/nan/denormal)", roundtrip(w))

w = torch.full((64, 1024), 1.5, dtype=torch.bfloat16, device=DEV)
check("single (sign,exp) class", roundtrip(w))

w = (torch.randn(2048, 4096, generator=g) * 0.01).bfloat16().to(DEV)
nib, lo, esc_bp, esc_hi, lut16 = bf12.encode_bf12(w)
packed = nib.numel() + lo.numel() + esc_hi.numel() + esc_bp.numel() * 8 + 16
bpw = 8.0 * packed / w.numel()
check(f"size: {bpw:.3f} bpw (expect ~12.0-12.1)", 12.0 <= bpw < 12.2)

# ---- 2. forward equivalence (manual layer) --------------------------------
print("== Bf12LinearMethod forward equivalence")
n, k = 1536, 4096
w0 = (torch.randn(n, k, generator=g) * 0.02).bfloat16().to(DEV)
layer = torch.nn.Module()
layer.weight = torch.nn.Parameter(w0.clone(), requires_grad=False)

nib, lo, esc_bp, esc_hi, lut16 = bf12.encode_bf12(layer.weight.data)
del layer._parameters["weight"]
layer.register_buffer("bf12_nib", nib)
layer.register_buffer("bf12_lo", lo)
layer.register_buffer("bf12_esc_bytepos", esc_bp)
layer.register_buffer("bf12_esc_hi", esc_hi)
layer.register_buffer("bf12_lut16", lut16)
meth = bf12.Bf12LinearMethod(n, k)

scratch_dev = torch.empty(0, device=DEV).device  # cuda -> cuda:0
for _skey in ((scratch_dev.index, "main"), (scratch_dev.index, "aux")):
    bf12._SCRATCH[_skey] = torch.empty(
        n * k * 2, dtype=torch.uint8, device=DEV)

for m in (1, 4, 16):
    x = (torch.randn(m, k, generator=g) * 0.5).bfloat16().to(DEV)
    ref = torch.nn.functional.linear(x, w0)
    out = meth.apply(layer, x)
    check(f"M={m} bitwise == F.linear on original", torch.equal(out, ref))

bias = (torch.randn(n, generator=g) * 0.1).bfloat16().to(DEV)
x = (torch.randn(2, k, generator=g)).bfloat16().to(DEV)
check("bias path bitwise",
      torch.equal(meth.apply(layer, x, bias),
                  torch.nn.functional.linear(x, w0, bias)))

if DEV == "cuda":
    x = (torch.randn(4, k, generator=g)).bfloat16().to(DEV)
    sx = x.clone()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            meth.apply(layer, sx)  # warmup on side stream
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout = meth.apply(layer, sx)
    sx.copy_(x)
    graph.replay()
    check("CUDA graph capture + replay bitwise",
          torch.equal(gout, torch.nn.functional.linear(x, w0)))
    graph.replay()
    check("second replay deterministic",
          torch.equal(gout, torch.nn.functional.linear(x, w0)))

    # -- aux-stream overlap: replicate the fused-MoE shared-experts
    # pattern (SharedExperts._run_in_aux_stream): the aux stream runs a
    # bf12 linear while the main stream runs a DIFFERENT bf12 linear
    # concurrently. Requires the per-stream scratch buffers.
    from vllm.utils.torch_utils import aux_stream
    aux = aux_stream()
    w1 = (torch.randn(n, k, generator=g) * 0.02).bfloat16().to(DEV)
    layer2 = torch.nn.Module()
    nib2, lo2, ebp2, ehi2, lut2 = bf12.encode_bf12(w1)
    layer2.register_buffer("bf12_nib", nib2)
    layer2.register_buffer("bf12_lo", lo2)
    layer2.register_buffer("bf12_esc_bytepos", ebp2)
    layer2.register_buffer("bf12_esc_hi", ehi2)
    layer2.register_buffer("bf12_lut16", lut2)
    layer2.bf12_layer_id = -1
    ok_overlap = True
    for it in range(20):
        xa = (torch.randn(4, k, generator=g)).bfloat16().to(DEV)
        xb = (torch.randn(4, k, generator=g)).bfloat16().to(DEV)
        aux.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(aux):
            ya = meth.apply(layer, xa)      # aux stream: "shared experts"
        yb = meth.apply(layer2, xb)         # main stream: "routed" work
        yb2 = meth.apply(layer2, yb[:, :k] if yb.shape[1] >= k else xb)
        torch.cuda.current_stream().wait_stream(aux)
        if not torch.equal(ya, torch.nn.functional.linear(xa, w0)):
            ok_overlap = False
        if not torch.equal(yb, torch.nn.functional.linear(xb, w1)):
            ok_overlap = False
    check("aux-stream overlap (SharedExperts pattern) bitwise", ok_overlap)

# ---- 3. convert_model walk -------------------------------------------------
print("== convert_model walk")
try:
    from vllm.config import VllmConfig, set_current_vllm_config
    _cfg_ctx = set_current_vllm_config(VllmConfig())
    _cfg_ctx.__enter__()
    import vllm.distributed.parallel_state as ps
    if ps._TP is None:
        ps.init_distributed_environment(world_size=1, rank=0,
                                        distributed_init_method="tcp://127.0.0.1:29587",
                                        local_rank=0, backend="gloo")
        ps.initialize_model_parallel(1, 1)
    from vllm.model_executor.layers.linear import ReplicatedLinear

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.big = ReplicatedLinear(2048, 1024, bias=False,
                                        params_dtype=torch.bfloat16)
            self.small = ReplicatedLinear(64, 64, bias=False,
                                          params_dtype=torch.bfloat16)

    toy = Toy().to(DEV)
    with torch.no_grad():
        toy.big.weight.normal_(0, 0.02)
        toy.small.weight.normal_(0, 0.02)
    w_big = toy.big.weight.data.clone()
    x = (torch.randn(3, 2048, generator=g)).bfloat16().to(DEV)
    ref, _ = toy.big(x)

    import os
    os.environ["VLLM_MOE_W2_BF12"] = "1"
    bf12.convert_model(toy)
    conv = not hasattr(toy.big, "weight") and hasattr(toy.big, "bf12_nib")
    check("big layer converted, param freed", conv)
    check("small layer skipped (< MIN_NUMEL)", hasattr(toy.small, "weight"))
    out, _ = toy.big(x)
    check("post-convert forward bitwise identical", torch.equal(out, ref))
    dec = torch.empty(w_big.numel() * 2, dtype=torch.uint8, device=DEV)
    bf12.decode_bf12(toy.big, dec)
    check("planes decode == original weight",
          torch.equal(dec.view(torch.bfloat16),
                      w_big.reshape(-1)))
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    check(f"convert_model walk (exception: {e})", False)

print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
