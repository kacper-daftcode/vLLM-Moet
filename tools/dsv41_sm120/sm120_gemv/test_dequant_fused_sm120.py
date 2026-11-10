#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The fused wo_a dequant (vllm_sm120_gemv_bmm.dequant_mxfp8_to_bf16_fused, its code-0 guard and the guarded entry) against vLLM's
dequant_mxfp8_to_bf16, bit for bit:
  (1) the raw fused kernel, exhaustive over the 256 x 256 (e4m3 value byte, ue8m0 scale byte) codes (a [256 rows = scale code][256 x 32 cols]
      tensor: row r has scale r in every block, block b holds the 32 copies of value byte b), compared on the int16 view - expected: identical
      everywhere except scale code 0 (the 2^-127 descale is an fp32 subnormal: carried by the six kernels, flushed by the Triton kernel);
  (2) the guard fused_dequant_safe: False for a tensor with a code-0 block next to a non-zero value, True for code 0 over zeros and for
      codes >= 1 (CPU tensors are enough for this part: --cpu-guard-only);
  (3) the guarded entry dequant_mxfp8_to_bf16_guarded: identical to the six kernels on the exhaustive grid (it routes a code-0 tensor to the
      six kernels) and on random weights of the wo_a shape ([2048, 4096] e4m3 with [2048, 128] scales) and odd shapes;
  (4) the runtime A/B key WOA_DEQUANT_FUSED of the control file (VLLM_MOET_RUNTIME_AB=1): the effective switch follows the file, a dropped key
      returns to the env value, a malformed file keeps the state;
  (5) timing of both paths on the wo_a shape.
Runs inside the serving image (vllm + triton + one GPU; --cpu-guard-only needs no GPU).

  python3 test_dequant_fused_sm120.py [--iters 20] [--cpu-guard-only]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time


def guard_tests(torch, mod, dev) -> bool:
    ok = True
    K = 1024
    x = torch.zeros(4, K, dtype=torch.uint8, device=dev)
    s = torch.full((4, K // 32), 120, dtype=torch.uint8, device=dev)
    x[1, 40] = 0x05  # a non-zero value byte in block 1 of row 1
    s[1, 1] = 0  # ... whose block has scale code 0 -> not safe
    r1 = mod.fused_dequant_safe(x.view(torch.float8_e4m3fn), s)
    s2 = s.clone()
    s2[1, 1] = 120
    s2[2, 5] = 0  # code 0 over a block of zeros (row 2 is all zeros) -> safe
    r2 = mod.fused_dequant_safe(x.view(torch.float8_e4m3fn), s2)
    x3 = x.clone()
    x3[3, 32 * 7 + 3] = 0x80  # a signed zero (byte 0x80) under code 0 -> still safe
    s3 = s2.clone()
    s3[3, 7] = 0
    r3 = mod.fused_dequant_safe(x3.view(torch.float8_e4m3fn), s3)
    r4 = mod.fused_dequant_safe(x.view(torch.float8_e4m3fn), torch.full_like(s, 114))  # the served codes (114-118): safe
    print(f"guard: code 0 with a non-zero value -> {'not safe' if not r1 else 'SAFE (wrong)'}; code 0 over zeros -> {'safe' if r2 else 'NOT SAFE (wrong)'}; "
          f"code 0 over a signed zero -> {'safe' if r3 else 'NOT SAFE (wrong)'}; codes >= 1 -> {'safe' if r4 else 'NOT SAFE (wrong)'}")
    ok &= (not r1) and r2 and r3 and r4
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--cpu-guard-only", action="store_true")
    a = ap.parse_args()
    import torch
    import vllm.model_executor.kernels.linear.mxfp8.sm120_gemv_bmm as mod
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import dequant_mxfp8_to_bf16

    if a.cpu_guard_only:
        ok = guard_tests(torch, mod, torch.device("cpu"))
        print(f"{'PASS' if ok else 'FAIL'}: the code-0 guard (CPU)")
        return 0 if ok else 1
    dev = torch.device("cuda")
    ok = True

    def cmp(tag, fn, x, s, expect_identical=True):
        nonlocal ok
        ref = dequant_mxfp8_to_bf16(x, s)
        got = fn(x, s)
        torch.cuda.synchronize()
        ri, gi = ref.view(torch.int16), got.view(torch.int16)
        same = ri == gi
        n_bad = int((~same).sum())
        rows = torch.nonzero((~same).any(dim=1)).flatten().tolist() if n_bad else []
        verdict = "IDENTICAL" if n_bad == 0 else f"DIFFER on {n_bad} elements (scale codes {rows[:6]}{' ...' if len(rows) > 6 else ''})"
        print(f"{tag}: {ref.numel()} elements, {verdict}")
        if expect_identical:
            ok &= n_bad == 0
        return n_bad, rows

    # (1) the raw fused kernel on the exhaustive grid: differences allowed at scale code 0 only
    vals = torch.arange(256, dtype=torch.uint8, device=dev).repeat_interleave(32)           # [8192]: block b = 32 x byte b
    x = vals.unsqueeze(0).expand(256, -1).contiguous().view(torch.float8_e4m3fn)          # [256, 8192]
    s = torch.arange(256, dtype=torch.uint8, device=dev).unsqueeze(1).expand(256, 256).contiguous()  # [256, 256]: row r = scale r
    n_bad, rows = cmp("(1) raw fused kernel, exhaustive 256 scale codes x 256 value bytes", mod.dequant_mxfp8_to_bf16_fused, x, s, expect_identical=False)
    raw_ok = rows == [0] or n_bad == 0
    print(f"    -> {'as expected: the only differing scale code is 0 (the subnormal descale)' if raw_ok else 'UNEXPECTED: differences outside scale code 0'}")
    ok &= raw_ok
    # (2) the guard
    ok &= guard_tests(torch, mod, dev)
    # (3) the guarded entry: identical everywhere (the exhaustive tensor carries code 0 with non-zero values -> routed to the six kernels)
    cmp("(3) guarded entry on the exhaustive grid", lambda x_, s_: mod.dequant_mxfp8_to_bf16_guarded(x_, s_), x, s)
    g = torch.Generator(device=dev).manual_seed(12)
    for shape in ((2048, 4096), (1024, 4096), (7, 2048), (3, 1024), (2, 512)):
        xr = torch.randint(0, 256, shape, device=dev, dtype=torch.uint8, generator=g).view(torch.float8_e4m3fn)
        sr = torch.randint(100, 141, (shape[0], shape[1] // 32), device=dev, dtype=torch.uint8, generator=g)
        cmp(f"(3) guarded entry, random {shape}", lambda x_, s_: mod.dequant_mxfp8_to_bf16_guarded(x_, s_), xr, sr)
        cmp(f"(3) raw fused kernel, random {shape} (codes 100-140)", mod.dequant_mxfp8_to_bf16_fused, xr, sr)
    x3 = torch.randint(0, 256, (2, 1024, 4096), device=dev, dtype=torch.uint8, generator=g).view(torch.float8_e4m3fn)
    s3 = torch.randint(100, 141, (2, 1024, 128), device=dev, dtype=torch.uint8, generator=g)
    cmp("(3) guarded entry, random [2, 1024, 4096]", lambda x_, s_: mod.dequant_mxfp8_to_bf16_guarded(x_, s_), x3, s3)
    # (4) the runtime A/B key (this process: RUNTIME_AB forced on, a private control file)
    with tempfile.TemporaryDirectory() as td:
        ctl = os.path.join(td, "ab.json")
        os.environ["VLLM_MOET_RUNTIME_AB"] = "1"
        os.environ["VLLM_MOET_RUNTIME_AB_FILE"] = ctl
        mod._WOA_RT["on"] = None
        mod._WOA_RT["next_poll"] = 0.0
        env_val = mod._WOA_DEQUANT_FUSED
        states = []
        for content in ('{"WOA_DEQUANT_FUSED": 1}', '{"WOA_DEQUANT_FUSED": 0}', '{ not json', '{}', '{"WOA_DEQUANT_FUSED": 7}'):
            with open(ctl, "w", encoding="utf-8") as f:
                f.write(content)
            time.sleep(0.12)
            mod._WOA_RT["next_poll"] = 0.0
            mod._woa_runtime_hooks()
            states.append(mod.woa_dequant_fused_effective())
        expect = [True, False, False, env_val, env_val]
        print(f"(4) runtime A/B key: effective after 1 / 0 / malformed / dropped / out-of-range = {[int(v) for v in states]} "
              f"(expected {[int(v) for v in expect]}, env {int(env_val)}) -> {'OK' if states == expect else 'MISMATCH'}")
        ok &= states == expect
        del os.environ["VLLM_MOET_RUNTIME_AB"]
        del os.environ["VLLM_MOET_RUNTIME_AB_FILE"]
    # (5) timing on the wo_a shape
    xw = torch.randint(0, 256, (2048, 4096), device=dev, dtype=torch.uint8, generator=g).view(torch.float8_e4m3fn)
    sw = torch.randint(100, 141, (2048, 128), device=dev, dtype=torch.uint8, generator=g)
    for name, fn in (("dequant_mxfp8_to_bf16 (6 kernels)", dequant_mxfp8_to_bf16), ("fused (1 Triton kernel)", mod.dequant_mxfp8_to_bf16_fused),
                     ("guarded entry (guard + fused)", lambda x_, s_: mod.dequant_mxfp8_to_bf16_guarded(x_, s_))):
        for _ in range(3):
            fn(xw, sw)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            fn(xw, sw)
        torch.cuda.synchronize()
        print(f"(5) time {name}: {(time.perf_counter() - t0) / a.iters * 1e6:.1f} us per call (host-timed, {a.iters} iterations, [2048, 4096])")
    print(f"{'PASS' if ok else 'FAIL'}: fused wo_a dequant with the code-0 guard {'bit-identical' if ok else 'DIFFERS'} with dequant_mxfp8_to_bf16")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
