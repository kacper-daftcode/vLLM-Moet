#!/usr/bin/env python3
"""Derive a Ripple-65B Preview checkpoint with the attention projections in MXFP8 from the ModelOpt NVFP4 checkpoint.

The NVFP4 export keeps q/k/v/g/o_proj in BF16 (82 "ignore" entries: lm_head, embed_tokens, every self_attn*). On the
4x RTX PRO 6000 decode step those BF16 projections are 6.6 GB of the 16.1 GB a TP rank reads per step (41 %); in MXFP8
(fp8-e4m3 values, one e8m0 scale per 32 elements along K, the OCP MX block format vLLM's ModelOpt loader reads as
quant_algo "MXFP8") the same layers are 3.4 GB. Measured on 4x RTX PRO 6000 (FlashInfer-CUTLASS MXFP8 with the dynamic activation quant included, per rank, M = 16 rows): qkg 31.8 -> 19.3 us,
o_proj 17.4 -> 10.3 us; the 256-wide v_proj is faster in BF16 (4.4 vs 12.0 us), so it stays BF16 by default.

Output: the same shard layout (one safetensors per input shard, same names), every selected
`model.layers.N.self_attn.<proj>.weight` replaced by `<proj>.weight` (F8_E4M3 [N, K]) + `<proj>.weight_scale` (U8 e8m0
bits [N, K/32]), every other tensor copied bit for bit, config.json with quantization_config.quant_algo =
"MIXED_PRECISION" and the per-layer `quantized_layers` map (MLP linears NVFP4 group 16 as before, the converted
projections MXFP8), the other files copied. The quantization follows vLLM's `_mxfp8_e4m3_quantize_torch` exactly
(scale = 2^ceil(log2(amax / 448)), values = round-to-nearest-even e4m3 of x / scale), so the loader's dequantization
reproduces the intended values.

The checkpoint's outputs differ from the NVFP4 checkpoint's (a numerics change): gate it (GSM8K, needle, tool calls,
parity against the NVFP4 arm) before serving - docs/ripple-sm120.md.

usage: make_mxfp8_attn_checkpoint.py SRC_DIR DST_DIR [--projs q_proj,k_proj,g_proj,o_proj] [--threads 32]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MXFP8_BLOCK = 32
E4M3_MAX = 448.0
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|g_proj|o_proj)\.weight$")


def mxfp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's MXFP8 convention (vllm/model_executor/layers/quantization/utils/mxfp8_utils.py): per 32-block e8m0 scale
    that fits the block amax into e4m3's finite range; returns (fp8 values [N, K], uint8 scale bits [N, K/32])."""
    n, k = x.shape
    assert k % MXFP8_BLOCK == 0, (n, k)
    xf = x.to(torch.float32).view(n, k // MXFP8_BLOCK, MXFP8_BLOCK)
    amax = xf.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    scale_biased = (torch.ceil(torch.log2(amax / E4M3_MAX)) + 127.0).clamp(0, 254)
    descale = torch.exp2(scale_biased - 127.0)
    q = (xf / descale.unsqueeze(-1)).view(n, k).to(torch.float8_e4m3fn)
    return q, scale_biased.to(torch.uint8).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--projs", default="q_proj,k_proj,g_proj,o_proj")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    projs = set(a.projs.split(","))
    src, dst = a.src.rstrip("/"), a.dst.rstrip("/")
    os.makedirs(dst, exist_ok=True)
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    new_map: dict[str, str] = {}
    total_bytes = 0
    converted: list[str] = []
    t_start = time.time()
    for shard in shards:
        t0 = time.time()
        out: dict[str, torch.Tensor] = {}
        n_conv = 0
        with safe_open(os.path.join(src, shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                m = LAYER_RE.match(name)
                if m and m.group(2) in projs:
                    assert t.dtype == torch.bfloat16, (name, t.dtype)
                    q, s = mxfp8_quantize(t)
                    base = name[: -len(".weight")]
                    out[name] = q
                    out[base + ".weight_scale"] = s
                    converted.append(base)
                    n_conv += 1
                else:
                    out[name] = t
        for name, t in out.items():
            new_map[name] = shard
            total_bytes += t.numel() * t.element_size()
        save_file(out, os.path.join(dst, shard), metadata={"format": "pt"})
        print(f"{shard}: {len(out)} tensors, {n_conv} projections -> MXFP8, {time.time() - t0:.0f} s", flush=True)
        del out
    index_out = {"metadata": {"total_size": total_bytes}, "weight_map": new_map}
    json.dump(index_out, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)

    cfg = json.load(open(os.path.join(src, "config.json")))
    qc = cfg["quantization_config"]
    num_layers = cfg["num_hidden_layers"]
    group_size = qc["config_groups"]["group_0"]["weights"]["group_size"]
    quantized_layers: dict[str, dict] = {}
    for i in range(num_layers):
        for p in ("gate_proj", "up_proj", "down_proj"):
            quantized_layers[f"model.layers.{i}.mlp.{p}"] = {"quant_algo": "NVFP4", "group_size": group_size}
    for base in converted:
        quantized_layers[base] = {"quant_algo": "MXFP8"}
    ignore = ["lm_head", "model.embed_tokens"]
    kept_bf16 = sorted({"q_proj", "k_proj", "v_proj", "g_proj", "o_proj"} - projs)
    cfg["quantization_config"] = {
        "quant_algo": "MIXED_PRECISION",
        "quant_method": "modelopt",
        "producer": qc.get("producer"),
        "group_size": group_size,
        "ignore": ignore,
        "quantized_layers": quantized_layers,
        "vllm_moet_derived": {
            "from": "ricursiveAI/Ripple-65B-Preview-NVFP4",
            "attention_projections_mxfp8": sorted(projs),
            "attention_projections_bf16": kept_bf16,
            "tool": "tools/ripple_sm120/make_mxfp8_attn_checkpoint.py",
        },
    }
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    for fn in os.listdir(src):
        p = os.path.join(src, fn)
        if fn in ("config.json", "model.safetensors.index.json") or fn.endswith(".safetensors") or fn == ".download-complete":
            continue
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(dst, fn), dirs_exist_ok=True)
        else:
            shutil.copy2(p, os.path.join(dst, fn))
    print(f"done: {len(converted)} projections -> MXFP8 ({sorted(projs)}), {total_bytes / 2**30:.1f} GiB of tensors, "
          f"{time.time() - t_start:.0f} s; BF16 kept: {kept_bf16}", flush=True)


if __name__ == "__main__":
    main()
