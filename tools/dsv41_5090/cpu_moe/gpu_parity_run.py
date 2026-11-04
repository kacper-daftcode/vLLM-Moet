"""GPU side of the CPU-vs-GPU expert parity check: runs inside a vLLM image that vendors DeepGEMM (sm_120).

    docker run --rm --gpus device=0 -v /mnt/nvme/cache/ds41/deep_gemm:/root/.deep_gemm \
      -v <cpu_moe dir>:/work:ro -v <parity dir>:/parity --entrypoint python3 vllm-moet-sm120:dsv41-w2 \
      /work/gpu_parity_run.py /parity/inputs.npz /parity/report.json

Builds the expert as vLLM lays it out for DeepGemmFP4Experts (TP4 slice, intermediate padded 576 -> 640, scales
through deepgemm_post_process_weight_scale_block), feeds the SAME fp8 codes + UE8M0 scales the CPU kernel got,
and compares: FC1 bf16 output vs the emulation; vLLM's fused SiLU-mul-quant (codes + scales) vs the emulation;
FC2 output vs the CPU kernel (with the GPU's own requantized activation and with the CPU's). Also checks the
activation quantizer itself: vLLM's per_token_group_quant_fp8_packed_for_deepgemm on the bf16 hidden states vs
the host emulation's codes / scales.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

inp, report_path = sys.argv[1], sys.argv[2]
Z = np.load(inp)
T = int(Z["xq"].shape[0])
HID, INTER_R, INTER_PAD = 5120, 576, 640
dev = torch.device("cuda:0")

from vllm.utils.deep_gemm import (  # noqa: E402
    get_mk_alignment_for_contiguous_layout,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    DeepGemmQuantScaleFMT,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    deepgemm_post_process_weight_scale_block,
    per_token_group_quant_fp8_packed_for_deepgemm,
    silu_mul_quant_fp8_packed_triton,
)

report = {"T": T, "layer": int(Z["layer"]), "expert": int(Z["expert"]), "rank": int(Z["rank"])}
fmt = DeepGemmQuantScaleFMT.from_oracle()
report["scale_fmt"] = str(fmt)
align = get_mk_alignment_for_contiguous_layout()[0]
M = max(align, ((T + align - 1) // align) * align)
report["m_alignment"] = int(align)
print(f"scale format {fmt}, M alignment {align}, M_pad {M}")


def u8(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev)


# ---- weights in vLLM's layout: w13 [1, 2*640, 2560] (gate rows, pad, up rows, pad), w2 [1, 5120, 320] -------
w13 = torch.zeros((1, 2 * INTER_PAD, HID // 2), dtype=torch.uint8, device=dev)
w13s = torch.zeros((1, 2 * INTER_PAD, HID // 32), dtype=torch.uint8, device=dev)
w13[0, :INTER_R] = u8(Z["w1n"]); w13s[0, :INTER_R] = u8(Z["w1s"])
w13[0, INTER_PAD:INTER_PAD + INTER_R] = u8(Z["w3n"]); w13s[0, INTER_PAD:INTER_PAD + INTER_R] = u8(Z["w3s"])
w2 = torch.zeros((1, HID, INTER_PAD // 2), dtype=torch.uint8, device=dev)
w2s = torch.zeros((1, HID, INTER_PAD // 32), dtype=torch.uint8, device=dev)
w2[0, :, :INTER_R // 2] = u8(Z["w2n"]); w2s[0, :, :INTER_R // 32] = u8(Z["w2s"])
w13s_p = deepgemm_post_process_weight_scale_block(ws=w13s, mn=2 * INTER_PAD, k=HID, quant_block_shape=(1, 32), num_groups=1)
w2s_p = deepgemm_post_process_weight_scale_block(ws=w2s, mn=HID, k=INTER_PAD, quant_block_shape=(1, 32), num_groups=1)
print("packed weight scales:", tuple(w13s_p.shape), w13s_p.dtype, tuple(w2s_p.shape), w2s_p.dtype)

# ---- activations: the CPU kernel's exact inputs (fp8 codes + UE8M0 bytes) in DeepGEMM's packed layout ----------
expert_ids = torch.full((M,), -1, dtype=torch.int32, device=dev)
expert_ids[:T] = 0


def pack_scales(sbytes: np.ndarray, M_pad: int) -> torch.Tensor:
    """[T, G] uint8 (e + 127) -> the layout silu_mul_quant_fp8_packed_triton produces: int32 [M, ceil(G/4)] view of
    a (packs, aligned_m) buffer (column-major), 4 bytes per int32, low byte = first group."""
    Tn, G = sbytes.shape
    packs = (G + 3) // 4
    buf = np.zeros((packs, M_pad), dtype=np.int32)
    for p in range(packs):
        v = np.zeros(Tn, dtype=np.int64)
        for j in range(4):
            g = p * 4 + j
            if g < G:
                v |= sbytes[:, g].astype(np.int64) << (8 * j)
        buf[p, :Tn] = v.astype(np.int32)
    return torch.from_numpy(buf).to(dev).T[:M_pad, :]


a1q = torch.zeros((M, HID), dtype=torch.uint8, device=dev)
a1q[:T] = u8(Z["xq"])
a1q = a1q.view(torch.float8_e4m3fn)
a1s = pack_scales(Z["xs"], M)

# the GPU's own quantizer on the bf16 hidden states vs the host emulation's codes / scales
x_bf16 = torch.from_numpy(Z["x_bf16"]).view(torch.bfloat16).to(dev)
try:
    gq, gs = per_token_group_quant_fp8_packed_for_deepgemm(x_bf16, 128, use_ue8m0=True)
    gq_b = gq.view(torch.uint8).cpu().numpy()
    gs_b = np.zeros_like(Z["xs"])
    gs_np = gs.cpu().numpy()
    for g in range(gs_b.shape[1]):
        gs_b[:, g] = (gs_np[:, g // 4] >> (8 * (g % 4))) & 0xFF
    report["act_quant_gpu_vs_emu"] = dict(code_mismatch=int((gq_b != Z["xq"]).sum()), codes=int(gq_b.size),
                                           scale_mismatch=int((gs_b != Z["xs"]).sum()), scales=int(gs_b.size))
    print("GPU activation quantizer vs host emulation:", report["act_quant_gpu_vs_emu"])
    report["act_scale_layout_ok"] = bool(torch.equal(a1s.contiguous(), pack_scales(gs_b, M).contiguous()) or
                                         int((gs_b != Z["xs"]).sum()) > 0)
except Exception as e:  # noqa: BLE001
    report["act_quant_gpu_vs_emu"] = f"failed: {e!r}"
    print("GPU activation quantizer check failed:", repr(e))


def cmp(name, got: torch.Tensor, ref: np.ndarray):
    g = got.float().cpu().numpy()
    r = ref.astype(np.float32)
    d = np.abs(g - r)
    rms = float(np.sqrt((r.astype(np.float64) ** 2).mean()))
    ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(r), 1e-30)))) * 2.0 ** -7
    res = dict(n=int(d.size), mismatches=int((d > 0).sum()), gt1ulp=int((d > ulp * 1.0001).sum()),
               max_abs=float(d.max()), max_rel_rms=float(d.max() / max(rms, 1e-30)))
    report[name] = res
    print(f"  {name:34s} mismatches {res['mismatches']:6d} / {res['n']}  >1ulp {res['gt1ulp']:5d}  "
          f"max|d| {res['max_abs']:.3e}  max|d|/rms {res['max_rel_rms']:.2e}")
    return res


# ---- FC1 ----------------------------------------------------------------------------------------------------
out1 = torch.empty((M, 2 * INTER_PAD), dtype=torch.bfloat16, device=dev)
m_grouped_fp8_fp4_gemm_nt_contiguous((a1q, a1s), (w13.view(torch.int8), w13s_p), out1, expert_ids,
                                     recipe_a=(1, 128), recipe_b=(1, 32))
torch.cuda.synchronize()
print("FC1 (DeepGEMM fp8 x fp4) vs the emulation's bf16 gate / up:")
cmp("fc1_gate", out1[:T, :INTER_R], Z["emu_g"])
cmp("fc1_up", out1[:T, INTER_PAD:INTER_PAD + INTER_R], Z["emu_u"])
pad_max = float(out1[:T, INTER_R:INTER_PAD].abs().max().item() + out1[:T, INTER_PAD + INTER_R:].abs().max().item())
report["fc1_pad_rows_abs_max"] = pad_max

# ---- activation + requant (vLLM's fused kernel, clamp 10) -----------------------------------------------------
a2q = torch.empty((M, INTER_PAD), dtype=torch.float8_e4m3fn, device=dev)
a2q, a2s = silu_mul_quant_fp8_packed_triton(input=out1, output_q=a2q, group_size=128, clamp_limit=10.0)
torch.cuda.synchronize()
a2q_b = a2q.view(torch.uint8)[:T, :INTER_R].cpu().numpy()
a2s_np = a2s[:T].cpu().numpy()
a2s_b = np.zeros((T, 5), dtype=np.uint8)
for g in range(5):
    a2s_b[:, g] = (a2s_np[:, g // 4] >> (8 * (g % 4))) & 0xFF
emu_ys = (Z["emu_ye"] + 127).astype(np.uint8)
report["act_gpu_vs_emu"] = dict(code_mismatch=int((a2q_b != Z["emu_yq"]).sum()), codes=int(a2q_b.size),
                                scale_mismatch=int((a2s_b != emu_ys).sum()), scales=int(emu_ys.size))
print("fused SiLU-mul-quant (GPU) vs emulation:", report["act_gpu_vs_emu"])

# ---- FC2 with the GPU's activation, and with the CPU's ---------------------------------------------------------
out2 = torch.empty((M, HID), dtype=torch.bfloat16, device=dev)
m_grouped_fp8_fp4_gemm_nt_contiguous((a2q, a2s), (w2.view(torch.int8), w2s_p), out2, expert_ids,
                                     recipe_a=(1, 128), recipe_b=(1, 32))
torch.cuda.synchronize()
cpu_out = torch.from_numpy(Z["cpu_out_bf16"].astype(np.int32) << 16).view(torch.float32).numpy()
print("FC2 (GPU activation) vs the CPU kernel / the emulation:")
cmp("fc2_gpuact_vs_cpu_kernel", out2[:T], cpu_out)
cmp("fc2_gpuact_vs_emu", out2[:T], Z["emu_out"])
a2q_cpu = torch.zeros((M, INTER_PAD), dtype=torch.uint8, device=dev)
a2q_cpu[:T, :INTER_R] = u8(Z["emu_yq"])
a2s_cpu = pack_scales(emu_ys, M)
out2b = torch.empty((M, HID), dtype=torch.bfloat16, device=dev)
m_grouped_fp8_fp4_gemm_nt_contiguous((a2q_cpu.view(torch.float8_e4m3fn), a2s_cpu), (w2.view(torch.int8), w2s_p), out2b,
                                     expert_ids, recipe_a=(1, 128), recipe_b=(1, 32))
torch.cuda.synchronize()
print("FC2 (CPU's requantized activation) vs the CPU kernel:")
cmp("fc2_cpuact_vs_cpu_kernel", out2b[:T], cpu_out)
with open(report_path, "w") as f:
    json.dump(report, f, indent=1)
print(f"wrote {report_path}")
