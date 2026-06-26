# Kernel manifest (SM120 / sm_120, RTX PRO 6000 Blackwell + RTX 5090)

Hand‑written SASS assembled by `cubit` (pinned @ `5912400`). Cubins are **SM120‑only**.
Assemble: `cubit asm sass/<SASS> -o <cubin> --kernel <kernel> --mercury-stub sass/qmma_e4m3.merc.stub`.

## 2‑bit MoE GEMM — the core contribution (no upstream equivalent)
| cubin (`cubins-sm120/`) | SASS (`sass/`) | kernel | purpose |
|---|---|---|---|
| `moe_w2_mm_k4096.cubin` | `moe_w2_mm.sass` | `moe_w2_mm` | 2‑bit MoE GEMM, MC=1 (decode), K=4096 (w13) |
| `moe_w2_mm_k2048.cubin` | `moe_w2_mm_k2048.sass` | `moe_w2_mm` | 2‑bit MoE GEMM, MC=1 (decode), K=2048 (w2) |
| `moe_w2_mm_mc2_k4096.cubin` | `moe_w2_mm_mc2.sass` | `moe_w2_mm` | MC=2 (prefill), K=4096 |
| `moe_w2_mm_mc2_k2048.cubin` | `moe_w2_mm_mc2_k2048.sass` | `moe_w2_mm` | MC=2 (prefill), K=2048 |
| `moe_w2_mm_mc4_k4096.cubin` | (via `gen/gen_moe_w2.py` MC=4) | `moe_w2_mm` | MC=4 (prefill, full QMMA‑M), K=4096 |
| `moe_w2_mm_mc4_k2048.cubin` | (via `gen/gen_moe_w2.py` MC=4) | `moe_w2_mm` | MC=4 (prefill), K=2048 |
| `moe_w2_mm_k1024.cubin` | `moe_w2_mm_k1024.sass` | `moe_w2_mm` | 2‑bit MoE GEMM, MC=1, **K=1024** (w2 under TP2) |
| `moe_w2_mm_mc2_k1024.cubin` | `moe_w2_mm_mc2_k1024.sass` | `moe_w2_mm` | MC=2 (prefill), K=1024 |
| `moe_w2_mm_mc4_k1024.cubin` | `moe_w2_mm_mc4_k1024.sass` | `moe_w2_mm` | MC=4 (prefill), K=1024 |
| `moe_w2_mm_k512.cubin` | `moe_w2_mm_k512.sass` | `moe_w2_mm` | 2‑bit MoE GEMM, MC=1, **K=512** (w2 under TP4), NWARP=4 |
| `moe_w2_mm_mc2_k512.cubin` | `moe_w2_mm_mc2_k512.sass` | `moe_w2_mm` | MC=2 (prefill), K=512 |
| `moe_w2_mm_mc4_k512.cubin` | `moe_w2_mm_mc4_k512.sass` | `moe_w2_mm` | MC=4 (prefill), K=512 |

2‑bit planes = sign‑symmetric `{−4,−1,1,4}` + UE8M0 block‑32 scales; PRMT‑LUT in‑register
decode → `QMMA.SF` tensor‑core. Regcount 64 → 4 CTA/SM.

**Contraction K & multi‑GPU.** K is the per‑cubin GEMM contraction: **4096** = w13/gate‑up (hidden
H, never sharded) and **2048** = w2/down (intermediate I) on a single GPU. Under **tensor parallelism
(TP2)** the w2‑side contraction shards to **K=1024** (the `*_k1024` cubins; w13 stays 4096), and
**TP4** shards it further to **K=512** (the `*_k512` cubins) — the delta tier shards its planes per
rank, so TPn requires the matching K cubins. **Pipeline parallelism (PP)** splits whole layers (each
expert stays intact, no intra‑layer shard) → it reuses the existing K=2048/4096 cubins unchanged.
Generators `gen/gen_moe_w2.py <out.sass> <K>` and `gen/gen_moe_w4.py <out.sass> <K>` emit any K;
**NWARP = split‑K warps is auto‑chosen as K/NWARP must be a multiple of 128 (K≥1024→8, K=512→4)**,
and the loader (`moe_w2_cubit.py::_nwarp_for_k`) launches the matching thread count per K. Op‑validated
by `gen/moe_w2_check.py` / `gen/moe_w4_check.py` (K=512 rel ~2–3e‑3, deterministic; M up to 16).

## FP4 "delta" tier GEMM
| cubin | SASS | kernel | purpose |
|---|---|---|---|
| `moe_w4_mm_k4096.cubin` | `moe_w4_mm.sass` | `moe_w4_mm` | FP4 (e2m1) hot‑expert delta GEMM, K=4096 |
| `moe_w4_mm_k2048.cubin` | `moe_w4_mm_k2048.sass` | `moe_w4_mm` | FP4 delta GEMM, K=2048 |
| `moe_w4_mm_k1024.cubin` | `moe_w4_mm_k1024.sass` | `moe_w4_mm` | FP4 delta GEMM, **K=1024** (w2 under TP2) |
| `moe_w4_mm_k512.cubin` | `moe_w4_mm_k512.sass` | `moe_w4_mm` | FP4 delta GEMM, **K=512** (w2 under TP4), NWARP=4 |

## Sparse‑MLA prefill (SM120)
| cubin | SASS | kernel | purpose |
|---|---|---|---|
| `mla_prefill_state.cubin` | `mla_prefill_state2.sass` | `mla_prefill_state2` | prefill accumulate, fp8 smem staging, 2 CTA/SM (~2.2× vs Triton) |

> Only the kernels the server actually loads are shipped (verified against prod logs). The cubit
> sparse‑MLA **decode** kernels and the **mHC** kernels are opt‑in/experimental (not loaded in the
> shipped config) and live in the `cubit` repo (@ `5912400`); they are intentionally omitted here.

Validation (vs torch/Triton reference, rel ~1–3e‑3, deterministic): `tools/test_moe_w2_planes.py`,
`tools/test_moe_w2_forward.py`, `tools/test_mqa_logits.py`.
