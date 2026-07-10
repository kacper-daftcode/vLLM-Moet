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
| `moe_w2_mm_k6144.cubin` | `moe_w2_mm_k6144.sass` | `moe_w2_mm` | 2‑bit MoE GEMM, MC=1, **K=6144** (gate‑up @ hidden 6144 — **GLM‑5.x**) |
| `moe_w2_mm_mc2_k6144.cubin` | `moe_w2_mm_mc2_k6144.sass` | `moe_w2_mm` | MC=2 (prefill), K=6144 |
| `moe_w2_mm_mc4_k6144.cubin` | `moe_w2_mm_mc4_k6144.sass` | `moe_w2_mm` | MC=4 (prefill), K=6144 |
| `moe_w2_mm_k7168.cubin` | `moe_w2_mm_k7168.sass` | `moe_w2_mm` | 2‑bit MoE GEMM, MC=1, **K=7168** (gate‑up @ hidden 7168 — **Kimi‑K2.x**) |
| `moe_w2_mm_mc2_k7168.cubin` | `moe_w2_mm_mc2_k7168.sass` | `moe_w2_mm` | MC=2 (prefill), K=7168 |
| `moe_w2_mm_mc4_k7168.cubin` | `moe_w2_mm_mc4_k7168.sass` | `moe_w2_mm` | MC=4 (prefill), K=7168 |
| `moe_w2_mm_mc4afrag_k4096.cubin` | `moe_w2_mm_mc4afrag_k4096.sass` | `moe_w2_mm` | **AFRAG** (prefill, fragment‑major A), K=4096 |
| `moe_w2_mm_mc4afrag_k2048.cubin` | `moe_w2_mm_mc4afrag_k2048.sass` | `moe_w2_mm` | AFRAG (prefill), K=2048 |
| `moe_w2_mm_mc4afrag_k1024.cubin` | `moe_w2_mm_mc4afrag_k1024.sass` | `moe_w2_mm` | AFRAG (prefill), K=1024 (TP2) |
| `moe_w2_mm_mc4afrag_k512.cubin` | `moe_w2_mm_mc4afrag_k512.sass` | `moe_w2_mm` | AFRAG (prefill), K=512 (TP4), NWARP=4 |
| `moe_w2_mm_mc4afrag_k6144.cubin` | `moe_w2_mm_mc4afrag_k6144.sass` | `moe_w2_mm` | AFRAG (prefill), K=6144 (GLM‑5.x) |
| `moe_w2_mm_mc4afrag_k7168.cubin` | `moe_w2_mm_mc4afrag_k7168.sass` | `moe_w2_mm` | AFRAG (prefill), K=7168 (Kimi‑K2.x) |

2‑bit planes = sign‑symmetric `{−4,−1,1,4}` + UE8M0 block‑32 scales; PRMT‑LUT in‑register
decode → `QMMA.SF` tensor‑core. Regcount 64 → 4 CTA/SM.

**AFRAG** (`mc4afrag`) = MC=4 with **fragment‑major activations**: each lane's m16k32 QMMA
A‑fragment loads in ONE `LDG.E.128` instead of 8 strided 4‑byte loads. Prefill moe_w2_mm is
L1/load‑issue bound (DRAM ~36%, L2 hit ~81%), so this cuts the dominant load class ~4× at
identical occupancy — measured **1.30×** (K=4096) / **1.27×** (K=2048) at the production
prefill shape (624 pairs, expert‑sorted, locked clocks). Bit‑identical outputs vs `mc4`
(validated per K incl. determinism; `MOEW2_MC=4 MOEW2_AFRAG=1 gen_moe_w2.py`). The glue
repacks A with a single‑pass triton kernel (`_afrag_repack`, ~65 µs @ [9984,4096]) into
dedicated `a1f/a2f` buffers; default ON via `VLLM_MOE_W2_AFRAG` (set `0` to fall back to mc4).

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
| `moe_w4_mm_k6144.cubin` | `moe_w4_mm_k6144.sass` | `moe_w4_mm` | FP4 delta GEMM, **K=6144** (gate‑up @ hidden 6144 — **GLM‑5.x**) |
| `moe_w4_mm_k7168.cubin` | `moe_w4_mm_k7168.sass` | `moe_w4_mm` | FP4 delta GEMM, **K=7168** (gate‑up @ hidden 7168 — **Kimi‑K2.x**) |
| `moe_w4_mm_k8192.cubin` | `moe_w4_mm_k8192.sass` | `moe_w4_mm` | FP4 GEMM, **K=8192** (dense wo_b — dense‑FP4 PoC, pairs=1 desc) |

## Sparse‑MLA prefill (SM120)
| cubin | SASS | kernel | purpose |
|---|---|---|---|
| `mla_prefill_state.cubin` | `mla_prefill_state2.sass` | `mla_prefill_state2` | prefill accumulate, fp8 smem staging, 2 CTA/SM (~2.2× vs Triton) |

> Only the kernels the server actually loads are shipped (verified against prod logs). The cubit
> sparse‑MLA **decode** kernels and the **mHC** kernels are opt‑in/experimental (not loaded in the
> shipped config) and live in the `cubit` repo (@ `5912400`); they are intentionally omitted here.

Validation (vs torch/Triton reference, rel ~1–3e‑3, deterministic): `tools/test_moe_w2_planes.py`,
`tools/test_moe_w2_forward.py`, `tools/test_mqa_logits.py`.
