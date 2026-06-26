# The patch — vs official vLLM `v0.19.2rc0`

`vllm-moet.patch` is the **complete runtime delta** that turns official vLLM `v0.19.2rc0`
(`aeee7ef9`) into the SM120 DeepSeek‑V4‑Flash + 2‑bit‑MoE engine. It's a single
`git diff` over `vllm/`, `csrc/`, and the build files (tests/docs/benchmarks excluded).

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout v0.19.2rc0
git apply /path/to/vLLM-Moet/patch/vllm-moet.patch
```

## What's in it (and who authored what)
The patch is one self‑contained diff against **official vLLM** — *not* a patch layered on a
fork. It bundles two distinct bodies of work:

1. **DeepSeek‑V4 enablement + SM120 support** — the DS4 model, sparse‑MLA, indexer, MTP, and the
   SM120 paths (CUTLASS/Triton/cubit instead of DeepGEMM, which has no SM120 kernels). Authored
   by vLLM contributors and the base‑fork (`kacper-daftcode`) authors. Needed because
   v0.19.2rc0 predates DS4 and upstream's later DS4 doesn't run on sm120.
2. **This project's contribution** — 2‑bit MoE + FP4 delta + cubit dispatch (the files below).

## `vllm-new-files/` — this project's wholly‑new files (quick review)
| file (→ vLLM path) | what it is |
|---|---|
| `moe_w2_planes.py` → `…/quantization/utils/` | 2‑bit sign‑symmetric expert‑plane quantizer/packers |
| `moe_w2_cubit.py` → same dir | plane build at load + capturable `moe_w2_forward` (→ `moe_w2_mm`) |
| `moe_w2_delta.py` → same dir | FP4 hot‑expert delta tier (host‑pinned planes, LFRU pool, `moe_w4_mm`) |
| `cubit_sparse_mla.py` → `…/v1/attention/backends/mla/` | ctypes dispatch for the cubit sparse‑MLA prefill kernel |

Our work also touches a few base files (hooks): `quantization/mxfp4.py`, `deepseek_v4_attention.py`,
`utils/deep_gemm.py` (SM120 triton MQA‑logits), `envs.py`. Those edits are in `vllm-moet.patch`.
