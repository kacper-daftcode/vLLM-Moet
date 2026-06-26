# Build & Run — vLLM‑Moet (SM120 / RTX PRO 6000)

Everything here is expressed as **a patch + kernels on top of an official vLLM tag**. (Rebasing
the patch onto a newer official tag is future work; see "Why this tag".)

## Pinned base
| component | source | ref | role |
|---|---|---|---|
| vLLM | `github.com/vllm-project/vllm` (**official**) | tag `v0.19.2rc0` (`aeee7ef9`) | base engine |
| **`patch/vllm-moet.patch`** | this repo | — | DS4‑Flash + SM120 + 2‑bit MoE + FP4 delta + cubit dispatch |
| cubit | `github.com/kacper-daftcode/cubit` | `5912400` | SASS assembler (only to rebuild cubins) |
| checkpoint | DeepSeek‑V4‑Flash (official) | — | FP4 MoE experts + FP8 dense; **not redistributed** |

## Why this tag (and not the latest official vLLM)
Official vLLM gained DeepSeek‑V4 in v0.20.0+, but its DS4 FP8 path routes through **DeepGEMM**
(o‑proj einsum, indexer MQA‑logits, …) and **DeepGEMM has no SM120 kernels** (its CUDA is
sm90/sm100 only; `support_deep_gemm()` = cap 90 or family‑100). So the latest official DS4 does
not run on consumer Blackwell (sm120). This patch carries the **SM120‑enabled** DS4 path
(CUTLASS/Triton/cubit instead of DeepGEMM) and was developed against `v0.19.2rc0`. A rebase onto
a newer tag would need SM120 fallbacks at each upstream DeepGEMM call site.

## 1. Build the image — one command, self-contained
The Dockerfile clones official vLLM at the pinned tag, applies `patch/vllm-moet.patch`, builds
for sm_120, and **bakes in the prebuilt SM120 cubins** — no manual clone/patch, no `cubit`:
```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120 -t vllm-moet-sm120:base .   # ~15-25 min
#   small host: --build-arg MAX_JOBS=8      different vLLM tag: --build-arg VLLM_REF=...
```
(Prefer a non-Docker build? The patch applies standalone: `git clone vllm && git checkout
v0.19.2rc0 && git apply patch/vllm-moet.patch`. `patch/vllm-new-files/` mirrors the wholly-ours
files for quick review.)

## 2. Rebuild kernels — OPTIONAL (`cubit` is NOT needed to run)
Prebuilt SM120 cubins ship in `kernels/cubins-sm120/` and are baked into the image. The SASS
sources, generators and QMMA stub live here too (`kernels/sass/`, `kernels/gen/`); `cubit` is
just the SM120 assembler, for rebuilding/auditing. The toolchain underneath is ours:
[`blackwell-isa`](https://github.com/kacper-daftcode/blackwell-isa) (machine‑readable SM120 ISA
database) → [`cubit`](https://github.com/kacper-daftcode/cubit) (SASS assembler/disassembler) →
the `.sass` kernels here.
```bash
git clone https://github.com/kacper-daftcode/cubit && cd cubit && cargo build --release  # @5912400
M=/path/to/vLLM-Moet/kernels
python3 "$M/gen/gen_moe_w2.py" "$M/sass/moe_w2_mm_k512.sass" 512          # any contraction K
./target/release/cubit asm "$M/sass/moe_w2_mm_k512.sass" --kernel moe_w2_mm \
    -o "$M/cubins-sm120/moe_w2_mm_k512.cubin" --mercury-stub "$M/sass/qmma_e4m3.merc.stub"
python3 "$M/gen/moe_w2_check.py"     # op-validate vs torch reference (rel ~2-3e-3)
```
`kernels/MANIFEST.md` maps every cubin ↔ SASS ↔ kernel (K=4096/2048 single-GPU, 1024 TP2, 512 TP4).

## 3. Quantization (no offline step to serve)
2‑bit MoE planes are built **at load** from the official checkpoint's FP4 codes
(`moe_w2_planes.py`/`moe_w2_cubit.py`); the FP4 delta planes are staged pinned‑host at load. You
only need the official checkpoint. `repack_expert_bits.py` is the *quality‑probe* tool, not a
serving step.

## 4. Serve
```bash
MODEL=/path/to/DeepSeek-V4-Flash bash tools/serve.sh 0 8000   # GPUs "0" (1×) / "0,1" (TP2); port 8000
```
Cubins are baked into the image — set `CUBINS=/path/to/kernels/cubins-sm120` only to override with a
local dir. `serve.sh` is fully generic (no host‑specific paths). Knobs via env (defaults shown):
`MOE_W2=1` (`0` = native FP4 reference), `GATE=0` (`1` = FP4 confidence gate), `DELTA_GB=1.0` (FP4
delta pool GiB; `0` disables), `MAX_LEN=262144`, `UTIL=0.97`, `NUM_LAYERS=44`, `PP=1`, `PP_PARTITION`,
`SEQS`.

## 5. Validate
```bash
python3 tools/probe_quant_quality.py <port>        # MTP acceptance + coherence + arithmetic
python3 tools/needle_probe.py <port> 48000 0.1     # long-context retrieval
python3 tools/prefill_probe.py <port> 16000,100000 # cold-prefill TTFT
```
See `docs/quality.md` (MTP acceptance ≥ official FP4, 12/12 coherence, arithmetic recovered with delta).
