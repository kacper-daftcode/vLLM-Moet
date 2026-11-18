# Inkling-2-Small on 4× RTX PRO 6000 (sm_120)

Thinking Machines' [`Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small) — a 594B
mixture-of-experts decoder with 17B active parameters per token (48 layers, d_model 4608, 48 query / 8 KV heads of
128; five 1024-token sliding-window layers per full-attention layer, relative-position logits instead of positional
embeddings, short convolutions on K, V and the residual stream, attention residuals over blocks of six layers; 384
experts per layer, 8 per token; a vision tower; 1M positions) — from its released checkpoint (394 GB: the experts of
layers 0–39 in NVFP4, of layers 40–47 in MXFP8, everything else BF16, a DSpark draft of five layers in the same
directory) **with the experts of layers 40–47 repacked from MXFP8 to six bits as they load - a lossy change of those
weights (next paragraph)**, TP4 with the experts expert-parallel over the TP group, the draft on. vLLM knows the
architecture in the day-0 image `vllm/vllm-openai:inkling-2-small` (`Inkling2ForConditionalGeneration`,
`--reasoning-parser inkling`, `--tool-call-parser inkling`), whose kernels are written for sm_100;
[`Dockerfile.sm120-inkling2`](../Dockerfile.sm120-inkling2) adds what sm_120 needs to run it. Kernels, patchers and
tests: [`tools/inkling2_sm120/README.md`](../tools/inkling2_sm120/README.md); every number below with its method:
[the measurement record](benchmarks/inkling2-sm120-2026-11-13.md).

**On four 96 GB cards the experts of layers 40–47 are held as e2m3, six bits per value, not as the checkpoint's
MXFP8** (`MXFP6=1`, the launcher's default; the image's own default is the released precision): 8.25 → 6.25 bits
per weight, 5.7 GiB less per card. It is a lossy change, made for room. A rank's weights take 85.9 GiB of a card's
95.0 packed and 91.6 GiB as released. As released the model starts on the four cards with 0.69 GiB left for the
paged cache - measured without the draft, with an 8 192-token context and 64-token prefill chunks, where the packed
model has 6.38 GiB. In the launcher's configuration (65 536 tokens, 512-token chunks, 32 sequences, the draft) the
packed model has 3.3 GiB of paged cache beside the draft's 1.0 GiB of weights and 1.4 GiB of CUDA graphs, and one
request of the full context reserves 2.4 GiB of it.

Within each block of 32 weights the three largest binades keep their values exactly and smaller weights are
rounded to an eighth of the third binade's lower end. Over 96 experts of the eight layers, 76–79 % of the weights
keep their value (the 66–69 % that lie in those binades and the smaller ones that fall on the grid), the largest
error is 1 / 64 of a block's largest magnitude, the error energy −38 to −39 dB of the weight energy.

Against the released weights, on the full model in the one configuration in which both precisions were served (the
same image and arguments: an 8K context, 64-token chunks, no draft): 194 of 200 GSM8K questions packed and 195 as
released, the same answer to 197 of the 200; of 24 greedy answers to short prompts 19 are the same text to the end,
where two passes through one server give 24 of 24; decode is 1 % faster packed at one request, up to 3 % at eight. A
six-layer cut of the checkpoint with three of the repacked layers, served on one card both ways, returns the same
greedy tokens in 8 of 8 probes of 48 tokens (|Δ logprob| median 0.003, max 0.65). In the launcher's configuration
the packed model answers 193 of 200 in each of three runs. The packing changes the text a greedy decode produces;
GSM8K shows no difference at 200 questions, where a loss below about three points would not show. Not measured:
the released precision beyond 8K tokens or with the draft. Everything else is the checkpoint's own arithmetic.

| step | change | switch | what it is for |
|---|---|---|---|
| 1 | the MoE's batched expert GEMM on `mma.sync` block-scaled MMAs — NVFP4 (`kind::mxf4nvf4`, m16n8k64), MXFP8 and MXFP8 × e2m3 (`kind::mxf8f6f4`, m16n8k32) — a persistent kernel that replays from a CUDA graph; expert weights in a step-major layout, loaded a few experts at a time | `VLLM_MOET_INKLING2_BMM=0` (vLLM's path: other architectures only), `VLLM_MOET_INKLING2_MXFP6=1` (the packing; the image's default is 0, the launcher sets 1) | vLLM's kernels here are FlashInfer's TRT-LLM-gen cubins for sm_100: there is none for the device |
| 2 | the relative attention of decode, verify, the DSpark draft and prefill in one `mma.sync` m16n8k16 kernel (paged K / V through a cp.async ring, the sheared relative logits, split-KV, gate and sink) | `VLLM_MOET_INKLING2_ATTN=kernel\|ref\|0` (0 = vLLM's path: other architectures only) | vLLM's decode kernel is tcgen05 with tensor memory, its prefill kernel tml_fa4's sm_100 forward; the fallback it has for sm_103 does not read the page table on capability 12 |
| 3 | linears of up to 32 rows on cuBLAS; the router's K-group partials on vLLM's `mma.sync` Triton kernel at every batch size, with the stage count that fits 99 KB of shared memory | `VLLM_MOET_INKLING2_LINEAR=0` (vLLM's path: other architectures only); `VLLM_MOET_INKLING2_LINEAR_KERNEL=1` (an `mma.sync` kernel instead of cuBLAS: no faster, off) | FlashInfer's CuTe DSL low-M kernels exist for capabilities 100 / 103 / 107; the Triton TMA kernel for 2048+ rows stages 197 KB |
| 4 | the vision tower's attention: tml_fa4's forward class for capability 12 keeps the Ampere code paths | — (build arg) | the class turns on a TMA output path its kernel has no atom for: the compile fails at start-up |
| 5 | a TP group's embedding without the fused kernel of vLLM's compiled library | `VLLM_MOET_FUSED_EMBEDDING=1` (vLLM's rule) | the base image's library has cubins for sm_80 / sm_89 / sm_90 / sm_100 only; the unchecked launch ends a TP server in its first warm-up step |
| 6 | the DSpark draft's argmax as each rank's first local maximum and an all-gather | `VLLM_MOET_INKLING2_DSPARK_ARGMAX_TP=0` | vLLM's kernel needs NVLink multicast with more than one rank and has no image for capability 12 |
| 7 | the TP group's all-reduces of small batches on the PCIe P2P push with NCCL's summation order (the DeepSeek line's module, compiled for rows of 4608) — built in, **off**: without the draft it measures 4 % slower than PyNCCL at one request and 5 % at eight | `VLLM_MOET_P2P_AR=1`, `VLLM_MOET_P2P_AR_M_MAX` / `_MAX_ROWS` | vLLM's fused all-reduce of the residual stream needs NVLink multicast; without it every sublayer's partial is a PyNCCL all-reduce, 96 per decode step |
| 8 | the DSpark draft's K / V write (`reshape_and_cache_flash`, the scatter into the paged cache) on vLLM's Triton kernel of the operation | `VLLM_MOET_CACHE_FLASH_C=1` (the compiled kernel) | the kernel of vLLM's compiled library has no image for capability 12: a server with the draft ends in the draft's first forward |
| 9 | with a draft length by batch size (`SPEC_SCHEDULE`), the draft's CUDA graphs captured only up to the largest batch size that drafts | `VLLM_MOET_DRAFT_GRAPHS_ALL=1` (vLLM's rule) | a DSpark draft graph holds logits over the vocabulary per request and Markov step, and vLLM captures one for every batch size up to `max_num_seqs` also where the schedule never runs the draft (1.4 GiB of graphs at 32 sequences, on a card whose paged cache is 3.3 GiB) |

## Build and serve

```bash
hf download thinkingmachines/Inkling-2-Small --local-dir /srv/models/Inkling-2-Small     # 394 GB
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-inkling2 -t vllm-moet-sm120:inkling2-b974b306-r13 .   # ~8 min
# GPUS: four FREE cards - there is no default, and the launcher refuses a card that holds a process or memory
MODEL_DIR=/srv/models/Inkling-2-Small GPUS=4,5,6,7 PORT=8002 docker/sm120/run-inkling2.sh
```

The measured image is this Dockerfile at commit `1740c25`; what the tree has gained since is in the
[changelog](../CHANGELOG.md) under "In the tree after `-r13`" (the launcher, tests and comments; nothing a server
runs differently).

`GPUS` is required and names four free cards: a load puts 87 GiB on each, so the launcher asks `nvidia-smi` first
and starts nothing when one of them holds a compute process or more than 1 GiB. On a host that runs anything else
give the server its own cores and NUMA node as well (`CPUS=... CPUSET_MEMS=...`): without `CPUS` its processes run
on every core of the host. The container's memory is capped at 96 GiB (`MEMORY`, `--memory` = `--memory-swap`), the
cap of the measured servers, which held about 23 GiB at the end of a load and at most 36 GiB while serving.

The engine's arguments the launcher sets by default are the measured ones: the e2m3 packing, the DSpark draft of 7
tokens at every batch size, 32 sequences, a 65 536-token context, 512-token prefill chunks, the `inkling` reasoning
and tool-call parsers. `SPEC_TOKENS=0` serves without the draft (1024-token chunks, 1.2 GiB more paged cache);
`SPEC_SCHEDULE='[[1,8,7],[9,64,0]]'` drafts nothing above 8 requests in a step. Every knob is an environment
variable documented in the header of [`docker/sm120/run-inkling2.sh`](../docker/sm120/run-inkling2.sh). The API
listens on `127.0.0.1:8002` (`BIND=` for all interfaces). Thinking is set per request with the template's
`reasoning_effort` (`"chat_template_kwargs": {"reasoning_effort": "none"}` turns it off; the default is 0.9).

## Measurements

4× RTX PRO 6000 Blackwell at their 400 W cap, TP4 with the experts expert-parallel, PyNCCL all-reduces, thinking
off; the experts of layers 40–47 as e2m3 except in the last row. `tools/sm120_perf/spec_matrix.py`: N concurrent
requests of 512 tokens (256 at 1 and 8 requests in the "no draft" row), aggregate tok/s in the window in which all
of them stream / per request @ engine steps per second; a range is two waves. Every row with its files:
[the record](benchmarks/inkling2-sm120-2026-11-13.md).

| configuration | code 1 | code 8 | code 16 | code 32 | prose 1 | prose 8 | prose 16 | prose 32 |
|---|---|---|---|---|---|---|---|---|
| vLLM's own kernels (this image, every step switched off) | does not load: `No supported CUDA architectures found for major versions [10]` | | | | | | | |
| no draft | 149 @ 148 | 768 / 96 @ 95 | 1077–1114 / 67–70 @ 68 | 1692–1694 / 53 @ 53 | 149 @ 148 | 677 / 85 @ 84 | 987–1118 / 60–62 @ 60 | 1478–1481 / 46 @ 45 |
| DSpark, 7 tokens at every batch size (the default) | **481–498 @ 75** | **1777–1893 / 223–237 @ 35** | **3050 / 191 @ 28.5** | **4190–4237 / 131 @ 19.7** | 167–193 @ 76 | 599–643 / 74–75 @ 34 | 982–1021 / 59–61 @ 26.5 | 1386 (one wave) / 42 @ 19 |
| no draft, 8K context, 64-token chunks: packed | 147 @ 146 | 647–652 / 81–82 @ 81 | | | 147 @ 144 | 634 / 79 @ 78 | | |
| the same at the released precision (`MXFP6=0`) | 145 @ 144 | 632–661 / 79–83 @ 79–82 | | | 145 @ 143 | 617–619 / 77 @ 76 | | |

(Without the draft at 16 and 32 requests: the rows of a server whose schedule drafts nothing above 8 requests; in
that server with the draft at 2 / 4 requests: code 785–799 / 1186–1261 tok/s, prose 248–295 / 458–460. The last
two rows are the pair of the two precisions - one image, one set of arguments; the second wave of the packed prose
at eight had a common window of 2.3 s and is left out.)

Reading: a step with the draft takes 13.3 ms at one request against 6.8 ms without - 8 rows through the target instead
of one, the draft's five layers and its Markov head - and pays by the tokens a step yields: 6.7 on code (78–84 % of
the draft accepted), 5.2 on GSM8K answers (52–60 %), 2.2–2.3 on prose (17–18 %). Code runs 3.2–3.3× faster at one
request, 2.3–2.5× at eight, 2.7–2.8× at 16 and 2.5× at 32; prose 12–30 % faster at one request and 5–12 % slower at
eight, within 3 % at 16 (59–61 against 60–62 tok/s per request), 6–10 % slower at 32. A verification of 256 rows takes
51 ms against 19–22 ms for a step of 32 rows without the draft: a step reads an expert's weights once however many
rows use it, so eight times the rows cost 2.4 times the step (below). For loads of free prose at high concurrency
vLLM's draft length by batch size (`SPEC_SCHEDULE`) turns the draft off above a batch size. In the pair of the two
precisions the packed model is 1 % faster at one request and 3 % faster per request at eight requests of prose
(code is within the waves' spread): an e2m3 launch reads a quarter fewer bytes than an MXFP8 one.

Prefill: a fresh prompt of 20.9K tokens reaches its first token after 2.4 s (8.7K tok/s), one of 60.3K tokens after
7.5 s (8.0K tok/s) - 512-token chunks, with the draft; 8.8K tok/s in 1024-token chunks without it (a second prompt
of the same size in the same server: 10.4K tok/s). The same 17.4K-token prefix with another question: 0.07 s (the
prefix cache keeps the conv state with the K / V).

Memory per card: weights 85.9 GiB (86.9 GiB with the draft; 91.6 GiB at the released precision); the weights load
in 113–115 s per rank (107 s without the packing), the model with its initialisation in 116–149 s, and a start
reaches `/health` after 177–233 s with warm kernel caches; the paged cache 4.5–4.6 GiB
without the draft (32 and 8 sequences) and 3.3 GiB with it (1.40 requests of 65 536 tokens by vLLM's count); with
an 8K context and no draft 6.38 GiB packed and 0.69 GiB at the released precision
([below](#what-is-left-of-a-card-the-paged-cache)).

Quality: GSM8K, 200 pinned test questions, 193 / 200 (96.5 %; Wilson 95 %: 93.0–98.3) without the draft at 8 requests,
193 / 200 with it at 8 and 193 / 200 with it at 32, no answer truncated - 3.9 s per question without the draft, 1.4 s
with it, and the 200 questions in 19 s at 32 requests. In the pair of the two precisions (no draft, 8 requests, an 8K
context): 195 / 200 as released (97.5 %; 94.3–98.9) and 194 / 200 packed (97.0 %; 93.6–98.6) - the same answer to 197
of the 200, two that only the released precision gets right and one that only the packed model does; of
`quality_cmp.py`'s 24 greedy answers 19 are the same text at both precisions (two passes through the packed server: 24
of 24). A needle at depths 0.2 and 0.7 of 28.8K and 57.6K tokens: 4 of 4 (packed, in the server with a draft
schedule); arithmetic 5 / 5 with thinking off and 5 / 5 with it on, 12 raw completions without a loop, a tool call
with its round trip, a strict JSON schema and an image request pass in that server and in both servers of the pair.

## Where the decode step goes

vLLM's torch profiler on rank 0 (`tools/sm120_perf/trace_agg.py`), no draft; the all-reduces of this one start ran
on the P2P push, every other kernel is the default path's. One request - 7.2 ms a step under the profiler, the GPU
busy 98 % of it:

| | launches per step | each | per step |
|---|---|---|---|
| all-reduces (two per layer and the embedding's) | 97 | 18.2 µs on the push, about 14 µs on PyNCCL | 1.4–1.8 ms |
| expert GEMMs (gate / up and down per layer) | 96 | 13.9 µs NVFP4, 21.0 µs e2m3 | 1.43 ms |
| attention output projection (cuBLAS, 4608 × 1536 on a rank) | 48 | 19.6 µs | 0.94 ms |
| q / k / v projection (cuBLAS) | 48 | 10.4 µs | 0.49 ms |
| AttnRes mix, short convolution | 96 + 96 | 4.9 µs, 2.5 µs | 0.71 ms |
| router | 48 | 7.3 µs | 0.35 ms |
| attention kernel, combine, prologue | 48 each | 4.9 µs, 2.4 µs, 3.6 µs | 0.52 ms |
| LM head (a rank's vocabulary shard) | 1 | 257 µs | 0.26 ms |

At eight requests (13.2 ms a step) the expert GEMMs are 6.0 ms - 59 µs a launch in an NVFP4 layer, 80 µs in an e2m3
layer: the weights of the experts eight tokens pick, read at the memory's rate - and the all-reduces 2.8 ms, most of
it the reduce kernel in which a rank waits for the slowest. The large launches are reads of weights at the rate the
card's memory delivers; what is left at one request is latency: a fifth of the step in all-reduces of one 9 KB row
(PyNCCL's ring and the P2P push both sit at 14–18 µs on four PCIe cards; vLLM fuses them into the next kernel over
NVLink multicast) and a quarter in a dozen 2–8 µs kernels per layer.

A prefill chunk of 1024 tokens (a fresh 17.4K-token prompt under the profiler): the PyNCCL all-reduces of
1024 × 4608 bf16 take a third of the GPU's time (398 µs each), the expert GEMMs 19 % (every expert of a rank is read
once per chunk), the attention kernel 8 % (41 µs a launch).

## Why the base image does not run on sm_120

vLLM's Inkling2 (`vllm/models/inkling2/`) is written for sm_100: every heavy operation of the model is a kernel
that capability 12 does not have, and each fails at a different moment of a start.

| operation | vLLM's kernel | what happens on capability 12 | step |
|---|---|---|---|
| expert GEMMs of the MoE (NVFP4 and MXFP8, 384 experts) | FlashInfer's TRT-LLM-gen batched GEMM: cubins for sm_100 | no kernel for the device: the model does not load | 1 |
| attention of decode, verify and the DSpark draft | `_rel_verify_kernel`, Triton Gluon: tcgen05 MMA into tensor memory, TMA | the instructions do not exist | 2 |
| attention of prefill | tml_fa4's sheared-bias forward (sm_100); the fallback for sm_103 is upstream cute FA4 with a score_mod bias | the fallback's forward for capability 12 takes the page table and does not read it: a wrong result without an error | 2 |
| linears of up to 32 rows | FlashInfer's CuTe DSL low-M kernels (capabilities 100 / 103 / 107) | the call raises in the first decode step | 3 |
| the router's K-group partials from 2048 rows | a Triton TMA kernel that stages 197 KB of shared memory | capability 12 has 99 KB: the launch fails in the profile run | 3 |
| the vision tower's attention | tml_fa4's `mma.sync` forward for capability 12 | its constructor turns on a TMA output path the kernel has no atom for: the compile fails at start-up | 4 |
| the embedding of a TP group | one kernel of vLLM's compiled library | the library has no image of it for capability 12; the unchecked launch surfaces at the next torch operation | 5 |
| the draft's argmax over the vocabulary shards | a kernel of the same library over multicast Lamport buffers (NVLink) | asserts multicast with more than one rank; no image for one rank | 6 |
| the residual stream's fused all-reduce | multicast Lamport buffers (NVLink), capability 10 | vLLM itself falls back to PyNCCL | 7 |
| the draft's K / V write into the paged cache | `reshape_and_cache_flash` of the same library | no image for capability 12: "no kernel image is available" in the draft's first forward (the target writes its cache in its own prologue and never calls it) | 8 |

How each row is known. The first row was seen: served with every runtime switch on vLLM's own path, a six-layer
cut of the checkpoint ends its start while the first layer's experts are built - FlashInfer compiles its
`fused_moe_trtllm_sm100` module for some minutes and raises `No supported CUDA architectures found for major
versions [10]`; no weight is loaded. That start does not reach the attention kernels, the small-batch linear, the
argmax and the all-reduce (steps 2, 6, 7 and the first row of step 3): those rows are read from vLLM's code, and
the official image itself was not started on these cards. The others are failures seen on a card with the steps
before them in place: the router's shared memory and the vision forward on one card with the six-layer cut
(`make_mini_checkpoint.py`), the embedding with the same cut on four cards, the draft's K / V write by calling the
operation alone in a process (a launch without a kernel image leaves a sticky CUDA error). The record has each.

## The expert GEMM (step 1)

One launch computes `out[r] = bf16(row_scale[r] * act[r] @ weight[expert(r)]^T)` for the routed rows of every local
expert: activations and weights block-scaled in the checkpoint's own formats, accumulated in fp32.

- **Instructions.** `mma.sync` with block scaling: `kind::mxf4nvf4` `scale_vec::4X` m16n8k64 (e2m1 × e2m1, a ue4m3
  scale per 16 elements) for NVFP4 and `kind::mxf8f6f4` `scale_vec::1X` m16n8k32 (e4m3 × e4m3, a ue8m0 scale per 32)
  for MXFP8; the same instruction with an e2m3 B operand for the packed experts. The operand layouts are established
  on the card (`test_qmma_layout_sm120.py`: one instruction with every lane's registers given by the host, 2560 of
  2560 outputs equal to exact arithmetic), not taken from a header.
- **Shape.** A persistent kernel: a grid of SMs × CTAs per SM takes work items (a row chunk of one expert's tile × a
  block of output columns) from a counter until none is left, and the last CTA out resets the counter, so the launch
  replays from a CUDA graph. K runs in steps of 64 elements through a ring of cp.async stages; the weights are read
  as one sequential stream per item from a step-major layout prepared at load (a byte permutation), prefetched into
  L2 ahead of the ring.
- **Measured** on one card: a decode launch reads its experts at 1.71–1.92 TB/s (the card's memory delivers about
  1.8): 16 experts × 8 rows of the NVFP4 gate / up projection in 106.5 µs, of the down projection in 55.3 µs; of an
  e2m3 gate / up projection in 147.5 µs against 188.4 µs as MXFP8. A prefill launch of 32 tiles × 128 rows of the
  gate / up projection takes 325.6 µs in NVFP4 (534 TFLOP/s), 641 µs in MXFP8 and 790 µs in e2m3: the six bits are
  unpacked per element, which a decode launch does not feel and a prefill launch does.
- **Test.** `test_bmm_sm120.py`: 53 cases - every format and tile against exact arithmetic on synthetic operands
  and on the checkpoint's own expert weights; an element passes when it is the nearest bf16 of the exact value or
  within the fp32 accumulation bound, at least 99.5 % must be the nearest; padding untouched; the result independent
  of the grid and of the run.

## The relative attention (step 2)

`logits[t, h, k] = scale * (q[t, h] . K[k, h / G]) + bias[t, h, pos(t) - k]` over the keys visible to token t (not
after its position; in a window layer less than 1024 positions before it), `out = softmax(logits) @ V`, and the
natural-log sum for the attention sink. K and V come from the paged cache; `bias` from the sheared table the
attention prologue writes.

- **One kernel for decode, verify, the draft and prefill.** A CTA holds 64 query rows - up to 64 / G tokens of one
  request times the G heads of a KV head - in four warps of 16 rows, with Q in registers as MMA A fragments, and walks
  the keys in steps of 64: K | V rows of the paged cache through a two-stage cp.async ring (a page-table lookup per
  row, 16-byte chunks XOR-swizzled by the row so that `ldmatrix` reads eight rows without bank conflicts), S = Q K^T
  on `mma.sync` m16n8k16, the online softmax in the log2 domain, P rounded to bf16 as the A fragments of O += P V
  (the C fragment of S has the A fragment's layout). 65 536 B of shared memory, 255 registers, no stack frame.
- **Decode** keeps vLLM's split plan: grid (splits, KV heads, requests), split s takes the 128-key blocks s,
  s + splits, ...; one split writes the gated output itself, more write fp32 partials that a combine kernel merges
  and gates. **Prefill** is a grid over 64 / G consecutive tokens of the batch; a CTA walks the keys once per request
  it meets among its tokens. Nothing reads a length on the host: the grid depends on the batch shape only, so the
  decode launch is captured in vLLM's FULL CUDA graphs.
- **Measured.** In a served decode step the kernel takes 5.6 µs and the combine 2.3 µs per layer on one card with all
  48 heads, 4.9 µs and 2.4 µs on a rank of four; a prefill of 8192 tokens takes 524 µs in a window layer and 1720 µs
  in a full layer on one rank's 12 heads, a 1024-token chunk of a served prefill 41 µs.
- **Test.** `test_attn_sm120.py`: 44 cases through both entries against the operation's definition in float64 - full
  and window layers, 1 to 94 splits, padding requests, the draft mode, multi-request prefill with and without
  context, page sizes 16 / 32 / 128, GQA groups 6 and 4. Everything the kernel must not read is NaN: cache slots past
  a request's last key, a poison page behind every page-table entry past a request's pages, the table's columns of
  invisible keys. The softmax weights are rounded to bf16 before P @ V, as in vLLM's kernels, so the criterion is
  `|out - ref| <= 2^-8 max|V| + 2^-8 |ref|` per element and 2e-3 on the LSE; two runs must agree bit for bit.

## The fit: experts of layers 40–47 as e2m3

Per rank at TP4 / EP4 (96 of the 384 experts of every layer on a card):

| | bits per weight | GiB per rank |
|---|---|---|
| NVFP4 experts, layers 0–39 | 4.5 (e2m1 + a scale per 16) | 64.1 |
| MXFP8 experts, layers 40–47, as released | 8.25 (e4m3 + a scale per 32) | 23.5 |
| MXFP8 experts, layers 40–47, as e2m3 | 6.25 | 17.8 |
| attention, router, embeddings, LM head, vision tower | 16 (bf16) | 3.3 |

On a card: 85.86 GiB after the load with the experts packed (the table's 85.2 GiB and the allocator's rounding),
86.9 GiB with the draft, 91.55 GiB as released - of 95.0.

`pack_e2m3`: per block of 32 weights the ue8m0 scale moves by a power of two so that the block's largest magnitude
lies in e2m3's top binade [4, 8). A weight in the block's top three binades keeps all three mantissa bits - it is
the same number; a smaller one is rounded to nearest even on e2m3's subnormal grid, and may become zero. Measured
on 96 experts (12 of each of the layers 40–47, both projections of each; `bmm/test_pack_e2m3.py --census 12`, CPU):
76.2–76.4 % of the gate / up weights and 77.5–78.6 % of the down weights keep their value (65.7–66.0 % and
67.5–69.1 % lie in the three binades), the error energy is −38.4 to −38.3 dB and −39.4 to −38.7 dB of the weight
energy, the largest error 1 / 64 of the block's largest magnitude. The kernel reads the 6-bit
stream directly (three 32-bit words per 16 values, unpacked into byte containers for the e2m3 operand of the same
MMA). The experts are registered in that form from the start and packed two at a time with the allocator's cache
emptied after each pair, so a rank never holds the e4m3 tensor: the packing's temporaries are 40 bytes per value,
and a card has no room for more of them beside 86 GiB of weights.

## What is left of a card: the paged cache

What vLLM measures after warm-up on a card, and the paged cache it sizes from it (the free memory less a margin
of 1 GiB or three quarters of the transient, less the transient of a prefill chunk):

| | packed, the draft | packed, no draft | packed, no draft, 8K | as released, no draft, 8K |
|---|---|---|---|---|
| configuration: context, prefill chunk, sequences | 65 536, 512, 32 | 65 536, 1024, 8 | 8 192, 64, 8 | 8 192, 64, 8 |
| weights and what is outside torch | 87.1 GiB | 86.0 GiB | 86.0 GiB | 91.7 GiB |
| CUDA graphs | 1.39 GiB | 0.15 GiB | 0.15 GiB | 0.15 GiB |
| free after warm-up | 5.2 GiB | 7.67 GiB | 7.67 GiB | 1.98 GiB |
| kept for a chunk's transients | 0.9 GiB | 1.74 GiB | 0.3 GiB | 0.3 GiB |
| the paged cache | 3.30 GiB | 4.64 GiB | 6.38 GiB | 0.69 GiB |
| requests of the configured context that fit, by vLLM's count | 1.40 | 1.28 | 20.5 | 2.21 |

What a request holds there, per rank - computed from vLLM's block sizes, not measured:

| | per token | kept | a request holds |
|---|---|---|---|
| conv state of the 48 layers (K, V and both residual streams, the short convolutions' taps) | 32 KiB a layer, 1.5 MiB a token | the last 3 tokens, in blocks of four - and every token of a chunk in flight | up to two blocks, 12 MiB, at rest |
| K / V of the 40 window layers | 1 KiB a layer | the last 1024 tokens, in blocks of 128 | up to nine blocks, 45 MiB |
| K / V of the 8 full-attention layers | 1 KiB a layer | every token | 8 MiB per 1000 tokens |

By this count a long context is cheap - eight requests of 65K tokens hold 4.5 GiB - and a prefill chunk is not:
vLLM reserves the conv state of two chunks in flight for a request of the maximum length (3 MiB per token of the
chunk) and refuses to start when the pool is smaller than one such request. Without the draft
`--max-num-batched-tokens 1024` fits beside `--max-model-len 65536` (3.6 GiB for the one request by vLLM's count;
4096 would need 12.9 GiB); with the draft 512 does (2.4 GiB of 3.3; a start with 768 was refused: 3.22 GiB needed,
2.83 GiB there). The chunk size costs the prefill little: 8.6–8.7K tok/s at 512 against 8.8K at 1024. At the
released precision 0.69 GiB are left - against the 2.4 GiB one request of 65 536 tokens reserves with the
launcher's 512-token chunks - and the draft's weights and graphs (2.3 GiB more than a server without it) exceed the
1.98 GiB free after warm-up. A start at the released precision with a longer context than 8 192 tokens or with the
draft was not tried.

The draft takes its room twice: 1.0 GiB of weights and 1.4 GiB of CUDA graphs - a draft graph holds logits over the
vocabulary per request and Markov step, and vLLM captures one per batch size up to `max_num_seqs` (0.19 GiB of graphs
without the draft at 32 sequences). vLLM's schedule of the draft length by batch size does not change that: the
target's graphs follow the schedule, the draft's do not (1.41 GiB with a schedule that drafts up to 8 of 32 requests,
1.39 GiB without one). Step 9 captures the draft's graphs up to the largest batch size that drafts: on the six-layer
cut with a schedule up to 2 of 8 requests, two draft graphs instead of eight; on the four cards it is not measured -
the default drafts at every batch size, where the step changes nothing.

## Loading

The target model is 363.3 GiB of the checkpoint's 366.5 (the draft is the other 3.2 GiB), and its experts are 350 GiB
of that. A rank registers its experts in the kernel's layout (and the MXFP8 layers in six bits) from the start and
loads them a few at a time - a byte permutation on the card, the 6-bit packing two experts at a time with the
allocator's cache emptied after each chunk - so a card never holds a second copy of a tensor. Each chunk's file pages
are released once the chunk is on the card (`madvise` and `posix_fadvise(DONTNEED)` on the range just read;
`VLLM_MOET_INKLING2_DROP_PAGES=0` turns it off): the server's host memory stays at 23 GiB through a load instead of
filling its limit with page cache within a minute, and a load leaves 13 GiB of the checkpoint cached instead of 82.
The weights load in 113 s per rank; a start, 177 s to `/health` with warm kernel caches (the model's Triton kernels
compile per batch shape at first use - the launcher keeps Triton's cache in `CACHE_DIR`; the first start takes about a
minute longer).

## Open items

- **Prefill** is 8.6–10.4K tok/s. A third of a chunk is the all-reduce of 9.4 MB per sublayer on PyNCCL's ring over
  PCIe; the P2P push of the other lines reproduces NCCL's order per row count from tables measured at start-up and is
  built here for up to 128 rows - a prefill chunk's 512 or 1024 rows are not covered. vLLM's own answer, a residual
  stream sharded by tokens between ranks, is written for NVLink multicast.
- **The decode step at one request** is latency: 97 all-reduces of a 9 KB row (a fifth) and a dozen 2–8 µs kernels
  per layer (a quarter). vLLM fuses the all-reduce, the short convolution, the AttnRes mix and the norm into one
  kernel of its compiled library over multicast buffers; a kernel of that fusion over NCCL's result is the next step
  for sm_120 and would have to return the unfused path's bits.
- **The draft** is on at every batch size: prose pays up to 12 % for it from eight requests up (5–12 % at eight,
  within 3 % at 16, 6–10 % at 32), code gains 2.3–2.8×.
  vLLM's verification by the draft's own confidence is not implemented for this draft, so the only finer control is
  the draft length by batch size, not by text; step 9's effect on the paged cache under such a schedule is measured
  on one card only.
- **The all-reduce of larger batches.** The P2P push is slower than PyNCCL at one and eight rows and, in the one
  comparison there is, 3–11 % faster at 32; the draft's verification all-reduces 64 to 256 rows at 8 to 32 requests.
  The push with a lower bound of rows is not measured.
- **The released precision** runs on the four cards with what 0.69 GiB of paged cache allows - an 8K context,
  64-token chunks, no draft (measured). A longer context there, or the draft, was not tried, so nothing compares the
  two precisions on long prompts.
- **The context** is bound by the paged cache, not by the model: 65 536 tokens is what one request may reserve with a
  512-token chunk in flight; the checkpoint allows 1M.
