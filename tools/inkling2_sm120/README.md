# tools/inkling2_sm120 — Inkling-2-Small on sm_120

The Inkling-2 line's kernels, patchers and checks, by Dockerfile step (`Dockerfile.sm120-inkling2`; what each step
does, the switches and the measurements: `docs/inkling2-sm120.md`). Shared with the other lines:
`tools/dsv41_sm120/p2p_ar/` and `tools/ripple_sm120/patch_vllm_p2p_allreduce_tp_sm120.py` (step 7),
`tools/sm120_perf/` (the measurement tools).

vLLM's Inkling2 (`vllm/models/inkling2/` in the base image) runs on kernels written for sm_100. Steps 1–4 replace
the four that capability 12 cannot run (steps 5, 6 and 8 take three operations of vLLM's compiled library, which has no
sm_120 image, off the model's paths; step 7 is the all-reduce module shared with the other lines; step 9 gives the
draft's unused CUDA graphs back to the paged cache); each replacement stands behind the entry vLLM already calls, takes the same
operands and has an op-level test against the operation's definition in exact arithmetic, with everything the
kernel must not read poisoned with NaN.

| file | step | what |
|---|---|---|
| `bmm/inkling2_bmm_sm120.cu`, `.py` | 1 | the MoE's batched expert GEMM: block-scaled activations × block-scaled expert weights → bf16 on `mma.sync` QMMAs (`kind::mxf4nvf4` `scale_vec::4X` m16n8k64 for NVFP4, `kind::mxf8f6f4` `scale_vec::1X` m16n8k32 for MXFP8 and for MXFP8 activations × e2m3 weights). A persistent kernel per format and tile (CTAs take work items from a counter; replays from a CUDA graph), K in cp.async stages of 64 elements, weights in a step-major layout prepared at load, L2 prefetch ahead of the ring. `pack_e2m3`: MXFP8 expert weights → e2m3, six bits per value |
| `bmm/test_qmma_layout_sm120.py` | 1 | the operand layouts of the three QMMAs established on the card: one instruction with every lane's registers given by the host, against exact arithmetic |
| `bmm/test_bmm_sm120.py` | 1 | the GEMM against exact arithmetic on synthetic operands and on the checkpoint's own expert weights (`--model-dir`), every tile, grid and run independence, untouched padding; `--bench`: decode and prefill shapes with the bytes read per second |
| `bmm/test_drop_file_pages.py` | 1 | `drop_file_pages` on CPU: a slice of a memory-mapped checkpoint tensor is in the page cache after it is read and gone after it is dropped, its neighbour keeps its pages (the expert loader releases every chunk's file pages once it is on the card: a load leaves 13 GiB of the checkpoint in the page cache instead of 82) |
| `bmm/test_pack_e2m3.py` | 1 | the 6-bit packing on CPU: round trip, which weights are exact, the error bound, the checkpoint's weights; `--census N`: the share of exact weights and the error energy over N experts of each of the layers 40–47, both projections |
| `patch_vllm_inkling2_bmm_sm120.py` | 1 | anchored patcher on `ops/trtllm_bmm.py` (`expert_plan`, `load_kernels`, `bmm`), `ops/block_scales.py` (row-major activation scales) and `moe.py` (expert weights into the prepared layout at load; `VLLM_MOET_INKLING2_MXFP6=1` packs the MXFP8 experts) |
| `attn/inkling2_attn_sm120.cu`, `.py` | 2 | the relative attention: decode, verify, the DSpark draft's block attention and prefill in one `mma.sync` m16n8k16 kernel (64 query rows = tokens × GQA group per CTA, steps of 64 keys from the paged cache through a two-stage cp.async ring, Q in registers, `ldmatrix` on XOR-swizzled K / V, the sheared relative logits of the attention prologue, online softmax in the log2 domain, split-KV partials and a combine kernel, the gate and the sink). `reference`: the operation in torch |
| `attn/test_attn_sm120.py` | 2 | the kernel through both entries against the float64 definition: full and window layers, 1 … 94 splits, padding requests, the draft mode, multi-request prefill with and without context, page sizes 16 / 32 / 128, GQA groups 6 and 4; `--bench` |
| `patch_vllm_inkling2_attn_sm120.py` | 2 | anchored patcher on `ops/attention.py`: `_decode_attention` and `Inkling2FA4RelAttentionKernel.kernel` take the module on capability 12 (`VLLM_MOET_INKLING2_ATTN=kernel\|ref\|0`) |
| `linear/inkling2_linear_sm120.cu`, `.py` | 3 | the small-batch bf16 linear (`x @ weight.T`, up to 32 rows, fp32 accumulation): one CTA per 32 output columns over all of K, cp.async stages of 64 elements of the row-major weight rows, `ldmatrix` on swizzled rows, L2 prefetch |
| `linear/test_linear_sm120.py` | 3 | the kernel against exact arithmetic on the model's shapes and odd ones, operands as slices of NaN tensors; `--bench`: the kernel and cuBLAS, warm and with the weight out of L2 |
| `patch_vllm_inkling2_linear_sm120.py` | 3 | anchored patcher on `ops/linear.py`: `linear` takes the module for batches of up to 32 rows on capability 12; `k_group_partials` keeps its `mma.sync` Triton kernel for every batch size with the stage count that fits 99 KB of shared memory |
| `patch_vllm_inkling2_vision_fa4_sm120.py` | 4 | anchored patcher on `third_party/tml_fa4/flash_fwd_sm120.py`: the forward class keeps `arch = sm_80` after the base constructor (the fix vLLM's other copy of the kernel carries), so the vision tower's attention compiles on capability 12 |
| `patch_vllm_tp_embedding_sm120.py` | 5 | anchored patcher on `vllm/model_executor/layers/vocab_parallel_embedding.py`: on capability 12 a TP group's embedding takes the unfused path (the fused kernel of vLLM's compiled library has no sm_120 image in the base; its unchecked launch ends the server in the first warm-up step) |
| `patch_vllm_inkling2_dspark_argmax_tp.py` | 6 | anchored patcher on `dspark.py`: without NVLink multicast, and on every capability-12 device (the compiled library has no image of vLLM's argmax kernel for it), the draft's argmax over the vocabulary shards is each rank's first local maximum and an all-gather of (maximum, global index) |
| `patch_vllm_cache_flash_sm120.py` | 8 | anchored patcher on `vllm/_custom_ops.py`: on capability 12 `reshape_and_cache_flash` — the scatter into the paged cache the DSpark draft's layers write their K / V with — runs on vLLM's Triton kernel of the same operation (the kernel of the compiled library has no sm_120 image in the base: a server with the draft ends in the draft's first forward) |
| `test_cache_flash_sm120.py` | 8 | the patched entry on the card against the operation's definition: both cache layouts, 1 … 1024 tokens, a batch longer than the slot mapping, every other byte of a random cache untouched |
| `patch_vllm_draft_graphs_by_schedule.py` | 9 | anchored patcher on `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`: with a draft length by batch size the draft's FULL CUDA graphs are captured only up to the largest batch size that drafts (the query pass never runs above it; the target's graphs already follow the schedule) — the memory of the larger graphs, logits over the vocabulary per request and Markov step, goes back to the paged cache |
| `test_draft_graphs_by_schedule.py` | 9 | the capture bound against schedules, without a card |
| `make_mini_checkpoint.py` | — | cuts a checkpoint of a few layers out of the released one (tensors copied byte for byte, layers and AttnRes queries renumbered): every kernel path of the model on one card; `--draft` attaches the DSpark draft to a cut that keeps one of its target layers (`fuse_weight` cut to the kept features), which drives the draft's whole path — load, K / V write, block attention, Markov head, argmax, verification, rejection — on one card |

Run a test inside the image on a capability-12 card, e.g.
`python3 /opt/vllm-moet/inkling2_sm120/attn/test_attn_sm120.py --bench`.

What the tests do not cover: steps 5 and 6 (the embedding path, the argmax all-gather) and the expert loader as a
whole (the order of weight and scale tensors, the scales that wait for their weights) are held by the served runs
only; the attention kernel is tested against this directory's own float64 reading of the operation, since none of
vLLM's kernels of it runs on these cards; the linear test covers the `mma.sync` kernel, which is off by default.
`VLLM_MOET_INKLING2_BMM=0`, `VLLM_MOET_INKLING2_ATTN=0` and `VLLM_MOET_INKLING2_LINEAR=0` hand the operation back
to vLLM's own kernels: they are for other architectures, on capability 12 those kernels do not run.

Checks the code does not make yet (none is reached by the model's shapes): the layout of the activations' block
scales is decided when `ops/block_scales.py` is imported, by its own capability query that reads any exception as
"not sm_120", apart from the kernel choice made at the first call; the loader clamps a repacked block's scale byte to
0–254 without counting the clamps; the prefill entry of the attention does not assert that the relative table's
width is a multiple of 128 keys as the decode entry does; the native entries of the three extensions rely on the
Python wrappers' assertions for dtypes and sizes, and `bmm_run` takes K values whose scale rows are not a multiple
of eight bytes.
