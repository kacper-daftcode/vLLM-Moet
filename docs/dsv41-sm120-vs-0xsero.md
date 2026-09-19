# DeepSeek‑V4.1‑Flash on 4× RTX PRO 6000: vLLM‑Moet vs 0xSero's SGLang recipe

Two independent ways to serve the **same official checkpoint** on four RTX PRO 6000 Blackwell
cards, measured **head‑to‑head on one host, on the same day, with the same client, the same
prompts and the same token ids**:

| | vLLM‑Moet (this repo) | [0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000) @ `45f538a` |
|---|---|---|
| engine | official `vllm/vllm-openai:deepseekv41-flash-0909` + this repo's sm_120 kernels (`Dockerfile.sm120-dsv41`) | pinned `lmsysorg/sglang@sha256:c4ca6511…` + adapter (`docker compose`, `compose.measured.yaml`) |
| parallelism | TP4 | TP4 / EP4 |
| attention on sm_120 | FlashInfer sparse‑MLA CUDA kernels, instantiated for the V4.1 page geometry (this port) | Triton `flash_mla_sm120` decode kernel; prefill sources re‑paged to 64‑token pages |
| routed experts | DeepGEMM FP8×FP4 grouped GEMM (SM120), glue kernels patched here | FlashInfer SM120 CUTLASS W4A8 MoE |
| dense decode GEMMs | this repo's MXFP8 tensor‑core GEMV | FlashInfer / CUTLASS |
| Engram (196B n‑gram table, 189 GiB FP8) | whole table pinned in host RAM (vLLM offload) | `nvme`: 64 GiB DDR5 row cache + exact NVMe reads on miss (**his default, for 128 GB hosts**); `ram`: whole table `mlock`ed (his optional mode) |
| speculative decoding | DSpark k=5 | DSpark block 5 |
| context / KV | 512K, `--gpu-memory-utilization 0.94`, fp8 KV, **1.49M‑token pool** | 524288, `MEMORY_FRACTION 0.95`, **4.2M tokens allocated** (`MAX_TOTAL_TOKENS`) |
| weights on GPU | 81 GiB/card | 72.6 GiB/card |
| vision | on | on |

Host: 8× RTX PRO 6000 Blackwell Server Edition (96 GB, sm_120) in a KVM guest, PCIe without NVLink,
600 W power limit, 708 GB RAM, model on an XFS **virtio disk (no bare NVMe)**. Each stack ran on
its own four cards (ours on GPUs 4–7 as deployed, his on GPUs 0–3 with `--gpus device=0,1,2,3`);
every measurement was taken with the other stack idle. His image was built from his repository as
published and launched with the settings of his `compose.measured.yaml`; his launcher's checkpoint
verification, smoke tests (arithmetic, JSON schema, tool round trip, 1024‑token vision) all passed.
2026‑09‑19.

## Quality: parity

Same weights, so the question is whether either kernel path damages the model. Neither does, within
what these probes can resolve (`tools/sm120_perf/quality_cmp.py`, `compare_outputs.py`; GSM8K via
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `--test-profile gsm8k`,
200 items, greedy, thinking off, C4, `--compare-baseline` for the paired exact McNemar test):

| probe | vLLM‑Moet | 0xSero |
|---|---|---|
| arithmetic, 5 multi‑step products/sums (thinking off / on) | 5/5, 5/5 | 5/5, 5/5 |
| coherence, 12 raw completions, loop detector | 0 degenerate | 0 degenerate |
| tool round trip (`lookup_fixture`), strict JSON schema | pass, pass | pass, pass |
| vision: red circle + blue square, 3024×588 | pass | pass |
| needle, 6‑digit code at depth 0.2 / 0.7 in 26.7K, 97.4K and **367.5K‑token** prompts | 6/6 | 6/6 |
| **GSM8K‑200**, greedy, thinking off | **96.5 %** (193/200), run 2: 96.5 % | **96.5 %** (`nvme`), **97.5 %** (`ram`) |
| paired McNemar vs 0xSero `nvme` run | Δ 0.0 pp, p = 1, flips 1 / 1 | — |
| completion tokens, mean | 158–159 | 158–159 |

The self‑comparison sets the noise floor: our two runs flip 1 item each way against each other, his
`nvme` and `ram` runs (identical weights and engine) differ by 2 items. Every accuracy delta above is
inside that floor. Greedy 192‑token answers to 24 chat prompts are byte‑identical between the two
stacks in 8/24; the rest diverge after 3–96 tokens into equally valid continuations (different
kernels, different rounding — neither stack is the reference).

One point in favour of the SGLang stack: **image tokenization**. The checkpoint's
`inference/image_processor.py` gives a 3024×588 image exactly 1024 tokens (14×72 grid); SGLang
reports 1024, the vLLM image produces **886** (13×67 — it resizes an image that already fits the
budget). On five other sizes (1024², 640×480, 1920×1080, 300×200, 4000×3000) both match the
reference exactly (`tools/sm120_perf/image_tokens_check.py`). This is an upstream vLLM processor
deviation at the token‑budget boundary, present in the official image, not introduced by this port.

## Performance

### Real prompts, single stream (`tools/sm120_perf/decode_bench.py`, greedy, thinking off, 512 output tokens)

| | decode steps/s | prose tok/s | code tok/s | tok/step prose / code (DSpark) |
|---|---|---|---|---|
| **vLLM‑Moet** | **66.7–67.0** | **148.5** | **346.4** | 2.23 / 5.17 |
| 0xSero `ram` | 37.4–40.2 | 84–89 | 206 | 2.10–2.22 / 5.12 |
| 0xSero `nvme`, first time the text is seen | 12.9–14.4 | 29 | 72 | 2.27 / 5.02 |
| 0xSero `nvme`, same prompt repeated (row cache warm) | 13.7–29.0 | 29 | 149 | 2.14 / 5.12 |

Tokens per step are the same on both stacks — DSpark accepts the same drafts, i.e. the target model
behaves the same. The difference is the step: **15 ms vs 25 ms** with the Engram table in RAM on
both sides. In his default `nvme` mode a decode step stalls on the row cache misses of every
new n‑gram: 29 tok/s on prose, and a **14.4K‑token prefill of unseen text took 65.6 s (220 tok/s)**
against 1.5 s (9.8k tok/s) in `ram` mode; the second run of the identical prompt took 0.12 s
(prefix cache). His published prefill and decode figures use repeated filler text, whose few
distinct n‑grams stay cache‑resident.

### 0xSero's burst methodology, same client for both (`tools/sm120_perf/openai_matrix.py`)

Token‑id prompts built exactly as his `benchmarks/matrix.py` does (chat prefix with a unique nonce,
repeated reference‑notes filler, the LRU‑cache instruction, `</think>` forced), 1024 forced output
tokens (`ignore_eos`), one wave per cell, his metric definitions: prefill = input tokens through the
last first token including queueing; **total decode** = tokens delivered by all streams inside the
shared window [last first token, first finish]. Engram in RAM on both sides:

| input | C | prefill tok/s Moet / 0xSero | **total decode tok/s Moet / 0xSero** | per request Moet / 0xSero | TTFT s Moet / 0xSero |
|---:|---:|---|---|---|---|
| 512 | 1 | 4 288 / 4 032 | **288 / 173** | 288 / 173 | 0.12 / 0.13 |
| 512 | 4 | 7 226 / 7 729 | **707 / 538** | 177 / 135 | 0.28 / 0.26 |
| 512 | 8 | 10 004 / 7 477 | **1 030 / 790** | 130 / 98 | 0.23 / 0.40 |
| 2 048 | 1 | 10 708 / 8 734 | **282 / 171** | 282 / 171 | 0.19 / 0.23 |
| 2 048 | 4 | 11 443 / 10 292 | **691 / 572** | 172 / 142 | 0.68 / 0.68 |
| 2 048 | 8 | 11 313 / 10 298 | **1 042 / 809** | 130 / 102 | 1.04 / 1.05 |
| 8 192 | 1 | 11 755 / 10 571 | **301 / 186** | 301 / 186 | 0.70 / 0.77 |
| 8 192 | 4 | 11 852 / 11 047 | **697 / 538** | 174 / 134 | 2.03 / 2.03 |
| 8 192 | 8 | 11 918 / 11 059 | **1 023 / 823** | 127 / 103 | 3.38 / 3.49 |
| 32 768 | 1 | 11 923 / 11 036 | **272 / 178** | 272 / 178 | 2.75 / 2.97 |
| 32 768 | 4 | 11 933 / 11 171 | **697 / 562** | 177 / 140 | 7.14 / 7.51 |
| 32 768 | 8 | 11 832 / 11 160 | **1 010 / 794** | 126 / 98 | 12.64 / 13.38 |
| 131 072 | 1 | 11 015 / 10 580 | **271 / 176** | 271 / 176 | 11.90 / 12.39 |
| 131 072 | 4 | 10 966 / 10 656 | **703 / 502** | 176 / 128 | 30.04 / 30.96 |
| 131 072 | 8 | 10 855 / 10 647 | 955† / 761 | 132 / 95 | 53.95 / 55.68 |
| 32 768, 4096 out | 8 | 11 702 / 10 894 | **1 064 / 841** | 134 / 105 | 12.88 / 13.96 |
| 131 072, 4096 out | 8 | 10 862 / 10 640 | **1 060 / 831** | 134 / 104 | 53.88 / 55.67 |

† 0.6 s shared window (the first stream had almost finished its 1024 tokens when the eighth
prefill ended) — the 4096‑output rows below it are the reliable C8 long‑context figures.

Single stream **+53–66 %**, C4 **+21–40 %**, C8 **+24–30 %**; prefill within ±10 % (both ≈ 11k tok/s
above 2K tokens; his chunked prefill is 2048, ours 4096). DSpark on this synthetic code workload:
3.06–3.48 accepted tokens per step on our side (vLLM metrics), his 78–88 % acceptance per his
tables — the same regime.

His default `nvme` mode on the same host, same matrix: first pass 64 → 144 tok/s single stream and
213 → 442 tok/s at C8 as the row cache warms across the 12 waves; a second pass 154–167 (C1) and
386–389 (C8). His published table (200–232 C1, 700–750 C8, 275 W, 8192‑token outputs) was taken on
his own hardware with a bare NVMe and a fully warm cache; our virtio disk penalizes the miss path,
so the `ram` rows above are the kernel‑level comparison.

### Real mixed workload, 4 concurrent streams (GSM8K‑200, C4, greedy, thinking off)

| | aggregate generation tok/s | per request | mean elapsed per item |
|---|---|---|---|
| **vLLM‑Moet** | **135** (143 in a first run that overlapped his RAM‑mode weight load) | 138 | 1.4 s |
| 0xSero `ram` | 113 | 115 | 1.5 s |
| 0xSero `nvme` | 19 | 19.5 | 9.3 s |

Short, diverse prompts and short answers: this is the case where the Engram row cache misses most.

## Where the difference comes from

Both stacks run the same DSpark (identical tokens per step), the same prefill throughput and the
same routed‑expert arithmetic (FP8 activations × FP4 weights), so the gap is the decode step of the
target model: 15 ms here against 25 ms. On this side that step is FlashInfer's sparse‑MLA CUDA
kernels compiled for the V4.1 page geometry (rather than a Triton decode kernel and re‑paging), the
~250 dense decode GEMMs on a tensor‑core MXFP8 GEMV, the grouped `wo_a` kept in MXFP8, NCCL over
PCIe P2P, and the DeepGEMM MoE glue kernels — all described in
[dsv41-sm120-port.md](dsv41-sm120-port.md) and validated bit‑exact or to one bf16 rounding against
the kernels they replace.

What his recipe does better: it serves the model on a **128 GB‑RAM host** at all (the bounded
DDR5 row cache with exact NVMe misses — vLLM's offload needs ~190 GiB pinned), it keeps
**4.2M KV tokens** on the same four cards (EP4 leaves 72.6 GiB of weights per card against our
81 GiB; our pool is 1.49M at 512K), and its image tokenization matches the checkpoint's reference
exactly. Its `ram` mode on a large‑memory host is a sound baseline at 40 steps/s.

## What was worth taking from the other stack (checked 2026-09-19)

- **Its MoE kernel — no.** vLLM can run the same FlashInfer CUTLASS W4A8 fused MoE with
  `--moe-backend flashinfer_cutlass_afp8`, and the B12x sm_120 kernels with `--moe-backend b12x`.
  Per layer at the served rank shape on an RTX 5090 (`tools/sm120_perf/moe_backends_bench.py`):
  DeepGEMM chain (served) 99 µs vs FlashInfer CUTLASS 207 µs vs B12x 157 µs at 6 tokens/step; 593 vs
  1000 vs 716 µs at 48 (C8). B12x wins only at 1 token/step (26 vs 43 µs), i.e. without DSpark.
  B12x's dense MXFP8 GEMM is 1.4–3× slower than this repo's GEMV at the decode shapes with
  identical numerics (`tools/sm120_perf/dense_b12x_bench.py`). Details: `tools/dsv41_sm120/README.md`.
- **Its KV capacity — not a knob here.** vLLM's accounting per card: 84.4 GiB weights + non-torch,
  1.8 GiB activation peak, 0.4 GiB graphs, 3.1 GiB KV; `expandable_segments` changes nothing (real
  allocations, not fragmentation). The only configuration lever is `GPU_MEM_UTIL=0.95` +
  `MAX_NUM_BATCHED_TOKENS=2048`: 2.21M tokens (+50 %) at −7 % prefill and 716 MiB of margin under
  the heaviest stress we run — documented as an opt-in capacity profile in
  [sm120-deploy.md](sm120-deploy.md), default unchanged. Closing the gap to 4.2M means the EP4
  weight layout (72.6 vs 81 GiB per card), a code-level change.
- **Its image tokenization — an upstream vLLM issue, not a local patch.** vLLM's `safe_resize`
  reserves `COMPRESS_PAD_TO − 1 = 3` tokens of `vision_max_n_token` because its image block carries
  the compressor-alignment pad (0–3 tokens), an even-row pad and a 2-token tail pad *inside* the
  1024-wide bidirectional SWA index rows (`max_image_tokens = vision_max_n_token` in
  `attention.py`, `image_width` in `combine_topk_swa_indices`). The reference processor has no such
  pads, so an image whose reference grid is 1022–1024 tokens (3024×588 → 14×72 = 1024) is shrunk
  to the next grid that fits 1021 (13×67 = 886). The fix belongs upstream: widen the SWA image
  span by the pad allowance instead of shrinking the image. Issue text below; not patched in this
  image because the span width is a compile-time constant of the index kernel.

<details>
<summary>Draft vLLM issue: DeepSeek-V4.1 image processor shrinks budget-boundary images (886 instead of 1024 tokens)</summary>

**Summary.** `vllm/models/deepseek_v4/common/mm_preprocess.py::safe_resize` subtracts
`COMPRESS_PAD_TO - 1` (= 3) from `vision_max_n_token` before checking whether an image fits. The
checkpoint's reference `inference/image_processor.py` checks against the full 1024. Any image whose
reference token grid is 1022–1024 tokens is therefore resized down in vLLM. Example: a 3024×588
PNG → reference 14×72 grid = 1024 tokens (SGLang reports `image_tokens: 1024`); vLLM produces 886
(13×67). Five other sizes (1024², 640×480, 1920×1080, 300×200, 4000×3000) match the reference
exactly (652, 206, 968, 189, 1001). Script: `tools/sm120_perf/image_tokens_check.py` in
vLLM‑Moet.

**Why the reserve exists.** `build_image_block` adds a position-dependent compressor pad
(0–3), an even-row pad and a 2-token tail pad inside the image span, and
`DeepseekV4Attention.max_image_tokens = vision_max_n_token` (also `image_width` in the
`combine_topk_swa_indices` warmup keys) bounds the bidirectional SWA index rows by exactly 1024, so
the block must stay ≤ 1024 including pads.

**Suggested fix.** Let `safe_resize` use the full `vision_max_n_token` (reference behaviour) and
widen the attention-side span bound to `vision_max_n_token + (COMPRESS_PAD_TO - 1) + row_len + 2`
(or pad the span bound to the next multiple of 128, which the prefill rows already use), so the
kernel constant covers the padded block. Version: official `vllm/vllm-openai:deepseekv41-flash-0909`
(v0.1.dev20904+g179dd0fa9), sm_120 and unchanged upstream code path.
</details>

## Reproduce

```bash
# his stack (his repo, his measured settings; GPUs 0-3 on an 8-card host), then ours
git clone https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000 && cd deepseek-v4.1-flash-4x-rtx-pro-6000
docker build -t deepseek-v41-4x6000:local .
docker run -d --name sero-ds41 --gpus '"device=0,1,2,3"' --ipc host --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -p 127.0.0.1:8010:8010 -v $MODEL_DIR:/models/DeepSeek-V4.1-Flash -v $PWD/state:/state -v $PWD/cache:/root/.cache \
  -e OFFLOAD_MODE=ram -e DSV41_CACHE_GIB=64 -e CONTEXT_LENGTH=524288 -e CHUNKED_PREFILL_SIZE=2048 \
  -e MEMORY_FRACTION=0.95 -e MAX_RUNNING_REQUESTS=8 -e MAX_TOTAL_TOKENS=4200000 -e API_KEY=$KEY deepseek-v41-4x6000:local
# same client against both
python3 tools/sm120_perf/openai_matrix.py --base http://127.0.0.1:8010 --api-key $KEY --model deepseek-v4.1-flash \
  --model-dir $MODEL_DIR --sizes 512,2048,8192,32768,131072 --concurrencies 1,4,8 --output-tokens 1024 --out sero.json
python3 tools/sm120_perf/openai_matrix.py --base http://127.0.0.1:8001 --model deepseek-ai/DeepSeek-V4.1-Flash \
  --model-dir $MODEL_DIR --sizes 512,2048,8192,32768,131072 --concurrencies 1,4,8 --output-tokens 1024 \
  --metrics-url http://127.0.0.1:8001/metrics --out moet.json
API_KEY=$KEY python3 tools/sm120_perf/decode_bench.py http://127.0.0.1:8010 deepseek-v4.1-flash '{"thinking": false}'
python3 tools/sm120_perf/quality_cmp.py --base http://127.0.0.1:8010 --api-key $KEY --model deepseek-v4.1-flash --out q_sero.json
python3 tools/sm120_perf/quality_cmp.py --base http://127.0.0.1:8001 --model deepseek-ai/DeepSeek-V4.1-Flash --out q_moet.json
python3 tools/sm120_perf/compare_outputs.py q_moet.json q_sero.json --tokenizer $MODEL_DIR/tokenizer.json
python3 llm_decode_bench.py --host http://127.0.0.1:8001 --model deepseek-ai/DeepSeek-V4.1-Flash --test-profile gsm8k \
  --profile-runs 200 --profile-concurrency 4 --max-tokens 8000 --completion-stats-temperature 0 \
  --request-overrides-json '{"chat_template_kwargs":{"thinking":false}}' --compare-baseline gsm8k_sero.json --output gsm8k_moet.json
```

Raw results (matrix, quality and GSM8K JSON for every run above):
[docs/benchmarks/dsv41-sm120-vs-0xsero/](benchmarks/dsv41-sm120-vs-0xsero/).
