# GLM-5.3-Flash on 4× RTX PRO 6000, stock vLLM nightly (2026-10-26)

The baseline the sm_120 work on this model starts from: the official FP8 checkpoint
([`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash): 45 layers — 34 KDA linear
attention + 11 DeepSeek sparse attention with the kpool indexer — 288 routed experts top-8, MTP) on the
unmodified `vllm/vllm-openai:nightly-49cb2bbf` image, TP4, 512K context, MTP k = 3, FP8 KV, a 64 GiB host
offload region plus a disk tier, the cards at 400 W. No vLLM-Moet kernels.

| | measured on the served configuration |
|---|---|
| decode, 1 stream | prose 175–190 tok/s, code 244–258 tok/s (71–75 steps/s; MTP accepts 1.4 / 2.6 of 3 drafts per step) |
| decode, 8 streams | prose 545–557 tok/s, code 710–726 tok/s aggregate |
| fresh prefill | 20K tokens in 2.0 s (10K tok/s), 169K in 17.8 s (9.4K), 406K in 47.6 s (8.5K) |
| KV cache | 613K tokens on the GPUs at `--gpu-memory-utilization 0.92` (1.17× concurrency at 512K), 64 GiB host offload, disk tier |
| quality | GSM8K-200 (reasoning effort low) 192/200, needle PASS at 29K / 106K / 400K, tool calls / JSON / vision pass |
