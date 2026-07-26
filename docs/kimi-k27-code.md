# Kimi-K2.7-Code bring-up

The supported checkpoint is `nvidia/Kimi-K2.7-Code-NVFP4`: a 1T-parameter
MoE with 384 routed experts, top-8 routing, hidden size 7168 and dense MLA.
Its native NVFP4 weights do not fit in 4×96 GiB, so the serving recipe
re-quantizes routed experts into the same sign-symmetric 2-bit format used by
the GLM path.

The current publication recipe is:

```text
bench/recipes/kimi-k2.7-code-nvfp4/pro6000x4-tp4-256k.yaml
```

The imported bring-up measurement reported 51 tok/s single-stream decode,
222 tok/s aggregate at concurrency 8, 2,448 tok/s 8K prefill and needle
retrieval through 248K. These values predate schema-v2 result bindings and
remain historical until re-measured from the pinned model revision.

## Checkpoint-specific findings

- The INT4→NVFP4 export writes exact zeros predominantly as positive zero.
  A sign-preserving 2-bit map would therefore introduce a systematic bias.
  `VLLM_MOE_W2_ZERO_MODE=auto` detects this case and deterministically
  alternates zero signs before packing.
- Kimi requires the K=7168 W2/W4/W4Q kernel family. Kernel source, generated
  SASS and cubin status are tracked in `kernels/MANIFEST.md`.
- Dense-MLA decode required an SM12x shared-memory fix.
- Long chunked prefill required correcting `merge_attn_states` stride
  handling beyond 64K.
- The checkpoint has no native MTP head. The production recipe therefore
  does not enable speculative decoding.

## Validation required for a fresh release

1. Run the pinned checkpoint revision from `bench/models.yaml`.
2. Record a schema-v2 live result bound to the exact recipe and image.
3. Repeat arithmetic, executable-code, tool-call and multi-depth needle
   probes.
4. Collect a paired native quality baseline on hardware where the stock
   checkpoint fits before assigning a native-grade label.
