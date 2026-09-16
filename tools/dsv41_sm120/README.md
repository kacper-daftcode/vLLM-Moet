# DeepSeek-V4.1-Flash on SM120 — kernel gap closure

Kernel-side patches that let the official `vllm/vllm-openai:deepseekv41-flash-0909` image serve
DeepSeek-V4.1-Flash on RTX PRO 6000 / RTX 5090 (sm_120). Port notes, gap inventory and
validation evidence: `docs/dsv41-sm120-port.md`; image: `Dockerfile.sm120-dsv41`.

| file | role |
|---|---|
| `sparse_mla_sm120_dsv41.cu` | new FlashInfer JIT TU: SM120 sparse-MLA instantiations for the V4.1 geometry (SWA page 32, compressed page 128/64, SWA rows 128/192/1152) + `sparse_mla_prefill_dispatch_dsv41` |
| `patch_flashinfer.py` | idempotent, anchored patcher for an installed flashinfer package (installs the TU, orchestrator hook, PBS=32 decode table, python dispatch) |
| `patch_deepgemm.py` | DeepGEMM `8b1392b9` host asserts: SM120 FP8 paged MQA logits on 128-row pages |
| `test_sparse_mla_sm120_dsv41.py` | op-level validation: torch reference + bit-exact re-paging parity vs stock PBS=64 kernels |
| `test_deepgemm_sm120_paged_mqa.py` | op-level validation: DeepGEMM reference + bit-exact parity block_kv 64 vs 128 |

Run the tests inside the built image on one SM120 GPU:

```bash
docker run --rm --gpus '"device=0"' --entrypoint bash vllm-moet-sm120:dsv41-0909 -c \
  'python3 /opt/vllm-moet/dsv41_sm120/test_sparse_mla_sm120_dsv41.py --quick &&
   python3 /opt/vllm-moet/dsv41_sm120/test_deepgemm_sm120_paged_mqa.py'
```
