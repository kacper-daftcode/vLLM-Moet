# The MoET hybrid: DeepSeek-V4.1-Flash on RTX 5090 with CPU-computed experts

What `Dockerfile.sm120-dsv41-nightly` steps 43 (`--build-arg VLLM_MOET_CPU_TIER_PATCH=1`) and 44
(`--build-arg VLLM_MOET_ENGRAM_V2_PATCH=1`) install into the DeepSeek-V4.1 image, and the tools around it. The
user-facing description, the knobs and every measurement: [`docs/moet-hybrid.md`](../../docs/moet-hybrid.md); the
record and the comparison with the field: [`docs/benchmarks/moet-hybrid-4x5090-2026-11-04.md`](../../docs/benchmarks/moet-hybrid-4x5090-2026-11-04.md).
The launcher is [`docker/sm120/run-dsv41-5090.sh`](../../docker/sm120/run-dsv41-5090.sh) (every knob in its header);
the host-side units (GPU power cap, kernel-thread affinity) are in [`docker/sm120/host/`](../../docker/sm120/host/README.md).

| directory | what | installed by | tests (no GPU unless noted) |
|---|---|---|---|
| `cpu_tier/` | the runtime: `vllm_moet_cpu_tier.py` (the vLLM hooks: resident rows, the placement manager, publish / inject, prefill streaming), `cpu_tier.cpp` + `cpu_tier.py` (the host dispatcher, the CCD pools, the mailboxes, the expert store and its NVMe tail, `TierGroup` for virtual ranks), `cpu_tier_cuda.cu` (the in-graph publish / wait-inject / promote kernels), `placement_manager.py` (static placement + LRU promotions), `patch_vllm_moe_cpu_tier.py` (the idempotent patcher; `--check`) | step 43 → `/opt/vllm-moet/cpu_tier` | `test_cpu_tier.py`, `test_promotion.py`, `test_cpu_tier_vranks.py` (one GPU: the tier end to end in captured graphs, fail-closed, the promotion protocol, V = 2 / 4 virtual ranks); `test_store_tail.py`, `test_store_load.py` (host); `mini_identity_runs.sh` (one GPU: the mini models, all-resident vs placement); `DEBUG_COMPACT=1` in a window = the within-boot identity check on the full model |
| `cpu_moe/` | the CPU expert kernel `cpu_moe_b12x.cpp` (bit-exact with b12x's compact W4A8 path; built into `libcpu_tier.so`), `build_store.py` (the expert store on NVMe, once per checkpoint), `dsv41_b12x.py` / `dsv41_mxfp4.py` (checkpoint access, the GPU-math emulation), `qmma_probe.cu` + `gpu_parity_b12x_*.py` (the sm_120 MMA's rounding rules and the kernel's GPU tables, once per GPU model), the benches (`bench_real_records_b12x.py`, `bench_pattern_b12x.py`, `bench_l3_vs_dram_b12x.py`, `ccd_bw.cpp`, `bench_b12x_tp_shapes.py`, `bench_expert_parallel.py`), the Engram probes (`engram_hash_probe.py`, `engram_rows_probe.py`) | step 43 (the kernel source) | `test_cpu_moe_b12x.py` (vs the scalar reference), `gpu_parity_b12x_compare.py` on the four recorded sets, `parity_sweep_{ranks,layer}.sh` |
| `engram_v2/` | the Engram tables on NVMe: `engram_v2.cpp` (io_uring fetcher, DRAM row cache), `engram_v2_cuda.cu` (mailbox → landing buffer), `vllm_moet_engram_v2.py` (vLLM's lookup kernel on the landing buffer), `patch_vllm_engram_v2.py`, `nvme_bench.cpp` | step 44 → `/opt/vllm-moet/engram_v2` | `test_engram_v2.py`, `engram_v2_harness.py` (one GPU, real shards) |
| `routing/` | the routing traces and the policies on them: `collect_routes.py` / `build_sequences.py` (native traces), `placement.py` (static placements from frequencies; the shipped ones in `placements/`), `promo_sim.py` (hit rate / CPU experts / promotions per step for a slot budget and admission policy), `nvme_tier_sim.py` (VRAM → DRAM → NVMe residency), `cache_sim.py`, `route_predict_eval.py` | — | — |
| `handoff/` | `handoff_proto.cu`: the standalone GPU ↔ CPU handoff inside one captured graph (the phase-2.0 prototype the design was decided on) | — | — |

Measurement tools (`cpu_tier/`): `tp4_window.sh TAG [KEY=VALUE …]` (one boot with the measurements — identity prompts,
decodes, the KPI lines, optional `PROFILE_STEPS`, `TTFT_LENGTHS`, `NEEDLE_LENGTHS`, `SOAK_MINUTES SOAK_CONC`, `HARD_PROBE`,
`DEBUG_COMPACT`; `TP` / `GPUS` for the 2× / 1× configurations), `ttft_probe.py`, `needle_probe.py`, `soak_probe.py`,
`profile_anatomy.py` (the decode step's GPU anatomy from a torch profile), `cpu_placement_probe.sh` (who runs on the pool
cores), `compare_mini_outputs.py` (identity across boots), `dump_compare.py`, `make_mini_real.py` / `serve_mini_hybrid.sh` (the
mini models), `serve_hybrid.sh` (the launcher's in-container half).

Historical (kept for the record, not used by the image): `cpu_moe/cpu_moe_mxfp4.cpp` with `test_cpu_moe_mxfp4.py`,
`bench_cpu_moe_mxfp4.py`, `gpu_parity_prepare.py` / `gpu_parity_run.py` — the v1 kernel with the DeepGEMM line's numerics and
its own pack layout, superseded by `cpu_moe_b12x.cpp` once the serving path moved to b12x.
