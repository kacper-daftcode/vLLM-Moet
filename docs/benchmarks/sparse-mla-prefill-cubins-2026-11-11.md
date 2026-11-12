# Sparse-MLA prefill kernel (sm_120) — the cubins: build, gates and times — 2026-11-11

[`kernels/MANIFEST.md`](../../kernels/MANIFEST.md) lists the cubins of the DeepSeek-V4.1 sparse-MLA prefill
attention kernel with one row each: what the cubin is and its status. This record holds what was measured for each
cubin and how it is built, with the conditions. What a variant changes in the kernel is in
[the internals](../dsv41-sm120-internals.md#step-46--the-prefill-attention-in-sass); the stores that three of the
cubins leave without a wait, the probe and `v43c` have
[their own record](sparse-mla-prefill-v43c-2026-11-10.md).

## Common to every cubin

| what | value |
|---|---|
| entry symbol | `sparse_mla_prefill_v41` with its 22 parameters in every cubin: one launcher loads them all — `VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN=<path>` at start, or as the second cubin of the runtime key (step 46b) |
| geometry | 1 token × 16 heads per CTA, 8 warps, 87 040 B of shared memory (87 KB: it decides the one CTA per SM), register prefetch of the next 64-candidate chunk |
| reference kernel | FlashInfer 0.7.2 `sparse_mla_prefill_dsv41_bf16_kernel<16,true>`; bit-identical means `torch.equal` on output and lse |
| synthetic suite | 8 cases: fp8 SWA + FP4 compressed caches, partial lengths, −1 holes, page sizes 32/64/128, sink / no sink, empty rows |
| dumps | 8 dumps of a real prefill (19K/163K, layers 0 and 2), written by `patch_vllm_sparse_mla_prefill_dump_sm120.py` + `replay_specB_dumps.py` |
| harness | `tools/dsv41_sm120/sparse_mla_sass/test_sparse_mla_sass_sm120.py` (`--synthetic`, `--time`, dumps); its four shapes are 128+512 p128 / 192+512 / 1152+512 / 128+512 p64 at 4096 rows, times in µs per launch |
| build script | `tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh`; each build reproduces the digest of its cubin |
| second host, reference host | a 4× RTX PRO 6000 Server Edition host; the 4× RTX PRO 6000 Workstation Edition host of "Measurements, as served" |

**Reproduction contract** (cubit KNOWN_ISSUES A20, 2026-11-06). A shipped cubin is reproduced byte for byte from
itself — `cubit disassemble --frozen <shipped.cubin> -o k.sass && cubit asm -T <shipped.cubin> k.sass -o rebuilt.cubin
&& cmp` — with any cubit ≥ 0.2.0 (the frozen path is gated 157/157 on every cubit push). Re-assembling the `.sass`
source is a regeneration: its control codes follow the current allocator, so it is a new schedule that needs in-situ
validation before it ships, with the switches its section names.

**The clock note.** The latencies of cubit's dependency table are counts of issue cycles measured at a core clock of
at most 2 430 MHz and re-measured unchanged on cards running at up to 2.9 GHz. Before a cubin with the
dependency-aware schedule (`v41s1` and every cubin after it) runs on cards with a higher core clock, repeat
`test_sparse_mla_sass_sm120.py --cubin <this> --synthetic` and the dump replay there.

## `sparse_mla_prefill_v41.cubin` — the default

| what | value |
|---|---|
| file | sha256 `c896234bc76d6938…`, 49 128 B; source `kernels/sass/sparse_mla_prefill_v41.sass`, generator `kernels/gen/gen_sparse_mla_prefill_v41.py` |
| assembled with | cubit `53e9ed8` + BUG-421/422 = `main`'s BUG-478/479 (`ed4d316`), `CUBIT_SB_SHARE=1 CUBIT_RB_EARLY=32 CUBIT_MMA_WAIT=1 CUBIT_MMA_WB_DRAIN=13`; strict round trip with cubit `main` 2 663 / 2 663 |
| rebuild | needs cubit `main` at or after `ed4d316` (BUG-478/479/480 there). The cubin reproduces byte for byte from `main` up to cubit `52767f9`; from `f064bed` (BUG-485, the allocator's WAW rule) `main` emits the same instruction text with two extra wait control words (`@P0 VIMNMX R18/R19`) — the shipped cubin stays pinned to its digest, a rebuild would need the in-situ A/B again |
| bits | bit-identical on the synthetic suite and on the dumps (8/8) |
| harness, the four shapes | 845/953/2466/981 µs against the reference kernel's 1290/1447/3547/1370: 0.66–0.72× its time |
| in situ: image `-r23` with `VLLM_MOET_SPARSE_MLA_PREFILL_SASS=1`, the second host, 2026-11-05 | fresh prefill 163K 7.70 → 7.30 s (−5.2 %), 19K 905 → 860 ms (−5.0 %); long_agree ×2 + agree ALL IDENTICAL; loaded on 4 ranks, 0 disabled, 0 Traceback |
| image | Dockerfile step 46, from image `-r25` on; the default cubin |

## `sparse_mla_prefill_v41s1.cubin` — the default's instruction words under the dependency-aware schedule

| what | value |
|---|---|
| file | sha256 `5924622b6289fca4…`, 47 160 B; the source and generator of `v41` |
| assembled with | cubit `main` at `38e4cc8` (the scheduler change is `49f8340`), the switches of `v41` plus `CUBIT_SCHED2=1 CUBIT_SCHED2_LAT=sm120 CUBIT_MMA_PTXAS=1`; build: `SCHEDULE=s1 tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh` |
| instructions | 2 540 — the 128 MMA writeback drains of the default schedule are gone, 5 NOP pads; strict round trip 2 540 / 2 540, the frozen listing re-assembles byte for byte |
| schedule, per token-warp at 128+512 | stall cycles 67 807 → 35 555 (−48 %), issued instructions 10 928 → 9 698 |
| bits | bit-identical with the reference kernel and with `v41`: the 8 synthetic cases and the 8 dumps, each on 30 repeated launches in one process and in 30 fresh processes, the default cubin as the control in the same processes — 0 differences |
| harness, the four shapes, the default cubin in the same rounds | 898 / 986 / 2 162 / 864 against 1 058 / 1 153 / 2 503 / 1 025 µs = −15.1 / −14.5 / −13.6 / −15.7 % (the reference kernel 1 583 / 1 753 / 4 005 / 1 556) |
| Nsight Compute, one launch at 128+512 | the fixed-stall wait falls from 5.34 to 2.78 warp-cycles per issued instruction of ≈ 10; the tensor pipe and the scoreboards take up part of it — lowering the MMA distances further (below ptxas) is still bit-identical and buys ≤ 0.5 %, so this is where stall rules end |
| in situ: image `-r28` as built with this cubin mounted and named by `VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN`, against the same image with the default cubin, the second host, 2026-11-10 | fresh prefill 163K 5 175 → 5 068 ms (−2.1 %), 19K 655 → 634 ms (−3.2 %), at a 300 W cap 163K 5 818 → 5 703 ms (−2.0 %); long_agree ×2 + agree ALL IDENTICAL against three references; 400 chunk-layer dumps × 5 states on 4 ranks ALL IDENTICAL; loaded on 4 ranks, 0 disabled, 0 Traceback; fresh prefill 163K under CPU load +4.7 % (limit ≤ +5 %; the default cubin +4.6 %) |
| image | copied from image `-r30` on (Dockerfile step 46b); nothing loads it by default |

## `sparse_mla_prefill_v41s1b.cubin` — `v41s1` with the vendor assembler's distance in front of the branches

| what | value |
|---|---|
| file | sha256 `cba10b0adafd20c3…`, 47 160 B; the source and generator of `v41`; equal to `v41s1` word for word (2 540 of 2 540 instruction texts; one control field differs: one byte of the file) |
| assembled with | cubit `main` at `ef7d335` or later with the switches of `v41s1` and one more row in its dependency table — an MMA holds every `BRA` 14 issue cycles off, the row `v43b` is built with — which the build script adds to a copy of cubit's `tables/dep_latency_sm120.json` and passes as `CUBIT_SCHED2_LAT`; build: `SCHEDULE=s1b tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh` |
| instructions | 2 540; `CUBIT_VERIFY` 0 / 0, `CUBIT_DEP_VERIFY` 0 against that table; strict round trip 2 540 / 2 540, the frozen listing re-assembles byte for byte; 128 registers |
| the one word | `BAR.SYNC` behind PV, stall 6 → 13: the last `HMMA` of PV stands 14 and 19 issue cycles in front of the two conditional branches behind it (`@!P5 BRA` to the epilogue, `@P6 BRA` to the FP4 dequant) instead of 7 and 12 |
| cost | 7 issue cycles per chunk — not measurable |
| harness, the four shapes, `v41s1` / `v41s1b` in the same rounds (Workstation Edition, core clock 2.7 GHz) | 731 / 801 / 1 751 / 753 against 732 / 801 / 1 754 / 754 µs (the default cubin 870 / 948 / 2 079 / 890, `v43b` 585 / 645 / 1 467 / 612) |
| real prefill inputs, the same rounds | 128+512 candidates 688 / 688, 126+448 630 / 629, 128+0 196 / 195 µs (the default 811 / 741 / 221, `v43b` 553 / 508 / 160) |
| bits | bit-identical with the reference kernel: the 8 synthetic cases and the 8 dumps, 30 repeated launches in one process and 30 fresh processes each, `v41s1` as the control — 0 differences, on two Workstation cards |
| in situ | not measured (the in-situ numbers of `v41s1` are of the same instruction stream); in no image |

## `sparse_mla_prefill_v42.cubin` — the dequant by the scaled BF16 unpack

| what | value |
|---|---|
| file | sha256 `dfdc0279ea5a3388…`, 29 784 B; source `kernels/sass/sparse_mla_prefill_v42.sass`, generator `kernels/gen/gen_sparse_mla_prefill_v42.py` — the generator of `v41` with a new `emit_dequant`; `SMLA_DEQUANT=v41` reproduces `sparse_mla_prefill_v41.sass` byte for byte |
| assembled with | cubit `main` at `db4cfd1` (the instruction's table row is `89564ac`), the policy and the schedule switches of `v41s1`: `CUBIT_SB_SHARE=1 CUBIT_RB_EARLY=32 CUBIT_MMA_WAIT=1 CUBIT_MMA_WB_DRAIN=13 CUBIT_SCHED2=1 CUBIT_SCHED2_LAT=sm120 CUBIT_MMA_PTXAS=1`; build: `KERNEL=v42 tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh` |
| instructions | 1 454; `CUBIT_VERIFY` 0 unprotected RAW / 0 broken WAR, `CUBIT_DEP_VERIFY` 0; strict round trip 1 454 / 1 454, the frozen listing re-assembles byte for byte. The verifier did not see 4 store → overwriter pairs that carry no wait on one path ([record](sparse-mla-prefill-v43c-2026-11-10.md)) |
| the dequant, per task | an fp8 chunk 44 → 10 instructions, an FP4 chunk 55 → 21; issued instructions per token-warp at 128+512: 9 698 → 6 986 (−28 %); stall cycles 35 555 → 30 211 |
| exactness of the new sequences on silicon (`tools/dsv41_sm120/sparse_mla_sass/probe_dequant_unpackb_sm120.py`: the instructions of `v42` against those of `v41` on every input, RTX PRO 6000) | 65 536 e4m3 pairs × 256 scale bytes in either half of the source word, 256 e2m1 bytes × 256 E4M3 block-scale codes through each of the four byte selectors, the block scale alone on all 256 codes — all equal, with junk in every unused byte, NaN codes and −0 included |
| bits | bit-identical with the reference kernel: the 8 synthetic cases and the 8 dumps, each on 30 repeated launches in one process and in 30 fresh processes, `v41s1` as the control in the same processes — 0 differences; the runtime key's helper with the default cubin and this one in either role 17 / 17 |
| harness, the four shapes, in the same rounds | 833 / 918 / 2 051 / 805 against `v41s1` 903 / 991 / 2 182 / 872 and the default cubin 1 060 / 1 154 / 2 506 / 1 025 µs = −7.8 / −7.4 / −6.0 / −7.7 % against `v41s1`, −21.4 / −20.5 / −18.2 / −21.5 % against the default (the reference kernel 1 605 / 1 753 / 4 035 / 1 568) |
| variants of the generator in the same rounds, all bit-identical | one register set 841 / 927 / 2 079 / 812; three sets 862 / 952 / 2 137 / 835; the shifts of `v41` instead of the byte selectors 868 / 958 / 2 148 / 839 (three sets); the shifts and the four-instruction block scale of `v41` 911 / 1 005 / 2 228 / 883 (three sets) |
| the instruction's latencies under cubit's table | measured on sm_120 silicon as a producer and as a consumer (cubit `tools/stallfloor/stall_floors_sm120_20261110_pairs_d.json`): the floors of the plain unpack / pack forms, at or below the cells that price them |
| in situ | not measured; in no image |

## `sparse_mla_prefill_v43.cubin` — two load waits taken out

| what | value |
|---|---|
| file | sha256 `43f04db5cc9e9874…`, 30 344 B; source `kernels/sass/sparse_mla_prefill_v43.sass`, generator `kernels/gen/gen_sparse_mla_prefill_v43.py` — the generator of `v42` with two switches; `SMLA_IDX_PREFETCH=0 SMLA_PROLOGUE_OVERLAP=0` reproduces `sparse_mla_prefill_v42.sass` byte for byte, with `SMLA_DEQUANT=v41` also `sparse_mla_prefill_v41.sass` |
| assembled with | cubit at `df4c557` — `main` `ef7d335` plus the opt-in colouring rule `CUBIT_SB_LONG_KEEP` —, the switches of `v42` and `CUBIT_SB_LONG_KEEP=1`; build: `KERNEL=v43 tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh` (it refuses a cubit without the rule) |
| instructions | 1 489; `CUBIT_VERIFY` 0 unprotected RAW / 0 broken WAR, `CUBIT_DEP_VERIFY` 0; strict round trip 1 489 / 1 489, the frozen listing re-assembles byte for byte; regcount 136, the index read ahead in registers R126–R129. The verifier did not see 3 store → overwriter pairs that carry no wait on one path ([record](sparse-mla-prefill-v43c-2026-11-10.md)) |
| where the warps of `v42` wait for something other than arithmetic (ncu per-instruction profile, real prefill inputs, 128+512 candidates) | the candidate index, 10.6 % of the samples (a load of the token's own index row: never cached, ≈ 800 cycles); the prologue's six global loads in a row, 4.9 % (more on a two-chunk layer); store-latch waits in the FP4 dequant, 5.7 % — tried with the outputs in registers no pending store reads: no gain, left out |
| built with `CUBIT_SB_LONG_KEEP=1` | the same text without the rule is legal and bit-identical, 1–4 % slower on real inputs and 5 % slower when every gather misses the cache |
| bits | bit-identical with the reference kernel: the 8 synthetic cases and the 8 dumps, each on 30 repeated launches in one process and in 30 fresh processes, `v42` as the control in the same processes — 0 differences, on two RTX PRO 6000 cards (Server Edition at a core clock of 2.4 GHz, Workstation Edition at 2.7–2.8 GHz) |
| real prefill inputs, 4096 rows, µs per launch in the same rounds, reference kernel / default cubin / `v41s1` / `v42` / `v43`, Server Edition | 128+512 candidates: 1 360 / 980 / 834 / 775 / 681; 126+448: 1 237 / 897 / 765 / 711 / 625; 128+0: 298 / 269 / 240 / 224 / 200 = −12.1 / −12.1 / −10.7 % against `v42`, −30.5 / −30.3 / −25.8 % against the default |
| the same on the Workstation card | 1 132 / 809 / 686 / 637 / 550, 1 027 / 740 / 629 / 584 / 507 and 244 / 221 / 195 / 183 / 160 = −13.7 / −13.2 / −12.6 % against `v42`, −32.0 / −31.5 / −27.4 % against the default |
| harness, the four shapes (random candidate indices — every gather a cache miss; real candidates are a sliding window and cost 7–8 % less) | 730 / 808 / 1 750 / 693 against `v42` 834 / 890 / 1 963 / 800 and the default 1 060 / 1 154 / 2 448 / 1 021 µs |
| variants of the generator in the same rounds, all bit-identical, real 128+512 input | the index prefetch alone 714; with the four `q` loads 691; this text without `CUBIT_SB_LONG_KEEP` 689 µs |
| in situ: the runtime key flipped inside one server lifetime, image `-r31` as built, the second host | 1.85 % ± 0.15 % faster than the default cubin at 163K |
| image | copied from image `-r31` on (Dockerfile step 46c); nothing loads it by default |

## `sparse_mla_prefill_v43b.cubin` — `v43` with the vendor assembler's distance in front of the branches

| what | value |
|---|---|
| file | sha256 `a163aea6d869c55e…`, 30 344 B; the source, generator and text of `v43`; equivalent to `v43` word for word (1 489 of 1 489 instruction texts; stall fields: one differs) |
| assembled with | cubit at `df4c557` or later with the switches of `v43` and one more row in its dependency table — an MMA holds every `BRA` 14 issue cycles off — which the build script adds to a copy of cubit's `tables/dep_latency_sm120.json` and passes as `CUBIT_SCHED2_LAT`; build: `KERNEL=v43b tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh` |
| instructions | 1 489; `CUBIT_VERIFY` 0 / 0, `CUBIT_DEP_VERIFY` 0 against that table; strict round trip 1 489 / 1 489, the frozen listing re-assembles byte for byte. The verifier did not see 3 store → overwriter pairs that carry no wait on one path ([record](sparse-mla-prefill-v43c-2026-11-10.md)) |
| the one word | `BAR.SYNC` behind PV, stall 6 → 13: the last `HMMA` of PV stands 14 and 20 issue cycles in front of the two conditional branches behind it (`@!P5 BRA` to the epilogue, `@P6 BRA` to the FP4 dequant) instead of 7 and 13 |
| why 14 | ptxas never issues a conditional branch closer than 14 cycles behind an MMA: 166 sites of the MoE kernels it compiled for this stack, 53 at exactly 14. In the default cubin the writeback drains kept 39 and 44 cycles there; the dependency-aware schedule removed the drains (`v41s1`, `v42`, `v43`: 7 and 12–13) and cubit's table had no row to put the distance back. No failure was observed at 7 |
| cost | 7 issue cycles per chunk — not measurable |
| real prefill inputs, 4096 rows, µs per launch in the same rounds, `v43` / `v43b` (Workstation Edition, core clock 2.7 GHz) | 128+512 candidates 554 / 554, 126+448 509 / 509, 128+0 161 / 161 (the default cubin 812 / 742 / 221, `v42` 643 / 589 / 183) |
| harness, the four shapes, the same rounds | 586 / 647 / 1 469 / 607 against 585 / 646 / 1 468 / 607 µs |
| bits | bit-identical with the reference kernel: the 8 synthetic cases and the 8 dumps, 30 repeated launches in one process and 30 fresh processes each, `v43` as the control — 0 differences |
| in situ: the runtime key flipped inside one server lifetime, image `-r32` as built, the second host | 1.83 % ± 0.12 % faster than the default cubin at 163K (8 pairs), 1.66 % ± 0.61 % at 19K |
| the four cards of the reference host, with the server stopped | bits and timing floors PASS on 4 of 4 ([record](dsv41-sm120-r32-2026-11-11.md#the-attention-kernel-on-the-four-cards)) |
| image | copied from image `-r32` on (Dockerfile step 46d); loads only when named. While `-r32` was the served image, `SERVED=1` named it as the second cubin of step 46b: loaded, compared with the default once, never selected |

## `sparse_mla_prefill_v43c.cubin` — `v43b` with every store of the dequant waited

sha256 `7e812fa2d9c47770…`, 30 344 B: `v43b` with one control word changed, so that every store of the dequant is
waited before a register it reads is rewritten, on every path — the branch behind the loop-top `BAR.SYNC` waits on
scoreboards 3, 4 and 5. Build: `KERNEL=v43c tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh`. The
instruction words, stalls and time of `v43b`; bit-identical with the reference kernel. Its gate, the
probe and its times: [record](sparse-mla-prefill-v43c-2026-11-10.md).

| what | value |
|---|---|
| in situ: the runtime key flipped inside one server lifetime, image `-r33` as built, the second host | 1.81 % ± 0.05 % faster than the default cubin at 163K (8 pairs) |
| in situ: the same key in the serving server of the reference host, image `-r33` | 2.569 % ± 0.323 % faster than the default cubin at 163K, 2.306 % ± 1.130 % at 19K (8 pairs each, read by order; [record](dsv41-sm120-r33-2026-11-12.md#the-runtime-keys-in-the-serving-server)) |
| the four cards of the reference host, with the server stopped | bits and timing floors PASS on 4 of 4; the runtime key's helper with the two cubins in both roles 17 of 17 on each ([record](dsv41-sm120-r33-2026-11-12.md#the-attention-kernel-on-the-four-cards)) |
| image | copied from image `-r33` on (Dockerfile step 46e) |
