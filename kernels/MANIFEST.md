# Kernel manifest (SM120 / sm_120, RTX PRO 6000 Blackwell + RTX 5090)

Hand‑written SASS assembled by `cubit` (shipped cubins: pinned @ `5912400`). Cubins are **SM120‑only**.
Assemble: `cubit asm sass/<SASS> -o <cubin> --kernel <kernel> --mercury-stub sass/qmma_e4m3.merc.stub`.

**Per-source allocator switches (cubit ≥ `f643e00`, 2026-11-06):** the five MC4 K ≥ 4096 sources are assembled with
`CUBIT_RB_CHAIN=1 CUBIT_SB_SHARE=1 cubit asm sass/<SASS> -o <cubin> --kernel moe_w2_mm --mercury-stub sass/qmma_e4m3.merc.stub`
— `moe_w2_mm_mc4_k4096_a32.sass`, `moe_w2_mm_mc4_k6144.sass`, `moe_w2_mm_mc4_k6144_a32.sass`, `moe_w2_mm_mc4_k7168.sass`,
`moe_w2_mm_mc4_k7168_a32.sass`. The two switches restore the `5912400` schedule of these kernels (the read-barrier
chain on the scale-byte loads + the shared-W0 prologue; cubit KNOWN_ISSUES A19): measured on a second host (RTX PRO
6000, the kernels' harness, 25 reps, E = 64; bits identical to the shipped cubins in every arm) the `main` default runs them at
1.37× / 1.38× / 1.65× / 1.21× / 1.67× of the shipped time and the switched build at 1.01× / 0.84× / 1.01× / 0.84× /
0.98× (`mc4_k4096_a32` 340.7 → 467.6 → 343.5 µs, `mc4_k6144` 475.8 → 658.6 → 400.6, `mc4_k6144_a32` 513.4 → 847.4 →
519.9, `mc4_k7168` 560.5 → 680.7 → 469.6, `mc4_k7168_a32` 599.8 → 1002.9 → 590.4). The switches cost 3–30 % on 57
of the other 79 kernels and stay OFF for every other source. The shipped cubins are unchanged: a regenerated cubin
of these five ships only after the in-situ validation pass (`tools/test_moe_w2_planes.py` + the GSM8K / agreement
battery) is repeated with the new assembler, like every other regeneration in this manifest.

**Sources re‑canonicalized for the current cubit (`main` ≥ `f8605f6`, strict encoder):** the
`.ROW/.COL` operand suffixes on `QMMA.SF` (no such bit exists in the SM120 encoding — the `5912400`
encoder silently dropped them, the current one refuses, BUG‑272) are gone from every `sass/*.sass`
and every generator in `gen/`, and the 4‑operand `IMAD.WIDE.U32 Rd, Ra, b, Rc` (silicon reads `c` as
the 64‑bit pair `Rc:Rc+1` — nvcc's address idiom, cubit BUG‑008 warning) is spelled in the explicit
5‑operand form `IMAD.WIDE.U32 Rd, PT, Ra, b, Rc`. Both spellings encode to the same 128‑bit word
(`cubit encode`, checked), the generators still reproduce their `.sass` byte for byte, all 90 sources
assemble with 0 errors / 0 warnings, and the result differs from the shipped cubins only in the NOP
tail (instruction‑level diff of all 86 matched kernels). The shipped `cubins-sm120/*.cubin` are
unchanged: a regenerated cubin ships only after its GPU validation is repeated with the new assembler.

## Activation format rev `_a32` (group‑size‑flexible A scales) — the LIVE set

All MoE GEMM cubins the server loads carry the `_a32` filename suffix: the
A‑side f32 scale plane is read at **PER‑32 stride** (f32 `[rows, K/32]`,
desc `as` row stride `(K/32)*4`) and folded into the split‑K masters **once
per k32**. Broadcasting one scale over consecutive 32‑groups is identical to
quantizing at a coarser group, so ONE cubin set serves activation groups
{32, 64, 128} per GEMM — selected host‑side (`VLLM_MOE_W2_A32_G1/G2`,
`VLLM_MOE_W2_A32_UE8M0`). Regcount stays 64 = 4 CTA/SM on `moe_w2_mm`; the
per‑k32 fold costs ≤6% on decode‑shape kernel micro‑bench (e2e within the
2×6000 anchor band).

**Default configuration: G1=128 G2=128 UE8M0=1** — numerically the a128
format (its per_token_group_quant inherited the platform's DeepGEMM‑E8M0
power‑of‑2 rounding). The per‑32 variants this rev was built to test
(activation‑precision handoff, plan A) were **E2E‑falsified** on GSM8K‑200
(2×6000, quintal τ1.0, weights bit‑exact):

| A format (G1/G2, scale) | accuracy | tokens avg |
|---|---|---|
| native FP8+MXFP4 (anchor) | 97.0% | 116 |
| a128 lineage (anchor) | 97.0% | 125 |
| 128/128 ue8m0 (= default, `_a32` kernels) | **97.0%** (flips 1↔1) | **122** |
| 32/128 f32 | 96.5% | 124 |
| 32/32 f32 | 95.5% | 127 |
| 32/32 e8m0 (native MXFP8 A format) | 95.5% | 122 |

Finer A groups do NOT shorten completions — the +8–11% token inflation vs
native does not come from activation‑scale granularity (nor from the
mid‑pipeline requant; the 32/128 mix isolates it) — and they consistently
cost accuracy against the 2‑bit/quintal weight planes. Op‑level the finer
groups measure BETTER (bf16‑reference worst_rel 1.0–1.2× lower), so the
loss is a serving‑distribution effect, not a kernel defect.

The e8m0‑per‑32 build also validated QMMA.SF's **sfa** operand as a working
alternative fold path: per‑row UE8M0 bytes, thread map HW‑probed
(`gen/gen_moe_sfa_probe.py` + `gen/moe_sfa_probe_check.py`, PASS: lane *t*
supplies the LOW byte for A row `8*(t&1) + (t>>2)`), host quant verified
byte‑identical to flashinfer `mxfp8_quantize` (`2^ceil(log2(amax/448))`).
The probe pair stays in `gen/` as the validated sfa documentation for any
future MX revisit.

**Format coherence is enforced by filename**: the loader
(`moe_w2_cubit._ensure_ready`) only opens `*_a32.cubin`, and
`_require_kernels` asserts the complete set — a stale/partial cubin dir
fails at weight load instead of silently mixing scale strides. The retired
`a128` files (no suffix) remain on disk ONLY for in‑flight a128 serving
sessions and drop out at the next regen. Retired for good (not regenerated
in `_a32`): the `mc2` prefill variant and the `MB=2` experiment — both
superseded by MC=4/AFRAG, never launched by the serving path.

## Dependency‑aware schedule set of the live MoE GEMM kernels — `cubins-sm120-s1/` (not loaded by default, in no image)

`cubins-sm120-s1/` (next to `cubins-sm120/`, not inside it) is the complete live `_a32` MoE GEMM set — the 43 file names
of the sections below — with cubit's dependency‑aware stall schedule wherever that schedule is faster: **37 files are
re‑scheduled, 6 are byte copies of the pinned cubin** (`cubins-sm120-s1/SHA256SUMS` lists all 43). **Nothing loads the
directory and no Dockerfile copies it.** The loader opens exact file names in `VLLM_MOE_W2_CUBIT_DIR` (default
`/cubit-share`, the image's copy of `cubins-sm120/`), so the pinned set stays what a server runs. `Dockerfile.sm120-v024`
copies `cubins-sm120/` with its subdirectories, which is why the set is not a subdirectory of it: an image changes only
through a step of its own. A deployment that wants the set adds that step — a `COPY` of `kernels/cubins-sm120-s1/` to a
directory of its own, checked there with `sha256sum -c SHA256SUMS` — or mounts `<checkout>/kernels/cubins-sm120-s1`,
and points `VLLM_MOE_W2_CUBIT_DIR` at that directory. The directory holds the MoE GEMM set only: anything else a
configuration looks up in the loader's directory (the opt‑in cubit sparse‑MLA kernels, all default off) is not in it.

**What a re‑scheduled file is.** The source of the pinned cubin of the same name (`sass/<name>.sass`) assembled by cubit
`main` (`ef8ff88`; byte‑identical at `ef7d335`) with `CUBIT_SCHED2=1 CUBIT_SCHED2_LAT=sm120 CUBIT_MMA_PTXAS=1` — the
switches of `sparse_mla_prefill_v41s1.cubin`: a stall is charged where a consumer needs it instead of behind every
producer (cubit's pair‑latency table for sm_120, enforced on every path), and the QMMA chain runs at ptxas's issue
distances instead of between writeback drains. The instructions are the source's — word for word those of today's
default assembly; against the pinned cubins (the `5912400` build) 328 words of the 37 kernels are another encoding of
the same instruction (230 `IADD3`, 98 `ISETP`: operand order, `RZ` for `#0`) and the control words are new. In a pinned
schedule every QMMA sits in 72 stall cycles (one drain in front, four behind); in the set a QMMA pair is
`QMMA S01 / NOP S15 / QMMA S12` (issue distance 16) and the fold reads a result exactly 28 cycles behind its QMMA.
Over the 37 kernels: stall cycles per warp 260 651 → 125 070 (−52.0 %), instructions 42 598 → 38 710
(the 4 320 drains are gone). cubit's read‑only verifiers on each of the 37: `CUBIT_VERIFY` 0 unprotected RAW / 0 broken WAR,
`CUBIT_DEP_VERIFY` 0 / 0 / 0 / 0. None of them carries a per‑source allocator switch (the three live sources that do
are among the six that keep the pinned bytes).

**The rule of the set** (fixed before the second sample was taken): a kernel takes the new schedule when its output is
identical to the pinned cubin's in every warm launch of both samples and in every fresh process, **and** the new schedule
is faster than the pinned one in both samples, **and** by at least 2 % in their mean (E = 64); every other kernel keeps
its pinned bytes, so that the directory is a complete loader directory.

**Gate (kernel level; `gen/moe_sched_ab.py`, the families' own harness inputs — packers, seeds, desc ABI and launch
geometry of the `gen/*_check.py` scripts; 4 096 rows).** Bits, all 43 kernels, output against the pinned cubin's byte
for byte: 30 warm launches per schedule taking turns in one process on each of three runs (RTX PRO 6000 Blackwell
Server Edition at a core clock of at most 2 362 MHz; Workstation Edition, twice, at up to 2 812 MHz — the second time
the committed files themselves), and fresh processes — 10 on the Server Edition card, 30 on the Workstation card, two
launches per schedule in each, one digest per kernel over schedules, launches and processes: **0 differences**. Time,
the 37 re‑scheduled kernels against their pinned cubins in the same rounds (CUDA events, 3 rounds × 25 launches,
schedules interleaved, E = 64 (expert, token‑group) pairs of random planes), median [range] per family:

| family | kernels | Server Edition card | Workstation card | Workstation card, the committed files |
|---|---|---|---|---|
| 2‑bit decode | 9 | −7.5 % [−10.1 … −3.9] | −6.8 % [−10.8 … −3.4] | −7.4 % [−11.0 … −3.9] |
| 2‑bit prefill, MC = 4 | 5 | −5.9 % [−9.8 … −5.3] | −6.2 % [−6.9 … −5.3] | −6.3 % [−9.6 … −4.9] |
| 2‑bit prefill, MC = 4, AFRAG | 8 | −9.7 % [−13.0 … −3.1] | −9.1 % [−13.4 … −2.1] | −9.2 % [−13.2 … −1.8] |
| FP4 delta | 9 | −6.0 % [−13.1 … −3.2] | −9.0 % [−14.6 … −4.3] | −6.0 % [−14.7 … −3.3] |
| split FP4 (quintal) | 6 | −11.4 % [−14.9 … −5.2] | −13.1 % [−14.5 … −4.0] | −12.6 % [−14.4 … −3.8] |
| **all 37** | 37 | **−8.0 %** [−14.9 … −3.1] | **−8.7 %** [−14.6 … −2.1] | **−7.9 %** [−14.7 … −1.8], faster on 37 / 37 |

Decode shape (E = 8; the 24 re‑scheduled MC = 1 kernels): −9.7 % [−21.6 … −0.4] on the Server Edition card, −10.5 % [−21.5 … +2.6] with the committed files; for 7 of the K ≤ 1 024 kernels
(11–18 µs per launch at this shape) the difference is inside the run‑to‑run spread (−15.1 … +2.6 % over the three runs, the sign not stable).
These are kernel times on the harness; the kernels wait on memory, so the gain is a fraction of the stall cycles removed
(Nsight Compute, `mc4afrag_k4096_a32`, one launch: the fixed‑stall wait falls from 5.71 to 2.27 warp‑cycles per issued
instruction, the long‑scoreboard wait rises from 9.58 to 11.88; 257 → 236 µs). No end‑to‑end number exists for the set.

| cubin (`cubins-sm120-s1/`) | family | schedule | sha256 | stall cycles per warp, pinned → s1 | µs per launch pinned → s1, Server Edition card | Workstation card, the committed files |
|---|---|---|---|---|---|---|
| `moe_w2_mm_k1024_a32.cubin` | 2‑bit decode | **s1** | `52f968497f2c08bc…` | 2 419 → 1 171 | 68.8 → 63.6 (−7.6 %) | 56.9 → 52.7 (−7.4 %) |
| `moe_w2_mm_k512_a32.cubin` | 2‑bit decode | **s1** | `9c5cfc32ec12d748…` | 2 556 → 1 244 | 40.0 → 36.0 (−10.1 %) | 34.5 → 30.8 (−10.8 %) |
| `moe_w2_mm_k384_a32.cubin` | 2‑bit decode | **s1** | `14336b23243813fc…` | 2 727 → 1 336 | 32.5 → 29.9 (−8.2 %) | 28.5 → 26.2 (−7.9 %) |
| `moe_w2_mm_k2048_a32.cubin` | 2‑bit decode | **s1** | `043910133a1bd9cc…` | 4 027 → 1 871 | 133.7 → 126.5 (−5.4 %) | 104.2 → 97.9 (−6.0 %) |
| `moe_w2_mm_k768_a32.cubin` | 2‑bit decode | **s1** | `a9cea1f489eb6de6…` | 4 335 → 2 036 | 44.9 → 40.8 (−9.1 %) | 38.8 → 34.6 (−11.0 %) |
| `moe_w2_mm_k4096_a32.cubin` | 2‑bit decode | **s1** | `8c07cd53086e829e…` | 7 243 → 3 271 | 241.3 → 229.0 (−5.1 %) | 188.0 → 179.6 (−4.5 %) |
| `moe_w2_mm_k3584_a32.cubin` | 2‑bit decode | **s1** | `fd65aef095a16d49…` | 7 489 → 3 413 | 230.9 → 213.7 (−7.5 %) | 183.6 → 167.6 (−8.8 %) |
| `moe_w2_mm_k6144_a32.cubin` | 2‑bit decode | **s1** | `1762fd5ceee995c5…` | 10 459 → 4 671 | 366.0 → 349.8 (−4.4 %) | 288.4 → 276.1 (−4.2 %) |
| `moe_w2_mm_k7168_a32.cubin` | 2‑bit decode | **s1** | `ebc0b65c5638fb87…` | 12 067 → 5 371 | 416.4 → 400.0 (−3.9 %) | 325.6 → 312.9 (−3.9 %) |
| `moe_w2_mm_mc4_k1024_a32.cubin` | 2‑bit prefill, MC = 4 | **s1** | `3629b12effec7814…` | 2 684 → 1 268 | 107.6 → 101.3 (−5.8 %) | 88.0 → 83.7 (−4.9 %) |
| `moe_w2_mm_mc4_k512_a32.cubin` | 2‑bit prefill, MC = 4 | = pinned | `1ba35b0fb6216e50…` | 2 821 (the new schedule: 1 341) | the new schedule: 56.4 → 56.4 (+0.0 %) | the new schedule: 46.2 → 44.7 (−3.2 %) — not taken |
| `moe_w2_mm_mc4_k384_a32.cubin` | 2‑bit prefill, MC = 4 | **s1** | `65432bb581d27f3b…` | 2 992 → 1 433 | 45.0 → 40.5 (−9.8 %) | 38.5 → 34.8 (−9.6 %) |
| `moe_w2_mm_mc4_k2048_a32.cubin` | 2‑bit prefill, MC = 4 | **s1** | `8fe6b77ff7a4eba1…` | 4 516 → 2 048 | 208.4 → 191.7 (−8.0 %) | 165.5 → 155.1 (−6.3 %) |
| `moe_w2_mm_mc4_k768_a32.cubin` | 2‑bit prefill, MC = 4 | **s1** | `2d408d26f50f01bd…` | 4 824 → 2 213 | 69.5 → 65.4 (−5.9 %) | 56.9 → 53.0 (−6.8 %) |
| `moe_w2_mm_mc4_k4096_a32.cubin` | 2‑bit prefill, MC = 4 | = pinned | `7cfa8b42fec44a4e…` | 8 180 (the new schedule: 3 608) | the new schedule: 345.7 → 337.6 (−2.4 %) | the new schedule: 279.4 → 275.6 (−1.4 %) — not taken |
| `moe_w2_mm_mc4_k3584_a32.cubin` | 2‑bit prefill, MC = 4 | **s1** | `45b009aa1bb88539…` | 8 426 → 3 750 | 373.4 → 353.7 (−5.3 %) | 294.6 → 277.3 (−5.9 %) |
| `moe_w2_mm_mc4_k6144_a32.cubin` | 2‑bit prefill, MC = 4 | = pinned | `aa3992fe75978597…` | 11 844 (the new schedule: 5 168) | the new schedule: 519.7 → 527.9 (+1.6 %) | the new schedule: 493.2 → 496.7 (+0.7 %) — not taken |
| `moe_w2_mm_mc4_k7168_a32.cubin` | 2‑bit prefill, MC = 4 | = pinned | `1278270682e6cdc9…` | 13 676 (the new schedule: 5 948) | the new schedule: 597.6 → 601.2 (+0.6 %) | the new schedule: 685.3 → 691.5 (+0.9 %) — not taken |
| `moe_w2_mm_mc4afrag_k1024_a32.cubin` | 2‑bit prefill, MC = 4, AFRAG | **s1** | `9b655b93301f5bae…` | 2 618 → 1 204 | 83.2 → 76.8 (−7.7 %) | 68.7 → 63.3 (−7.9 %) |
| `moe_w2_mm_mc4afrag_k512_a32.cubin` | 2‑bit prefill, MC = 4, AFRAG | **s1** | `fdbdc57c99b49c0e…` | 2 755 → 1 277 | 44.2 → 40.1 (−9.3 %) | 37.9 → 34.5 (−8.9 %) |
| `moe_w2_mm_mc4afrag_k384_a32.cubin` | 2‑bit prefill, MC = 4, AFRAG | **s1** | `6b37f64091b5db2c…` | 2 926 → 1 369 | 36.7 → 31.9 (−13.0 %) | 30.7 → 27.7 (−9.8 %) |
| `moe_w2_mm_mc4afrag_k2048_a32.cubin` | 2‑bit prefill, MC = 4, AFRAG | **s1** | `6c4bc3ab497f4ebb…` | 4 402 → 1 936 | 162.5 → 146.1 (−10.1 %) | 128.5 → 116.2 (−9.6 %) |
| `moe_w2_mm_mc4afrag_k768_a32.cubin` | 2‑bit prefill, MC = 4, AFRAG | **s1** | `957cf8fe8cdfa565…` | 4 710 → 2 101 | 54.4 → 48.8 (−10.3 %) | 44.9 → 40.5 (−9.8 %) |
| `moe_w2_mm_mc4afrag_k4096_a32.cubin` | 2‑bit prefill, MC = 4, AFRAG | **s1** | `59066b0fff420bd5…` | 7 970 → 3 400 | 282.2 → 257.6 (−8.7 %) | 218.8 → 200.8 (−8.2 %) |
| `moe_w2_mm_mc4afrag_k3584_a32.cubin` | 2‑bit prefill, MC = 4, AFRAG | **s1** | `651a3f005f241f8b…` | 8 216 → 3 542 | 260.8 → 226.8 (−13.0 %) | 204.3 → 177.4 (−13.2 %) |
| `moe_w2_mm_mc4afrag_k6144_a32.cubin` | 2‑bit prefill, MC = 4, AFRAG | **s1** | `61a20d1b9f2c5cf9…` | 11 538 → 4 864 | 423.2 → 410.1 (−3.1 %) | 329.2 → 323.1 (−1.8 %) |
| `moe_w2_mm_mc4afrag_k7168_a32.cubin` | 2‑bit prefill, MC = 4, AFRAG | = pinned | `65920010ec01268f…` | 13 322 (the new schedule: 5 596) | the new schedule: 494.9 → 511.6 (+3.4 %) | the new schedule: 463.9 → 462.2 (−0.4 %) — not taken |
| `moe_w4_mm_k1024_a32.cubin` | FP4 delta | **s1** | `8b9c0a1c818b890a…` | 2 686 → 1 269 | 139.9 → 121.6 (−13.1 %) | 108.0 → 92.1 (−14.7 %) |
| `moe_w4_mm_k512_a32.cubin` | FP4 delta | **s1** | `b9d93382851dcae4…` | 2 823 → 1 342 | 51.1 → 48.4 (−5.4 %) | 44.0 → 42.4 (−3.5 %) |
| `moe_w4_mm_k384_a32.cubin` | FP4 delta | **s1** | `a6932eef23dcd71e…` | 2 994 → 1 434 | 44.8 → 42.1 (−6.0 %) | 38.8 → 36.4 (−6.0 %) |
| `moe_w4_mm_k2048_a32.cubin` | FP4 delta | **s1** | `20d77e07d725d994…` | 4 630 → 2 109 | 228.1 → 212.5 (−6.8 %) | 181.8 → 167.5 (−7.9 %) |
| `moe_w4_mm_k768_a32.cubin` | FP4 delta | **s1** | `e28263629b530477…` | 4 938 → 2 274 | 62.6 → 56.4 (−9.8 %) | 53.0 → 47.0 (−11.3 %) |
| `moe_w4_mm_k4096_a32.cubin` | FP4 delta | **s1** | `143af46538ecd63b…` | 8 518 → 3 789 | 411.2 → 394.8 (−4.0 %) | 325.0 → 310.9 (−4.3 %) |
| `moe_w4_mm_k3584_a32.cubin` | FP4 delta | **s1** | `fa160ea73554ce99…` | 8 764 → 3 931 | 390.6 → 354.9 (−9.1 %) | 312.7 → 278.9 (−10.8 %) |
| `moe_w4_mm_k6144_a32.cubin` | FP4 delta | **s1** | `317b1362c21d97db…` | 12 406 → 5 469 | 627.5 → 607.3 (−3.2 %) | 493.0 → 476.6 (−3.3 %) |
| `moe_w4_mm_k7168_a32.cubin` | FP4 delta | **s1** | `2887b7fe28a18ab6…` | 14 350 → 6 309 | 724.5 → 698.1 (−3.6 %) | 568.7 → 548.4 (−3.6 %) |
| `moe_w4_mm_k8192_a32.cubin` | FP4 delta | = pinned | `56f0cd312e8ef1da…` | 16 294 (the new schedule: 7 149) | the new schedule: 789.1 → 790.1 (+0.1 %) | the new schedule: 724.2 → 687.5 (−5.1 %) — not taken |
| `moe_w4q_mm_k1024_a32.cubin` | split FP4 (quintal) | **s1** | `eab90d298bf9a675…` | 4 216 → 2 347 | 164.6 → 140.0 (−14.9 %) | 134.7 → 116.1 (−13.8 %) |
| `moe_w4q_mm_k512_a32.cubin` | split FP4 (quintal) | **s1** | `88fdf2e25d1fe5eb…` | 4 353 → 2 420 | 64.6 → 61.3 (−5.2 %) | 53.2 → 51.2 (−3.8 %) |
| `moe_w4q_mm_k2048_a32.cubin` | split FP4 (quintal) | **s1** | `3f578247a7de3bf6…` | 7 630 → 4 227 | 282.3 → 243.3 (−13.8 %) | 226.8 → 194.1 (−14.4 %) |
| `moe_w4q_mm_k4096_a32.cubin` | split FP4 (quintal) | **s1** | `fa9c1a02f93b0db3…` | 14 458 → 7 987 | 516.9 → 454.4 (−12.1 %) | 415.0 → 359.7 (−13.3 %) |
| `moe_w4q_mm_k6144_a32.cubin` | split FP4 (quintal) | **s1** | `168efc9edad3984d…` | 21 286 → 11 747 | 758.4 → 678.7 (−10.5 %) | 605.7 → 534.2 (−11.8 %) |
| `moe_w4q_mm_k7168_a32.cubin` | split FP4 (quintal) | **s1** | `956addc2e004a00b…` | 24 700 → 13 627 | 876.4 → 781.9 (−10.8 %) | 699.1 → 615.9 (−11.9 %) |

**What the gate does not cover, and what to repeat before a deployment selects the set.** (1) The files are
regenerations in this manifest's sense: the in‑situ validation pass (`tools/test_moe_w2_planes.py` + the GSM8K /
agreement battery) has not been run with them. (2) The table's latencies are counts of issue cycles; the pair floors
measured for these kernels (cubit `tools/stallfloor/`, harness v5e) are the same at core clocks of 2 362 and 2 752 MHz, and
the set was gated at up to 2 812 MHz:
before it runs on cards with a higher core clock, repeat `python3 kernels/gen/moe_sched_ab.py` and
`python3 kernels/gen/moe_sched_ab.py --cold 30` there (exit status 0 = every output identical). (3) Fewer stall cycles
are not always less time: `moe_w2_mm_mc4_k3584_a32` runs 373 / 304 / 354 µs (Server Edition card) and
296 / 251 / 280 µs (Workstation card) as pinned cubin / today's default assembly / this set's file — all three
bit‑identical. The set carries the new schedule for it (it qualifies by the rule); a default re‑assembly of that one
source is the faster file and a regeneration of its own.

Rebuild and check: `CUBIT=<cubit>/target/release/cubit TABLE=<cubit>/tables/sm120.json.gz kernels/gen/build_moe_s1.sh`
(every re‑scheduled file from its source, compared with the committed one; `s1 set: 37 re-scheduled + 6 pinned files; 0 problem(s)`).

## Split FP4 MoE GEMM — moe_w4q_mm (serving: opt‑in `VLLM_MOE_W2_DELTA_SPLIT=1`, default off)
**Bit‑exact** FP4 GEMM reading the resident 2‑bit base plane + a **radix‑5
"quintal" refinement plane at 2.5 bits/elem**: the base code narrows the e2m1
magnitude to the small ({0,.5,1,1.5,2}) or big ({3,4,6}) class, one base‑5
digit per element covers both, and 4 digits pack into a 10‑bit word per QMMA
quad (5⁴ = 625 ≤ 1024). All 16 e2m1 nibbles reconstruct exactly — **zeros
included** (the superseded 2‑bit refinement `moe_w4s_mm` merged mag 0 → 0.5:
3.6% of elements on GLM but **11.6% on DS4**, a measured GSM8K decay growing
with FP4‑pool coverage). Slot = **5/8 of the non‑split nibble plane** → 1.6×
experts/GiB (1.7× over the base cache, where non‑split slots also carry scale
sections). Decode: funnel‑extract → magic /5 ((x·0x334)≫12, exact for x<1024,
brute‑forced in the generator) → selector = digit + 5·is_big → one PRMT over
the full e2m1 magnitude pool → sign from the base code. Desc ABI: 64 B/pair
`{a, as, base, ref, bs, c, m_rows, pad}` (identical to w4s). Generated by
`gen/gen_moe_w4q.py`, validated by `gen/moe_w4q_check.py` (reference = TRUE
e2m1, no merge; worst_rel 2.7–3.8e‑3, deterministic, PASS on RTX 5090; kernel
cost vs `moe_w4s_mm` +11% at K=4096, +6% at K=2048/512 — decode ALU, slot
read bytes 10 B vs 8 B per lane‑k64) and by the HW micro‑probe
`gen/gen_moe_w4q_probe.py` + `gen/moe_w4q_probe_check.py` (byte‑exact decode
of all 16 nibbles).

| cubin (`cubins-sm120/`) | SASS (`sass/`) | kernel | purpose |
|---|---|---|---|
| `moe_w4q_mm_k4096_a32.cubin` | `moe_w4q_mm_k4096_a32.sass` | `moe_w4q_mm` | quintal split FP4, K=4096 (DS4 w13) |
| `moe_w4q_mm_k2048_a32.cubin` | `moe_w4q_mm_k2048_a32.sass` | `moe_w4q_mm` | quintal split FP4, K=2048 (DS4 w2) |
| `moe_w4q_mm_k1024_a32.cubin` | `moe_w4q_mm_k1024_a32.sass` | `moe_w4q_mm` | quintal split FP4, K=1024 (TP2 w2) |
| `moe_w4q_mm_k512_a32.cubin` | `moe_w4q_mm_k512_a32.sass` | `moe_w4q_mm` | quintal split FP4, K=512 (TP4 w2), NWARP=4 |
| `moe_w4q_mm_k6144_a32.cubin` | `moe_w4q_mm_k6144_a32.sass` | `moe_w4q_mm` | quintal split FP4, K=6144 (GLM‑5.x) |
| `moe_w4q_mm_k7168_a32.cubin` | `moe_w4q_mm_k7168_a32.sass` | `moe_w4q_mm` | quintal split FP4, K=7168 (Kimi‑K2.x) |

## Split FP4 MoE GEMM — moe_w4s_mm (LEGACY — superseded by moe_w4q_mm, not loaded by the server)
2‑bit refinement variant (half the slot bytes, but mag 0 merges into 0.5 —
the split‑FP4 zero loss). Kept for history/A‑B tooling: `gen/gen_moe_w4s.py`,
`gen/moe_w4s_check.py` (NOTE: its golden reference CONTAINS the merge),
`gen/moe_w4s_nesting_study.py` (the GLM study that chose the merge). Pack tag
`fp4s` (orphaned; `fp4q` is the live tag). Still the **a128** activation
format (f32 per‑128 scales) — never regenerated for `_a32`.

| cubin (`cubins-sm120/`) | SASS (`sass/`) | kernel | purpose |
|---|---|---|---|
| `moe_w4s_mm_k4096.cubin` | `moe_w4s_mm_k4096.sass` | `moe_w4s_mm` | legacy split FP4, K=4096 |
| `moe_w4s_mm_k2048.cubin` | `moe_w4s_mm_k2048.sass` | `moe_w4s_mm` | legacy split FP4, K=2048 |
| `moe_w4s_mm_k1024.cubin` | `moe_w4s_mm_k1024.sass` | `moe_w4s_mm` | legacy split FP4, K=1024 |
| `moe_w4s_mm_k512.cubin` | `moe_w4s_mm_k512.sass` | `moe_w4s_mm` | legacy split FP4, K=512, NWARP=4 |
| `moe_w4s_mm_k6144.cubin` | `moe_w4s_mm_k6144.sass` | `moe_w4s_mm` | legacy split FP4, K=6144 |
| `moe_w4s_mm_k7168.cubin` | `moe_w4s_mm_k7168.sass` | `moe_w4s_mm` | legacy split FP4, K=7168 |

## 2‑bit MoE GEMM — the core contribution (no upstream equivalent)
| cubin (`cubins-sm120/`) | SASS (`sass/`) | kernel | purpose |
|---|---|---|---|
| `moe_w2_mm_k4096_a32.cubin` | `moe_w2_mm_k4096_a32.sass` | `moe_w2_mm` | 2‑bit MoE GEMM, MC=1 (decode), K=4096 (w13) |
| `moe_w2_mm_k2048_a32.cubin` | `moe_w2_mm_k2048_a32.sass` | `moe_w2_mm` | 2‑bit MoE GEMM, MC=1 (decode), K=2048 (w2) |
| `moe_w2_mm_mc4_k4096_a32.cubin` | `moe_w2_mm_mc4_k4096_a32.sass` | `moe_w2_mm` | MC=4 (prefill, full QMMA‑M), K=4096 — build with `CUBIT_RB_CHAIN=1 CUBIT_SB_SHARE=1` (shipped 340.7 µs; main default 1.37×, switched 1.01×; bits identical) |
| `moe_w2_mm_mc4_k2048_a32.cubin` | `moe_w2_mm_mc4_k2048_a32.sass` | `moe_w2_mm` | MC=4 (prefill), K=2048 |
| `moe_w2_mm_k1024_a32.cubin` | `moe_w2_mm_k1024_a32.sass` | `moe_w2_mm` | 2‑bit MoE GEMM, MC=1, **K=1024** (w2 under TP2) |
| `moe_w2_mm_mc4_k1024_a32.cubin` | `moe_w2_mm_mc4_k1024_a32.sass` | `moe_w2_mm` | MC=4 (prefill), K=1024 |
| `moe_w2_mm_k512_a32.cubin` | `moe_w2_mm_k512_a32.sass` | `moe_w2_mm` | 2‑bit MoE GEMM, MC=1, **K=512** (w2 under TP4), NWARP=4 |
| `moe_w2_mm_mc4_k512_a32.cubin` | `moe_w2_mm_mc4_k512_a32.sass` | `moe_w2_mm` | MC=4 (prefill), K=512 |
| `moe_w2_mm_k6144_a32.cubin` | `moe_w2_mm_k6144_a32.sass` | `moe_w2_mm` | 2‑bit MoE GEMM, MC=1, **K=6144** (gate‑up @ hidden 6144 — **GLM‑5.x**) |
| `moe_w2_mm_mc4_k6144_a32.cubin` | `moe_w2_mm_mc4_k6144_a32.sass` | `moe_w2_mm` | MC=4 (prefill), K=6144 — build with `CUBIT_RB_CHAIN=1 CUBIT_SB_SHARE=1` (shipped 513.4 µs; main default 1.65×, switched 1.01×; bits identical) |
| `moe_w2_mm_k7168_a32.cubin` | `moe_w2_mm_k7168_a32.sass` | `moe_w2_mm` | 2‑bit MoE GEMM, MC=1, **K=7168** (gate‑up @ hidden 7168 — **Kimi‑K2.x**) |
| `moe_w2_mm_mc4_k7168_a32.cubin` | `moe_w2_mm_mc4_k7168_a32.sass` | `moe_w2_mm` | MC=4 (prefill), K=7168 — build with `CUBIT_RB_CHAIN=1 CUBIT_SB_SHARE=1` (shipped 599.8 µs; main default 1.67×, switched 0.98×; bits identical) |
| `moe_w2_mm_mc4afrag_k4096_a32.cubin` | `moe_w2_mm_mc4afrag_k4096_a32.sass` | `moe_w2_mm` | **AFRAG** (prefill, fragment‑major A), K=4096 |
| `moe_w2_mm_mc4afrag_k2048_a32.cubin` | `moe_w2_mm_mc4afrag_k2048_a32.sass` | `moe_w2_mm` | AFRAG (prefill), K=2048 |
| `moe_w2_mm_mc4afrag_k1024_a32.cubin` | `moe_w2_mm_mc4afrag_k1024_a32.sass` | `moe_w2_mm` | AFRAG (prefill), K=1024 (TP2) |
| `moe_w2_mm_mc4afrag_k512_a32.cubin` | `moe_w2_mm_mc4afrag_k512_a32.sass` | `moe_w2_mm` | AFRAG (prefill), K=512 (TP4), NWARP=4 |
| `moe_w2_mm_mc4afrag_k6144_a32.cubin` | `moe_w2_mm_mc4afrag_k6144_a32.sass` | `moe_w2_mm` | AFRAG (prefill), K=6144 (GLM‑5.x) |
| `moe_w2_mm_mc4afrag_k7168_a32.cubin` | `moe_w2_mm_mc4afrag_k7168_a32.sass` | `moe_w2_mm` | AFRAG (prefill), K=7168 (Kimi‑K2.x) |

2‑bit planes = sign‑symmetric `{−4,−1,1,4}` + UE8M0 block‑32 scales; PRMT‑LUT in‑register
decode → `QMMA.SF` tensor‑core. Regcount 64 → 4 CTA/SM.

**AFRAG** (`mc4afrag`) = MC=4 with **fragment‑major activations**: each lane's m16k32 QMMA
A‑fragment loads in ONE `LDG.E.128` instead of 8 strided 4‑byte loads. Prefill moe_w2_mm is
L1/load‑issue bound (DRAM ~36%, L2 hit ~81%), so this cuts the dominant load class ~4× at
identical occupancy — measured **1.30×** (K=4096) / **1.27×** (K=2048) at the production
prefill shape (624 pairs, expert‑sorted, locked clocks). Bit‑identical outputs vs `mc4`
(validated per K incl. determinism; `MOEW2_MC=4 MOEW2_AFRAG=1 gen_moe_w2.py`). The glue
repacks A with a single‑pass triton kernel (`_afrag_repack`, ~65 µs @ [9984,4096]) into
dedicated `a1f/a2f` buffers; default ON via `VLLM_MOE_W2_AFRAG` (set `0` to fall back to mc4).

**Contraction K & multi‑GPU.** K is the per‑cubin GEMM contraction: **4096** = w13/gate‑up (hidden
H, never sharded) and **2048** = w2/down (intermediate I) on a single GPU. Under **tensor parallelism
(TP2)** the w2‑side contraction shards to **K=1024** (the `*_k1024` cubins; w13 stays 4096), and
**TP4** shards it further to **K=512** (the `*_k512` cubins) — the delta tier shards its planes per
rank, so TPn requires the matching K cubins. **Pipeline parallelism (PP)** splits whole layers (each
expert stays intact, no intra‑layer shard) → it reuses the existing K=2048/4096 cubins unchanged.
Generators `gen/gen_moe_w2.py <out.sass> <K>` and `gen/gen_moe_w4.py <out.sass> <K>` emit any K;
**NWARP = split‑K warps is auto‑chosen as K/NWARP must be a multiple of 128 (K≥1024→8, K=512→4)**,
and the loader (`moe_w2_cubit.py::_nwarp_for_k`) launches the matching thread count per K. Op‑validated
by `gen/moe_w2_check.py` / `gen/moe_w4_check.py` (K=512 rel ~2–3e‑3, deterministic; M up to 16;
`_a32` PASS across all K on RTX PRO 6000, AFRAG bit‑exact vs MC=4 per K via `gen/moe_w2_afrag_check.py`).

## FP4 "delta" tier GEMM
| cubin | SASS | kernel | purpose |
|---|---|---|---|
| `moe_w4_mm_k4096_a32.cubin` | `moe_w4_mm_k4096_a32.sass` | `moe_w4_mm` | FP4 (e2m1) hot‑expert delta GEMM, K=4096 |
| `moe_w4_mm_k2048_a32.cubin` | `moe_w4_mm_k2048_a32.sass` | `moe_w4_mm` | FP4 delta GEMM, K=2048 |
| `moe_w4_mm_k1024_a32.cubin` | `moe_w4_mm_k1024_a32.sass` | `moe_w4_mm` | FP4 delta GEMM, **K=1024** (w2 under TP2) |
| `moe_w4_mm_k512_a32.cubin` | `moe_w4_mm_k512_a32.sass` | `moe_w4_mm` | FP4 delta GEMM, **K=512** (w2 under TP4), NWARP=4 |
| `moe_w4_mm_k6144_a32.cubin` | `moe_w4_mm_k6144_a32.sass` | `moe_w4_mm` | FP4 delta GEMM, **K=6144** (gate‑up @ hidden 6144 — **GLM‑5.x**) |
| `moe_w4_mm_k7168_a32.cubin` | `moe_w4_mm_k7168_a32.sass` | `moe_w4_mm` | FP4 delta GEMM, **K=7168** (gate‑up @ hidden 7168 — **Kimi‑K2.x**) |
| `moe_w4_mm_k8192_a32.cubin` | `moe_w4_mm_k8192_a32.sass` | `moe_w4_mm` | FP4 GEMM, **K=8192** (dense wo_b — dense‑FP4 PoC, pairs=1 desc) |
| `moe_w4_mm_k3584_a32.cubin` | `moe_w4_mm_k3584_a32.sass` | `moe_w4_mm` | FP4 GEMM, **K=3584** (latent gate‑up — **Kimi‑K3 native**), **NWARP=7** |
| `moe_w4_mm_k384_a32.cubin` | `moe_w4_mm_k384_a32.sass` | `moe_w4_mm` | FP4 GEMM, **K=384** (w2 @ inter 3072 TP8 — **Kimi‑K3 native**), **NWARP=3** |
| `moe_w4_mm_k768_a32.cubin` | `moe_w4_mm_k768_a32.sass` | `moe_w4_mm` | FP4 GEMM, **K=768** (w2 @ inter 3072 **TP4** — Kimi‑K3), **NWARP=3** |

### Kimi‑K3 TP4 set — K=768 family

K3 under **TP4** shards the w2 contraction to **K=768** (w13 stays K=3584;
per‑rank geometry `1536,3584,3584,768` — the glue's N13/2 == K2 invariant
holds natively). Full `_a32` family shipped: `moe_w2_mm_k768_a32`,
`moe_w2_mm_mc4_k768_a32`, `moe_w2_mm_mc4afrag_k768_a32`,
`moe_w4_mm_k768_a32`. NWARP=3 (KSLICE=256; same non‑pow2 NTHR=96 class as
k384 — generator tail‑pass fix applies). Validated on RTX 5090: op‑level
worst_rel ≤3.5e‑3 deterministic (w2 MC1/MC4, w4), AFRAG bit‑exact vs mc4
0/48; existing k3584 cubins re‑checked at the TP4 w13 shape N=1536 (PASS).
Serving needs `VLLM_MOE_W2_KS=...,3584,384,768` and TP4‑sharded packs
(`k3_build_packs.py --tp 4`).

### Kimi‑K3 native set — removes the +52% padding tax

K3's native per‑rank TP8 geometry is `N13=768, K13=3584, N2=3584, K2=384`
(vs the bring‑up packs zero‑padded to `1024,4096,4096,512`). The full `_a32`
family ships for both new Ks: `moe_w2_mm_k{3584,384}_a32`,
`moe_w2_mm_mc4_k{3584,384}_a32`, `moe_w2_mm_mc4afrag_k{3584,384}_a32`,
`moe_w4_mm_k{3584,384}_a32`. **These are the first non‑pow2 split‑K
geometries** (K=3584 → 28 k128 blocks → NWARP=7; K=384 → NWARP=3): the
generators' epilogue slot‑pass loop (`range(0,256,NTHR)`) previously
over‑ranged past the 256 reduce slots on the tail pass (OOB smem reads +
stores into the next n‑block's columns) — fixed in `gen_moe_w2.py` /
`gen_moe_w4.py` by an unconditional slot clamp (`AND 0xff`) + a tid store
guard on overflowing passes only; pow2 regens stay **byte‑identical**
(verified k512 w2+w4). Validation: op‑level checks at both native shapes
(`moe_w2_check`/`moe_w4_check` worst_rel ≤3.8e‑3, deterministic; AFRAG
bit‑exact vs mc4 0/96) + real‑weight K3 parity at native geometry
(`kimi-k3-analysis/k3_parity_native.py`: silu 1.743e‑02/0.999926, SiTU
2.453e‑02/0.999839, exact‑FP4 1.976e‑02/0.999928 — identical to the padded
run) + full‑model golden battery vs B300 (bit‑identical PASS). Serving
needs `VLLM_MOE_W2_KS=7168,6144,4096,2048,1024,512,3584,384`.

## EXL3 decode-wave M=8-native canons (`exl3-wave-m8/`)

Hand-SASS trellis GEMV canons of the EXL3 decode-wave M∈[2,8] route
(charter M8 + charter CAPTURE-SAFE): each canon decodes its experts'
trellis ONCE and runs HMMA over 8 token rows (`d_a[slot][8][K]` half,
rows ≥M zeroed by the ext stage kernels). Loaded by the exllamav3 ext
(`VLLM_MOE_W2_EXL3_WAVE_M8_CUBIN_DIR` → this directory), grid 768×512,
serving pair base(2,2,1)+Δ(2,2,3). The `*_m8g` set adds the EMPTY-GROUP
GUARD prologue (4th param `d_rmap`, (6,8) int32): a CTA whose expert has
`row_map[e][0] == -1` exits immediately — **rows must be PREFIX-PACKED
from row 0** (hard contract, both group builders guarantee it). The
guarded set is REQUIRED by the capture-safe in-graph route (fixed G=T
groups per captured size); the plain set serves the eager route only.
Generators: `gen_exl3_wave_{uni,cb2,k1,dn3}.py` (lab) + `gen/gen_m8_guard.py`
(guard injector; audits register liveness and instruction counts).

| cubin | kernel | solo cold (µs) | guard cost | empty CTA-set |
|---|---|---|---|---|
| `uni_m8[g].cubin` | `exl3_wave_uni_k2cb0` | 33.14 → 33.58 | +1.33% | 4.16 µs |
| `ucb2_m8[g].cubin` | `exl3_wave_ucb2` | 30.27 → 30.73 | +1.54% | 4.69 µs |
| `k1_m8[g].cubin` | `exl3_wave_k1cb0` | 19.23 → 19.71 | +2.48% | 4.71 µs |
| `dn3_m8[g].cubin` | `exl3_wave_dn3` | 19.28 → 19.75 | +2.41% | 4.71 µs |

Validation: guarded vs plain BIT-exact on full maps, sentinel-intact on
empty maps, partial-map mixes ✓ (real l00 packs, GPU 3); full matrices of
the plain canons per DESIGN §5r/§5s (Gate1/Gate2b/degenerates/det/A-B);
layer M=8 = 13.88 µs/tok (6.9× vs wave M=1); in-graph route: capture +
replay with swapped routing rel ≤8.1e-8 vs eager recompute; serving
boot-cert PIECEWISE 16/16 + FULL 8/8 with the route captured; paired
quality GSM8K-200/GPQA-198 NT: no significant regression (McNemar
p=0.500/1.000). Empty ext-call 16.2 µs (stage early-outs under guarded).

### GLM family (K=6144, base-only)

Regenerated canons for the GLM-5.2 trellis geometry (H=6144, I=2048,
serving pair base (2,2,1) cb0 — **no delta side stream**: the ext skips
ucb2/dn3 and the glue keeps the delta buffer quadrants zeroed +
zero-svh'd; DESIGN §5t/§5t′/§5t″). k1@N=6144 has GROUPS=192 (non-pow2):
the CTAID decode uses an exact magic-div (`(x>>6)*0xAAAB>>17`, grid 1152
= 3.06 waves — the fuller wave amortizes the tail, +2% vs the DS4
canon despite 1.5× columns). Guard variant for it: `div192` mode of
`gen/gen_m8_guard.py`. Generators: parameterized `gen_exl3_wave_{uni,k1}.py`
(lab; `UNI_KDIM`/`K1_NDIM`, defaults regenerate the DS4 canons
byte-identically — verified 3×).

| cubin | kernel | solo cold (µs) | grid |
|---|---|---|---|
| `uni_m8[g]_k6144.cubin` | `exl3_wave_uni_k2cb0` | 38.11 (rot8 @1680; gate ≤55) | 768 |
| `k1_m8[g]_n6144.cubin` | `exl3_wave_k1cb0` | 20.12 (rot24 @1680; gate ≤32) | 1152 |

Validation: Gate1 vs numpy oracle rel ~2.07-2.08e-4 (+ M=1 lab reference
at N=6144: degenerate row0 BIT-identical, rows 1-7 bit-zero); real GLM
packs l03+l50 4/4 PASS (spread experts, w1/w3/w2); det ×2/×3 everywhere;
base-only layer window 96.1 µs = 12.02 µs/tok @M=8 (burst-first-block,
SETS=4/12); ext seam parity vs lab wrapper 4.5e-8, map linearity incl.
empty expert through the guard 4.2e-8, d_in BIT-det (out: fp32
atomic-order 4.7e-10 — the route's only atomic surface). In-serving
(GLM-5.2 TP4 resident + MTP k=2): route ACTIVE, C1-MTP code 53-67 tok/s
(mgemm verify: 36-44), C2 103-104 agg, C4 parity (verify T=12 is above
the M∈[2,8] window by design). Guards runtime-certified in-serving via
the captured graph route (fixed G=T groups). Paired quality GSM8K-200
(P6, same tool/config/items, OFF→ON): 93.0% → 93.5%, discordants 4/5,
McNemar exact p=1.000 — no significant regression.

### GLM family M=16 (dual-set with M8)

M=16-native canons widen the decode-wave window to **T∈[2,16]** — the
HMMA tile is m16n8k16 and the m8 canons zeroed A-fragment rows 8-15;
m16 fills the existing zeros with token data (hot loop cycle-identical
to m8: 72.5/95.4 cyc/tile — 8 extra token rows cost zero additional
HMMA and zero additional trellis decode). The flat row-8-15 offset does
not fit the 12-bit LDGSTS immediate (and uni has no free register
pair), so the A side is **INTERLEAVED**:
`d_a16[slot][r%8][k//2][r//8][k%2]` — pairs {a01,a89}/{a23,a1011} land
as 2× `LDGSTS.E.64` per slice (same instruction count, same
barrier/pending discipline as the m8 cert). The epilogue folds rows
8-15 through the epilogue-dead ping-pong region (+A_OFF): **zero
additional BARs** vs m8, warp early-exit kept. ABI: `d_rmap` (6,16)
int32 for the guarded set (`rows16` mode of `gen/gen_m8_guard.py`,
stride 64 B; `div192 rows16` for k1). Group contract:
**G = ceil(k·T/6)** — the previous G=T was a top-6 invariant and can
silently overflow the slot buffers at top-8 (the fix ships with the
glue and applies retroactively to the m8 route). Served as a
**dual-set** with m8: the ext loader holds both, the glue picks m8 for
T∈[2,8] (m16 pays ~15%/tok at M=8) and m16 for T∈[9,16].

| cubin | kernel | solo cold (µs) | grid |
|---|---|---|---|
| `uni_m16[g]_k6144.cubin` | `exl3_wave_uni_k2cb0` | 45.63 (rot8 @1680; gate ≤46) | 768 |
| `k1_m16[g]_n6144.cubin` | `exl3_wave_k1cb0` | 24.55 (rot24 @1680; gate ≤25) | 1152 |

Validation: Gate1 vs numpy oracle rel 2.06-2.09e-4 (×2 kernels ×2
runs); degenerate (rows 8-15 zero in): rows 0-7 **BIT-identical to the
m8 canon** + rows 8-15 bit-zero out; real GLM packs l03+l50 4/4 PASS;
det ×2/×3 everywhere; m8 baselines reproduced on the same card
(37.99/20.12 µs); per-token −40%/−39% vs m8 (2.852/1.534 µs/tok @M=16).
Guards m16g: +0.34%/+1.03% full map, empty map ~5.2 µs. Base-only
layer window: M=16 = 7.34 µs/tok, M=12 = 9.63 µs/tok (the C4+MTP
verify shape; layer stays flat in M — M=1→16 = +10%). Ext seam parity
3/3 (5.5e-8; map linearity incl. empty expert 4.8e-8; d_in16 BIT-det,
out fp32 atomic-order 4.7e-10). Builder tests 47+79 GREEN incl. the
top-8 OOB regression (U=24 @T=3 covered by G=4). In-serving (GLM-5.2
TP4 resident + MTP k=2, dual-set m8+m16): **C4-MTP agg 59-60 →
194-203 tok/s (3.3×)** — verify T=12 moves off the mgemm union tax;
C1/C2 within the ±3% gate (the G-fix empty groups cost ~2-2.5% at
T≤8); cold prefill untouched (554 vs 559). Paired quality GSM8K-200
**at C4** (fresh boots per arm; the m16 route never fires at C1),
**r1/r2/r3 with same-config floor controls**: OFF 94.0/93.0/93.5% vs
ON 91.0/94.0/91.5%; same-config pairs (OFF×OFF, ON×ON) flip 9-15
items/pair at C4 batching — the near-tie drift floor — and every
cross-arm pair (7-14 flips, McNemar p=0.146-1.0) sits inside that
band, with the r1 asymmetry (9:3) reversed in r2 (4:6). **No
detectable quality regression.** Boot note: graph capture at T=12
needs deterministic KV headroom — pin `--kv-cache-memory-bytes`
≈2 GiB/rank at the 8k window; adding exact verify sizes {3,6} to
`cudagraph_capture_sizes` trims the G-fix empty groups at C1/C2
(G 6→4 / 11→8) at zero numerical delta (pad rows are weight-0).

### GLM family M=16×4 — prefill-native

Prefill-native canons amortize the trellis decode over **4 slabs of
16 rows** (64 tokens per expert slot): 16 warps = 4 k-splits × 4 slab
groups, the decode runs at the m16 canon rate while HMMA covers 4×
the rows; decoded fragments are shared through smem with UNIFORM-REG
addressing, accumulators stay slab-private in registers. The A-side
ABIs carry the rotation in the LAYOUT (`d_a16x4` per-slab slice
rotation; `d_a16x4k1` slice-pair swap) because LDGSTS has no UR
addend on the global side. Consumed by the expert-major prefill route
(`VLLM_MOE_W2_EXL3_WAVE_PREFILL=1`, eager chunks above the graph
envelope; ext `exl3_decode_wave_x4_layer`, 5 launches per group of 6
experts). Generators: `internal` wave-sass-lab
`gen_exl3_wave_uni_x4.py` (X4_DIRECT=1 X4_ADV2=1 X4_PRE=1) and
`gen_exl3_wave_k1_x4.py`.

| cubin | kernel | solo cold µs (lock-1680 class¹) | grid |
|---|---|---|---|
| `uni_m16x4d_k6144.cubin` | `exl3_wave_uni_k2cb0` | 134.27 (rot8 @1680; ≤0.6×4×anchor PASS 0.5975) | 768 |
| `k1_m16x4_n6144.cubin` | `exl3_wave_k1cb0` | 74.63 (rot24 @1680; 0.5646) | 1152 |

¹ Absolute numbers measured at a VERIFIED sm-lock 1680 under load
(older certification absolutes were measured at boost clocks).

Validation: Gate1 oracle rel 2.05-2.09e-4 (both kernels, ×5/×5 runs);
degenerate per slab: out-of-slab rows bit-zero ×4/×4; real GLM packs
l03+l50 24/24 + 12/12; det ×10+ at 2 CTA/SM; CUBIT_VERIFY 0/0/0.
Layer window @chunk 2048 top-8 on real planes: **2.21× MoE-only vs
the exl3_moe_dual equivalent** (28.9 → 13.1 ms/layer; 2.09× with the
stage kernels). Route seam parity vs the moe_dual route 3/3 (rel
6.6e-4; scatter repeat-band 1e-9 class). In-serving boot-cert
(GLM-5.2 TP4 + MTP k=2, fresh boots per arm): **cold prefill
2051-2152 tok/s vs the line's moe_dual 870-881 = 2.33-2.44×**
(vs the paired same-recipe baseline: 4.8×); needle 4/4 both arms;
paired prompt-logprob KLD sits exactly at the cross-boot floor
(0.2887/5.45% flips vs a moe_dual↔moe_dual cross-boot control at
0.2884/5.67%); 15-min mixed soak (~890k tokens) clean with the KLD
still at the floor. Ops notes: derived pointer tables must be rebuilt
per call on paged-base cells (stale-pointer incident caught by the
needle probe and fixed — fork `a913b0a07`); a two-stream group
launcher measured a wash (13.61 vs 13.82 ms — co-residency eats the
gap-fill) and the launcher stays single-stream.

## Sparse‑MLA prefill (SM120)
| cubin | SASS | kernel | purpose |
|---|---|---|---|
| `mla_prefill_state.cubin` | `mla_prefill_state2.sass` | `mla_prefill_state2` | prefill accumulate, fp8 smem staging, 2 CTA/SM (~2.2× vs Triton) |
| `sparse_mla_prefill_v41.cubin` (sha256 c896234bc76d6938…, 49 128 B; assembled with cubit 53e9ed8 + BUG-421/422 = main's BUG-478/479 (`ed4d316`), `CUBIT_SB_SHARE=1 CUBIT_RB_EARLY=32 CUBIT_MMA_WAIT=1 CUBIT_MMA_WB_DRAIN=13`; strict round trip with cubit main 2 663 / 2 663) | `sparse_mla_prefill_v41.sass` (generator `gen/gen_sparse_mla_prefill_v41.py`) | `sparse_mla_prefill_v41` | DeepSeek‑V4.1 sparse‑MLA prefill attention (step 40 class), **bit‑identical** (`torch.equal`, output + lse) with FlashInfer 0.7.2 `sparse_mla_prefill_dsv41_bf16_kernel<16,true>` on the synthetic suite (8 cases: fp8 SWA + FP4 compressed caches, partial lengths, −1 holes, page sizes 32/64/128, sink/no sink, empty rows); 1 token × 16 heads per CTA, 8 warps, 87 040 B smem, 1 CTA/SM, register prefetch of the next 64‑candidate chunk; 0.66–0.72× of the reference time on GPU 4 (845/953/2466/981 µs vs 1290/1447/3547/1370 for 128+512 p128 / 192+512 / 1152+512 / 128+512 p64, 4096 rows). Assemble with `tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh` — needs cubit `main` at or after `ed4d316` (V's fixes = BUG‑478/479/480 there; the script's defaults point at `/workspace/cubit` since 2026‑11‑05 and the cubin reproduces byte for byte from `main` up to cubit `52767f9`; from `f064bed` (BUG‑485, the allocator's WAW rule) `main` emits the same instruction text with **two extra wait control words** (`@P0 VIMNMX R18/R19`) — the shipped cubin stays pinned to this digest, a rebuild would need the in‑situ A/B again). **Reproduction contract (cubit KNOWN_ISSUES A20, 2026‑11‑06):** a shipped cubin is reproduced byte for byte from itself — `cubit disassemble --frozen <shipped.cubin> -o k.sass && cubit asm -T <shipped.cubin> k.sass -o rebuilt.cubin && cmp` — with any cubit ≥ 0.2.0 (the frozen path is gated 157/157 on every cubit push); re‑assembling the `.sass` source is a *regeneration* whose control codes follow the current allocator and is a new schedule that needs in‑situ validation before it ships and `CUBIT_SB_SHARE=1 CUBIT_RB_EARLY=32 CUBIT_MMA_WAIT=1 CUBIT_MMA_WB_DRAIN=13`. Harness `tools/dsv41_sm120/sparse_mla_sass/test_sparse_mla_sass_sm120.py` (`--synthetic`, `--time`, dumps from the real prefill via `patch_vllm_sparse_mla_prefill_dump_sm120.py` + `replay_specB_dumps.py`). **Dumps from the real prefill: 8/8 bit‑identical (19K/163K, layers 0 and 2). In situ (4× RTX PRO 6000 SE, r23 image + `VLLM_MOET_SPARSE_MLA_PREFILL_SASS=1`, 2026‑11‑05): fresh prefill 163K 7.70 → 7.30 s (−5.2 %), 19K 905 → 860 ms (−5.0 %), long_agree ×2 + agree ALL IDENTICAL, loaded on 4 ranks, 0 disabled, 0 Traceback.** Dockerfile step 46 (image `-r25`). |
| `sparse_mla_prefill_v41s1.cubin` (sha256 5924622b6289fca4…, 47 160 B; **the same source and the same instruction words as `sparse_mla_prefill_v41.cubin`, a new schedule**: cubit `main` at `38e4cc8` (the scheduler change is `49f8340`), the policy of the row above plus `CUBIT_SCHED2=1 CUBIT_SCHED2_LAT=sm120 CUBIT_MMA_PTXAS=1`; 2 540 instructions — the 128 MMA writeback drains of the default schedule are gone, 5 NOP pads; strict round trip 2 540 / 2 540, the frozen listing re‑assembles byte for byte) | `sparse_mla_prefill_v41.sass` (the same source and generator) | `sparse_mla_prefill_v41` | The dependency‑aware stall schedule of the same kernel: a stall is charged where a consumer needs it instead of behind every producer (cubit's pair‑latency table for sm_120 — ptxas's practice on 1.8 M instructions, measured on RTX PRO 6000 silicon — enforced on every control‑flow path; ptxas's issue distances for the HMMA chain instead of writeback drains). Stall cycles per token‑warp at 128+512: 67 807 → 35 555 (−48 %), issued instructions 10 928 → 9 698. **Bit‑identical** (`torch.equal`, output + lse) with FlashInfer's `sparse_mla_prefill_dsv41_bf16_kernel<16,true>` and with `sparse_mla_prefill_v41.cubin`: the 8 synthetic cases and the 8 dumps of the real prefill, each on 30 repeated launches in one process and in 30 fresh processes, the default cubin as the control in the same processes — 0 differences. **Harness, 4096 rows, µs per launch with the default cubin in the same rounds: 898 / 986 / 2 162 / 864 vs 1 058 / 1 153 / 2 503 / 1 025 (128+512 p128 / 192+512 / 1152+512 / 128+512 p64) = −15.1 / −14.5 / −13.6 / −15.7 %** (the reference kernel 1 583 / 1 753 / 4 005 / 1 556). Nsight Compute, one launch at 128+512: the fixed‑stall wait falls from 5.34 to 2.78 warp‑cycles per issued instruction of ≈ 10; the tensor pipe and the scoreboards take up part of it — lowering the MMA distances further (below ptxas) is still bit‑identical and buys ≤ 0.5 %, so this is where stall rules end. **In situ (4× RTX PRO 6000 SE, the `-r28` image as built with this cubin bind‑mounted and `VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN` pointing at it, 2026‑11‑10, against the same image with the default cubin): fresh prefill 163K 5 175 → 5 068 ms (−2.1 %), 19K 655 → 634 ms (−3.2 %), at a 300 W cap 163K 5 818 → 5 703 ms (−2.0 %); long_agree ×2 + agree ALL IDENTICAL against three references, 400 chunk‑layer dumps × 5 states on 4 ranks ALL IDENTICAL, loaded on 4 ranks, 0 disabled, 0 Traceback; the prefill‑under‑load gate 163K +4.7 % (≤ +5 %; the default cubin +4.6 %).** Not loaded by default and not copied into the image: the default stays `sparse_mla_prefill_v41.cubin` with the Dockerfile's digest check; select this one at run time with `VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN=<path>` (same kernel name, same launcher). Build: `SCHEDULE=s1 tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh` (reproduces this digest). The table's latencies are counts of issue cycles measured at a core clock of at most 2 430 MHz: before the cubin runs on cards with a higher core clock, repeat `test_sparse_mla_sass_sm120.py --cubin <this> --synthetic` and the dump replay there. |
| `sparse_mla_prefill_v42.cubin` (sha256 dfdc0279ea5a3388…, 29 784 B; cubit `main` at `db4cfd1` — the instruction's table row is `89564ac` —, the policy and the schedule switches of `sparse_mla_prefill_v41s1.cubin`: `CUBIT_SB_SHARE=1 CUBIT_RB_EARLY=32 CUBIT_MMA_WAIT=1 CUBIT_MMA_WB_DRAIN=13 CUBIT_SCHED2=1 CUBIT_SCHED2_LAT=sm120 CUBIT_MMA_PTXAS=1`; 1 454 instructions, `CUBIT_VERIFY` 0 unprotected RAW / 0 broken WAR, `CUBIT_DEP_VERIFY` 0; strict round trip 1 454 / 1 454, the frozen listing re‑assembles byte for byte) | `sparse_mla_prefill_v42.sass` (generator `gen/gen_sparse_mla_prefill_v42.py`: v41's generator with a new `emit_dequant`; `SMLA_DEQUANT=v41` reproduces `sparse_mla_prefill_v41.sass` byte for byte) | `sparse_mla_prefill_v41` (the entry symbol and the 22 parameters are v41's: the cubin loads through the same launcher, `VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN=<path>` or as the second cubin of the runtime A/B key) | **v41 with the dequant of the staged records done by sm_120's scaled BF16 unpack** — `F2FP.BF16.<E4M3\|E2M1>.UNPACK_B Rd, Rs[.sel], Rscale.B0`: two codes of `Rs` → two bf16 halves, each times 2^(byte 0 of `Rscale` − 127). The instruction is named by nvdisasm for SM120 and executes there; ptxas does not emit it for this target (0 of the 1.8 M instructions of the sm_120 census). fp8 chunk: e4m3 pair × UE8M0 block scale in one instruction, the scale byte straight from its staging register (v41: unpack to f16, 2 × widen to f32, pack to bf16, `HMUL2` by the unpacked scale) — 44 → 10 instructions per task. FP4 chunk: e2m1 pair → bf16 with the unit scale code, the four bytes of a word through the source's byte selector instead of three shifts, the E4M3 block scale by one more scaled unpack, then v41's `HMUL2.BF16_V2` — 55 → 21. A task's store data rotates over two register sets (a store latches its data when the memory pipe serves it). Everything else is v41's text. Issued instructions per token‑warp at 128+512: 9 698 → 6 986 (−28 %); stall cycles 35 555 → 30 211. **Exactness of the new sequences on silicon** (`tools/dsv41_sm120/sparse_mla_sass/probe_dequant_unpackb_sm120.py`, v42's instructions against v41's on every input, RTX PRO 6000): 65 536 e4m3 pairs × 256 scale bytes in either half of the source word, 256 e2m1 bytes × 256 E4M3 block‑scale codes through each of the four byte selectors, the block scale alone on all 256 codes — all equal, with junk in every unused byte, NaN codes and −0 included. **Bit‑identical** (`torch.equal`, output + lse) with FlashInfer's `sparse_mla_prefill_dsv41_bf16_kernel<16,true>`: the 8 synthetic cases and the 8 dumps of the real prefill, each on 30 repeated launches in one process and in 30 fresh processes, `sparse_mla_prefill_v41s1.cubin` as the control in the same processes — 0 differences; the runtime A/B helper with the default cubin and this one in either role 17 / 17. **Harness, 4096 rows, µs per launch in the same rounds: 833 / 918 / 2 051 / 805 vs `v41s1` 903 / 991 / 2 182 / 872 vs the default cubin 1 060 / 1 154 / 2 506 / 1 025 (128+512 p128 / 192+512 / 1152+512 / 128+512 p64) = −7.8 / −7.4 / −6.0 / −7.7 % against `v41s1`, −21.4 / −20.5 / −18.2 / −21.5 % against the default** (the reference kernel 1 605 / 1 753 / 4 035 / 1 568). Variants of the generator in the same rounds, all bit‑identical: one register set 841 / 927 / 2 079 / 812, three sets 862 / 952 / 2 137 / 835, v41's shifts instead of the byte selectors 868 / 958 / 2 148 / 839 (three sets), the shifts and v41's four‑instruction block scale 911 / 1 005 / 2 228 / 883 (three sets). The instruction's latencies under cubit's table: measured on sm_120 silicon as a producer and as a consumer (cubit `tools/stallfloor/stall_floors_sm120_20261110_pairs_d.json`) — the floors of the plain unpack / pack forms, at or below the cells that price them. **Not measured in situ yet; not loaded by default and in no image.** Build: `KERNEL=v42 tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh` (reproduces this digest). As for `v41s1`: the table's latencies are counts of issue cycles measured at a core clock of at most 2 430 MHz — before the cubin runs on cards with a higher core clock, repeat `test_sparse_mla_sass_sm120.py --cubin <this> --synthetic` and the dump replay there. Correction to "0 broken WAR": 4 store → overwriter pairs carry no wait on one path ([record](../docs/benchmarks/sparse-mla-prefill-v43c-2026-11-10.md)). |
| `sparse_mla_prefill_v43.cubin` (sha256 43f04db5cc9e9874…, 30 344 B; cubit at `df4c557` — `main` `ef7d335` plus the opt‑in colouring rule `CUBIT_SB_LONG_KEEP` —, the switches of `sparse_mla_prefill_v42.cubin` and `CUBIT_SB_LONG_KEEP=1`; 1 489 instructions, `CUBIT_VERIFY` 0 unprotected RAW / 0 broken WAR, `CUBIT_DEP_VERIFY` 0; strict round trip 1 489 / 1 489, the frozen listing re‑assembles byte for byte) | `sparse_mla_prefill_v43.sass` (generator `gen/gen_sparse_mla_prefill_v43.py`: v42's generator with two switches; `SMLA_IDX_PREFETCH=0 SMLA_PROLOGUE_OVERLAP=0` reproduces `sparse_mla_prefill_v42.sass` byte for byte, with `SMLA_DEQUANT=v41` also `sparse_mla_prefill_v41.sass`) | `sparse_mla_prefill_v41` (the entry symbol and the 22 parameters are v41's: the cubin loads through the same launcher, `VLLM_MOET_SPARSE_MLA_PREFILL_SASS_CUBIN=<path>` or as the second cubin of the runtime A/B key) | **v42 with two load waits taken out — the same values, the same arithmetic, the same order; loads are issued earlier.** A per‑instruction profile of v42 on real prefill inputs (ncu, 128+512 candidates) shows three places where warps wait for something other than arithmetic; v43 removes two. **(1) The candidate index one chunk ahead (10.6 % of the samples):** v41 / v42 load the candidate index of chunk c+1 at the top of the loop and store it into `valid[]` at once — warps 0–1 wait for a load of the token's own index row (never cached, ≈ 800 cycles) and the other six wait for them at the barrier. v43 issues the load for chunk c+2 behind the gather loads of chunk c+1 (after the barrier, branch‑free, the `valid[]` threads selected by the load's predicate), reads it once in front of the dequant and stores it at the next loop top; registers R126–R129, regcount 136 (one CTA per SM as before: the 87 KB of shared memory decide that). **(2) The prologue's global loads back to back (4.9 %; more on a two‑chunk layer):** the two length loads and the four `q` loads are issued together and stored afterwards — v41 / v42 ran six load latencies in a row per token. The third place (store‑latch waits in the FP4 dequant, 5.7 %) was tried with the outputs in registers no pending store reads: no gain, left out. **Built with `CUBIT_SB_LONG_KEEP=1`:** scoreboards are counters, and under cubit's sharing policy other uses ended up on the scoreboard of the gather prefetch (in v41s1 / v42 the fifth `LDSM` of QK waits for all sixteen gather loads); with the prologue change the gather moved to a scoreboard the first QK instruction waits on. The rule gives long loads a scoreboard of their own. The same text without it is legal and bit‑identical, 1–4 % slower on real inputs and 5 % slower when every gather misses the cache. **Bit‑identical** (`torch.equal`, output + lse) with FlashInfer's `sparse_mla_prefill_dsv41_bf16_kernel<16,true>`: the 8 synthetic cases and the 8 dumps of the real prefill, each on 30 repeated launches in one process and in 30 fresh processes, `sparse_mla_prefill_v42.cubin` as the control in the same processes — 0 differences, on two RTX PRO 6000 cards (Server Edition at a core clock of 2.4 GHz, Workstation Edition at 2.7–2.8 GHz). **Real prefill inputs, 4096 rows, µs per launch in the same rounds, reference kernel / default cubin / `v41s1` / `v42` / this — 128+512 candidates: 1 360 / 980 / 834 / 775 / 681; 126+448: 1 237 / 897 / 765 / 711 / 625; 128+0: 298 / 269 / 240 / 224 / 200 = −12.1 / −12.1 / −10.7 % against `v42`, −30.5 / −30.3 / −25.8 % against the default** (Server Edition); on the Workstation card 1 132 / 809 / 686 / 637 / 550, 1 027 / 740 / 629 / 584 / 507 and 244 / 221 / 195 / 183 / 160 (−13.7 / −13.2 / −12.6 % against `v42`, −32.0 / −31.5 / −27.4 % against the default). Harness shapes (random candidate indices — every gather a cache miss; real candidates are a sliding window and cost 7–8 % less): 730 / 808 / 1 750 / 693 vs `v42` 834 / 890 / 1 963 / 800 vs the default 1 060 / 1 154 / 2 448 / 1 021 (128+512 p128 / 192+512 / 1152+512 / 128+512 p64). Variants of the generator in the same rounds, all bit‑identical, real 128+512 input: the index prefetch alone 714, with the four `q` loads 691, this text without `CUBIT_SB_LONG_KEEP` 689. **Not measured in situ yet; not loaded by default and in no image.** Build: `KERNEL=v43 tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh` (reproduces this digest; refuses a cubit without the rule). As for `v41s1` and `v42`: the table's latencies are counts of issue cycles measured at a core clock of at most 2 430 MHz and re‑measured unchanged on cards running at up to 2.9 GHz — before the cubin runs on cards with a higher core clock, repeat `test_sparse_mla_sass_sm120.py --cubin <this> --synthetic` and the dump replay there. Correction to "0 broken WAR": 3 store → overwriter pairs carry no wait on one path ([record](../docs/benchmarks/sparse-mla-prefill-v43c-2026-11-10.md)). |
| `sparse_mla_prefill_v43b.cubin` (sha256 a163aea6d869c55e…, 30 344 B; **the source and the instruction words of `sparse_mla_prefill_v43.cubin`, one control word different**; cubit at `df4c557` or later with `v43`'s switches and one more row in its dependency table — an MMA holds every `BRA` 14 issue cycles off — which the build script adds to a copy of cubit's `tables/dep_latency_sm120.json` and passes as `CUBIT_SCHED2_LAT`; 1 489 instructions, `CUBIT_VERIFY` 0 / 0, `CUBIT_DEP_VERIFY` 0 against that table; strict round trip 1 489 / 1 489, the frozen listing re‑assembles byte for byte) | `sparse_mla_prefill_v43.sass` (the same generator and text as `v43`) | `sparse_mla_prefill_v41` | **`v43` with the last `HMMA` of PV at least 14 issue cycles in front of the two conditional branches behind it** (`@!P5 BRA` to the epilogue, `@P6 BRA` to the FP4 dequant): 14 and 20 cycles instead of 7 and 13. The one word: `BAR.SYNC` behind PV, stall 6 → 13. Why: ptxas never issues a conditional branch closer than 14 cycles behind an MMA (166 sites of the MoE kernels it compiled for this stack, 53 at exactly 14) — the result of a warp‑wide MMA is to land before an instruction that can change the warp's set of threads. In the default cubin the writeback drains kept 39 and 44 cycles there; the dependency‑aware schedule removed the drains (`v41s1`, `v42`, `v43`: 7 and 12–13), and cubit's table had no row to put the distance back. Both branches are uniform over the CTA (their predicates depend on the token's chunk counter only) and `v41s1` / `v42` / `v43` are bit‑identical in every gate, so no failure was observed at 7; `v43b` is the build that does not rely on that. **Cost: 7 issue cycles per chunk — not measurable:** real prefill inputs, 4096 rows, µs per launch in the same rounds, `v43` / `v43b` (RTX PRO 6000 Workstation Edition, core clock 2.7 GHz): 128+512 candidates 554 / 554, 126+448 509 / 509, 128+0 161 / 161 (the default cubin 812 / 742 / 221, `v42` 643 / 589 / 183); harness shapes 586 / 647 / 1 469 / 607 vs 585 / 646 / 1 468 / 607. **Bit‑identical** (`torch.equal`, output + lse) with the reference kernel: the 8 synthetic cases and the 8 dumps of the real prefill, 30 repeated launches in one process and 30 fresh processes each, `v43` as the control — 0 differences. Equivalent to `v43` word for word (1 489 of 1 489 instruction texts; stall fields: one differs). **Not measured in situ (the in‑situ numbers of `v43` are of the same instruction stream); not loaded by default.** Build: `KERNEL=v43b tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh` (reproduces this digest). The clock note of `v43` applies. Correction to "0 broken WAR": 3 store → overwriter pairs carry no wait on one path ([record](../docs/benchmarks/sparse-mla-prefill-v43c-2026-11-10.md)). |
| `sparse_mla_prefill_v41s1b.cubin` (sha256 cba10b0adafd20c3…, 47 160 B; **the source and the instruction words of `sparse_mla_prefill_v41s1.cubin` — those of the default cubin —, one control word different**; cubit `main` at `ef7d335` or later with `v41s1`'s switches and one more row in its dependency table — an MMA holds every `BRA` 14 issue cycles off, the row `sparse_mla_prefill_v43b.cubin` is built with — which the build script adds to a copy of cubit's `tables/dep_latency_sm120.json` and passes as `CUBIT_SCHED2_LAT`; 2 540 instructions, `CUBIT_VERIFY` 0 / 0, `CUBIT_DEP_VERIFY` 0 against that table; strict round trip 2 540 / 2 540, the frozen listing re‑assembles byte for byte) | `sparse_mla_prefill_v41.sass` (the same source and generator) | `sparse_mla_prefill_v41` | **`v41s1` with the last `HMMA` of PV at least 14 issue cycles in front of the two conditional branches behind it** (`@!P5 BRA` to the epilogue, `@P6 BRA` to the FP4 dequant): 14 and 19 cycles instead of 7 and 12. The one word: `BAR.SYNC` behind PV, stall 6 → 13 — the word and the reason of `v43b` (ptxas never issues a conditional branch closer than 14 cycles behind an MMA). This is the dependency‑aware schedule of the default kernel's own instruction words that does not go below that practice: no new instruction form, no load issued earlier, 128 registers. **Cost: 7 issue cycles per chunk — not measurable:** harness shapes, 4096 rows, µs per launch in the same rounds, `v41s1` / `v41s1b` (RTX PRO 6000 Workstation Edition, core clock 2.7 GHz): 731 / 801 / 1 751 / 753 vs 732 / 801 / 1 754 / 754 (the default cubin 870 / 948 / 2 079 / 890, `v43b` 585 / 645 / 1 467 / 612); real prefill inputs: 128+512 candidates 688 / 688, 126+448 630 / 629, 128+0 196 / 195 (the default 811 / 741 / 221, `v43b` 553 / 508 / 160). **Bit‑identical** (`torch.equal`, output + lse) with the reference kernel: the 8 synthetic cases and the 8 dumps of the real prefill, 30 repeated launches in one process and 30 fresh processes each, `v41s1` as the control — 0 differences, on two Workstation cards. Equal to `v41s1` word for word (2 540 of 2 540 instruction texts; one control field differs: one byte of the file). **Not measured in situ (the in‑situ numbers of `v41s1` are of the same instruction stream); not loaded by default and in no image.** Build: `SCHEDULE=s1b tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh` (reproduces this digest). The clock note of `v41s1` applies. |
| `sparse_mla_prefill_v43c.cubin` (sha256 7e812fa2d9c47770…, 30 344 B; `sparse_mla_prefill_v43b.cubin` with one control word changed; build: `KERNEL=v43c tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh`) | `sparse_mla_prefill_v43.sass` (the same generator and text as `v43`) | `sparse_mla_prefill_v41` | `v43b` with every store of the dequant waited before a register it reads is rewritten, on every path: the branch behind the loop‑top `BAR.SYNC` waits on scoreboards 3, 4 and 5. The instruction words, stalls and time of `v43b`; bit‑identical with the reference kernel; not loaded by default and in no image. Gate, probe and times: [record](../docs/benchmarks/sparse-mla-prefill-v43c-2026-11-10.md). |

> Only the kernels the server actually loads are shipped (verified against the server's load logs). The cubit
> sparse‑MLA **decode** kernels and the **mHC** kernels are opt‑in/experimental (not loaded in the
> shipped config) and live in the `cubit` repo (@ `5912400`); they are intentionally omitted here.

Validation (vs torch/Triton reference, rel ~1–3e‑3, deterministic): `tools/test_moe_w2_planes.py`,
`tools/test_moe_w2_forward.py`, `tools/test_mqa_logits.py`.

## CUDA C++ decode kernels for the official-image deployments (sm_120, JIT via torch cpp_extension)

Not SASS/cubins: plain `.cu` sources compiled into the `Dockerfile.sm120-dsv41` / `Dockerfile.sm120-qwen38`
images (`torch.utils.cpp_extension.load`, `TORCH_CUDA_ARCH_LIST=12.0`). `mma.sync.m16n8k32` e4m3 → f32,
one MMA per 32-wide scale block, fp32 accumulation.

| source | entry points | replaces | validation |
|---|---|---|---|
| `tools/dsv41_sm120/sm120_gemv/mxfp8_gemv_sm120.cu` | `mxfp8_gemv` (dense M ≤ 16, F8_128x4 scales; v3 transport for M ≤ 8 since 2026-09-20, `VLLM_MOET_GEMV_IMPL=v1` for the previous kernel), `mxfp8_gemv_grouped` (DeepSeek-V4 `wo_a`, T ≤ 64, row-major + DeepGEMM MN-packed scales); scalar fallback `VLLM_MOET_GEMV_IMPL=scalar`, experiment `=v2` | FlashInfer CUTLASS `mm_mxfp8` for decode shapes; BF16-weight cuBLAS bmm for `wo_a` | `sm120_gemv/test_mxfp8_gemv_sm120.py` (vs CUTLASS + fp32 ref, maxrel 3.8e-3 = CUTLASS), `test_wo_a_gemv_sm120.py`, integration tests |
| `tools/qwen38_sm120/moe_gemv/fused_moe_gemv_sm120.cu` | `fused_moe_gemv` (vLLM `fused_moe_kernel` contract, naive + `moe_align_block_size` paths, routed weights), `fused_moe_gemv_act` (silu·up + UE8M0 per-32 quant fused, naive path) | Triton `fused_moe_kernel` at ≤ 320 (token, expert) pairs; `act_and_mul` + `per_token_group_quant_fp8` | `moe_gemv/test_fused_moe_gemv_sm120.py` (vs Triton: 0 differing elements, and fp32 ref), `test_fused_moe_integration.py` (dispatch + `TritonExperts.apply` with vLLM's workspace layout) |
