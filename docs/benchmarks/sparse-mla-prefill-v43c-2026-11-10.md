# Sparse-MLA prefill kernel (sm_120) — 2026-11-10: stores whose data registers are rewritten without a wait, the probe, and `v43c`

`sparse_mla_prefill_v42`, `v43` and `v43b` carry 4 / 3 / 3 places where the data registers of a shared-memory store are
rewritten on some control-flow path before anything has waited on that store. They never produced a wrong bit; a probe on
two card models shows the reason — the rewriting instruction is a shared-memory load, which the memory pipe serves behind
the store — and that no issue distance would have protected a register rewritten by arithmetic. `v43c` is `v43b` with
those stores waited: one control word, the same time.

## The pairs

A store reads its address and data registers when the memory pipe serves it, not when it is issued. A register it reads
may therefore be rewritten only behind a wait on the store's read barrier. cubit places that wait at the register's next
overwriter; for a store in front of a loop's back edge it looked for the overwriter in the text of the next iteration.
In the chunk loop of `v42` / `v43` / `v43b` the iteration of a token's last chunk branches around the block whose
fragment loads carry those waits (`v43b`, instruction indices of its frozen listing):

```
[1348] STS.128 [R84+0xe380], R64      read barrier 5        last stores of the in-loop dequant
[1349] STS.128 [R84+0xe390], R68      read barrier 3
[1351] BRA L_2380                                          back edge
[ 568] L_2380: IADD3 R11, ...         waits on 0, 1, 2
[ 570] @!P5 BRA L_25e0                taken: no further chunk
[ 606] L_25e0: BAR.SYNC               waits on 2
[ 607] @!P5 BRA L_2da0                taken
[ 730] L_2da0: LDSM.16.M88.4 R64, [R76]          rewrites R64..R67, 58 issue cycles behind [1348]
[ 733] LDSM.16.M88.4 R68, [R76+0x20]             rewrites R68..R71, 67 issue cycles behind [1349]
```

Nothing on that path waits on scoreboard 5 or 3. cubit's verifier followed the text of the next iteration and reported
"0 broken WAR" for all three.

`tools/dsv41_sm120/sparse_mla_sass/store_war_paths.py <frozen listing> [--loads] [--stores]` walks, for every store and
every register it reads, every control-flow path (both outcomes of each guarded branch, loops included; a predicate is
known from the last branch on it until an instruction names it) to the first overwriter, and reports the overwriters
reached without a wait on the store's read barrier at least two issue cycles behind the store.

| cubin | sha256 | stores | pairs without a wait on some path | pairs behind a wait on every path |
|---|---|---|---|---|
| `sparse_mla_prefill_v41.cubin` (default) | `c896234b…` | 96 | 0 | 424 |
| `sparse_mla_prefill_v41s1.cubin` | `5924622b…` | 96 | 0 | 424 |
| `sparse_mla_prefill_v41s1b.cubin` | `cba10b0a…` | 96 | 0 | 424 |
| `sparse_mla_prefill_v42.cubin` | `dfdc0279…` | 96 | 4 (53 / 58 / 75 / 75 issue cycles) | 404 |
| `sparse_mla_prefill_v43.cubin` | `43f04db5…` | 96 | 3 (58 / 67 / 67) | 392 |
| `sparse_mla_prefill_v43b.cubin` | `a163aea6…` | 96 | 3 (58 / 67 / 67) | 392 |
| `sparse_mla_prefill_v43c.cubin` | `7e812fa2…` | 96 | 0 | 395 |

The address registers of the scoreboarded loads (245–250 per kernel): no pair in any of the seven. Over the 116 cubins
of `kernels/cubins-sm120/` (1 760 stores, 825 without a read barrier, none of those with a register rewritten behind
it): 112 kernels without a pair, the three above, and `mla_prefill_state.cubin` with 12 at 167 issue cycles and more.
Without predicate knowledge the default cubin, `v41s1` and `v41s1b` show 2 pairs for one store — a path that takes
`@!P5 BRA` and then does not take the same, unwritten predicate — and `v43c` none.

## The probe

Probe kernels, 256 iterations per thread: a quad of registers computed from the iteration index; 0 or 15 `STS.128` in
front (stall 9 or 4); a bare `STS.128` of the quad into the thread's own shared-memory slot; d issue cycles later an
instruction that rewrites its data registers — `LDS.128`, `LDSM.16.M88.4`, `LOP3` on the last register, or nothing; the
slot read back and folded into a checksum. 188 CTAs × 256 threads (8 warps per SM, the attention kernel's geometry) and
32 CTAs × 1 024 threads (32 warps per SM); d = 1 … 15 and on to 75, and 105; 20 launches per kernel, 672 kernels. Every
thread's checksum is computed on the host from the kernel's arithmetic and each thread-launch is classed: the intended
quad in all 256 iterations ("good"), the rewritten quad in all 256 ("late"), or a mixture. A comparison against a
reference kernel with the overwriter 105 cycles behind the store is not enough: for `LOP3` that reference is itself wrong.

RTX PRO 6000 Blackwell Server Edition (core clock 2.37 GHz while busy) / Workstation Edition (2.89 GHz); where one
number is given the two cards agree to 0.4 points. Share of thread-launches that are "late":

| overwriter, d cycles behind the store | 8 warps per SM, nothing in front | 8 warps, 15 stores in front (stall 9 / 4) | 32 warps, nothing in front | 32 warps, 15 in front (stall 9 / 4) |
|---|---|---|---|---|
| nothing | 0 % (good 100 %) | 0 % (good 100 %) | 0 % (good 100 %) | 0 % (good 100 %) |
| `LDS.128`, every d from 1 to 75, and 105 | **0 % (good 100 %)** | **0 % (good 100 %)** | **0 % (good 100 %)** | **0 % (good 100 %)** |
| `LDSM.16.M88.4`, d = 1 … 15, 30, 45, 60, 75, 105 | **0 % (good 100 %)** | **0 % (good 100 %)** | **0 % (good 100 %)** | **0 % (good 100 %)** |
| `LOP3`, d = 1 / 2 / 3 | 98.6 / 98.9 / 70.5 % | 100 % | 96.2 / 96.4 / 91.2 % | 100 % |
| `LOP3`, d = 6 … 30 | the first iteration of every thread, the other 255 good | 100 % | 62 – 93 % | 100 % |
| `LOP3`, d = 60 | the first iteration | 99.2 / 98.6 % | 41 % | 99.9 % |
| `LOP3`, d = 75 | the first iteration | 98.6 / 97.0 % | 95 % | 99.1 / 97.9 % |
| `LOP3`, d = 105 | the first iteration | 87.0 / 88.6 %, the rest a mixture, good 0 % | 54 % | 95.5 / 95.6 % |

About 7 × 10^10 store → load-overwrite events per card, none wrong.

- A shared-memory load that rewrites a store's data registers is served behind the store: its result is not written
  before the store has read them, at any distance, with or without stores queued in front. This is what has protected
  the pairs of `v42` / `v43` / `v43b`.
- A fixed-latency instruction in the same place is not held back by anything: with 15 stores queued in front, at 8 warps
  per SM, it reaches the store in every iteration 30 cycles later and in 87 % of the threads 105 cycles later; with
  nothing queued the steady state is clean from 6 cycles, but the first store behind the kernel's opening stores is
  late at every distance. An issue distance is not a protection for a store's data; a wait on its read barrier is.
- The result is a measurement of two load forms behind `STS.128` on two card models, not a statement of the
  architecture.

## `v43c`

`v43b`'s 1 489 instruction words and stalls with one control word changed: the conditional branch behind the loop-top
`BAR.SYNC` (`[607] @!P5 BRA`) waits on scoreboards 3, 4 and 5 — the read barriers of the dequant's stores that the loop
top (`[568]`: 0, 1, 2) does not wait on. Behind that branch every store issued in front of the loop top has been
served, on every path. Two bytes of the cubin differ from `v43b`.

Build: `KERNEL=v43c tools/dsv41_sm120/sparse_mla_sass/build_sparse_mla_sass.sh` takes the committed `v43b` cubin
(checked by digest), disassembles it, changes that wait mask and assembles the listing as it stands; nothing is
scheduled again. The script ends every build with `store_war_paths.py` and writes a `v43c` only when it reports no pair.

| check | result |
|---|---|
| `store_war_paths.py --loads` | 0 pairs, 0 load-address pairs; 0 also without predicate knowledge |
| cubit's read-only verifiers | `CUBIT_VERIFY` 0 unprotected RAW / 0 broken WAR, `CUBIT_DEP_VERIFY` 0 / 0 / 0 / 0 |
| strict round trip; frozen listing | 1 489 / 1 489; re-assembles byte for byte |
| against `v43b` | 1 489 of 1 489 instruction texts equal; one control field differs (wait mask 0 → scoreboards 3, 4, 5) |
| bits (RTX PRO 6000 Blackwell Server Edition, the r32 image) | identical (`torch.equal`, output + lse) with the reference kernel on the 8 synthetic cases and the 8 dumps of a real prefill; with `v43b` as the control 30 launches × 16 cases in one process and 30 fresh processes × 16 cases: 0 differences |
| the runtime A/B key's helper, `v43c` and the default cubin in either role | 17 / 17 both ways |

Times, µs per launch, 4096 rows, the kernels in the same rounds (RTX PRO 6000 Blackwell Server Edition, core clock
2.34 GHz while busy):

| input | default | `v41s1b` | `v43b` | `v43c` |
|---|---|---|---|---|
| harness 128+512, page 128 | 1 058 | 899 | 724 | 724 |
| harness 192+512 | 1 154 | 986 | 800 | 800 |
| harness 1152+512 | 2 508 | 2 162 | 1 826 | 1 826 |
| harness 128+512, page 64 | 1 056 | 897 | 724 | 724 |
| real prefill input, 128+512 candidates | 978 | 831 | 675 | 675 |
| real prefill input, 126+448 | 895 | 763 | 621 | 621 |
| real prefill input, 128+0 | 269 | 239 | 200 | 200 |

In situ, measured after this record: in the serving server of the reference host `v43c` is 2.569 % ± 0.323 %
faster than the default cubin at 163K (8 pairs, read by order;
[record](dsv41-sm120-r33-2026-11-12.md#the-runtime-keys-in-the-serving-server)); the served forward's numbers of `v43` / `v43b` are of the
same instruction stream.
