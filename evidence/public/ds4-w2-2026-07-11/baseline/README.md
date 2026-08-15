# Corrected 32K baseline re-score

All rows were re-scored from the supplied raw `content` with the same robust
n-gram, repeated-line, vocabulary, special-token, and max-token rules. The only
semantic correction here is that max-token scoring now reads the field the
runner actually emitted: `ct`.

The three-run aggregate below used 32K, base 8 GiB, delta 6 GiB, LRU, and gate
tau 0.67. It is not a tau-0.75 control arm.

| Eval seed | Source file | SHA-256 | Clean | Sinks | Sink IDs |
|---:|---|---|---:|---:|---|
| 42 | `raw-rig-32k-base8-delta6-lru-s42.jsonl` | `04baf5e34635706839cb07cdefcde1952992c6478f4001200004f61778fa5815` | 27/40 | 5/40 | `r07`, `c10`, `r14`, `r15`, `c18` |
| 43 | `raw-rig-32k-b8d6-lru-s43.jsonl` | `0b863f6440f099ba73197d52c65ef81e384b0e29553a08ea0fa2bfeb28bfc37d` | 22/40 | 7/40 | `r04`, `r06`, `r07`, `r12`, `r14`, `r15`, `r16` |
| 44 | `raw-rig-32k-b8d6-lru-s44.jsonl` | `66f3b203e5578df3b12b10320fd712dd4645e2725e5024260079d9d281efb69e` | 27/40 | 4/40 | `r01`, `c03`, `r10`, `c10` |

Combined: **16/120 sinks (13.3%)**, with a 4-7 sink range. This does not meet
the requested stable 1-2/40 bar.

The tau-0.75 seed-42 file also changes under the alias fix:

- `raw-rig-32k-b8d6-lru-tau075-s42.jsonl`
- SHA-256 `605f1538493754b5126eaddad3371f32decfaa50a2b340731c9451ee792a6a67`
- corrected: 27/40 clean, 5/40 sinks (`c07`, `c13`, `c16`, `c17`, `c20`)
- old standalone re-scorer: 28/40 clean, 3/40 sinks

The original standalone re-scorer undercounted seed 42 by one sink, seed 43 by
one, and tau-0.75 seed 42 by two. Seed 44 happened to be unchanged because its
max-token row also tripped an n-gram rule.
