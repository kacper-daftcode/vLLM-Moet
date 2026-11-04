"""Match compact (T <= 8) MoE calls between two eager dumps by (layer, T, identical input x), compare the partials."""
import glob, os, sys, torch
from collections import defaultdict

A, B = sys.argv[1], sys.argv[2]
for rank in range(4):
    fa = sorted(glob.glob(os.path.join(A, f"rank{rank}_*.pt")))
    fb = sorted(glob.glob(os.path.join(B, f"rank{rank}_*.pt")))
    da = [torch.load(f) for f in fa]
    idx = defaultdict(list)
    for a in da:
        if a["T"] <= 8:
            idx[(a["layer"], a["T"])].append(a)
    n_match = n_diff = 0
    reports = []
    for f in fb:
        b = torch.load(f)
        if b["T"] > 8:
            continue
        cands = [a for a in idx.get((b["layer"], b["T"]), []) if torch.equal(a["x"], b["x"]) and torch.equal(a["ids"], b["ids"])]
        if not cands:
            continue
        a = cands[0]
        n_match += 1
        d = (a["out"] - b["out"]).abs()
        if d.max() > 0:
            n_diff += 1
            if len(reports) < 4:
                rows = (d.max(dim=1).values > 0).nonzero().flatten().tolist()
                sl = b["slots"]
                ids = b["ids"]
                cpu_e = sorted({int(e) for t in rows for e in ids[t].tolist() if e >= 0 and int(sl[e]) < 0})
                all_e = sorted({int(e) for t in range(b["T"]) for e in ids[t].tolist() if e >= 0})
                reports.append(f"   layer {b['layer']} T {b['T']} (B call {b['call']}, step {b['step']}): max|d| {d.max():.3e} of |out| {a['out'].abs().max():.3e}, "
                               f"tokens differing {len(rows)}/{b['T']}; experts routed {all_e}; on the CPU in B {cpu_e}")
    print(f"== rank {rank}: compact calls matched {n_match}, with different partials {n_diff}")
    for r in reports:
        print(r)
