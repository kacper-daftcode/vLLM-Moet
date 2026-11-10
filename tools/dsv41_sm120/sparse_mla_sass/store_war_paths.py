#!/usr/bin/env python3
"""Path-sensitive write-after-read check of the memory instructions of a frozen sm_120 listing
(`cubit disassemble --frozen`).

A store reads its address and data registers when the memory pipe serves it, not when it is issued; under load that
is tens of issue cycles later (measured on RTX PRO 6000 silicon: with every warp of an SM storing, a register rewritten
48 cycles behind a shared-memory store still reached the store). A later write of one of those registers is therefore
safe only behind a wait on the store's read barrier - a distance is not a guarantee.

For every store and every register it reads, this walks EVERY path of the control-flow graph from the store - both
outcomes of each guarded branch, loops included - up to the first overwriter of the register, and reports the
overwriters that are reached WITHOUT a wait on the store's read barrier at or before them, with the smallest issue
distance (sum of the stall counts) at which that happens.
With --loads the address registers of scoreboarded loads are checked the same way (protected by a wait on the load's
read barrier or on its write barrier: data back means the address was read).

What the walk assumes, and where it is deliberately on the safe side:
  * a predicate is known from the last guarded branch on it until ANY instruction names it as an operand (sources
    included: more paths than can execute, never fewer); `R2P` forgets every predicate. A path that takes a branch on
    a predicate and later does not take a branch on the same, unwritten predicate is not walked;
  * a wait counts wherever it sits - on a guarded instruction whose guard is false too (the control code is honoured
    before the guard) - but only from 2 issue cycles behind the store on (closer, the barrier may not be armed yet:
    such a wait is reported as `unarmed` and does not protect);
  * every store is taken as executed (a guarded store whose guard is false needs nothing: over-reported, never under);
  * a guarded overwriter whose guard is known false on the path is skipped; unknown or true: it overwrites;
  * multi-register results: .128 / LDSM...4 / an MMA = 4 registers, any 64-bit form / IMAD.WIDE / CS2R / LDSM...2 = 2
    (wider than the instruction where unsure: an overwrite too many, never one too few);
  * control flow: BRA (plain, guarded, .U / .DIV forms) and EXIT are followed; BSSY / BSYNC / BREAK fall through;
    CALL / RET / BRX / JMX / JMP are NOT followed - a kernel with one of them is flagged `control flow not modeled`.

Why it exists: cubit protects such an operand by a wait at its next overwriter, and for a store in front of a loop's
back edge it looks for that overwriter in the text of the next iteration. An iteration that leaves the text order
through a guarded forward branch can reach a different first overwriter, which then carries no wait, and cubit's own
verifier follows the same text (v42, v43 and v43b of the sparse-MLA prefill kernel: 4, 3 and 3 such pairs on the path
of a token's last chunk; v43c closes them). This check walks paths instead.

usage: store_war_paths.py <frozen.sass> [--loads] [--via] [--all] [--stores] [--json out.json] [--require-none]
  --loads          also the address registers of scoreboarded loads
  --via            print the branches, barrier syncs and waits on each reported path
  --all            print the pairs beyond 400 issue cycles too (they are always counted)
  --stores         the full listing: every store, every register it reads, every first overwriter on any path with the
                   wait that protects it (the evidence behind a "0 pairs")
  --require-none   exit status 1 when a pair is reported
"""
import heapq
import json
import re
import sys

INS = re.compile(r'^\s*(?:([A-Za-z_.$][\w.$]*):\s*)?\[B([0-5-]{6}):R([0-5-]):W([0-5-]):([Y-]):S(\d+)\]\s+(.*?)\s*;')
NODEST = ('BAR.', 'BRA', 'EXIT', 'NOP', 'STS', 'STG', 'STL', 'ISETP', 'FSETP', 'DSETP', 'PSETP', 'PLOP3', 'HSETP2',
          'RET', 'CALL', 'JMP', 'JMX', 'BRX', 'KILL', 'BPT', 'YIELD', 'WARPSYNC', 'CCTL', 'MEMBAR', 'ERRBAR', 'DEPBAR',
          'BSSY', 'BSYNC', 'BREAK', 'R2P', 'RED', 'NANOSLEEP', 'TRAP')
UNMODELED = ('CALL', 'RET', 'BRX', 'JMX', 'JMP')
ARMING = 2          # a wait closer than this behind the instruction that set its barrier may pass it unarmed


def parse(lines):
    ins, labels = [], {}
    for line in lines:
        m = INS.match(line)
        if not m:
            continue
        lab, wait, rb, wb, y, st, text = m.groups()
        pred = None
        pm = re.match(r'^(@!?U?P[\dT])\s+(.*)$', text)
        if pm:
            pred, text = pm.group(1), pm.group(2)
        op, _, rest = text.partition(' ')
        if lab:
            labels[lab] = len(ins)
        ins.append(dict(i=len(ins), wait={int(c) for c in wait if c != '-'}, rb=None if rb == '-' else int(rb),
                        wb=None if wb == '-' else int(wb), stall=int(st), pred=pred, op=op, rest=rest.strip(),
                        text=(pred + ' ' if pred else '') + text))
    return ins, labels


def width(op):
    if op.startswith('LDSM'):
        return 4 if op.endswith('.4') else 2 if op.endswith('.2') else 1
    if re.match(r'^[A-Z]MMA', op):
        return 4
    if '.128' in op:
        return 4
    if '64' in op or op.startswith(('IMAD.WIDE', 'DADD', 'DMUL', 'DFMA', 'DMNMX')) or op == 'CS2R':
        return 2
    return 1


def dests(x):
    op = x['op']
    if op.startswith(NODEST):
        return set()
    ops = [o.strip() for o in x['rest'].split(',')]
    for o in ops:                      # first GPR operand that is not a predicate destination
        if re.fullmatch(r'!?U?P\d|PT|UPT|PR', o):
            continue
        m = re.fullmatch(r'R(\d+)', o)
        if not m:
            return set()               # uniform destination (UR..) or nothing
        r = int(m.group(1))
        return set(range(r, r + width(op)))
    return set()


def addr_regs(s):
    out = set()
    for a in re.finditer(r'R(\d+)(\.64)?', re.sub(r'UR\d+', '', s)):
        r = int(a.group(1))
        out |= {r, r + 1} if a.group(2) else {r}
    return out


def store_reads(x):
    """(address registers, data registers) of a store / atomic / reduction, None for anything else."""
    op = x['op']
    if not op.startswith(('STS', 'STG', 'STL', 'ATOM', 'RED')):
        return None
    m = re.match(r'^(?:R\d+\s*,\s*)?(?:desc\[UR\d+\])?\[(.*?)\]\s*(?:,\s*(.*))?$', x['rest'])
    if not m:
        raise SystemExit(f"store not understood: {x['text']}")
    data = set()
    w = 4 if '.128' in op else 2 if '64' in op else 1
    for o in [o.strip() for o in (m.group(2) or '').split(',') if o.strip()]:
        dm = re.fullmatch(r'R(\d+)', o)
        if dm:                          # RZ, a uniform register or an immediate: no general register to rewrite
            d = int(dm.group(1))
            data |= set(range(d, d + w))
    return addr_regs(m.group(1)), data


def load_reads(x):
    """address registers of a load that holds a scoreboard barrier (a variable-latency load), None otherwise."""
    op = x['op']
    if not op.startswith(('LDS', 'LDG', 'LDL', 'LDC', 'LD.')) or op.startswith('LDCU'):
        return None
    if x['rb'] is None and x['wb'] is None:
        return None
    m = re.search(r'\[([^\[\]]*)\]\s*$', x['rest'])
    return addr_regs(m.group(1)) if m else None


def pred_of(x):
    """(predicate number, value the instruction needs to execute) or None."""
    if x['pred'] is None:
        return None
    m = re.fullmatch(r'@(!?)P(\d)', x['pred'])
    return (int(m.group(2)), m.group(1) == '') if m else None


def preds_named(x):
    """predicates an instruction names as operands (destinations and sources alike): forgotten behind it."""
    if x['op'].startswith('R2P'):
        return set(range(8))
    return {int(m.group(1)) for m in re.finditer(r'(?<![UA-Za-z_])P(\d)\b', x['rest'])}


def is_bra(x):
    return x['op'] == 'BRA' or x['op'].startswith('BRA.')


def succ(ins, labels, i, facts):
    """[(next instruction, facts)] - the outcomes the predicate values known on this path allow."""
    x = ins[i]
    pr = pred_of(x)
    if is_bra(x) or x['op'] == 'EXIT':
        ops = [o.strip() for o in x['rest'].split(',') if o.strip()]
        tgt = [labels[ops[-1]]] if is_bra(x) else []
        f0 = dict(facts)
        cond_operand = is_bra(x) and len(ops) > 1          # BRA.U !UP0, L / BRA P0, L: a condition we do not track
        if x['pred'] is None and not cond_operand:
            return [(t, frozenset(f0.items())) for t in tgt]
        if pr is None or cond_operand:                     # uniform predicate or operand condition: both outcomes
            return [(t, frozenset(f0.items())) for t in tgt + [i + 1]]
        n, need = pr
        out = []
        if f0.get(n, need) == need:                        # taken
            f = dict(f0); f[n] = need
            out += [(t, frozenset(f.items())) for t in tgt]
        if f0.get(n, not need) != need:                    # not taken
            f = dict(f0); f[n] = not need
            out.append((i + 1, frozenset(f.items())))
        return out
    named = preds_named(x)
    f = frozenset((k, v) for k, v in facts if k not in named)
    return [(i + 1, f)] if i + 1 < len(ins) else []


def executes(x, facts):
    if x['pred'] in ('@!PT', '@!UPT'):          # a guard that is never true: the instruction only carries its control code
        return False
    pr = pred_of(x)
    return pr is None or dict(facts).get(pr[0], pr[1]) == pr[1]


def walk(ins, labels, s, reg, barriers):
    """Dijkstra over (instruction, known predicates, first armed waiter) from memory instruction s.
    Returns {overwriter: {'dist': least distance, 'unprot': least distance reached with no armed wait or None,
    'unprot_path': ..., 'guards': set of first armed waiters on protected arrivals, 'unarmed': set}}."""
    out, seen = {}, set()
    heap = [(ins[s]['stall'], j, tuple(sorted(f)), -1, (s,), ()) for j, f in succ(ins, labels, s, frozenset())]
    heapq.heapify(heap)
    while heap:
        d, i, f, guard, path, unarmed = heapq.heappop(heap)
        if (i, f, guard) in seen:
            continue
        seen.add((i, f, guard))
        x = ins[i]
        if guard < 0 and barriers & x['wait']:
            if d >= ARMING:
                guard = i
            else:
                unarmed = unarmed + (i,)
        if reg in dests(x) and executes(x, f):
            o = out.setdefault(i, dict(dist=d, unprot=None, unprot_path=None, guards=set(), unarmed=set()))
            o['dist'] = min(o['dist'], d)
            if guard < 0:
                if o['unprot'] is None or d < o['unprot']:
                    o['unprot'], o['unprot_path'] = d, path + (i,)
                o['unarmed'] |= set(unarmed)
            else:
                o['guards'].add(guard)
            continue
        keep = is_bra(x) or x['op'].startswith('BAR.') or bool(x['wait'])
        for j, g in succ(ins, labels, i, frozenset(f)):
            g = tuple(sorted(g))
            if (j, g, guard) not in seen:
                heapq.heappush(heap, (d + x['stall'], j, g, guard, path + (i,) if keep else path, unarmed))
    return out


def kernels(path):
    """[(name, [lines])] - one entry per `.entry` block (a file without directives is one kernel)."""
    out, cur, name = [], [], None
    for line in open(path):
        m = re.match(r'^\s*\.entry\s+(\S+)', line)
        if m:
            if cur and name is None and any(INS.match(l) for l in cur):
                out.append(('?', cur))
            cur, name = [], m.group(1)
            continue
        if re.match(r'^\s*\.endentry', line):
            out.append((name, cur)); cur, name = [], None
            continue
        cur.append(line)
    if cur and any(INS.match(l) for l in cur):
        out.append((name or '?', cur))
    return out


def analyse(ins, labels, loads):
    """rows: one per (memory instruction, role, first overwriter) with its protection."""
    rows, mems = [], []
    for x in ins:
        sr = store_reads(x)
        if sr is not None:
            sets, bars, kind = (('addr', sr[0]), ('data', sr[1])), {x['rb']} - {None}, 'store'
        else:
            la = load_reads(x) if loads else None
            if la is None:
                continue
            # a load that rewrites its own address register is its own business
            sets, bars, kind = (('addr', la - dests(x)),), {x['rb'], x['wb']} - {None}, 'load'
        mem = dict(kind=kind, i=x['i'], text=x['text'], rb=x['rb'], wb=x['wb'], regs={}, bare=not bars)
        for role, regs in sets:
            for r in sorted(regs):
                res = walk(ins, labels, x['i'], r, bars)
                mem['regs'][(role, r)] = res
                for o, v in res.items():
                    rows.append(dict(kind=kind, store=x['i'], store_text=x['text'], rb=x['rb'], wb=x['wb'], role=role, reg=r,
                                     over=o, over_text=ins[o]['text'], over_wait=sorted(ins[o]['wait']), dist=v['dist'],
                                     unprotected=v['unprot'] is not None, unprot_dist=v['unprot'],
                                     guards=sorted(v['guards']), unarmed=sorted(v['unarmed']),
                                     via=[f"[{q}] {ins[q]['text']} (waits {''.join(map(str, sorted(ins[q]['wait']))) or '-'})"
                                          for q in (v['unprot_path'] or ())[1:-1]]))
        mems.append(mem)
    return rows, mems


def fmt_regs(rs):
    rs = sorted(rs)
    return f"R{rs[0]}" if len(rs) == 1 else f"R{rs[0]}..R{rs[-1]}" if rs == list(range(rs[0], rs[-1] + 1)) else ','.join(f"R{r}" for r in rs)


def listing(ins, mems, out):
    """--stores: the evidence, one block per store."""
    for mem in mems:
        if mem['kind'] != 'store':
            continue
        b = f"read barrier {mem['rb']}" if mem['rb'] is not None else "NO read barrier"
        print(f"[{mem['i']:4}] {mem['text']:<44} {b}", file=out)
        groups = {}
        for (role, r), res in mem['regs'].items():
            key = (role, tuple(sorted((o, v['dist'], v['unprot'], tuple(sorted(v['guards']))) for o, v in res.items())))
            groups.setdefault(key, []).append(r)
        if not mem['regs']:
            print("         reads no general register", file=out)
        for (role, key), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[1])):
            if not key:
                print(f"         {role} {fmt_regs(rs)}: not rewritten on any path", file=out)
                continue
            for o, dist, unprot, guards in key:
                if unprot is not None:
                    how = f"UNPROTECTED: reached at {unprot} cycles with no wait on the barrier"
                    if guards:
                        how += f" (other paths: behind a wait at {', '.join(f'[{g}]' for g in guards)})"
                else:
                    how = "behind a wait on it at " + ', '.join(f"[{g}] {ins[g]['text'].split(',')[0]}" + (" (the overwriter itself)" if g == o else '') for g in guards) + " on every path"
                print(f"         {role} {fmt_regs(rs)}: first rewritten by [{o}] {ins[o]['text']} at >= {dist} cycles - {how}", file=out)


def main():
    a = sys.argv[1:]
    path = a[0]
    loads = '--loads' in a
    allrows, total_bad = [], 0
    for name, lines in kernels(path):
        ins, labels = parse(lines)
        try:
            rows, mems = analyse(ins, labels, loads)
        except KeyError as e:
            print(f"{path} [{name}]: not analysed (branch target {e} not in the listing)")
            total_bad += 1
            continue
        for r in rows:
            r['kernel'] = name
        allrows += rows
        bad = [r for r in rows if r['unprotected']]
        pairs = {}
        for r in bad:                    # one pair = (memory instruction, overwriter); its registers together
            pairs.setdefault((r['kind'], r['store'], r['role'], r['over']), []).append(r)
        st = [k for k in pairs if k[0] == 'store']
        ld = [k for k in pairs if k[0] == 'load']
        total_bad += len(pairs)
        nst = sum(1 for m in mems if m['kind'] == 'store')
        nbare = sum(1 for m in mems if m['kind'] == 'store' and m['bare'])
        nld = sum(1 for m in mems if m['kind'] == 'load')
        nprot = len({(r['kind'], r['store'], r['role'], r['over']) for r in rows if not r['unprotected']} - set(pairs))
        unm = sorted({x['op'].split('.')[0] for x in ins if x['op'].startswith(UNMODELED)})
        print(f"{path} [{name}]: {len(ins)} instructions, {nst} stores ({nbare} without a read barrier)"
              + (f", {nld} scoreboarded loads" if loads else '')
              + f": {len(st)} store -> overwriter pair(s) reached with NO wait on the store's read barrier"
              + (f", {len(ld)} load-address pair(s) with no wait on the load's barriers" if loads else '')
              + f"; pairs behind a wait on every path: {nprot}"
              + (f"; CONTROL FLOW NOT MODELED: {' '.join(unm)}" if unm else ''))
        for key, rs in sorted(pairs.items(), key=lambda kv: (kv[0][0] != 'store', min(r['unprot_dist'] for r in kv[1]), kv[0][1])):
            r = min(rs, key=lambda r: r['unprot_dist'])
            if '--all' not in a and r['unprot_dist'] > 400:
                continue
            b = f"R{r['rb'] if r['rb'] is not None else '-'}" + (f"W{r['wb'] if r['wb'] is not None else '-'}" if r['kind'] == 'load' else '')
            print(f"  [{r['store']:4}] {r['store_text']:<40} ({b}) {r['role']} {fmt_regs([x['reg'] for x in rs]):<9} "
                  f"-> [{r['over']:4}] {r['over_text']:<44} waits {''.join(map(str, r['over_wait'])) or '-':<4} at {r['unprot_dist']:4} cycles"
                  + (f"  (unarmed wait at {r['unarmed']})" if r['unarmed'] else ''))
            if '--via' in a:
                for v in r['via']:
                    print(f"           via {v}")
        if '--stores' in a:
            listing(ins, mems, sys.stdout)
    if '--json' in a:
        json.dump(allrows, open(a[a.index('--json') + 1], 'w'), indent=1)
    if '--require-none' in a and total_bad:
        sys.exit(1)


if __name__ == '__main__':
    main()
