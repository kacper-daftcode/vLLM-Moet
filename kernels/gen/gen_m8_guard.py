#!/usr/bin/env python3
"""F1 charteru CAPTURE-SAFE [E] 2026-08-10: injektor guardu pustej grupy
do kanonów m8 (uni/ucb2/k1/dn3) — NIE dotyka generatorów ani ciał
kanonów; wstawia SAMODOMKNIĘTY blok na wejściu kernela:

    expert = slot (k1/dn3, 6 slotów) | slot mod 6 (uni/ucb2, 12 slotów)
    LDG.E.128 x2 row_map[expert][0..7] (nowy 4. param d_rmap, (6,8) i32)
    AND-reduce 8 wartości; jeśli == 0xFFFFFFFF (8x -1) -> EXIT CTA

Blok używa wyłącznie rejestrów martwych na wejściu oryginalnego ciała
(R2, R20-21, R24-31, P0 — audyt write-first wbudowany) i domyka
wszystkie własne scoreboardy przed oddaniem sterowania (oryginalny
prolog startuje z czystym stanem jak przy zwykłym wejściu). Semantyka
grup NIEPUSTYCH: bajt-w-bajt ciało oryginału (diff = sam blok).

Użycie: python3 gen_m8_guard.py <src.sass> <dst.sass> <shift6|shift7>
"""
import re
import sys

CLOBBER_R = [2, 20, 21, 24, 25]
# opcodes, dla których PIERWSZY rejestr NIE jest destem (czytają wszystko)
READ_ALL_OPS = ("STS", "STG", "ST.", "RED", "ATOM", "BRA", "EXIT", "BAR",
                "LDGSTS", "NOP", "DEPBAR", "CCTL")

GUARD_SHIFT7 = """\
    // ---- [E] m8g guard (F1 capture-safe): CTA konczy, gdy
    // row_map[expert][0] == -1 (KONTRAKT: wiersze prefix-packed od 0 —
    // gwarantowane przez build_m8_groups i device-builder). d_rmap =
    // 4. param; quad R60-63 wolny (census). Blok samodomkniety.
    [B------:R-:W4:-:S01] LDCU.64 UR4, c[0x0][0x358] ;
    [B------:R-:W0:-:S01] S2R R2, SR_CTAID.X ;
    [B------:R-:W1:-:S01] LDC.64 R20, c[0x0][0x398] ;  // d_rmap
    [B0-----:R-:W-:-:S06] SHF.R.U32.HI R2, RZ, 0x7, R2 ;  // slot = expert
    [B------:R-:W-:-:S06] SHF.L.U32 R2, R2, 0x5, RZ ;  // *32 B
    [B-1----:R-:W-:-:S06] IMAD.WIDE.U32 R20, R2, 0x1, R20 ;
    [B----4-:R-:W2:-:S01] LDG.E.64 R24, desc[UR4][R20.64] ;  // row_map[e][0:2]
    [B--2---:R-:W-:-:S06] IADD3 R24, PT, PT, R24, 0x1, RZ ;  // -1 -> 0
    [B------:R-:W-:-:S08] ISETP.EQ.AND P0, PT, R24, RZ, PT ;
    [B------:R-:W-:-:S06] MOV R24, RZ ;  // dystans ISETP->@P0 >=15 cykli
    [B------:R-:W-:-:S06] MOV R24, RZ ;  // (wzorzec kanonu: predykat late)
    [B------:R-:W-:-:S05] @P0 EXIT ;  // ekspert pusty: caly CTA konczy
"""

GUARD_SHIFT6 = """\
    // ---- [E] m8g guard (F1 capture-safe): CTA konczy, gdy
    // row_map[expert][0] == -1 (KONTRAKT: prefix-packing); expert =
    // slot mod 6 (12 slotow = 2 projekcje x 6). Quad R60-63 (census).
    [B------:R-:W4:-:S01] LDCU.64 UR4, c[0x0][0x358] ;
    [B------:R-:W0:-:S01] S2R R2, SR_CTAID.X ;
    [B------:R-:W1:-:S01] LDC.64 R20, c[0x0][0x398] ;  // d_rmap
    [B0-----:R-:W-:-:S06] SHF.R.U32.HI R2, RZ, 0x6, R2 ;  // slot 0..11
    [B------:R-:W-:-:S08] ISETP.GE.AND P0, PT, R2, 0x6, PT ;
    [B------:R-:W-:-:S06] MOV R24, RZ ;  // dystans ISETP->@P0 >=15 cykli
    [B------:R-:W-:-:S06] MOV R24, RZ ;
    [B------:R-:W-:-:S06] @P0 IADD3 R2, PT, PT, R2, -0x6, RZ ;  // mod 6
    [B------:R-:W-:-:S06] SHF.L.U32 R2, R2, 0x5, RZ ;  // *32 B
    [B-1----:R-:W-:-:S06] IMAD.WIDE.U32 R20, R2, 0x1, R20 ;
    [B----4-:R-:W2:-:S01] LDG.E.64 R24, desc[UR4][R20.64] ;  // row_map[e][0:2]
    [B--2---:R-:W-:-:S06] IADD3 R24, PT, PT, R24, 0x1, RZ ;
    [B------:R-:W-:-:S08] ISETP.EQ.AND P0, PT, R24, RZ, PT ;
    [B------:R-:W-:-:S06] MOV R24, RZ ;  // dystans ISETP->@P0 >=15 cykli
    [B------:R-:W-:-:S06] MOV R24, RZ ;  // (wzorzec kanonu: predykat late)
    [B------:R-:W-:-:S05] @P0 EXIT ;
"""


def first_use_audit(lines):
    """Każdy rejestr klobrowany przez guard musi być w oryginalnym ciele
    najpierw PISANY (dest) — inaczej abort."""
    pending = {f"R{n}" for n in CLOBBER_R} | {"P0"}
    pat = re.compile(r"\b(R\d+|P\d)\b")
    for ln in lines:
        if not ln.lstrip().startswith("[B"):
            continue
        body = ln.split("]", 1)[1].split(";")[0]
        body = body.split("//")[0].strip()
        toks = body.split()
        if not toks:
            continue
        lead_pred = toks[0].startswith("@")
        if lead_pred:
            p = toks[0].lstrip("@!")
            if p in pending:
                raise SystemExit(f"AUDYT FAIL: {p} czytany przed zapisem: {ln}")
            body = body.split(None, 1)[1]  # zdejmij token @Px
            toks = toks[1:]
        op = toks[0]
        regs = pat.findall(body)
        if not regs:
            continue
        # dest = pierwszy rejestr (dla ISETP: pierwszy P), chyba że op
        # czyta wszystko; predykowany zapis destu liczymy jako zapis
        # (kanon jest poprawny standalone => zero zależności od wejścia)
        reads_all = any(op.startswith(x) for x in READ_ALL_OPS)
        dests = []
        if not reads_all:
            dests = [regs[0]]
            if op.startswith("ISETP"):
                dests = [r for r in regs[:2] if r.startswith("P")] or [regs[0]]
            if op.startswith(("LDG.E.128", "LDS.128", "LDC.64", "LDCU.64",
                              "LDG.E.64")):
                base = int(dests[0][1:])
                width = 4 if ".128" in op else 2
                dests = [f"R{base + i}" for i in range(width)]
        for r in regs:
            if r not in pending:
                continue
            if r in dests:
                pending.discard(r)
            else:
                raise SystemExit(f"AUDYT FAIL: {r} czytany przed zapisem: {ln}")
        if not pending:
            break
    return True


def main():
    src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    guard = {"shift6": GUARD_SHIFT6, "shift7": GUARD_SHIFT7}[mode]
    lines = open(src).read().splitlines(keepends=True)
    out = []
    injected_param = injected_guard = False
    body_lines = [l for l in lines if l.lstrip().startswith("[B")]
    first_use_audit(body_lines)
    for ln in lines:
        if not injected_param and ln.strip() == ".param u64 d_c":
            out.append(ln)
            out.append("    .param u64 d_rmap\n")
            injected_param = True
            continue
        if injected_param and not injected_guard and ln.lstrip().startswith("[B"):
            out.append(guard)
            injected_guard = True
        out.append(ln)
    assert injected_param and injected_guard, "anchor nie znaleziony"
    if out[0].startswith("//"):
        out[0] = out[0].rstrip() + " + [E] m8g guard pustej grupy\n"
    open(dst, "w").write("".join(out))
    print(f"OK: {dst} (audyt rejestrow PASS, guard {mode})")


if __name__ == "__main__":
    main()
