#!/usr/bin/env python3
"""Pinned texts for the twins of the fast prefill MoE's FC1 / FC2 kernels (b12x_moe_prefill_fast_sm120,
VLLM_MOET_B12X_MOE_PREFILL_FAST_FC_TEXT and the runtime key B12X_MOE_PREFILL_FAST_FC_TEXT).

The extension carries, beside each of four kernels (fc_texts/fc_texts.json: `kernels`), two twins - the same source instantiated
twice more (template parameter TX = 1 and 2) - which the compiler turns into the text of the kernel itself. This step writes a
pinned text (fc_texts/<file>: the bytes of that kernel's .text section in another assembler's schedule of the same instructions)
over the TX = 1 twin inside the built extension: the sm_120a cubin lies verbatim in the .so's fat binary, the section keeps its
size, nothing else moves. The TX = 2 twin and the kernel itself keep the compiler's text.

  splice_fc_texts.py [--so PATH] [--texts DIR]              write every text of the manifest into its twin (idempotent)
  splice_fc_texts.py [--so PATH] [--texts DIR] --check      0 = every text of the manifest stands in its twin and every other
                                                            section named in the manifest holds the compiler's text; else 1
  splice_fc_texts.py --so PATH --out PATH --over-kernel     for an extension WITHOUT twins: a copy (--out, never the input) whose
                                                            kernels themselves carry the texts - one state per file, no key

It refuses (exit 1, nothing written) unless every kernel of the manifest - with a text or without - has the compiler's text the
manifest names (`compiler_sha256`: the text a pinned one was made from) in its own section and in its TX = 2 twin, and the TX = 1
twin holds that text or already the pinned one. After writing it reads the file again: each twin holds its pinned bytes and every
byte outside the written sections is the byte it was. Default --so: the image's JIT build,
$TORCH_EXTENSIONS_DIR/<name>/<name>.so. CPU only: no device, no import of torch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
import sys
from pathlib import Path

NAME = "vllm_moet_b12x_moe_prefill_fast_sm120"
HERE = Path(__file__).resolve().parent


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def elf_sections(buf: bytes, base: int):
    """{section name: (offset in buf, size)} of the ELF64 little-endian image that starts at `base`, or None if none lies there."""
    if buf[base:base + 4] != b"\x7fELF" or len(buf) < base + 0x40 or buf[base + 4] != 2 or buf[base + 5] != 1:
        return None
    shoff, = struct.unpack_from("<Q", buf, base + 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", buf, base + 0x3A)
    if shentsize != 64 or shnum == 0 or shstrndx >= shnum or base + shoff + shnum * 64 > len(buf):
        return None
    raw = [struct.unpack_from("<IIQQQQIIQQ", buf, base + shoff + i * 64) for i in range(shnum)]
    so, ss = raw[shstrndx][4], raw[shstrndx][5]
    if base + so + ss > len(buf):
        return None
    strtab = buf[base + so:base + so + ss]
    out = {}
    for name, typ, _flags, _addr, off, size, _link, _info, _align, _entsize in raw:
        end = strtab.find(b"\0", name)
        if name >= len(strtab) or end < 0:
            return None
        if typ != 8:  # SHT_NOBITS has no bytes in the file
            if base + off + size > len(buf):
                return None
            out[strtab[name:end].decode("ascii", "replace")] = (base + off, size)
    return out


def find_cubin(buf: bytes, symbols: list[str]):
    """The one ELF image inside buf that has the .text section of any of `symbols`: its section table."""
    hits = []
    i = buf.find(b"\x7fELF")
    while i >= 0:
        secs = elf_sections(buf, i)
        if secs is not None and any(".text." + s in secs for s in symbols):
            hits.append(secs)
        i = buf.find(b"\x7fELF", i + 1)
    if len(hits) != 1:
        raise SystemExit(f"fc texts: REFUSED - {len(hits)} cubin image(s) with the manifest's kernels in the extension, exactly one is "
                         "expected (0: an extension built from a source without the twins, or not this module)")
    return hits[0]


def load_manifest(texts: Path):
    """The manifest with every text's bytes read and checked against its digest and its kernel's size."""
    man = json.loads((texts / "fc_texts.json").read_text(encoding="utf-8"))
    kernels = man["kernels"]
    if sorted(k["bit"] for k in kernels.values()) != list(range(len(kernels))):
        raise SystemExit("fc texts: the manifest's kernel bits are not 0 .. n-1, each once")
    seen = set()
    for t in man["texts"]:
        k = kernels.get(t["kernel"])
        if k is None or t["kernel"] in seen:
            raise SystemExit(f"fc texts: text {t['name']}: unknown kernel {t['kernel']!r}, or a second text for it")
        seen.add(t["kernel"])
        body = (texts / t["file"]).read_bytes()
        if sha(body) != t["sha256"]:
            raise SystemExit(f"fc texts: {t['file']} has sha256 {sha(body)}, the manifest pins {t['sha256']}")
        if len(body) != k["text_bytes"]:
            raise SystemExit(f"fc texts: {t['file']} has {len(body)} bytes, the text of {t['kernel']} has {k['text_bytes']}")
        if sha(body) == k["compiler_sha256"]:
            raise SystemExit(f"fc texts: {t['file']} IS the compiler's text of {t['kernel']} - not a text to pin")
        t["_bytes"] = body
    return man


def sec(buf: bytes, secs: dict, symbol: str):
    hit = secs.get(".text." + symbol)
    return None if hit is None else (hit[0], hit[1], buf[hit[0]:hit[0] + hit[1]])


def survey(buf: bytes, man: dict, over_kernel: bool):
    """Per kernel: what its own section and its two twins hold. Returns (rows, problems, writes)."""
    kernels, texts = man["kernels"], {t["kernel"]: t for t in man["texts"]}
    sym = (lambda k, tx: k["symbol"].replace("Li{TX}E", "")) if over_kernel else (lambda k, tx: k["symbol"].replace("{TX}", str(tx)))
    secs = find_cubin(buf, [sym(k, 0) for k in kernels.values()])
    rows, problems, writes = [], [], []
    for kid, k in kernels.items():
        t = texts.get(kid)
        own = sec(buf, secs, sym(k, 0))
        if own is None:
            problems.append(f"{kid}: the extension has no kernel {k['kernel'].replace('TX', '0')}")
            continue
        d_own = sha(own[2])
        what = lambda d: "the compiler's text" if d == k["compiler_sha256"] else (f"the pinned text {t['name']}" if t and d == t["sha256"] else f"ANOTHER text ({d[:16]})")
        if own[1] != k["text_bytes"]:
            problems.append(f"{kid}: the kernel's text has {own[1]} bytes, the manifest says {k['text_bytes']}")
            continue
        if over_kernel:
            if t is None:
                rows.append(f"{kid} [{k['kernel']}]: kernel = {what(d_own)}; no pinned text for it")
                if d_own != k["compiler_sha256"]:
                    problems.append(f"{kid}: the kernel holds {what(d_own)}, not the compiler's text {k['compiler_sha256'][:16]}")
                continue
            rows.append(f"{kid} [{k['kernel']}]: kernel = {what(d_own)}")
            if d_own == k["compiler_sha256"]:
                writes.append((kid, t, own, own[2]))
            elif d_own != t["sha256"]:
                problems.append(f"{kid}: the kernel holds {what(d_own)} - neither the compiler's text nor the pinned one")
            continue
        tw1, tw2 = sec(buf, secs, sym(k, 1)), sec(buf, secs, sym(k, 2))
        if tw1 is None or tw2 is None:
            problems.append(f"{kid}: the extension has no twins of {k['kernel']} (built from a source without TX?)")
            continue
        d1, d2 = sha(tw1[2]), sha(tw2[2])
        rows.append(f"{kid} [{k['kernel']}]: kernel = {what(d_own)} | twin 2 = {what(d2)} | twin 1 = {what(d1)}"
                    + ("" if t else " (no pinned text for this kernel)"))
        if d_own != k["compiler_sha256"]:
            problems.append(f"{kid}: the kernel's own text is {d_own[:16]}, the manifest's compiler text is {k['compiler_sha256'][:16]}"
                            " - this build's compiler gives another text than the one a pinned text is made from")
        if d2 != k["compiler_sha256"]:
            problems.append(f"{kid}: twin 2 holds {what(d2)}, not the compiler's text")
        if t is None:
            if d1 != k["compiler_sha256"]:
                problems.append(f"{kid}: twin 1 holds {what(d1)} and the manifest has no text for this kernel")
        elif d1 == k["compiler_sha256"]:
            writes.append((kid, t, tw1, own[2]))
        elif d1 != t["sha256"]:
            problems.append(f"{kid}: twin 1 holds {what(d1)} - neither the compiler's text nor the pinned one")
    return rows, problems, writes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--so", type=Path, default=None)
    ap.add_argument("--texts", type=Path, default=HERE / "fc_texts")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--over-kernel", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    so = a.so or Path(os.environ.get("TORCH_EXTENSIONS_DIR", "/opt/vllm-moet/torch_extensions")) / NAME / (NAME + ".so")
    if a.over_kernel and (a.out is None or a.out.exists() or a.check):
        print("fc texts: --over-kernel needs --out PATH (a file that does not exist yet) and no --check")
        return 1
    if a.out is not None and not a.over_kernel:
        print("fc texts: --out belongs to --over-kernel")
        return 1
    man = load_manifest(a.texts)
    buf = so.read_bytes()
    print(f"fc texts: {so} ({len(buf)} bytes, sha256 {sha(buf)}); manifest {a.texts / 'fc_texts.json'}: {len(man['texts'])} text(s) for {len(man['kernels'])} kernel(s)")
    rows, problems, writes = survey(buf, man, a.over_kernel)
    for r in rows:
        print("  " + r)
    if problems:
        for p in problems:
            print("  REFUSED: " + p)
        print(f"fc texts: {len(problems)} problem(s) - nothing written")
        return 1
    if a.check:
        if writes:
            for kid, t, _where, _ref in writes:
                print(f"  NOT SPLICED: {kid} - twin 1 still holds the compiler's text, the manifest pins {t['name']}")
            print(f"fc texts: check FAILED - {len(writes)} of {len(man['texts'])} text(s) not in their twins")
            return 1
        print(f"fc texts: check ok - {len(man['texts'])} of {len(man['texts'])} text(s) in their twins, every other section named in the manifest holds the compiler's text")
        return 0
    if not writes:
        print(f"fc texts: nothing to write - {len(man['texts'])} of {len(man['texts'])} text(s) already in place")
        return 0
    new = bytearray(buf)
    spans = []
    for kid, t, (off, size, _old), ref in writes:
        new[off:off + size] = t["_bytes"]
        spans.append((off, size))
        ndiff = sum(1 for x, y in zip(ref, t["_bytes"]) if x != y)
        print(f"  WRITE {kid}: {t['name']} ({size} bytes, sha256 {t['sha256']}; {ndiff} byte(s) differ from the compiler's text) at offset {off}")
    new = bytes(new)
    dst = a.out if a.over_kernel else so
    tmp = dst.with_name(dst.name + ".splice.tmp")
    tmp.write_bytes(new)
    os.chmod(tmp, stat.S_IMODE(os.stat(so).st_mode))
    os.replace(tmp, dst)
    got = dst.read_bytes()  # read back: the written file is what was meant, and nothing outside the sections moved
    outside_same = len(got) == len(buf)
    pos = 0
    for off, size in sorted(spans):
        outside_same = outside_same and got[pos:off] == buf[pos:off]
        pos = off + size
    outside_same = outside_same and got[pos:] == buf[pos:]
    _rows, problems, left = survey(got, man, a.over_kernel)
    if not outside_same or problems or left:
        print(f"fc texts: READ-BACK FAILED (bytes outside the sections unchanged: {outside_same}; problems {problems}; unwritten {[w[0] for w in left]})")
        return 1
    print(f"fc texts: wrote {len(writes)} text(s) into {dst} (sha256 {sha(got)}); read back: each holds its pinned bytes, "
          f"every byte outside the {len(spans)} section(s) is unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
