#!/usr/bin/env python3
"""The build step of the pinned texts (splice_fc_texts.py) and their manifest (fc_texts/fc_texts.json), on the CPU: no GPU, no CUDA
call, no torch, no build. The cases:

  manifest   the tree's manifest and its text files: every text has its kernel, its kernel's size and the pinned digest, is not the
             compiler's text, and the kernels' bits are 0 .. n-1.
  synthetic  a SYNTHETIC extension - junk, a host-like ELF header that is no cubin, then an ELF64 image with two kernels and their
             twins - and a manifest of its own: --check fails before the step and passes after it; the step writes the text into
             the TX = 1 twin and nowhere else (every byte outside that section stays), a second run writes nothing; it refuses,
             writing nothing, when the kernel's own text or the TX = 2 twin is not the compiler's text of the manifest, when the
             TX = 1 twin holds a third text, when a kernel without a text has a changed twin, when a text file is not the pinned
             one or has another size, when the extension has no twins or two cubin images; --over-kernel writes a copy whose
             kernel carries the text, leaves its input alone and never overwrites a file.
  --so PATH  (optional) the tree's manifest against a built extension: prints the step's --check of it.

  python3 test_splice_fc_texts.py [--so PATH]
Exit 0 only if every case behaves as described (and, with --so, the check passes)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "splice_fc_texts.py"
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import splice_fc_texts as sp  # noqa: E402


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def blob(seed: str, n: int) -> bytes:
    out = b""
    i = 0
    while len(out) < n:
        out += hashlib.sha256(f"{seed}:{i}".encode()).digest()
        i += 1
    return out[:n]


def elf(sections: list[tuple[str, bytes | None]]) -> bytes:
    """An ELF64 little-endian image with these sections (None = a section without bytes in the file)."""
    names = b"\0"
    name_off = {}
    for name, _ in sections + [(".shstrtab", b"")]:
        name_off[name] = len(names)
        names += name.encode() + b"\0"
    body = bytearray(b"\0" * 64)
    heads = [(0, 0, 0, 0)]  # the null section
    for name, data in sections:
        if data is None:
            heads.append((name_off[name], 8, len(body), 4096))
        else:
            heads.append((name_off[name], 1, len(body), len(data)))
            body += data
    heads.append((name_off[".shstrtab"], 3, len(body), len(names)))
    body += names
    shoff = len(body)
    for name, typ, off, size in heads:
        body += struct.pack("<IIQQQQIIQQ", name, typ, 0, 0, off, size, 0, 0, 1, 0)
    body[0:16] = b"\x7fELF" + bytes([2, 1, 1]) + b"\0" * 9
    struct.pack_into("<HHIQQQIHHHHHH", body, 16, 2, 190, 1, 0, 0, shoff, 0, 64, 0, 0, 64, len(heads), len(heads) - 1)
    return bytes(body)


SYM = {"ka": "_ZN4test2kaILi4ELi1ELi{TX}EEEvNS_4ArgsE", "kb": "_ZN4test2kbILi4ELi2ELi{TX}EEEvNS_4ArgsE"}
SIZE = {"ka": 256, "kb": 512}
COMP = {k: blob("compiler " + k, n) for k, n in SIZE.items()}
TEXT = blob("pinned ka", SIZE["ka"])


def extension(twins: bool = True, images: int = 1, change: dict | None = None) -> bytes:
    """The synthetic extension; change = {(kernel, tx): bytes} replaces one section's bytes."""
    change = change or {}
    secs: list[tuple[str, bytes | None]] = [(".nv.info", blob("info", 100))]
    for k in ("ka", "kb"):
        for tx in ((0, 1, 2) if twins else (None,)):
            name = SYM[k].replace("Li{TX}E", "") if tx is None else SYM[k].replace("{TX}", str(tx))
            secs.append((".text." + name, change.get((k, tx), COMP[k])))
            secs.append((".nv.shared." + name, None))
    image = elf(secs)
    host = b"\x7fELF" + bytes([2, 1, 1]) + blob("host", 700)  # an ELF magic that is no image with these kernels
    return host + blob("pad", 300) + b"".join(image + blob(f"gap{i}", 64) for i in range(images)) + blob("tail", 200)


def manifest(d: Path, **over) -> None:
    man = {"kernels": {k: {"bit": i, "kernel": f"{k}<4, {i + 1}, TX>", "symbol": SYM[k], "text_bytes": SIZE[k], "compiler_sha256": sha(COMP[k])}
                       for i, k in enumerate(("ka", "kb"))},
           "texts": [{"kernel": "ka", "name": "ka.t1", "file": "ka.t1.text", "sha256": over.get("sha256", sha(TEXT))}]}
    d.mkdir(parents=True, exist_ok=True)
    (d / "fc_texts.json").write_text(json.dumps(man))
    (d / "ka.t1.text").write_bytes(over.get("body", TEXT))


def run(*args) -> tuple[int, str]:
    p = subprocess.run([sys.executable, "-B", str(TOOL), *map(str, args)], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", type=Path, default=None, help="a built extension to check against the tree's manifest")
    a = ap.parse_args()
    fails = [0]

    def check(cond: bool, what: str) -> None:
        print(("PASS " if cond else "FAIL ") + what, flush=True)
        fails[0] += 0 if cond else 1

    # ---- the tree's manifest
    man = sp.load_manifest(HERE / "fc_texts")
    ks = man["kernels"]
    check(len(man["texts"]) >= 1 and all(len(t["_bytes"]) == ks[t["kernel"]]["text_bytes"] for t in man["texts"]),
          f"the tree's manifest: {len(man['texts'])} text(s) for {len(ks)} kernels, each with its kernel's size and its pinned digest: "
          + ", ".join(f"{t['name']} -> {t['kernel']} ({len(t['_bytes'])} bytes)" for t in man["texts"]))
    check(sorted(k["bit"] for k in ks.values()) == list(range(len(ks))) and len({k["symbol"] for k in ks.values()}) == len(ks)
          and all("Li{TX}E" in k["symbol"] for k in ks.values()), "the kernels' bits are 0 .. n-1 and every symbol carries its TX place")

    # ---- the synthetic extension
    tmp = Path(tempfile.mkdtemp(prefix="fc_texts_test_"))
    try:
        texts = tmp / "texts"
        manifest(texts)
        so = tmp / "ext.so"
        orig = extension()
        so.write_bytes(orig)
        secs = sp.find_cubin(orig, [s.replace("{TX}", "0") for s in SYM.values()])
        where = {(k, tx): secs[".text." + SYM[k].replace("{TX}", str(tx))] for k in SYM for tx in (0, 1, 2)}
        rc, out = run("--so", so, "--texts", texts, "--check")
        check(rc == 1 and "NOT SPLICED: ka" in out and so.read_bytes() == orig, "--check before the step: 1, names the kernel, writes nothing")
        rc, out = run("--so", so, "--texts", texts)
        got = so.read_bytes()
        off, size = where[("ka", 1)]
        check(rc == 0 and got[off:off + size] == TEXT and got[:off] == orig[:off] and got[off + size:] == orig[off + size:] and len(got) == len(orig),
              "the step: 0; the TX = 1 twin of the kernel holds the pinned text, every byte outside that section is the byte it was")
        check(all(got[o:o + s] == COMP[k] for (k, tx), (o, s) in where.items() if (k, tx) != ("ka", 1)),
              "the kernel itself, its TX = 2 twin and the other kernel's three sections keep the compiler's text")
        rc, out = run("--so", so, "--texts", texts)
        check(rc == 0 and "nothing to write" in out and so.read_bytes() == got, "a second run writes nothing")
        rc, out = run("--so", so, "--texts", texts, "--check")
        check(rc == 0 and "check ok" in out, "--check after the step: 0")

        def refused(label: str, ext: bytes, needle: str, tdir: Path = texts) -> None:
            p = tmp / "r.so"
            p.write_bytes(ext)
            rc, out = run("--so", p, "--texts", tdir)
            check(rc == 1 and needle in out and p.read_bytes() == ext, f"refused, nothing written: {label}")

        other = blob("a third text", SIZE["ka"])
        refused("the kernel's own text is not the manifest's compiler text", extension(change={("ka", 0): other}), "the kernel's own text is")
        refused("the TX = 2 twin is not the compiler's text", extension(change={("ka", 2): other}), "twin 2 holds")
        refused("the TX = 1 twin holds a third text", extension(change={("ka", 1): other}), "neither the compiler's text nor the pinned one")
        refused("a kernel without a text has a changed TX = 1 twin", extension(change={("kb", 1): blob("x", SIZE["kb"])}), "no text for this kernel")
        refused("an extension without twins", extension(twins=False), "0 cubin image(s)")
        refused("two cubin images with these kernels", extension(images=2), "2 cubin image(s)")
        bad = tmp / "bad_digest"
        manifest(bad, sha256=sha(b"not it"))
        refused("a text file that is not the pinned one", orig, "the manifest pins", bad)
        short = tmp / "bad_size"
        manifest(short, body=TEXT[:-16], sha256=sha(TEXT[:-16]))
        refused("a text of another size than its kernel's", orig, "bytes, the text of", short)
        same = tmp / "same"
        manifest(same, body=COMP["ka"], sha256=sha(COMP["ka"]))
        refused("a text that IS the compiler's text", orig, "not a text to pin", same)

        # ---- --over-kernel: an extension without twins, a copy whose kernel carries the text
        plain = tmp / "plain.so"
        pb = extension(twins=False)
        plain.write_bytes(pb)
        outp = tmp / "over.so"
        rc, out = run("--so", plain, "--texts", texts, "--over-kernel", "--out", outp)
        psecs = sp.find_cubin(pb, [s.replace("Li{TX}E", "") for s in SYM.values()])
        off, size = psecs[".text." + SYM["ka"].replace("Li{TX}E", "")]
        ob = outp.read_bytes() if outp.exists() else b""
        check(rc == 0 and plain.read_bytes() == pb and ob[off:off + size] == TEXT and ob[:off] == pb[:off] and ob[off + size:] == pb[off + size:],
              "--over-kernel: a copy whose kernel holds the pinned text and nothing else differs; the input is untouched")
        rc, out = run("--so", plain, "--texts", texts, "--over-kernel", "--out", outp)
        check(rc == 1 and outp.read_bytes() == ob, "--over-kernel never overwrites an existing file")
        rc, out = run("--so", plain, "--texts", texts, "--out", tmp / "x.so")
        check(rc == 1 and not (tmp / "x.so").exists(), "--out without --over-kernel is refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if a.so is not None:
        rc, out = run("--so", a.so, "--check")
        print(out.rstrip())
        check(rc == 0, f"--check of {a.so} against the tree's manifest")
    print("ALL PASS" if fails[0] == 0 else f"{fails[0]} FAILED")
    return 1 if fails[0] else 0


if __name__ == "__main__":
    sys.exit(main())
