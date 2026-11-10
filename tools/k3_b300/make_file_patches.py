#!/usr/bin/env python3
"""Cut a vLLM-Moet file patch set (the input of `apply_file_patches.py`) from two directory trees.

    python3 make_file_patches.py --name k31-r2 --before DIR --after DIR [--note TEXT ...] [--out SET.patch]

BEFORE and AFTER hold the same relative paths (e.g. `vllm/...` copied out of two images with `docker cp`). Every file
of AFTER that differs from BEFORE, or does not exist there, becomes one manifest line and one unified diff (three lines
of context); a file that exists only in BEFORE is an error - a set does not delete. The result is applied to BEFORE in
memory before it is written: the set is only emitted when that reproduces AFTER byte for byte.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_file_patches as A  # noqa: E402


def files_of(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def diff_text(rel: str, old: bytes | None, new: bytes) -> bytes:
    a, b = A.split_lines(old or b""), A.split_lines(new)
    out = [b"--- /dev/null" if old is None else b"--- a/" + rel.encode(), b"+++ b/" + rel.encode()]

    def emit(tag: bytes, line: bytes) -> None:
        if line.endswith(b"\n"):
            out.append(tag + line[:-1])
        else:
            out.extend([tag + line, A.NO_NEWLINE])

    for group in difflib.SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(3):
        a0, a1, b0, b1 = group[0][1], group[-1][2], group[0][3], group[-1][4]
        out.append(f"@@ -{a0 + 1 if a1 > a0 else a0},{a1 - a0} +{b0 + 1 if b1 > b0 else b0},{b1 - b0} @@".encode())
        for op, i0, i1, j0, j1 in group:
            if op == "equal":
                for line in a[i0:i1]:
                    emit(b" ", line)
                continue
            for line in a[i0:i1]:
                emit(b"-", line)
            for line in b[j0:j1]:
                emit(b"+", line)
    return b"\n".join(out) + b"\n"


def build(name: str, before: Path, after: Path, notes: list[str]) -> bytes:
    gone = [f for f in files_of(before) if not (after / f).is_file()]
    if gone:
        raise A.PatchError(f"only in {before}: {gone[:5]} - a patch set does not delete files")
    manifest, body = [], []
    for rel in files_of(after):
        new = (after / rel).read_bytes()
        old = (before / rel).read_bytes() if (before / rel).is_file() else None
        if old == new:
            continue
        for what, data in (("before", old), ("after", new)):
            if data is not None:
                try:
                    data.decode("utf-8")
                except UnicodeDecodeError:
                    raise A.PatchError(f"{rel} ({what}) is not UTF-8 text") from None
        manifest.append(f"# sha256 {A.sha256(old) if old is not None else '-'} {A.sha256(new)} {rel}".encode())
        body.append(diff_text(rel, old, new))
    if not manifest:
        raise A.PatchError("the two trees are identical")
    head = [f"# vllm-moet file patch set: {name}".encode()] + [f"# {n}".encode() for n in notes]
    text = b"\n".join(head + manifest) + b"\n" + b"".join(body)
    # the set must reproduce AFTER from BEFORE, byte for byte, through the applier's own code
    _, listed, hunks = A.parse(text)
    for rel, _, sha_after in listed:
        old = (before / rel).read_bytes() if (before / rel).is_file() else b""
        if A.apply_hunks(rel, old, hunks[rel]) != (after / rel).read_bytes() or A.sha256((after / rel).read_bytes()) != sha_after:
            raise A.PatchError(f"{rel}: the generated diff does not reproduce the file")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--note", action="append", default=[], help="a '# ...' header line (repeatable)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        text = build(args.name, args.before, args.after, args.note)
    except (A.PatchError, OSError) as exc:
        print(f"make_file_patches: {exc}", file=sys.stderr)
        return 1
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(text)
        n = sum(1 for line in text.split(b"\n") if line.startswith(b"# sha256 "))
        print(f"{args.out}: {n} files, {len(text)} bytes")
    else:
        sys.stdout.buffer.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
