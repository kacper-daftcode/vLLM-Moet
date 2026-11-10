#!/usr/bin/env python3
"""Apply a vLLM-Moet file patch set: unified diffs anchored on the sha256 of every file they touch.

A patch set (`tools/k3_b300/k31/*.patch`, written by `make_file_patches.py`) is one text file:

    # vllm-moet file patch set: <name>
    # sha256 <before> <after> <path>        one line per file; <before> is "-" for a file the set creates
    --- a/<path>
    +++ b/<path>
    @@ -l,s +l,s @@
    ...

Paths are relative to `--root` (the site-packages directory that holds `vllm/`). For every file the applier

  * leaves it alone when its sha256 is already <after> (idempotent),
  * otherwise requires its sha256 to be <before> - the set was cut against exactly that file, so a different base
    image stops the build here instead of being patched approximately,
  * applies the hunks with every context and removed line compared (no fuzz, no offsets),
  * requires the sha256 of the result to be <after>.

Nothing is written unless every file of the set passes; `--check` never writes. A patched file keeps its permission
bits. The diffs are plain unified diffs of "\\n"-terminated lines: a set reviews as one, and `patch -p1` or
`git apply` read it too (without the anchors). Usage:

    python3 apply_file_patches.py SET.patch [--root DIR] [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/usr/local/lib/python3.12/dist-packages")
HUNK = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
NO_NEWLINE = b"\\ No newline at end of file"


class PatchError(Exception):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_lines(data: bytes) -> list[bytes]:
    """Lines with their terminators, split on "\\n" only (the unit of the diffs; a last line may have none)."""
    parts = data.split(b"\n")
    out = [p + b"\n" for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def parse(text: bytes) -> tuple[str, list[tuple[str, str, str]], dict[str, list]]:
    """(name, [(path, before, after)], {path: [hunk, ...]}); a hunk is (old_start, old_count, [[tag, line bytes], ...])."""
    name, manifest, hunks = "", [], {}
    path = None
    lines = text.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(b"# vllm-moet file patch set:"):
            name = line.split(b":", 1)[1].strip().decode()
        elif line.startswith(b"# sha256 "):
            parts = line.decode().split()
            if len(parts) != 5:
                raise PatchError(f"malformed manifest line: {line.decode()!r}")
            manifest.append((parts[4], parts[2], parts[3]))
        elif line.startswith(b"--- "):
            if i + 1 >= len(lines) or not lines[i + 1].startswith(b"+++ b/"):
                raise PatchError(f"line {i + 1}: '---' without a '+++ b/<path>' line")
            path = lines[i + 1][len(b"+++ b/"):].decode()
            if path in hunks:
                raise PatchError(f"{path}: two sections for one file")
            hunks[path] = []
            i += 1
        elif line.startswith(b"@@"):
            m = HUNK.match(line)
            if not m or path is None:
                raise PatchError(f"line {i + 1}: malformed hunk header")
            old_n = int(m.group(2)) if m.group(2) is not None else 1
            new_n = int(m.group(4)) if m.group(4) is not None else 1
            hunk = (int(m.group(1)), old_n, [])
            # the header's counts say how many lines belong to the hunk: a removed line that happens to read
            # "-- ..." is never taken for a file header
            while old_n > 0 or new_n > 0:
                i += 1
                if i >= len(lines):
                    raise PatchError(f"{path}: the last hunk is truncated")
                body = lines[i]
                if body == NO_NEWLINE:
                    if not hunk[2]:
                        raise PatchError(f"line {i + 1}: stray no-newline marker")
                    hunk[2][-1][1] = hunk[2][-1][1][:-1]
                    continue
                tag = body[:1] or b" "                   # a blank context line whose space was stripped
                if tag == b" ":
                    old_n, new_n = old_n - 1, new_n - 1
                elif tag == b"-":
                    old_n -= 1
                elif tag == b"+":
                    new_n -= 1
                else:
                    raise PatchError(f"line {i + 1}: unexpected line inside a hunk: {body[:60]!r}")
                hunk[2].append([tag.decode(), body[1:] + b"\n"])
            if old_n < 0 or new_n < 0:
                raise PatchError(f"{path}: a hunk has more lines than its header says")
            if i + 1 < len(lines) and lines[i + 1] == NO_NEWLINE:
                hunk[2][-1][1] = hunk[2][-1][1][:-1]
                i += 1
            hunks[path].append(hunk)
        elif line.startswith(b"#") or line == b"":
            pass
        else:
            raise PatchError(f"line {i + 1}: unexpected text in a patch set: {line[:60]!r}")
        i += 1
    if not manifest:
        raise PatchError("no '# sha256' manifest lines")
    listed = [p for p, _, _ in manifest]
    if len(set(listed)) != len(listed):
        raise PatchError("a path is listed twice in the manifest")
    if set(listed) != set(hunks):
        raise PatchError(f"manifest and diff sections name different files: {sorted(set(listed) ^ set(hunks))}")
    return name, manifest, hunks


def apply_hunks(path: str, old: bytes, hunks: list) -> bytes:
    src = split_lines(old)
    out, pos = [], 0                                     # pos = index into src (0-based)
    for start, old_count, body in hunks:
        at = start if old_count == 0 else start - 1      # "-l,0": insert after line l
        if at < pos or at > len(src):
            raise PatchError(f"{path}: hunk at line {start} overlaps the previous one or lies past the end of the file")
        out += src[pos:at]
        pos = at
        for tag, text in body:
            if tag == "+":
                out.append(text)
                continue
            if pos >= len(src) or src[pos] != text:
                have = src[pos] if pos < len(src) else b"<end of file>"
                raise PatchError(f"{path}: line {pos + 1} is {have[:70]!r}, the set expects {text[:70]!r}")
            if tag == " ":
                out.append(text)
            pos += 1
    return b"".join(out + src[pos:])


def plan(patch_file: Path, root: Path) -> tuple[str, list[tuple[Path, bytes | None, str]]]:
    """(name, [(file, new content or None when already applied, status text)]) - raises PatchError on any mismatch."""
    name, manifest, hunks = parse(patch_file.read_bytes())
    todo = []
    for rel, before, after in manifest:
        if os.path.isabs(rel) or ".." in Path(rel).parts:
            raise PatchError(f"{rel}: path leaves the root")
        target = root / rel
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise PatchError(f"{rel} is not a regular file")
        old = target.read_bytes() if target.is_file() else None
        have = sha256(old) if old is not None else "-"
        if have == after:
            todo.append((target, None, "already applied"))
            continue
        if have != before:
            what = "is missing" if old is None else f"has sha256 {have[:16]}..."
            want = "no file" if before == "-" else f"{before[:16]}..."
            raise PatchError(f"{rel} {what}; the set was cut against {want} "
                             "(another base image, an earlier step missing or a later set already applied)")
        new = apply_hunks(rel, old or b"", hunks[rel])
        if sha256(new) != after:
            raise PatchError(f"{rel}: the patched file has sha256 {sha256(new)[:16]}..., the set records {after[:16]}...")
        todo.append((target, new, "created" if old is None else "patched"))
    return name, todo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patch_set", type=Path)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--check", action="store_true", help="verify that the set applies; write nothing")
    args = ap.parse_args(argv)
    try:
        name, todo = plan(args.patch_set, args.root)
    except (PatchError, OSError) as exc:
        print(f"{args.patch_set.name}: {exc}", file=sys.stderr)
        return 1
    for target, new, status in todo:
        if new is not None and not args.check:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".vllm-moet-tmp")
            tmp.write_bytes(new)
            if target.is_file():                         # a patched file keeps its permission bits
                os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
            os.replace(tmp, target)
        print(f"{name}: {target.relative_to(args.root)}: {status}{' (not written)' if args.check and new is not None else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
