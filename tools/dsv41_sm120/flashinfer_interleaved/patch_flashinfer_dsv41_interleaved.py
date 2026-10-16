#!/usr/bin/env python3
"""FlashInfer's SM120 sparse MLA reading DeepSeek-V4.1 rows with interleaved scales.

vLLM after vllm#61371 (FlashMLA 2026.09.30) writes each V4.1 token's scales right after its own data: the 528-byte
FP8 main (SWA) row and the 288-byte V41_FP4 compressed row. FlashInfer's SM120 reader (0.7.1, 0.7.2, main of
2026-10-16; the "sparse" and "cake" backends) reads the footer placement only: a page's data rows, then its scale
rows. Same bytes per page, another order, so a mismatch decodes garbage without an error.

dsv41-interleaved.patch (19 files, all-or-nothing) adds `kv_scale_placement="footer"|"interleaved"` to
`trtllm_batch_decode_sparse_mla_dsv4` and SparseMLASm120Wrapper (sparse backend): one more field in the execution
metadata (19th, optional - 18-field tuples still mean footer), a runtime flag in the DSV4_1 sections, and
FooterScaleLayout offsets that take it - decode ordinary / mixed / full-BF16, prefill SG / mixed / full-BF16. The
footer path compiles to the same arithmetic (its outputs are bit-identical to an unpatched build).

--build compiles the patched sparse_mla_sm120 module (FLASHINFER_CUDA_ARCH_LIST, default 12.0f) and installs it as
data/aot/sparse_mla_sm120/sparse_mla_sm120.so, the artifact FlashInfer loads before any jit-cache provider's - so
the served image needs no JIT at start and a bind-mounted ~/.cache cannot hide it.
Usage: patch_flashinfer_dsv41_interleaved.py [--flashinfer-dir DIR] [--check] [--build]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PATCH = Path(__file__).with_name("dsv41-interleaved.patch")


def parse(diff: str) -> dict[str, list[tuple[str, str]]]:
    files: dict[str, list[tuple[str, str]]] = {}
    for block in re.split(r"^diff --git ", diff, flags=re.M)[1:]:
        lines = block.splitlines(keepends=True)
        path = re.match(r"a/(\S+) b/", lines[0]).group(1)
        hunks, i = [], 0
        while i < len(lines) and not lines[i].startswith("@@"):
            i += 1
        while i < len(lines):
            i += 1
            before, after = [], []
            while i < len(lines) and not lines[i].startswith("@@"):
                tag, body = lines[i][:1], lines[i][1:]
                if tag in " -":
                    before.append(body)
                if tag in " +":
                    after.append(body)
                i += 1
            hunks.append(("".join(before), "".join(after)))
        files[path] = hunks
    return files


def patched_text(path: Path, hunks: list[tuple[str, str]]) -> str | None:
    s = path.read_text()
    for n, (before, after) in enumerate(hunks):
        if s.count(before) == 1:
            s = s.replace(before, after)
        elif before not in s and s.count(after) == 1:
            continue
        else:
            print(f"{path}: hunk {n + 1} does not apply (found {s.count(before)}x)", file=sys.stderr)
            return None
    return s


def build(fi_dir: Path) -> int:
    env = dict(os.environ)
    env.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0f")
    env.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/flashinfer-dsv41-interleaved")
    code = (
        "from flashinfer.jit.mla import gen_sparse_mla_sm120_module as g\n"
        "s = g()\n"
        "s.build(verbose=False)\n"
        "print(s.jit_library_path)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if out.returncode:
        print(out.stdout[-4000:], out.stderr[-4000:], file=sys.stderr)
        return out.returncode
    built = Path(out.stdout.strip().splitlines()[-1])
    dest = fi_dir / "data/aot/sparse_mla_sm120/sparse_mla_sm120.so"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, dest)
    shutil.rmtree(env["FLASHINFER_WORKSPACE_BASE"], ignore_errors=True)
    print(f"{dest}: sparse_mla_sm120 built for {env['FLASHINFER_CUDA_ARCH_LIST']} ({dest.stat().st_size >> 20} MiB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    spec = importlib.util.find_spec("flashinfer")
    default = Path(spec.submodule_search_locations[0]) if spec and spec.submodule_search_locations else None
    ap.add_argument("--flashinfer-dir", type=Path, default=default)
    ap.add_argument("--check", action="store_true", help="verify every hunk applies, write nothing")
    ap.add_argument("--build", action="store_true", help="then compile and install the module as its AOT artifact")
    args = ap.parse_args()
    if args.flashinfer_dir is None:
        print("flashinfer is not installed", file=sys.stderr)
        return 1
    out = {}
    for rel, hunks in parse(PATCH.read_text()).items():
        out[args.flashinfer_dir / rel] = patched_text(args.flashinfer_dir / rel, hunks)
    if any(s is None for s in out.values()):
        print("nothing written", file=sys.stderr)
        return 1
    for path, s in out.items():
        if path.suffix == ".py":
            compile(s, str(path), "exec")
    changed = [path for path, s in out.items() if s != path.read_text()]
    if args.check:
        print(f"{len(out)} files: {len(changed)} would change, {len(out) - len(changed)} already patched")
        return 0
    for path in changed:
        path.write_text(out[path])
    print(f"{args.flashinfer_dir}: {len(changed)} of {len(out)} files patched (kv_scale_placement)")
    return build(args.flashinfer_dir) if args.build else 0


if __name__ == "__main__":
    sys.exit(main())
