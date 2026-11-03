#!/usr/bin/env python3
"""Every kernel an sm_120 image ships, through cubit, bit for bit.

Extracts the sm_120 cubins of the prebuilt Torch extensions of a vllm-moet image (the
/opt/vllm-moet/torch_extensions/*/*.so the server loads) and runs cubit's strict semantic
round-trip on each: decode every 128-bit instruction to ordinary SASS, re-parse, re-encode, and
require all 128 bits back and the rebuilt ELF byte-identical to the input. No raw-word escapes.
A kernel that fails here uses an encoding the SM120 ISA table does not describe - the signal to
pin it (blackwell-isa) before anyone edits the kernel at the SASS level.

usage: cubit_gate.py IMAGE [--cubit PATH] [--table PATH] [--out DIR] [--only NAME ...]
    cubit_gate.py vllm-moet-sm120:dsv41-nightly-df3bd094-r21

Needs docker (to read the image), cuobjdump (CUDA toolkit on the host or inside the image) and a
cubit binary built from kacper-daftcode/cubit `main` with its vendored tables/sm120.json. Writes
<out>/<extension>/{*.cubin, report.json, failures.jsonl} and a summary JSON; exit 1 on any failure.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_CUBIT = os.environ.get("CUBIT", "/workspace/cubit/target/release/cubit")
DEFAULT_TABLE = os.environ.get("CUBIT_TABLE", "/workspace/cubit/tables/sm120.json")
EXT_DIR = "/opt/vllm-moet/torch_extensions"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, **kw)


def extract(image: str, out: Path, only: list[str] | None) -> list[tuple[str, Path]]:
    """Copy the extensions' .so out of the image and pull their sm_120 cubins with cuobjdump."""
    out.mkdir(parents=True, exist_ok=True)
    cid = run(["docker", "create", image]).stdout.strip()
    if not cid:
        sys.exit(f"docker create {image} failed")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            r = run(["docker", "cp", f"{cid}:{EXT_DIR}", tmp])
            if r.returncode:
                sys.exit(f"docker cp failed: {r.stderr.strip()}")
            root = Path(tmp) / "torch_extensions"
            found = []
            for ext in sorted(p for p in root.iterdir() if p.is_dir()):
                if only and ext.name not in only:
                    continue
                sos = sorted(ext.glob("*.so"))
                if not sos:
                    continue
                dest = out / ext.name
                dest.mkdir(exist_ok=True)
                for so in sos:
                    shutil.copy2(so, dest / so.name)
                    # cuobjdump writes the ELFs into the current directory
                    r = run(["cuobjdump", "-xelf", "all", str(dest / so.name)], cwd=dest)
                    if r.returncode:
                        sys.exit(f"cuobjdump on {so.name} failed: {r.stderr.strip()}")
                cubins = sorted(dest.glob("*.cubin"))
                for c in cubins:
                    found.append((ext.name, c))
            return found
    finally:
        run(["docker", "rm", cid])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--cubit", default=DEFAULT_CUBIT)
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--out", type=Path, default=None, help="work/report directory (default: a fresh dir under /tmp)")
    ap.add_argument("--only", nargs="*", default=None, help="extension names to gate (default: all)")
    args = ap.parse_args()
    if shutil.which("cuobjdump") is None:
        sys.exit("cuobjdump not on PATH (CUDA toolkit); run on a host with the toolkit or from inside the image")
    if not Path(args.cubit).is_file():
        sys.exit(f"cubit binary not found: {args.cubit} (build kacper-daftcode/cubit main: cargo build --release)")
    out = args.out or Path(tempfile.mkdtemp(prefix="cubit_gate_"))
    cubit_ver = run([args.cubit, "--version"]).stdout.strip()
    cubins = extract(args.image, out, args.only)
    if not cubins:
        sys.exit("no extension cubins found in the image")
    summary = {"image": args.image, "cubit": args.cubit, "cubit_version": cubit_ver, "table": args.table,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "extensions": {}}
    failed = 0
    for ext, cubin in cubins:
        d = cubin.parent
        report, failures = d / f"{cubin.stem}.report.json", d / f"{cubin.stem}.failures.jsonl"
        for stale in (report, failures):
            stale.unlink(missing_ok=True)
        rebuilt = d / f"{cubin.stem}.rebuilt"
        shutil.rmtree(rebuilt, ignore_errors=True)
        t0 = time.monotonic()
        r = run([args.cubit, "roundtrip-strict", "-t", args.table, str(cubin), "--report", str(report),
                 "--failures", str(failures), "--rebuilt-dir", str(rebuilt)])
        dt = time.monotonic() - t0
        ok = r.returncode == 0 and "PASS" in r.stdout
        rep = json.loads(report.read_text()) if report.is_file() else None
        entry = {"cubin": cubin.name, "pass": ok, "seconds": round(dt, 2),
                 "summary_line": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-300:]}
        if rep and isinstance(rep, list) and rep:
            rp = rep[0]
            entry["instructions"] = rp.get("report", {}).get("instructions")
            entry["elf_byte_identical"] = rp.get("rebuilt_elf_byte_identical")
            entry["kernels"] = [{k: v for k, v in kr.items() if k in ("instructions", "matched", "mismatches", "decode_errors",
                                                                      "encode_errors", "parse_errors", "forbidden")}
                                for kr in rp.get("report", {}).get("kernels", [])]
        summary["extensions"].setdefault(ext, []).append(entry)
        failed += 0 if ok else 1
        print(f"[{'ok ' if ok else 'BAD'}] {ext}/{cubin.name}: {entry['summary_line']} ({dt:.1f} s)")
    summary["failed"] = failed
    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "cubit_gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    total = sum(e.get("instructions") or 0 for v in summary["extensions"].values() for e in v)
    print(f"{'ALL PASS' if failed == 0 else f'{failed} FAILED'}: {len(cubins)} cubins, {total:,} instructions, reports in {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
