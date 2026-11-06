#!/usr/bin/env python3
"""Public-tree hygiene: no host paths, session jargon or internal references in
what the repository publishes; every relative Markdown link resolves; Python
tools compile; launchers parse.

    python3 tools/docs_lint.py          # exit 1 on any finding

Data records keep their provenance strings: bench/results/, the rendered
reports under docs/benchmarks/ and the generated patch are not scanned for
wording, only for links where applicable.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORBIDDEN = [
    (re.compile(r"/workspace/tmp"), "host scratch path"),
    (re.compile(r"\bpar31\b"), "session scratch directory"),
    (re.compile(r"/root/work\b"), "host work path"),
    (re.compile(r"/mnt/nvme/vLLM-Moet"), "host checkout path"),
    (re.compile(r"(?<![\w./-])internal/"), "reference to the private notes"),
    (re.compile(r"\bsessions? [A-Z]\b(?!-)"), "agent session letter"),
    (re.compile(r"\bsessions? \[[A-Z]\]|\(\[[A-Z]\]\)|\breproduces? \[[A-Z]\]"), "agent session letter"),
    (re.compile(r"\bsesj[aie]\b", re.I), "Polish session note"),
    (re.compile(r"\bowner'?s (decision|call|approval)|owner-approved|\bthe owner decides\b|\b[Dd]ecided \d{4}-\d\d-\d\d"), "decision narrative"),
    (re.compile(r"\bserv(ed|ing) since\b|\bin production\b|\bproduction box\b|\bprod logs\b"), "deployment narrative"),
    (re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?Z\b"), "deployment timestamp"),
    (re.compile(r"\b(test|dev) host\b|\btesthost\b"), "test-host reference"),
    (re.compile(r"agent-transcripts|/root/\.cursor|NEEDS_OWNER|\bK/REPORT"), "agent tooling or session report"),
    (re.compile(r"private note|operators' notes|hand-?off directory|\bHANDOFF[-_](20\d\d|moet)"), "reference to the private notes"),
    (re.compile(r"\bripple65b\b"), "deployment slot path"),
]
# Files that legitimately name internal/ (the ignore rules, the contract) or carry provenance strings.
WORDING_EXEMPT = {
    ".gitignore", ".dockerignore", "AGENTS.md", "CONTRIBUTING.md", "tools/docs_lint.py",
    "tools/import_quality_results.py", "tools/import_tau060_quality.py",
}
WORDING_EXEMPT_PREFIXES = ("patch/", "bench/results/", "docs/benchmarks/", "kernels/cubins-sm120/", "kernels/sass/")
TEXT_SUFFIXES = (".md", ".py", ".sh", ".yml", ".yaml", ".cu", ".cuh", ".h", ".txt", "Dockerfile")

LINK_RE = re.compile(r"\]\(([^)\s#]+)(#[^)]*)?\)")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "-C", ROOT, "ls-files", "-z"], check=True, capture_output=True).stdout
    return [f for f in out.decode().split("\0") if f]


def is_text(path: str) -> bool:
    base = os.path.basename(path)
    return path.endswith(TEXT_SUFFIXES) or base.startswith("Dockerfile")


def check_wording(files: list[str]) -> list[str]:
    findings = []
    for f in files:
        if f in WORDING_EXEMPT or f.startswith(WORDING_EXEMPT_PREFIXES) or not is_text(f):
            continue
        try:
            with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
                for n, line in enumerate(fh, 1):
                    for rx, why in FORBIDDEN:
                        if rx.search(line):
                            findings.append(f"{f}:{n}: {why}: {line.strip()[:120]}")
        except UnicodeDecodeError:
            continue
    return findings


def check_links(files: list[str]) -> list[str]:
    findings = []
    for f in files:
        if not f.endswith(".md"):
            continue
        base = os.path.dirname(os.path.join(ROOT, f))
        with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                for m in LINK_RE.finditer(line):
                    target = m.group(1)
                    if re.match(r"^[a-z]+:", target) or target.startswith("mailto:"):
                        continue
                    path = os.path.normpath(os.path.join(base, target))
                    if not os.path.exists(path):
                        findings.append(f"{f}:{n}: broken link {target}")
    return findings


def check_sources(files: list[str]) -> list[str]:
    findings = []
    py = [f for f in files if f.endswith(".py") and (f.startswith("tools/") or f.startswith("docker/") or f.startswith("bench/"))]
    for f in py:
        r = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(ROOT, f)], capture_output=True, text=True)
        if r.returncode != 0:
            findings.append(f"{f}: does not compile: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")
    sh = [f for f in files if f.endswith(".sh")]
    for f in sh:
        r = subprocess.run(["bash", "-n", os.path.join(ROOT, f)], capture_output=True, text=True)
        if r.returncode != 0:
            findings.append(f"{f}: bash -n: {r.stderr.strip()[:160]}")
    return findings


def main() -> int:
    files = tracked_files()
    findings = check_wording(files) + check_links(files) + check_sources(files)
    for line in findings:
        print(line)
    if findings:
        print(f"docs-lint: {len(findings)} finding(s)")
        return 1
    print(f"docs-lint OK ({len(files)} tracked files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
