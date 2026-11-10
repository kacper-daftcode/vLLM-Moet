#!/usr/bin/env python3
"""Public-tree hygiene: no host paths, session jargon or internal references in
what the repository publishes; every relative Markdown link resolves; Python
tools compile; launchers parse; the reference docs keep the form of a
reference: README.md and docs/*.md have no table cell longer than 400
characters and no line longer than 1 200, and the switch table of
docs/dsv41-sm120.md has its rows in step order with a `served` cell that is a
state (on / off / not built / a value, plus at most one short parenthesis).

    python3 tools/docs_lint.py          # exit 1 on any finding

Data records keep their provenance strings: bench/results/, the rendered
reports under docs/benchmarks/ and the generated patch are not scanned for
wording or form, only for links where applicable.
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

# The form of a reference doc (README.md and docs/*.md; the records under docs/benchmarks/ keep their raw tables and
# logs): a table cell holds a fact, not a narrative. What does not fit goes where AGENTS.md says - the mechanism to the
# line's internals doc, the measurement to a record under docs/benchmarks/, the image history to CHANGELOG.md.
MAX_CELL = 400  # characters of one Markdown table cell
MAX_LINE = 1200  # characters of one line
FORM_DOC_RE = re.compile(r"README\.md|docs/[^/]+\.md")
# Rows that were over MAX_CELL when the check was introduced, in docs of other lines of work that the introducing
# commit did not rewrite: (file, first cell of the row) -> the cell length the row may keep. The way out is to shorten
# the row and delete its entry: an entry whose row is gone or fits MAX_CELL is itself a finding.
CELL_EXEMPT = {
    # "Repository layout": the row of the Kimi-K3.1-Pro files
    ("README.md", "`Dockerfile.b300-k31`, `tools/k3_b300/patch_vllm_k31_experts_grid.py`, `tools/k3_b300/k31/`, "
                  "`tools/k3_b300/apply_file_patches.py`"): 497,
    # "Measurements": two rows of the MoET hybrid's table
    ("docs/moet-hybrid.md", "concurrency"): 515,
    ("docs/moet-hybrid.md", "host RAM while up"): 531,
}
# The switch table of the DeepSeek-V4.1 doc (the table whose first header cell is "step"): one row per step in step
# order (3, 3b, 4, ... 42, 42b, 43); every `served` cell of that doc's tables is a state - on, off, not built or a
# value, plus at most one parenthesis of at most 40 characters - not the story of how the state came about.
SWITCH_TABLE_DOC = "docs/dsv41-sm120.md"
STEP_RE = re.compile(r"(\d+)([a-z]?)")
SERVED_RE = re.compile(r"(?:on|off|not built|`[^\s`|]+`|[^\s`|()]+)(?: \([^()]{1,40}\))?")


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


def table_cells(line: str) -> list[str] | None:
    """The cells of a Markdown table row (a line that starts with '|'); None for any other line."""
    s = line.strip()
    if not s.startswith("|"):
        return None
    s = s[1:-1] if len(s) > 1 and s.endswith("|") and not s.endswith("\\|") else s[1:]
    return [c.strip() for c in re.split(r"(?<!\\)\|", s)]


def md_tables(path: str) -> list[list[tuple[int, list[str]]]]:
    """The tables of a Markdown file outside code fences: per table its rows as (line number, cells), header first,
    the delimiter row left out."""
    tables, cur, fence = [], [], False
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.lstrip().startswith("```"):
                fence = not fence
            cells = None if fence else table_cells(line)
            if cells is None:
                if cur:
                    tables.append(cur)
                cur = []
            elif not all(re.fullmatch(r":?-+:?", c) for c in cells):
                cur.append((n, cells))
    if cur:
        tables.append(cur)
    return tables


def check_form(files: list[str]) -> list[str]:
    findings, used = [], set()
    for f in files:
        if not FORM_DOC_RE.fullmatch(f):
            continue
        path = os.path.join(ROOT, f)
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                length = len(line.rstrip("\n"))
                if length > MAX_LINE:
                    findings.append(f"{f}:{n}: line of {length} characters (limit {MAX_LINE}): {line.strip()[:80]}")
        for table in md_tables(path):
            for n, cells in table:
                longest = max(len(c) for c in cells)
                limit = CELL_EXEMPT.get((f, cells[0]), MAX_CELL)
                if limit != MAX_CELL:
                    used.add((f, cells[0]))
                    if longest <= MAX_CELL:
                        findings.append(f"{f}:{n}: the row fits {MAX_CELL} characters per cell: delete its CELL_EXEMPT entry: {cells[0][:60]}")
                if longest > limit:
                    findings.append(f"{f}:{n}: table cell of {longest} characters (limit {limit}): | {' | '.join(cells)[:80]}")
    for f, first in sorted(set(CELL_EXEMPT) - used):
        if f in files:
            findings.append(f"{f}: CELL_EXEMPT entry matches no table row: {first[:60]}")
    return findings


def check_switch_table(files: list[str]) -> list[str]:
    f = SWITCH_TABLE_DOC
    if f not in files:
        return []
    findings, found = [], 0
    for table in md_tables(os.path.join(ROOT, f)):
        header = table[0][1]
        served = header.index("served") if "served" in header else None
        is_switch_table = header[0] == "step" and served is not None
        found += is_switch_table
        prev = None  # (sort key, cell text) of the last step row that was in order
        for n, cells in table[1:]:
            if is_switch_table:
                m = STEP_RE.fullmatch(cells[0])
                key = (int(m.group(1)), m.group(2)) if m else None
                if key is None:
                    findings.append(f"{f}:{n}: switch table: the first cell must be a step (3, 3b, 42): {cells[0][:40]}")
                elif prev is not None and key <= prev[0]:
                    findings.append(f"{f}:{n}: switch table: step {cells[0]} stands after step {prev[1]} (one row per step, in step order)")
                else:
                    prev = (key, cells[0])
            if served is not None:
                cell = cells[served] if served < len(cells) else ""
                if not SERVED_RE.fullmatch(cell):
                    findings.append(f"{f}:{n}: served cell of {len(cell)} characters: must be on, off, not built or a value, "
                                    f"plus at most one parenthesis of 40 characters: {cell[:60]}")
    if found != 1:
        findings.append(f"{f}: expected one switch table (first header cell 'step', a 'served' column), found {found}")
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
    findings = check_wording(files) + check_links(files) + check_form(files) + check_switch_table(files) + check_sources(files)
    for line in findings:
        print(line)
    if findings:
        print(f"docs-lint: {len(findings)} finding(s)")
        return 1
    print(f"docs-lint OK ({len(files)} tracked files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
