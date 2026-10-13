#!/usr/bin/env python3
"""Host-side one-second watchdog for a guarded W2 cold pack restage."""

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

GIB = 1 << 30


def read_int(path: Path) -> int | None:
    try:
        raw = path.read_text().strip()
        return None if raw == "max" else int(raw)
    except (OSError, ValueError):
        return None


def read_kv(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) == 2:
                values[fields[0]] = int(fields[1])
    except (OSError, ValueError):
        pass
    return values


def cgroup_headrooms(current: int, maximum: int, high: int | None) -> tuple[int, int]:
    """Return hard-limit and soft-throttle headroom without conflating them."""
    max_headroom = maximum - current
    high_headroom = high - current if high is not None else max_headroom
    return max_headroom, high_headroom


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.split()[0]) * 1024
    return values


def pressure_full_avg10() -> float:
    try:
        for line in Path("/proc/pressure/memory").read_text().splitlines():
            if line.startswith("full "):
                fields = dict(field.split("=", 1) for field in line.split()[1:])
                return float(fields["avg10"])
    except (OSError, KeyError, ValueError):
        pass
    return -1.0


def container_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def container_cgroup(name: str) -> Path:
    pid = subprocess.check_output(
        ["docker", "inspect", "-f", "{{.State.Pid}}", name], text=True
    ).strip()
    for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0":
            return Path("/sys/fs/cgroup") / fields[2].lstrip("/")
    raise RuntimeError(f"no cgroup-v2 path for container {name}")


def health_code(port: int) -> int:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=1
        ) as response:
            return response.status
    except (urllib.error.URLError, TimeoutError):
        return 0


def committed_layers(sidecar: Path) -> tuple[int, int]:
    try:
        meta = json.loads(sidecar.read_text())
        return int(meta.get("version", -1)), len(meta.get("layers", []))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return -1, 0


def write_reason(path: Path, reason: str) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(reason + "\n")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--abort-file", required=True, type=Path)
    parser.add_argument("--done-file", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--deadline-seconds", type=int, default=2400)
    parser.add_argument("--progress-timeout-seconds", type=int, default=300)
    parser.add_argument("--min-cgroup-max-headroom-gib", type=int, default=4)
    args = parser.parse_args()
    if args.deadline_seconds <= 0 or args.progress_timeout_seconds <= 0:
        parser.error("deadlines must be positive")
    if args.min_cgroup_max_headroom_gib <= 0:
        parser.error("cgroup max headroom floor must be positive")

    start = time.monotonic()
    first_layer_at: float | None = None
    last_progress_at = start
    last_layers = 0
    full_psi_hits = 0
    recent_available: deque[tuple[float, int]] = deque(maxlen=5)
    read_failures = 0
    baseline_events: dict[str, int] | None = None
    baseline_swap_events: dict[str, int] | None = None
    baseline_oom_kill: int | None = None

    fields = [
        "epoch",
        "elapsed_s",
        "mem_available",
        "cached",
        "dirty",
        "writeback",
        "swap_free",
        "cg_current",
        "cg_peak",
        "cg_max",
        "cg_high",
        "cg_headroom",
        "cg_high_headroom",
        "cg_anon",
        "cg_file",
        "cg_file_mapped",
        "cg_file_dirty",
        "cg_file_writeback",
        "cg_swap_current",
        "cg_swap_max",
        "event_high",
        "event_max",
        "event_oom",
        "event_oom_kill",
        "swap_event_max",
        "psi_full_avg10",
        "pgscan",
        "pgsteal",
        "pswpin",
        "pswpout",
        "host_oom_kill",
        "pgmajfault",
        "pack_version",
        "pack_layers",
        "health",
    ]
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    with args.trace.open("w", buffering=1) as trace:
        trace.write("\t".join(fields) + "\n")
        while True:
            now = time.monotonic()
            if args.done_file.exists():
                return 0
            if now - start > args.deadline_seconds:
                reason = f"{args.deadline_seconds}-second watchdog deadline exceeded"
                break
            if not container_running(args.container):
                reason = "container exited before guarded success"
                break
            try:
                cgroup = container_cgroup(args.container)
                memory = meminfo()
                stat = read_kv(cgroup / "memory.stat")
                events = read_kv(cgroup / "memory.events")
                swap_events = read_kv(cgroup / "memory.swap.events")
                vmstat = read_kv(Path("/proc/vmstat"))
                current = read_int(cgroup / "memory.current")
                maximum = read_int(cgroup / "memory.max")
                high = read_int(cgroup / "memory.high")
                peak = read_int(cgroup / "memory.peak")
                swap_current = read_int(cgroup / "memory.swap.current") or 0
                swap_max = read_int(cgroup / "memory.swap.max")
                if current is None or maximum is None:
                    raise RuntimeError("finite cgroup memory.current/max unavailable")
                # memory.high is a soft reclaim/throttle boundary, not the
                # hard allocation ceiling. Keep both distances observable but
                # reserve the fail-closed headroom gate for memory.max; using
                # min(max, high) aborts before clean mmap cache can be reclaimed.
                headroom, high_headroom = cgroup_headrooms(current, maximum, high)
                read_failures = 0
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                read_failures += 1
                if read_failures >= 10:
                    reason = f"watchdog telemetry unavailable: {error}"
                    break
                time.sleep(1)
                continue

            version, layers = committed_layers(args.sidecar)
            if layers > last_layers:
                last_layers = layers
                last_progress_at = now
                first_layer_at = first_layer_at or now
            health = health_code(args.port)
            psi = pressure_full_avg10()
            pgscan = sum(
                value for key, value in vmstat.items() if key.startswith("pgscan")
            )
            pgsteal = sum(
                value for key, value in vmstat.items() if key.startswith("pgsteal")
            )
            host_oom_kill = vmstat.get("oom_kill", 0)
            if baseline_events is None:
                baseline_events = events.copy()
                baseline_swap_events = swap_events.copy()
                baseline_oom_kill = host_oom_kill

            row = {
                "epoch": int(time.time()),
                "elapsed_s": f"{now - start:.1f}",
                "mem_available": memory["MemAvailable"],
                "cached": memory.get("Cached", 0),
                "dirty": memory.get("Dirty", 0),
                "writeback": memory.get("Writeback", 0),
                "swap_free": memory.get("SwapFree", 0),
                "cg_current": current,
                "cg_peak": peak or 0,
                "cg_max": maximum,
                "cg_high": high or 0,
                "cg_headroom": headroom,
                "cg_high_headroom": high_headroom,
                "cg_anon": stat.get("anon", 0),
                "cg_file": stat.get("file", 0),
                "cg_file_mapped": stat.get("file_mapped", 0),
                "cg_file_dirty": stat.get("file_dirty", 0),
                "cg_file_writeback": stat.get("file_writeback", 0),
                "cg_swap_current": swap_current,
                "cg_swap_max": swap_max or 0,
                "event_high": events.get("high", 0),
                "event_max": events.get("max", 0),
                "event_oom": events.get("oom", 0),
                "event_oom_kill": events.get("oom_kill", 0),
                "swap_event_max": swap_events.get("max", 0),
                "psi_full_avg10": psi,
                "pgscan": pgscan,
                "pgsteal": pgsteal,
                "pswpin": vmstat.get("pswpin", 0),
                "pswpout": vmstat.get("pswpout", 0),
                "host_oom_kill": host_oom_kill,
                "pgmajfault": vmstat.get("pgmajfault", 0),
                "pack_version": version,
                "pack_layers": layers,
                "health": health,
            }
            trace.write("\t".join(str(row[field]) for field in fields) + "\n")

            available = memory["MemAvailable"]
            recent_available.append((now, available))
            if available < 24 * GIB:
                reason = f"MemAvailable crossed 24 GiB floor: {available}"
                break
            max_headroom_floor = args.min_cgroup_max_headroom_gib * GIB
            if headroom < max_headroom_floor:
                reason = (
                    "cgroup max headroom crossed "
                    f"{args.min_cgroup_max_headroom_gib} GiB floor: {headroom}"
                )
                break
            for key in ("max", "oom", "oom_kill"):
                if events.get(key, 0) > baseline_events.get(key, 0):
                    reason = f"cgroup memory.events {key} incremented"
                    break
            else:
                reason = ""
            if reason:
                break
            if swap_events.get("max", 0) > baseline_swap_events.get("max", 0):
                reason = "cgroup memory.swap.events max incremented"
                break
            if memory.get("SwapFree", 0) < 2 * GIB:
                reason = "host SwapFree crossed 2 GiB floor"
                break
            if swap_current > 3 * GIB:
                reason = f"cgroup swap crossed 3 GiB ceiling: {swap_current}"
                break
            if host_oom_kill > (baseline_oom_kill or 0):
                reason = "host /proc/vmstat oom_kill incremented"
                break
            dirty = memory.get("Dirty", 0) + memory.get("Writeback", 0)
            if dirty > 8 * GIB and available < 40 * GIB:
                reason = "dirty+writeback exceeded 8 GiB below 40 GiB available"
                break
            full_psi_hits = (
                full_psi_hits + 1 if psi >= 10 and available < 40 * GIB else 0
            )
            if full_psi_hits >= 3:
                reason = "memory full PSI >=10% for three samples below 40 GiB"
                break
            if len(recent_available) == recent_available.maxlen:
                dt = recent_available[-1][0] - recent_available[0][0]
                slope = (recent_available[-1][1] - recent_available[0][1]) / dt
                if slope < -2 * GIB and available < 40 * GIB:
                    reason = f"MemAvailable five-sample slope {slope:.0f} B/s"
                    break
            pack_complete = version == 2 and layers >= 43
            if not pack_complete and first_layer_at is None and now - start > 900:
                reason = "no first committed pack layer after 15 minutes"
                break
            if (
                not pack_complete
                and first_layer_at is not None
                and now - last_progress_at > args.progress_timeout_seconds
            ):
                reason = (
                    "no committed pack-layer progress for "
                    f"{args.progress_timeout_seconds} seconds"
                )
                break
            time.sleep(1)

    write_reason(args.abort_file, reason)
    subprocess.run(["docker", "kill", args.container], check=False)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
