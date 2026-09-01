"""Real-kernel probes for W2 cold-restage memory safety.

These probes use synthetic files only. Run them inside the serving image on
the same filesystem/cgroup shape as production; they never read or mutate a
model checkpoint or production pack.
"""

import argparse
import ctypes
import gc
import json
import mmap
import os
import tempfile
from datetime import UTC, datetime
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.model_executor.layers.quantization.utils import moe_w2_store as store
from vllm.model_executor.model_loader import weight_utils


def resident_bytes(path: str) -> int:
    """Return file pages currently resident according to Linux mincore(2)."""
    size = os.path.getsize(path)
    page = os.sysconf("SC_PAGE_SIZE")
    pages = (size + page - 1) // page
    fd = os.open(path, os.O_RDONLY)
    try:
        mapping = mmap.mmap(
            fd,
            size,
            flags=mmap.MAP_PRIVATE,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        try:
            anchor = ctypes.c_char.from_buffer(mapping)
            vector = (ctypes.c_ubyte * pages)()
            libc = ctypes.CDLL(None, use_errno=True)
            result = libc.mincore(
                ctypes.c_void_p(ctypes.addressof(anchor)),
                ctypes.c_size_t(size),
                vector,
            )
            if result != 0:
                errno = ctypes.get_errno()
                raise OSError(errno, os.strerror(errno))
            return sum(1 for value in vector if value & 1) * page
        finally:
            del anchor
            mapping.close()
    finally:
        os.close(fd)


def cgroup_sample() -> dict:
    status = store._cgroup_memory_status()
    return {
        key: status.get(key)
        for key in (
            "known",
            "version",
            "path",
            "limited",
            "max_available",
            "high_available",
            "current",
            "anon",
            "file",
            "file_mapped",
            "swap_current",
            "swap_limit",
            "events",
        )
    }


def provenance() -> dict:
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "image": os.getenv("W2_PROBE_IMAGE", "unknown"),
        "image_id": os.getenv("W2_PROBE_IMAGE_ID", "unknown"),
        "kernel": os.uname().release,
    }


def checkpoint_residency(root: str, mib: int) -> dict:
    os.makedirs(root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="w2-residency-", dir=root) as work:
        path = os.path.join(work, "synthetic.safetensors")
        tensor = torch.full((mib * (1 << 20),), 0x5A, dtype=torch.uint8)
        save_file({"weight": tensor}, path)
        del tensor
        gc.collect()
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

        before_drop: list[int] = []
        original_drop = store._drop_path_page_cache

        def measured_drop(drop_path: str, label: str) -> bool:
            before_drop.append(resident_bytes(drop_path))
            return original_drop(drop_path, label)

        env = {
            "VLLM_MOE_W2": "1",
            "VLLM_MOE_W2_STORE_DIR": work,
            "VLLM_MOE_W2_PACK_ID": "synthetic-residency-v1",
            "VLLM_MOE_W2_CACHE_CONTROL": "required",
            "VLLM_MOE_W2_MIN_MEM_AVAILABLE_GB": "4",
            "VLLM_MOE_W2_MIN_CGROUP_HEADROOM_GB": "0.125",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(
                store, "_drop_path_page_cache", side_effect=measured_drop
            ):
                rows = list(
                    weight_utils.safetensors_weights_iterator(
                        [path], use_tqdm_on_load=False
                    )
                )
            immediate = resident_bytes(path)
            store.checkpoint_cleanup_pending()
            after_retry = resident_bytes(path)
        checksum = int(rows[0][1][::4096].sum())
        del rows
        gc.collect()

        file_bytes = os.path.getsize(path)
        result = {
            "probe": "checkpoint-residency",
            "provenance": provenance(),
            "file_bytes": file_bytes,
            "resident_before_dontneed": max(before_drop, default=0),
            "resident_after_immediate": immediate,
            "resident_after_retry": after_retry,
            "checksum": checksum,
            "cgroup": cgroup_sample(),
        }
        if result["resident_before_dontneed"] < file_bytes * 0.8:
            raise RuntimeError(f"synthetic shard was not fully faulted: {result}")
        if after_retry > result["resident_before_dontneed"] * 0.25:
            raise RuntimeError(f"DONTNEED did not bound shard residency: {result}")
        return result


def checkpoint_retry_residency(root: str, mib: int) -> dict:
    """Prove the next-shard retry drops pages a live mmap protected earlier."""
    os.makedirs(root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="w2-retry-", dir=root) as work:
        first = os.path.join(work, "first.safetensors")
        second = os.path.join(work, "second.safetensors")
        tensor = torch.full((mib * (1 << 20),), 0x5A, dtype=torch.uint8)
        save_file({"weight": tensor}, first)
        del tensor
        save_file({"weight": torch.ones(4096, dtype=torch.uint8)}, second)
        gc.collect()
        for path in (first, second):
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)

        env = {
            "VLLM_MOE_W2": "1",
            "VLLM_MOE_W2_STORE_DIR": os.path.join(work, "store"),
            "VLLM_MOE_W2_PACK_ID": "synthetic-retry-v1",
            "VLLM_MOE_W2_CACHE_CONTROL": "required",
            "VLLM_MOE_W2_MIN_MEM_AVAILABLE_GB": "4",
            "VLLM_MOE_W2_MIN_CGROUP_HEADROOM_GB": "0.125",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with safe_open(first, framework="pt") as handle:
                mapped = handle.get_tensor("weight")
                checksum = int(mapped[::4096].sum())
                before = resident_bytes(first)
                store.checkpoint_file_done(first)
                immediate = resident_bytes(first)
            del mapped, handle
            gc.collect()
            store.checkpoint_file_preflight(second)
            retried = resident_bytes(first)
            store.checkpoint_cleanup_pending()

        file_bytes = os.path.getsize(first)
        result = {
            "probe": "checkpoint-next-shard-retry",
            "provenance": provenance(),
            "file_bytes": file_bytes,
            "resident_before_first_dontneed": before,
            "resident_after_live_mapping_dontneed": immediate,
            "resident_after_next_shard_retry": retried,
            "checksum": checksum,
            "cgroup": cgroup_sample(),
        }
        if before < file_bytes * 0.8:
            raise RuntimeError(f"synthetic shard was not fully faulted: {result}")
        if immediate < before * 0.8:
            raise RuntimeError(
                f"live mmap did not preserve the first-drop test condition: {result}"
            )
        if retried > before * 0.25:
            raise RuntimeError(
                f"next-shard retry did not bound shard residency: {result}"
            )
        return result


def constrained_pack_build(root: str, layers: int, experts: int, slot_mib: int) -> dict:
    os.makedirs(root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="w2-pack-", dir=root) as work:
        env = {
            "VLLM_MOE_W2": "1",
            "VLLM_MOE_W2_STORE_DIR": work,
            "VLLM_MOE_W2_PACK_ID": "synthetic-pack-v1",
            "VLLM_MOE_W2_CACHE_CONTROL": "required",
            "VLLM_MOE_W2_MIN_MEM_AVAILABLE_GB": "4",
            "VLLM_MOE_W2_MIN_CGROUP_HEADROOM_GB": "0.125",
            "VLLM_MOE_W2_TIER_PREHEAT": "0",
        }
        slot_bytes = slot_mib * (1 << 20)
        samples = []
        with mock.patch.dict(os.environ, env, clear=False):
            pack = store.TieredPackStore(
                work,
                "base",
                n_layers=layers,
                n_experts=experts,
                slot_bytes=slot_bytes,
                ram_gb=0.125,
            )
            try:
                part = torch.full((experts, slot_bytes), 0xA5, dtype=torch.uint8)
                for layer in range(layers):
                    pack.add_layer(layer, (part,))
                    sample = cgroup_sample()
                    sample["layer"] = layer
                    samples.append(sample)
                stats = pack.stats()
                with open(pack._sidecar_path) as f:
                    sidecar = json.load(f)
            finally:
                pack.release()
        if stats["write_cache_drop_calls"] != layers:
            raise RuntimeError(f"not every durable layer dropped cache: {stats}")
        if sidecar.get("layers") != list(range(layers)):
            raise RuntimeError(f"pack sidecar was not complete: {sidecar}")
        if any(sample.get("events", {}).get("oom", 0) for sample in samples):
            raise RuntimeError(f"cgroup OOM event during synthetic build: {samples}")
        return {
            "probe": "constrained-pack-build",
            "provenance": provenance(),
            "layers": layers,
            "experts": experts,
            "slot_bytes": slot_bytes,
            "logical_pack_bytes": layers * experts * slot_bytes,
            "arena_bytes": max(16, int(0.125 * 2**30) // slot_bytes) * slot_bytes,
            "write_cache_drop_calls": stats["write_cache_drop_calls"],
            "write_cache_drop_bytes": stats["write_cache_drop_bytes"],
            "min_cgroup_max_available": min(
                sample["max_available"]
                for sample in samples
                if sample["max_available"] is not None
            ),
            "min_cgroup_high_available": min(
                (
                    sample["high_available"]
                    for sample in samples
                    if sample["high_available"] is not None
                ),
                default=None,
            ),
            "max_cgroup_current": max(sample["current"] for sample in samples),
            "max_cgroup_file": max((sample["file"] or 0) for sample in samples),
            "max_cgroup_anon": max((sample["anon"] or 0) for sample in samples),
            "final_events": samples[-1]["events"],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("checkpoint-residency", "checkpoint-retry", "pack-build"),
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--mib", type=int, default=128)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--slot-mib", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "checkpoint-residency":
        result = checkpoint_residency(args.root, args.mib)
    elif args.mode == "checkpoint-retry":
        result = checkpoint_retry_residency(args.root, args.mib)
    else:
        result = constrained_pack_build(
            args.root, args.layers, args.experts, args.slot_mib
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
