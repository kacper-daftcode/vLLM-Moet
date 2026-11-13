#!/usr/bin/env python3
"""CPU test of `drop_file_pages` (inkling2_bmm_sm120.py): the pages of a memory-mapped safetensors tensor are in the
page cache after a slice of it is read and gone after the slice is dropped; the neighbouring slice keeps its pages;
the data read again is the same.

    python3 test_drop_file_pages.py [DIR]     (DIR: where the 96 MB test file is written; default the system's tmp)
"""

from __future__ import annotations

import ctypes
import mmap
import os
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inkling2_bmm_sm120 as m  # noqa: E402


def cached_pages(path: str, offset: int, size: int) -> int:
    """Pages of the file range that are in the page cache (mincore on a private mapping of the file)."""
    page = mmap.PAGESIZE
    lo = offset // page * page
    length = offset + size - lo
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    fd = os.open(path, os.O_RDONLY)
    addr = libc.mmap(None, length, 1, 1, fd, lo)
    vec = (ctypes.c_ubyte * ((length + page - 1) // page))()
    libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(length), vec)
    libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(length))
    os.close(fd)
    return sum(b & 1 for b in vec)


def main() -> int:
    where = sys.argv[1] if len(sys.argv) > 1 else tempfile.gettempdir()
    path = os.path.join(where, f"drop_file_pages_{os.getpid()}.safetensors")
    experts, row = 24, 4 * 1024 * 1024
    data = torch.randint(0, 256, (experts, row), dtype=torch.uint8)
    save_file({"w": data}, path)
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # a cold file
    os.close(fd)
    ok = True
    try:
        with safe_open(path, framework="pt") as f:
            w = f.get_tensor("w")
            base = m._file_mapping(w.data_ptr())
            if base is None:
                print("  the tensor is not file-backed here (safetensors copied it): nothing to drop, drop returns 0")
                return 0 if m.drop_file_pages(w[:4]) == 0 else 1
            lo, _, offset, _ = base
            def file_range(t):
                return offset + t.data_ptr() - lo, t.numel()
            first, second = w[:8], w[8:16]
            a, b = first.clone(), second.clone()  # reads both slices
            before = cached_pages(path, *file_range(first)), cached_pages(path, *file_range(second))
            released = m.drop_file_pages(first)
            after = cached_pages(path, *file_range(first)), cached_pages(path, *file_range(second))
            again = bool(torch.equal(first, a)) and bool(torch.equal(second, b)) and bool(torch.equal(a, data[:8]))
            pages = first.numel() // mmap.PAGESIZE
            print(f"  slice of {first.numel() >> 20} MiB: {before[0]} of {pages} pages cached before, {after[0]} after "
                  f"({released >> 20} MiB released); the next slice {before[1]} -> {after[1]}; data read again equal: {again}")
            # the slice's first and last page are shared with its neighbours and stay
            ok = before[0] >= pages and after[0] <= 2 and after[1] >= before[1] - 1 and again and released >= first.numel() - 2 * mmap.PAGESIZE
    finally:
        os.unlink(path)
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
