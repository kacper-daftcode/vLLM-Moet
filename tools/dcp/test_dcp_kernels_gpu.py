"""GPU unit tests for the DCP-modified Triton kernels.

Extracts the kernels from the real source files (ast-based, no copy drift)
and validates them against brute-force references on a real GPU.
"""
import ast
import sys
import types

import torch
import triton
import triton.language as tl

# ---- stub vllm.triton_utils so the extracted sources see tl/triton ----
stub = types.ModuleType("vllm.triton_utils")
stub.tl = tl
stub.triton = triton
sys.modules["vllm.triton_utils"] = stub


def extract(path, names):
    """Write the kernels to a real temp module (triton needs file-backed src)."""
    import importlib.util
    import hashlib

    src = open(path).read()
    tree = ast.parse(src)
    parts = ["import triton\nimport triton.language as tl\n"]
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef,)) and node.name in names:
            parts.append("@triton.jit\n" + ast.get_source_segment(src, node) + "\n")
            found.add(node.name)
    missing = set(names) - found
    assert not missing, f"kernels not found: {missing}"
    mod_src = "\n".join(parts)
    tag = hashlib.md5(mod_src.encode()).hexdigest()[:8]
    mod_path = f"/tmp/_dcp_k_{tag}.py"
    with open(mod_path, "w") as f:
        f.write(mod_src)
    spec = importlib.util.spec_from_file_location(f"_dcp_k_{tag}", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return {n: getattr(mod, n) for n in names}


SU = "/root/workspace/vllm-v0.24.0/vllm/v1/attention/backends/mla/sparse_utils.py"
IX = "/root/workspace/vllm-v0.24.0/vllm/v1/attention/backends/mla/indexer.py"

conv = extract(SU, ["_convert_req_index_to_global_index_kernel"])[
    "_convert_req_index_to_global_index_kernel"
]
pfx = extract(IX, ["_build_prefill_chunk_metadata_kernel"])[
    "_build_prefill_chunk_metadata_kernel"
]

dev = "cuda"
torch.manual_seed(0)


def brute_owner(p, world, interleave):
    return (p // interleave) % world


def brute_local(p, world, interleave):
    return (p // interleave // world) * interleave + p % interleave


def test_convert(world, rank, interleave, block_size, topk=2048):
    num_tokens, num_reqs = 8, 3
    max_blocks = 512
    L = 3000  # global positions in [0, L)
    req_id = torch.randint(0, num_reqs, (num_tokens,), dtype=torch.int32, device=dev)
    block_table = torch.randint(
        1, 30000, (num_reqs, max_blocks), dtype=torch.int32, device=dev
    )
    tok = torch.randint(0, L, (num_tokens, topk), dtype=torch.int32, device=dev)
    tok[torch.rand_like(tok, dtype=torch.float32) < 0.2] = -1

    out = torch.empty_like(tok)
    valid = torch.zeros(num_tokens, dtype=torch.int32, device=dev)
    BLOCK_N = 128
    grid = (num_tokens, topk // BLOCK_N)
    conv[grid](
        req_id, block_table, tok, out, valid,
        None, None,
        max_blocks, block_size, BLOCK_N,
        False, True,
        world, rank, interleave,
        block_table.stride(0), block_table.stride(1),
        tok.stride(0), tok.stride(1),
        out.stride(0), out.stride(1),
    )
    tok_c, out_c, req_c = tok.cpu(), out.cpu(), req_id.cpu()
    bt_c = block_table.cpu()
    n_valid_ref = torch.zeros(num_tokens, dtype=torch.int32)
    for t in range(num_tokens):
        for j in range(topk):
            g = int(tok_c[t, j])
            if g < 0 or brute_owner(g, world, interleave) != rank:
                assert out_c[t, j] == -1, (t, j, g, int(out_c[t, j]))
                continue
            lp = brute_local(g, world, interleave)
            ref = int(bt_c[req_c[t], lp // block_size]) * block_size + lp % block_size
            assert out_c[t, j] == ref, (t, j, g, lp, ref, int(out_c[t, j]))
            n_valid_ref[t] += 1
    assert (valid.cpu() == n_valid_ref).all(), (valid.cpu(), n_valid_ref)
    print(f"convert OK world={world} rank={rank} il={interleave} bs={block_size}")


def test_prefill_ke(world, rank, interleave):
    # 2 requests: seq lens 700/1300, query lens 5/9 (chunked prefill tail)
    seq_lens = [700, 1300]
    query_lens = [5, 9]
    num_reqs = 2
    qsl = torch.tensor([0, 5, 14], dtype=torch.int32, device=dev)
    useq = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    local_lens = [
        sum(1 for p in range(L) if brute_owner(p, world, interleave) == rank)
        for L in seq_lens
    ]
    cu = torch.tensor([0, local_lens[0], local_lens[0] + local_lens[1]],
                      dtype=torch.int32, device=dev)
    total_q = 14
    tts = torch.empty(cu[-1].item() or 1, dtype=torch.int32, device=dev)
    ks = torch.empty(total_q, dtype=torch.int32, device=dev)
    ke = torch.empty(total_q, dtype=torch.int32, device=dev)
    pfx[(num_reqs,)](
        qsl, useq, cu, tts, ks, ke, 0, total_q,
        BLOCK_SIZE=1024, COMPRESS_RATIO=1,
        DCP_WORLD=world, DCP_RANK=rank, CP_INTERLEAVE=interleave,
    )
    ks_c, ke_c = ks.cpu(), ke.cpu()
    row = 0
    for r in range(num_reqs):
        start_pos = seq_lens[r] - query_lens[r]
        seq_start = int(cu[r])
        for o in range(query_lens[r]):
            gl = start_pos + 1 + o  # causal global prefix
            loc = sum(
                1 for p in range(gl) if brute_owner(p, world, interleave) == rank
            )
            assert ks_c[row] == seq_start, (r, o, int(ks_c[row]), seq_start)
            assert ke_c[row] == seq_start + loc, (r, o, int(ke_c[row]), seq_start + loc)
            row += 1
    print(f"prefill ks/ke OK world={world} rank={rank} il={interleave}")


for world, rank in ((1, 0), (4, 0), (4, 2), (4, 3), (2, 1)):
    for interleave in (1, 256):
        test_convert(world, rank, interleave, 256)
        test_prefill_ke(world, rank, interleave)
test_convert(4, 1, 64, 64)
print("ALL-GPU-OK")
