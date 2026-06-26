# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cubit hand-written SASS sparse-MLA decode (SM120, experimental, opt-in).

Replaces the Triton pair `accumulate_fp8ds_global_slots_sparse_mla_attention_
chunk_multihead` + `finish_sparse_mla_attention_with_sink` with ONE fused
hand-scheduled SASS kernel (`mla_decode_cache_mw8` from the cubit assembler):
fp8 QMMA QK + softmax-with-sink + bf16 HMMA PV, reading the paged `fp8_ds_mla`
cache directly (448B fp8-e4m3 NoPE + per-64-block UE8M0 scales + 128B bf16
RoPE per token, 64-token blocks).

Enable with VLLM_SPARSE_MLA_CUBIT=1. Requirements / notes:
  - SM120, 16 active heads, head_dim 512 (448 NoPE + 64 RoPE), cache block 64.
  - The kernel quantizes Q to fp8 (e4m3 + UE8M0 per-64-block scales); vs the
    bf16-Q Triton path this introduces ~2-3e-2 relative output difference.
  - Launches via the CUDA driver API (ctypes) on the current torch context;
    NOT CUDA-graph capturable - the wrapper detects capture and falls back.
  - Candidate slots are masked to -1 beyond each token's length and padded to
    a multiple of 64 (kernel contract: the 8-warp QK phase tiles candidates in
    units of 64; N <= 512; -1 = invalid).
  - The cubin is taken from VLLM_SPARSE_MLA_CUBIT_CUBIN, or assembled on first
    use from the cubit repo (VLLM_SPARSE_MLA_CUBIT_REPO, default
    cubit).

Any unsupported shape or setup failure returns False and the caller falls
back to the Triton path.
"""

import ctypes
import os
import subprocess

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

LOG2E = 1.4426950408889634
NOPE, ROPE, HEADS, OUT_DIM = 448, 64, 16, 512
NOPE_TILES, ROPE_TILES = NOPE // 32, ROPE // 16
QUANT_BLK, NSCALE = 64, NOPE // 64
FP8_MAX = 448.0
OUT_TILES = 64
_OUT_BYTES = 0x8000          # per-launch O slab (64 C-frag tiles x 32 lanes x 4 f32)
_KERN = b"mla_decode_cache_mw8"

# ---------------------------------------------------------------------------
# fragment-layout index maps (fixed permutations; built once, cached per device)
# ---------------------------------------------------------------------------


def _build_maps():
    nope = np.zeros((NOPE_TILES, 32, 4, 4), np.int64)
    for kt in range(NOPE_TILES):
        for lane in range(32):
            g, t = lane // 4, lane % 4
            for w, (hh, off) in enumerate([(g, 4 * t), (g + 8, 4 * t),
                                           (g, 16 + 4 * t), (g + 8, 16 + 4 * t)]):
                for b in range(4):
                    nope[kt, lane, w, b] = hh * NOPE + (32 * kt + off + b)
    rope = np.zeros((ROPE_TILES, 32, 4, 2), np.int64)
    for kt in range(ROPE_TILES):
        for lane in range(32):
            g, t = lane // 4, lane % 4
            for w, (hh, off) in enumerate([(g, 2 * t), (g + 8, 2 * t),
                                           (g, 2 * t + 8), (g + 8, 2 * t + 8)]):
                for p in range(2):
                    rope[kt, lane, w, p] = hh * ROPE + (16 * kt + off + p)
    lane2row = np.full(32, -1, np.int64)
    for r in range(HEADS):
        lane2row[(r % 8) * 4 + (r // 8)] = r
    scale_blk = np.array([kt // 2 for kt in range(NOPE_TILES)], np.int64)
    o_lane = np.zeros((HEADS, 8), np.int64)
    o_word = np.zeros((HEADS, 8), np.int64)
    for h in range(HEADS):
        g = h % 8
        for j in range(8):
            o_lane[h, j] = g * 4 + (j // 2)
            o_word[h, j] = (0 if h < 8 else 2) + (j % 2)
    return nope, rope, lane2row, scale_blk, o_lane, o_word


_NOPE, _ROPE, _LANE2ROW, _SCALE_BLK, _O_LANE, _O_WORD = _build_maps()
_DEV_MAPS: dict = {}


def _maps(device):
    m = _DEV_MAPS.get(device)
    if m is None:
        t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
        m = dict(nope=t(_NOPE.reshape(-1)), rope=t(_ROPE.reshape(-1)),
                 lane2row=t(_LANE2ROW), scale_blk=t(_SCALE_BLK),
                 o_lane=t(_O_LANE.reshape(-1)), o_word=t(_O_WORD.reshape(-1)))
        _DEV_MAPS[device] = m
    return m


def _quant_ue8m0(x):
    """x[...,448] f32 -> (fp8 bits u8, scale bytes u8[...,7]) per-64-block UE8M0."""
    *batch, _ = x.shape
    xb = x.reshape(*batch, NSCALE, QUANT_BLK)
    amax = xb.abs().amax(-1).clamp_min(1e-4)
    exp = torch.ceil(torch.log2(amax / FP8_MAX))
    q = (xb / torch.exp2(exp)[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q.view(torch.uint8).reshape(*batch, NOPE),
            (exp + 127.0).clamp(0, 255).to(torch.uint8))


def _pack_q_into(q16: torch.Tensor, b: dict) -> int:
    """Pack q16 [T,16,512] bf16 into the persistent buffers b['qn'/'sfa'/'qr'][:T]."""
    dev = q16.device
    m = _maps(dev)
    T = q16.shape[0]
    fp8, sb = _quant_ue8m0(q16[:, :, :NOPE].to(torch.float32))
    g = fp8.reshape(T, HEADS * NOPE).to(torch.int32).index_select(1, m["nope"])
    g = g.reshape(T, NOPE_TILES * 32, 4, 4)
    b["qn"][:T].copy_(g[..., 0] | (g[..., 1] << 8) | (g[..., 2] << 16) | (g[..., 3] << 24))

    valid = m["lane2row"] >= 0
    by_lane = sb.to(torch.int32).index_select(1, m["lane2row"].clamp_min(0))  # [T,32,7]
    sel = by_lane.index_select(2, m["scale_blk"]).movedim(2, 1)               # [T,14,32]
    packed = (torch.full_like(sel, 0x7F7F7F7F) & ~0xFF) | sel
    b["sfa"][:T].copy_(
        torch.where(valid.expand_as(packed), packed,
                    torch.full_like(packed, 0x7F7F7F7F)).reshape(T, -1))

    qrb = q16[:, :, NOPE:].contiguous().view(torch.uint16)
    g = qrb.reshape(T, HEADS * ROPE).to(torch.int32).index_select(1, m["rope"])
    g = g.reshape(T, ROPE_TILES * 32, 4, 2)
    b["qr"][:T].copy_(g[..., 0] | (g[..., 1] << 16))
    return T


def _unpack_o(c: torch.Tensor):
    """c: [T, 64*32, 4] f32 (C-frag) -> O[T,16,512] f32 (same device)."""
    m = _maps(c.device)
    T = c.shape[0]
    sel = c.reshape(T, OUT_TILES, 32, 4).index_select(2, m["o_lane"])        # [T,64,128,4]
    w = m["o_word"].view(1, 1, HEADS * 8, 1).expand(T, OUT_TILES, HEADS * 8, 1)
    sel = torch.gather(sel, 3, w).squeeze(3).reshape(T, OUT_TILES, HEADS, 8)
    return sel.movedim(2, 1).reshape(T, HEADS, OUT_DIM)


# ---------------------------------------------------------------------------
# driver-API launcher
# ---------------------------------------------------------------------------

_cu = None
_fns: dict = {}              # kind ("decode"|"state") -> function handle
_state = "uninit"            # uninit | ready | unavailable
# Default directory for cubit cubins (assemble-on-first-use outputs + the search
# fallback). Inside the serving container this is a HOST-PERSISTENT dir bind-mounted
# by serve_w2.sh; it is deliberately NOT /tmp, which is wiped on reboot and silently
# drops every kernel back to the slow Triton fallback. Override with the same
# VLLM_MOE_W2_CUBIT_DIR the rest of the stack uses.
_CUBIN_DIR = os.getenv("VLLM_MOE_W2_CUBIT_DIR") or "/cubit-share"
_KERNELS = {
    "decode": (b"mla_decode_cache_mw8", "sass/mla_decode_cache_mw8.sass",
               "VLLM_SPARSE_MLA_CUBIT_CUBIN",
               os.path.join(_CUBIN_DIR, "vllm_cubit_mla_mw8.cubin")),
    # chunked state kernel: grid (nchunks, head_blocks); CTA (cx, cy) reduces
    # candidate chunk cx (64 slots) for 16-head block cy to an online-softmax state
    "cstate": (os.getenv("VLLM_SPARSE_MLA_CUBIT_CSTATE_KERNEL",
                         "mla_decode_cache_mw8_cstate").encode(),
               os.getenv("VLLM_SPARSE_MLA_CUBIT_CSTATE_SASS",
                         "sass/mla_decode_cache_mw8_cstate.sass"),
               "VLLM_SPARSE_MLA_CUBIT_CSTATE_CUBIN",
               os.path.join(_CUBIN_DIR, "vllm_cubit_mla_mw8_cstate.cubin")),
    "cchunk": (b"mla_decode_cchunk",
               os.getenv("VLLM_SPARSE_MLA_CUBIT_CCHUNK_SASS",
                         "sass/mla_decode_cchunk.sass"),
               "VLLM_SPARSE_MLA_CUBIT_CCHUNK_CUBIN",
               os.path.join(_CUBIN_DIR, "vllm_cubit_mla_cchunk.cubin")),
    "merge":  (b"mla_state_merge",
               os.getenv("VLLM_SPARSE_MLA_CUBIT_MERGE_SASS",
                         "sass/mla_state_merge.sass"),
               "VLLM_SPARSE_MLA_CUBIT_MERGE_CUBIN",
               os.path.join(_CUBIN_DIR, "vllm_cubit_mla_merge.cubin")),
    "qpack":  (b"mla_qpack",
               os.getenv("VLLM_SPARSE_MLA_CUBIT_QPACK_SASS",
                         "sass/mla_qpack.sass"),
               "VLLM_SPARSE_MLA_CUBIT_QPACK_CUBIN",
               os.path.join(_CUBIN_DIR, "vllm_cubit_mla_qpack.cubin")),
}
_NHB_MAX = 4                 # up to 64 TP-local heads (4 x 16-head blocks)
_NCHUNK_MAX = 10             # 512 compressed (8 chunks) + 128 SWA (2) in one launch
LN2 = 0.6931471805599453
_STATE_SLAB = 0x8100         # acc C-frags (0x8000) + m2[16] f32 + denom[16] f32, padded
# Max decode tokens per call: larger decode batches fall back to Triton. Kept small
# so the packing transients stay tiny in every captured CUDA graph's private pool
# (each captured size pools its own peak transients; at 1024 this inflated the graph
# memory estimate to ~9 GiB and starved the KV cache).
_MAX_T = int(os.getenv("VLLM_SPARSE_MLA_CUBIT_MAX_T", "64"))
_MAX_N = 640                 # slots row: 512 compressed + 128 SWA (fused dual-subset)
_BUFS: dict = {}             # device -> persistent kernel I/O buffers


def _bufs(device):
    """Persistent kernel I/O buffers (~45 MB), allocated ONCE per device in eager.
    The kernel-argument pointers and the graph-pool footprint stay tiny and stable:
    per-call allocations here would otherwise be duplicated into every captured
    CUDA graph's private pool (observed: 9 GiB graph-memory estimate vs 3.5 GiB
    baseline -> KV-cache OOM)."""
    b = _BUFS.get(device)
    if b is None:
        t = lambda *s, dt=torch.int32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        b = dict(
            qn=t(_MAX_T * _NHB_MAX, NOPE_TILES * 32, 4),
            sfa=t(_MAX_T * _NHB_MAX, NOPE_TILES * 32),
            qr=t(_MAX_T * _NHB_MAX, ROPE_TILES * 32, 4),
            slots=t(_MAX_T, _MAX_N),
            obuf=t(_MAX_T * _NHB_MAX * _NCHUNK_MAX * (_STATE_SLAB // 4),
                   dt=torch.float32),
            sink2=t(_NHB_MAX * HEADS, dt=torch.float32),
            npad=t(1),
            j=torch.arange(_MAX_N, device=device, dtype=torch.int32),
        )
        _BUFS[device] = b
    return b


def _driver():
    global _cu
    if _cu is None:
        cu = ctypes.CDLL("libcuda.so.1")
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [
            ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        cu.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                           ctypes.c_char_p]
        _cu = cu
    return _cu


def _ck(r, what):
    if r:
        raise RuntimeError(f"cubit sparse-MLA: CUDA error {r} in {what}")


def _cubin_path(kind: str) -> str:
    kern, sass, env, out = _KERNELS[kind]
    pre = os.getenv(env)
    if pre:
        if not os.path.isfile(pre):
            raise FileNotFoundError(pre)
        return pre
    repo = os.getenv("VLLM_SPARSE_MLA_CUBIT_REPO", "cubit")
    r = subprocess.run(
        [os.path.join(repo, "target/release/cubit"), "asm",
         os.path.join(repo, sass), "-o", out, "--kernel", kern.decode(),
         "--mercury-stub", os.path.join(repo, "sass/qmma_e4m3.merc.stub")],
        capture_output=True, text=True, cwd=repo)
    if "0 failed" not in (r.stdout + r.stderr):
        raise RuntimeError(f"cubit asm failed: {r.stdout[-500:]} {r.stderr[-500:]}")
    return out


def _ensure_ready() -> bool:
    global _state
    if _state == "ready":
        return True
    if _state == "unavailable":
        return False
    try:
        cu = _driver()
        for kind, (kern, _, _, _) in _KERNELS.items():
            mod = ctypes.c_void_p()
            _ck(cu.cuModuleLoad(ctypes.byref(mod), _cubin_path(kind).encode()),
                "cuModuleLoad")
            fn = ctypes.c_void_p()
            _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, kern),
                "cuModuleGetFunction")
            _fns[kind] = fn
        # Pre-build the device index maps and persistent I/O buffers NOW (eager):
        # the maps are numpy->GPU copies (illegal inside CUDA-graph capture) and the
        # buffers must not land in per-graph private pools.
        dev = torch.device("cuda", torch.cuda.current_device())
        _maps(dev)
        _bufs(dev)
        _state = "ready"
        logger.info("cubit sparse-MLA kernels loaded (decode + state)")
        return True
    except Exception as e:  # noqa: BLE001 - any setup failure means Triton fallback
        logger.warning("cubit sparse-MLA unavailable (%s); using Triton fallback", e)
        _state = "unavailable"
        return False


def cubit_sparse_mla_decode(
    q: torch.Tensor,            # [T, padded_heads, 512] bf16 (first 16 heads active)
    k_cache: torch.Tensor,      # fp8_ds_mla paged cache, uint8
    slot_ids: torch.Tensor,     # [T, N] or [T, 1, N] int32 global slots (-1 invalid)
    lens: torch.Tensor,         # [T] int32 valid-candidate counts
    block_size: int,
    scale: float,
    attn_sink: torch.Tensor,    # [>=16] f32
    output: torch.Tensor,       # [T, padded_heads, 512] bf16 (heads 16: written here)
    num_heads: int,
) -> bool:
    """Fused decode via the cubit SASS kernel. Returns False when the shape /
    environment is unsupported (caller must then run the Triton path)."""
    if torch.cuda.is_current_stream_capturing() and (
            _state != "ready" or q.device not in _DEV_MAPS):
        return False               # cannot load modules / build maps mid-capture
    if not _ensure_ready():
        return False
    if q.dim() == 4:                # [T, 1, padded_heads, dim] form
        q = q[:, 0]
    # The kernel computes 16 heads per launch; larger TP-local head counts are
    # served in independent 16-head blocks (attention is head-independent).
    if num_heads % HEADS != 0 or q.shape[-1] != OUT_DIM or block_size != QUANT_BLK:
        return False
    if k_cache.dtype != torch.uint8:
        return False
    if slot_ids.dim() == 3:
        slot_ids = slot_ids[:, 0]
    T, n_raw = slot_ids.shape
    if n_raw > 512 or n_raw == 0 or T == 0:
        return False
    if not torch.cuda.is_current_stream_capturing() and int(lens.min()) == 0:
        # Empty subset -> all-(-inf) softmax; let Triton handle it. (Host sync: only
        # checked in eager. Under graph capture the caller must guarantee lens > 0.)
        return False

    if T > _MAX_T:
        return False
    for hb in range(0, num_heads, HEADS):
        slab = _launch(_fns["decode"], q[:, hb:hb + HEADS], k_cache, slot_ids, lens,
                       scale, attn_sink[hb:hb + HEADS], _OUT_BYTES)
        o = _unpack_o(slab.reshape(T, OUT_TILES * 32, 4))
        output[:T, hb:hb + HEADS] = o.to(output.dtype)
    return True


def _launch(fn, q, k_cache, slot_ids, lens, scale, attn_sink, slab_bytes):
    """Shared core: mask/pad candidates, pack Q, launch one grid-1 kernel per token.
    All kernel I/O lives in the persistent per-device buffers; returns the obuf
    slab view [T, slab_bytes/4] f32 (GPU)."""
    T, n_raw = slot_ids.shape
    b = _bufs(q.device)
    # mask beyond-length entries to -1 and pad N to a multiple of 64 (the 8-warp
    # QK phase processes candidates in 64-token tiles: tiles/warp = N>>6)
    n_pad = (n_raw + 63) & ~63
    slots = b["slots"][:T, :n_pad]
    if n_pad != n_raw:
        slots[:, n_raw:].fill_(-1)
    slots[:, :n_raw].copy_(slot_ids)
    slots[:, :n_raw].masked_fill_(b["j"][:n_raw][None, :] >= lens[:T, None], -1)

    _pack_q_into(q[:, :HEADS], b)
    b["sink2"][:HEADS].copy_(attn_sink[:HEADS].to(torch.float32) * LOG2E)
    qn, sfa, qr, obuf = b["qn"], b["sfa"], b["qr"], b["obuf"]
    scale2 = int(np.float32(scale * LOG2E).view(np.uint32))

    qn_s, sfa_s = qn.stride(0) * 4, sfa.stride(0) * 4
    qr_s, slots_s = qr.stride(0) * 4, b["slots"].stride(0) * 4
    cu = _driver()
    # Stream-ordered launch on torch's CURRENT stream: ordered after the packing ops
    # and before the unpack below, with no device syncs — eager-correct and CUDA-graph
    # capturable (during capture the driver records the launches into the graph; the
    # argument pointers are the persistent buffers, identical on every replay).
    stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
    keep = []
    for t in range(T):
        a = [ctypes.c_uint64(qn.data_ptr() + t * qn_s),
             ctypes.c_uint64(qr.data_ptr() + t * qr_s),
             ctypes.c_uint64(sfa.data_ptr() + t * sfa_s),
             ctypes.c_uint64(b["sink2"].data_ptr()),
             ctypes.c_uint64(k_cache.data_ptr()),
             ctypes.c_uint64(b["slots"].data_ptr() + t * slots_s),
             ctypes.c_uint64(obuf.data_ptr() + t * _STATE_SLAB),
             ctypes.c_uint32(scale2 & 0xFFFFFFFF),
             ctypes.c_uint32(n_pad),
             ctypes.c_uint32(int(k_cache.stride(0)))]
        argv = (ctypes.c_void_p * len(a))(
            *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in a])
        keep.append((a, argv))
        _ck(cu.cuLaunchKernel(fn, 1, 1, 1, 256, 1, 1, 0, stream, argv, None), "launch")
    del keep
    return obuf[: _MAX_T * (_STATE_SLAB // 4)].view(
        _MAX_T, _STATE_SLAB // 4)[:T, :slab_bytes // 4]


def cubit_sparse_mla_state(
    q: torch.Tensor,            # [T, padded_heads, 512] bf16 (first 16 heads active)
    k_cache: torch.Tensor,      # fp8_ds_mla paged cache (64-token blocks), uint8
    slot_ids: torch.Tensor,     # [T, N] or [T, 1, N] int32 global slots (-1 invalid)
    lens: torch.Tensor,         # [T] int32 valid-candidate counts (must be > 0)
    block_size: int,
    scale: float,
    max_score: torch.Tensor,    # [T, 16] f32 out (ln-domain max)
    denom: torch.Tensor,        # [T, 16] f32 out
    acc: torch.Tensor,          # [T, 16, 512] f32 out (unnormalized)
    num_heads: int,
    skip_pack: bool = False,    # reuse Q fragments packed by the previous call
) -> bool:
    """Online-softmax STATE of one candidate subset (no sink, unnormalized), matching
    the contract of `accumulate_fp8ds_global_slots_..._multihead`, so the result can be
    merged with another subset via `finish_two_sparse_mla_attention_states_with_sink`.
    Returns False when unsupported (caller must run the Triton accumulate instead)."""
    if torch.cuda.is_current_stream_capturing() and (
            _state != "ready" or q.device not in _DEV_MAPS):
        return False
    if not _ensure_ready():
        return False
    if q.dim() == 4:
        q = q[:, 0]
    if num_heads % HEADS != 0 or q.shape[-1] != OUT_DIM or block_size != QUANT_BLK:
        return False
    if k_cache.dtype != torch.uint8:
        return False
    if slot_ids.dim() == 3:
        slot_ids = slot_ids[:, 0]
    T, n_raw = slot_ids.shape
    if n_raw > 512 or n_raw == 0 or T == 0:
        return False
    if not torch.cuda.is_current_stream_capturing() and int(lens.min()) == 0:
        return False

    if T > _MAX_T:
        return False
    m_ln, d, a = _cstate_launch(q, k_cache, slot_ids, lens, scale, num_heads,
                                skip_pack=skip_pack)
    max_score[:T, :num_heads] = m_ln
    denom[:T, :num_heads] = d
    acc[:T, :num_heads] = a
    return True


def _cstate_launch(q, k_cache, slot_ids, lens, scale, num_heads, skip_pack=False):
    """ONE chunked-state launch per token, grid (nchunks, head_blocks): every CTA
    reduces a 64-candidate chunk for one 16-head block in parallel (~wall time of a
    single chunk instead of N/64 x head_blocks sequential grid-1 kernels). Chunk
    states are merged here (log2-domain, empty chunks masked) into one subset state:
    (max_ln [T,H], denom [T,H], unnormalized acc [T,H,512]).
    skip_pack=True reuses the Q fragments already packed by a previous call (the
    two candidate subsets of one layer share the same Q)."""
    T, n_raw = slot_ids.shape
    nhb = num_heads // HEADS
    n_pad = (n_raw + 63) & ~63
    nchunks = n_pad // 64
    b = _bufs(q.device)

    slots = b["slots"][:T, :n_pad]
    if n_pad != n_raw:
        slots[:, n_raw:].fill_(-1)
    slots[:, :n_raw].copy_(slot_ids)
    slots[:, :n_raw].masked_fill_(b["j"][:n_raw][None, :] >= lens[:T, None], -1)
    b["npad"].fill_(n_pad)        # rows are pre-masked; in-kernel lens mask idles

    if not skip_pack:
        _pack_q_into(q[:, :num_heads].reshape(T * nhb, HEADS, OUT_DIM), b)
    scale2 = int(np.float32(scale * LOG2E).view(np.uint32))
    qn, sfa, qr, obuf = b["qn"], b["sfa"], b["qr"], b["obuf"]
    qn_s, sfa_s = qn.stride(0) * 4, sfa.stride(0) * 4
    qr_s, slots_s = qr.stride(0) * 4, b["slots"].stride(0) * 4
    tok_slab = nhb * nchunks * _STATE_SLAB
    cu = _driver()
    stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
    keep = []
    for t in range(T):
        srow = b["slots"].data_ptr() + t * slots_s
        a = [ctypes.c_uint64(qn.data_ptr() + t * nhb * qn_s),
             ctypes.c_uint64(qr.data_ptr() + t * nhb * qr_s),
             ctypes.c_uint64(sfa.data_ptr() + t * nhb * sfa_s),
             ctypes.c_uint64(b["sink2"].data_ptr()),       # unused by the state kernel
             ctypes.c_uint64(k_cache.data_ptr()),
             ctypes.c_uint64(srow),
             ctypes.c_uint64(obuf.data_ptr() + t * tok_slab),
             ctypes.c_uint32(scale2 & 0xFFFFFFFF),
             ctypes.c_uint32(64),
             ctypes.c_uint32(int(k_cache.stride(0))),
             ctypes.c_uint32(nchunks),
             ctypes.c_uint64(k_cache.data_ptr()),
             ctypes.c_uint32(nchunks),
             ctypes.c_uint32(int(k_cache.stride(0))),
             ctypes.c_uint64(srow),
             ctypes.c_uint64(b["npad"].data_ptr()),
             ctypes.c_uint64(b["npad"].data_ptr()),
             ctypes.c_uint32(0),    # slots1 tok-stride (unused: grid Z=1)
             ctypes.c_uint32(0),    # slots2 tok-stride
             ctypes.c_uint32(nhb)]
        argv = (ctypes.c_void_p * len(a))(
            *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in a])
        keep.append((a, argv))
        _ck(cu.cuLaunchKernel(_fns["cstate"], nchunks, nhb, 1, 256, 1, 1, 0,
                              stream, argv, None), "launch")
    del keep

    slab_w = _STATE_SLAB // 4
    v = obuf[: T * nhb * nchunks * slab_w].view(T, nhb, nchunks, slab_w)
    m2 = v[..., 0x2000:0x2000 + 16].permute(0, 1, 3, 2).reshape(T, num_heads, nchunks)
    dn = v[..., 0x2010:0x2010 + 16].permute(0, 1, 3, 2).reshape(T, num_heads, nchunks)
    big_m2 = m2.amax(-1)                                       # [T,H] log2-domain
    w = torch.exp2(m2 - big_m2[..., None])
    w = torch.where(torch.isnan(w) | torch.isneginf(m2), torch.zeros_like(w), w)
    # fully-masked chunks emit NaN denom/acc (exp2 over all -inf logits): w=0 must
    # zero them, so scrub NaNs before the weighted sums
    d = (torch.nan_to_num(dn) * w).sum(-1)
    if T * nhb * nchunks <= 256:
        # batched unpack of every chunk + one weighted reduction (fewest kernels;
        # transient [T*nhb*C,16,512] f32 is small for the latency-critical sizes)
        u = _unpack_o(v[..., :_OUT_BYTES // 4].reshape(-1, OUT_TILES * 32, 4))
        u = u.view(T, nhb, nchunks, HEADS, OUT_DIM).movedim(3, 2)   # [T,nhb,16,C,D]
        u = u.reshape(T, num_heads, nchunks, OUT_DIM)
        acc = (torch.nan_to_num(u) * w[..., None]).sum(2)
    else:
        acc = torch.zeros(T, num_heads, OUT_DIM, dtype=torch.float32, device=q.device)
        for c in range(nchunks):
            u = _unpack_o(v[:, :, c, :_OUT_BYTES // 4]
                          .reshape(T * nhb, OUT_TILES * 32, 4))
            acc += torch.nan_to_num(u.view(T, num_heads, OUT_DIM)) * w[..., c, None]
    return big_m2 * LN2, d, acc


def cubit_sparse_mla_fused_single(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    lens: torch.Tensor,
    block_size: int,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
) -> bool:
    """Single-subset fused decode (e.g. the SWA-only / MTP-draft layers): the
    dual-subset pipeline with every chunk bound to subset 1. Token-batched and
    graph-capturable like the dual variant."""
    return cubit_sparse_mla_fused_decode(
        q=q, k_cache1=k_cache, slot_ids1=slot_ids, lens1=lens,
        k_cache2=k_cache, slot_ids2=slot_ids, lens2=lens,
        block_size=block_size, scale=scale, attn_sink=attn_sink,
        output=output, num_heads=num_heads, single_subset=True)


def cubit_sparse_mla_fused_decode(
    q: torch.Tensor,            # [T, padded_heads, 512] bf16
    k_cache1: torch.Tensor,     # compressed fp8_ds_mla paged cache, uint8
    slot_ids1: torch.Tensor,    # [T, N1] or [T, 1, N1] int32 (-1 invalid)
    lens1: torch.Tensor,        # [T] int32 (> 0)
    k_cache2: torch.Tensor,     # SWA fp8_ds_mla paged cache, uint8
    slot_ids2: torch.Tensor,    # [T, N2] int32
    lens2: torch.Tensor,        # [T] int32 (> 0)
    block_size: int,
    scale: float,
    attn_sink: torch.Tensor,    # [>= num_heads] f32
    output: torch.Tensor,       # [T, padded_heads, 512] bf16, written in place
    num_heads: int,
    single_subset: bool = False,  # bind every chunk to subset 1 (SWA-only/draft)
) -> bool:
    """Fully fused dual-subset decode: ONE `mla_decode_cchunk` launch covers both
    candidate subsets (chunks [0, C1) gather from k_cache1, [C1, C) from k_cache2;
    the slot rows are concatenated), then ONE `mla_state_merge` launch k-way-merges
    the chunk states with the attention sink and writes bf16 O directly into
    `output` - no torch-side state extraction at all. Returns False when the
    shape/environment is unsupported (caller falls back)."""
    if torch.cuda.is_current_stream_capturing() and (
            _state != "ready" or q.device not in _DEV_MAPS):
        return False
    if not _ensure_ready():
        return False
    if q.dim() == 4:
        q = q[:, 0]
    if num_heads % HEADS != 0 or q.shape[-1] != OUT_DIM or block_size != QUANT_BLK:
        return False
    if k_cache1.dtype != torch.uint8 or k_cache2.dtype != torch.uint8:
        return False
    if slot_ids1.dim() == 3:
        slot_ids1 = slot_ids1[:, 0]
    if slot_ids2.dim() == 3:
        slot_ids2 = slot_ids2[:, 0]
    # merge kernel writes output rows at the fixed 512-element head stride;
    # qpack kernel reads q head rows at the same stride
    if output.stride(-1) != 1 or output.stride(-2) != OUT_DIM:
        return False
    if q.stride(-1) != 1 or q.stride(-2) != OUT_DIM:
        return False
    # raw index rows / length scalars are read directly by the kernels
    if (slot_ids1.stride(-1) != 1 or slot_ids2.stride(-1) != 1
            or slot_ids1.dtype != torch.int32 or slot_ids2.dtype != torch.int32
            or lens1.dtype != torch.int32 or lens2.dtype != torch.int32
            or lens1.stride(-1) != 1 or lens2.stride(-1) != 1
            or attn_sink.dtype != torch.float32 or attn_sink.stride(-1) != 1):
        return False
    T, n1 = slot_ids1.shape
    T2, n2 = slot_ids2.shape
    n1_pad = (n1 + 63) & ~63
    n2_pad = 0 if single_subset else (n2 + 63) & ~63
    if (T != T2 or T == 0 or T > _MAX_T or n1 == 0
            or (n2 == 0 and not single_subset)
            or n1_pad + n2_pad > _MAX_N):
        return False
    # NOTE: no lens > 0 requirement (no host sync): empty subsets/chunks yield
    # -inf states which the merge kernel weighs to zero against the sink.

    nhb = num_heads // HEADS
    nchunks1, nchunks = n1_pad // 64, (n1_pad + n2_pad) // 64
    b = _bufs(q.device)
    scale2 = int(np.float32(scale * LOG2E).view(np.uint32))
    qn, sfa, qr, obuf = b["qn"], b["sfa"], b["qr"], b["obuf"]
    qn_s, sfa_s, qr_s = qn.stride(0) * 4, sfa.stride(0) * 4, qr.stride(0) * 4
    s1_s, s2_s = slot_ids1.stride(0) * 4, slot_ids2.stride(0) * 4
    out_s = output.stride(0) * 2
    q_s = q.stride(0) * 2
    if T > 1 and (lens1.stride(0) != 1 or lens2.stride(0) != 1):
        return False              # kernels index lens rows at a fixed 4B stride
    cu = _driver()
    stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
    # ONE launch trio covers the whole (MTP/speculative) token batch: the kernels
    # select the token via CTAID (qpack: Y; cchunk/merge: Z) and advance their
    # q/slot/len/slab/output bases by the per-token strides passed below.
    keep = []
    p = [ctypes.c_uint64(q.data_ptr()),
         ctypes.c_uint64(qn.data_ptr()),
         ctypes.c_uint64(sfa.data_ptr()),
         ctypes.c_uint64(qr.data_ptr()),
         ctypes.c_uint32(q_s),
         ctypes.c_uint32(nhb)]
    pargv = (ctypes.c_void_p * len(p))(
        *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in p])
    keep.append((p, pargv))
    _ck(cu.cuLaunchKernel(_fns["qpack"], nhb, T, 1, 256, 1, 1, 0,
                          stream, pargv, None), "launch qpack")
    a = [ctypes.c_uint64(qn.data_ptr()),
         ctypes.c_uint64(qr.data_ptr()),
         ctypes.c_uint64(sfa.data_ptr()),
         ctypes.c_uint64(attn_sink.data_ptr()),        # unused by cchunk
         ctypes.c_uint64(k_cache1.data_ptr()),
         ctypes.c_uint64(slot_ids1.data_ptr()),
         ctypes.c_uint64(obuf.data_ptr()),
         ctypes.c_uint32(scale2 & 0xFFFFFFFF),
         ctypes.c_uint32(64),
         ctypes.c_uint32(int(k_cache1.stride(0))),
         ctypes.c_uint32(nchunks),
         ctypes.c_uint64(k_cache2.data_ptr()),
         ctypes.c_uint32(nchunks1),
         ctypes.c_uint32(int(k_cache2.stride(0))),
         ctypes.c_uint64(slot_ids2.data_ptr()),
         ctypes.c_uint64(lens1.data_ptr()),
         ctypes.c_uint64(lens2.data_ptr()),
         ctypes.c_uint32(s1_s),
         ctypes.c_uint32(s2_s),
         ctypes.c_uint32(nhb)]
    argv = (ctypes.c_void_p * len(a))(
        *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in a])
    keep.append((a, argv))
    _ck(cu.cuLaunchKernel(_fns["cchunk"], nchunks, nhb, T, 256, 1, 1, 0,
                          stream, argv, None), "launch cchunk")
    m = [ctypes.c_uint64(obuf.data_ptr()),
         ctypes.c_uint64(attn_sink.data_ptr()),        # raw ln-domain sink
         ctypes.c_uint64(output.data_ptr()),
         ctypes.c_uint32(nchunks),
         ctypes.c_uint32(out_s),
         ctypes.c_uint32(nhb)]
    margv = (ctypes.c_void_p * len(m))(
        *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in m])
    keep.append((m, margv))
    _ck(cu.cuLaunchKernel(_fns["merge"], nhb, 4, T, 256, 1, 1, 0,
                          stream, margv, None), "launch merge")
    del keep
    return True


# ===========================================================================
# PREFILL accumulate (mla_prefill_state2: fp8 smem staging, 2 CTA/SM, ~2.2x
# vs Triton at T=256). Drop-in for the Triton accumulate_fp8ds_global_slots_
# sparse_mla_attention_chunk_multihead in the cache-direct prefill path,
# behind VLLM_SPARSE_MLA_PREFILL_CUBIT (default OFF). Self-contained: loads
# its own cubin + maps so a missing prefill cubin never affects the decode path.
# Contract: q[T,H,512] bf16, fp8_ds_mla k_cache, slot_ids[T,C] (-1 invalid),
# lens[T], in-place online state max[T,NH]/denom[T,NH]/acc[T,NH,512] f32
# (natural log domain). Processes 64 candidates/launch (grid.x=T); chunks of
# C>64 are sub-tiled here. Falls back (returns False) on any unsupported shape.
# ===========================================================================
_PF_TILE = 64
# Supported cache paging block sizes (tokens/page). The kernel decodes a global
# slot into (page, offset): page = slot >> log2bs, off = slot & (bs-1). The
# fp8_ds_mla page is SoA [bs x 576B entry][bs x 8B UE8M0 scale], so the per-page
# scale region starts at bs*576 (= scale_base). bs=64 is the C4A compressed leg,
# 2 is C128A, 4/8 are the small-window SWA legs (all powers of two).
_PF_BLOCK_SIZES = (2, 4, 8, 64)
_PF_ENTRY_BYTES = NOPE + 2 * ROPE       # 576 = 448 fp8 NoPE + 128 bf16 RoPE (bs-independent)
# e2e (GPU3, 256k, 2026-06-15) found routing bs!=64 (the C128A bs=2 + small SWA
# legs) onto state2 is NET-NEGATIVE: the per-64-cand sub-tile launches + acc-RMW
# are launch-bound at the C128A candidate count, slower than Triton there (TTFT
# regressed vs the C4A-only prod config). So bs!=64 is OPT-IN (default OFF -> only
# the validated C4A bs=64 leg runs on cubit). Re-enable once the chunked kernel
# (single 3D launch, no per-chunk acc RMW) handles the high-candidate legs.
_PF_BS_ALL = os.environ.get("VLLM_SPARSE_MLA_PREFILL_BS_ALL", "0") == "1"
_PF_KERN = b"mla_prefill_state2"
_pf_fn = None
_pf_state = "uninit"
_PF_LO: dict = {}          # (num_heads, device) -> low-bf16 index map
_PF_BUFS: dict = {}        # device -> dict(slots64) persistent
_PF_MAX_T = 256
_pf_ran_seen: set = set()  # one-time per-(block_size) run/fallback log keys


def _pf_cubin() -> str:
    pre = os.getenv("VLLM_SPARSE_MLA_PREFILL_CUBIN")
    if pre:
        if not os.path.isfile(pre):
            raise FileNotFoundError(pre)
        return pre
    for d in (_CUBIN_DIR, "cubit-share"):
        p = os.path.join(d, "mla_prefill_state.cubin")
        if os.path.isfile(p):
            return p
    # assemble from the cubit repo as a last resort
    repo = os.getenv("VLLM_SPARSE_MLA_CUBIT_REPO", "cubit")
    out = os.path.join(_CUBIN_DIR, "mla_prefill_state.cubin")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = subprocess.run(
        [os.path.join(repo, "target/release/cubit"), "asm",
         os.path.join(repo, "sass/mla_prefill_state2.sass"), "-o", out,
         "--kernel", "mla_prefill_state2",
         "--mercury-stub", os.path.join(repo, "sass/qmma_e4m3.merc.stub")],
        capture_output=True, text=True, cwd=repo)
    if "0 failed" not in (r.stdout + r.stderr):
        raise RuntimeError(f"prefill cubit asm failed: {r.stdout[-400:]} {r.stderr[-400:]}")
    return out


def _pf_ensure() -> bool:
    global _pf_fn, _pf_state
    if _pf_state == "ready":
        return True
    if _pf_state == "unavailable":
        return False
    try:
        cu = _driver()
        mod = ctypes.c_void_p()
        _ck(cu.cuModuleLoad(ctypes.byref(mod), _pf_cubin().encode()), "load prefill")
        fn = ctypes.c_void_p()
        _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, _PF_KERN), "getfn prefill")
        _pf_fn = fn
        _pf_state = "ready"
        logger.info("cubit sparse-MLA PREFILL kernel loaded (mla_prefill_state2)")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("cubit sparse-MLA prefill unavailable (%s); using Triton", e)
        _pf_state = "unavailable"
        return False


def _pf_lo_idx(num_heads: int, device) -> torch.Tensor:
    """bf16 HMMA m16n8k16 A-fragment low-element index map (per token, flattened
    [NHG][32 ktiles][32 lanes][4 u32]); each u32 packs the bf16 at idx and idx+1."""
    key = (num_heads, device)
    t = _PF_LO.get(key)
    if t is None:
        nhg = num_heads // 16
        idx = np.zeros(nhg * 32 * 32 * 4, np.int64)
        p = 0
        for gp in range(nhg):
            for kt in range(32):
                for L in range(32):
                    g, tt = L // 4, L % 4
                    for (hh, off) in ((g, 2 * tt), (g + 8, 2 * tt),
                                      (g, 2 * tt + 8), (g + 8, 2 * tt + 8)):
                        idx[p] = (gp * 16 + hh) * 512 + (16 * kt + off)
                        p += 1
        t = torch.from_numpy(idx).to(device)
        _PF_LO[key] = t
    return t


def _pf_pack_q(q: torch.Tensor, num_heads: int) -> torch.Tensor:
    """q[T,H,512] bf16 -> qn[T, NHG*32*32*4] int32 (HMMA A-frags). Packed fresh
    every call (cheap GPU gather) -- no data_ptr cache (stale-storage hazard)."""
    T = q.shape[0]
    lo = _pf_lo_idx(num_heads, q.device)
    qf = q[:, :num_heads].reshape(T, num_heads * 512).contiguous().view(torch.int16)
    glo = qf.index_select(1, lo).to(torch.int32) & 0xFFFF
    ghi = qf.index_select(1, lo + 1).to(torch.int32) & 0xFFFF
    return (glo | (ghi << 16)).contiguous()


def _pf_slots_buf(device, T: int) -> torch.Tensor:
    b = _PF_BUFS.get(device)
    if b is None:
        b = {"slots64": torch.empty(_PF_MAX_T, _PF_TILE, dtype=torch.int32,
                                    device=device)}
        _PF_BUFS[device] = b
    return b["slots64"]


def cubit_sparse_mla_prefill_accumulate(
    q: torch.Tensor,            # [T, H, 512] bf16 (first NH heads active)
    k_cache: torch.Tensor,      # fp8_ds_mla paged cache (>=2D), uint8
    slot_ids: torch.Tensor,     # [T, C] int32 global slots (-1 invalid)
    lens: torch.Tensor,         # [T] int32 valid-candidate counts
    block_size: int,
    scale: float,
    max_score: torch.Tensor,    # [T, NH] f32 in/out (natural log domain)
    denom: torch.Tensor,        # [T, NH] f32 in/out
    acc: torch.Tensor,          # [T, NH, 512] f32 in/out (unnormalized)
    candidate_offset: int = 0,
) -> bool:
    """In-place online-softmax accumulate of one candidate chunk via the cubit
    SASS prefill kernel. Returns False (caller runs the Triton kernel) on any
    unsupported shape/environment."""
    if not _pf_ensure():
        return False
    if q.dim() == 4:
        q = q[:, 0]
    if slot_ids.dim() == 3:
        slot_ids = slot_ids[:, 0]
    _allowed = _PF_BLOCK_SIZES if _PF_BS_ALL else (QUANT_BLK,)
    if q.dim() != 3 or q.shape[-1] != OUT_DIM or block_size not in _allowed:
        if ("fb", block_size) not in _pf_ran_seen:
            _pf_ran_seen.add(("fb", block_size))
            logger.info("cubit prefill FELL BACK to Triton (block_size=%d head_dim=%d)",
                        block_size, q.shape[-1])
        return False
    if k_cache.dtype != torch.uint8 or k_cache.dim() < 2:
        return False
    T, H, _ = q.shape
    NH = max_score.shape[1]
    if NH % HEADS != 0 or NH > 64 or H < NH or T == 0 or T > _PF_MAX_T:
        if ("fb", block_size, NH) not in _pf_ran_seen:
            _pf_ran_seen.add(("fb", block_size, NH))
            logger.info("cubit prefill FELL BACK to Triton (NH=%d H=%d T=%d block_size=%d)",
                        NH, H, T, block_size)
        return False
    if slot_ids.shape[0] != T or lens.shape[0] != T:
        return False
    # require the contiguous [T,NH] / [T,NH,512] layout the kernel hardcodes
    if (max_score.stride(0) != NH or denom.stride(0) != NH
            or acc.stride(0) != NH * OUT_DIM or acc.stride(1) != OUT_DIM
            or acc.stride(2) != 1):
        return False
    if max_score.dtype != torch.float32 or denom.dtype != torch.float32 \
            or acc.dtype != torch.float32:
        return False

    NHG = NH // HEADS
    dev = q.device
    Cchunk = slot_ids.shape[1]
    block_stride = int(k_cache.stride(0))     # bytes (uint8 cache)
    scale2 = int(np.float32(scale * LOG2E).view(np.uint32))

    qn = _pf_pack_q(q, NH)
    # mask this chunk's slots to -1 beyond the per-token valid length
    jj = candidate_offset + torch.arange(Cchunk, device=dev, dtype=torch.int32)
    slots_m = slot_ids.to(torch.int32)
    slots_m = torch.where(jj[None, :] >= lens[:, None], slots_m.new_full((), -1),
                          slots_m)
    slots64 = _pf_slots_buf(dev, T)

    cu = _driver()
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    nhg_c = ctypes.c_uint32(NHG)
    bs_c = ctypes.c_uint32(block_stride)
    s2_c = ctypes.c_uint32(scale2 & 0xFFFFFFFF)
    # block_size-dependent cache addressing (slot -> page/offset and per-page scale
    # region). block_stride (bs_c) is the byte page stride; these are the logical
    # tokens-per-page decomposition. scale_base = bs*576 derived from the real
    # fp8_ds_mla SoA page layout [bs x 576B entry][bs x 8B scale].
    log2bs_c = ctypes.c_uint32(block_size.bit_length() - 1)
    bsmask_c = ctypes.c_uint32(block_size - 1)
    scale_base_c = ctypes.c_uint32(block_size * _PF_ENTRY_BYTES)
    keep = []
    for sc in range(0, Cchunk, _PF_TILE):
        ce = min(sc + _PF_TILE, Cchunk)
        n = ce - sc
        if n < _PF_TILE:
            slots64[:T, n:].fill_(-1)
        slots64[:T, :n].copy_(slots_m[:, sc:ce])
        a = [ctypes.c_uint64(qn.data_ptr()),
             ctypes.c_uint64(k_cache.data_ptr()),
             ctypes.c_uint64(slots64.data_ptr()),
             ctypes.c_uint64(max_score.data_ptr()),
             ctypes.c_uint64(denom.data_ptr()),
             ctypes.c_uint64(acc.data_ptr()),
             s2_c, bs_c, nhg_c, log2bs_c, bsmask_c, scale_base_c]
        argv = (ctypes.c_void_p * len(a))(
            *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in a])
        keep.append((a, argv))
        _ck(cu.cuLaunchKernel(_pf_fn, T, 1, 1, 256, 1, 1, 0, stream, argv, None),
            "launch prefill")
    del keep
    if ("ran", block_size) not in _pf_ran_seen:
        _pf_ran_seen.add(("ran", block_size))
        logger.info("cubit prefill accumulate RAN (T=%d NH=%d C=%d block_size=%d, %d sub-tiles)",
                    T, NH, Cchunk, block_size, (Cchunk + _PF_TILE - 1) // _PF_TILE)
    return True


# ===========================================================================
# PREFILL chunked-state (mla_prefill_chunked accumulate + mla_prefill_merge):
# the WHOLE candidate set in ONE 3D-grid launch over (token, chunk, head-group)
# -> per-chunk PARTIAL online states in scratch (NO per-chunk acc RMW), then ONE
# merge -> final in-place state. Fixes the C128A (bs=2, high candidate count) leg
# where the per-64-cand sub-tile loop + acc-RMW of `_accumulate` above is launch/
# RMW-bound and regressed TTFT vs Triton e2e. Opt-in VLLM_SPARSE_MLA_PREFILL_CHUNKED
# (default OFF); returns False on any unsupported shape (caller keeps its loop path).
# Scratch (per token-pass): part_max/part_denom [nchunks,Tp,NH] f32, part_acc
# [nchunks,Tp,NH,512] bf16. part_acc dominates (nchunks*Tp*NH*512*2 B); tokens/pass
# Tp is bounded so it fits VLLM_SPARSE_MLA_PREFILL_CHUNKED_SCRATCH_MB (default 512).
# ===========================================================================
_PFC_ACC_KERN = b"mla_prefill_chunked"
_PFC_MRG_KERN = b"mla_prefill_merge"
_pfc_acc_fn = None
_pfc_mrg_fn = None
_pfc_state = "uninit"
# scratch budget for part_acc; tokens/pass (tcap) derived so part_acc <= this. Clamped to
# <2 GiB so the kernel's 32-bit acc byte-offset (s_idx*0x4000) never overflows.
_PFC_SCRATCH_MB = min(int(os.environ.get("VLLM_SPARSE_MLA_PREFILL_CHUNKED_SCRATCH_MB", "512")), 2048)
# max candidate chunks (64 cands each) per token the persistent pool covers. C128A is
# frozen at C=2048 -> 32 chunks; raise via env if a leg uses a larger candidate count.
_PFC_NCHUNK_CAP = int(os.environ.get("VLLM_SPARSE_MLA_PREFILL_CHUNKED_NCHUNK_CAP", "32"))
_PFC_BUFS: dict = {}        # (device, NH) -> persistent CAPTURE-SAFE I/O pool (allocated in eager)
_pfc_launches = 0           # cumulative cuLaunchKernel count (flat in steady state once captured)
_PFC_TRACE = os.environ.get("VLLM_SPARSE_MLA_PREFILL_CHUNKED_TRACE", "0") == "1"
_pfc_ran_seen: set = set()


def _pfc_cubin(kern: bytes, sass: str, env: str, out: str) -> str:
    pre = os.getenv(env)
    if pre:
        if not os.path.isfile(pre):
            raise FileNotFoundError(pre)
        return pre
    repo = os.getenv("VLLM_SPARSE_MLA_CUBIT_REPO", "cubit")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    r = subprocess.run(
        [os.path.join(repo, "target/release/cubit"), "asm",
         os.path.join(repo, sass), "-o", out, "--kernel", kern.decode(),
         "--mercury-stub", os.path.join(repo, "sass/qmma_e4m3.merc.stub")],
        capture_output=True, text=True, cwd=repo)
    if "0 failed" not in (r.stdout + r.stderr):
        raise RuntimeError(
            f"cubit asm failed ({kern.decode()}): {r.stdout[-400:]} {r.stderr[-400:]}")
    return out


def _pfc_ensure() -> bool:
    global _pfc_acc_fn, _pfc_mrg_fn, _pfc_state
    if _pfc_state == "ready":
        return True
    if _pfc_state == "unavailable":
        return False
    try:
        cu = _driver()
        specs = (
            (_PFC_ACC_KERN, "sass/mla_prefill_chunked.sass",
             "VLLM_SPARSE_MLA_PREFILL_CHUNKED_CUBIN",
             os.path.join(_CUBIN_DIR, "pf_chunked_bs.cubin"), "acc"),
            (_PFC_MRG_KERN, "sass/mla_prefill_merge.sass",
             "VLLM_SPARSE_MLA_PREFILL_MERGE_CUBIN",
             os.path.join(_CUBIN_DIR, "pf_merge.cubin"), "mrg"),
        )
        for kern, sass, env, out, slot in specs:
            mod = ctypes.c_void_p()
            _ck(cu.cuModuleLoad(ctypes.byref(mod), _pfc_cubin(kern, sass, env, out).encode()),
                "load prefill-chunked")
            fn = ctypes.c_void_p()
            _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, kern), "getfn prefill-chunked")
            if slot == "acc":
                _pfc_acc_fn = fn
            else:
                _pfc_mrg_fn = fn
        _pfc_state = "ready"
        logger.info("cubit sparse-MLA PREFILL-CHUNKED kernels loaded (accumulate + merge)")
        return True
    except Exception as e:  # noqa: BLE001 - any setup failure -> caller fallback
        logger.warning("cubit sparse-MLA prefill-chunked unavailable (%s); using fallback", e)
        _pfc_state = "unavailable"
        return False


def _pfc_bufs(device, NH: int) -> dict:
    """Persistent CAPTURE-SAFE I/O pool for the chunked prefill, allocated ONCE per
    (device, NH) in eager (never during graph capture). Stable pointers + zero per-call
    allocation are the prerequisites for the launch trio to be recorded into a CUDA graph
    and replayed with ZERO host launches. Mirrors the decode `_BUFS` design.
    Sized for tcap tokens/pass (so part_acc <= the MB budget) x _PFC_NCHUNK_CAP chunks."""
    key = (device, NH)
    b = _PFC_BUFS.get(key)
    if b is None:
        NHG = NH // HEADS
        ncap = _PFC_NCHUNK_CAP
        npad = ncap * _PF_TILE
        tcap = max(1, (_PFC_SCRATCH_MB << 20) // (ncap * NH * OUT_DIM * 2))
        z = lambda *s, dt: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        b = dict(
            tcap=tcap, ncap=ncap, npad=npad,
            qn=z(tcap, NHG * 32 * 32 * 4, dt=torch.int32),        # HMMA A-frags (q-pack out)
            slot_pad=z(tcap, npad, dt=torch.int32),               # padded/masked slots [T,npad]
            slots_ch=z(ncap, tcap, _PF_TILE, dt=torch.int32),     # chunk-major slots (kernel in)
            mask=z(tcap, npad, dt=torch.bool),                    # ge(jcol,lens) scratch
            p_max=z(ncap, tcap, NH, dt=torch.float32),            # partial states (scratch)
            p_den=z(ncap, tcap, NH, dt=torch.float32),
            p_acc=z(ncap, tcap, NH, OUT_DIM, dt=torch.bfloat16),  # dominates the budget
            lo_half=(_pf_lo_idx(NH, device) // 2),                # q-pack int32 gather index
            jcol=torch.arange(npad, device=device, dtype=torch.int32),
        )
        _PFC_BUFS[key] = b
        logger.info("cubit prefill-chunked pool: NH=%d tcap=%d ncap=%d (part_acc=%.0f MB)",
                    NH, tcap, ncap, ncap * tcap * NH * OUT_DIM * 2 / (1 << 20))
    return b


def cubit_sparse_mla_prefill_chunked_reserve(device, num_heads: int) -> None:
    """Profile-time hook: load the cubins + allocate the persistent pool eagerly so the
    memory profiler's peak covers the (large) part_acc scratch and graph capture later
    finds the pool already resident. No-op if unsupported. Call from the dummy run."""
    if num_heads % HEADS != 0 or num_heads > 64:
        return
    try:
        if _pfc_ensure():
            _pfc_bufs(device, num_heads)
    except Exception as e:  # noqa: BLE001
        logger.warning("cubit prefill-chunked reserve skipped (%s)", e)


def cubit_sparse_mla_prefill_chunked_launches() -> int:
    """Cumulative cuLaunchKernel count for the chunked prefill path. With graph capture
    this stops incrementing in steady state (the graph replays device-side)."""
    return _pfc_launches


def cubit_sparse_mla_prefill_chunked(
    q: torch.Tensor,            # [T, H, 512] bf16 (first NH heads active)
    k_cache: torch.Tensor,      # fp8_ds_mla paged cache (>=2D), uint8
    slot_ids: torch.Tensor,     # [T, C] int32 global slots (-1 invalid) - FULL candidate set
    lens: torch.Tensor,         # [T] int32 valid-candidate counts
    block_size: int,
    scale: float,
    max_score: torch.Tensor,    # [T, NH] f32 out (natural log domain) - written (fresh accumulate)
    denom: torch.Tensor,        # [T, NH] f32 out
    acc: torch.Tensor,          # [T, NH, 512] f32 out (unnormalized)
) -> bool:
    """Accumulate the WHOLE candidate set [T, C] in one chunked 2-kernel pass (no per-chunk
    acc RMW): ONE 3D-grid (token, chunk, head-group) accumulate -> per-chunk PARTIAL states
    in scratch, then ONE merge -> final state, matching the per-64-cand accumulate loop run
    from a fresh (max=-inf, denom=0, acc=0) state. Output state is OVERWRITTEN.

    CAPTURE-SAFE: every per-call buffer is from the persistent pool (_pfc_bufs); q-pack is a
    single non-allocating index_select(out=), slot mask/pad/chunk-major are in-place ops into
    pooled buffers; the two cuLaunchKernels are stream-ordered with stable pointers and there
    is NO host sync / allocation. So the launch trio is recorded into the model's CUDA graph
    and replayed with zero host launches in steady prefill. Returns False (caller keeps its
    loop/Triton path) on any unsupported shape/environment."""
    global _pfc_launches
    capturing = torch.cuda.is_current_stream_capturing()
    if not _pfc_ensure():
        return False
    if q.dim() == 4:
        q = q[:, 0]
    if slot_ids.dim() == 3:
        slot_ids = slot_ids[:, 0]
    if q.dim() != 3 or q.shape[-1] != OUT_DIM or block_size not in _PF_BLOCK_SIZES:
        if ("fb", block_size) not in _pfc_ran_seen:
            _pfc_ran_seen.add(("fb", block_size))
            logger.info("cubit prefill-chunked FELL BACK (block_size=%d head_dim=%d)",
                        block_size, q.shape[-1])
        return False
    if k_cache.dtype != torch.uint8 or k_cache.dim() < 2:
        return False
    T, H, _ = q.shape
    NH = max_score.shape[1]
    C = slot_ids.shape[1]
    if NH % HEADS != 0 or NH > 64 or H < NH or T == 0 or C == 0:
        if ("fb2", NH, C) not in _pfc_ran_seen:
            _pfc_ran_seen.add(("fb2", NH, C))
            logger.info("cubit prefill-chunked FELL BACK (NH=%d H=%d T=%d C=%d)", NH, H, T, C)
        return False
    if slot_ids.shape[0] != T or lens.shape[0] != T:
        return False
    # contiguity the no-alloc views require (q reshape, slot/lens raw rows)
    if (not q.is_contiguous() or slot_ids.stride(-1) != 1 or slot_ids.dtype != torch.int32
            or lens.dtype != torch.int32 or lens.stride(-1) != 1):
        return False
    if (max_score.stride(0) != NH or denom.stride(0) != NH
            or acc.stride(0) != NH * OUT_DIM or acc.stride(1) != OUT_DIM
            or acc.stride(2) != 1):
        return False
    if (max_score.dtype != torch.float32 or denom.dtype != torch.float32
            or acc.dtype != torch.float32):
        return False

    dev = q.device
    NHG = NH // HEADS
    nchunks = (C + _PF_TILE - 1) // _PF_TILE
    key = (dev, NH)
    if key not in _PFC_BUFS:
        if capturing:
            return False               # cannot allocate the persistent pool mid-capture
        _pfc_bufs(dev, NH)
    b = _PFC_BUFS[key]
    tcap, ncap, npad = b["tcap"], b["ncap"], b["npad"]
    if nchunks > ncap or C > npad:
        if ("fb3", C) not in _pfc_ran_seen:
            _pfc_ran_seen.add(("fb3", C))
            logger.info("cubit prefill-chunked FELL BACK (C=%d nchunks=%d > pool cap %d; "
                        "raise VLLM_SPARSE_MLA_PREFILL_CHUNKED_NCHUNK_CAP)", C, nchunks, ncap)
        return False

    block_stride = int(k_cache.stride(0))
    scale2 = int(np.float32(scale * LOG2E).view(np.uint32))
    log2bs = block_size.bit_length() - 1
    bsmask = block_size - 1
    scale_base = block_size * _PF_ENTRY_BYTES
    qn, slot_pad, slots_ch, mask = b["qn"], b["slot_pad"], b["slots_ch"], b["mask"]
    p_max, p_den, p_acc, lo_half, jcol = (b["p_max"], b["p_den"], b["p_acc"],
                                          b["lo_half"], b["jcol"])
    # full-H q as int32 (each lane reads a bf16 pair as one u32); contiguous view, no copy
    qH = q.reshape(T, H * OUT_DIM).view(torch.int32)
    cu = _driver()
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    keep = []
    for t0 in range(0, T, tcap):
        t1 = min(t0 + tcap, T)
        ts = t1 - t0
        # ---- q-pack: ONE non-allocating gather into the pooled A-frag buffer ----
        torch.index_select(qH[t0:t1], 1, lo_half, out=qn[:ts])
        # ---- slots: pad/mask/chunk-major, all in-place into pooled buffers (no alloc) ----
        if C < npad:
            slot_pad[:ts, C:].fill_(-1)
        slot_pad[:ts, :C].copy_(slot_ids[t0:t1])
        torch.ge(jcol[None, :C], lens[t0:t1, None], out=mask[:ts, :C])  # cand >= len -> mask
        slot_pad[:ts, :C].masked_fill_(mask[:ts, :C], -1)
        slots_ch[:nchunks, :ts].copy_(
            slot_pad[:ts].view(ts, ncap, _PF_TILE)[:, :nchunks].transpose(0, 1))
        # ---- accumulate: grid (ts, nchunks, 1); kernel addresses with pool token stride tcap ----
        a = [ctypes.c_uint64(qn.data_ptr()),
             ctypes.c_uint64(k_cache.data_ptr()),
             ctypes.c_uint64(slots_ch.data_ptr()),
             ctypes.c_uint64(p_max.data_ptr()),
             ctypes.c_uint64(p_den.data_ptr()),
             ctypes.c_uint64(p_acc.data_ptr()),
             ctypes.c_uint32(scale2 & 0xFFFFFFFF),
             ctypes.c_uint32(block_stride),
             ctypes.c_uint32(NHG),
             ctypes.c_uint32(tcap),                          # T (pool token stride)
             ctypes.c_uint32(NHG),                           # gpc = NHG (grid.z = 1)
             ctypes.c_uint32(log2bs),
             ctypes.c_uint32(bsmask),
             ctypes.c_uint32(scale_base)]
        argv = (ctypes.c_void_p * len(a))(
            *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in a])
        _ck(cu.cuLaunchKernel(_pfc_acc_fn, ts, nchunks, 1, 256, 1, 1, 0, stream, argv, None),
            "launch prefill-chunked accumulate")
        # ---- merge: grid (ts*NH,); part chunk stride tn = tcap*NH ----
        ms, ds, ac = max_score[t0:t1], denom[t0:t1], acc[t0:t1]
        m = [ctypes.c_uint64(ms.data_ptr()),
             ctypes.c_uint64(ds.data_ptr()),
             ctypes.c_uint64(ac.data_ptr()),
             ctypes.c_uint64(p_max.data_ptr()),
             ctypes.c_uint64(p_den.data_ptr()),
             ctypes.c_uint64(p_acc.data_ptr()),
             ctypes.c_uint32(nchunks),
             ctypes.c_uint32(tcap * NH)]
        margv = (ctypes.c_void_p * len(m))(
            *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in m])
        _ck(cu.cuLaunchKernel(_pfc_mrg_fn, ts * NH, 1, 1, 128, 1, 1, 0, stream, margv, None),
            "launch prefill-chunked merge")
        _pfc_launches += 2
        keep.append((a, argv, m, margv))
    del keep
    if _PFC_TRACE or ("ran", block_size) not in _pfc_ran_seen:
        _pfc_ran_seen.add(("ran", block_size))
        logger.info("cubit prefill-chunked RAN (T=%d NH=%d C=%d bs=%d nchunks=%d tcap=%d "
                    "passes=%d capturing=%d launches=%d)", T, NH, C, block_size, nchunks,
                    tcap, (T + tcap - 1) // tcap, int(capturing), _pfc_launches)
    return True
