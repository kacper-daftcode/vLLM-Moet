# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""1-GPU DeepSeek-V4 routed experts on 2-bit tensor-sym planes (cubit moe_w2).

Opt-in via VLLM_MOE_W2=1. Replaces the Marlin w4a16 routed-expert path:

  weights : checkpoint mxfp4 e2m1 codes -> {-4,-1,1,4} 2-bit planes built on
            GPU at load (QUANT_PROBE tensor-sym K=4: acceptance 2.73 vs 2.68
            baseline, 12/12 coherent). Block-32 UE8M0 scale bytes verbatim.
  compute : cubit `moe_w2_mm` SASS GEMM (M<=4 per pair, PRMT-LUT decode,
            QMMA.SF block-32 sfb, f32 act-scale fold) for BOTH w13 and w2.
  glue    : moe_align_block_size(block=4) pairs, fp8 group-128 activation
            quant, silu*up in torch, weighted scatter-add unpermute. All
            steps are tensor ops or driver launches on the current stream:
            CUDA-graph capturable, registered as one custom op.

VRAM: planes+scales ~1.73 GiB/layer (vs ~3.2 GiB raw fp4) -> 43 layers fit
a single 96 GB SM120 board together with the fp8 dense stack and KV.
The MTP drafter keeps its original (Marlin) path: layer names containing
"mtp" are excluded, matching the QUANT_PROBE protocol (drafter unmodified).
"""

import ctypes
import functools
import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
    mxfp4_to_codes,
    pack_fragment_major,
    pack_scales,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_KERN = b"moe_w2_mm"
_DIR = os.getenv("VLLM_MOE_W2_CUBIT_DIR", "/cubit-share")
_BLOCK = 4                      # tokens per pair == kernel M limit
_NTHR = 256                     # NWARP=8 (K>=1024)


def _nwarp_for_k(k: int) -> int:
    """Split-K warp count baked into each cubin by gen_moe_w2.py (KSLICE=K/NWARP
    must be a multiple of 128). K>=1024 -> 8 warps; K=512 (the w2 GEMM under TP4)
    shards to 4. The launch block MUST match the cubin or the extra warps index
    past K (KSLICE*wid) and read garbage. Mirrors the generator's `_nwarp`."""
    nb = k // 128
    cap = 8 if k >= 1024 else 4
    for n in range(min(cap, nb), 0, -1):
        if nb % n == 0:
            return n
    return 1

_cu = None
_fns: dict = {}
_state = "uninit"
# PREFILL LEVER (opt-in, default OFF): fragment-major activations so each lane's
# m16k32 QMMA A-fragment loads in ONE LDG.128 (vs 8 strided 4-byte loads). Profile
# showed prefill moe_w2_mm is L1/load-issue bound (NOT weight-DRAM bound), so this
# cuts the dominant load class ~4x at identical occupancy -> ~1.3x prefill GEMM.
# Numerics are bit-identical to mc4. Needs moe_w2_mm_mc4afrag_k{K}.cubin present.
_AFRAG = os.getenv("VLLM_MOE_W2_AFRAG", "0") == "1"
_afrag_ok = False


def _to_fragment_major(a: torch.Tensor, pairs: int, K: int) -> torch.Tensor:
    """[pairs*16, K] fp8 row-major -> fragment-major per 16-token tile (matches the
    AFRAG kernel layout / tools.moe_w2_prefill_bench.pack_a_fragment_major):
    dims [pair, g2, g, j, quad, t, b] -> [pair, j, g, t, quad, g2, b].

    `a` MUST have EXACTLY pairs*16 rows (complete tiles). Callers pass the
    tile-aligned region ws['a1'][:pairs*16] -- NOT ws['a1'][:slots] (slots is the
    over-allocated, non-16-multiple sorted_ids size)."""
    assert a.shape[0] == pairs * 16, (a.shape, pairs)
    v = a.view(torch.uint8).view(pairs, 2, 8, K // 64, 4, 4, 4)
    v = v.permute(0, 3, 2, 5, 4, 1, 6).reshape(pairs * 16, K)
    return v.contiguous().view(a.dtype)

# layer_key -> dict(planes13, sc13, planes2, sc2, top_k, inter)
_LAYERS: dict[int, dict] = {}
_WS: dict = {}                  # shared workspaces, sized lazily


def enabled() -> bool:
    return os.getenv("VLLM_MOE_W2", "0") == "1"


def is_w2_layer(layer_name: str) -> bool:
    """Main-model routed experts only. The MTP drafter (layer index >=
    num_hidden_layers, e.g. model.layers.43.* for the 43-layer main stack)
    keeps its original Marlin path: QUANT_PROBE's acceptance numbers were
    measured with the drafter unmodified."""
    if not enabled():
        return False
    name = layer_name or ""
    if "mtp" in name:
        return False
    import re
    m = re.search(r"\.layers\.(\d+)\.", name)
    if m is None:
        return False
    cutoff = int(os.getenv("VLLM_MOE_W2_NUM_LAYERS", "43"))
    return int(m.group(1)) < cutoff


def _driver():
    global _cu
    if _cu is None:
        cu = ctypes.CDLL("libcuda.so.1")
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [
            ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p]
        cu.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_char_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.c_void_p, ctypes.c_char_p]
        _cu = cu
    return _cu


def _ck(r, what):
    if r:
        raise RuntimeError(f"moe_w2_cubit: CUDA error {r} in {what}")


def _ensure_ready() -> bool:
    global _state
    if _state == "ready":
        return True
    if _state == "unavailable":
        return False
    try:
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        cu = _driver()
        for tier, kern in (("w2", b"moe_w2_mm"), ("w4", b"moe_w4_mm"),
                           ("w2mc2", b"moe_w2_mm"), ("w2mc4", b"moe_w2_mm")):
            # GEMM contraction K: 4096 (w13) + 2048 (w2) on a single GPU. Under
            # tensor parallelism the w2-side K shards (1024 @ TP2, 512 @ TP4);
            # those cubins are OPTIONAL -- loaded only when shipped, so the
            # single-GPU path still needs just k4096/k2048 (and fails loudly if
            # those mandatory ones are missing).
            for k in (4096, 2048, 1024, 512):
                if tier in ("w2mc2", "w2mc4"):
                    fname = f"moe_w2_mm_{tier[2:]}_k{k}.cubin"
                else:
                    fname = f"moe_{tier}_mm_k{k}.cubin"
                path = os.path.join(_DIR, fname)
                if k in (1024, 512) and not os.path.exists(path):
                    continue
                mod = ctypes.c_void_p()
                _ck(cu.cuModuleLoad(ctypes.byref(mod), path.encode()),
                    f"cuModuleLoad {path}")
                fn = ctypes.c_void_p()
                _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, kern),
                    "cuModuleGetFunction")
                _fns[(tier, k)] = fn
        global _afrag_ok
        if _AFRAG:
            try:
                for k in (4096, 2048, 1024, 512):
                    path = os.path.join(_DIR, f"moe_w2_mm_mc4afrag_k{k}.cubin")
                    if k in (1024, 512) and not os.path.exists(path):
                        continue
                    mod = ctypes.c_void_p()
                    _ck(cu.cuModuleLoad(ctypes.byref(mod), path.encode()),
                        f"cuModuleLoad {path}")
                    fn = ctypes.c_void_p()
                    _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"moe_w2_mm"),
                        "cuModuleGetFunction afrag")
                    _fns[("w2mc4afrag", k)] = fn
                _afrag_ok = True
                logger.info("moe_w2_cubit: AFRAG prefill cubins loaded")
            except Exception as e:  # noqa: BLE001
                logger.warning("moe_w2_cubit: AFRAG unavailable (%s); using mc4", e)
                _afrag_ok = False
        _state = "ready"
        logger.info("moe_w2_cubit: cubins loaded: %s", sorted(_fns))
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("moe_w2_cubit unavailable: %s", e)
        _state = "unavailable"
        return False


# --------------------------------------------------------------------------
# Load-time plane building
# --------------------------------------------------------------------------

def build_layer_planes(layer, layer_key: int) -> None:
    """Quantize one FusedMoE layer's experts to 2-bit planes (GPU, chunked).

    Reads the CPU-resident checkpoint params (w13_weight [E,2I,K/2] u8 etc.),
    builds fragment-major code planes + scale planes on the GPU, then
    replaces the originals with empty stubs.
    """
    assert _ensure_ready(), "moe_w2 cubins missing"
    dev = torch.device("cuda")
    w13 = layer.w13_weight.data          # [E, 4096, 2048] u8 (cpu)
    s13 = layer.w13_weight_scale.data    # [E, 4096, 128] u8
    w2 = layer.w2_weight.data            # [E, 4096, 1024] u8
    s2 = layer.w2_weight_scale.data      # [E, 4096, 64] u8
    E, N13, _ = w13.shape
    _, N2, _ = w2.shape
    K13, K2 = N2, N13 // 2               # 4096, 2048

    planes13 = torch.empty(E, N13 * K13 // 4, dtype=torch.uint8, device=dev)
    sc13 = torch.empty(E, N13 * K13 // 32, dtype=torch.uint8, device=dev)
    planes2 = torch.empty(E, N2 * K2 // 4, dtype=torch.uint8, device=dev)
    sc2 = torch.empty(E, N2 * K2 // 32, dtype=torch.uint8, device=dev)

    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        mxfp4_to_nibbles, pack_fp4_fragment_major)
    # Pass the PER-RANK FP4 plane sizes (N*K//2 bytes/expert) so the delta tier's
    # slots, host store, and pool indexing match the (TP-sharded) planes. On TP1
    # these equal the module constants -> the single-GPU path is unchanged.
    tier = moe_w2_delta.get_tier(dev=dev, w13_bytes=N13 * K13 // 2,
                                 w2_bytes=N2 * K2 // 2)
    fp13 = fp2 = None
    if tier is not None:
        fp13 = torch.empty(E, N13 * K13 // 2, dtype=torch.uint8, device=dev)
        fp2 = torch.empty(E, N2 * K2 // 2, dtype=torch.uint8, device=dev)

    chunk = 32
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        wg = w13[e0:e1].to(dev, non_blocking=True)
        sg = s13[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            nib = mxfp4_to_nibbles(wg[i])
            planes13[e0 + i] = pack_fragment_major(mxfp4_to_codes(wg[i]))
            sc13[e0 + i] = pack_scales(sg[i])
            if fp13 is not None:
                fp13[e0 + i] = pack_fp4_fragment_major(nib)
        wg = w2[e0:e1].to(dev, non_blocking=True)
        sg = s2[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            nib = mxfp4_to_nibbles(wg[i])
            planes2[e0 + i] = pack_fragment_major(mxfp4_to_codes(wg[i]))
            sc2[e0 + i] = pack_scales(sg[i])
            if fp2 is not None:
                fp2[e0 + i] = pack_fp4_fragment_major(nib)

    if tier is not None:
        tier.add_layer_host_planes(layer_key, fp13, fp2)
        del fp13, fp2
        # (the background manager is started by get_tier when the tier is
        # created; the old "start on layer NUM_LAYERS-1" trigger never fired
        # under PP, where layer_keys are local per rank and never reach 42)

    _LAYERS[layer_key] = dict(
        planes13=planes13, sc13=sc13, planes2=planes2, sc2=sc2,
        N13=N13, K13=K13, N2=N2, K2=K2, E=E,
    )
    # Release checkpoint copies; keep CUDA stubs so device probes stay happy.
    stub = torch.empty(0, dtype=torch.uint8, device=dev)
    for name in ("w13_weight", "w13_weight_scale", "w2_weight",
                 "w2_weight_scale"):
        layer.register_parameter(
            name, torch.nn.Parameter(stub, requires_grad=False))
    logger.info("moe_w2: layer %d planes built (%.2f GiB)", layer_key,
                (planes13.nbytes + sc13.nbytes + planes2.nbytes + sc2.nbytes)
                / 2**30)


# --------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------

def _workspaces(slots: int, tokens: int, dev, inter: int = 2048) -> dict:
    # `inter` = per-rank expert intermediate size I (2048 on 1 GPU; 1024 @ TP2,
    # 512 @ TP4 as the experts shard). The hidden H (4096) is NOT sharded, so
    # the A-side (a1), x-quant (xq) and w2 output (c2) buffers stay H-wide; only
    # the gate/up output (c13 = 2I), the intermediate activation (act/a2 = I) and
    # its group-128 scales (as2 = I/128) follow the shard. On 1 GPU (inter=2048)
    # every shape is byte-identical to before.
    if (_WS.get("slots", 0) < slots or _WS.get("tokens", 0) < tokens
            or _WS.get("inter") != inter):
        slots = max(slots, _WS.get("slots", 0))
        tokens = max(tokens, _WS.get("tokens", 0))
        _WS.update(
            slots=slots,
            tokens=tokens,
            inter=inter,
            # token-side quant buffers; the LAST row is the permanent zero
            # pad row (gather source for filler slots) — quant only ever
            # writes rows [:T].
            xq=torch.zeros(tokens + 1, 4096, dtype=torch.float8_e4m3fn,
                           device=dev),
            xs=torch.zeros(tokens + 1, 32, dtype=torch.float32, device=dev),
            a1=torch.zeros(slots + 4, 4096, dtype=torch.float8_e4m3fn,
                           device=dev),
            as1=torch.zeros(slots + 4, 32, dtype=torch.float32, device=dev),
            # zeros, not empty: pad-pair rows are never written by the kernel
            # (early EXIT) yet flow through silu/scatter math with weight 0;
            # uninitialized inf/nan would poison 0*x.
            c13=torch.zeros(slots + 4, 2 * inter, dtype=torch.bfloat16,
                            device=dev),
            act=torch.zeros(slots + 4, inter, dtype=torch.bfloat16, device=dev),
            a2=torch.zeros(slots + 4, inter, dtype=torch.float8_e4m3fn,
                           device=dev),
            as2=torch.zeros(slots + 4, max(inter // 128, 1),
                            dtype=torch.float32, device=dev),
            c2=torch.zeros(slots + 4, 4096, dtype=torch.bfloat16, device=dev),
            desc=torch.empty(4, slots // _BLOCK, 6, dtype=torch.int64,
                             device=dev),
            no_slots=torch.full((256,), -1, dtype=torch.int32, device=dev),
        )
    return _WS


import triton
import triton.language as tl


@triton.jit
def _desc_build_kernel(
    eids_ptr, npost_ptr, slot_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    p13b, s13b, p2b, s2b, poolb,
    p13s, s13s, p2s, s2s,
    slot_bytes, w13_bytes,
    c13_rb, a2_rb, as2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """All four moe desc tables in one launch (24 columns per pair).

    d_ptr = [4, cap, 6] i64: 0 = w2-tier w13, 1 = w2-tier w2,
    2 = w4-tier w13, 3 = w4-tier w2. A pair is routed to exactly one tier
    via the m_rows field (the other tier's kernel sees m=0 -> early EXIT).
    slot_ptr = this layer's row of the delta slot table (-1 = base tier);
    poolb = delta pool base (w13 plane at slot start, w2 at +w13_bytes).
    """
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    is4 = slot >= 0
    m2 = tl.where(live & ~is4, mblock, 0).to(tl.int64)
    m4 = tl.where(live & is4, mblock, 0).to(tl.int64)
    base = p.to(tl.int64) * mblock
    slot_c = tl.maximum(slot, 0)
    a1 = a1b + base * 4096
    as1 = as1b + base * 128
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * 8192
    bs13 = s13b + e * s13s
    bs2 = s2b + e * s2s
    for gi in tl.static_range(4):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c, m = p13b + e * p13s, bs13, a1, as1, c13, m2
        elif gi == 1:
            b, s, a, as_, c, m = p2b + e * p2s, bs2, a2, as2, c2, m2
        elif gi == 2:
            b, s, a, as_, c, m = (poolb + slot_c * slot_bytes, bs13,
                                  a1, as1, c13, m4)
        else:
            b, s, a, as_, c, m = (poolb + slot_c * slot_bytes + w13_bytes,
                                  bs2, a2, as2, c2, m4)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m, mask=mask)


def _launch(tier: str, K: int, desc: torch.Tensor, n_rows: int, pairs: int,
            stream):
    fn = _fns[(tier, K)]
    args = [ctypes.c_uint64(desc.data_ptr()),
            ctypes.c_uint32(K),
            ctypes.c_uint32(K // 64),
            ctypes.c_uint32(n_rows * 2),
            ctypes.c_uint32(K // 128)]
    argv = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in args])
    _ck(_driver().cuLaunchKernel(fn, n_rows // 16, pairs, 1,
                                 _nwarp_for_k(K) * 32, 1, 1, 0,
                                 stream, argv, None), "launch")


def _moe_w2_forward(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils import prefill_timers
    with prefill_timers.span("moe_w2"):
        return _moe_w2_forward_timed(x, topk_weights, topk_ids, layer_key)


def _moe_w2_forward_timed(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    st = _LAYERS[layer_key]
    T, H = x.shape
    top_k = topk_ids.shape[1]
    dev = x.device
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    # decode-sized calls use the proven 4-token kernel + delta tier;
    # prefill-sized calls use the MC4 kernel (16 tokens per pair-entry = full
    # QMMA-M, plane reads amortized 4x, ~1.5x over MC2) on the 2-bit base only.
    # 96 = the largest cudagraph capture size: anything above is necessarily a
    # prefill chunk; short tail chunks keep the delta-quality path.
    prefill = T > 96
    mblock = 16 if prefill else _BLOCK
    sorted_ids, expert_blocks, num_post = moe_align_block_size(
        topk_ids, mblock, st["E"])
    slots = sorted_ids.numel()
    pairs = slots // mblock
    # st["K2"] = per-rank expert intermediate I (w2 contraction) -> sizes the
    # intermediate workspaces correctly under tensor parallelism.
    ws = _workspaces(slots, T, dev, inter=st["K2"])

    # ---- activation quant (group-128) into the padded buffer; the buffer's
    # last row is the permanent zero pad row for filler slots.
    xq = ws["xq"]
    pad_row = xq.shape[0] - 1
    _, xs = per_token_group_quant_fp8(x, 128, out_q=xq[:T])
    ws["xs"][:T] = xs
    valid = sorted_ids < T * top_k
    rows = torch.where(valid, sorted_ids // top_k,
                       torch.full_like(sorted_ids, pad_row))
    torch.index_select(xq.view(torch.uint8), 0, rows,
                       out=ws["a1"][:slots].view(torch.uint8))
    torch.index_select(ws["xs"], 0, rows, out=ws["as1"][:slots])

    # ---- all four desc tables in ONE triton launch (w2 + w4 tiers)
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    tier = moe_w2_delta._TIER       # peek only; created by the plane builder
    if tier is not None and not prefill:
        if torch.cuda.is_current_stream_capturing():
            tier.notify_capture()
        slot_row = tier.slot_table[layer_key]
        pool_ptr = tier.pool.data_ptr()
        moe_w2_delta.mark_seen(tier.seen[layer_key], topk_ids.view(-1).long())
    else:
        if tier is not None:
            moe_w2_delta.mark_seen(tier.seen[layer_key], topk_ids.view(-1).long())
        slot_row = ws["no_slots"]
        pool_ptr = ws["a1"].data_ptr()      # never dereferenced (m4=0)
    d = ws["desc"]
    cap = d.shape[1]
    _desc_build_kernel[(triton.cdiv(pairs, 256),)](
        expert_blocks, num_post, slot_row, d,
        ws["a1"].data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
        ws["a2"].data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
        st["planes13"].data_ptr(), st["sc13"].data_ptr(),
        st["planes2"].data_ptr(), st["sc2"].data_ptr(), pool_ptr,
        st["planes13"].shape[1], st["sc13"].shape[1],
        st["planes2"].shape[1], st["sc2"].shape[1],
        (tier.slot_bytes if tier is not None else moe_w2_delta.SLOT_BYTES),
        (tier.w13_bytes if tier is not None else moe_w2_delta.W13_BYTES),
        # per-rank intermediate-buffer row strides (bytes): c13 bf16 [2I],
        # a2 fp8 [I], as2 f32 [I/128]. K2 = I -> identical to the old 8192/2048/64
        # literals on 1 GPU (I=2048); halved under TP2 (I=1024).
        4 * st["K2"], st["K2"], (st["K2"] // 128) * 4,
        st["E"], pairs, cap * 6, mblock, BLOCK=256)

    # ---- w13 GEMMs (both tiers) -> fused silu*up -> quant -> w2 GEMMs
    # AFRAG prefill: repack the activation to fragment-major in-place (desc 'a'
    # pointers are unchanged -- only the per-tile byte order differs) so the GEMM
    # loads each m16k32 A-fragment in one LDG.128. Numerics bit-identical to mc4.
    use_afrag = prefill and _afrag_ok
    w2tier = ("w2mc4afrag" if use_afrag else "w2mc4") if prefill else "w2"
    # AFRAG repacks COMPLETE 16-row tiles. `slots` is moe_align's OVER-ALLOCATED
    # row count (sorted_ids.numel() = topk*T + E*15), NOT a multiple of 16; the
    # desc/kernel only ever touch the first `pairs*16` rows (num_post <= pairs*16),
    # so repack exactly that tile-aligned region. Rows [pairs*16:slots] are unused
    # filler (left untouched, never read). Capacity is fine: pairs*16 <= slots <=
    # a1.shape[0]-4. (Row-major `[:slots]` here would mis-shape -> hard crash.)
    n_af = pairs * 16
    if use_afrag:
        ws["a1"][:n_af].copy_(_to_fragment_major(ws["a1"][:n_af], pairs, st["K13"]))
    _launch(w2tier, st["K13"], d[0], st["N13"], pairs, stream)
    if tier is not None and not prefill:
        _launch("w4", st["K13"], d[2], st["N13"], pairs, stream)
    act = ws["act"][:slots]
    torch.ops._C.silu_and_mul(act, ws["c13"][:slots])
    _, qs2 = per_token_group_quant_fp8(act, 128, out_q=ws["a2"][:slots])
    ws["as2"][:slots] = qs2
    if use_afrag:
        ws["a2"][:n_af].copy_(_to_fragment_major(ws["a2"][:n_af], pairs, st["K2"]))
    _launch(w2tier, st["K2"], d[1], st["N2"], pairs, stream)
    if tier is not None and not prefill:
        _launch("w4", st["K2"], d[3], st["N2"], pairs, stream)

    # ---- weighted unpermute (pad slots masked out)
    w = topk_weights.reshape(-1)[sorted_ids.clamp(max=T * top_k - 1)]
    w = torch.where(valid, w, torch.zeros_like(w)).to(torch.float32)
    out = torch.zeros(T, H, dtype=torch.float32, device=dev)
    out.index_add_(0, rows.clamp(max=T - 1),
                   ws["c2"][:slots].float() * w.unsqueeze(1))
    return out.to(x.dtype)


def _moe_w2_forward_fake(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    "moe_w2_forward",
    _moe_w2_forward,
    fake_impl=_moe_w2_forward_fake,
)


def moe_w2_forward(x, topk_weights, topk_ids, layer_key):
    return torch.ops.vllm.moe_w2_forward(x, topk_weights, topk_ids, layer_key)


@functools.cache
def ready() -> bool:
    return enabled() and _ensure_ready()
