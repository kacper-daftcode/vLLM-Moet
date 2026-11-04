"""Engram v2 inside vLLM: the Engram tables stay on NVMe (the checkpoint's own safetensors shards), a DRAM row cache
+ an io_uring fetcher thread per TP rank serve every step's rows, and vLLM's own lookup kernel dequantizes them from
a device copy - bit-identical to the pinned-table path (the same bytes, the same kernel). Installed next to vLLM by
patch_vllm_engram_v2.py together with engram_v2.py (the ctypes / extension glue), engram_v2_cuda.cu, engram_v2_abi.h
and libengram_v2.so. Everything is a no-op unless VLLM_MOET_ENGRAM_MODE=nvme.

What the patched engram.py calls:
  allocate_weights(emb)            ParallelEngramEmbedding._allocate_weights: no 23.6 GiB pinned table - two 1-row
                                   placeholders (the loader never sees the table: the weight iterator skips it)
  on_embedding_created(emb, ...)   Engram._create_embedding: the table's file / offsets / this rank's row range
  lookup(emb, indices, out, bg)    ParallelEngramEmbedding.lookup: publish this rank's row ids of the call to the
                                   fetcher (mapped mailbox, in-stream), a bounded in-stream wait for its ack, the host's
                                   staging rows copied to device landing arrays, vLLM's _engram_lookup_kernel over them
  runner_after_step()              the error word (a wait that timed out / an NVMe read that failed): the step fails
                                   closed (RuntimeError, like the CPU tier's)
  runner_before_step(scheduler_output)   the prefill lookahead (VLLM_MOET_ENGRAM_LOOKAHEAD=1): the prompt's hashes on the
                                   host, the next chunks' rows queued for background reads into the row cache

Environment:
  VLLM_MOET_ENGRAM_MODE=nvme            on (default ram = inert: vLLM's pinned host tables)
  VLLM_MOET_ENGRAM_CACHE_GIB=2          DRAM row cache per rank (plain memory, split evenly over the Engram layers)
  VLLM_MOET_ENGRAM_RESIDENT_SCALES=1    this rank's ue8m0 scales stay in RAM (768 MB per rank): one NVMe read per row, not two
  VLLM_MOET_ENGRAM_POLLED=1             the CPU tier's dispatcher thread polls the fetcher from its dedicated core (default; the
                                        fetcher's own thread runs only until the tier is created, and alone without the tier)
  VLLM_MOET_ENGRAM_FETCH_CPU=<int>      the fetcher thread's CPU when it has one (default 8*rank+33: the dispatcher core's SMT
                                        sibling - shared with the engine core / API server, which cost the 200-us window on 11-04)
  VLLM_MOET_ENGRAM_QD=64                io_uring depth per rank (reads in flight)
  VLLM_MOET_ENGRAM_TIMEOUT_MS=2000      the in-stream wait's bound; on expiry the error word is armed (fail-closed)
  VLLM_MOET_ENGRAM_MODEL_DIR=<dir>      the checkpoint (default: vLLM's model path)
  VLLM_MOET_ENGRAM_DIRECT=1             O_DIRECT reads (0 = buffered: test files on tmpfs)
  VLLM_MOET_ENGRAM_LOOKAHEAD=1          prefill lookahead (default 1; needs the scheduler output = the V2 runner hook)
  VLLM_MOET_ENGRAM_LOOKAHEAD_CHUNKS=4   how many future chunks (of max_num_batched_tokens) to keep warm
  VLLM_MOET_ENGRAM_LOG=1 / _LOG_EVERY=100   a stats line every N steps
"""
from __future__ import annotations

import atexit
import json
import os
import struct
import sys
import threading
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ENABLED = os.environ.get("VLLM_MOET_ENGRAM_MODE", "ram") == "nvme"
CACHE_GIB = float(os.environ.get("VLLM_MOET_ENGRAM_CACHE_GIB", "2"))
QD = int(os.environ.get("VLLM_MOET_ENGRAM_QD", "64"))
TIMEOUT_MS = float(os.environ.get("VLLM_MOET_ENGRAM_TIMEOUT_MS", "2000"))
DIRECT = os.environ.get("VLLM_MOET_ENGRAM_DIRECT", "1") == "1"
LOG = os.environ.get("VLLM_MOET_ENGRAM_LOG", "0") == "1"
LOG_EVERY = int(os.environ.get("VLLM_MOET_ENGRAM_LOG_EVERY", "100"))
LOOKAHEAD = os.environ.get("VLLM_MOET_ENGRAM_LOOKAHEAD", "1") == "1"
LOOKAHEAD_CHUNKS = int(os.environ.get("VLLM_MOET_ENGRAM_LOOKAHEAD_CHUNKS", "4"))
# VLLM_MOET_ENGRAM_POLLED=1 (default): the CPU tier's dispatcher thread polls the fetcher (its dedicated spinning core);
# 0 = the fetcher keeps its own thread on VLLM_MOET_ENGRAM_FETCH_CPU
POLLED_BY_TIER = os.environ.get("VLLM_MOET_ENGRAM_POLLED", "1") == "1"
# VLLM_MOET_ENGRAM_RESIDENT_SCALES=1 (default): this rank's ue8m0 scales (8 B per row, 768 MB per rank of plain memory) stay in
# RAM, so a row costs one NVMe read (its 256 values) instead of two - Fraser Price's recipe does the same; 0 = read both
RESIDENT_SCALES = os.environ.get("VLLM_MOET_ENGRAM_RESIDENT_SCALES", "1") == "1"
ROW_VALUES, ROW_SCALES = 256, 8
# the weight iterator skips the tables (patch_vllm_engram_v2.py patches should_skip_weight to honour this regex)
SKIP_WEIGHT_RE = r"(^|\.)engram\.embed\.(weight|scale)$"
if ENABLED and not os.environ.get("VLLM_MOET_SKIP_WEIGHT_RE"):
    os.environ["VLLM_MOET_SKIP_WEIGHT_RE"] = SKIP_WEIGHT_RE


def _rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return int(get_tensor_model_parallel_rank())
    except Exception:
        return 0


def _log(msg: str) -> None:
    print(f"[moet-engram-v2] rank {_rank()}: {msg}", flush=True)


class _Table:
    """One Engram layer's table of this rank: the file, the row range, the mailbox, the staging and landing."""

    def __init__(self, emb, layer_id: int, layer_hash_index: int):
        self.emb, self.layer_id, self.idx = emb, layer_id, layer_hash_index
        self.path = ""
        self.values_off = self.scales_off = self.n_rows = 0
        self.max_rows = 0
        self.req = self.ack = self.st_values = self.st_scales = 0
        self.req_d = self.ack_d = self.st_values_d = self.st_scales_d = 0
        self.land_values = self.land_scales = self.land_values_fp8 = None
        self.seq_dev = None
        self.arange = None


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.tables: dict[int, _Table] = {}    # layer_hash_index -> table
        self.n_layers = 0
        self.created = False
        self.eng = None                        # engram_v2.EngramV2 (the fetcher handle + extension)
        self.step = 0
        self.max_tokens = 0
        self.device = None
        self.lookahead = None                  # engram_v2.Lookahead
        self.kpi = dict(last_log_t=time.time(), lookups=0)


S = _State()


def _model_dir() -> str:
    d = os.environ.get("VLLM_MOET_ENGRAM_MODEL_DIR")
    if d:
        return d
    from vllm.config import get_current_vllm_config
    return get_current_vllm_config().model_config.model


def _tensor_location(model_dir: str, name: str) -> tuple[str, int, int, list[int]]:
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][name]
    path = os.path.join(model_dir, shard)
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    ent = hdr[name]
    base = 8 + n + ent["data_offsets"][0]
    return path, base, ent["data_offsets"][1] - ent["data_offsets"][0], ent["shape"]


# ---------------------------------------------------------------------------------------------------------------
# ParallelEngramEmbedding._allocate_weights
# ---------------------------------------------------------------------------------------------------------------
def allocate_weights(emb):
    """No pinned table: two placeholders (the loader skips the checkpoint's engram.embed tensors)."""
    if not ENABLED:
        return None
    scale_dim = emb.dim // emb.block_size
    w = torch.empty(1, emb.dim, dtype=torch.float8_e4m3fn, device="cpu")
    s = torch.empty(1, scale_dim, dtype=torch.uint8, device="cpu")
    return w, s


# ---------------------------------------------------------------------------------------------------------------
# Engram._create_embedding
# ---------------------------------------------------------------------------------------------------------------
def on_embedding_created(emb, layout, layer_hash_index: int) -> None:
    if not ENABLED:
        return
    if emb.dp_size > 1 or getattr(emb, "dp_shared_memory", False):
        raise RuntimeError("[moet-engram-v2] DP-sharded / DP-shared Engram tables are not supported in nvme mode")
    layer_id = int(layout.layer_ids[layer_hash_index])
    T = _Table(emb, layer_id, layer_hash_index)
    md = _model_dir()
    path_w, base_w, nbytes_w, shape_w = _tensor_location(md, f"layers.{layer_id}.engram.embed.weight")
    path_s, base_s, nbytes_s, shape_s = _tensor_location(md, f"layers.{layer_id}.engram.embed.scale")
    if path_w != path_s:
        raise RuntimeError(f"[moet-engram-v2] layer {layer_id}: weight and scale in different shards")
    if shape_w != [emb.num_embeddings, emb.dim] or shape_s != [emb.num_embeddings, emb.dim // emb.block_size]:
        raise RuntimeError(f"[moet-engram-v2] layer {layer_id}: table shape {shape_w} / {shape_s} != "
                           f"{[emb.num_embeddings, emb.dim]} / {[emb.num_embeddings, emb.dim // emb.block_size]}")
    T.path, T.values_off, T.scales_off, T.n_rows = path_w, base_w, base_s, int(emb.num_embeddings)
    from vllm.config import get_current_vllm_config
    S.max_tokens = int(get_current_vllm_config().scheduler_config.max_num_batched_tokens)
    T.max_rows = S.max_tokens * int(emb.part_n_hash_cols)
    S.tables[layer_hash_index] = T
    S.n_layers = len(layout.layer_ids)
    emb._moet_engram_v2_table = T
    _log(f"layer {layer_id} (hash index {layer_hash_index}): table {os.path.basename(path_w)} rows {T.n_rows} "
         f"(this rank's heads {emb.head_start}..{emb.head_start + emb.part_n_hash_cols - 1} = rows "
         f"[{emb.vocab_start_idx}, {emb.vocab_end_idx})), staging {T.max_rows} rows; no pinned table "
         f"({emb.part_num_embeddings * 264 / 2**30:.2f} GiB saved)")


def _create(device: torch.device) -> None:
    if S.created:
        return
    from engram_v2 import EngramV2

    tables = [S.tables[i] for i in sorted(S.tables)]
    if len(tables) != S.n_layers:
        raise RuntimeError(f"[moet-engram-v2] {len(tables)} of {S.n_layers} Engram layers registered at the first lookup")
    r = _rank()
    fetch_cpu = int(os.environ.get("VLLM_MOET_ENGRAM_FETCH_CPU", str(8 * r + 33)))
    S.eng = EngramV2([dict(path=T.path, values_off=T.values_off, scales_off=T.scales_off, n_rows=T.n_rows,
                           vocab_start=int(T.emb.vocab_start_idx), vocab_end=int(T.emb.vocab_end_idx), max_rows=T.max_rows)
                      for T in tables],
                     cache_bytes=int(CACHE_GIB * 2**30), fetch_cpu=fetch_cpu, queue_depth=QD, direct=DIRECT, device=device,
                     timeout_ms=TIMEOUT_MS, resident_scales=RESIDENT_SCALES)
    for i, T in enumerate(tables):
        L = S.eng.layers[i]
        T.req, T.ack, T.st_values, T.st_scales = L.req, L.ack, L.st_values, L.st_scales
        T.req_d, T.ack_d, T.st_values_d, T.st_scales_d = L.req_d, L.ack_d, L.st_values_d, L.st_scales_d
        T.land_values, T.land_scales, T.seq_dev, T.arange = L.land_values, L.land_scales, L.seq_dev, L.arange
        T.land_values_fp8 = L.land_values.view(torch.float8_e4m3fn)
        T.eng_index = i
    S.eng.start()
    S.device = device
    S.created = True
    if LOOKAHEAD:
        try:
            from engram_v2 import Lookahead
            S.lookahead = Lookahead(S.eng, tables, _model_dir(), S.max_tokens, LOOKAHEAD_CHUNKS)
        except Exception as ex:
            _log(f"lookahead disabled: {ex!r}")
            S.lookahead = None

    def _shutdown():
        try:
            if LOG:
                _log("exit: " + S.eng.stats_line())
            S.eng.close()
        except Exception:
            pass

    atexit.register(_shutdown)
    _log(f"fetcher started: {len(tables)} tables, row cache {CACHE_GIB:.1f} GiB per rank "
         f"({S.eng.cache_rows()} rows), io_uring depth {QD}, {'O_DIRECT' if DIRECT else 'buffered'}, fetch CPU {fetch_cpu}, "
         f"wait timeout {TIMEOUT_MS:.0f} ms, lookahead {'on' if S.lookahead else 'off'}, "
         f"resident scales {'on (' + str(S.eng.stats()['resident_scale_bytes'] // 2**20) + ' MiB)' if RESIDENT_SCALES else 'off'}")
    # the CPU tier's dispatcher polls the fetcher from its dedicated core when the tier exists (else the tier attaches
    # us when it is created: vllm_moet_cpu_tier._create_tier_if_needed -> attach_dispatcher)
    if POLLED_BY_TIER:
        try:
            import vllm_moet_cpu_tier as _tier
            if _tier.ENABLED and _tier.S.created and _tier.S.tier is not None:
                attach_dispatcher(_tier.S.tier)
        except ImportError:
            pass


def attach_dispatcher(tier) -> None:
    """Called by the CPU tier (or by _create above): the fetcher's work moves to the tier's dispatcher thread."""
    if not ENABLED or not S.created or not POLLED_BY_TIER or S.eng is None or getattr(S.eng, "attached", None) is not None:
        return
    S.eng.attach_dispatcher(tier)
    _log("fetcher attached to the CPU tier's dispatcher (polled from its core; the fetcher thread stopped)")


# ---------------------------------------------------------------------------------------------------------------
# ParallelEngramEmbedding.lookup
# ---------------------------------------------------------------------------------------------------------------
def lookup(emb, indices: torch.Tensor, out: torch.Tensor, background: bool) -> bool:
    """Publish -> bounded wait + landing copy -> vLLM's lookup kernel over the landing. Returns True = handled."""
    if not ENABLED:
        return False
    T = getattr(emb, "_moet_engram_v2_table", None)
    if T is None:
        raise RuntimeError("[moet-engram-v2] lookup on an embedding that was not registered (patch order?)")
    if not S.created:
        with S.lock:
            _create(indices.device)
    if out.dtype == torch.uint8:
        raise RuntimeError("[moet-engram-v2] the packed (DP-batched) Engram path is not supported in nvme mode")
    n_tokens = int(indices.shape[0])
    rows = n_tokens * int(emb.part_n_hash_cols)
    if rows > T.max_rows:
        raise RuntimeError(f"[moet-engram-v2] {rows} rows > staging {T.max_rows}")
    ext = S.eng.ext
    ext.publish(T.req_d, T.seq_dev, indices, int(emb.head_start), int(emb.part_n_hash_cols), n_tokens, T.max_rows)
    ext.wait_copy(T.ack_d, T.req_d, T.seq_dev, T.st_values_d, T.st_scales_d, T.land_values, T.land_scales, S.eng.err_d,
                  S.eng.timeout_ns, rows)
    from vllm.models.deepseek_v41.common.engram import _engram_lookup_kernel
    from vllm.triton_utils import triton

    tiles = triton.cdiv(rows, 16)
    grid = min(tiles, emb._num_sms // 2 if background else emb._num_sms)
    ids = T.arange[:rows]
    _engram_lookup_kernel[(grid,)](
        T.land_values_fp8, T.land_scales, ids, ids, out,
        0, T.max_rows, rows, 0, 0,
        HEAD_START=emb.head_start, LOCAL_HEADS=emb.part_n_hash_cols, TOTAL_HEADS=emb.n_hash_cols, DIM=emb.dim,
        QUANT_BLOCK=emb.block_size, BLOCK_R=16, GRID=grid, SORTED=True, PACKED=False, OUT_STRIDE=out.stride(0))
    S.kpi["lookups"] += 1
    return True


# ---------------------------------------------------------------------------------------------------------------
# around the model runner's execute_model
# ---------------------------------------------------------------------------------------------------------------
def runner_before_step(scheduler_output=None) -> None:
    if not S.created or S.lookahead is None or scheduler_output is None or torch.cuda.is_current_stream_capturing():
        return
    try:
        S.lookahead.on_schedule(scheduler_output)
    except Exception as ex:   # the lookahead is an optimisation: never fail a step for it
        if not S.kpi.get("la_err_reported"):
            S.kpi["la_err_reported"] = True
            _log(f"lookahead error (disabled for this process): {ex!r}")
        S.lookahead = None


def runner_after_step() -> None:
    if not S.created or torch.cuda.is_current_stream_capturing():
        return
    S.step += 1
    err = S.eng.error()
    if err:
        raise RuntimeError(f"[moet-engram-v2] the Engram fetcher failed closed (error word {err}: "
                           f"{'wait timeout' if err == 1 else 'host status ' + str(err - 16)}); {S.eng.stats_line()}")
    if LOG and (S.step % LOG_EVERY == 0 or S.step <= 2):
        now = time.time()
        _log(f"step {S.step}: {S.eng.stats_line(reset=True)}; {LOG_EVERY / max(1e-6, now - S.kpi['last_log_t']):.1f} steps/s"
             + (f"; lookahead {S.lookahead.stats_line()}" if S.lookahead is not None else ""))
        S.kpi["last_log_t"] = now
