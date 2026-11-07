"""Engram v2 GPU harness (one GPU, inside the serving image; the mini-real models have no Engram, so this is the test
vehicle short of a TP4 boot): vLLM's own hash kernel on real tokenized text, the stock lookup kernel over a pinned
table slice as the reference, the NVMe path (publish -> fetcher -> bounded wait + landing copy -> the same lookup
kernel over the landing) compared bit for bit, the host-side hash port checked against the kernel, the access
pattern measured (rows per token, distinct rows per chunk), the publish -> ack latency on the GPU timeline, and a
CUDA-graph replay loop of the decode-size path.

    flock -w 7200 /tmp/exl3diff-gpu1.lock docker run --rm --gpus '"device=1"' --name moet-eng-harness \
      --security-opt seccomp=$REPO/docker/sm120/seccomp-io_uring.json \
      -v /mnt/nvme/models/DeepSeek-V4.1-Flash:/model:ro -v $REPO:/repo:ro -v /mnt/nvme/cache/ds41/dot-cache:/root/.cache \
      --entrypoint python3 vllm-moet-sm120:dsv41-nightly-df3bd094-hyb2 /repo/tools/dsv41_5090/engram_v2/engram_v2_harness.py \
      --model-dir /model --rank 0 --tp 4 [--cache-gib 1] [--text /repo/README.md,/repo/docs/moet-hybrid.md] [--json /root/.cache/engram-v2-harness.json]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def tensor_location(model_dir: str, name: str):
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][name]
    path = os.path.join(model_dir, shard)
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    ent = hdr[name]
    return path, 8 + n + ent["data_offsets"][0], ent["shape"]


class Cfg:
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/model")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--cache-gib", type=float, default=1.0)
    ap.add_argument("--qd", type=int, default=64)
    ap.add_argument("--fetch-cpu", type=int, default=33)
    ap.add_argument("--text", default="/repo/README.md,/repo/docs/moet-hybrid.md,/repo/tools/dsv41_5090/cpu_tier/vllm_moet_cpu_tier.py")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=500, help="decode-size iterations (eager) per token count")
    ap.add_argument("--graph-steps", type=int, default=1000)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    results: dict = {"args": vars(args)}

    from vllm.models.deepseek_v41.common.engram import (EngramLayout, _engram_lookup_kernel, _hash_ids_kernel,
                                                        build_compressed_token_map, compute_hash_multipliers)
    from vllm.triton_utils import triton
    from engram_v2 import EngramV2, HostHash
    from transformers import AutoTokenizer

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    cfg = cfg.get("text_config", cfg)
    c = Cfg()
    for k, v in cfg.items():
        setattr(c, k, v)
    layout = EngramLayout(c)
    n_cols = layout.n_hash_cols
    local_heads = triton.cdiv(n_cols, args.tp)
    head_start = args.rank * local_heads
    head_sizes = [tuple(size for order in layout.primes[i] for size in order) for i in range(len(layout.layer_ids))]
    vocab_start = [sum(hs[:head_start]) for hs in head_sizes]
    vocab_end = [sum(hs[:head_start + local_heads]) for hs in head_sizes]
    print(f"layout: layers {layout.layer_ids}, {n_cols} hash columns, rank {args.rank}/{args.tp}: heads {head_start}..{head_start + local_heads - 1}, "
          f"rows {[(a, b) for a, b in zip(vocab_start, vocab_end)]}", flush=True)

    # ---- the token stream and vLLM's hashes --------------------------------------------------------------------
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    token_map, vocab = build_compressed_token_map(tok)
    assert vocab == layout.compressed_vocab_size
    text = "".join(open(p).read() for p in args.text.split(",") if os.path.isfile(p))
    ids_all = np.array(tok.encode(text, add_special_tokens=False), dtype=np.int64)
    print(f"tokenizer + token map {time.time() - t0:.1f} s; corpus {len(ids_all)} tokens", flush=True)
    assert len(ids_all) >= args.max_tokens + 16, "corpus too short"
    token_map_t = torch.tensor(token_map, dtype=torch.int32, device=dev)
    primes_t = torch.tensor(layout.primes, device=dev)
    offsets_t = layout.offsets.to(dev)
    mult_t = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab).to(dev)
    pad_id = token_map[layout.pad_token_id]
    n_layers, max_ngram = mult_t.shape
    lookback_depth = max_ngram - 1

    def gpu_hashes(ids: np.ndarray, start_pos: int, lookback: np.ndarray | None) -> torch.Tensor:
        """vLLM's _hash_ids_kernel for one request chunk: tokens ids at positions start_pos.., lookback = the 3 ids before."""
        T = len(ids)
        input_ids = torch.tensor(ids, dtype=torch.int32, device=dev)
        positions = torch.arange(start_pos, start_pos + T, dtype=torch.int64, device=dev)
        qsl = torch.tensor([0, T], dtype=torch.int32, device=dev)
        dead = torch.zeros(T, dtype=torch.bool, device=dev)
        lb = torch.full((1, lookback_depth), -1, dtype=torch.int32, device=dev)
        if lookback is not None:   # [depth]: ids at positions start_pos-1, start_pos-2, ... (col 0 = the one just before)
            lb[0, : len(lookback)] = torch.tensor(np.ascontiguousarray(lookback), dtype=torch.int32, device=dev)
        lb_dead = torch.zeros_like(lb, dtype=torch.bool)
        out = torch.empty((T, n_layers, n_cols), dtype=torch.int32, device=dev)
        _hash_ids_kernel[(triton.cdiv(T, 32), n_layers)](
            input_ids, token_map_t, dead, positions, None, qsl, mult_t, primes_t, offsets_t, None, lb, lb_dead, out, T, 0, pad_id,
            input_stride=input_ids.stride(0), mask_stride=dead.stride(0), position_stride=positions.stride(0), table_stride=0,
            table_col_stride=0, query_stride=qsl.stride(0), num_query_rows=1, num_table_rows=0, max_blocks=0, cache_block_size=1,
            MAX_NGRAM=max_ngram, num_heads=layout.n_heads, BLOCK_T=32, BLOCK_H=triton.next_power_of_2(layout.n_heads), dead_id=-1,
            lookback_depth=lookback_depth, lookback_row_stride=lb.stride(0), lookback_col_stride=lb.stride(1),
            lookback_mask_row_stride=lb_dead.stride(0), lookback_mask_col_stride=lb_dead.stride(1), num_warps=4)
        return out

    # ---- 1. the host hash port against the kernel ------------------------------------------------------------------
    hh = HostHash(args.model_dir, layout=layout, token_map=token_map)
    T = args.max_tokens
    h_gpu = gpu_hashes(ids_all[:T], 0, None).cpu().numpy()          # [T, L, 24], sequence start
    ok = True
    for li in range(n_layers):
        h_host = hh.hash_positions(ids_all[:T], 0, T, li)
        eq = np.array_equal(h_host, h_gpu[:, li])
        ok &= eq
        print(f"host hash vs kernel, layer {layout.layer_ids[li]}, {T} tokens from position 0: {'EQUAL' if eq else 'DIFFER'}")
    # a later chunk: positions 1000.. with the lookback supplied (as the V2 runner does)
    s0, s1 = 1000, 1000 + 2048
    h_gpu2 = gpu_hashes(ids_all[s0:s1], s0, ids_all[s0 - 1:s0 - 1 - lookback_depth:-1]).cpu().numpy()
    for li in range(n_layers):
        h_host = hh.hash_positions(ids_all, s0, s1, li)
        eq = np.array_equal(h_host, h_gpu2[:, li])
        ok &= eq
        print(f"host hash vs kernel, layer {layout.layer_ids[li]}, chunk at position {s0}: {'EQUAL' if eq else 'DIFFER'}")
    results["host_hash_equal"] = bool(ok)

    # ---- 2. the access pattern -------------------------------------------------------------------------------------
    pat = {}
    for li in range(n_layers):
        cols = h_gpu[:, li, head_start:head_start + local_heads]
        per_chunk = {}
        for win in (1, 8, 2048, 4096):
            if win > T:
                continue
            u = [len(np.unique(cols[i:i + win].reshape(-1))) for i in range(0, T - win + 1, max(1, win))][:64]
            per_chunk[str(win)] = float(np.mean(u))
        pat[str(layout.layer_ids[li])] = dict(rows_per_token_rank=local_heads, rows_per_token_all=n_cols, distinct_rows_per_window=per_chunk)
        print(f"access pattern layer {layout.layer_ids[li]} (rank {args.rank}: {local_heads} rows/token, {local_heads * 264} B/token; all ranks {n_cols} rows = {n_cols * 264} B/token): "
              + ", ".join(f"{w}-token window: {v:.0f} distinct rows" for w, v in per_chunk.items()))
    results["access_pattern"] = pat

    # ---- 3. the NVMe path vs the stock kernel over a pinned slice ---------------------------------------------------
    layers_cfg = []
    for li, lid in enumerate(layout.layer_ids):
        pw, bw, shw = tensor_location(args.model_dir, f"layers.{lid}.engram.embed.weight")
        ps, bs, shs = tensor_location(args.model_dir, f"layers.{lid}.engram.embed.scale")
        assert pw == ps and shw == [layout.num_embeddings[li], layout.head_dim]
        layers_cfg.append(dict(path=pw, values_off=bw, scales_off=bs, n_rows=shw[0], vocab_start=vocab_start[li], vocab_end=vocab_end[li],
                               max_rows=args.max_tokens * local_heads))
    eng = EngramV2(layers_cfg, cache_bytes=int(args.cache_gib * 2**30), fetch_cpu=args.fetch_cpu, queue_depth=args.qd, direct=True,
                   device=dev, timeout_ms=10000)
    eng.start()
    mm = [np.memmap(l["path"], dtype=np.uint8, mode="r") for l in layers_cfg]

    def slice_table(li: int, rows: np.ndarray):
        """pinned mini-table of the given global rows (sorted, unique) + the remap; the reference's bytes come from a plain read"""
        l = layers_cfg[li]
        K = len(rows)
        W = torch.empty(K, 256, dtype=torch.uint8, pin_memory=True)
        Sc = torch.empty(K, 8, dtype=torch.uint8, pin_memory=True)
        Wn, Sn = W.numpy(), Sc.numpy()
        for k, r in enumerate(rows.tolist()):
            ov, os_ = l["values_off"] + r * 256, l["scales_off"] + r * 8
            Wn[k] = mm[li][ov:ov + 256]
            Sn[k] = mm[li][os_:os_ + 8]
        return W, Sc

    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    num_sms = torch.cuda.get_device_properties(dev).multi_processor_count

    def stock_lookup(li: int, hashes_T: torch.Tensor, out: torch.Tensor):
        """the stock kernel over a pinned slice holding exactly the rows this call needs (ids remapped to the slice)"""
        h = hashes_T[:, head_start:head_start + local_heads].cpu().numpy()
        rows = np.unique(h.reshape(-1))
        rows = rows[(rows >= 0)]
        W, Sc = slice_table(li, rows)
        remap = {int(r): k for k, r in enumerate(rows.tolist())}
        ids = hashes_T.clone()
        hl = ids[:, head_start:head_start + local_heads].cpu().numpy()
        for t in range(hl.shape[0]):
            for j in range(local_heads):
                hl[t, j] = remap.get(int(hl[t, j]), -1)
        ids[:, head_start:head_start + local_heads] = torch.tensor(hl, dtype=torch.int32, device=dev)
        n_rows = ids.shape[0] * local_heads
        grid = min(triton.cdiv(n_rows, 16), num_sms)
        _engram_lookup_kernel[(grid,)](
            get_accelerator_view_from_cpu_tensor(W.view(torch.float8_e4m3fn)), get_accelerator_view_from_cpu_tensor(Sc), ids, ids, out,
            0, len(rows), n_rows, ids.stride(0), ids.stride(1), HEAD_START=head_start, LOCAL_HEADS=local_heads, TOTAL_HEADS=n_cols,
            DIM=layout.head_dim, QUANT_BLOCK=32, BLOCK_R=16, GRID=grid, SORTED=False, PACKED=False, OUT_STRIDE=out.stride(0))
        torch.cuda.synchronize()
        return W, Sc   # keep alive until the kernel is done

    def nvme_lookup(li: int, hashes_T: torch.Tensor, out: torch.Tensor, stream=None):
        L = eng.layers[li]
        n_tokens = hashes_T.shape[0]
        rows = n_tokens * local_heads
        eng.ext.publish(L.req_d, L.seq_dev, hashes_T, head_start, local_heads, n_tokens, L.max_rows)
        eng.ext.wait_copy(L.ack_d, L.req_d, L.seq_dev, L.st_values_d, L.st_scales_d, L.land_values, L.land_scales, eng.err_d, eng.timeout_ns, rows)
        grid = min(triton.cdiv(rows, 16), num_sms)
        ids = L.arange[:rows]
        _engram_lookup_kernel[(grid,)](
            L.land_values.view(torch.float8_e4m3fn), L.land_scales, ids, ids, out, 0, L.max_rows, rows, 0, 0,
            HEAD_START=head_start, LOCAL_HEADS=local_heads, TOTAL_HEADS=n_cols, DIM=layout.head_dim, QUANT_BLOCK=32, BLOCK_R=16,
            GRID=grid, SORTED=True, PACKED=False, OUT_STRIDE=out.stride(0))

    ident = {}
    for T_ in (1, 4, 8, 2048, args.max_tokens):
        for li in range(n_layers):
            for rep in range(2):   # cold, then warm (the same rows: cache hits)
                pos = 100 + rep * 0   # the same chunk twice
                h = gpu_hashes(ids_all[pos:pos + T_], pos, ids_all[pos - 1:pos - 1 - lookback_depth:-1])[:, li].contiguous()
                out_ref = torch.zeros(T_, local_heads, layout.head_dim, dtype=torch.bfloat16, device=dev)
                out_nv = torch.zeros_like(out_ref)
                keep = stock_lookup(li, h, out_ref)
                ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ev0.record()
                nvme_lookup(li, h, out_nv)
                ev1.record()
                torch.cuda.synchronize()
                eq = torch.equal(out_ref.view(torch.int16), out_nv.view(torch.int16))
                st = eng.stats()
                key = f"T{T_}_L{layout.layer_ids[li]}_{'cold' if rep == 0 else 'warm'}"
                ident[key] = dict(equal=bool(eq), gpu_ms=float(ev0.elapsed_time(ev1)), hits=st["hits"], misses=st["misses"])
                print(f"identity T={T_} layer {layout.layer_ids[li]} {'cold' if rep == 0 else 'warm'}: nvme path == stock pinned path: {eq}; "
                      f"publish+wait+copy+lookup {ev0.elapsed_time(ev1):.2f} ms on the GPU timeline; |out| max {out_ref.float().abs().max().item():.3g}; "
                      f"fetcher: {eng.stats_line(reset=True)}", flush=True)
                assert eq, "MISMATCH"
                del keep
    results["identity"] = ident
    assert eng.error() == 0, f"error word {eng.error()}"

    # ---- 4. the decode-size latency, eager: publish -> ack on the GPU timeline (fresh rows every step) -------------
    lat = {}
    for T_ in (1, 8):
        times = []
        base = 6000 if T_ == 8 else 10000   # text the identity tests above did not touch: cold rows (the real decode case)
        for s in range(args.steps):
            pos = base + (s * T_) % (len(ids_all) - base - 300)
            h_all = gpu_hashes(ids_all[pos:pos + T_], pos, ids_all[pos - 1:pos - 1 - lookback_depth:-1])
            outs = [torch.empty(T_, local_heads, layout.head_dim, dtype=torch.bfloat16, device=dev) for _ in range(n_layers)]
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True)
            evs = [torch.cuda.Event(enable_timing=True) for _ in range(n_layers)]
            ev0.record()
            for li in range(n_layers):
                nvme_lookup(li, h_all[:, li].contiguous(), outs[li])
                evs[li].record()
            torch.cuda.synchronize()
            times.append([ev0.elapsed_time(e) for e in evs])
            time.sleep(0.005)   # a decode step's cadence
        a = np.array(times)
        lat[f"T{T_}"] = dict(layer1_ms=dict(median=float(np.median(a[:, 0])), p90=float(np.percentile(a[:, 0], 90)), p99=float(np.percentile(a[:, 0], 99)), max=float(a[:, 0].max())),
                           both_ms=dict(median=float(np.median(a[:, -1])), p90=float(np.percentile(a[:, -1], 90)), p99=float(np.percentile(a[:, -1], 99)), max=float(a[:, -1].max())))
        print(f"decode-size T={T_} x {args.steps} (eager, fresh text each step, 5 ms gaps): layer-1 rows ready at median {np.median(a[:, 0]) * 1e3:.0f} us "
              f"p90 {np.percentile(a[:, 0], 90) * 1e3:.0f} p99 {np.percentile(a[:, 0], 99) * 1e3:.0f} max {a[:, 0].max() * 1e3:.0f}; both layers median "
              f"{np.median(a[:, -1]) * 1e3:.0f} us p99 {np.percentile(a[:, -1], 99) * 1e3:.0f}; fetcher: {eng.stats_line(reset=True)}", flush=True)
    results["decode_latency"] = lat

    # ---- 5. prefill-size cold vs lookahead-warm -------------------------------------------------------------------
    pf = {}
    for T_ in (2048, args.max_tokens):
        for mode in ("cold", "warm"):
            pos = 300 + (0 if T_ == 2048 else 2500)
            pos = min(pos, len(ids_all) - T_ - 10)
            h_all = gpu_hashes(ids_all[pos:pos + T_], pos, ids_all[pos - 1:pos - 1 - lookback_depth:-1])
            if mode == "warm":   # the lookahead's effect: the rows prefetched into the cache first
                for li in range(n_layers):
                    rows = np.unique(h_all[:, li, head_start:head_start + local_heads].cpu().numpy().reshape(-1))
                    eng.prefetch(li, rows)
                t_pf = time.perf_counter()
                for _ in range(2000):   # until every queued id was read or found cached (bounded: 20 s)
                    st = eng.stats()
                    if st["prefetch_queued"] > 0 and st["prefetch_rows"] + st["prefetch_hits"] >= st["prefetch_queued"]:
                        break
                    time.sleep(0.01)
                print(f"  lookahead warm-up of {st['prefetch_queued']} rows took {time.perf_counter() - t_pf:.2f} s "
                      f"({st['prefetch_reads']} reads)", flush=True)
            outs = [torch.empty(T_, local_heads, layout.head_dim, dtype=torch.bfloat16, device=dev) for _ in range(n_layers)]
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True)
            evs = [torch.cuda.Event(enable_timing=True) for _ in range(n_layers)]
            ev0.record()
            for li in range(n_layers):
                nvme_lookup(li, h_all[:, li].contiguous(), outs[li])
                evs[li].record()
            torch.cuda.synchronize()
            t = [ev0.elapsed_time(e) for e in evs]
            pf[f"T{T_}_{mode}"] = dict(layer1_ms=t[0], both_ms=t[-1])
            print(f"prefill-size T={T_} {mode}: layer-1 rows ready at {t[0]:.1f} ms, both layers {t[-1]:.1f} ms; fetcher: {eng.stats_line(reset=True)}", flush=True)
    results["prefill_latency"] = pf

    # ---- 6. CUDA graph: the decode path captured and replayed (static hash buffer, new ids copied in before each replay)
    T_ = 1
    static_h = torch.zeros(T_, n_layers, n_cols, dtype=torch.int32, device=dev)
    outs = [torch.zeros(T_, local_heads, layout.head_dim, dtype=torch.bfloat16, device=dev) for _ in range(n_layers)]
    side = torch.cuda.Stream(device=dev)
    # warm the Triton JIT paths outside the capture
    for li in range(n_layers):
        nvme_lookup(li, static_h[:, li].contiguous(), outs[li])
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        for li in range(n_layers):
            nvme_lookup(li, static_h[:, li], outs[li])
    torch.cuda.synchronize()
    mism = 0
    rep_t = []
    for s in range(args.graph_steps):
        pos = 500 + s
        h_all = gpu_hashes(ids_all[pos:pos + T_], pos, ids_all[pos - 1:pos - 1 - lookback_depth:-1])
        static_h.copy_(h_all)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        rep_t.append((time.perf_counter() - t0) * 1e6)
        if s % 50 == 0:
            for li in range(n_layers):
                ref = torch.zeros_like(outs[li])
                keep = stock_lookup(li, h_all[:, li].contiguous(), ref)
                if not torch.equal(ref.view(torch.int16), outs[li].view(torch.int16)):
                    mism += 1
                del keep
        time.sleep(0.004)
    rep_t = np.array(rep_t)
    results["graph"] = dict(replays=args.graph_steps, checked_mismatches=mism, replay_us=dict(median=float(np.median(rep_t)), p99=float(np.percentile(rep_t, 99)), max=float(rep_t.max())))
    print(f"CUDA graph: {args.graph_steps} replays of the T=1 publish/wait/copy/lookup path for both layers, checked every 50th against the stock "
          f"kernel: {mism} mismatches; replay wall time median {np.median(rep_t):.0f} us p99 {np.percentile(rep_t, 99):.0f} max {rep_t.max():.0f}; "
          f"error word {eng.error()}; fetcher: {eng.stats_line(reset=True)}", flush=True)
    assert mism == 0 and eng.error() == 0
    eng.close()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.json}")
    print("HARNESS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
