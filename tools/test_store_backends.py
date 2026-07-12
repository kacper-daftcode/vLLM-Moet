# Sanity for moe_w2_store backends: pinned vs mmap-pack vs tiered must
# deliver BYTE-IDENTICAL rows into the GPU pool, the pack must persist
# across a store re-open (boot-from-pack), and the tiered arena must stay
# correct under eviction pressure and batch overflow. Run inside the
# serving image:
#   docker run --rm --gpus '"device=7"' -v /root/workspace/vllm-v0.24.0:/vllm-024 \
#     -v /workspace/moet-serve:/serve --entrypoint python3 \
#     vllm-moet-sm120:v024-dev /serve/test_store_backends.py
import os
import shutil
import sys

import torch

sys.path.insert(0, "/vllm-024")

PACK_DIR = "/serve/packs-test"
N_LAYERS, E = 4, 16
W13, W2 = (1 << 20) + 4096, (1 << 19) + 512   # slot NOT 4K-aligned -> stride pads
SLOT = W13 + W2


def build_parts(gen):
    torch.manual_seed(gen)
    return [
        (torch.randint(0, 256, (E, W13), dtype=torch.uint8, device="cuda"),
         torch.randint(0, 256, (E, W2), dtype=torch.uint8, device="cuda"))
        for _ in range(N_LAYERS)
    ]


def ref_row(parts, li, ei):
    return torch.cat((parts[li][0][ei], parts[li][1][ei]))


def check_tier(tier, parts, label):
    ok = 0
    # direct promote path (manager idiom)
    with tier._lock:
        for li in range(N_LAYERS):
            for ei in (0, 3, E - 1):
                slot = tier._take_slot()
                assert slot is not None
                tier._promote(li, ei, slot)
                got = tier.pool[slot]
                assert torch.equal(got, ref_row(parts, li, ei)), (
                    label, "promote", li, ei)
                ok += 1
    # ensure_resident path (prefill idiom, batched rows_for)
    ids = torch.tensor([1, 5, 7, 11], device="cuda")
    n = tier.ensure_resident(2, ids)
    assert n == 4, (label, "ensure_resident fetched", n)
    for ei in ids.tolist():
        slot = int(tier._mirror[2, ei])
        assert slot >= 0
        assert torch.equal(tier.pool[slot], ref_row(parts, 2, ei)), (
            label, "resident", ei)
        ok += 1
    print(f"[{label}] {ok} slots byte-identical  OK")


def make_tier(tag):
    from vllm.model_executor.layers.quantization.utils.moe_w2_delta import (
        DeltaTier)
    return DeltaTier(N_LAYERS, E, torch.device("cuda"),
                     w13_bytes=W13, w2_bytes=W2, pool_gb=0.1,
                     policy="freq", tag=tag)


def slots_for_gb(gb):
    stride = (SLOT + 4095) // 4096 * 4096
    return max(int(gb * 2**30) // stride, 16)


def check_store_rows(store, parts, pairs, label, scan=False):
    rows = store.rows_for(pairs, scan=scan)
    for (li, ei), row in zip(pairs, rows):
        assert row.shape[0] == SLOT, (label, row.shape)
        assert torch.equal(row, ref_row(parts, li, ei).cpu()), (
            label, li, ei)


def main():
    parts = build_parts(1234)

    os.environ.pop("VLLM_MOE_W2_STORE_DIR", None)
    os.environ.pop("VLLM_MOE_W2_BASE_RAM_GB", None)
    tier = make_tier("t-pinned")
    for li in range(N_LAYERS):
        tier.add_layer_host_planes(li, *parts[li])
    check_tier(tier, parts, "pinned")

    shutil.rmtree(PACK_DIR, ignore_errors=True)
    os.environ["VLLM_MOE_W2_STORE_DIR"] = PACK_DIR
    tier = make_tier("t-mmap")
    for li in range(N_LAYERS):
        tier.add_layer_host_planes(li, *parts[li])
    cold_st = tier._store.stats()
    assert cold_st["write_cache_drop_calls"] == N_LAYERS, cold_st
    assert cold_st["write_cache_drop_bytes"] == (
        N_LAYERS * E * tier._store.stride), cold_st
    check_tier(tier, parts, "mmap-pack cold")

    # boot-from-pack: new tier, same dir; add_layer must SKIP (feed garbage
    # to prove rows come from the pack, not from these tensors)
    tier2 = make_tier("t-mmap")
    garbage = [(torch.zeros_like(parts[li][0]), torch.zeros_like(parts[li][1]))
               for li in range(N_LAYERS)]
    for li in range(N_LAYERS):
        tier2.add_layer_host_planes(li, *garbage[li])
    check_tier(tier2, parts, "mmap-pack reboot")

    st = tier2._store.stats()
    assert st["reads"] > 0
    print(f"pack reads: {st['reads']} rows, {st['read_bytes']/2**20:.1f} MiB")

    # ---- tiered backend (pinned arena + O_DIRECT), tag must be "base" ----
    shutil.rmtree(PACK_DIR, ignore_errors=True)
    os.environ["VLLM_MOE_W2_BASE_RAM_GB"] = "0.1"   # 67 slots > 64 rows
    tier = make_tier("base")
    from vllm.model_executor.layers.quantization.utils.moe_w2_store import (
        TieredPackStore)
    assert isinstance(tier._store, TieredPackStore), type(tier._store)
    for li in range(N_LAYERS):
        tier.add_layer_host_planes(li, *parts[li])
    cold_st = tier._store.stats()
    assert cold_st["write_cache_drop_calls"] == N_LAYERS, cold_st
    assert cold_st["write_cache_drop_bytes"] == (
        N_LAYERS * E * tier._store.stride), cold_st
    check_tier(tier, parts, "tiered cold")
    st = tier._store.stats()
    assert st["miss_rows"] > 0, st

    # warm pass: every row already in the arena -> pure ram hits
    miss_before = tier._store.stats()["miss_rows"]
    all_pairs = [(li, ei) for li in range(N_LAYERS) for ei in range(E)]
    check_store_rows(tier._store, parts, all_pairs, "tiered warm sweep")
    check_store_rows(tier._store, parts, all_pairs, "tiered warm sweep2")
    st = tier._store.stats()
    assert st["hit_rows"] >= len(all_pairs), st            # 2nd sweep all-hit
    assert st["miss_rows"] <= len(all_pairs), st           # each row read <=1x
    print(f"[tiered warm] {st['hit_rows']} ram hits / {st['miss_rows']} nvme "
          f"rows, arena {st['arena_used']}/{st['arena_slots']}  OK")

    # reboot-from-pack with garbage staging: rows must come from the pack
    tier3 = make_tier("base")
    for li in range(N_LAYERS):
        tier3.add_layer_host_planes(li, *garbage[li])
    check_tier(tier3, parts, "tiered reboot")

    # eviction stress: arena of 24 slots < 64 rows; sweep everything twice
    # in small batches -> forced evictions + re-reads, bytes always right
    os.environ["VLLM_MOE_W2_BASE_RAM_GB"] = str(24 * ((SLOT + 4095) // 4096 *
                                                      4096) / 2**30)
    tier4 = make_tier("base")
    assert tier4._store.n_arena == 24, tier4._store.n_arena
    for li in range(N_LAYERS):
        tier4.add_layer_host_planes(li, *garbage[li])
    for sweep in range(2):
        for li in range(N_LAYERS):
            for e0 in range(0, E, 8):
                pairs = [(li, ei) for ei in range(e0, e0 + 8)]
                check_store_rows(tier4._store, parts, pairs,
                                 f"evict s{sweep}")
    st = tier4._store.stats()
    assert st["miss_rows"] > 64, st     # re-reads prove evictions happened
    print(f"[tiered evict] arena 24 slots, {st['miss_rows']} nvme rows "
          f"(evict-driven re-reads), {st['hit_rows']} hits  OK")

    # overflow: ONE batch of 64 rows > 24-slot arena -> stage fallback
    check_store_rows(tier4._store, parts, all_pairs, "tiered overflow")
    print("[tiered overflow] 64-row batch over a 24-slot arena  OK")

    # ---- scan resistance: scan batches fill FREE slots but never evict ----
    os.environ["VLLM_MOE_W2_BASE_RAM_GB"] = str(24 * ((SLOT + 4095) // 4096 *
                                                      4096) / 2**30)
    tier5 = make_tier("base")
    st5 = tier5._store
    assert st5.n_arena == 24 and st5.scan_enabled
    for li in range(N_LAYERS):
        tier5.add_layer_host_planes(li, *garbage[li])
    # scan 1 (16 rows, empty arena): fills free slots only
    scan1 = [(0, ei) for ei in range(16)]
    check_store_rows(st5, parts, scan1, "scan fills free", scan=True)
    assert st5.stats()["arena_used"] == 16, st5.stats()
    # scan 2 (16 NEW rows): 8 into remaining free, 8 via stage, NO eviction
    scan2 = [(1, ei) for ei in range(16)]
    check_store_rows(st5, parts, scan2, "scan no-evict", scan=True)
    assert st5.stats()["arena_used"] == 24, st5.stats()
    assert all((0, ei) in st5._pos for ei in range(16)), "scan1 evicted!"
    # decode batch (4 new rows, scan=False) DOES evict on a full arena
    small = [(2, ei) for ei in range(4)]
    check_store_rows(st5, parts, small, "decode evicts")
    assert st5.stats()["arena_used"] == 24
    assert all(k in st5._pos for k in small), "decode batch not inserted"
    print("[tiered scan-resist] scans fill-only, decode evicts  OK")

    # ---- preheat: dump heat, reopen, rows served from arena ----
    keys = sorted(st5._pos, key=lambda k: st5._last[st5._pos[k]],
                  reverse=True)
    st5._dump_heat([list(k) for k in keys])
    tier6 = make_tier("base")
    st6 = tier6._store
    for li in range(N_LAYERS):
        tier6.add_layer_host_planes(li, *garbage[li])
    assert st6.stats()["arena_used"] == 24, st6.stats()
    hot = [k for k in keys][:8]
    check_store_rows(st6, parts, hot, "preheat bytes")
    st = st6.stats()
    assert st["hit_rows"] == 8 and st["miss_rows"] == 0, st
    print(f"[tiered preheat] {st6.stats()['arena_used']} rows preheated, "
          f"8/8 ram hits  OK")

    os.environ.pop("VLLM_MOE_W2_BASE_RAM_GB", None)
    os.environ.pop("VLLM_MOE_W2_STORE_DIR", None)
    shutil.rmtree(PACK_DIR, ignore_errors=True)
    print("ALL OK")


if __name__ == "__main__":
    main()
