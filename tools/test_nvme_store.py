"""Standalone test of the hybrid RAM/NVMe base store (moe_w2_nvme).

Fakes a minimal tier (slot_bytes, E, dev), stages 4 layers of random rows
through build_layer, then reads every (layer, expert) back via copy_into on
a side stream and compares bit-exactly against the source. Exercises both
the pinned-RAM rows and the O_DIRECT NVMe rows, plus prefetch and the
staging-ring generation check (staging ring smaller than one layer's NVMe
row count).
"""
import os
import sys

os.environ["VLLM_MOE_W2_BASE_NVME_RATIO"] = "1:2"
os.environ["VLLM_MOE_W2_BASE_NVME_DIR"] = "/root/models/base-store-test"
os.environ["VLLM_MOE_W2_BASE_NVME_STAGING"] = "24"   # force ring wrap
os.environ["VLLM_MOE_W2_BASE_NVME_THREADS"] = "4"


import torch  # noqa: E402
from vllm.model_executor.layers.quantization.utils import moe_w2_nvme  # noqa: E402


class FakeTier:
    slot_bytes = 4096 * 3          # 12 KiB, 4096-aligned
    E = 64
    dev = torch.device("cuda", 0)


def main():
    torch.cuda.set_device(0)
    tier = FakeTier()
    n_layers = 4
    stream = torch.cuda.Stream(tier.dev)

    assert moe_w2_nvme.enabled()
    mask = moe_w2_nvme.ram_mask(tier.E)
    n_ram = sum(mask)
    print(f"ram_mask: {n_ram}/{tier.E} in RAM "
          f"(expected ~{tier.E // 3}), first 12: {mask[:12]}")
    assert abs(n_ram - tier.E / 3) <= 1

    src = {}
    stores = {}
    for li in range(n_layers):
        p13 = torch.randint(0, 256, (tier.E, tier.slot_bytes * 2 // 3),
                            dtype=torch.uint8, device=tier.dev)
        p2 = torch.randint(0, 256, (tier.E, tier.slot_bytes // 3),
                           dtype=torch.uint8, device=tier.dev)
        src[li] = torch.cat((p13, p2), dim=1).cpu()
        stores[li] = moe_w2_nvme.build_layer(tier, li, (p13,), (p2,))

    dst = torch.empty(tier.slot_bytes, dtype=torch.uint8, device=tier.dev)

    # 1) individual copy_into, all experts (RAM + NVMe direct path)
    bad = 0
    for li in range(n_layers):
        for ei in range(tier.E):
            stores[li].copy_into(ei, dst, stream)
            stream.synchronize()
            if not torch.equal(dst.cpu(), src[li][ei]):
                bad += 1
    print(f"direct copy_into: {n_layers * tier.E} rows, mismatches: {bad}")
    assert bad == 0

    # 2) prefetch batch larger than the staging ring, then consume
    store = stores[2]
    eis = list(range(tier.E))
    store.prefetch(eis)
    bad = 0
    for ei in eis:
        store.copy_into(ei, dst, stream)
        stream.synchronize()
        if not torch.equal(dst.cpu(), src[2][ei]):
            bad += 1
    print(f"prefetch+consume (ring wrap): {len(eis)} rows, mismatches: {bad}")
    assert bad == 0

    # 3) interleaved prefetch of two layers (gen check across layers)
    stores[0].prefetch(list(range(0, tier.E, 2)))
    stores[1].prefetch(list(range(1, tier.E, 2)))
    bad = 0
    for ei in range(0, tier.E, 2):
        stores[0].copy_into(ei, dst, stream)
        stream.synchronize()
        bad += 0 if torch.equal(dst.cpu(), src[0][ei]) else 1
    for ei in range(1, tier.E, 2):
        stores[1].copy_into(ei, dst, stream)
        stream.synchronize()
        bad += 0 if torch.equal(dst.cpu(), src[1][ei]) else 1
    print(f"interleaved 2-layer prefetch: mismatches: {bad}")
    assert bad == 0

    st = moe_w2_nvme._BACKEND.stats()
    print(f"backend stats: {st['reads']} NVMe reads, {st['gib']:.3f} GiB, "
          f"{st['gib'] / max(st['sec'], 1e-9):.2f} GiB/s aggregate")
    print("ALL OK")


if __name__ == "__main__":
    main()
