#!/usr/bin/env python3
"""KV offload connector: the copy path off the runner's main thread (`VLLM_MOET_KV_SWAP_ASYNC=1`; session K-10).

Patches vLLM's `v1/kv_offload/cpu/gpu_worker.py` (vLLM 0.32.1rc1.dev117+gdf3bd094a, the r26/r27 images) and installs
`kv_swap_async.py` next to it as `vllm.v1.kv_offload.cpu.kv_swap_async`. Default behaviour unchanged (the switch off):

  1. `pin_mmap_region`: the four TP workers' `cudaHostRegister` of the shared /dev/shm region run under a cross-process
     file lock with retries (today they race and three of four fail with cudaErrorInvalidValue -> pageable region ->
     host-synchronous copies on those ranks; rank 0 = the store writer, 85-220 ms per 8 192-token step on its main
     thread). `VLLM_MOET_KV_SWAP_PIN=0` skips the registration (the robustness case: every rank pageable).
  2. `SingleDirectionOffloadingHandler`: when the region is pageable, stores go device -> pinned staging on the transfer
     stream + a drain thread (memmove staging -> region), loads go region -> pinned staging in a feeder thread which
     issues staging -> device; completion / `wait()` follow the connector's protocol (see kv_swap_async.py).

Idempotent, anchor-based; every old block must match verbatim. Usage:
    python3 patch_vllm_kv_swap_async_sm120.py [--dist /usr/local/lib/python3.12/dist-packages/vllm] [--check]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "# [vllm-moet] K-10 kv_swap_async"
REL = "v1/kv_offload/cpu/gpu_worker.py"
HELPER = "kv_swap_async.py"

EDITS: list[tuple[str, str, str]] = [
    (
        "import",
        "from vllm.v1.kv_offload.cpu.swap_blocks_triton import (\n    THRESHOLD_BYTES,\n    swap_blocks_batch,\n)\n",
        "from vllm.v1.kv_offload.cpu.swap_blocks_triton import (\n    THRESHOLD_BYTES,\n    swap_blocks_batch,\n)\n"
        f"from vllm.v1.kv_offload.cpu import kv_swap_async as _kv_swap_async  {MARKER}\n",
    ),
    (
        "Transfer.async_state",
        "    batch_src: torch.Tensor\n    batch_dst: torch.Tensor\n    batch_sizes: torch.Tensor\n\n\ndef compute_sub_block_ptrs(\n",
        "    batch_src: torch.Tensor\n    batch_dst: torch.Tensor\n    batch_sizes: torch.Tensor\n"
        f"    async_state: \"_kv_swap_async.AsyncState | None\" = None  {MARKER}\n\n\ndef compute_sub_block_ptrs(\n",
    ),
    (
        "pin_mmap_region: the registration chunk",
        "    rows_per_chunk = max(MAX_HOST_REGISTER_CHUNK_BYTES // region._row_stride, 1)\n",
        f"    rows_per_chunk = max(_kv_swap_async.pin_chunk_bytes(MAX_HOST_REGISTER_CHUNK_BYTES) // region._row_stride, 1)  {MARKER}\n",
    ),
    (
        "pin_mmap_region: lock + retries + skip",
        "    addresses: list[int] = []\n"
        "    registered_bytes = 0\n"
        "    for offset, size in ranges:\n"
        "        address = base_ptr + offset\n"
        "        if host_register(address, size):\n"
        "            addresses.append(address)\n"
        "            registered_bytes += size\n"
        "            continue\n"
        "        logger.warning(\n"
        "            \"host_register failed for rank=%d at %.2f of %.2f GB; \"\n"
        "            \"the offload region stays pageable\",\n"
        "            rank,\n"
        "            offset / 1e9,\n"
        "            total_size / 1e9,\n"
        "        )\n"
        "        for registered in reversed(addresses):\n"
        "            host_unregister(registered)\n"
        "        return\n"
        "\n"
        "    region.pinned_addresses.extend(addresses)\n",
        f"    {MARKER}: the four workers register the same pages one after another (file lock) with retries; PIN=0 skips\n"
        "    if not _kv_swap_async.pin_enabled():\n"
        "        logger.warning(\"KV offload: host registration skipped (VLLM_MOET_KV_SWAP_PIN=0); the offload region stays pageable\")\n"
        "        return\n"
        "    addresses: list[int] = []\n"
        "    registered_bytes = 0\n"
        "    with _kv_swap_async.pin_lock(region):\n"
        "        for offset, size in ranges:\n"
        "            address = base_ptr + offset\n"
        "            if _kv_swap_async.register_with_retries(host_register, address, size):\n"
        "                addresses.append(address)\n"
        "                registered_bytes += size\n"
        "                continue\n"
        "            logger.warning(\n"
        "                \"host_register failed for rank=%d at %.2f of %.2f GB; \"\n"
        "                \"the offload region stays pageable\",\n"
        "                rank,\n"
        "                offset / 1e9,\n"
        "                total_size / 1e9,\n"
        "            )\n"
        "            for registered in reversed(addresses):\n"
        "                host_unregister(registered)\n"
        "            return\n"
        "\n"
        "    region.pinned_addresses.extend(addresses)\n",
    ),
    (
        "pin_mmap_region: the success line at INFO (the arms count the pinned ranks)",
        "    region.is_pinned = True\n"
        "    logger.debug(\n"
        "        \"Host-registered mmap region rank=%d %.2f GB in %d chunk(s)\",\n",
        "    region.is_pinned = True\n"
        f"    logger.info(  {MARKER}\n"
        "        \"Host-registered mmap region rank=%d %.2f GB in %d chunk(s)\",\n",
    ),
    (
        "handler init: the async state",
        "        # list of pinned descriptor buffer sets available for re-use\n"
        "        self._buffer_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []\n",
        "        # list of pinned descriptor buffer sets available for re-use\n"
        "        self._buffer_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []\n"
        f"        {MARKER}: asynchronous copies through pinned staging when the shared region is pageable\n"
        "        self._async: \"_kv_swap_async.HandlerState | None\" = None\n"
        "        self._transfer_async: dict[int, \"_kv_swap_async.AsyncState\"] = {}\n"
        "        if _kv_swap_async.enabled() and not host_memory_is_pinned:\n"
        "            self._async = _kv_swap_async.HandlerState(gpu_to_cpu, gpu_tensors[0].device)\n",
    ),
    (
        "transfer_async: the issue",
        "        # Stores must wait for the model to finish writing the KV they read.\n"
        "        # Loads must wait for pending writes (including zeroing) to their\n"
        "        # destination blocks: the scheduler leaves a block unzeroed only in\n"
        "        # the step that allocates it for the load and cannot retract zeroing\n"
        "        # shipped in an earlier step for a since-reallocated block; with async\n"
        "        # scheduling nothing else orders that zeroing against this copy.\n"
        "        stream.wait_stream(current_platform.current_stream())\n"
        "        if self._transfers:\n"
        "            last_transfer: Transfer = self._transfers[-1]\n"
        "            last_event = last_transfer.end_event\n"
        "            # assure job will start only after the previous one completes\n"
        "            stream.wait_event(last_event)\n"
        "        # CPU->GPU reads from host memory, which is never written\n"
        "        # by a concurrent GPU stream, so CU_MEMCPY_SRC_ACCESS_ORDER_ANY is\n"
        "        # safe and lets the driver pipeline source reads. GPU->CPU reads\n"
        "        # from the live GPU KV cache, which the compute stream keeps\n"
        "        # writing; we must keep STREAM ordering so source reads are gated\n"
        "        # by the transfer stream's wait_stream(compute) barrier.\n"
        "        is_src_access_order_any = not self.gpu_to_cpu\n"
        "        with current_platform.stream(stream):\n"
        "            start_event.record(stream)\n"
        "            if op_idx > 0:\n"
        "                self._swap_blocks_batch(\n"
        "                    src,\n"
        "                    dst,\n"
        "                    sizes,\n"
        "                    is_src_access_order_any=is_src_access_order_any,\n"
        "                )\n"
        "            end_event.record(stream)\n"
        "\n"
        "        self._transfer_events[job_id] = end_event\n"
        "        self._transfers.append(\n"
        "            Transfer(\n"
        "                job_id=job_id,\n"
        "                stream=stream,\n"
        "                start_event=start_event,\n"
        "                end_event=end_event,\n"
        "                num_bytes=num_transfer_bytes,\n"
        "                batch_src=batch_src,\n"
        "                batch_dst=batch_dst,\n"
        "                batch_sizes=batch_sizes,\n"
        "            )\n"
        "        )\n",
        f"        {MARKER}: the pageable region's asynchronous path (pinned staging + the direction's thread); else vLLM's issue\n"
        "        async_state = None\n"
        "        if self._async is not None and op_idx > 0:\n"
        "            async_state = _kv_swap_async.AsyncState(is_store=self.gpu_to_cpu)\n"
        "            self._async.submit(\n"
        "                async_state, stream, start_event, end_event, self._swap_blocks_batch, src, dst, sizes,\n"
        "                current_platform.current_stream(), self._transfers[-1].end_event if self._transfers else None,\n"
        "            )\n"
        "            self._transfer_async[job_id] = async_state\n"
        "        else:\n"
        "            # Stores must wait for the model to finish writing the KV they read.\n"
        "            # Loads must wait for pending writes (including zeroing) to their\n"
        "            # destination blocks: the scheduler leaves a block unzeroed only in\n"
        "            # the step that allocates it for the load and cannot retract zeroing\n"
        "            # shipped in an earlier step for a since-reallocated block; with async\n"
        "            # scheduling nothing else orders that zeroing against this copy.\n"
        "            stream.wait_stream(current_platform.current_stream())\n"
        "            if self._transfers:\n"
        "                last_transfer: Transfer = self._transfers[-1]\n"
        "                last_event = last_transfer.end_event\n"
        "                # assure job will start only after the previous one completes\n"
        "                stream.wait_event(last_event)\n"
        "            # CPU->GPU reads from host memory, which is never written\n"
        "            # by a concurrent GPU stream, so CU_MEMCPY_SRC_ACCESS_ORDER_ANY is\n"
        "            # safe and lets the driver pipeline source reads. GPU->CPU reads\n"
        "            # from the live GPU KV cache, which the compute stream keeps\n"
        "            # writing; we must keep STREAM ordering so source reads are gated\n"
        "            # by the transfer stream's wait_stream(compute) barrier.\n"
        "            is_src_access_order_any = not self.gpu_to_cpu\n"
        "            with current_platform.stream(stream):\n"
        "                start_event.record(stream)\n"
        "                if op_idx > 0:\n"
        "                    self._swap_blocks_batch(\n"
        "                        src,\n"
        "                        dst,\n"
        "                        sizes,\n"
        "                        is_src_access_order_any=is_src_access_order_any,\n"
        "                    )\n"
        "                end_event.record(stream)\n"
        "\n"
        "        self._transfer_events[job_id] = end_event\n"
        "        self._transfers.append(\n"
        "            Transfer(\n"
        "                job_id=job_id,\n"
        "                stream=stream,\n"
        "                start_event=start_event,\n"
        "                end_event=end_event,\n"
        "                num_bytes=num_transfer_bytes,\n"
        "                batch_src=batch_src,\n"
        "                batch_dst=batch_dst,\n"
        "                batch_sizes=batch_sizes,\n"
        "                async_state=async_state,\n"
        "            )\n"
        "        )\n",
    ),
    (
        "get_finished: completion by the async state",
        "        results: list[TransferResult] = []\n"
        "        while self._transfers and self._transfers[0].end_event.query():\n"
        "            transfer = self._transfers.popleft()\n"
        "            transfer_time = (\n"
        "                transfer.start_event.elapsed_time(transfer.end_event) * 1e-3\n"
        "            )  # elapsed_time is in milliseconds\n"
        "            result = TransferResult(\n"
        "                job_id=transfer.job_id,\n"
        "                success=True,\n"
        "                transfer_size=transfer.num_bytes,\n"
        "                transfer_time=transfer_time,\n"
        "            )\n"
        "\n"
        "            results.append(result)\n",
        "        results: list[TransferResult] = []\n"
        f"        {MARKER}: an asynchronous transfer is finished when its thread says so (store: the memmove into the region done)\n"
        "        while self._transfers and (\n"
        "            self._transfers[0].async_state.is_done()\n"
        "            if self._transfers[0].async_state is not None\n"
        "            else self._transfers[0].end_event.query()\n"
        "        ):\n"
        "            transfer = self._transfers.popleft()\n"
        "            failed = transfer.async_state is not None and transfer.async_state.error is not None\n"
        "            transfer_time = (\n"
        "                None if failed else transfer.start_event.elapsed_time(transfer.end_event) * 1e-3\n"
        "            )  # elapsed_time is in milliseconds\n"
        "            result = TransferResult(\n"
        "                job_id=transfer.job_id,\n"
        "                success=not failed,\n"
        "                transfer_size=transfer.num_bytes,\n"
        "                transfer_time=transfer_time,\n"
        "            )\n"
        "            self._transfer_async.pop(transfer.job_id, None)\n"
        "            if self._async is not None and self._async.transfers % 50 == 0 and transfer.async_state is not None:\n"
        "                logger.info(\"KV offload async stats: %s\", self._async.stats())\n"
        "\n"
        "            results.append(result)\n",
    ),
    (
        "wait: by the async state",
        "    def wait(self, job_ids: set[int]):\n"
        "        for job_id in job_ids:\n"
        "            event = self._transfer_events.get(job_id)\n",
        "    def wait(self, job_ids: set[int]):\n"
        "        for job_id in job_ids:\n"
        f"            state = self._transfer_async.get(job_id)  {MARKER}: the GPU copy and the host memmove\n"
        "            if state is not None:\n"
        "                state.wait()\n"
        "                continue\n"
        "            event = self._transfer_events.get(job_id)\n",
    ),
    (
        "shutdown: drain the thread",
        "        while self._transfers:\n"
        "            transfer = self._transfers[0]\n"
        "            try:\n"
        "                transfer.end_event.synchronize()\n",
        "        while self._transfers:\n"
        "            transfer = self._transfers[0]\n"
        "            try:\n"
        f"                if transfer.async_state is not None:  {MARKER}\n"
        "                    transfer.async_state.wait()\n"
        "                else:\n"
        "                    transfer.end_event.synchronize()\n",
    ),
    (
        "shutdown: stop the thread",
        "        self._transfer_events.clear()\n"
        "        self._stream_pool.clear()\n",
        "        self._transfer_events.clear()\n"
        f"        if self._async is not None:  {MARKER}\n"
        "            logger.info(\"KV offload async stats at shutdown: %s\", self._async.stats())\n"
        "            self._async.shutdown()\n"
        "            self._async = None\n"
        "        self._transfer_async.clear()\n"
        "        self._stream_pool.clear()\n",
    ),
]


def apply(text: str) -> tuple[str, list[str]]:
    done: list[str] = []
    for name, old, new in EDITS:
        if new in text:
            done.append(f"{name}: already applied")
            continue
        if text.count(old) != 1:
            raise SystemExit(f"anchor for '{name}' found {text.count(old)} times (expected 1) - wrong vLLM version?")
        text = text.replace(old, new, 1)
        done.append(f"{name}: applied")
    return text, done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="/usr/local/lib/python3.12/dist-packages/vllm", help="the vllm package directory")
    ap.add_argument("--check", action="store_true", help="only report whether the patch is applied")
    a = ap.parse_args()
    target = Path(a.dist) / REL
    text = target.read_text()
    helper_dst = target.parent / HELPER
    if a.check:
        n = sum(1 for _, _, new in EDITS if new in text)
        print(f"{target}: {n}/{len(EDITS)} edits present; helper {'present' if helper_dst.exists() else 'missing'}")
        return 0 if n == len(EDITS) and helper_dst.exists() else 1
    new_text, done = apply(text)
    if new_text != text:
        target.write_text(new_text)
    shutil.copyfile(HERE / HELPER, helper_dst)
    for d in done:
        print(d)
    print(f"helper installed: {helper_dst}")
    compile(new_text, str(target), "exec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
