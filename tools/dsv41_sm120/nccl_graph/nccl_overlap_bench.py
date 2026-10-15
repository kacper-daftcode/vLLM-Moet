#!/usr/bin/env python3
"""Does a TP all-reduce overlap with the other micro-batch's compute on the 4 PCIe GPUs of the serving host?

The premise of dual-batch overlap (DBO) for the DeepSeek-V4.1 prefill: a 4096-token chunk's two all-reduces per layer
([4096, 5120] bf16, 42 MB, 1.6 ms each on the PCIe ring) are a third of the prefill, and the ring kernel occupies a handful
of SMs, so a chunk split into two micro-batches could run one half's all-reduce ([2048, 5120], 21 MB) behind the other
half's GEMMs. This bench replays that GPU timeline without the model - vLLM's own PyNccl communicator on an explicit comm
stream, events between the streams exactly as `vllm/v1/worker/ubatching.py` records them, cuBLAS / cuBLASLt GEMMs plus an
elementwise pass as the compute of one half-chunk between two comm points - and reports, per schedule, the time of
L layers x 2 phases (attention boundary, FFN boundary):

  serial    one batch: compute(2H) then all-reduce([2H, 5120]) on the same stream        (today's prefill)
  dbo       two halves: compute(h) on the compute stream, all-reduce(h) on the comm stream after it, the next phase's
            compute(h) after that all-reduce - the dependency graph the two ubatch threads produce
  compute   the compute alone (both halves / the full batch), comm the all-reduces alone (halves / full)

Overlap efficiency = (compute_halves + comm_halves - dbo) / min(compute_halves, comm_halves). Each configuration is one
process group (NCCL reads its environment at init), e.g. inside the serving image on GPUs 0-3 next to the serving job:
    NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node=4 nccl_overlap_bench.py [--gemm-dtype fp8]
        [--half-tokens 2048] [--gemm-n 6144] [--gemms 3] [--elementwise 2] [--layers 10] [--comm-priority -1]
Rank 0 prints the medians over --iters repetitions (per rank the four agree within a few percent). The allocator is capped
at 3 % of the GPU."""
import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--half-tokens", type=int, default=2048, help="tokens per micro-batch (H); the chunk is 2H")
parser.add_argument("--hidden", type=int, default=5120)
parser.add_argument("--gemm-n", type=int, default=6144, help="N of each [tokens, hidden] x [hidden, N] GEMM")
parser.add_argument("--gemms", type=int, default=3, help="GEMMs per half-chunk between two comm points")
parser.add_argument(
    "--gemm-dtype",
    choices=["bf16", "fp8", "mxfp8", "triton-persistent"],
    default="bf16",
    help="bf16 cuBLAS, fp8 cuBLASLt (_scaled_mm), mxfp8 = the served FlashInfer CUTLASS SM120 GEMM (persistent "
    "scheduler), triton-persistent = a Triton matmul with one CTA per SM looping over tiles",
)
parser.add_argument("--elementwise", type=int, default=2, help="read+write passes over [tokens, 4, hidden] bf16 per phase")
parser.add_argument("--layers", type=int, default=10, help="layers per repetition; 2 phases (all-reduces) per layer")
parser.add_argument("--iters", type=int, default=10)
parser.add_argument("--comm-priority", type=int, default=0, help="comm stream priority (-1 = high, 0 = default)")
parser.add_argument("--compute-priority", type=int, default=0)
parser.add_argument("--no-elementwise-overlap", action="store_true", help="elementwise only in the full-batch schedules")
args = parser.parse_args()

dist.init_process_group("gloo")
rank, ws = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
torch.cuda.set_per_process_memory_fraction(0.03)
device = torch.device("cuda", rank)

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator  # noqa: E402

comm = PyNcclCommunicator(group=dist.group.WORLD, device=device)
assert comm.available and not comm.disabled

H, D, N = args.half_tokens, args.hidden, args.gemm_n
compute_stream = torch.cuda.Stream(device=device, priority=args.compute_priority)
comm_stream = torch.cuda.Stream(device=device, priority=args.comm_priority)

# Inputs: one [2H, D] activation whose halves are the micro-batches; weights shared by every GEMM call.
x_full = torch.randn(2 * H, D, device=device, dtype=torch.bfloat16)
ar_out_full = torch.empty_like(x_full)
res_full = torch.randn(2 * H, 4, D, device=device, dtype=torch.bfloat16)
if args.gemm_dtype == "bf16":
    weights = [torch.randn(D, N, device=device, dtype=torch.bfloat16) for _ in range(args.gemms)]
    x_in = x_full
elif args.gemm_dtype == "fp8":
    fp8 = torch.float8_e4m3fn
    weights = [torch.randn(N, D, device=device).to(fp8).t() for _ in range(args.gemms)]  # column-major [D, N]
    x_in = x_full.to(fp8)
    scale_a = torch.ones((), device=device)
    scale_b = torch.ones((), device=device)
elif args.gemm_dtype == "mxfp8":
    # The served dense GEMM: FlashInfer's CUTLASS SM120 MXFP8 kernel on swizzled per-32 scales (as
    # model_executor/kernels/linear/mxfp8/flashinfer.py calls it above the GEMV range).
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.utils import flashinfer as vllm_flashinfer

    weights = []
    for _ in range(args.gemms):
        w_q, w_sf = mxfp8_e4m3_quantize(
            torch.randn(N, D, device=device, dtype=torch.bfloat16), is_sf_swizzled_layout=True
        )
        weights.append((w_q, w_sf))
    # One quantized activation per half and for the whole chunk: the swizzled scale layout is per row block, so
    # a half's rows are quantized on their own.
    x_q = {}
    for name, rows in (("full", slice(0, 2 * H)), ("h0", slice(0, H)), ("h1", slice(H, 2 * H))):
        x_q[name] = mxfp8_e4m3_quantize(x_full[rows], is_sf_swizzled_layout=True)
else:
    import triton
    import triton.language as tl

    @triton.jit
    def _persistent_matmul(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                           stride_cn, NUM_SMS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr):
        # One CTA per SM, each looping over its share of the output tiles (the shape of DeepGEMM's and
        # CUTLASS's persistent schedulers).
        pid = tl.program_id(0)
        num_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
        num_n = tl.cdiv(N, BLOCK_N)
        for tile in range(pid, num_tiles, NUM_SMS):
            pid_m = tile // num_n
            pid_n = tile % num_n
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, K, BLOCK_K):
                a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
                b = tl.load(b_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)
                acc = tl.dot(a, b, acc)
            tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc.to(tl.bfloat16))

    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    weights = [torch.randn(D, N, device=device, dtype=torch.bfloat16) for _ in range(args.gemms)]
    x_in = x_full
gemm_out = torch.empty(2 * H, N, device=device, dtype=torch.bfloat16)


def compute(rows: slice, elementwise: bool = True) -> None:
    """The compute of one (half-)batch between two comm points: GEMMs + elementwise passes over the residual stream."""
    out = gemm_out[rows]
    for w in weights:
        if args.gemm_dtype == "bf16":
            torch.matmul(x_in[rows], w, out=out)
        elif args.gemm_dtype == "fp8":
            torch._scaled_mm(x_in[rows], w, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16, out=out)
        elif args.gemm_dtype == "mxfp8":
            a_q, a_sf = x_q["full" if rows == FULL else ("h0" if rows.start == 0 else "h1")]
            vllm_flashinfer.mm_mxfp8(a_q, w[0].t(), a_sf, w[1], out_dtype=torch.bfloat16, backend="cutlass")
        else:
            a = x_in[rows]
            m = a.shape[0]
            _persistent_matmul[(num_sms,)](
                a, w, out, m, N, D, a.stride(0), a.stride(1), w.stride(0), w.stride(1), out.stride(0),
                out.stride(1), NUM_SMS=num_sms, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8,
            )
    if elementwise:
        r = res_full[rows]
        for _ in range(args.elementwise):
            r.mul_(1.0001).add_(0.5)


def all_reduce(rows: slice, stream: torch.cuda.Stream) -> None:
    comm.all_reduce(x_full[rows], ar_out_full[rows], stream=stream)


FULL = slice(0, 2 * H)
HALVES = [slice(0, H), slice(H, 2 * H)]
phases = 2 * args.layers


def run_serial() -> None:
    with torch.cuda.stream(compute_stream):
        for _ in range(phases):
            compute(FULL)
            all_reduce(FULL, compute_stream)


def run_compute(halves: bool) -> None:
    with torch.cuda.stream(compute_stream):
        for _ in range(phases):
            if halves:
                for h in HALVES:
                    compute(h, elementwise=not args.no_elementwise_overlap)
            else:
                compute(FULL)


def run_comm(halves: bool) -> None:
    with torch.cuda.stream(compute_stream):
        for _ in range(phases):
            if halves:
                for h in HALVES:
                    all_reduce(h, compute_stream)
            else:
                all_reduce(FULL, compute_stream)


def run_dbo() -> None:
    """compute(h) waits for the previous phase's all-reduce(h); all-reduce(h) waits for compute(h) - the events
    UBatchContext records in yield_and_switch_from_compute_to_comm / switch_to_compute_sync."""
    ar_done = [None, None]
    with torch.cuda.stream(compute_stream):
        for _ in range(phases):
            for i, h in enumerate(HALVES):
                if ar_done[i] is not None:
                    compute_stream.wait_event(ar_done[i])
                compute(h, elementwise=not args.no_elementwise_overlap)
                done = torch.cuda.Event()
                done.record(compute_stream)
                comm_stream.wait_event(done)
                all_reduce(h, comm_stream)
                ar_done[i] = torch.cuda.Event()
                ar_done[i].record(comm_stream)
        for ev in ar_done:
            compute_stream.wait_event(ev)


def timed(fn) -> float:
    """Median wall time in ms of one call, every rank synchronized before each repetition."""
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    ts = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    dist.barrier()
    return statistics.median(ts)


results = {
    "compute_full": timed(lambda: run_compute(halves=False)),
    "compute_halves": timed(lambda: run_compute(halves=True)),
    "comm_full": timed(lambda: run_comm(halves=False)),
    "comm_halves": timed(lambda: run_comm(halves=True)),
    "serial": timed(run_serial),
    "dbo": timed(run_dbo),
}
# Every rank's numbers to rank 0.
gathered = [None] * ws
dist.all_gather_object(gathered, results)
if rank == 0:
    cfg = {k: v for k, v in sorted(os.environ.items()) if k.startswith("NCCL_") and k not in ("NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL")}
    tag = " ".join(f"{k.replace('NCCL_', '')}={v}" for k, v in cfg.items()) or "NCCL defaults"
    print(
        f"{tag}; {ws} ranks, H={H} tokens x {D}, {args.gemms} x GEMM[{D}x{N}] {args.gemm_dtype} + {args.elementwise} x "
        f"elementwise[4x{D}] per phase, {args.layers} layers = {phases} phases; comm stream priority {args.comm_priority}"
    )
    per_phase = {k: results[k] / phases * 1e3 for k in results}
    print("per phase (us), rank 0:   " + "  ".join(f"{k}={v:7.1f}" for k, v in per_phase.items()))
    spread = {k: max(r[k] for r in gathered) / min(r[k] for r in gathered) for k in results}
    print("rank spread (max/min):    " + "  ".join(f"{k}={v:5.3f}" for k, v in spread.items()))
    r = results
    hidden = r["compute_halves"] + r["comm_halves"] - r["dbo"]
    eff = hidden / min(r["compute_halves"], r["comm_halves"])
    print(
        f"serial {r['serial']:.2f} ms -> dbo {r['dbo']:.2f} ms ({(r['dbo'] / r['serial'] - 1) * 100:+.1f} %); "
        f"floor max(compute_halves, comm_halves) = {max(r['compute_halves'], r['comm_halves']):.2f} ms; "
        f"overlap efficiency {eff * 100:.0f} % of min(compute, comm); "
        f"halving cost: compute {r['compute_halves'] / r['compute_full'] - 1:+.1%}, comm {r['comm_halves'] / r['comm_full'] - 1:+.1%}"
    )
dist.barrier()
comm.destroy()
dist.destroy_process_group()
