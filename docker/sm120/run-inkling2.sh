#!/usr/bin/env bash
# Inkling-2-Small (thinkingmachines) on 4x RTX PRO 6000 (sm_120), TP4 with the experts expert-parallel over the TP
# group, the DSpark draft of the checkpoint, the `inkling` reasoning and tool-call parsers.
# Image: Dockerfile.sm120-inkling2 (docs/inkling2-sm120.md).
#
# Required:  MODEL_DIR   the checkpoint directory (thinkingmachines/Inkling-2-Small: NVFP4 experts in layers 0-39,
#                        MXFP8 experts in layers 40-47, everything else BF16, mtp.safetensors = the DSpark draft; 394 GB)
#            GPUS        the four cards of this server, indices or UUIDs (e.g. 4,5,6,7). There is no default: a load
#                        puts 87 GiB on each card, so the cards are named by whoever knows that they are free. The
#                        launcher refuses a card that holds a compute process or more than GPU_BUSY_MIB (1024) MiB
#                        (an `nvidia-smi` query; GPU_CHECK=0 skips it)
#
# Fit:       MXFP6 (1)   1 = the MXFP8 experts (layers 40-47) are packed to e2m3, six bits per value, as they load
#                        (step 1's VLLM_MOET_INKLING2_MXFP6): 5.7 GiB less per card at TP4 - a lossy change of those
#                        weights, with its measurements in docs/inkling2-sm120.md. 0 = the checkpoint's bytes:
#                        91.6 GiB of weights per card, which left 0.69 GiB of paged cache in the one start measured
#                        (SPEC_TOKENS=0 MAX_MODEL_LEN=8192 MAX_NUM_BATCHED_TOKENS=64 MAX_NUM_SEQS=8); the defaults
#                        below need more than that (the draft alone takes 2.3 GiB of a card)
#
# Drafter:   SPEC_TOKENS (7)   DSpark draft tokens per step (the draft is in the checkpoint and drafts blocks of 8:
#                              the only other value is 0 = no speculative decoding)
#            SPEC_SCHEDULE ("")   vLLM's draft length by batch size, a JSON list of [batch_from, batch_to, tokens];
#                              "" = the draft at every batch size - the default: at 16 and 32 requests code runs
#                              2.7-2.8x and 2.5x faster with it, prose within 3 % and 6-10 % slower.
#                              '[[1,8,7],[9,64,0]]' drafts nothing above 8 requests (free prose at high
#                              concurrency); with a schedule the draft's CUDA graphs are captured only for the
#                              batch sizes that draft (step 9)
#
# All-reduce: P2P_AR (0)       1 = the TP group's all-reduces of up to P2P_AR_ROWS (128) rows on the PCIe P2P push with
#                              NCCL's summation order (step 7: VLLM_MOET_P2P_AR) - measured slower than PyNCCL on this
#                              model (one row is 9 KB: the all-reduce is latency-bound); 0 = PyNCCL
#
# Container: IMAGE (vllm-moet-sm120:inkling2-b974b306-r13)  NAME (inkling2-small)
#            TP (4)  PORT (8002)  BIND (127.0.0.1:)  "" publishes on all interfaces - put an authenticating proxy in front
#            CPUS ("")  --cpuset-cpus, CPUSET_MEMS ("")  --cpuset-mems: the cores and the NUMA node of GPUS. REQUIRED
#                        ON A HOST THAT RUNS ANYTHING ELSE: without CPUS the server's processes (four workers, the
#                        engine, the API server) run on every core of the host. The launcher says so when it is empty
#            MEMORY (96g)  --memory = --memory-swap, the cap the measured servers ran under (they held about 23 GiB
#                        of host memory at the end of a load and at most 36 GiB while serving); "" = no cap
#            IPC (private)  an own IPC namespace with SHM_SIZE (16g) of /dev/shm - several servers on one host do not
#                        share /dev/shm; host = the host's
#            CACHE_DIR ($PWD/cache-$NAME)  Triton / CuTe DSL / FlashInfer caches; the first start fills it (Triton's
#                        cache is pointed into it: the model's Triton kernels compile per batch shape at first use,
#                        seconds each, and vLLM's warm-up does not cover every shape)
#            SERVED_NAME (thinkingmachines/Inkling-2-Small)  the model name the API answers to; SERVED_ALIASES ("")
#            NCCL_P2P_LEVEL (SYS)    P2P over PCIe also in KVM guests NCCL classifies as PHB
#            EXTRA_ARGS / EXTRA_DOCKER_ARGS   appended to `vllm serve` / `docker run`
#
# Engine:    MAX_MODEL_LEN (65536)   the checkpoint allows 1,048,576; what fits is set by the paged cache a card has
#                        left (4.5 GiB without the draft, 3.3 GiB with it): one request of MAX_MODEL_LEN must fit beside the conv
#                        state of two prefill chunks in flight (3 MiB per token of MAX_NUM_BATCHED_TOKENS), or the
#                        server refuses to start and names the length that would fit
#            MAX_NUM_SEQS (32)       MAX_NUM_BATCHED_TOKENS ("" = 512 with the draft, 1024 without: the prefill chunk;
#                        the prefill reads 8.7K tok/s at 512 and 8.8K at 1024)
#            GPU_MEM_UTIL ("" = 0.99 with the draft, 0.98 without)
#            PREFIX_CACHING (1)      0 = --no-enable-prefix-caching
#
# Thinking is set per request: "chat_template_kwargs": {"reasoning_effort": "none"} turns it off (default 0.9).
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR (Inkling-2-Small checkpoint directory) is required}"
IMAGE="${IMAGE:-vllm-moet-sm120:inkling2-b974b306-r13}"
NAME="${NAME:-inkling2-small}"
: "${GPUS:?GPUS (the four cards of this server, e.g. GPUS=4,5,6,7) is required: there is no default}"
GPU_CHECK="${GPU_CHECK:-1}"
GPU_BUSY_MIB="${GPU_BUSY_MIB:-1024}"
TP="${TP:-4}"
PORT="${PORT:-8002}"
BIND="${BIND-127.0.0.1:}"
CPUS="${CPUS:-}"
CPUSET_MEMS="${CPUSET_MEMS:-}"
MEMORY="${MEMORY-96g}"
IPC="${IPC:-private}"
SHM_SIZE="${SHM_SIZE:-16g}"
CACHE_DIR="${CACHE_DIR:-$PWD/cache-$NAME}"
SERVED_NAME="${SERVED_NAME:-thinkingmachines/Inkling-2-Small}"
SERVED_ALIASES="${SERVED_ALIASES-}"
MXFP6="${MXFP6:-1}"
SPEC_TOKENS="${SPEC_TOKENS:-7}"
SPEC_SCHEDULE="${SPEC_SCHEDULE-}"
P2P_AR="${P2P_AR:-0}"
P2P_AR_ROWS="${P2P_AR_ROWS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
if [[ "$SPEC_TOKENS" == "0" ]]; then
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"; GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.98}"
else
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"; GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.99}"
fi
PREFIX_CACHING="${PREFIX_CACHING:-1}"
NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"

[[ "$GPUS" =~ ^[A-Za-z0-9-]+(,[A-Za-z0-9-]+)*$ ]] || { echo "GPUS must be a comma-separated list of card indices or UUIDs, got $GPUS" >&2; exit 1; }
[[ -f "$MODEL_DIR/config.json" ]] || { echo "no config.json in MODEL_DIR=$MODEL_DIR" >&2; exit 1; }
for v in SPEC_TOKENS MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_MODEL_LEN TP P2P_AR_ROWS GPU_BUSY_MIB; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "$v must be a non-negative integer, got ${!v}" >&2; exit 1; }
done
case "$PREFIX_CACHING$MXFP6$P2P_AR$GPU_CHECK" in
  [01][01][01][01]) ;;
  *) echo "PREFIX_CACHING, MXFP6, P2P_AR and GPU_CHECK must be 0 or 1" >&2; exit 1 ;;
esac
case "$SPEC_TOKENS" in 0 | 7) ;; *) echo "SPEC_TOKENS must be 7 (the draft's block of 8) or 0" >&2; exit 1 ;; esac
case "$IPC" in host | private) ;; *) echo "IPC must be host or private" >&2; exit 1 ;; esac
if [[ "$SPEC_TOKENS" != "0" && ! -f "$MODEL_DIR/mtp.safetensors" ]]; then
  echo "SPEC_TOKENS=$SPEC_TOKENS but MODEL_DIR has no mtp.safetensors (the DSpark draft); set SPEC_TOKENS=0" >&2; exit 1
fi
if [[ "$GPU_CHECK" == "1" ]]; then
  # queries only: the cards' memory in use and the cards that hold a compute process
  command -v nvidia-smi > /dev/null || { echo "nvidia-smi not found: cannot tell whether GPUS=$GPUS are free (GPU_CHECK=0 skips the check)" >&2; exit 1; }
  cards=$(nvidia-smi -i "$GPUS" --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits) \
    || { echo "nvidia-smi does not know GPUS=$GPUS" >&2; exit 1; }
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader) || { echo "nvidia-smi: the process query failed" >&2; exit 1; }
  IFS=, read -r -a named <<< "$GPUS"
  [[ $(grep -c . <<< "$cards") == "${#named[@]}" ]] || { echo "nvidia-smi does not know every card of GPUS=$GPUS" >&2; exit 1; }
  busy=0
  while IFS=', ' read -r index uuid used; do
    [[ -n "$index" ]] || continue
    pids=$(grep -F "$uuid" <<< "$apps" | cut -d, -f2 | tr -d ' ' | paste -sd' ' -) || true
    if [[ -n "$pids" || "$used" -gt "$GPU_BUSY_MIB" ]]; then
      echo "card $index ($uuid) is in use: $used MiB${pids:+, compute process(es) $pids}" >&2; busy=1
    fi
  done <<< "$cards"
  [[ "$busy" == "0" ]] || { echo "GPUS=$GPUS are not free; nothing started (GPU_CHECK=0 skips the check)" >&2; exit 1; }
fi
if [[ -z "$CPUS" ]]; then
  echo "note: CPUS is not set - the server's processes may run on every core of this host; on a host that runs anything else set CPUS and CPUSET_MEMS to the cores and the NUMA node of GPUS=$GPUS" >&2
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "container $NAME exists; remove it first (docker stop -t 120 $NAME && docker rm $NAME)" >&2; exit 1
fi
mkdir -p "$CACHE_DIR/dot-cache"

ARGS=(--model /model --served-model-name "$SERVED_NAME")
for alias in $SERVED_ALIASES; do ARGS+=("$alias"); done
ARGS+=(--tensor-parallel-size "$TP"
       --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
       --gpu-memory-utilization "$GPU_MEM_UTIL"
       --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice
       --enable-prompt-tokens-details)
[[ "$PREFIX_CACHING" == "0" ]] && ARGS+=(--no-enable-prefix-caching)
if [[ "$SPEC_TOKENS" != "0" ]]; then
  SPEC_EXTRA=""
  [[ -n "$SPEC_SCHEDULE" ]] && SPEC_EXTRA=",\"num_speculative_tokens_per_batch_size\":${SPEC_SCHEDULE}"
  ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\"${SPEC_EXTRA}}")
fi
DOCKER_OPTS=()
if [[ "$P2P_AR" == "1" ]]; then
  DOCKER_OPTS+=(-e VLLM_MOET_P2P_AR=1 -e VLLM_MOET_P2P_AR_M_MAX="$P2P_AR_ROWS" -e VLLM_MOET_P2P_AR_MAX_ROWS="$P2P_AR_ROWS")
fi
[[ -n "$CPUS" ]] && DOCKER_OPTS+=(--cpuset-cpus "$CPUS")
[[ -n "$CPUSET_MEMS" ]] && DOCKER_OPTS+=(--cpuset-mems "$CPUSET_MEMS")
[[ -n "$MEMORY" ]] && DOCKER_OPTS+=(--memory "$MEMORY" --memory-swap "$MEMORY")
if [[ "$IPC" == "host" ]]; then DOCKER_OPTS+=(--ipc host); else DOCKER_OPTS+=(--shm-size "$SHM_SIZE"); fi

docker run -d --name "$NAME" --restart no \
  --gpus "\"device=${GPUS}\"" \
  "${DOCKER_OPTS[@]}" \
  -p "${BIND}${PORT}:${PORT}" \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  -v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR/dot-cache:/root/.cache" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e VLLM_MOET_INKLING2_MXFP6="$MXFP6" \
  ${EXTRA_DOCKER_ARGS} \
  "$IMAGE" \
  "${ARGS[@]}" \
  ${EXTRA_ARGS} \
  --host 0.0.0.0 --port "$PORT"
echo "container $NAME started; logs: docker logs -f $NAME ; API: http://127.0.0.1:${PORT}/v1"
