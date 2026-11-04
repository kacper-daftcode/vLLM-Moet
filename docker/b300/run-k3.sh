#!/usr/bin/env bash
# Kimi K3 (moonshotai/Kimi-K3) and Kimi K3.1 (moonshotai/Kimi-K3.1-Pro), MXFP4, on 8x B300 (hgx-02): TP8 + DCP8, FP8 KV,
# prefix caching, KV offload to host RAM and an NVMe tier, tool calling and reasoning. Image: the official
# vllm/vllm-openai release (K3: v0.32.0; K3.1: the day-0 kimi-k31 image until the first release with K3.1), or
# vLLM-Moet's Dockerfile.b300-k3 on top of it (K3 on v0.32.0 needs it for USE_RECOVERSSM=1 together with KV_OFFLOAD);
# the flags follow recipes.vllm.ai/moonshotai/Kimi-K3 and /Kimi-K3.1-Pro (Blackwell TP8 profile with its
# decode-context-parallel option). The model family is read from MODEL_DIR/config.json (model_type kimi_k3 / kimi_k31):
# K3's 24 MLA layers run on TOKENSPEED_MLA; K3.1 replaces them with Gated DSA (sparse MLA, top-2048 indexer), served
# by vLLM's sparse MLA backend (FLASHINFER_MLA_SPARSE with FP8 KV), so the K3 attention flags are not sent for it.
# Conventions follow vLLM-Moet's docker/sm120/run-dsv41.sh: one detached container with --restart no, kept up by
# docker/sm120/ds41-watchdog.sh from cron (WRAPPER_DIR = the directory of the wrapper run.sh that sets the knobs).
# The KV budget behind the knobs below: docs/k3-b300-dspark-kv.md (tools/k3_b300/kv_model.py reproduces it).
#
# Required:  MODEL_DIR   directory with the moonshotai/Kimi-K3 or moonshotai/Kimi-K3.1-Pro checkpoint
# Optional:  IMAGE (vllm/vllm-openai:v0.32.0 for K3, vllm/vllm-openai:kimi-k31 for K3.1)  NAME (vllm-k3)  PORT (8000)
#            BIND (127.0.0.1: only local; "" = all interfaces; several space-separated "ADDR:" prefixes publish the
#                        port on each of them, e.g. "10.100.0.12: 127.0.0.1:" = the LAN plus the loopback the
#                        watchdog probes, nothing on Tailscale)
#            API_KEY_FILE ("")       docker env file with VLLM_API_KEY=...; "" serves without a key
#            API_KEYS_FILE ("")      file with one API key per line ('#' comments): every key is accepted
#                        (vllm serve --api-key k1 k2 k3), for per-client keys (a gateway, a second gateway, a person).
#                        Takes precedence over API_KEY_FILE. The keys land in the container's command line, which only
#                        root on the host can read. vLLM guards /v1, /v2, /inference and /cohere only: /health and
#                        /metrics stay open (the watchdog probes /health), and the Python API server's /invocations
#                        is unauthenticated by design - the network is still the boundary, keys are for telling
#                        clients apart and revoking one of them.
#            CACHE_DIR   host dir with two subdirs mounted at /root/.cache and /root/.deep_gemm (FlashInfer JIT,
#                        torch.compile, vLLM and DeepGEMM caches; later starts reuse them)
#            SERVED_NAME (moonshotai/Kimi-K3 or moonshotai/Kimi-K3.1-Pro, by the model family)  one or more names,
#                        space-separated: the first is the name in responses, the others are aliases (a gateway whose
#                        deployment still asks for moonshotai/Kimi-K3 keeps working while K3.1 is served)
#            MAX_MODEL_LEN (1048576)  MAX_NUM_SEQS (128)  MAX_NUM_BATCHED_TOKENS (8192)
#            PREFIX_MATCH_UNIT (128) --prefix-match-unit: the prefix-cache hash unit; every prefix-cacheable KV group's
#                        block must be a multiple of it (K3 at TP8: 1664-token blocks; K3.1's packed DSA row is sized
#                        in units of 128 at TP8). 64 halves the hit granularity's cost in hashes where 128 does not divide.
#            GPU_MEM_UTIL (0.93)     the recipe uses 0.95
#            DCP (8)                 decode context parallel size. TP8 alone replicates the MLA KV on every rank;
#                        DCP8 shards it, ~8x the tokens per GPU for long agent sessions. 1 = off. On K3.1 DCP
#                        shards the indexer key cache too (the recipe's DCP8 row: ~7x the KV tokens of TP8).
#            ATTENTION_BACKEND       "" = the family's default: TOKENSPEED_MLA for K3 (with the prefill-side
#                        attention config below), vLLM's choice for K3.1 (FLASHINFER_MLA_SPARSE with FP8 KV,
#                        FLASHMLA_SPARSE next). Any other value is passed as --attention-backend as is.
#            ATTENTION_CONFIG        "" = the family's default --attention-config JSON (K3: TRTLLM_RAGGED prefill with
#                        query quantization under DCP, TOKENSPEED_MLA prefill without; K3.1: none unless
#                        INDEXER_KV_DTYPE is set); a JSON object replaces it.
#            KV_CACHE_DTYPE (fp8)    --kv-cache-dtype. K3.1 also takes nvfp4_ds_mla: the MLA latent in NVFP4
#                        (352 instead of 576 bytes per token per layer, ~1.5x the KV tokens; the indexer keys stay
#                        FP8, so the top-2048 selection is unchanged; the recipe measured GSM8K equal to FP8 and
#                        no long-context accuracy yet).
#            INDEXER_KV_DTYPE ("")   K3.1 only: "" = FP8 indexer keys (132 B per token per DSA layer); mxfp4 = 68 B
#                        (SM10x, the fused key insert); goes into the attention config as indexer_kv_dtype.
#            DECODE_BACKEND ("")     latency_specialized = the fused CuTe DSL decode path of the recipes' "low-latency
#                        decode" row (--additional-config decode_backend): decode steps 1.6-1.7x faster at batch 1 and
#                        1.2-1.3x at 32-64 on 8x B300 per the K3.1 recipe; needs the V2 runner, FULL decode CUDA
#                        graphs, TP8 or TP8 + DCP8 and no expert parallelism. "" = vLLM's default decode path.
#            SPEC_TOKENS (0)         DSpark draft tokens (the K3 recipe uses 7, the K3.1 recipe 5); needs DRAFT_DIR
#            DRAFT_DIR ("")          Inferact/Kimi-K3-DSpark (K3) or Inferact/Kimi-K3.1-Pro-DSpark-preview (K3.1: 5 dense
#                        MLA layers, its own FP8 MLA cache next to the target's packed DSA cache) checkpoint
#            DRAFT_ATTENTION_BACKEND (FLASHINFER_MLA)  the Inferact drafts are MLA-native; RedHatAI's speculator
#                        (RedHatAI/Kimi-K3-speculator.dspark: 5 Qwen3-style GQA layers, sliding window 2048, trained
#                        for k=8) is a GQA draft that FLASHINFER_MLA rejects: set "" to let vLLM pick (FLASHINFER).
#                        It needs the Dockerfile.b300-k3 -r5 image (tools/k3_b300/patch_vllm_swa_draft_dcp.py: the
#                        draft next to DCP, and a SlidingWindowManager whose prefix hits land where K3's KDA state
#                        is - the -r2 image lost prefix caching with it); its window-bounded KV leaves ~13% more
#                        pool than the Inferact draft at the same k (24.48M vs 21.61M tokens at k=4, 2026-10-25) at
#                        the cost of 20 KV cache groups.
#            DRAFT_QUANTIZATION ("")  quantization of the draft's weights, e.g. fp8 (online, from the BF16 checkpoint)
#            DRAFT_KV_CACHE_DTYPE ("")  "" = the target's --kv-cache-dtype (fp8). The draft's MLA layers are K3's own
#                        (models/kimi_k3/nvidia/mla.py), so DCP shards the draft KV like the target's. On K3.1 vLLM
#                        defaults an MLA draft to its own plain fp8 cache on FLASHINFER_MLA (no dense kernel reads
#                        the target's packed DSA pages).
#            REJECTION_SAMPLE_METHOD (block for K3, "" for K3.1)  DSpark's rejection sampler: block = the K3 recipe's
#                        and the served K3 value; "" = vLLM's default (standard), which the K3.1 recipe uses.
#            USE_RECOVERSSM (0)      1 = --use-replayssm, which on K3 selects the KDA RecoverSSM verify: the k draft
#                        tokens are verified off one checkpoint state instead of one full KDA state slot per
#                        draft token (MambaSpec.num_speculative_blocks k -> 0). With SPEC_TOKENS=4 that is
#                        12 blocks x 25.7 MB = 308 MB per request per GPU returned to the pool: GPU KV 19.26M ->
#                        21.61M tokens at 1M-token requests, a 32K request 539 -> 250 MB (2026-10-24, hgx-02);
#                        costs ~3% single-stream decode. The official image rejects it next to a KV connector
#                        (KV_OFFLOAD != off): use the Dockerfile.b300-k3 image (tools/k3_b300/
#                        patch_vllm_recoverssm_offload.py lifts the check for K3, whose RecoverSSM commits the
#                        state in-step). Needs SPEC_TOKENS > 0.
#            MAMBA_SSM_CACHE_DTYPE ("")  --mamba-ssm-cache-dtype for the KDA recurrent state: "" = vLLM's default
#                        (float32); bfloat16 halves the KDA page (the InferenceX AgentX lanes serve it), which
#                        halves the attention block (1664 -> 1024 tokens with RecoverSSM) and the per-request KDA
#                        state cost; a quality trade-off, measure before serving it.
#            MAX_CUDAGRAPH_CAPTURE_SIZE ("")  largest CUDA graph batch in tokens; "" = vLLM's default (1024). A decode
#                        step is at most MAX_NUM_SEQS x (1 + SPEC_TOKENS) tokens.
#            KV_OFFLOAD (off)        where KV blocks and KDA states evicted from the GPUs go:
#                        disk   = SimpleCPUOffloadConnector (the connector vLLM's K3 recipe pairs with DCP8) on its disk
#                                 backend: one slab file per rank, KV_OFFLOAD_DIR/k3.rank_N, read through the page
#                                 cache, so free host RAM holds the hot blocks and the NVMe the rest; ~5 min starts.
#                                 Validated 2026-10-23: a 30K-token prompt evicted from the GPUs came back in 0.14 s
#                                 instead of a 3.2 s prefill, greedy output token-identical.
#                        cpu    = the same connector on KV_OFFLOAD_GIB of pinned host RAM; pinning 1 TiB adds ~13 min
#                                 to every start
#                        native = OffloadingConnector (--kv-offloading-size: a /dev/shm region, --ipc host, /etc/fstab
#                                 sizes /dev/shm to 2T), with an fs tier in KV_OFFLOAD_DIR if set. On v0.32.0 with
#                                 DCP8 the fs tier writes only rank 0's attention group and never hits; vLLM never
#                                 deletes its files (vLLM-Moet's kvcache-ttl.sh)
#                        off
#            KV_OFFLOAD_GIB (0)      offload capacity over all 8 ranks
#            KV_OFFLOAD_DIR ("")     disk: directory for the slab files; native: root of the fs tier
#            KV_OFFLOAD_PROMPT_ONLY (0)  native only. 0 = offload generated tokens too: K3 is served in preserved-thinking
#                        mode, the client sends the previous answer, reasoning and tool calls back, so they open the
#                        next prompt and their blocks hit.
#            LOAD_FORMAT (fastsafetensors)  "dummy" = random weights for configuration tests; skips the download
#                        check and keeps the offload files under KV_OFFLOAD_DIR/dummy
#            LANGUAGE_ONLY (0)       1 = --language-model-only (no vision encoder)
#            RUST_FRONTEND (1)       0 = vLLM's Python API server
#            FABRIC_WAIT_MIN (15)    how long to wait for nvidia-fabricmanager to register all GPUs (after a boot)
#            GPU_FREE_WAIT_MIN (10)  how long to wait for the previous server's GPU memory to be released
#            EXTRA_ARGS / EXTRA_DOCKER_ARGS   appended to `vllm serve` / `docker run`
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR (moonshotai/Kimi-K3 or moonshotai/Kimi-K3.1-Pro checkpoint directory) is required}"
[[ -f "$MODEL_DIR/config.json" ]] || { echo "no config.json in MODEL_DIR=$MODEL_DIR" >&2; exit 1; }
# The model family decides the attention flags, the default image and the served name.
MODEL_TYPE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("model_type", ""))' "$MODEL_DIR/config.json")"
case "$MODEL_TYPE" in
  kimi_k3)  FAMILY=k3;  DEFAULT_IMAGE=vllm/vllm-openai:v0.32.0;  DEFAULT_SERVED_NAME=moonshotai/Kimi-K3 ;;
  kimi_k31) FAMILY=k31; DEFAULT_IMAGE=vllm/vllm-openai:kimi-k31; DEFAULT_SERVED_NAME=moonshotai/Kimi-K3.1-Pro ;;
  *) echo "MODEL_DIR=$MODEL_DIR: model_type '$MODEL_TYPE' is not kimi_k3 or kimi_k31" >&2; exit 1 ;;
esac
IMAGE="${IMAGE:-$DEFAULT_IMAGE}"
NAME="${NAME:-vllm-k3}"
PORT="${PORT:-8000}"
BIND="${BIND-127.0.0.1:}"
API_KEY_FILE="${API_KEY_FILE:-}"
API_KEYS_FILE="${API_KEYS_FILE:-}"
CACHE_DIR="${CACHE_DIR:-$PWD/cache-$NAME}"
SERVED_NAME="${SERVED_NAME:-$DEFAULT_SERVED_NAME}"
# shellcheck disable=SC2206
SERVED_NAMES=($SERVED_NAME)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
PREFIX_MATCH_UNIT="${PREFIX_MATCH_UNIT:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.93}"
DCP="${DCP:-8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
ATTENTION_CONFIG="${ATTENTION_CONFIG:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
INDEXER_KV_DTYPE="${INDEXER_KV_DTYPE:-}"
DECODE_BACKEND="${DECODE_BACKEND:-}"
SPEC_TOKENS="${SPEC_TOKENS:-0}"
DRAFT_DIR="${DRAFT_DIR:-}"
DRAFT_ATTENTION_BACKEND="${DRAFT_ATTENTION_BACKEND-FLASHINFER_MLA}"
DRAFT_QUANTIZATION="${DRAFT_QUANTIZATION:-}"
DRAFT_KV_CACHE_DTYPE="${DRAFT_KV_CACHE_DTYPE:-}"
# K3 is served with the block rejection sampler (the K3 recipe's DSpark row); the K3.1 recipe leaves vLLM's default.
if [[ "$FAMILY" == k3 ]]; then REJECTION_SAMPLE_METHOD="${REJECTION_SAMPLE_METHOD-block}"; else REJECTION_SAMPLE_METHOD="${REJECTION_SAMPLE_METHOD-}"; fi
USE_RECOVERSSM="${USE_RECOVERSSM:-0}"
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
KV_OFFLOAD="${KV_OFFLOAD:-off}"
KV_OFFLOAD_GIB="${KV_OFFLOAD_GIB:-0}"
KV_OFFLOAD_DIR="${KV_OFFLOAD_DIR:-}"
KV_OFFLOAD_PROMPT_ONLY="${KV_OFFLOAD_PROMPT_ONLY:-0}"
LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"
LANGUAGE_ONLY="${LANGUAGE_ONLY:-0}"
RUST_FRONTEND="${RUST_FRONTEND:-1}"
FABRIC_WAIT_MIN="${FABRIC_WAIT_MIN:-15}"
GPU_FREE_WAIT_MIN="${GPU_FREE_WAIT_MIN:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"

# The download unit writes this marker once `hf download` has finished and every file's size has been verified.
if [[ "$LOAD_FORMAT" != dummy && ! -e "$MODEL_DIR/.download-complete" ]]; then
  echo "$MODEL_DIR/.download-complete is missing: the weights are still downloading or unverified (systemctl list-units 'hf-*')" >&2
  exit 1
fi
case "$KV_CACHE_DTYPE" in fp8|auto|bfloat16) ;; nvfp4_ds_mla|fp8_ds_mla)
  [[ "$FAMILY" == k31 ]] || { echo "KV_CACHE_DTYPE=$KV_CACHE_DTYPE is a K3.1 (DSA) layout; K3 takes fp8" >&2; exit 1; } ;;
  *) echo "KV_CACHE_DTYPE must be fp8, auto, bfloat16, nvfp4_ds_mla or fp8_ds_mla, got $KV_CACHE_DTYPE" >&2; exit 1 ;; esac
case "$INDEXER_KV_DTYPE" in "") ;; mxfp4|fp8|auto)
  [[ "$FAMILY" == k31 ]] || { echo "INDEXER_KV_DTYPE applies to K3.1 only (K3 has no indexer)" >&2; exit 1; } ;;
  *) echo "INDEXER_KV_DTYPE must be \"\", auto, fp8 or mxfp4, got $INDEXER_KV_DTYPE" >&2; exit 1 ;; esac
case "$DECODE_BACKEND" in ""|latency_specialized) ;; *) echo "DECODE_BACKEND must be \"\" or latency_specialized, got $DECODE_BACKEND" >&2; exit 1 ;; esac
if [[ -n "$ATTENTION_CONFIG" ]]; then
  python3 -c 'import json,sys; o=json.loads(sys.argv[1]); assert isinstance(o, dict)' "$ATTENTION_CONFIG" 2>/dev/null \
    || { echo "ATTENTION_CONFIG must be a JSON object, got: $ATTENTION_CONFIG" >&2; exit 1; }
fi
[[ -z "$API_KEY_FILE" || -r "$API_KEY_FILE" ]] || { echo "API_KEY_FILE=$API_KEY_FILE is not readable" >&2; exit 1; }
[[ -z "$API_KEYS_FILE" || -r "$API_KEYS_FILE" ]] || { echo "API_KEYS_FILE=$API_KEYS_FILE is not readable" >&2; exit 1; }
KEY_ARGS=()
if [[ -n "$API_KEYS_FILE" ]]; then
  mapfile -t API_KEYS < <(grep -vE '^[[:space:]]*(#|$)' "$API_KEYS_FILE" | tr -d '\r' | sed 's/[[:space:]]*$//')
  (( ${#API_KEYS[@]} > 0 )) || { echo "API_KEYS_FILE=$API_KEYS_FILE contains no keys" >&2; exit 1; }
  KEY_ARGS=(--api-key "${API_KEYS[@]}")
fi
case "$DCP" in 1|2|4|8) ;; *) echo "DCP must divide TP 8 (1, 2, 4 or 8), got $DCP" >&2; exit 1 ;; esac
[[ "$PREFIX_MATCH_UNIT" =~ ^[1-9][0-9]*$ ]] || { echo "PREFIX_MATCH_UNIT must be a positive integer, got $PREFIX_MATCH_UNIT" >&2; exit 1; }
case "$KV_OFFLOAD_PROMPT_ONLY" in 0|1) ;; *) echo "KV_OFFLOAD_PROMPT_ONLY must be 0 or 1, got $KV_OFFLOAD_PROMPT_ONLY" >&2; exit 1 ;; esac
case "$KV_OFFLOAD" in off|disk|cpu|native) ;; *) echo "KV_OFFLOAD must be off, disk, cpu or native, got $KV_OFFLOAD" >&2; exit 1 ;; esac
case "$USE_RECOVERSSM" in 0|1) ;; *) echo "USE_RECOVERSSM must be 0 or 1, got $USE_RECOVERSSM" >&2; exit 1 ;; esac
case "$MAMBA_SSM_CACHE_DTYPE" in ""|float32|bfloat16) ;; *) echo "MAMBA_SSM_CACHE_DTYPE must be \"\", float32 or bfloat16, got $MAMBA_SSM_CACHE_DTYPE" >&2; exit 1 ;; esac
if [[ "$USE_RECOVERSSM" == 1 ]]; then
  [[ "$SPEC_TOKENS" != 0 ]] || { echo "USE_RECOVERSSM=1 needs SPEC_TOKENS > 0 (K3 accepts --use-replayssm only with speculative decoding)" >&2; exit 1; }
  # The v0.32.0 release rejects --use-replayssm next to a KV connector; the K3.1 line (kimi-k31 and later) admits
  # KDA RecoverSSM there itself, which is what tools/k3_b300/patch_vllm_recoverssm_offload.py adds to v0.32.0.
  if [[ "$KV_OFFLOAD" != off && "$IMAGE" == vllm/vllm-openai:v0.32.0* ]]; then
    echo "USE_RECOVERSSM=1 with KV_OFFLOAD=$KV_OFFLOAD: the official v0.32.0 image rejects --use-replayssm next to a KV connector;" >&2
    echo "build Dockerfile.b300-k3 (IMAGE=vllm-moet-b300:k3-v0.32.0-r5) or set KV_OFFLOAD=off" >&2
    exit 1
  fi
fi
if [[ "$KV_OFFLOAD" != off ]]; then
  [[ "$KV_OFFLOAD_GIB" =~ ^[1-9][0-9]*$ ]] || { echo "KV_OFFLOAD=$KV_OFFLOAD needs KV_OFFLOAD_GIB > 0, got '$KV_OFFLOAD_GIB'" >&2; exit 1; }
  [[ "$KV_OFFLOAD" != disk || -n "$KV_OFFLOAD_DIR" ]] || { echo "KV_OFFLOAD=disk needs KV_OFFLOAD_DIR" >&2; exit 1; }
fi
mkdir -p "$CACHE_DIR/dot-cache" "$CACHE_DIR/deep_gemm"

ARGS=()
MOUNTS=(-v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR/dot-cache:/root/.cache" -v "$CACHE_DIR/deep_gemm:/root/.deep_gemm")
[[ "$DCP" != 1 ]] && ARGS+=(--decode-context-parallel-size "$DCP" --dcp-comm-backend a2a)
# Attention: K3's dense MLA on TOKENSPEED_MLA with the recipe's prefill side (TRTLLM_RAGGED under DCP, where the
# sharded KV is gathered for prefill; TOKENSPEED_MLA's own prefill without), K3.1's sparse MLA on vLLM's choice.
if [[ -z "$ATTENTION_BACKEND" && "$FAMILY" == k3 ]]; then ATTENTION_BACKEND=TOKENSPEED_MLA; fi
if [[ -z "$ATTENTION_CONFIG" ]]; then
  if [[ "$FAMILY" == k3 ]]; then
    if [[ "$DCP" != 1 ]]; then ATTENTION_CONFIG='{"use_prefill_query_quantization":true,"mla_prefill_backend":"TRTLLM_RAGGED"}'
    else ATTENTION_CONFIG='{"use_prefill_query_quantization":true,"mla_prefill_backend":"TOKENSPEED_MLA"}'; fi
  elif [[ -n "$INDEXER_KV_DTYPE" ]]; then
    ATTENTION_CONFIG="{\"indexer_kv_dtype\":\"$INDEXER_KV_DTYPE\"}"
  fi
elif [[ -n "$INDEXER_KV_DTYPE" ]]; then
  ATTENTION_CONFIG="$(python3 -c 'import json,sys; o=json.loads(sys.argv[1]); o["indexer_kv_dtype"]=sys.argv[2]; print(json.dumps(o))' "$ATTENTION_CONFIG" "$INDEXER_KV_DTYPE")"
fi
[[ -n "$ATTENTION_BACKEND" ]] && ARGS+=(--attention-backend "$ATTENTION_BACKEND")
[[ -n "$ATTENTION_CONFIG" ]] && ARGS+=(--attention-config "$ATTENTION_CONFIG")
[[ -n "$DECODE_BACKEND" ]] && ARGS+=(--additional-config "{\"decode_backend\":\"$DECODE_BACKEND\"}")
if [[ "$SPEC_TOKENS" != 0 ]]; then
  [[ -f "$DRAFT_DIR/config.json" ]] || { echo "SPEC_TOKENS=$SPEC_TOKENS needs DRAFT_DIR with the DSpark checkpoint" >&2; exit 1; }
  MOUNTS+=(-v "$DRAFT_DIR:/draft:ro")
  DRAFT_OPTS=""
  [[ -n "$DRAFT_ATTENTION_BACKEND" ]] && DRAFT_OPTS+="\"attention_backend\":\"$DRAFT_ATTENTION_BACKEND\","
  [[ -n "$DRAFT_QUANTIZATION" ]] && DRAFT_OPTS+="\"quantization\":\"$DRAFT_QUANTIZATION\","
  [[ -n "$DRAFT_KV_CACHE_DTYPE" ]] && DRAFT_OPTS+="\"kv_cache_dtype\":\"$DRAFT_KV_CACHE_DTYPE\","
  [[ -n "$REJECTION_SAMPLE_METHOD" ]] && DRAFT_OPTS+="\"rejection_sample_method\":\"$REJECTION_SAMPLE_METHOD\","
  ARGS+=(--speculative-config
    "{\"model\":\"/draft\",\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},${DRAFT_OPTS}\"draft_sample_method\":\"probabilistic\"}")
fi
[[ "$USE_RECOVERSSM" == 1 ]] && ARGS+=(--use-replayssm)
[[ -n "$MAMBA_SSM_CACHE_DTYPE" ]] && ARGS+=(--mamba-ssm-cache-dtype "$MAMBA_SSM_CACHE_DTYPE")
[[ "$LANGUAGE_ONLY" == 1 ]] && ARGS+=(--language-model-only)
[[ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]] && ARGS+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
OFFLOAD_DIR="$KV_OFFLOAD_DIR"
# Offloaded KV is keyed by token content, not by the load format: blocks computed with random weights would be
# served to the real model.
[[ -n "$OFFLOAD_DIR" && "$LOAD_FORMAT" == dummy ]] && OFFLOAD_DIR="$OFFLOAD_DIR/dummy"
if [[ -n "$OFFLOAD_DIR" && ( "$KV_OFFLOAD" == disk || "$KV_OFFLOAD" == native ) ]]; then
  mkdir -p "$OFFLOAD_DIR"
  MOUNTS+=(-v "$OFFLOAD_DIR:$OFFLOAD_DIR")
fi
PER_RANK_BYTES=$(( KV_OFFLOAD_GIB * 1024 * 1024 * 1024 / 8 ))
case "$KV_OFFLOAD" in
  disk)
    # The disk backend writes <disk_path>.rank_N next to disk_path: disk_path is a file prefix inside the mounted
    # directory, or the slabs land in the container's own layer on the root filesystem.
    ARGS+=(--kv-transfer-config
      "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"kv_offload_backend\":\"disk\",\"disk_path\":\"$OFFLOAD_DIR/k3\",\"disk_capacity_bytes\":$PER_RANK_BYTES,\"disk_buffer_slots\":16,\"use_page_cache\":true,\"lazy_offload\":false}}") ;;
  cpu)
    ARGS+=(--kv-transfer-config
      "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$PER_RANK_BYTES,\"lazy_offload\":false}}") ;;
  native)
    # --kv-offloading-size adds cpu_bytes_to_use to the connector's extra config given here.
    ARGS+=(--kv-offloading-size "$KV_OFFLOAD_GIB")
    OFFLOAD_EXTRA=""
    [[ "$KV_OFFLOAD_PROMPT_ONLY" == 0 ]] && OFFLOAD_EXTRA="\"offload_prompt_only\":false"
    [[ -n "$OFFLOAD_DIR" ]] && OFFLOAD_EXTRA="\"spec_name\":\"TieringOffloadingSpec\",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$OFFLOAD_DIR\",\"n_read_threads\":32,\"n_write_threads\":16,\"locality\":\"LOCAL\"}]${OFFLOAD_EXTRA:+,$OFFLOAD_EXTRA}"
    [[ -n "$OFFLOAD_EXTRA" ]] && ARGS+=(--kv-transfer-config
      "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{$OFFLOAD_EXTRA}}") ;;
esac
ENV_ARGS=()
[[ -n "$API_KEY_FILE" ]] && ENV_ARGS+=(--env-file "$API_KEY_FILE")
# One -p per BIND prefix ("" = one unrestricted publish).
PUBLISH=()
if [[ -z "$BIND" ]]; then PUBLISH+=(-p "${PORT}:${PORT}"); else
  for prefix in $BIND; do PUBLISH+=(-p "${prefix}${PORT}:${PORT}"); done
fi

if docker ps -q -f name="^${NAME}$" -f status=running | grep -q .; then
  echo "container $NAME already running - stop it first (docker stop -t 120 $NAME)"; exit 0
fi

# CUDA fails with "system not yet initialized" until nvidia-fabricmanager and NVLSM have registered every GPU.
states=""
for ((i = 0; i < FABRIC_WAIT_MIN * 12; i++)); do
  states="$(timeout 30 nvidia-smi --query-gpu=fabric.state --format=csv,noheader 2>/dev/null | sort -u)"
  [[ "$states" == Completed ]] && break
  sleep 5
done
[[ "$states" == Completed ]] || { echo "NVLink fabric not ready after $FABRIC_WAIT_MIN min (fabric.state: ${states//$'\n'/, }); check nvidia-fabricmanager" >&2; exit 1; }
# A stopped server keeps its GPU memory for minutes after its container is gone, while the driver unpins the
# host offload buffers (nvidia-smi blocks meanwhile, hence the timeout).
used=""
for ((i = 0; i < GPU_FREE_WAIT_MIN * 12; i++)); do
  used="$(timeout 30 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -n1)"
  [[ -n "$used" ]] && (( used < 4096 )) && break
  sleep 5
done
[[ -n "$used" ]] && (( used < 4096 )) || { echo "GPU memory still in use (${used:-?} MiB on the busiest GPU) after $GPU_FREE_WAIT_MIN min" >&2; exit 1; }

docker rm -f "$NAME" >/dev/null 2>&1 || true
if [[ "$KV_OFFLOAD" == native ]]; then
  # An offload region its server did not remove on exit stays in the host's /dev/shm (RAM) until deleted;
  # drop the ones no process maps any more.
  for region in /dev/shm/vllm_offload_*.mmap; do
    [[ -e "$region" ]] || continue
    grep -qs -- "$region" /proc/[0-9]*/maps || { echo "removing stale offload region $region"; rm -f "$region"; }
  done
fi

# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart no \
  --gpus all \
  --ipc host \
  "${PUBLISH[@]}" \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  "${MOUNTS[@]}" \
  "${ENV_ARGS[@]}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_ALLREDUCE_USE_FLASHINFER=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_RUST_FRONTEND="$RUST_FRONTEND" \
  ${EXTRA_DOCKER_ARGS} \
  "$IMAGE" \
  --model /model --served-model-name "${SERVED_NAMES[@]}" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --load-format "$LOAD_FORMAT" \
  --no-enable-flashinfer-autotune \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --enable-prefix-caching --prefix-match-unit "$PREFIX_MATCH_UNIT" \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 --enable-auto-tool-choice \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  "${ARGS[@]}" \
  ${EXTRA_ARGS} \
  "${KEY_ARGS[@]}" \
  --host 0.0.0.0 --port "$PORT"

echo "container $NAME started; logs: docker logs -f $NAME ; API: $(for prefix in ${BIND:-0.0.0.0:}; do printf 'http://%s%s/v1 ' "$prefix" "$PORT"; done)${API_KEYS_FILE:+ (${#API_KEYS[@]} API keys from $API_KEYS_FILE)}"
exit 0
