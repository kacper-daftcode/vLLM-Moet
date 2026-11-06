#!/bin/bash
# Mixed steps (decode rows batched with a prefill chunk) under load: three decode streams of one prompt (distinct fixed
# cache-bust nonces, a fresh prefill of the prompt each time, 2048 greedy tokens, per-token logprobs) run (a) alone, one
# at a time; (b) together, no other load; (c) together under back-to-back fresh 163K prefills (with
# VLLM_MOET_PREFILL_PIPE=2 the 8192-token steps carry the decode rows in group 0). Then mixed_compare.py: token-by-token
# identity of every stream across a/b/c. Note: decoding is not batch-invariant on this stack (kernels chosen by the row
# count, NCCL chunking, the prompts' prefills batched differently), so (b) and (c) differ from (a) with or without the
# pipeline; the pipeline's own oracle is the same server with VLLM_MOET_PREFILL_PIPE_FORCE_SEQUENTIAL=1 compared on the
# per-layer dumps of the prefill rows (compare_dumps.py --kind prefill_rows), which MIXED_DUMP_DIR arranges: the first
# 163K prefill of (c) is a seeded prompt (seed 4711) dumped alone (ENABLE present during it), its files end up in
# MIXED_DUMP_DIR/mixed/ (an earlier dump's files are kept aside meanwhile). With VLLM_MOET_PREFILL_PIPE_LOG_STEPS=1 the
# server logs one "prefill pipeline step:" line per mixed step (the batch composition).
#   usage: PROMPT_FILE=<prompt text> mixed_steps.sh <base-url> <model> <out-prefix>
#   env: STREAMS=3 PREFILLS=12 REPS=1 MIXED_DUMP_DIR=<the server's dump dir on the host>
set -uo pipefail
B=$1; M=$2; OUTP=$3
K3=$(cd "$(dirname "$0")" && pwd)   # decode_stream.py, prefill_loop.py, mixed_compare.py live next to this script
STREAMS=${STREAMS:-3}; PREFILLS=${PREFILLS:-12}; REPS=${REPS:-1}; MIXED_DUMP_DIR=${MIXED_DUMP_DIR:-}
PROMPT_FILE=${PROMPT_FILE:?PROMPT_FILE=<prompt text file> (a few thousand tokens)}
DS="python3 $K3/decode_stream.py --base $B --model $M --reps $REPS --fresh-prefill --prompt-file $PROMPT_FILE"
ts() { date -u +%H:%M:%SZ; }
rm -f "$OUTP".*.jsonl "$OUTP".stop
echo "[$(ts)] (a) alone: $STREAMS streams one at a time"
for i in $(seq 1 $STREAMS); do
  timeout 900 $DS --nonce k3stream$i --out "$OUTP.alone.s$i.jsonl" 2>&1 | tail -1
done
echo "[$(ts)] (b) together: $STREAMS streams concurrently, no other load"
for i in $(seq 1 $STREAMS); do
  ( timeout 1200 $DS --nonce k3stream$i --out "$OUTP.together.s$i.jsonl" > "$OUTP.together.s$i.log" 2>&1 ) &
done
wait
for i in $(seq 1 $STREAMS); do tail -1 "$OUTP.together.s$i.log"; done
echo "[$(ts)] (c) under load: $STREAMS streams concurrently + up to $PREFILLS fresh 163K prefills back to back (started 2 s in)"
PIDS=""
for i in $(seq 1 $STREAMS); do
  ( timeout 1800 $DS --nonce k3stream$i --out "$OUTP.under.s$i.jsonl" > "$OUTP.under.s$i.log" 2>&1 ) &
  PIDS="$PIDS $!"
done
sleep 2
DUMP_ARGS=""
if [[ -n "$MIXED_DUMP_DIR" ]]; then
  mkdir -p "$MIXED_DUMP_DIR/19k" "$MIXED_DUMP_DIR/mixed"; mv "$MIXED_DUMP_DIR"/rank*.jsonl "$MIXED_DUMP_DIR/19k/" 2>/dev/null
  DUMP_ARGS="--seed-first 4711 --dump-dir $MIXED_DUMP_DIR"
fi
timeout 1700 python3 $K3/prefill_loop.py $B $M $PREFILLS "$OUTP.stop" $DUMP_ARGS > "$OUTP.under.prefills.log" 2>&1 &
PP=$!
wait $PIDS
touch "$OUTP.stop"
wait $PP
rm -f "$OUTP.stop"
if [[ -n "$MIXED_DUMP_DIR" ]]; then
  sleep 2; mv "$MIXED_DUMP_DIR"/rank*.jsonl "$MIXED_DUMP_DIR/mixed/" 2>/dev/null; mv "$MIXED_DUMP_DIR"/19k/rank*.jsonl "$MIXED_DUMP_DIR/" 2>/dev/null
  echo "  mixed-step dump lines: $(cat "$MIXED_DUMP_DIR"/mixed/rank*.jsonl 2>/dev/null | wc -l) (prefill_rows entries: $(grep -c prefill_rows "$MIXED_DUMP_DIR"/mixed/rank*.jsonl 2>/dev/null | paste -sd,))"
fi
for i in $(seq 1 $STREAMS); do tail -1 "$OUTP.under.s$i.log"; done
echo "  prefills under the streams: $(grep -c 'fresh prefill' "$OUTP.under.prefills.log") x 163K: $(grep -h 'fresh prefill' "$OUTP.under.prefills.log" | sed -E 's/.*tok: ([0-9]+) ms.*/\1/' | paste -sd/) ms"
echo "[$(ts)] compare"
python3 $K3/mixed_compare.py "$OUTP"
echo "[$(ts)] done"
