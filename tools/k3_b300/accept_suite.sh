#!/usr/bin/env bash
# DSpark acceptance suite for K3: one stream, greedy, four prompt types (Fraser code, 1500-word prose, a number-theory
# proof at reasoning_effort high, a 30K-token design document + coding task), accepted length and tok/s per type.
# usage: accept_suite.sh LABEL [REPS] [BASE]   (API_KEY in the environment if the server needs one; MODEL = the served
#        model name, moonshotai/Kimi-K3 by default - moonshotai/Kimi-K3.1-Pro for K3.1)
# Appends one line per prompt type to accept-suite.log in the current directory.
set -euo pipefail
LABEL=$1; REPS=${2:-2}; BASE=${3:-http://127.0.0.1:8000}
MODEL=${MODEL:-moonshotai/Kimi-K3}
HERE=$(cd "$(dirname "$0")" && pwd)
PROMPTS=$HERE/prompts
[[ -f $PROMPTS/code.txt ]] || cp "$HERE/../sm120_perf/fraser_prompt.txt" "$PROMPTS/code.txt"
[[ -f $PROMPTS/longctx.txt ]] || python3 "$HERE/make_longctx_prompt.py" > "$PROMPTS/longctx.txt"
for p in code prose math longctx; do
  effort=low; max=2048
  [[ $p == math ]] && effort=high
  [[ $p == prose ]] && max=1800
  python3 "$HERE/k3_code_bench.py" --base "$BASE" --model "$MODEL" --prompt-file "$PROMPTS/$p.txt" --reps $REPS \
    --max-tokens $max --reasoning-effort $effort --label "$LABEL/$p" 2>&1 | tail -1 | tee -a accept-suite.log
done
