#!/bin/bash
# Every expert of one layer (one rank) against b12x's compact path, 16 experts x 8 tokens per set.
#   moet_parity_layer.sh GPU RANK LAYER
set -u
GPU="${1:-1}"; RANK="${2:-0}"; LAYER="${3:-0}"
REPO=/mnt/nvme/vLLM-Moet-cpu
P=$REPO/internal/dsv41-5090-cpu/parity/sweep
mkdir -p $P
PY=/root/work/venv/bin/python
cd $REPO/tools/dsv41_5090/cpu_moe
bad=0; sets=0
for first in $(seq 0 16 368); do
  tag=r${RANK}_l${LAYER}_e${first}
  $PY gpu_parity_b12x_prepare.py --layer $LAYER --rank $RANK --experts 16 --first-expert $first --tokens 8 --seed $((500 + first)) --out $P/in_$tag.npz > $P/prep_$tag.log 2>&1 || { echo "$tag prepare FAILED"; continue; }
  flock -w 3600 /tmp/exl3diff-gpu$GPU.lock docker run --rm --gpus "\"device=$GPU\"" --name moet-parity-sweep -v /mnt/nvme/cache/ds41/dot-cache:/root/.cache \
    -v $PWD:/work:ro -v $P:/parity --entrypoint python3 vllm/vllm-openai:nightly-df3bd094aeea829f27baa0a5282830464f601d9f \
    /work/gpu_parity_b12x_run.py /parity/in_$tag.npz /parity/gpu_$tag.npz > $P/run_$tag.log 2>&1 || { echo "$tag gpu run FAILED: $(tail -2 $P/run_$tag.log)"; continue; }
  $PY gpu_parity_b12x_compare.py --inputs $P/in_$tag.npz --gpu $P/gpu_$tag.npz --report $P/report_$tag.json --cpus 9-15 > $P/cmp_$tag.log 2>&1
  line=$(grep -E "^2\. kernel vs b12x" $P/cmp_$tag.log | sed 's/.*route_output/route_output/' | cut -c1-80)
  verdict=$(grep -o "G1b [A-Z]*" $P/cmp_$tag.log)
  sets=$((sets+1)); [ "$verdict" = "G1b PASS" ] || bad=$((bad+1))
  echo "$tag: $line -> $verdict"
done
echo "layer $LAYER rank $RANK: $sets sets, $bad not PASS"
