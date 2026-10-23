#!/bin/bash
# CPU kernel v2 vs b12x's compact path on other ranks / layers than G1b covered (rank 0, layers 0/1/20/39):
# for each (rank, layer): 16 real experts, 8 tokens -> GPU run in the base image on GPU $GPU under its lock -> compare.
set -u
GPU="${1:-1}"
REPO=/mnt/nvme/vLLM-Moet-cpu
P=$REPO/internal/dsv41-5090-cpu/parity/sweep
mkdir -p $P
PY=/root/work/venv/bin/python
cd $REPO/tools/dsv41_5090/cpu_moe
for rank in 1 2 3 0; do
  for layer in 5 14 30 39; do
    tag=r${rank}_l${layer}
    $PY gpu_parity_b12x_prepare.py --layer $layer --rank $rank --experts 16 --first-expert $((layer * 7 % 300)) --tokens 8 --seed $((100 + layer + rank)) --out $P/in_$tag.npz > $P/prep_$tag.log 2>&1 || { echo "$tag prepare FAILED"; continue; }
    flock -w 3600 /tmp/exl3diff-gpu$GPU.lock docker run --rm --gpus "\"device=$GPU\"" --name moet-parity-sweep -v /mnt/nvme/cache/ds41/dot-cache:/root/.cache \
      -v $PWD:/work:ro -v $P:/parity --entrypoint python3 vllm/vllm-openai:nightly-df3bd094aeea829f27baa0a5282830464f601d9f \
      /work/gpu_parity_b12x_run.py /parity/in_$tag.npz /parity/gpu_$tag.npz > $P/run_$tag.log 2>&1 || { echo "$tag gpu run FAILED: $(tail -2 $P/run_$tag.log)"; continue; }
    $PY gpu_parity_b12x_compare.py --inputs $P/in_$tag.npz --gpu $P/gpu_$tag.npz --report $P/report_$tag.json --cpus 9-15 > $P/cmp_$tag.log 2>&1
    echo "$tag: $(grep -E "^2\. kernel vs b12x|G1b" $P/cmp_$tag.log | tr '\n' ' ' | cut -c1-230)"
  done
done
