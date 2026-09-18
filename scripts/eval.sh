#!/bin/bash
#SBATCH --job-name=kid-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=koes_gpu
#SBATCH --gres=gpu:1
#SBATCH -C "L40|A100"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# KID evaluation: baseline (1000 steps) + Q7 step-count ablation (900..100).
#
# Usage:
#   sbatch scripts/eval.sh                    # auto-picks newest checkpoint
#   sbatch scripts/eval.sh path/to/ckpt.pt    # explicit checkpoint
#
# Outputs land in results/: kid_<steps>.txt logs and one q7_sample_<steps>steps.png
# per setting (grabbed before evaluate_torch_fidelity.sh wipes its generated dir).

source .venv-cuda*/bin/activate

# Newest .pt wins the default: the final checkpoint is written last.
CKPT=${1:-$(ls -t logs/*/checkpoints/*.pt checkpoints/*.pt 2>/dev/null | head -1)}
# fidelity scans non-recursively, so the reference must be the images/ dir itself.
DATA=$(find data/celeba -maxdepth 3 -type d -name images | head -1)

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found (looked in logs/*/checkpoints/ and checkpoints/)"
    exit 1
fi
if [ -z "$DATA" ]; then
    echo "ERROR: no images/ directory found under data/celeba"
    exit 1
fi

echo "Checkpoint: $CKPT"
echo "Reference:  $DATA ($(ls "$DATA" | wc -l) images)"

GEN=$(dirname "$CKPT")/samples/generated   # matches evaluate_torch_fidelity.sh
mkdir -p results

for n in 1000 900 700 500 300 100; do
    echo "===== num_steps = $n ====="
    ./scripts/evaluate_torch_fidelity.sh --checkpoint "$CKPT" --method ddpm \
        --dataset-path "$DATA" --num-steps "$n" 2>&1 | tee "results/kid_${n}.txt"
    cp "$GEN/000000.png" "results/q7_sample_${n}steps.png"
done

echo "===== KID summary ====="
grep -iH "kernel_inception" results/kid_*.txt
