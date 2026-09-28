#!/bin/bash
#SBATCH --job-name=ddpm-exp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=koes_gpu
#SBATCH --gres=gpu:1
#SBATCH -C "L40|A100"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

# Train one experiment config, then KID-evaluate its final checkpoint.
# Usage: sbatch --job-name=exp-b128 scripts/train_eval.sh configs/ddpm_b128.yaml

CONFIG=${1:?usage: sbatch scripts/train_eval.sh <config.yaml>}
TAG=$(basename "$CONFIG" .yaml)

source .venv-cuda*/bin/activate
mkdir -p results

echo "===== training: $CONFIG ====="
# Marker so we can identify OUR run dir afterwards: "newest dir" is a race
# when jobs run concurrently (bit us 2026-09-28: a resume job's eval grabbed
# a still-training sibling's dir). A candidate must be newer than the marker
# AND actually finished (has ddpm_final.pt) — excludes concurrent runs.
MARKER="logs/.start_${SLURM_JOB_ID:-manual}"
mkdir -p logs
touch "$MARKER"
python train.py --method ddpm --config "$CONFIG" || exit 1

RUN_DIR=""
for d in $(find logs -maxdepth 1 -type d -name 'ddpm_2*' -newer "$MARKER" | sort); do
    [ -f "$d/checkpoints/ddpm_final.pt" ] && RUN_DIR="$d"
done
rm -f "$MARKER"
CKPT="$RUN_DIR/checkpoints/ddpm_final.pt"
if [ -z "$RUN_DIR" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no finished run dir found (marker-based search)"; exit 1
fi
DATA=$(find data/celeba -maxdepth 3 -type d -name images | head -1)

echo "===== evaluating: $CKPT ====="
./scripts/evaluate_torch_fidelity.sh --checkpoint "$CKPT" --method ddpm \
    --dataset-path "$DATA" --num-steps 1000 2>&1 | tee "results/kid_${TAG}.txt"

echo "===== done: $TAG ====="
grep -iH "kernel_inception" "results/kid_${TAG}.txt"
