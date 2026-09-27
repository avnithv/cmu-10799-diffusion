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
python train.py --method ddpm --config "$CONFIG" || exit 1

# newest run dir belongs to the training that just finished
RUN_DIR=$(ls -td logs/ddpm_2* | head -1)
CKPT="$RUN_DIR/checkpoints/ddpm_final.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: no final checkpoint at $CKPT"; exit 1
fi
DATA=$(find data/celeba -maxdepth 3 -type d -name images | head -1)

echo "===== evaluating: $CKPT ====="
./scripts/evaluate_torch_fidelity.sh --checkpoint "$CKPT" --method ddpm \
    --dataset-path "$DATA" --num-steps 1000 2>&1 | tee "results/kid_${TAG}.txt"

echo "===== done: $TAG ====="
grep -iH "kernel_inception" "results/kid_${TAG}.txt"
