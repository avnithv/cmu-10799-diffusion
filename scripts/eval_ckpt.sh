#!/bin/bash
#SBATCH --job-name=eval-ckpt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH -C "2080Ti|L40|A100"
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00

# Standalone KID eval of one checkpoint (for when train_eval.sh's chained eval
# didn't run, or to re-evaluate anything). Runs fine on a free 2080Ti.
# Usage: sbatch scripts/eval_ckpt.sh <checkpoint.pt> <result-tag> [num_steps]
#   -> results/kid_<result-tag>.txt

CKPT=${1:?usage: sbatch scripts/eval_ckpt.sh <checkpoint.pt> <tag> [num_steps]}
TAG=${2:?result tag required (e.g. ddpm_resume1e5)}
STEPS=${3:-1000}

source .venv-cuda*/bin/activate
DATA=$(find data/celeba -maxdepth 3 -type d -name images | head -1)
mkdir -p results

./scripts/evaluate_torch_fidelity.sh --checkpoint "$CKPT" --method ddpm \
    --dataset-path "$DATA" --num-steps "$STEPS" --batch-size 128 \
    2>&1 | tee "results/kid_${TAG}.txt"

grep -iH "kernel_inception" "results/kid_${TAG}.txt"
