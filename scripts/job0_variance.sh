#!/bin/bash
#SBATCH --job-name=job0-posterior-var
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH -C 2080Ti
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

# Job 0: evaluate the EXISTING baseline checkpoint with posterior-variance
# (beta-tilde) sampling instead of sigma^2 = beta.
#
# PREREQUISITE (your code): sample.py must accept `--variance posterior` and
# thread it through to DDPM.sample()/reverse_process(). This script assumes
# that contract; it will fail loudly on the argparse error if it's missing.
#
# Runs on a 2080Ti (usually free): inference fits in 11GB at batch 128.

source .venv-cuda*/bin/activate

CKPT=logs/ddpm_20260917_205522/checkpoints/ddpm_final.pt
DATA=$(find data/celeba -maxdepth 3 -type d -name images | head -1)
OUT=samples_posterior_var
mkdir -p results
rm -rf "$OUT"

echo "===== generating 1000 samples, posterior variance ====="
python sample.py --checkpoint "$CKPT" --method ddpm \
    --num_samples 1000 --batch_size 128 --num_steps 1000 \
    --variance posterior --output_dir "$OUT" || exit 1

echo "===== KID ====="
fidelity --gpu 0 --kid --batch-size 128 \
    --input1 "$OUT" --input2 "$DATA" 2>&1 | tee results/kid_1000_posteriorvar.txt

grep -iH "kernel_inception" results/kid_1000_posteriorvar.txt
