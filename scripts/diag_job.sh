#!/bin/bash
#SBATCH --job-name=border-diag
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH -C 2080Ti
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

# Border diagnostics for a finished run:
#   1. border_stats on its generated eval samples (vs baseline set + reference)
#   2. trajectory diagnostic on its checkpoint (when do borders darken?)
#   3. same trajectory on the best (resume1e5) model for comparison
#
# Usage: sbatch scripts/diag_job.sh [run_dir]     (default: newest finished run)

source .venv-cuda*/bin/activate

RUN_DIR=${1:-}
if [ -z "$RUN_DIR" ]; then
    for d in $(ls -td logs/ddpm_2*); do
        [ -f "$d/checkpoints/ddpm_final.pt" ] && RUN_DIR=$d && break
    done
fi
TAG=$(basename "$RUN_DIR")
CKPT="$RUN_DIR/checkpoints/ddpm_final.pt"
echo "diagnosing: $RUN_DIR (tag $TAG)"
grep -E "padding_mode|learning_rate:|batch_size:" "$RUN_DIR/config.yaml" | head -4
mkdir -p "results/$TAG"

echo "===== [1/3] border statistics ====="
python scripts/border_stats.py \
    "$RUN_DIR/checkpoints/samples/generated" witness_samples/baseline --reference \
    2>&1 | tee "results/$TAG/border_stats.txt"

echo "===== [2/3] trajectory diagnostic: $TAG ====="
python scripts/trajectory_diag.py --checkpoint "$CKPT" --tag "$TAG"

echo "===== [3/3] trajectory diagnostic: best model (resume1e5) for comparison ====="
python scripts/trajectory_diag.py \
    --checkpoint logs/ddpm_20260927_222014/checkpoints/ddpm_final.pt --tag resume1e5

echo "done — see results/$TAG/ and results/resume1e5/"
