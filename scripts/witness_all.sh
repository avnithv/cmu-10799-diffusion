#!/bin/bash
#SBATCH --job-name=witness-all
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH -C 2080Ti
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

# Witness analysis across all four experiment models.
# Fast path: reuses each run's existing 1000-step eval samples where valid;
# regenerates only where the on-disk set is known-stale (REGEN list below).
#
# REGEN notes:
#   - baseline: its generated/ was last written by the 300-step debug rerun.
#   - add resume1e5 here if scripts/eval.sh (the Q7 ablation) has been re-run
#     since 2026-09-28 — its last loop iteration leaves a 100-step set behind.
REGEN="baseline"

source .venv-cuda*/bin/activate
mkdir -p results

declare -A RUNS=(
  [baseline]=logs/ddpm_20260917_205522
  [resume1e5]=logs/ddpm_20260927_222014
  [lr1e3]=logs/ddpm_20260927_011049
  [c192]=logs/ddpm_20260928_111807
)

for tag in baseline resume1e5 lr1e3 c192; do
    run=${RUNS[$tag]}
    echo "===== $tag : $run ====="
    # prove in the log which config this tag really is
    grep -E "base_channels:|learning_rate:|num_iterations:" "$run/config.yaml" | head -3

    if [[ " $REGEN " == *" $tag "* ]]; then
        out=witness_samples/$tag
        echo "regenerating 1000 fresh 1000-step samples -> $out"
        rm -rf "$out"
        python sample.py --checkpoint "$run/checkpoints/ddpm_final.pt" --method ddpm \
            --num_samples 1000 --batch_size 64 --num_steps 1000 --seed 0 \
            --output_dir "$out" || { echo "$tag generation FAILED"; continue; }
    else
        out=$run/checkpoints/samples/generated
        if [ ! -d "$out" ]; then echo "$tag: no existing samples at $out — skipping"; continue; fi
        echo "using existing set: $(ls "$out" | wc -l) images at $out"
    fi

    python scripts/witness.py --generated "$out" --tag "$tag" || echo "$tag witness FAILED"
done

echo "===== summary ====="
ls -la results/witness_*.csv results/grids/witness_*.png 2>/dev/null
