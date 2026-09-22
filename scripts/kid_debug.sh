#!/bin/bash
#SBATCH --job-name=kid-debug
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=koes_gpu
#SBATCH --gres=gpu:1
#SBATCH -C "L40|A100"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00

# KID debugging: why is the baseline ~0.045 instead of <0.005?
#   [1] EMA sanity      — are the checkpoint's EMA weights distinct from raw weights?
#   [2] real-vs-real    — KID between two halves of the reference set (pipeline bias
#                         control; healthy value is ~0.000x)
#   [3] rerun 1000      — run-to-run variance of the baseline (fresh sampling seed)
#   [4] rerun 300       — is the 0.076 outlier reproducible or sampling noise?

source .venv-cuda*/bin/activate

CKPT=logs/ddpm_20260917_205522/checkpoints/ddpm_final.pt
REF=$(readlink -f data/celeba/train/images)
mkdir -p results

echo "===== [1/4] EMA sanity check ====="
python - <<'PY'
import torch
ckpt = torch.load("logs/ddpm_20260917_205522/checkpoints/ddpm_final.pt", map_location="cpu")
model, ema = ckpt["model"], ckpt["ema"]
print("ema step:", ema.get("step"), "| decay:", ema.get("decay"))
shadow = ema["shadow"]
tot, diff = 0, 0.0
identical = 0
for name, sh in shadow.items():
    if name in model:
        rel = (sh - model[name]).norm() / (model[name].norm() + 1e-12)
        diff += rel.item(); tot += 1
        if rel.item() < 1e-9: identical += 1
print(f"params compared: {tot} | exactly identical to raw: {identical}")
print(f"mean relative EMA-vs-raw difference: {diff/max(tot,1):.6f}")
print("VERDICT:", "EMA IS A NO-OP (bug!)" if tot and identical == tot else "EMA weights are distinct (healthy)")
PY

echo ""
echo "===== [2/4] real-vs-real KID control ====="
CTRL=kid_control
rm -rf "$CTRL"; mkdir -p "$CTRL/a" "$CTRL/b"
find "$REF" -name '*.png' | sort > "$CTRL/all.txt"
awk 'NR % 2 == 1' "$CTRL/all.txt" | xargs -I{} ln -s {} "$CTRL/a/"
awk 'NR % 2 == 0' "$CTRL/all.txt" | xargs -I{} ln -s {} "$CTRL/b/"
echo "half A: $(ls "$CTRL/a" | wc -l) images, half B: $(ls "$CTRL/b" | wc -l) images"
fidelity --gpu 0 --kid --batch-size 256 \
    --input1 "$CTRL/a" --input2 "$CTRL/b" 2>&1 | tee results/kid_control.txt

echo ""
echo "===== [3/4] rerun baseline (1000 steps, fresh seed) ====="
./scripts/evaluate_torch_fidelity.sh --checkpoint "$CKPT" --method ddpm \
    --dataset-path "$REF" --num-steps 1000 2>&1 | tee results/kid_1000_rerun.txt

echo ""
echo "===== [4/4] rerun the outlier (300 steps, fresh seed) ====="
./scripts/evaluate_torch_fidelity.sh --checkpoint "$CKPT" --method ddpm \
    --dataset-path "$REF" --num-steps 300 2>&1 | tee results/kid_300_rerun.txt

echo ""
echo "===== summary ====="
grep -iH "kernel_inception" results/kid_control.txt results/kid_1000_rerun.txt results/kid_300_rerun.txt
