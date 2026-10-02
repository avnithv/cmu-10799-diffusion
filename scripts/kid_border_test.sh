#!/bin/bash
#SBATCH --job-name=kid-border-test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH -C 2080Ti
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00

# Border-artifact quantification: KID on original images vs KID with the
# border zone center-cropped away (SAME crop applied to generated AND
# reference, so the delta isolates the border's contribution).
#
# Usage: sbatch scripts/kid_border_test.sh <generated_dir> <tag> [crop_px=4]
#   e.g. sbatch scripts/kid_border_test.sh witness_samples/baseline baseline

GEN=${1:?usage: sbatch scripts/kid_border_test.sh <generated_dir> <tag> [crop_px]}
TAG=${2:?tag required}
CROP=${3:-4}

source .venv-cuda*/bin/activate
REF=$(find data/celeba -maxdepth 3 -type d -name images | head -1)
WORK=border_test/$TAG
mkdir -p "results/$TAG"

echo "===== cropping both sets by ${CROP}px per side ====="
python - "$GEN" "$REF" "$WORK" "$CROP" <<'PY'
import glob, os, sys
from PIL import Image
gen, ref, work, crop = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
for src, dst in ((gen, f"{work}/gen"), (ref, f"{work}/ref")):
    os.makedirs(dst, exist_ok=True)
    files = glob.glob(os.path.join(src, "*.png")) + glob.glob(os.path.join(src, "*.jpg"))
    for i, f in enumerate(files):
        im = Image.open(f)
        w, h = im.size
        im.crop((crop, crop, w - crop, h - crop)).save(os.path.join(dst, os.path.basename(f)))
        if i % 5000 == 0: print(f"  {dst}: {i}/{len(files)}", flush=True)
    print(f"  {dst}: {len(files)} done", flush=True)
PY

echo "===== KID uncropped ====="
fidelity --gpu 0 --kid --batch-size 128 --input1 "$GEN" --input2 "$REF" \
    2>&1 | tee "results/$TAG/kid_border_uncropped.txt"

echo "===== KID cropped (${CROP}px) ====="
fidelity --gpu 0 --kid --batch-size 128 --input1 "$WORK/gen" --input2 "$WORK/ref" \
    2>&1 | tee "results/$TAG/kid_border_cropped.txt"

echo "===== verdict ====="
grep -H "kernel_inception_distance_mean" "results/$TAG"/kid_border_*.txt
echo "big drop when cropped = border artifact is a major KID contributor"
echo "note: cropped copies in $WORK (~300MB) — rm -rf border_test when done"
