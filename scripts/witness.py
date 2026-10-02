#!/usr/bin/env python
"""
Witness-score analysis: rank generated samples by how much they hurt KID.

KID is three averages of the kernel k(x,y) = (x.y/2048 + 1)^3 over Inception
features. The per-sample "witness" score of a generated sample x is

    w(x) = mean_{y in real} k(x, y)  -  mean_{x' in gen, x' != x} k(x, x')

Low w(x) = the metric considers x atypical of the real distribution (and/or
over-typical of the model's own quirks). The bottom-ranked tiles answer, in
one look, whether a KID floor is a tail problem (a few terrible samples), a
diversity problem (near-duplicates inflating gen-gen similarity), or a global
shift (bottom tiles look unremarkable).

Outputs: results/witness_<tag>.csv (every file, scored, worst first) and two
grids in results/grids/: witness_worst_<tag>.png and witness_best_<tag>.png.

Usage (GPU strongly recommended; CPU works but is slow):
    python scripts/witness.py --generated logs/<run>/checkpoints/samples/generated --tag ddpm_resume1e5
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_grid import build_grid, COLORS  # reuse the divider-grid renderer


def list_images(d, limit=None, rng=None):
    files = sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))
    if not files:
        raise FileNotFoundError(f"no images in {d}")
    if limit and len(files) > limit:
        idx = (rng or np.random.default_rng(0)).choice(len(files), size=limit, replace=False)
        files = [files[i] for i in sorted(idx)]
    return files


def extract_features(files, device, batch_size=64):
    """2048-d inception-v3-compat features — the same extractor torch-fidelity
    uses for KID, so the ranking sees exactly what the metric sees."""
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    ext = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(files), batch_size):
            batch = np.stack([np.asarray(Image.open(f).convert("RGB"), np.uint8)
                              for f in files[i:i + batch_size]])
            t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)  # uint8 BCHW
            out = ext(t)
            f = out["2048"] if isinstance(out, dict) else out[0]
            feats.append(f.cpu().double().numpy())
            print(f"\r  features {min(i + batch_size, len(files))}/{len(files)}", end="", flush=True)
    print()
    return np.concatenate(feats)


def witness_scores(gen, real):
    """w(x) per generated sample; kernel k(a,b) = (a.b/d + 1)^3, d=2048."""
    d = gen.shape[1]
    k_cross = (gen @ real.T / d + 1.0) ** 3          # (n_gen, n_real)
    k_gen = (gen @ gen.T / d + 1.0) ** 3             # (n_gen, n_gen)
    np.fill_diagonal(k_gen, 0.0)                     # exclude self-similarity
    return k_cross.mean(1) - k_gen.sum(1) / (len(gen) - 1)


def tiles_from(files):
    imgs = [Image.open(f).convert("RGB") for f in files]
    size = imgs[0].size
    return np.stack([np.asarray(im if im.size == size else im.resize(size), np.uint8)
                     for im in imgs])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generated", required=True, help="dir of generated PNGs (e.g. the eval's generated/)")
    ap.add_argument("--reference", default=None, help="dir of reference images (default: auto-find data/celeba/*/images)")
    ap.add_argument("--tag", required=True, help="label for output files")
    ap.add_argument("--num_reference", type=int, default=5000,
                    help="reference subsample for the cross term (default 5000; plenty for a stable mean)")
    ap.add_argument("--grid", default="5x5", help="size of the worst/best grids (default 5x5)")
    ap.add_argument("--color", default="magenta")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ref_dir = args.reference
    if ref_dir is None:
        hits = glob.glob("data/celeba/*/images") + glob.glob("data/celeba/images")
        if not hits:
            sys.exit("could not auto-find reference images; pass --reference")
        ref_dir = hits[0]

    gen_files = list_images(args.generated)
    ref_files = list_images(ref_dir, limit=args.num_reference)
    print(f"generated: {len(gen_files)} from {args.generated}")
    print(f"reference: {len(ref_files)} (subsampled) from {ref_dir}")

    gen_feat = extract_features(gen_files, device)
    ref_feat = extract_features(ref_files, device)
    w = witness_scores(gen_feat, ref_feat)

    order = np.argsort(w)  # ascending: worst first
    # per-run folder: results/<tag>/witness{.csv,_worst.png,_best.png}
    run_out = os.path.join(args.outdir, args.tag)
    os.makedirs(run_out, exist_ok=True)
    csv_path = os.path.join(run_out, "witness.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "file", "witness"])
        for r, i in enumerate(order):
            wr.writerow([r, os.path.basename(gen_files[i]), f"{w[i]:.6f}"])

    rows, cols = (int(v) for v in args.grid.lower().split("x"))
    n = rows * cols
    color = COLORS.get(args.color, (255, 0, 255))
    for name, idx in (("worst", order[:n]), ("best", order[::-1][:n])):
        canvas = build_grid(tiles_from([gen_files[i] for i in idx]), rows, cols, 4, color)
        out = os.path.join(run_out, f"witness_{name}.png")
        Image.fromarray(canvas).save(out)
        print(f"{name} grid -> {out}")

    print(f"scores -> {csv_path}")
    print(f"witness: mean {w.mean():.4f} | worst {w[order[0]]:.4f} ({os.path.basename(gen_files[order[0]])}) "
          f"| best {w[order[-1]]:.4f}")
    print(f"bottom-5%% mean {np.sort(w)[:max(1,len(w)//20)].mean():.4f} vs overall {w.mean():.4f} "
          f"-> big gap = tail problem, small gap = global shift")


if __name__ == "__main__":
    main()
