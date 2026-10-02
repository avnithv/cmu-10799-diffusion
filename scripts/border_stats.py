#!/usr/bin/env python
"""
Border statistics for sample sets: quantifies the black-border artifact.

Per image: mean intensity of the outer k-px frame vs the interior.
Reports per directory: frame/interior ratio, full black-frame rate, and
one-sided black-band rate (the signature of translate-augmentation fill).

CPU-only, no model needed. Compare any number of dirs side by side:
    python scripts/border_stats.py logs/<run>/checkpoints/samples/generated \
        witness_samples/baseline --reference
(--reference appends the real images dir as the control row.)
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image


def stats_for_dir(d, k=3, limit=2000):
    files = sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))
    if not files:
        raise FileNotFoundError(f"no images in {d}")
    if len(files) > limit:
        idx = np.random.default_rng(0).choice(len(files), size=limit, replace=False)
        files = [files[i] for i in sorted(idx)]
    frames, inners, full_frame, one_band = [], [], 0, 0
    for f in files:
        a = np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0
        sides = [a[:k], a[-k:], a[k:-k, :k], a[k:-k, -k:]]
        frame = np.concatenate([s.ravel() for s in sides])
        inner = a[k:-k, k:-k]
        fm, im = frame.mean(), inner.mean()
        frames.append(fm); inners.append(im)
        if fm < 0.15 and im > fm + 0.1:
            full_frame += 1
        elif any(s.mean() < 0.10 and im > s.mean() + 0.15 for s in sides):
            one_band += 1
    n = len(files)
    return dict(n=n, frame=np.mean(frames), inner=np.mean(inners),
                ratio=float(np.mean(np.array(frames) / (np.array(inners) + 1e-6))),
                full_pct=100 * full_frame / n, band_pct=100 * one_band / n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="sample directories to analyze")
    ap.add_argument("--reference", action="store_true",
                    help="also analyze the real reference images as a control row")
    ap.add_argument("--k", type=int, default=3, help="frame thickness in px (default 3)")
    args = ap.parse_args()

    dirs = list(args.dirs)
    if args.reference:
        hits = glob.glob("data/celeba/*/images") + glob.glob("dataset/*/images")
        if hits:
            dirs.append(hits[0])

    print(f"{'directory':55s} {'n':>5s} {'frame':>6s} {'inner':>6s} {'ratio':>6s} {'frame%':>7s} {'band%':>6s}")
    for d in dirs:
        s = stats_for_dir(d, k=args.k)
        print(f"{d[:55]:55s} {s['n']:5d} {s['frame']:6.3f} {s['inner']:6.3f} "
              f"{s['ratio']:6.2f} {s['full_pct']:7.1f} {s['band_pct']:6.1f}")
    print("\nratio ~1.0 + frame%<~7 = matches real data; low ratio / high %s = border artifact")


if __name__ == "__main__":
    main()
