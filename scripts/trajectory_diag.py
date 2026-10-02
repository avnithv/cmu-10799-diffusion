#!/usr/bin/env python
"""
Trajectory diagnostic: when during sampling do borders turn dark?

Samples a batch with return_trajectory=True and tracks mean border-frame vs
interior intensity at every reverse step. Distinguishes:
  - steady divergence from early on  -> static learned bias
  - tracking until mid-trajectory, then divergence that saturates
    -> mode commitment / compounding dynamics
Outputs a plot + CSV into results/<tag>/.

GPU recommended (a 2080Ti is fine):
    python scripts/trajectory_diag.py --checkpoint <ckpt.pt> --tag <tag>
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample import load_checkpoint
from src.methods import DDPM


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--num_steps", type=int, default=1000)
    ap.add_argument("--variance", default="beta", choices=["beta", "posterior"])
    ap.add_argument("--k", type=int, default=3, help="frame thickness in px")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, config, ema = load_checkpoint(args.checkpoint, device)
    ema.apply_shadow()
    method = DDPM.from_config(model, config, device)
    method.eval_mode()
    dc = config["data"]
    shape = (dc["channels"], dc["image_size"], dc["image_size"])

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        _, traj = method.sample(batch_size=args.num_samples, image_shape=shape,
                                num_steps=args.num_steps, variance=args.variance,
                                return_trajectory=True)
    traj = traj.cpu()                       # (T+1, B, C, H, W), values ~[-1,1]
    traj01 = ((traj + 1) / 2).clamp(0, 1)   # view in image space
    k = args.k
    frame = torch.cat([
        traj01[..., :k, :].flatten(2), traj01[..., -k:, :].flatten(2),
        traj01[..., k:-k, :k].flatten(2), traj01[..., k:-k, -k:].flatten(2)], dim=2)
    inner = traj01[..., k:-k, k:-k].flatten(2)
    frame_m = frame.mean(dim=(1, 2)).numpy()   # per trajectory step
    inner_m = inner.mean(dim=(1, 2)).numpy()

    run_out = os.path.join(args.outdir, args.tag)
    os.makedirs(run_out, exist_ok=True)
    with open(os.path.join(run_out, "trajectory_border.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["traj_index", "frame_mean", "interior_mean"])
        for i, (fm, im) in enumerate(zip(frame_m, inner_m)):
            wr.writerow([i, f"{fm:.5f}", f"{im:.5f}"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(frame_m))             # 0 = pure noise, last = final sample
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, inner_m, label=f"interior mean (inside {k}px)", lw=2)
    plt.plot(x, frame_m, label=f"border frame mean (outer {k}px)", lw=2)
    plt.plot(x, inner_m - frame_m, label="gap (interior − frame)", lw=1, ls="--")
    plt.xlabel(f"sampling progress (0 = x_T noise → {len(x)-1} = final)")
    plt.ylabel("mean intensity [0,1]")
    plt.title(f"Border vs interior along the sampling trajectory — {args.tag}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(run_out, "trajectory_border.png")
    plt.savefig(out, dpi=140)
    print(f"plot -> {out}")
    print(f"final: interior {inner_m[-1]:.3f}, frame {frame_m[-1]:.3f}, gap {inner_m[-1]-frame_m[-1]:.3f}")


if __name__ == "__main__":
    main()
