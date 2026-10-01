#!/usr/bin/env python
"""
Generate comparison sample grids from past run checkpoints.

One PNG per run, tiles separated by a bright divider color, and — key for
comparison — the SAME seed is used for every run, so tile (i,j) starts from
the identical x_T noise across models: differences you see are model
differences, not luck of the draw.

Usage (on a GPU node):
    python scripts/sample_grid.py \
        --runs logs/ddpm_20260917_205522 logs/ddpm_20260927_222014 \
        --grid 5x5 --steps 1000 --seed 0 --color magenta --outdir results/grids

--runs accepts run directories (uses checkpoints/ddpm_final.pt inside) or
direct paths to .pt files. Output: results/grids/grid_<run>_<RxC>_<steps>steps.png
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

# repo root on sys.path so this works from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample import load_checkpoint          # model + config + EMA plumbing
from src.methods import DDPM
from src.data import unnormalize

COLORS = {
    "magenta": (255, 0, 255),
    "lime": (60, 255, 60),
    "cyan": (0, 230, 255),
    "orange": (255, 140, 0),
    "yellow": (255, 220, 0),
    "white": (255, 255, 255),
}


def resolve_checkpoint(path: str) -> str:
    """Accept a run dir or a .pt path; return the .pt path."""
    if path.endswith(".pt"):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no such checkpoint: {path}")
        return path
    cand = os.path.join(path, "checkpoints", "ddpm_final.pt")
    if os.path.isfile(cand):
        return cand
    # fall back to the newest .pt in the run's checkpoints dir
    ckpt_dir = os.path.join(path, "checkpoints")
    pts = sorted(
        (os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")),
        key=os.path.getmtime,
    ) if os.path.isdir(ckpt_dir) else []
    if pts:
        print(f"  note: no ddpm_final.pt in {path}, using newest: {pts[-1]}")
        return pts[-1]
    raise FileNotFoundError(f"no checkpoints found under: {path}")


def run_tag(ckpt_path: str) -> str:
    """A short name for output files: the run dir's name, or the .pt stem."""
    parts = os.path.normpath(ckpt_path).split(os.sep)
    for p in parts:
        if p.startswith("ddpm_2"):
            return p
    return os.path.splitext(os.path.basename(ckpt_path))[0]


def build_grid(tiles: np.ndarray, rows: int, cols: int, pad: int, color) -> np.ndarray:
    """tiles: (N, H, W, 3) uint8 -> canvas with colored dividers (and border)."""
    n, h, w, _ = tiles.shape
    canvas = np.zeros((rows * h + pad * (rows + 1), cols * w + pad * (cols + 1), 3), np.uint8)
    canvas[:, :] = color
    for i in range(min(n, rows * cols)):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = tiles[i]
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs (logs/ddpm_<ts>) and/or .pt checkpoint paths")
    ap.add_argument("--grid", default="10x10", help="RxC, e.g. 10x10 (default)")
    ap.add_argument("--steps", type=int, default=1000, help="sampling steps (default 1000)")
    ap.add_argument("--seed", type=int, default=0,
                    help="same seed is reused for every run -> comparable tiles (default 0)")
    ap.add_argument("--color", default="magenta",
                    help=f"divider color: {', '.join(COLORS)} or R,G,B (default magenta)")
    ap.add_argument("--pad", type=int, default=4, help="divider thickness in px (default 4)")
    ap.add_argument("--batch_size", type=int, default=25,
                    help="samples generated per chunk (default 25; keeps 10x10 grids within a 2080Ti's 11GB)")
    ap.add_argument("--variance", default="beta", choices=["beta", "posterior"])
    ap.add_argument("--no_ema", action="store_true", help="use raw training weights")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="results/grids")
    args = ap.parse_args()

    rows, cols = (int(v) for v in args.grid.lower().split("x"))
    n = rows * cols
    color = COLORS[args.color] if args.color in COLORS else tuple(int(v) for v in args.color.split(","))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    for run in args.runs:
        ckpt = resolve_checkpoint(run)
        tag = run_tag(ckpt)
        print(f"== {tag}: {ckpt}")

        model, config, ema = load_checkpoint(ckpt, device)
        if not args.no_ema:
            ema.apply_shadow()
        method = DDPM.from_config(model, config, device)
        method.eval_mode()

        dc = config["data"]
        image_shape = (dc["channels"], dc["image_size"], dc["image_size"])

        torch.manual_seed(args.seed)           # identical x_T across runs
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        # chunked generation: one batch of rows*cols OOMs small GPUs. Note the
        # noise draws depend on the chunking, so keep --batch_size constant
        # across runs you want tile-comparable (the default always is).
        with torch.no_grad():
            chunks, remaining = [], n
            while remaining > 0:
                b = min(args.batch_size, remaining)
                chunks.append(method.sample(batch_size=b, image_shape=image_shape,
                                            num_steps=args.steps, variance=args.variance))
                remaining -= b
            samples = torch.cat(chunks)

        imgs = unnormalize(samples.detach().cpu()).clamp(0, 1)
        tiles = (imgs.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
        canvas = build_grid(tiles, rows, cols, args.pad, color)

        out = os.path.join(args.outdir, f"grid_{tag}_{rows}x{cols}_{args.steps}steps.png")
        Image.fromarray(canvas).save(out)
        print(f"   -> {out}")

        if not args.no_ema:
            ema.restore()                      # leave the model clean before the next run


if __name__ == "__main__":
    main()
