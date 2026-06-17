#!/usr/bin/env python3
"""
Plot reconstruction attention leakage curves from one or more runs.

Each run must have produced an attn_alignment.json file (requires
track_recon_alignment = true in config.ini).

Usage:
  # Single run — per-image curves + averaged curve
  python plot_attn_alignment.py output/007/attn_alignment.json

  # Compare two runs (e.g. full method vs ablation)
  python plot_attn_alignment.py \
      output/007/attn_alignment.json --label "Full method" \
      output/008/attn_alignment.json --label "Ablation (no recon attn)"

  # Save to a specific directory
  python plot_attn_alignment.py output/007/attn_alignment.json --out-dir figures/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_run(path: Path):
    with open(path) as f:
        records = json.load(f)
    max_pos = max(len(r["steps"]) for r in records)
    leak_by_pos  = [[] for _ in range(max_pos)]
    prog_by_pos  = [[] for _ in range(max_pos)]
    all_curves   = []
    for rec in records:
        curve = [s["leakage"] for s in rec["steps"]]
        all_curves.append(curve)
        for pos, s in enumerate(rec["steps"]):
            leak_by_pos[pos].append(s["leakage"])
            prog_by_pos[pos].append(s["progress"])
    means = np.array([np.mean(v) for v in leak_by_pos if v])
    stds  = np.array([np.std(v)  for v in leak_by_pos if v])
    xs    = np.array([np.mean(p) for p in prog_by_pos if p])
    return xs, means, stds, all_curves, records


def plot_single(path: Path, out_dir: Path):
    """Per-image curves + averaged curve for a single run."""
    xs, means, stds, all_curves, records = load_run(path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-image curves
    ax = axes[0]
    for i, (rec, curve) in enumerate(zip(records, all_curves)):
        n = len(curve)
        x = xs[:n] if n <= len(xs) else np.linspace(xs[0], xs[-1], n)
        ax.plot(x, curve, alpha=0.4, linewidth=0.9)
    ax.set_xlabel("Denoising progress", fontsize=11)
    ax.set_ylabel("Leakage (attention outside concept mask)", fontsize=11)
    ax.set_title(f"Per-image leakage curves\n({len(records)} images)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: mean ± std
    ax = axes[1]
    ax.plot(xs, means, linewidth=2.5, label="mean")
    ax.fill_between(xs, means - stds, means + stds, alpha=0.25, label="±1 std")
    ax.set_xlabel("Denoising progress", fontsize=11)
    ax.set_ylabel("Leakage (attention outside concept mask)", fontsize=11)
    ax.set_title("Averaged leakage curve", fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)

    fig.suptitle("Reconstruction attention leakage over denoising steps", fontsize=12, fontweight="bold")
    plt.tight_layout()

    stem = "attn_alignment_single"
    for ext in ("png", "pdf"):
        out = out_dir / f"{stem}.{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close()


def plot_comparison(paths, labels, out_dir: Path):
    """Averaged leakage curves for multiple runs overlaid."""
    COLORS = ["#4C72B0", "#C44E52", "#55A868", "#DD8452", "#8172B3"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (path, label) in enumerate(zip(paths, labels)):
        xs, means, stds, _, records = load_run(path)
        color = COLORS[i % len(COLORS)]
        ax.plot(xs, means, linewidth=2.5, color=color, label=f"{label} (n={len(records)})")
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel("Denoising progress (0 = start, 1 = clean image)", fontsize=11)
    ax.set_ylabel("Leakage  (attention outside concept mask)", fontsize=11)
    ax.set_title("Reconstruction attention leakage — method comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10)
    plt.tight_layout()

    stem = "attn_alignment_comparison"
    for ext in ("png", "pdf"):
        out = out_dir / f"{stem}.{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", type=Path,
                        help="One or more attn_alignment.json files")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Display labels for each run (same order as json_files)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for plots (default: same dir as first JSON)")
    args = parser.parse_args()

    out_dir = args.out_dir or args.json_files[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels or [p.parent.name for p in args.json_files]
    if len(labels) < len(args.json_files):
        labels += [p.parent.name for p in args.json_files[len(labels):]]

    if len(args.json_files) == 1:
        plot_single(args.json_files[0], out_dir)
    else:
        plot_comparison(args.json_files, labels, out_dir)


if __name__ == "__main__":
    main()
