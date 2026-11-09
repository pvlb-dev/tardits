#!/usr/bin/env python
"""Analyse the learned RAT decay ``gamma`` from a checkpoint.

The fast-weight memory decays each element by ``|gamma|`` per token. This script
turns that into the data-porn views used in the v0.3.0 docs:

* per-layer |gamma| mean / spread / share above the explosion threshold 1.0
* the effective memory horizon ``1 / (1 - |gamma|)`` (tokens)
* complex-plane density of gamma (Re/Im)

Usage:
    .venv/bin/python docs/scripts/analyze_gamma.py \
        --checkpoint examples/saved/shakespeare/rat_l1_winner.pt \
        --out docs/assets/gamma
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from tardits.models import RAT  # noqa: E402


def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = RAT(payload["config"])
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def gamma_stats(gamma):
    abs_g = gamma.abs().flatten().numpy()
    phase = torch.angle(gamma).flatten().numpy()
    denom = 1.0 - abs_g
    # Avoid the divide-by-zero warning for |gamma| == 1; those elements have an
    # infinite horizon by definition.
    with np.errstate(divide="ignore", invalid="ignore"):
        horizon = np.where(abs_g < 0.999, 1.0 / denom, np.inf)
    return abs_g, phase, horizon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="docs/assets/gamma")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.checkpoint)

    per_layer = []
    for idx, block in enumerate(model.blocks):
        abs_g, phase, horizon = gamma_stats(block.sa.gamma.detach().cpu())
        finite_horizon = horizon[np.isfinite(horizon)]
        per_layer.append(
            {
                "layer": idx,
                "mean_abs": float(abs_g.mean()),
                "std_abs": float(abs_g.std()),
                "frac_gt_1": float((abs_g > 1.0).mean()),
                "mean_phase": float(phase.mean()),
                "median_horizon": float(np.median(finite_horizon))
                if finite_horizon.size
                else float("inf"),
                "p90_horizon": float(np.percentile(finite_horizon, 90))
                if finite_horizon.size
                else float("inf"),
                "abs": abs_g,
                "horizon": horizon,
            }
        )

    with open(os.path.join(args.out, "gamma_stats.csv"), "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(
            [
                "layer",
                "mean_abs",
                "std_abs",
                "frac_gt_1",
                "mean_phase",
                "median_horizon",
                "p90_horizon",
            ]
        )
        for row in per_layer:
            writer.writerow(
                [
                    row["layer"],
                    f"{row['mean_abs']:.6f}",
                    f"{row['std_abs']:.6f}",
                    f"{row['frac_gt_1']:.6f}",
                    f"{row['mean_phase']:.6f}",
                    f"{row['median_horizon']:.2f}",
                    f"{row['p90_horizon']:.2f}",
                ]
            )

    # 1. Per-layer decay and threshold share
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    layers = [row["layer"] for row in per_layer]
    means = [row["mean_abs"] for row in per_layer]
    stds = [row["std_abs"] for row in per_layer]
    ax1.bar(layers, means, yerr=stds, capsize=4, color="purple", alpha=0.75)
    ax1.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="|γ| = 1")
    ax1.set_xlabel("RAT layer")
    ax1.set_ylabel("mean |γ| ± std")
    ax1.set_title("Learned decay per layer")
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot(
        layers,
        [row["frac_gt_1"] * 100 for row in per_layer],
        color="black",
        marker="o",
        linewidth=1.2,
        label="% |γ| > 1",
    )
    ax2.set_ylabel("% of weights above 1")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "gamma_layers.png"), dpi=150)
    plt.close(fig)

    # 2. Effective memory horizon (tokens)
    finite = np.concatenate(
        [row["horizon"][np.isfinite(row["horizon"])] for row in per_layer]
    )
    finite = finite[finite > 0]
    if finite.size:
        fig, ax = plt.subplots(figsize=(9, 5))
        bins = np.logspace(0, np.log10(max(10.0, finite.max())), 60)
        ax.hist(finite, bins=bins, color="teal", alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("effective memory horizon (tokens, 1/(1-|γ|))")
        ax.set_ylabel("count")
        ax.set_title("How far the fast weights remember")
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "gamma_horizon.png"), dpi=150)
        plt.close(fig)

    # 3. Complex-plane density
    real = np.concatenate(
        [block.sa.gamma.detach().cpu().real.flatten().numpy() for block in model.blocks]
    )
    imag = np.concatenate(
        [block.sa.gamma.detach().cpu().imag.flatten().numpy() for block in model.blocks]
    )
    fig, ax = plt.subplots(figsize=(6.5, 6))
    hb = ax.hexbin(real, imag, gridsize=80, cmap="twilight", mincnt=1)
    circle = Circle((0, 0), 1.0, fill=False, color="red", linestyle="--", lw=1.2)
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(γ)")
    ax.set_ylabel("Im(γ)")
    ax.set_title("γ in the complex plane (red = unit circle)")
    fig.colorbar(hb, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "gamma_complex_plane.png"), dpi=150)
    plt.close(fig)

    print(f"wrote gamma analysis to {args.out}")
    for row in per_layer:
        print(
            f"layer {row['layer']:2d}: |γ|={row['mean_abs']:.4f} "
            f"±{row['std_abs']:.4f}  >1={row['frac_gt_1'] * 100:5.2f}%  "
            f"median horizon={row['median_horizon']:.1f} tokens"
        )


if __name__ == "__main__":
    main()
