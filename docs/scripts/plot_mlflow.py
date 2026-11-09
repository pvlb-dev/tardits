#!/usr/bin/env python
"""Export curves and a summary table from the local MLflow store.

Usage:
    .venv/bin/python docs/scripts/plot_mlflow.py \
        --experiment tardits-0.3.0-shakespeare \
        --out docs/assets/curves

Produces (in ``--out``):
    loss_curves.png      train/val loss vs step, one pair per run
    param_efficiency.png final val loss vs parameter count
    runtime.png          final val loss vs wall-clock minutes
    gamma_layers.png     layer-wise mean |gamma| heatmap (RAT runs)
    summary.csv          one row per finished run

The MLflow store is the repo-local ``sqlite:///mlflow.db`` by default.
"""

import argparse
import csv
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _metric(client, run_id, key):
    history = client.get_metric_history(run_id, key)
    return [(m.step, m.value) for m in history]


def _final(client, run_id, key):
    history = client.get_metric_history(run_id, key)
    return history[-1].value if history else None


def _num(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", default="tardits-0.3.0-shakespeare", help="MLflow experiment name"
    )
    parser.add_argument("--out", default="docs/assets/curves", help="output directory")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"experiment not found: {args.experiment}")

    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time ASC"],
        max_results=500,
    )
    runs = [r for r in runs if r.info.status == "FINISHED"]
    if not runs:
        raise SystemExit(f"no finished runs in {args.experiment}")

    rows = []
    for r in runs:
        params = r.data.params
        rows.append(
            {
                "run_id": r.info.run_id,
                "run_name": r.info.run_name or r.info.run_id[:8],
                "arch": params.get("arch", ""),
                "level": params.get("level", ""),
                "n_head": params.get("n_head", ""),
                "n_embd": params.get("n_embd", ""),
                "n_layer": params.get("n_layer", ""),
                "params": _num(params.get("num_params")),
                "train_loss": _final(client, r.info.run_id, "train loss"),
                "val_loss": _final(client, r.info.run_id, "val loss"),
                "minutes": round(
                    (r.info.end_time - r.info.start_time) / 60000.0, 1
                )
                if r.info.end_time
                else "",
            }
        )

    rows.sort(key=lambda row: (row["level"], row["params"] or math.inf))

    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # 1. Loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in runs:
        name = r.info.run_name or r.info.run_id[:8]
        val = _metric(client, r.info.run_id, "val loss")
        train = _metric(client, r.info.run_id, "train loss")
        if val:
            (line,) = ax.plot(
                [s for s, _ in val], [v for _, v in val], label=name, linewidth=1.8
            )
            if train:
                ax.plot(
                    [s for s, _ in train],
                    [v for _, v in train],
                    linestyle=":",
                    color=line.get_color(),
                    alpha=0.5,
                    linewidth=1.0,
                )
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"{args.experiment}: train (dotted) / val (solid)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "loss_curves.png"), dpi=150)
    plt.close(fig)

    # 2. Parameter-efficiency frontier
    points = [row for row in rows if row["params"] and row["val_loss"] is not None]
    if points:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for row in points:
            ax.scatter(row["params"], row["val_loss"], s=45)
            ax.annotate(
                row["run_name"],
                (row["params"], row["val_loss"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
            )
        ax.set_xlabel("trainable parameters")
        ax.set_ylabel("final val loss")
        ax.set_title("Parameter efficiency (lower-left is better)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "param_efficiency.png"), dpi=150)
        plt.close(fig)

    # 3. Val loss vs wall time
    timed = [row for row in rows if row["minutes"] != "" and row["val_loss"] is not None]
    if timed:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for row in timed:
            ax.scatter(row["minutes"], row["val_loss"], s=45)
            ax.annotate(
                row["run_name"],
                (row["minutes"], row["val_loss"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
            )
        ax.set_xlabel("wall-clock minutes (CPU)")
        ax.set_ylabel("final val loss")
        ax.set_title("Cost vs quality")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "runtime.png"), dpi=150)
        plt.close(fig)

    # 4. Layer-wise gamma heatmap. Prefer the latest level-1 (8-layer) RAT run;
    # otherwise use the latest run that logged gamma metrics.
    gamma_candidates = [
        r
        for r in runs
        if any(k.startswith("gamma/layer_") for k in r.data.metrics)
    ]
    level1 = [r for r in gamma_candidates if r.data.params.get("level") == "1"]
    gamma_run = (level1[-1] if level1 else gamma_candidates[-1]) if gamma_candidates else None
    if gamma_run is not None:
        n_layers = (
            max(
                int(k.split("/")[1].split("_")[1])
                for k in gamma_run.data.metrics
                if k.startswith("gamma/layer_")
            )
            + 1
        )
        histories = {
            i: _metric(client, gamma_run.info.run_id, f"gamma/layer_{i}/mean_abs")
            for i in range(n_layers)
        }
        steps = sorted({s for h in histories.values() for s, _ in h})
        matrix = []
        for i in range(n_layers):
            by_step = dict(histories[i])
            matrix.append([by_step.get(s, float("nan")) for s in steps])
        if steps:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            im = ax.imshow(
                matrix,
                aspect="auto",
                origin="lower",
                extent=[steps[0], steps[-1], -0.5, n_layers - 0.5],
                cmap="magma",
                interpolation="nearest",
            )
            ax.set_xlabel("step")
            ax.set_ylabel("RAT layer")
            ax.set_title(f"Mean |gamma| per layer - {gamma_run.info.run_name}")
            fig.colorbar(im, ax=ax, label="mean |gamma|")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, "gamma_layers.png"), dpi=150)
            plt.close(fig)

    print(f"wrote plots + summary.csv to {args.out}")
    for row in rows:
        print(
            f"{row['run_name']:28s} L{row['level']:>2s} "
            f"params={row['params']} val={row['val_loss']}"
        )


if __name__ == "__main__":
    main()
