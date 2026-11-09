#!/usr/bin/env python
"""Graceful-degradation probe: remove residual channels and measure val loss.

Eval-only. Compares the 11M RAT against the parameter-matched 11M RealGPT on
held-out FineWeb tokens. A "channel" is one residual-stream dimension
(``n_embd``); the same *fraction* of channels is removed from each model, so the
comparison is at matched relative width rather than matched parameter count.

For each fraction and ordering, the selected residual channels are zeroed:
- at the token embedding output, and
- at every block's output,
i.e. those dimensions carry no signal through the stack. Orderings:
- ``magnitude``: largest mean |activation| first (calibrated on val batches)
- ``random``: seeded uniform subsets, averaged over ``--random-seeds``

The thesis under test: if RAT's complex interference stores information more
distributed, its degradation curve should be flatter than RealGPT's. If the
curves overlap, capacity is just as localised. No training, no claims beyond
this test.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from tardits.models import RAT, RealGPT  # noqa: E402


def load_model(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    cls = RealGPT if cfg.level == 0 else RAT
    model = cls(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, cfg


def make_batches(data: torch.Tensor, batch_size: int, block_size: int, n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(data) - block_size - 1, (n, batch_size), generator=g)
    return [
        (
            torch.stack([data[s : s + block_size] for s in row]),
            torch.stack([data[s + 1 : s + block_size + 1] for s in row]),
        )
        for row in starts
    ]


@torch.no_grad()
def eval_loss(model, batches) -> float:
    losses = []
    for xb, yb in batches:
        _, loss = model(xb, yb)
        losses.append(float(loss))
    return float(np.mean(losses))


def embedding_module(model):
    for name in ("token_embedding", "token_embedding_table"):
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError("no token embedding found")


def _zero_hook(indices: torch.Tensor):
    def hook(module, inputs, output):
        def zero(x):
            # residual channel dim is 2 for both [B,T,C] and hyper [B,T,C,H]
            x = x.clone()
            x[:, :, indices] = 0
            return x
        if isinstance(output, tuple):
            return (zero(output[0]), *output[1:])
        return zero(output)
    return hook


@torch.no_grad()
def activation_magnitude(model, batches) -> torch.Tensor:
    """Mean |residual activation| per channel, averaged over blocks/batches."""
    sums = None
    count = 0
    handles = []

    def capture(module, inputs, output):
        nonlocal sums, count
        x = output[0] if isinstance(output, tuple) else output
        reduce_dims = tuple(d for d in range(x.dim()) if d != 2)
        v = x.detach().abs().mean(dim=reduce_dims)
        sums = v if sums is None else sums + v
        count += 1

    handles.append(embedding_module(model).register_forward_hook(capture))
    for block in model.blocks:
        handles.append(block.register_forward_hook(capture))
    for xb, _ in batches:
        model(xb)
    for h in handles:
        h.remove()
    return sums / max(count, 1)


@torch.no_grad()
def eval_with_zeroed(model, batches, indices: torch.Tensor) -> float:
    if len(indices) == 0:
        return eval_loss(model, batches)
    handles = [embedding_module(model).register_forward_hook(_zero_hook(indices))]
    for block in model.blocks:
        handles.append(block.register_forward_hook(_zero_hook(indices)))
    try:
        return eval_loss(model, batches)
    finally:
        for h in handles:
            h.remove()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rat", default=".cache/kaggle/rat22m11m-out/rat_l1_11m.pt")
    ap.add_argument("--real", default=".cache/kaggle/real11m-rat6m-out/real11m.pt")
    ap.add_argument("--val-bin", default=".cache/kaggle/prep-out/fineweb/val.bin")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--random-seeds", type=int, default=3)
    ap.add_argument("--fractions", default="0.01,0.05,0.10,0.25,0.50")
    ap.add_argument("--out", default="docs/assets/curves/graceful_degradation.csv")
    ap.add_argument("--plot", default="docs/assets/curves/graceful_degradation.png")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    data = torch.from_numpy(np.fromfile(args.val_bin, dtype=np.uint16)).long()
    batches = make_batches(data, args.batch_size, args.block_size, args.eval_batches, 20261031)
    calib = make_batches(data, args.batch_size, args.block_size, args.calib_batches, 20261031 + 1)
    fractions = [float(x) for x in args.fractions.split(",")]

    rows = []
    for label, path in (("rat11m", args.rat), ("real11m", args.real)):
        model, cfg = load_model(path)
        n_embd = int(cfg.n_embd)
        intact = eval_loss(model, batches)
        mags = activation_magnitude(model, calib)
        print(f"\n=== {label}: {type(model).__name__} n_embd={n_embd} params={model.get_num_params():,} intact={intact:.4f}")
        for frac in fractions:
            k = int(round(frac * n_embd))
            if k <= 0:
                continue
            # magnitude order: largest activations first
            order = torch.argsort(mags, descending=True)[:k]
            loss = eval_with_zeroed(model, batches, order)
            rows.append(dict(model=label, arch=type(model).__name__, n_embd=n_embd,
                             params=model.get_num_params(), order="magnitude", fraction=frac,
                             removed=k, val_loss=round(loss, 6), delta=round(loss - intact, 6)))
            print(f"  magnitude {frac:5.2f}  k={k:3d}  val={loss:.4f}  delta={loss-intact:+.4f}")
            # random subsets, average over seeds
            rlosses = []
            for s in range(args.random_seeds):
                g = torch.Generator().manual_seed(1000 + s)
                idx = torch.randperm(n_embd, generator=g)[:k]
                rlosses.append(eval_with_zeroed(model, batches, idx))
            rmean = float(np.mean(rlosses))
            rows.append(dict(model=label, arch=type(model).__name__, n_embd=n_embd,
                             params=model.get_num_params(), order="random", fraction=frac,
                             removed=k, val_loss=round(rmean, 6), delta=round(rmean - intact, 6)))
            print(f"  random    {frac:5.2f}  k={k:3d}  val={rmean:.4f}  delta={rmean-intact:+.4f} (mean of {args.random_seeds})")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")

    # plot delta from intact; solid=magnitude, dashed=random
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.2))
    styles = {"rat11m": ("RAT 11.2M", "tab:green"), "real11m": ("RealGPT 11.1M", "tab:blue")}
    for label, (name, color) in styles.items():
        for order, ls in (("magnitude", "-"), ("random", "--")):
            pts = [r for r in rows if r["model"] == label and r["order"] == order]
            xs = [0.0] + [r["fraction"] for r in pts]
            ys = [0.0] + [r["delta"] for r in pts]
            ax.plot(xs, ys, ls, marker="o", color=color, label=f"{name} {order}")
    ax.set_xlabel("fraction of residual channels zeroed")
    ax.set_ylabel("val loss increase (nats)")
    ax.set_title("Graceful degradation - 11M RAT vs 11M RealGPT")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=150); plt.close(fig)
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
