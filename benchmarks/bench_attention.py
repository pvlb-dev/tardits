"""Benchmark the level-1 (complex) attention cost as a function of ``n_head``.

The linear fast-weight memory costs ``O(block_size * head_size**2)`` per head,
and ``head_size = n_embd / n_head``, so attention time scales with
``n_embd**2 / n_head``. More heads is ~1.8-2x faster at an essentially
unchanged parameter count, but it does change ``head_size`` (so runs are not
directly comparable).

Usage:
    python benchmarks/bench_attention.py
    python benchmarks/bench_attention.py -c examples/configs/rat.yaml
    python benchmarks/bench_attention.py --heads 6,12,14,21 --iters 3
"""

import argparse
import os
import sys
import time

import torch
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from tardits.models import ModelConfig, RAT  # noqa: E402
from tardits.modules.base import get_device  # noqa: E402


def _measure(model, idx, tgt, device, iters, warmup):
    """Return forward+backward milliseconds per iteration (best-of over iters)."""
    model.train()
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        _, loss = model(idx.to(device), tgt.to(device))
        loss.backward()

    times = []
    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        _, loss = model(idx.to(device), tgt.to(device))
        loss.backward()
        times.append(time.perf_counter() - t0)
    return min(times) * 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config", default="examples/configs/rat.yaml",
        help="YAML config to take the model shape from",
    )
    parser.add_argument(
        "--heads", default="6,12,14,21",
        help="comma-separated n_head values to sweep",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vocab-size", type=int, default=65,
        help="fixed vocab for stable timing (configs use char-level data)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]
    n_embd = int(mcfg["n_embd"])
    block = int(mcfg["block_size"])
    n_layer = int(mcfg["n_layer"])
    level = int(mcfg["level"])
    dropout = float(mcfg.get("dropout", 0.0))

    device = get_device(cfg.get("device", "auto"))
    torch.manual_seed(args.seed)
    idx = torch.randint(0, args.vocab_size, (args.batch_size, block))
    tgt = torch.randint(0, args.vocab_size, (args.batch_size, block))

    print(
        f"config={args.config} level={level} n_embd={n_embd} n_layer={n_layer} "
        f"block_size={block} batch={args.batch_size} device={device}"
    )
    header = f"{'n_head':>7} {'head_size':>10} {'hs^2':>5} {'params':>10} "
    header += f"{'ms/iter':>9} {'speedup':>8}"
    print(header)

    base = None
    for head in (int(h) for h in args.heads.split(",")):
        if n_embd % head != 0:
            print(f"{head:7d}  skipped: n_embd={n_embd} not divisible by n_head={head}")
            continue

        model_cfg = ModelConfig(
            level=level, block_size=block, vocab_size=args.vocab_size,
            n_head=head, n_embd=n_embd, n_layer=n_layer, dropout=dropout,
        )
        model = RAT(model_cfg).to(device)
        ms = _measure(model, idx, tgt, device, args.iters, args.warmup)
        params = sum(p.numel() for p in model.parameters())

        if base is None:
            base = ms
        head_size = n_embd // head
        print(
            f"{head:7d} {head_size:10d} {head_size ** 2:5d} {params:10,} "
            f"{ms:9.1f} {base / ms:7.2f}x"
        )


if __name__ == "__main__":
    main()
