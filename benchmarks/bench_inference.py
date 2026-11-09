#!/usr/bin/env python
"""Measure decode speed vs context length for RealGPT (KV-cached) vs RAT.

This is the *fair* comparison: RealGPT uses a KV cache (prefill chunked by
``block_size``, then O(T) attention work per new token), not the uncached
full-window ``generate`` path. RAT keeps a recurrent fast-weight state and does
O(1) state work per token, so its decode speed should stay flat as the context
grows while the cached real model still pays O(T) per token.

Long-context rows (> ``block_size``, the 128-token training window) are
extrapolation: this measures *speed*, not quality. For retrieval quality beyond
the window see the NIAH pilot.

The timed loop excludes prompt prefill for both: RealGPT fills its cache once
(chunked), RAT warms its recurrent state with a chunked scan. Timing is the minimum of 12 repeats (robust to
background host load); prefill is reported separately - it is O(T) sequential for RAT and O(T^2) parallel-ish for RealGPT,
which is itself an interesting long-context trade-off. The reported tokens/s
therefore measures generation, not prompt ingestion.

Usage:
    .venv/bin/python benchmarks/bench_inference.py
    .venv/bin/python benchmarks/bench_inference.py --prompt-lengths 128,512,1024
    .venv/bin/python benchmarks/bench_inference.py --new-tokens 16 --repeats 3

Output:
    docs/assets/curves/inference_scaling.csv
    docs/assets/curves/inference_scaling.png
"""

import argparse
import csv
import os
import statistics
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from tardits.models import KVCache, RAT, RealGPT  # noqa: E402

DEFAULT_MODELS = [
    ("real", "examples/saved/shakespeare/real_baseline.pt"),
    ("rat-l1", "examples/saved/shakespeare/rat_l1_h21_5k.pt"),
    ("rat-l2", "examples/saved/shakespeare/rat_l2_h16_l5_5k.pt"),
]


def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    cls = RealGPT if config.level == 0 else RAT
    model = cls(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config


def _sample_next(logits, temp=1.0, top_k=10):
    logits = logits / temp
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float("Inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_decode(model, vocab_size, prompt_len, new_tokens, repeats, device):
    """Return (prefill_seconds, decode_seconds, tokens/s) - minimum of repeats.

    The decode timing excludes prefill; ``prefill_seconds`` is measured on the
    same fresh prompt so both costs are visible. Using the minimum (rather than
    the median) keeps the number close to the uncontended speed on this shared
    desktop host; the wall-clock spread is reported in the CSV as the max.
    """
    prefill_times, decode_times = [], []
    for _ in range(repeats):
        torch.manual_seed(1337)
        prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)

        if isinstance(model, RAT):
            with torch.no_grad():
                _sync(device)
                t0 = time.perf_counter()
                # Vectorised scan prefill (not token-by-token: see issue #21).
                logits, states = model.prefill(prompt)
                _sync(device)
                prefill_times.append(time.perf_counter() - t0)
                token = _sample_next(logits[:, -1, :])
            pos = prompt_len
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(new_tokens):
                    logits, states = model.step(token, states, pos)
                    token = _sample_next(logits[:, -1, :])
                    pos += 1
                _sync(device)
            decode_times.append(time.perf_counter() - t0)
        else:
            # Fair RealGPT decode: chunked KV-cached prefill, then one cached
            # step/token. ``prefill`` grows the RoPE cache past block_size.
            caches = [KVCache() for _ in model.blocks]
            with torch.no_grad():
                _sync(device)
                t0 = time.perf_counter()
                logits = model.prefill(prompt, caches)
                _sync(device)
                prefill_times.append(time.perf_counter() - t0)
                token = _sample_next(logits[:, -1, :])
            idx = torch.cat((prompt, token), dim=1)
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(new_tokens):
                    pos = idx.shape[1] - 1
                    logits, _ = model(token, kv_cache=caches, start_pos=pos)
                    token = _sample_next(logits[:, -1, :])
                    idx = torch.cat((idx, token), dim=1)
                _sync(device)
            decode_times.append(time.perf_counter() - t0)

    prefill_seconds = min(prefill_times)
    seconds = min(decode_times)
    return prefill_seconds, seconds, new_tokens / seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(f"{name}={path}" for name, path in DEFAULT_MODELS),
        help="comma-separated name=checkpoint pairs",
    )
    parser.add_argument(
        "--prompt-lengths",
        default="16,64,128,256,512,1024,2048",
        help="comma-separated lengths; >128 is beyond the training window",
    )
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--out", default="docs/assets/curves/inference_scaling.csv"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")

    specs = []
    for pair in args.models.split(","):
        name, path = pair.split("=", 1)
        model, config = load_model(path)
        model.to(device)
        specs.append((name, model, config))
        # warmup
        model.generate(
            torch.randint(0, config.vocab_size, (1, 8)), max_new_tokens=4, top_k=10
        )

    rows = []
    for name, model, config in specs:
        params = model.get_num_params()
        print(f"\n{name}: {type(model).__name__} level {config.level} ({params:,} params)")
        for prompt_len in (int(x) for x in args.prompt_lengths.split(",")):
            prefill_s, seconds, tps = time_decode(
                model, config.vocab_size, prompt_len, args.new_tokens,
                args.repeats, device
            )
            rows.append(
                {
                    "model": name,
                    "arch": type(model).__name__,
                    "level": config.level,
                    "params": params,
                    "prompt_len": prompt_len,
                    "new_tokens": args.new_tokens,
                    "prefill_seconds": f"{prefill_s:.6f}",
                    "seconds": f"{seconds:.6f}",
                    "decode_tokens_per_sec": f"{tps:.2f}",
                }
            )
            print(
                f"  prompt {prompt_len:4d}: prefill {prefill_s * 1000:7.1f} ms  "
                f"decode {seconds * 1000:6.1f} ms  {tps:8.2f} tokens/s"
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # Decode speed (left) and prefill cost (right) vs context length.
    fig, (ax, ax_pre) = plt.subplots(1, 2, figsize=(13, 5.5))
    for name, _, _ in specs:
        points = [r for r in rows if r["model"] == name]
        lengths = [r["prompt_len"] for r in points]
        ax.plot(
            lengths,
            [float(r["decode_tokens_per_sec"]) for r in points],
            marker="o",
            label=name,
        )
        ax_pre.plot(
            lengths,
            [float(r["prefill_seconds"]) * 1000 for r in points],
            marker="o",
            label=name,
        )

    for axis in (ax, ax_pre):
        axis.set_xscale("log", base=2)
        axis.set_xlabel("context length (tokens, log scale)")
        axis.axvline(
            128,
            color="grey",
            linestyle="--",
            alpha=0.7,
            label="training window (128)",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    ax.set_ylabel("decode speed (tokens/s, prefill excluded)")
    ax.set_title(f"Decode speed vs context length ({device.type.upper()})")
    ax_pre.set_ylabel("prefill time (ms)")
    ax_pre.set_title("Prompt prefill cost")
    fig.suptitle("KV-cached RealGPT vs RAT: short vs long context", fontsize=11)
    fig.tight_layout()
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    print(f"\nwrote {args.out} and {png}")


if __name__ == "__main__":
    main()
