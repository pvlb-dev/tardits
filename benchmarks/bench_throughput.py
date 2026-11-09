#!/usr/bin/env python
"""Batch throughput: KV-cached RealGPT vs recurrent RAT.

The single-stream benchmark hides the asymptotic difference on a GPU. This one
feeds B independent prompts through prefill + generate and reports decode
throughput (B * new_tokens / decode_seconds), with prefill throughput and peak
CUDA memory when available.
"""
import argparse
import csv
import os
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


def run_once(model, config, batch, prompt_len, new_tokens, device):
    torch.manual_seed(1337)
    prompt = torch.randint(0, config.vocab_size, (batch, prompt_len), device=device)
    if isinstance(model, RAT):
        with torch.no_grad():
            _sync(device)
            t0 = time.perf_counter()
            logits, states = model.prefill(prompt)
            _sync(device)
            prefill_s = time.perf_counter() - t0
            token = _sample_next(logits[:, -1, :])
            _sync(device)
            t0 = time.perf_counter()
            for _ in range(new_tokens):
                logits, states = model.step(token, states, prompt_len)
                token = _sample_next(logits[:, -1, :])
                prompt_len += 1
            idx = None
            decode_s = None
    else:
        caches = [KVCache() for _ in model.blocks]
        with torch.no_grad():
            _sync(device)
            t0 = time.perf_counter()
            logits = model.prefill(prompt, caches)
            _sync(device)
            prefill_s = time.perf_counter() - t0
            token = _sample_next(logits[:, -1, :])
            idx = torch.cat((prompt, token), dim=1)
            _sync(device)
            t0 = time.perf_counter()
            for _ in range(new_tokens):
                pos = idx.shape[1] - 1
                logits, _ = model(token, kv_cache=caches, start_pos=pos)
                token = _sample_next(logits[:, -1, :])
                idx = torch.cat((idx, token), dim=1)
            decode_s = None
    _sync(device)
    decode_s = time.perf_counter() - t0
    return prefill_s, decode_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="name=checkpoint,...")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--batch-sizes", default="1,4,8")
    ap.add_argument("--prompt-lengths", default="128,512,2048")
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="docs/assets/curves/throughput_scaling.csv")
    args = ap.parse_args()
    device = torch.device(args.device)
    rows = []
    for pair in args.models.split(","):
        name, path = pair.split("=", 1)
        model, config = load_model(path)
        model.to(device)
        model.eval()
        with torch.no_grad():
            warm = torch.randint(0, config.vocab_size, (1, 8), device=device)
            if isinstance(model, RAT):
                state = model.init_states(1, device)
                logits, _ = model.prefill(warm)
            else:
                caches = [KVCache() for _ in model.blocks]
                logits = model.prefill(warm, caches)
        params = model.get_num_params()
        print(f"\n{name}: {type(model).__name__} ({params:,} params)")
        for batch in [int(x) for x in args.batch_sizes.split(",")]:
            for prompt_len in [int(x) for x in args.prompt_lengths.split(",")]:
                best = None
                for _ in range(args.repeats):
                    result = run_once(model, config, batch, prompt_len,
                                      args.new_tokens, device)
                    if best is None or result[1] < best[1]:
                        best = result
                prefill_s, decode_s = best
                peak = (torch.cuda.max_memory_allocated(device) / 1e9
                        if device.type == "cuda" else 0.0)
                row = {
                    "model": name,
                    "arch": type(model).__name__,
                    "params": params,
                    "batch": batch,
                    "prompt_len": prompt_len,
                    "new_tokens": args.new_tokens,
                    "prefill_s": f"{prefill_s:.6f}",
                    "decode_s": f"{decode_s:.6f}",
                    "decode_tokens_per_sec": f"{batch * args.new_tokens / decode_s:.2f}",
                    "prefill_tokens_per_sec": f"{batch * prompt_len / prefill_s:.0f}",
                    "peak_gb": f"{peak:.2f}",
                }
                rows.append(row)
                print(f"  B={batch:<2d} L={prompt_len:<5d} "
                      f"decode {row['decode_tokens_per_sec']:>8s} tok/s "
                      f"prefill {row['prefill_tokens_per_sec']:>8s} tok/s "
                      f"peak {row['peak_gb']} GB")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    names = sorted({r["model"] for r in rows})
    batches = sorted({r["batch"] for r in rows})
    fig, axes = plt.subplots(1, len(batches), figsize=(5 * len(batches), 4.2), squeeze=False)
    for ax, batch in zip(axes[0], batches):
        for name in names:
            pts = [r for r in rows if r["model"] == name and r["batch"] == batch]
            ax.plot([r["prompt_len"] for r in pts],
                    [float(r["decode_tokens_per_sec"]) for r in pts],
                    marker="o", label=name)
        ax.set_xscale("log", base=2)
        ax.set_title(f"B={batch} decode throughput")
        ax.set_xlabel("context length")
        ax.set_ylabel("tokens/s")
        ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.tight_layout()
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=150)
    print("wrote", args.out, "and", png)


if __name__ == "__main__":
    main()
