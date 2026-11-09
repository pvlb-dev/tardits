#!/usr/bin/env python
"""Overfit one fixed text, then measure where generation drifts.

Tiny CPU test of generation stability vs trained block size. A model is trained
on random windows of one non-periodic random-ASCII text at block_size L; if it
overfits the mapping, it can reproduce the text from a prefix. We measure:
  - teacher-forced next-char accuracy by position (no generation compounding)
  - greedy generation: first mismatch / longest exact prefix
RAT uses its recurrent state; RealGPT uses its KV cache (the usual asymmetry).
No claim about "winning"; this is a stability/attractor probe.
"""
import argparse, csv, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tardits.models import ModelConfig, RAT, RealGPT
from tardits.models.real import KVCache


class TrainableRoPE(nn.Module):
    """Real-valued analogue of the trainable GyRoPE prototype.

    Classical RoPE uses fixed inverse frequencies; this version makes one
    frequency per head/pair trainable, initialised to the classical values.
    Same q/k pair rotation, same API as ``RotaryPositionalEmbedding``.
    """
    def __init__(self, head_size, block_size, n_head):
        super().__init__()
        self.head_size = head_size
        base = 1.0 / (10000.0 ** (torch.arange(0, head_size, 2).float() / head_size))
        self.inv_freq = nn.Parameter(base.repeat(n_head, 1))  # [nh, hs/2]
        self.register_buffer("t", torch.arange(block_size, dtype=torch.float32))

    def forward(self, x, start_pos=0):
        B, nh, T, hs = x.shape
        t = torch.arange(start_pos, start_pos + T, dtype=self.inv_freq.dtype,
                         device=x.device)
        freqs = torch.einsum("hp,t->htp", self.inv_freq[:nh], t)
        cos = torch.cos(freqs).unsqueeze(0)  # [1,nh,T,pairs]
        sin = torch.sin(freqs).unsqueeze(0)
        x1, x2 = x[..., :hs // 2], x[..., hs // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def make_text(n, seed):
    rng = random.Random(seed)
    chars = [chr(c) for c in range(32, 127) if c not in (92,)]  # printable ASCII-92
    return "".join(rng.choice(chars) for _ in range(n))


def encode(text, stoi): return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def train_model(arch, level, block, n_embd, n_layer, n_head, text, steps, seed, lr=1e-3, device="cpu", rope_type="classical"):
    torch.manual_seed(seed)
    cfg = ModelConfig(level=level, block_size=block, vocab_size=96, n_head=n_head,
                      n_embd=n_embd, n_layer=n_layer, dropout=0.0)
    model = (RAT if level else RealGPT)(cfg).to(device)
    if arch == "real" and rope_type == "trainable":
        head_size = n_embd // n_head
        for b in model.blocks:
            b.sa.rope = TrainableRoPE(head_size, block, n_head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    data = encode(text, {c: i for i, c in enumerate(sorted(set(text)))})
    data = data.to(device)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for step in range(steps):
        starts = torch.randint(0, len(data) - block - 1, (32,), generator=g)
        xb = torch.stack([data[s:s+block] for s in starts])
        yb = torch.stack([data[s+1:s+1+block] for s in starts])
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    return model, data


@torch.no_grad()
def train_acc(model, data, block):
    g = torch.Generator().manual_seed(99)
    correct = total = 0
    for _ in range(20):
        starts = torch.randint(0, len(data)-block-1, (32,), generator=g)
        xb = torch.stack([data[s:s+block] for s in starts])
        yb = torch.stack([data[s+1:s+1+block] for s in starts])
        logits, _ = model(xb)
        correct += (logits.argmax(-1) == yb).sum().item()
        total += yb.numel()
    return correct/total


@torch.no_grad()
def teacher_forced(model, data):
    x = data[:-1].unsqueeze(0); y = data[1:].unsqueeze(0)
    if isinstance(model, RealGPT):
        caches = [KVCache() for _ in model.blocks]
        chunks = []
        for start in range(0, x.shape[1], model.config.block_size):
            logits, _ = model(x[:, start:start + model.config.block_size], kv_cache=caches, start_pos=start)
            chunks.append(logits)
        logits = torch.cat(chunks, dim=1)
    else:
        logits, _ = model(x)
    pred = logits.argmax(-1)
    ok = (pred[0] == y[0]).float()
    return ok.cpu().numpy()


def generate(model, data, block, max_new):
    if isinstance(model, RealGPT):
        idx = data[:block].unsqueeze(0)
        try:
            out = model.generate_cached(idx, max_new_tokens=max_new, temp=1.0, top_k=1)
        except Exception:
            out = model.generate(idx, max_new_tokens=max_new, temp=1.0, top_k=1)
    else:
        idx = data[:block].unsqueeze(0)
        out = model.generate(idx, max_new_tokens=max_new, temp=1.0, top_k=1)
    return out[0, block:].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--blocks", default="16,32")
    ap.add_argument("--sizes", default="64,96,128")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--archs", default="rat,real")
    ap.add_argument("--rope", default="classical", choices=("classical", "trainable"))
    ap.add_argument("--out", default="docs/assets/curves/memorize.csv")
    args = ap.parse_args()
    text = make_text(args.text_len, 1234)
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    data = encode(text, stoi)
    rows = []
    for block in [int(x) for x in args.blocks.split(",")]:
        for size in [int(x) for x in args.sizes.split(",")]:
            for arch, level in (("rat", 1), ("real", 0)):
                if arch not in args.archs.split(","):
                    continue
                for seed in [int(x) for x in args.seeds.split(",")]:
                    t0 = time.time()
                    model, data_d = train_model(arch, level, block, size, 2, 8, text, args.steps, seed, rope_type=args.rope)
                    acc = train_acc(model, data_d, block)
                    tf = teacher_forced(model, data_d)
                    gen = generate(model, data_d, block, len(data_d)-block)
                    target = data_d[block:].cpu().numpy()
                    first = len(target)
                    for i,(a,b) in enumerate(zip(gen, target)):
                        if a != b:
                            first = i; break
                    rows.append(dict(arch=arch, rope=args.rope, block=block, n_embd=size, params=model.get_num_params(),
                                     seed=seed, train_acc=round(acc,4),
                                     tf_acc=round(float(tf.mean()),4),
                                     gen_first_mismatch=first, gen_exact_prefix=first,
                                     secs=round(time.time()-t0,1)))
                    print(rows[-1], flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
