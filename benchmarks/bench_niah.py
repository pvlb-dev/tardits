#!/usr/bin/env python
"""Smallest practical needle-in-a-haystack (NIAH) pilot for the tardits char models.

Why this protocol
-----------------
Both shipped checkpoints are ~0.6-0.9M parameter *char-level* models trained for
5000 iters on shakespeare with ``block_size=128``.  The question is whether
retrieval survives past that window, and specifically whether RAT's fixed-size
recurrent fast-weight state can hold a fact that RealGPT's RoPE (trained only to
position 128) cannot.

The cheapest protocol that still asks that question:

* **Haystack** = a contiguous slice of the *training* split of
  ``examples/data/shakespeare.txt`` (in-distribution chars, real word statistics)
  -- not random token noise.
* **Needle** = ``"okonkwo"`` (7 chars, an OOV proper noun, absent from the
  corpus).  It is lower-case on purpose: the char-level model has no
  upper-case/numeric induction signal at this scale (a first pilot with
  ``"OKONKWO-3"`` failed even the adjacent-copy sanity check because ``O`` is
  OOD; lower-case gives retrieval its best chance).  It is inserted as
  ``"\\nthe secret word is okonkwo\\n"``, replacing a same-length slice of
  filler so the total prompt length is unchanged by the insertion.
* **Query** = append ``"\\nthe secret word is "`` to the end of the haystack and
  evaluate the *next* ``len(needle)`` chars.  This is a pure continuation task,
  which is the only kind a char-level LM can do; no special prompt format, no
  retraining, no sampling stochasticity.
* **Depth** = where the needle block starts as a fraction of the available
  positions, ``pos = round(depth * (L - len(block)))``.  Depth 1.0 means the
  needle sits immediately before the query cue: a *copy sanity check* -- if the
  model cannot retrieve an adjacent needle, the long-range negatives are
  uninformative about long-range retrieval.
* **Chance baseline** = the identical prompt/query with the needle **absent**
  (filler + query cue only, same total length), scored against the same answer.
  This measures how often a model that never saw the needle outputs it anyway.

Two decoding regimes are reported:

* **greedy** (primary): argmax char by char -- the NIAH score in the literature.
* **teacher-forced** (sensitivity): feed the correct prefix at every step.  This
  decouples a weak retrieval signal from a cascading generation failure, and the
  *rank* of the correct answer chars is reported too (65-char vocab, so rank 32
  is chance).  If the rank at depth 1.0 is already bad, the model cannot copy at
  all; if it is good at 1.0 and bad at 0.9, the receptive field/state is the
  limit.

All arms for one ``(model, length)`` are evaluated in a single batch per decode
regime, so the CPU cost is a handful of prefills, not one per sample.

Usage:
    .venv/bin/python benchmarks/bench_niah.py
    .venv/bin/python benchmarks/bench_niah.py --lengths 128,256,512,1024 --samples 8
    .venv/bin/python benchmarks/bench_niah.py --models rat-l1=examples/saved/shakespeare/rat_l1_h21_5k.pt

Output: table on stdout + ``docs/assets/curves/niah_pilot.csv``.
"""

import argparse
import csv
import os
import random
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from tardits.models import KVCache, RAT, RealGPT  # noqa: E402

DEFAULT_MODELS = [
    ("real", "examples/saved/shakespeare/real_baseline.pt"),
    ("rat-l1", "examples/saved/shakespeare/rat_l1_h21_5k.pt"),
    ("rat-l2", "examples/saved/shakespeare/rat_l2_h16_l5_5k.pt"),
]
DEFAULT_NEEDLE = "okonkwo"
DEFAULT_CUE = "the secret word is "
TEST_SPLIT = 0.8  # same 80/20 split as training; keep haystacks in-distribution


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
def load_model(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    cls = RealGPT if config.level == 0 else RAT
    model = cls(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config


def load_vocab(corpus_path: str, vocab_path: str | None) -> dict[str, int]:
    """Char -> id map. Prefer the trained vocab file; fall back to the corpus."""
    with open(corpus_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as fh:
            chars = list(fh.read())
    else:
        chars = sorted(set(text))
    missing = sorted(set(text) - set(chars))
    assert not missing, (
        f"{vocab_path} is missing corpus chars {missing!r}"
    )
    # Extra vocab chars are allowed: a held-out slice may not exercise every
    # character of the training charset (e.g. FineWeb val misses one of 96).
    return {ch: i for i, ch in enumerate(chars)}


def build_haystacks(
    text: str, lengths: list[int], samples: int, seed: int
) -> dict[int, list[str]]:
    """Deterministic shakespeare slices per (length, sample).

    Drawn independently per length so a row at 512 is not simply the 128-row's
    haystack padded; each stays inside the training split.
    """
    split = int(len(text) * TEST_SPLIT)
    haystacks: dict[int, list[str]] = {}
    for length in lengths:
        rng = random.Random(f"{seed}:{length}")
        out = []
        for _ in range(samples):
            start = rng.randrange(0, split - length - 40)
            out.append(text[start : start + length])
        haystacks[length] = out
    return haystacks


def encode(prompts: list[str], stoi: dict[str, int]) -> torch.Tensor:
    """[B, T] long tensor. Raises if any char is outside the trained vocab."""
    unknown = sorted({ch for p in prompts for ch in p if ch not in stoi})
    if unknown:
        raise ValueError(f"prompt chars outside the trained vocab: {unknown!r}")
    return torch.tensor(
        [[stoi[ch] for ch in p] for p in prompts], dtype=torch.long
    )


def decode(ids: list[int], stoi: dict[str, int]) -> str:
    itos = {i: ch for ch, i in stoi.items()}
    return "".join(itos[i] for i in ids)


# --------------------------------------------------------------------------- #
# completion (deterministic; greedy or teacher-forced)
# --------------------------------------------------------------------------- #
def _real_prefill(model: RealGPT, prompt: torch.Tensor):
    """Prefill the KV cache; returns (next-token logits, caches, mode).

    Current ``RealGPT.prefill`` chunks the prompt in ``block_size`` pieces and
    the RoPE cache grows lazily, so any length works.  Older trees raised for
    T > block_size; in that case fall back to a truncated window and label it.
    """
    block = model.config.block_size
    T = prompt.shape[1]
    caches = [KVCache() for _ in model.blocks]
    try:
        if hasattr(model, "prefill"):
            logits = model.prefill(prompt, caches)
            return logits, caches, ("full" if T <= block else "chunked")
        logits, _ = model(prompt, kv_cache=caches, start_pos=0)
        return logits, caches, "full"
    except (ValueError, RuntimeError):
        # Pre-long-context tree: only the trailing block_size tokens are visible.
        caches = [KVCache() for _ in model.blocks]
        window = prompt[:, -block:]
        logits, _ = model(window, kv_cache=caches, start_pos=0)
        return logits, caches, f"windowed({block})"


@torch.no_grad()
def run_completion(
    model,
    prompt: torch.Tensor,
    n_new: int,
    targets: torch.Tensor | None = None,
):
    """Decode ``n_new`` chars after ``prompt``.

    ``targets`` is a 1-D tensor of the correct next chars for teacher forcing;
    ``None`` means greedy argmax.  Returns ``(gen_ids [B, n_new], per-step logits
    [B, n_new, vocab], mode, context_len_seen)``.
    """
    B = prompt.shape[0]
    out_ids: list[torch.Tensor] = []
    out_logits: list[torch.Tensor] = []

    def step_from(logits: torch.Tensor, i: int):
        step_logits = logits[:, -1, :]  # [B, V] predicts char i
        out_logits.append(step_logits.unsqueeze(1))
        # Always score the *model's* prediction; teacher forcing only changes
        # what token is fed back into the recurrent state / KV cache.
        pred = step_logits.argmax(dim=-1, keepdim=True)
        out_ids.append(pred)
        if targets is None:
            return pred
        return targets[i].view(1, 1).expand(B, 1)

    if isinstance(model, RAT):
        states = model.init_states(B, torch.device("cpu"))
        logits, states = model._warm_states(prompt, states)
        context_len = prompt.shape[1]
        pos = context_len
        for i in range(n_new):
            nxt = step_from(logits, i)
            logits, states = model.step(nxt, states, pos)
            pos += 1
        mode = "recurrent"
    else:
        logits, caches, mode = _real_prefill(model, prompt)
        context_len = len(caches[0])
        pos = context_len
        for i in range(n_new):
            nxt = step_from(logits, i)
            logits, _ = model(nxt, kv_cache=caches, start_pos=pos)
            pos += 1

    return (
        torch.cat(out_ids, dim=1),
        torch.cat(out_logits, dim=1),
        mode,
        context_len,
    )


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score(gen: torch.Tensor, answer: torch.Tensor, step_logits: torch.Tensor) -> dict:
    """Exact match / char acc / prefix length plus the rank of the answer chars.

    ``step_logits`` are the argmax path's logits at each position, so the rank
    metrics answer "how close was the model to the right char, even if greedy
    decoding picked something else?".  Rank 0 = argmax; chance ~ vocab/2.
    """
    match = gen == answer.unsqueeze(0)
    prefix = []
    for row in match.tolist():
        n = 0
        for hit in row:
            if not hit:
                break
            n += 1
        prefix.append(n)

    B, n, _ = step_logits.shape
    answer_logits = step_logits.gather(
        -1, answer.view(1, n, 1).expand(B, -1, -1)
    )  # [B, n, 1]
    ranks = (step_logits > answer_logits).sum(dim=-1).float()  # 0 = top-1
    return {
        "exact": match.all(dim=1).float().mean().item(),
        "char": match.float().mean().item(),
        "first": match[:, 0].float().mean().item(),
        "prefix": sum(prefix) / len(prefix),
        "mean_rank": ranks.mean().item(),
        "first_rank": ranks[:, 0].mean().item(),
    }


# --------------------------------------------------------------------------- #
# prompting
# --------------------------------------------------------------------------- #
def build_prompt(
    haystack: str, needle_block: str, cue_tail: str, depth: float | None
) -> tuple[str, int]:
    """Insert the needle block at ``depth`` (replacing filler); append query cue.

    ``depth=None`` is the needle-absent arm: the haystack is untouched.
    Returns ``(prompt, needle_start_offset)``.
    """
    if depth is None:
        return haystack + cue_tail, -1
    pos = int(round(depth * (len(haystack) - len(needle_block))))
    pos = max(0, min(len(haystack) - len(needle_block), pos))
    prompt = (
        haystack[:pos]
        + needle_block
        + haystack[pos + len(needle_block) :]
        + cue_tail
    )
    return prompt, pos


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal NIAH pilot for char-level tardits checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        default=",".join(f"{name}={path}" for name, path in DEFAULT_MODELS),
        help="comma-separated name=checkpoint pairs",
    )
    parser.add_argument("--lengths", default="128,256,512")
    parser.add_argument(
        "--depths",
        default="0.1,0.5,0.9,1.0",
        help="needle depth fractions; 1.0 = adjacent copy sanity check",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--needle", default=DEFAULT_NEEDLE)
    parser.add_argument("--cue", default=DEFAULT_CUE)
    parser.add_argument("--corpus", default="examples/data/shakespeare.txt")
    parser.add_argument("--vocab", default="examples/saved/shakespeare/vocab.txt")
    parser.add_argument("--out", default="docs/assets/curves/niah_pilot.csv")
    args = parser.parse_args()

    with open(args.corpus, "r", encoding="utf-8") as fh:
        text = fh.read()
    stoi = load_vocab(args.corpus, args.vocab)
    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]

    needle_block = "\n" + args.cue + args.needle + "\n"
    cue_tail = "\n" + args.cue
    if len(needle_block) >= min(lengths):
        raise ValueError(
            f"needle block ({len(needle_block)} chars) does not fit in "
            f"the shortest haystack ({min(lengths)})"
        )
    answer_text = args.needle
    answer = torch.tensor([stoi[ch] for ch in answer_text], dtype=torch.long)
    k = len(answer_text)
    # Sanity: the needle must not exist in the corpus, otherwise "retrieval"
    # could be plain corpus memorisation.
    assert text.count(args.needle) == 0, (
        f"needle {args.needle!r} occurs in the corpus; pick a rarer string"
    )

    haystacks = build_haystacks(text, lengths, args.samples, args.seed)

    print(
        f"NIAH pilot | needle={args.needle!r} ({k} chars, absent from corpus) "
        f"| cue={args.cue!r}\n"
        f"samples={args.samples} seed={args.seed} depths={depths} "
        f"lengths={lengths} vocab={len(stoi)}"
    )

    rows: list[dict] = []
    for pair in args.models.split(","):
        name, path = pair.split("=", 1)
        model, config = load_model(path)
        params = model.get_num_params()
        print(
            f"\n=== {name}: {type(model).__name__} level={config.level} "
            f"layers={config.n_layer} {params:,} params ==="
        )
        print(
            f"  {'len':>4} {'depth':>7} {'greedy':>6} {'char':>6} "
            f"{'tfchar':>6} {'rank':>6} {'first_rk':>8} {'mode':>13} "
            f"{'ctx':>5} {'time':>6}"
        )

        def report(length, label, prompts):
            """Run both regimes on one batched arm and print/record a row."""
            t0 = time.perf_counter()
            gen, greedy_logits, mode, ctx = run_completion(
                model, encode(prompts, stoi), k
            )
            greedy = score(gen, answer, greedy_logits)
            tf_ids, tf_logits, _, _ = run_completion(
                model, encode(prompts, stoi), k, targets=answer
            )
            tf = score(tf_ids, answer, tf_logits)
            dt = time.perf_counter() - t0
            rows.append(
                {
                    "model": name,
                    "arch": type(model).__name__,
                    "level": config.level,
                    "n_layer": config.n_layer,
                    "params": params,
                    "haystack_len": length,
                    "prompt_len": len(prompts[0]),
                    "context_len": ctx,
                    "depth": label,
                    "arm": "control" if label == "none" else "needle",
                    "mode": mode,
                    "needle": args.needle,
                    "cue": args.cue.strip(),
                    "samples": args.samples,
                    "k": k,
                    "greedy_exact": f"{greedy['exact']:.4f}",
                    "greedy_char": f"{greedy['char']:.4f}",
                    "greedy_first_char": f"{greedy['first']:.4f}",
                    "greedy_mean_prefix": f"{greedy['prefix']:.4f}",
                    "greedy_mean_rank": f"{greedy['mean_rank']:.4f}",
                    "greedy_first_rank": f"{greedy['first_rank']:.4f}",
                    "tf_exact": f"{tf['exact']:.4f}",
                    "tf_char": f"{tf['char']:.4f}",
                    "tf_first_char": f"{tf['first']:.4f}",
                    "tf_mean_rank": f"{tf['mean_rank']:.4f}",
                    "tf_first_rank": f"{tf['first_rank']:.4f}",
                    "seconds": f"{dt:.3f}",
                }
            )
            print(
                f"  {length:>4} {label:>7} {greedy['exact']:>6.2f} "
                f"{greedy['char']:>6.3f} {tf['char']:>6.3f} "
                f"{tf['mean_rank']:>6.1f} {tf['first_rank']:>8.1f} "
                f"{mode:>13} {ctx:>5} {dt:>5.1f}s"
            )
            return gen, prompts

        for length in lengths:
            hays = haystacks[length]
            control_prompts = [
                build_prompt(h, needle_block, cue_tail, None)[0] for h in hays
            ]
            gen, prompts = report(length, "none", control_prompts)
            print(
                f"       control greedy sample: {decode(gen[0].tolist(), stoi)!r}"
            )

            for depth in depths:
                prompts = [
                    build_prompt(h, needle_block, cue_tail, depth)[0] for h in hays
                ]
                gen, prompts = report(length, f"{depth:.2f}", prompts)
                if depth in (depths[0], depths[-1]):
                    tail = prompts[0][-38:]
                    print(
                        f"       L={length} depth={depth:.2f} prompt tail "
                        f"{tail!r} -> greedy {decode(gen[0].tolist(), stoi)!r} "
                        f"(answer {args.needle!r})"
                    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = [
        "model",
        "arch",
        "level",
        "n_layer",
        "params",
        "haystack_len",
        "prompt_len",
        "context_len",
        "depth",
        "arm",
        "mode",
        "needle",
        "cue",
        "samples",
        "k",
        "greedy_exact",
        "greedy_char",
        "greedy_first_char",
        "greedy_mean_prefix",
        "greedy_mean_rank",
        "greedy_first_rank",
        "tf_exact",
        "tf_char",
        "tf_first_char",
        "tf_mean_rank",
        "tf_first_rank",
        "seconds",
    ]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
