# RAT state probe - decay, horizons and state-delta diagnostics

**Status: instrumentation / usage guide (2026-10-30).** This is the training-free probe that
was built for the the RAT2 / Rettention study (research tree) pre-flight, kept in the main repo
after the RAT2 idea was put **on hold / stopped**. It answers questions about a *trained* RAT
checkpoint - how far back its fast-weight state actually remembers, how much of a readout
comes from older blocks, and how a block's state delta behaves - without training anything.
The RAT2 study that used it is the post-mortem (research tree); the code is
`experiments/rat2/` (research tree).

## Why it is useful for RAT research

RAT's "fixed-size memory" claim depends on one learned number per state element: the complex
decay `γ`. The probe measures what that decay does in an actual checkpoint:

- **Effective memory horizon** `1/(1-|γ|)` per layer - the model writes a fast-weight memory,
  but if the horizon is ~5–20 tokens it is a *local* language model, and long-context
  retrieval claims are unsupported (see #20).
- **Readout history share** - the fraction of the block-end readout that comes from the
  carried state (earlier blocks) rather than the current block. A few percent means the state
  is effectively a suffix cache; any state-injection idea has to overcome that.
- **`|γ| > 1` audit** - a single over-unit channel is an unbounded gain and grows the state
  without limit over long contexts (`1.001^48000 ≈ 7e20`), silently corrupting long-context
  probes. The trainer now caps `|γ| ≤ 1` after each step
  (`acd75ad`), but existing checkpoints
  must be audited. See #27.
- **State-delta capacity** - how many per-block deltas can be superposed before readout
  interference destroys the signal, and whether a delta-based cosine key is a usable
  retrieval key. This is a reusable protocol for *any* state-memory proposal, not just RAT2.

## Quickstart

Checkpoints and corpora are gitignored local artifacts (e.g. from Kaggle runs under
`.cache/kaggle/`). The probe reads a checkpoint saved by `LMCoreMixin.save_checkpoint`. Run
from the repo root with the project venv:

```bash
# 0. one-time
UV_CACHE_DIR="$PWD/.uv-cache" uv sync --extra mlflow --locked

# 1. exactness self-test: exact float64 scan vs the recurrent step path (~2e-7 rel)
.venv/bin/python -m experiments.rat2.probe self-test \
  --checkpoint .cache/rat2/rat_l1_45m_22k.pt --block-len 128

# 2. extract per-block deltas from a char-level text slice
.venv/bin/python -m experiments.rat2.probe extract \
  --checkpoint .cache/rat2/rat_l1_45m_22k.pt \
  --text .cache/fineweb-probe/val.txt --vocab .cache/fineweb-probe/vocab.txt \
  --block-len 128 --chars 48000 --threads 8 \
  --out .cache/rat2/probe_22k45m_L128.pt --self-test

# 3. readout history share + K-vs-SIR capacity curves (+ PNG/Markdown)
.venv/bin/python -m experiments.rat2.capacity \
  --artifact .cache/rat2/probe_22k45m_L128.pt \
  --out-prefix .cache/rat2/capacity_22k45m_L128 --threads 4

# 4. retrieval keys on a synthetic topic-marker DB (mid = neutral, end = recency control)
.venv/bin/python -m experiments.rat2.synthetic build --flavor mid
.venv/bin/python -m experiments.rat2.retrieval \
  --checkpoint .cache/rat2/rat_l1_45m_22k.pt --flavor mid \
  --out-prefix .cache/rat2/retrieval_22k45m_L128

# 5. parked RAT2 gate prototype tests (not needed for the probe)
.venv/bin/python -m unittest experiments.rat2.test_memory_gate -v
```

Artifacts land under `.cache/rat2/` (gitignored); the retrieval extraction cache is
checkpoint-tagged, so runs on different checkpoints cannot silently reuse each other's
tensors. Use `--threads 4–8` on the shared box.

## Fast checkpoint audit

The most reusable part is a one-glance audit of any RAT checkpoint:

```python
import torch
from experiments.rat2.probe import load_rat

model, cfg = load_rat(".cache/rat2/rat_l1_45m_22k.pt")
for i, block in enumerate(model.blocks):
    g = block.sa.gamma.detach().abs()
    n_over = int((g > 1).sum())
    horizon = 1.0 / max(1e-9, 1.0 - float(g.mean()))
    print(f"L{i:2d}  mean|g|={float(g.mean()):.4f}  horizon={horizon:5.1f} tok  "
          f"max|g|={float(g.max()):.6f}  channels>1={n_over}")
```

Red flags: `max|g| > 1` (unbounded state growth - #27),
a horizon in the single digits (the model cannot use long-range state), or a large spread of
`mean|g|` across layers (layer specialisation worth reporting).

## What the numbers mean

Definitions are fixed in `experiments/rat2/README.md` (research tree);
the short version:

| quantity | meaning |
|---|---|
| `D_b = S_end − γ^L ⊙ S_start` | exact block contribution, no training; lossless decomposition of the carried state |
| history share | `‖readout(γ^L S_start)‖ / ‖readout(S_end)‖` at a block end - how much older blocks contribute |
| self share | `‖readout(D_b)‖ / ‖readout(S_end)‖` - how much the current block contributes (norm shares are not additive) |
| decay horizon | `1/(1−mean |γ|)` tokens, plus per-layer min/max across state elements |
| unit-gain SIR | signal/interference when `K−1` equal-power distractor deltas are superposed: capacity is `O(1)` when the threshold is 1–2 |
| retrieval keys | raw/centred delta cosine, query-probe readout similarity; controls: raw state, lexical n-gram, random and shuffled labels |

Two regimes are reported on purpose: **natural decay** (`γ^Δ D_b`, what the carried state
holds) is a decay artefact - old blocks are invisible, so SIR is huge - while **unit gain**
(`D_b/‖D_b‖`, the bounded-injection idea) is the real superposition question.

## What the current results say

Three checkpoints measured with the same protocol (L=128 and L=512, FineWeb val slice,
doc-disjoint; details and per-layer tables in the RAT2 post-mortem (research tree) §10):

| checkpoint | horizon | history share L128 / L512 | unit SIR K=2 | mid-retrieval delta MRR | end control |
|---|---|---|---|---|---|
| 13.5k 44M (pre-`#23`) | 5.8–22.3 tok | 3.10 % / 0.22 % | 0.998 | 0.224 (chance 0.188) | 1.000 |
| 22k clean 44M | 4.8–14.9 | 4.56 % / 0.37 % | 0.994 | 0.253 | 1.000 |
| 22k 33M (9 layers) | 5.2–14.6 | 3.32 % / 0.07 % | 0.996 | 0.306 | 1.000 |

Reading: at this scale a trained RAT L1 is a **local** model (horizon ≪ block length), so a
block's state delta is a **recency/suffix packet**, not a block summary; superposition
capacity is one packet; and cosine similarity over the state does not track topic relevance.
The `end`-flavor MRR of 1.000 is the recency control (a lexical n-gram over the last 16 chars
scores the same), not semantic retrieval. RAT2 was stopped on these results.

## Caveats

- These are **diagnostics, not training**; they say nothing about what a model could learn
  with a different decay, selective gating or a learned readout (see the reopen conditions in
  the RAT2 post-mortem (research tree) §7).
- Single checkpoint/seed/slice; char-level 45M models; level 1 only; synthetic topic markers
  are out-of-distribution words; capacity queries are block-end only.
- The probe's scan is exact to ~2e-7 relative **by construction**; the measured difference
  between a checkpoint's training scan and the `step` path (`#23`) was 8.9e-3 on the pre-fix
  checkpoint and 1.2e-4 / 2.1e-4 on the 22k checkpoints - audit it per checkpoint, do not
  assume it is zero.
- Artifacts under `.cache/` are gitignored; committed figures are under
  `docs/assets/rat2/` (research tree).

## File map

| file | role |
|---|---|
| `experiments/rat2/probe.py` (research tree) | exact per-block state/delta extraction, checkpoint audit, self-test |
| `experiments/rat2/capacity.py` (research tree) | readout self/history share, K-vs-SIR curves, baselines |
| `experiments/rat2/synthetic.py` (research tree) | marker/topic blocks for retrieval tests (flavors `mid`/`tail`/`end`/`markeronly`) |
| `experiments/rat2/retrieval.py` (research tree) | retrieval keys, recall@k/MRR, controls |
| `experiments/rat2/memory_gate.py` (research tree), `experiments/rat2/GATE_SPEC.md` (research tree) | parked RAT2 zero-init bounded-gate prototype (page-only, not release code) |
