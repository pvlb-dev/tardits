# RealGPT review - v0.3.0 baseline

Date: 2026-10-25. Scope: `src/tardits/models/real.py` plus the shared config/training
plumbing it depends on. The goal was a worthy, trustworthy real-valued baseline for the
RAT level-1/2 comparison - not a rewrite.

## Verdict

The core model is mathematically sound: causal softmax attention with RoPE, RMSNorm,
SwiGLU, residual stream, and a standard LM head. It trains, produces finite gradients,
respects causality, and can overfit a fixed tiny batch. No NaN/instability was observed.

The architecture is intentionally **not** nanoGPT's canonical pre-norm. The block is a
"post-norm branch":

```python
x = x + ln1(sa(x))
x = x + ln2(ffwd(x))
```

This normalises the sublayer *output* before the residual add. It was compared against
other norm layouts during the original development and kept. It stays; the purpose of
this review was to catch defects, not to re-litigate a validated design choice.

## Checks run

- Strict causality: perturbing tokens at positions `t >= 5` leaves logits at `t < 5`
  unchanged (`tests/test_real.py`).
- Forward/backward finite; no NaNs in gradients.
- Tiny fixed batch can be overfit (optimisation path works).
- Checkpoint save/load round-trip (state dict identical).
- `n_embd % n_head == 0`; RoPE needs an even `head_size`.
- Context longer than `block_size` fails with a clear error instead of a shape mismatch.
- `seed: 0` now seeds (the old `if seed:` check skipped it).
- MLflow deep phase logging no longer raises for RealGPT (it has no `.sail`).

## Changes made

| Area | Change | Why |
|---|---|---|
| `models/base.py` | `ModelConfig.__post_init__` validates positive dims and head divisibility | fail early instead of inside a `view()` |
| `models/real.py` | require even `head_size`; reject `T > block_size` | RoPE buffer slicing and causal mask are shape-sensitive |
| `train.py` | `if seed is not None` | `seed: 0` was silently ignored |
| `modules/logging.py` | skip phase hooks when a block has no `.sail` | RealGPT uses SwiGLU, not SAIL |
| `modules/trainer.py` | only save when `checkpoint` is non-empty | allow screening configs without a checkpoint |
| `benchmarks/bench_attention.py` | default config is `rat.yaml` | the old `peter.yaml` was deleted |

## Adjacent bug found (not RealGPT)

`GyRoPE` packed the rotor's real components as first-half/second-half. For level ≥ 2 that
permutes the hypercomplex axes: a quaternion `i`-rotation landed on `j`. Fixed to the
interleaved `0::2` / `1::2` order required by `cd_mul`'s real-dim convention, with
`tests/test_gyrope.py`. This affects `HyperFormer` checkpoints/benchmarks, not RAT (RAT's
GyRoPE is commented out); see the "rerun hyper benchmarks" backlog issue.

## Post-norm vs pre-norm sanity A/B

`experiments/ablation/norm_ablation.py --iters 400` runs the same shakespeare split and
baseline shape (96 embd, 6 heads, 8 layers) for the current branch-norm block and a local
pre-norm monkeypatch. Full curve in `docs/assets/curves/norm_ablation.csv`.

| layout | val @250 | val @399 |
|---|---:|---:|
| branch-norm (current) | 2.1954 | 2.0460 |
| pre-norm | 2.1467 | **1.9801** |

Pre-norm converges faster in this short run (1.9801 vs 2.0460, −0.066). That is enough to
flag the layout for a proper re-test, but **not** enough to change the release baseline:
400 iterations is early, the original development comparison was full-length and favored
branch-norm, and the real baseline + RAT finals in this report all use branch-norm. A
silent switch would invalidate the comparison. The follow-up is a 5k A/B at the same
shape before 0.3.x changes the default - see
#11.

### Full 5k A/B (result)

The layout is configurable (`model.norm: branch|pre`); the full 5k test ran
`experiments/configs/norm/real_pre_5k.yaml` against the branch baseline, same 96/6/8
shape, seed 1337, 5k iters, 200 eval batches:

| layout | train | final val | best val |
|---|---:|---:|---:|
| branch (project default) | 1.3402 | **1.6280** | **1.6280** @5k |
| pre (canonical) | 1.3197 | 1.6356 | 1.6383 @4.5k |

**Branch wins by 0.0076 at 5k.** Pre-norm was ahead at 400 iters (1.9801 vs 2.0460),
which is exactly why the short test was not enough: pre-norm converges faster but ends
with a larger train/val gap (0.316 vs 0.288) and a worse final val. The default stays
`branch`; `norm: pre` remains available for future experiments. #11 is resolved.

## What was deliberately not changed

- Norm placement (see above).
- Biases on `proj`/`lm_head`, and the untied embedding/LM head. They are part of the
  measured 0.90M-param baseline; changing them would invalidate the comparison.
- RoPE cache/buffers and dropout placement.
- No LR schedule / weight decay tuning in the baseline; the sweep uses the historical
  constant `5e-4` AdamW setting for comparability.

## Baseline configuration

`experiments/configs/shakespeare/final/real_baseline.yaml`:

| field | value |
|---|---|
| arch / level | real / 0 |
| block_size | 128 |
| n_head / n_embd / n_layer | 6 / 96 / 8 |
| dropout | 0.2 |
| params | 0.90M |
| iters / eval | 5000 / every 500 × 200 batches |
| CPU actual | 92.3 min (train 1.3402, val 1.6280) |

## Open questions

- A short LR/weight-decay sweep for the real baseline is still worth doing later; the
  current run is a historical-settings baseline, not a tuned GPT.
- Weight tying may improve the parameter-efficiency comparison; keep it out until the
  RAT table is stable.
