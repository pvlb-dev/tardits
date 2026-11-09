# 2026-10-25 - shakespeare real vs RAT (v0.3.0)

Status: **in progress**. This report is filled as the staged sweep completes.

## Question

On the repo's own shakespeare corpus (char-level, vocab 65 - no tokenizer), can RAT level
1 and level 2 match the real GPT baseline's val loss at equal or lower parameter count?
The previous kropotkin experiments are historical (different corpus/vocab) and are not a
valid comparison.

## Norm-layout A/B (done)

`model.norm` is now selectable. Full 5k test, same 96/6/8 shape/seed/protocol:

| layout | train | final val |
|---|---:|---:|
| branch (default) | 1.3402 | **1.6280** |
| pre | 1.3197 | 1.6356 |

Branch wins by 0.0076 at 5k; the early 400-iter signal was misleading. Default stays
`branch`. Details: [real-model-review.md](../findings/real-model-review.md);
#11 resolved.

## Dataset-size context (historical kropotkin)

The old `LinA_init` runs used a larger corpus (`vocab 138`, kropotkin) and the sign was
the other way around - RAT beat real at 5k:

| run | params | kropotkin val | shakespeare val |
|---|---:|---:|---:|
| real 96/6/8 | 0.91M | 1.3325 | 1.6280 |
| RAT L1 84/12/8 | 0.86M | **1.3034** | 1.6849 |
| RAT L2 64/8/5 | 0.66M | **1.2992** | 1.6778 |

This is consistent with the small-dataset overfitting explanation, but it is **historical
evidence, not a controlled reproduction**: kropotkin was removed, the code has changed,
and the old runs are not directly comparable to today's shakespeare table. Reproduce it
with a licensed larger corpus in #13.

## Shared setup

| field | value |
|---|---|
| data | `examples/data/shakespeare.txt`, 80/20 split |
| tokenisation | char-level, vocab 65 |
| block_size | 128 |
| batch_size | 32 |
| dropout | 0.2 |
| seed | 1337 |
| optimiser | AdamW, lr 5e-4, weight_decay 0.01 (0.05 in the regularised follow-up), no schedule |
| screen | 2000 iters, eval 250 × 100 batches |
| final | 5000 iters, eval 500 × 200 batches |
| hardware | t14btw, CPU only (12 cores) |

## Runs

| phase | run | arch | L | n_embd | heads | layers | params | iters | status |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| final | final-real-96x8 | real | 0 | 96 | 6 | 8 | 0.90M | 5000 | done: val 1.6280 (92.3m) |
| screen | screen-rat-l1-h12 | rat | 1 | 84 | 12 | 8 | 0.83M | 2000 | TODO |
| screen | screen-rat-l1-h14 | rat | 1 | 84 | 14 | 8 | 0.83M | 2000 | TODO |
| screen | screen-rat-l1-h21 | rat | 1 | 84 | 21 | 8 | 0.83M | 2000 | TODO |
| screen | screen-rat-l1-72-h12 | rat | 1 | 72 | 12 | 8 | 0.62M | 2000 | TODO |
| screen | screen-rat-l2-h16-l5 | rat | 2 | 64 | 16 | 5 | 0.62M | 2000 | TODO |
| screen | screen-rat-l2-h16-l6 | rat | 2 | 64 | 16 | 6 | 0.74M | 2000 | TODO |
| final | final-rat-l1-h21 | rat | 1 | 84 | 21 | 8 | 0.83M | 5000 | running |
| final | final-rat-l2-h16-l5 | rat | 2 | 64 | 16 | 5 | 0.62M | 5000 | queued |

## Baseline result

The real GPT baseline finished before the sweep:

| run | train | val | params | minutes | run id |
|---|---:|---:|---:|---:|---|
| final-real-96x8 | 1.3402 | **1.6280** | 0.90M | 92.3 | `ce8e38df26534a11afb17bc57fa1c332` |

This is the number the RAT L1/L2 finalists have to match. Note it is a historical-recipe
baseline (constant LR, no schedule/weight-decay tuning), not a tuned GPT.

## Screen results

Filled from MLflow (`tardits-0.3.0-shakespeare`). Values are the step-2000 eval.

| run | train | val | params | minutes |
|---|---:|---:|---:|---:|
| screen-rat-l1-h12 | 1.3575 | 1.7099 | 0.83M | 43.1 |
| screen-rat-l1-h14 | 1.3551 | 1.7043 | 0.83M | 38.7 |
| screen-rat-l1-h21 | 1.3583 | **1.7031** | 0.83M | 32.1 |

At the 2k mark the three head counts are within 0.007 val loss and all are effectively
tied with the real baseline (1.7104). h21 wins by a hair and is also the fastest
(32.1 min vs 43.1), so it is the L1 finalist. The single L2 candidate (`h16/l5`) was
started as a 2k screen but superseded by the 5k final (the final run contains the same
2000-step point), so that screen run is marked KILLED rather than reported.

## Final results

| run | train | val | params | minutes | verdict |
|---|---:|---:|---:|---:|---|
| real baseline | 1.3402 | **1.6280** | 0.90M | 92.4 | reference |
| RAT L1 (h21) | 1.1886 | 1.6849 | 0.83M | 79.5 | gap +0.0569; did not match |
| RAT L2 (h16/l5) | 1.1719 | 1.6778 | 0.62M | 192.8 | gap +0.0498; did not match |
| RAT L1 + dropout/WD | 1.2399 | 1.6790 | 0.83M | 92.4 | gap +0.0510; marginal gain |
| RAT L2 + dropout/WD | 1.2208 | **1.6585** | 0.62M | 203.7 | gap +0.0305; best RAT |

Success criterion: L1 val ≤ real baseline; L2 val ≤ baseline + 0.02; params ≤ baseline.
The regularised L2 gets closest (+0.0305) but still misses the +0.02 bar; L1 misses by
more. The gap is real, not a report of a cherry-picked checkpoint. Wall-clock minutes are
inflated by Firefox/desktop contention on the shared laptop; loss numbers are unaffected.

### The interesting part: RAT is ahead early, real wins at 5k

Both RAT curves beat or match the real baseline through ~2.5k steps and then lose:

| step | real val | RAT L1 | RAT L2 |
|---:|---:|---:|---:|
| 2000 | 1.7104 | 1.7060 | **1.6751** |
| 2500 | 1.6967 | **1.6833** | **1.6678** |
| 3000 | **1.6693** | 1.6944 | 1.6662 |
| 3500 | **1.6571** | 1.6788 | 1.6611 |
| 5000 | **1.6280** | 1.6849 | 1.6778 |

RAT converges faster per parameter - L2 peaks at 1.6611 (step 3500) with 31% fewer
parameters - but then overfits: the train/val gap is 0.51 (L1) / 0.51 (L2) vs 0.29 for
the real model, and val turns upward while the real baseline is still improving. RAT only
applies dropout to the attention output; RealGPT also drops attention weights and the
SwiGLU output. The next experiment is a **dropout / AdamW weight-decay sweep** at the
same shapes (0.3–0.4), plus an equal-parameter (~0.62M) RealGPT baseline for a clean
parameter-efficiency comparison. The 5k finals stand as the honest release numbers.

### L1 verdict

1.6849 vs 1.6280 (+0.0569) at 0.83M params; best checkpoint 1.6788 at 3.5k. Regularised
L1 is 1.6790 (best 1.6769) - a marginal gain. Not a match.

### L2 verdict

1.6778 vs 1.6280 (+0.0498) at 0.62M params; best checkpoint 1.6611 at 3.5k. Regularised
L2 is **1.6585**, the best RAT result, still +0.0305 from the real baseline but 31% fewer
parameters and ahead of the real baseline until ~2.5k. Not a match on the absolute
baseline; the strongest parameter-efficiency result in the sweep.

## Regularised follow-up

Two 5k runs at the same shapes with `dropout 0.3` and AdamW `weight_decay 0.05`:

| run | config | val | best val | status |
|---|---|---:|---:|---|
| final-rat-l1-h21-d03-wd005 | `final/rat_l1_h21_d03.yaml` | 1.6790 | 1.6769 @4k | done |
| final-rat-l2-h16-l5-d03-wd005 | `final/rat_l2_h16_l5_d03.yaml` | **1.6585** | **1.6585 @5k** | done |

These are the direct test of the overfitting hypothesis. L1 improves only marginally:
1.6790 vs 1.6849 unregularised (+0.0059 at 5k; best-to-best 1.6769 vs 1.6788), with the
train/val gap shrinking from ~0.50 to ~0.44. L2 responds better: 1.6585 vs 1.6778
(+0.0193), the best RAT result in the sweep, but still +0.0305 from the real baseline.
Dropout/weight decay is therefore part of the story, not all of it - especially for L1.
The unregularised runs remain in the main table as the baseline comparison.

## Curves

- `docs/assets/curves/loss_curves.png`
- `docs/assets/curves/param_efficiency.png`
- `docs/assets/curves/runtime.png`
- `docs/assets/curves/gamma_layers.png`

Generated with:

```bash
.venv/bin/python docs/scripts/plot_mlflow.py \
  --experiment tardits-0.3.0-shakespeare \
  --out docs/assets/curves
```

## Decode speed (after training, KV-cached baseline)

`benchmarks/bench_inference.py` (prefill excluded, batch 1, CPU) now uses a **KV-cached**
RealGPT for a fair comparison. The corrected result: RealGPT and RAT L1 are tied at short
context, RAT L1 crosses over between 128 and 256 tokens and stays roughly flat
(~317–363 tok/s) while the cached real model falls from ~392 to ~215 tok/s at 2048
(1.48×; #18). RAT L2 is slower everywhere
(~105 tok/s). With the vectorised chunked prefill (#21) RAT is also faster end-to-end
beyond ~512 tokens (2.1× at 2048 for 16 generated tokens). The earlier
“185 → 108” real curve was uncached and is **not** a valid baseline; do not use it in the
README/paper. Full table/curve:
[docs/findings/inference-speed.md](../findings/inference-speed.md).

## Media

- 2D SAIL: `docs/assets/sail/4-layer.mp4`, `docs/assets/sail/8-layer.mp4`
- 3D SAIL: `docs/assets/sail/4-layer-3d.mp4`, `docs/assets/sail/8-layer-3d.mp4`
- gamma evolution: `docs/assets/gamma/gamma_evolution.mp4`
- posters: `docs/assets/sail/*-poster.png`, `docs/assets/gamma/poster.png`

The 4/8-layer interference runs are in `tardits-0.3.0-interference` (120 steps each,
0.26M / 0.50M params, val 2.3459 / 2.3648). The MP4s are gitignored; upload them to
YouTube and link the URLs in the README/docs (issue #3).

## MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# experiment: tardits-0.3.0-shakespeare
# experiment: tardits-0.3.0-interference
```

Run ids:

| run | run id |
|---|---|
| real baseline | `ce8e38df26534a11afb17bc57fa1c332` |
| L1 final h21 | `e45083777ecf4aad89f5c28e3f90e1c2` |
| L2 final h16/l5 | `4a14da2302544570b4f953548f93be69` |
| L1 dropout/WD | `c29151b0402e4ad79c3e83849de08df4` |
| L2 dropout/WD | `f1aa9aac79d0444492b9a1dc1a092988` |
| interference 4l | `c5cfb1f3c24446c89783e75870a820b3` |
| interference 8l | `1100f340617b4426b835410e61fee0f3` |

The full table is in `docs/assets/curves/summary.csv`.

## Notes / observations

- More heads remains the cheap speed lever at level 1; the screen tests whether it also
  helps quality at this corpus size.
- Level 2 is ~1.7–2.5× slower per step; the screen keeps only two L2 candidates to stay
  within the CPU budget.
- The real baseline uses the historical constant-LR AdamW setup, not a tuned GPT; the
  comparison is therefore "same recipe, different architecture".
