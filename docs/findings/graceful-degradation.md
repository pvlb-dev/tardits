# Graceful degradation - is RAT's capacity more distributed?

**Eval-only, 2026-10-31.** 11.24M RAT vs parameter-matched 11.13M RealGPT,
FineWeb checkpoints, no training. The question: if RAT's complex interference
stores information more densely, does zeroing residual channels hurt it less?

## Method

- Models: RAT 296×9 h37 (11.24M) and RealGPT 320×9 h8 (11.13M), both 22k iters
  / 90.1M FineWeb chars, seed 1337.
- Unit: one **residual-stream channel** (`n_embd`). The selected channels are
  zeroed at the token embedding output and at every block output, so those
  dimensions carry no signal through the stack.
- Fractions: 1, 5, 10, 25, 50 % of `n_embd` (296 vs 320; same *relative* width).
- Orderings:
  - **random**: seeded subsets, 2 masks averaged;
  - **magnitude**: largest mean |residual activation| first, calibrated on val
    batches.
- Evaluation: block 512, batch 4, 12 held-out FineWeb val batches. Delta from
  the intact checkpoint. Plot/CSV: `docs/assets/curves/graceful_degradation.{png,csv}`.
- Code: `benchmarks/bench_degradation.py`.

## Result

| fraction | RAT random Δ | Real random Δ | RAT magnitude Δ | Real magnitude Δ |
|---:|---:|---:|---:|---:|
| 1 % | +0.0024 | +0.0038 | +0.0466 | +0.0275 |
| 5 % | +0.0174 | +0.0194 | +0.1000 | +0.1065 |
| 10 % | +0.0435 | +0.0487 | +0.1388 | +0.1892 |
| 25 % | +0.1988 | +0.2090 | +0.3340 | +0.6181 |
| 50 % | +1.0451 | +1.0842 | +1.4448 | +2.1319 |

Intact val: RAT 1.3333, RealGPT 1.3403 (block 512).

## Reading

- **Random removal is essentially identical** between RAT and RealGPT. On the
  direct “distributed capacity” test, there is **no evidence** that RAT is more
  robust to arbitrary channel loss. The thesis, as stated, is not supported.
- **Magnitude-ordered removal does separate them at the high end**: at 25 % and
  50 %, RAT degrades substantially less (+0.334 vs +0.618; +1.445 vs +2.132).
  This says RAT's highest-activation channels are less individually essential -
  consistent with information being spread over more of the high-magnitude set -
  but it does **not** make random removal flat.
- Both models are **not uniformly distributed**: random removal hurts far less
  than magnitude-ordered removal (at 50 %, RAT 1.045 vs 1.445; Real 1.084 vs
  2.132). A small set of high-magnitude channels carries disproportionate loss in
  both architectures.

**Verdict:** the broad “complex interference → distributed capacity” claim is
not supported. The magnitude-order difference is a real, narrow signal worth one
follow-up (more seeds/batches, a single-layer ablation, and the 6.5M RAT point),
but it is not the predicted graceful-degradation story.

## Caveats

- Block 512 evaluation (models trained at 1024) and 12 batches / 2 random masks;
  noise is visible at the 1 % and 5 % points.
- Residual-channel ablation is one of several possible "units" (heads, MLP
  neurons, state channels); the conclusion is about residual width.
- Magnitude order is calibrated per model; the rank function differs, so the
  magnitude curves are not a controlled comparison of the same channels.
- 11M only; no 6.5M point yet.
- No training; this says nothing about what a differently regularised model
  could learn.
