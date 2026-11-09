# Model internals - algebras, RAT, config, invariants

Deep reference for the code. The map is [`../AGENTS.md`](../AGENTS.md); workflow
(git/issues/tests) is [`../AGENTS.md`](../AGENTS.md).

## `level` - the hypercomplex dimension

`level` selects the algebra. All complex-valued tensors use `torch.complex64`
with a trailing `hyper_dim` axis of `2**(level-1)` complex components
(= `2**level` real dims):

| `level` | Algebra | `hyper_dim` | Real dims | Notes |
|--------:|---------|------------:|----------:|-------|
| 0 | real | - | 1 | `RealGPT` only (`arch: real`) |
| 1 | complex ℂ | 1 | 2 | commutative, associative |
| 2 | quaternion ℍ | 2 | 4 | non-commutative, associative |
| 3 | octonion 𝕆 | 4 | 8 | non-commutative, **non-associative but alternative** |
| 4 | sedenion 𝕊 | 8 | 16 | **non-alternative** (zero divisors) |
| ≥5 | 2^level-dim | `2^(level-1)` | `2^level` | e.g. level 7 = 128-D; extremely slow |

The alternative/non-alternative split at 3→4 is load-bearing for RAT's memory
(see invariants). `RAT` requires `level >= 1`; `resolve_arch` raises for
`arch: rat` + `level: 0`.

## Cayley-Dickson algebra (`src/tardits/modules/core.py`)

- `cd_mul(a, b, level)` / `cd_conjugate(x, level)` - recursive CD multiplication
  and conjugation on the trailing `hyper_dim` complex axis. Correct for all
  levels (level 1 = plain complex multiply). Used for **elementwise** products
  (attention outer products, SAIL interference).
- `HyperLinear` - hypercomplex linear layer.
  - level 1: complex `nn.Linear` fast path.
  - level ≥ 2: real-matrix BLAS matmul via the cached
    `_cd_multiplication_table(level)`. The weight stays `[out, in, hyper_dim]`
    complex (checkpoint compatible); it is spread into a real
    `[out*n, in*n]` matrix and applied with `F.linear`.
- `HyperRMSNorm` - normalizes amplitudes over the feature axis, preserves phase.
- `SAIL` - *Self-Activating Interference Layer*: project to `w1(x)`, `w2(x)`,
  hypercomplex-multiply them, project back with `w3`. SwiGLU without the
  activation; the nonlinearity is the hypercomplex product. Exposed as
  `model.blocks[i].sail` (what MLflow phase plots hook). `RealGPT` uses a
  SwiGLU `block.ffwd`, so deep SAIL logging is a no-op for it.
- `HyperEmbedding`, `HyperLMHead` - token ↔ hypercomplex features; the LM head
  flattens real+imag parts back to real logits.
- `HyperDropout` - complex-safe dropout: `nn.Dropout` cannot run on complex
  tensors (invariant 5), so one real mask is shared across real/imag parts.
  Used by RAT on its attention output.
- `GyRoPE` - learnable hypercomplex RoPE; used by `HyperFormer`, commented out
  in `rat.py` (RAT has no active positional rotation).

## RAT fast-weight attention (`src/tardits/models/rat.py`)

Linear attention replaces softmax attention with an associative memory:

```
S_t = S_{t-1} ⊗ γ + v_t ⊗ k̄_t          # decay + outer-product ingestion
out_t = (S_t ⊗ q_t) · scale             # recall
```

- `Rattention.forward` trains in parallel (prefix scan); `Rattention.step` runs
  the recurrence for generation. (`Rattention` is the renamed ex-`LinearAttention`;
  the module attribute is still `block.sa`, so checkpoints/state-dict keys are
  unchanged.)
- Levels 1–3 use the **de Moivre parallel scan** (`_compute_gamma_powers`),
  provably identical to the recurrence (associativity; octonion alternative ⇒
  Artin). Level 1 has a cheaper `gamma**t` branch for training.
- Level ≥ 4 uses the **exact sequential recurrence** (`_decay_add`): the
  non-alternative algebra breaks scan/recursion equivalence.
- `γ` is a parameter initialised on the unit hypersphere; its magnitude is the
  per-step decay/gain. `|gamma|` distributions are logged to MLflow by
  `MLFlowLogger._log_gamma_histograms` (RAT-only, guarded by `hasattr`).
- **Matched data order:** `TrainerConfig.seed` drives a dedicated data RNG;
  models with the same seed see identical training and eval batches regardless
  of how their initialisation consumed the global RNG. Use it for every
  architecture comparison.
- **Decay bound:** RAT's rotor is initialised on the unit sphere, but nothing
  stops training from pushing `|gamma| > 1`, which turns the decay into a gain
  and grows the state without bound (`1.001^48000 ≈ 7e20`). `Trainer` now
  projects each `gamma` magnitude back onto the `gamma_max` ball (default 1.0)
  after every optimizer step; a no-op while all |gamma| <= 1. See
  #27.
- **Checkpoint audit:** `experiments/rat2/probe.py` extracts per-block state deltas
  exactly (float64 prefix scan, self-tested against `step`) and reports per-layer
  mean `|gamma|`, decay horizon, max `|gamma|` and channels `> 1`; `capacity.py` /
  `retrieval.py` measure readout history share and state-delta capacity. Audit
  `max |gamma|` before any long-context state work. Usage:
  [`docs/findings/rat-state-probe.md`](findings/rat-state-probe.md).
- **State-size law:** the fast-weight state is `n_head · head_size² =
  n_embd · head_size`. At fixed `n_embd`, **more heads = smaller state**; the
  capacity direction is *fewer, fatter* heads. Softmax head-count intuition is
  backwards for RAT memory (and the GPU probe measured h64 as 1.5× faster than
  h32 at ~equal params).
- **Bidirectional mode:** set `model.bidirectional: true` for a denoiser; the
  kernel is the symmetric `S_t = Σ_s γ^{|t−s|} A_s = scan(A)_t + scan(A.flip)_t.flip − A_t`
  (no RoPE needed), full-window only (≤ the scan chunk), and generation/recurrent
  state are rejected (`RAT.generate/step/prefill` raise). Reversal-equivariance is
  tested for levels 1–4; see #24 and
  the dagoRAT port notes (research tree).
- **Bidirectional mode** is documented above; the dagoRAT write-up is digested
  in the dagoRAT port notes (research tree) and
  the feature is tracked as #24.
- Inference prefill is `RAT.prefill`: a **chunked** scan (chunk bounded by the
  trained `|gamma|`) with the state carried between chunks, plus a forced de
  Moivre branch for level 1. It matches `_warm_states` and is much faster than
  token-by-token warmup; see `TestRATPrefill` and
  [`findings/inference-speed.md`](findings/inference-speed.md).
- Known training bug: the level-1 `gamma**t` inverse branch loses memory once
  `|gamma_t| < 1e-8` (inside a 128-token window for trained checkpoints), so
  `forward != step` there. Tracked as
  #23; inference is unaffected.

## Configuration

Training and generation are driven by a YAML file; each section maps to a
dataclass that lives next to the class which owns it.

| YAML section | Fields | Defined in |
|---|---|---|
| `model` | `ModelConfig` + `arch` selector | `src/tardits/models/base.py`, arch handled by `registry.py` |
| `trainer` | `TrainerConfig` | `src/tardits/modules/trainer.py` |
| `mlflow` | `MLFlowConfig` | `src/tardits/modules/logging.py` |

```yaml
device: "auto"        # auto | cuda | cpu | mps
seed: 1337
model:
  arch: rat           # real | hyper | rat  (optional; legacy: level<1 ? real : hyper)
  level: 1
  block_size: 128     # training window
  n_head: 12
  n_embd: 84
  n_layer: 8
  dropout: 0.2        # real/hyper on softmax weights; RAT on attention output
  norm: branch        # RealGPT only: branch (default) | pre
  bidirectional: false # RAT only: symmetric decay for diffusion; generation disabled
trainer:
  batch_size: 32
  max_iters: 5000
  eval_interval: 500
  learning_rate: 0.0005
  weight_decay: 0.01  # AdamW
  gamma_max: 1.0      # RAT only: project |gamma| <= 1 after each step
  seed: 1337          # data-order RNG, independent of model init
  eval_iters: 200
  save_interval: 0
  checkpoint: "examples/saved/shakespeare/rat_l1.pt"
mlflow:               # logging only enabled when `experiment` is non-empty
  experiment: "tardits-0.3.0-shakespeare"
  run_name: "rat-l1-84x8-h12"
  tags: {dataset: shakespeare, seed: "1337"}
  log_interval: 10
  deep_log_interval: 0
  system_metrics: false
data:
  input_path: "examples/data/shakespeare.txt"
  vocab_save_path: "examples/saved/shakespeare/vocab.txt"
```

Shipped configs (validated by `tests/test_registry.py`): `complex`,
`quaternion`, `octonion`, `sedenion` → `hyper`; `rat`, `inter` → `rat`;
`real` → `real`. Tokenization is **char-level**: the vocab is built at train
time and written to `data.vocab_save_path`; generation re-reads it.

### Model selection

`ModelConfig` does not carry the family; the config key `model.arch`
(`real|hyper|rat`, `-a` override) selects the class through
`MODEL_REGISTRY` in `src/tardits/models/registry.py`. Aliases: `lina`→`rat`,
`gpt`/`realgpt`→`real`, `hyperformer`→`hyper`; matching ignores case, spaces,
`-`, `_`. Without `arch`, legacy rule: `level < 1` → `real`, else `hyper`. New
configs should be explicit. `arch` is popped before `ModelConfig(**...)`, so a
new selector needs popping and a new hyperparameter needs a dataclass field.

## Checkpoints

`LMCoreMixin.save_checkpoint` writes `{"model_state_dict": …, "config":
<ModelConfig>}` via `torch.save`. `load_from_checkpoint(cls, path, device)`
reconstructs `cls(config)` then `load_state_dict` (strict); `weights_only=False`
because the config is a dataclass.

- The checkpoint does **not** record which class wrote it; `tgen` needs the YAML
  or `-a`. Self-describing metadata is
  #5.
- RAT state-dict keys are identical to pre-rename `LinA`, so old
  `examples/saved/LinA_*.pt` load into `RAT` unchanged.
- `examples/saved/` is gitignored except `.gitkeep`. Some old committed `.pt`
  files predate the current `Rattention` layout (`rope`/`norm` submodules) and no
  longer load - pre-existing, not a regression.
- Config class path matters: checkpoints saved before the dataclasses moved out
  of `conf.py` reference `tardits.modules.conf.ModelConfig` and will not
  unpickle; current ones reference `tardits.models.base.ModelConfig`.

## Invariants and gotchas

1. **`forward` must equal `step` (RAT).** Route every memory update through
   `Rattention._decay_add`. Do **not** extend the de Moivre scan to level ≥ 4.
   `tests/test_rat_equivalence.py` checks levels 1–4 (tiny/unit-gamma regime;
   see #23 for trained-gamma L1).
2. **`HyperLinear` level ≥ 2 relies on `_cd_multiplication_table`.** Keep the
   weight `[out, in, hyper_dim]` complex; the real matrix is derived per
   forward (cheap) and must stay consistent with recursive `cd_mul`.
3. **Namespace package.** `tardits` resolves to `src/tardits` (no top-level
   `__init__.py`); it imports from the repo root without installation. Do not
   add model code at the repo root again.
4. **RAT overrides both generation paths.** `LMCoreMixin.generate` /
   `generate_stream` re-run the full window per token (O(T²)); RAT overrides
   with the O(1) recurrent `step`, so `RAT.generate` ignores `block_size` and
   `generate_stream` never re-reads the prompt. Prompt ingestion is
   `RAT.prefill` (chunked scan, state carry). `RealGPT` uses the mixin for
   `generate`; its long-context path is `prefill` + `generate_cached`. Adding a
   model: decide which path it gets.
5. **`nn.Dropout` cannot run on complex tensors** (`bernoulli_scalar_cpu_` not
   implemented for `ComplexFloat`). Use `HyperDropout` for any complex
   activation. RAT applies it after `out_proj` in both forward/step.
6. **Level cost is exponential.** Hypercomplex multiply is O(4^level) real ops;
   level 4 ≈ 60× level 1 per element. Prefer small `n_embd`/`block_size` for high
   levels (`examples/configs/sedenion.yaml`). `HyperLinear` is BLAS-backed; the
   remaining hot path is elementwise `cd_mul` in attention/SAIL.
7. **Level-1 attention cost scales as `n_embd² / n_head`.** The memory is
   `[B, n_head, T, head_size, head_size, hyper_dim]`, so cost is
   `B·T·n_embd²/n_head` with `head_size = n_embd/n_head`: **more heads is
   quadratically cheaper at ~unchanged params**. Measured on `rat.yaml`
   (n_embd=84, T=128, 8 layers, CPU): n_head 6 ≈ 2013 ms/iter, n_head 12 ≈
   972 ms/iter. `head_size² ≈ block_size` is the crossover; 12 heads
   (head_size 7) is the current sweet spot. Reproduce:
   `.venv/bin/python benchmarks/bench_attention.py -c examples/configs/rat.yaml`.
8. **Adding a config key.** `model:` keys are splatted into `ModelConfig(**…)`:
   a selector must be popped (like `arch`), a hyperparameter must be a dataclass
   field, or `ttrain` fails with `unexpected keyword argument`.
9. **CPU-first environment.** No CUDA/MPS here; `get_device("auto")` → CPU.
   Validate timing/tuning assumptions on CPU. Kaggle T4 path:
   the GPU run notes (research tree).
10. **Shape preconditions live in `ModelConfig.__post_init__`.** `n_embd`
    divisible by `n_head`, all dims positive. `RealGPT` needs even `head_size`
    (RoPE pairs); uncached/training forward with `T > block_size` raises, but
    `RealGPT.prefill` feeds `block_size` chunks while the RoPE cache grows
    lazily (`_extend_cache`). RAT is not window-limited for generation. Beyond
    `block_size` both models are extrapolating - speed-only claims.
11. **Branch norm is the default.** `model.norm: branch|pre` (RealGPT only):
    `branch` = `x + ln(sublayer(x))`, `pre` = `x + sublayer(ln(x))`. Full 5k A/B:
    branch 1.6280 vs pre 1.6356 (pre overfits more). Keep `branch` unless a new
    full-length run says otherwise. See
    [`findings/real-model-review.md`](findings/real-model-review.md).
12. **RealGPT has no `block.sail`.** Deep SAIL phase logging is hyper/RAT-only;
    the logger returns early. `_log_gamma_histograms` is RAT-only.
13. **MLflow is optional.** `uv sync --extra mlflow --locked` installs it;
    without it `MLFlowLogger` no-ops. Tracked runs must log to local
    `mlflow.db`/`mlruns/`.
14. **`seed: 0` must seed.** `train.py` checks `if seed is not None`, not
    truthiness; keep that when touching the config path.
15. **Media artifacts are large.** Keep checkpoints/GIFs/MP4s under
    `examples/saved/` (gitignored); commit only curves/posters/art in
    `docs/assets/`. Videos are generated locally, not committed.
16. **Never quote decode speed without the KV-cached RealGPT baseline.** The old
    uncached chart (real 185→108 tok/s) made RAT look 2.7× faster; it is invalid.
    With `RealGPT.prefill` + `RAT.prefill`, real decode is 392→215 tok/s over
    16→2048 while RAT L1 is ~flat (363→317, 1.48× at 2048); RAT is also faster
    end-to-end beyond ~512 tokens (2.1× at 2048). Use
    `benchmarks/bench_inference.py` (`--repeats 12`, minimum) and
    [`findings/inference-speed.md`](findings/inference-speed.md). Beyond 128 is
    speed-only - NIAH is a clean null
    (#20).
