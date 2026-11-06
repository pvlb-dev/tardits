# AGENTS.md - TARDITS working map

> **If unsure, ask.** One short question beats a wrong assumption or a long
> detour. Do not guess, and do not disappear into research to avoid asking.

**This file is a map, not the manual.** The harness loads it as instructions and
truncates past 65,536 bytes from the tail, so detail lives in the linked topic
files. If a task touches a row in the trigger table below, read that file first.

## Orientation - pvlb.dev

This repo lives on the pvlb.dev forge: <https://git.pvlb.dev/ml-lab/tardits>.
`master` is the default branch; `release/0.3.0` is the current release tree.

Before starting, read the bootcamp:
<https://git.pvlb.dev/ops/srvbtw/wiki/DSH-Bootcamp>. It covers the box (srvbtw,
Arch, 4 threads, 8 GiB, CPU-only), the `dsh <dsh@pvlb.dev>` identity (global and
GPG-signed), `git` over SSH versus the `gitea_*` tools, and the house rules.

- Every issue is labeled at creation with `type/*`, `area/*` and `priority/*`;
  add `needs-operator` only for work only the operator can do.
- The project has an **input box**: `input/` -> `devops/input/` (local only,
  gitignored). A change outside the project gets a `note` issue in `ops/srvbtw`;
  a task needing a fresh context gets `NaNs` plus a handoff in the body; a
  bootcamp gap gets `bootcamp`.
- This repo is **public**. Never commit secrets, private paths, checkpoints or
  unpublished data.
- Commit as `dsh`, conventional and signed, then **push**. Never force-push
  `master`; never rewrite a release commit or another project's history.
- Be gentle: the box is shared with the DSH and CI. No CUDA.

Rules below are project-specific and win over the bootcamp.

## What this project is

**TARDITS** - *Text And Rotating Dimensions In Token Space* - is a research and
portfolio library of complex- and hypercomplex-valued language models in
PyTorch. Hypercomplex multiplication is itself non-linear, so the transformer
needs no activation function (the **SAIL** feed-forward layer); the
Cayley-Dickson construction scales one architecture from the complex plane
through quaternion, octonion and sedenion. **RAT** is the linear-attention
family, formerly `LinA`, which is the dev and research branch.

| `arch` | Class | Attention | Notes |
|---|---|---|---|
| `real` | `RealGPT` | causal softmax, RoPE, SwiGLU | KV cache, chunked prefill |
| `hyper` | `HyperFormer` | Hermitian `QK` dagger, real part, softmax | GyRoPE |
| `rat` | `RAT` | linear fast-weight memory | learned complex decay, O(1) decode |

Char-level tokenizer, Python >= 3.13, `src/` layout, CPU-first. `ttrain` and
`tgen` drive every family from `model.arch`; shipped configs are validated by
`tests/test_registry.py`.

> Transition note: on `master` before PR #3 merges, `docs/`, `tests/`,
> `benchmarks/` and `RAT` exist only on `release/0.3.0`. The rows below describe
> the merged 0.3.0 tree.

## Quick start

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv sync --extra mlflow --locked
.venv/bin/ttrain -c examples/configs/rat.yaml
.venv/bin/tgen   -c examples/configs/rat.yaml -p "To be or not to be" -l 200
.venv/bin/pytest -q            # 60 tests; unittest discover also works
```

MLflow is optional: without the extra the logger no-ops. Checkpoints, MP4s and
GIFs stay in `examples/saved/` (gitignored); committed graphics live in
`docs/assets/`.

## Trigger table - read before you touch it

| If the task touches ... | Read |
|---|---|
| git identity, commits, signing, a new clone | [`GIT.md`](GIT.md), local only |
| issues, input box, `note` and `NaNs`, experiments, media, release | [`devops/workflow.md`](devops/workflow.md), local only |
| model code, algebras, RAT attention, configs, checkpoints, CLIs | [`docs/model-internals.md`](docs/model-internals.md) |
| file and directory layout, commands, test and bench entry points | [`docs/repo-map.md`](docs/repo-map.md) |
| benchmarks, or any speed claim | [`docs/findings/inference-speed.md`](docs/findings/inference-speed.md) |
| long context, NIAH, retrieval claims | [`docs/findings/niah-pilot.md`](docs/findings/niah-pilot.md), [`docs/findings/niah-fineweb.md`](docs/findings/niah-fineweb.md) |
| Kaggle, GPU, research-scale FineWeb runs | [`docs/findings/kaggle-training.md`](docs/findings/kaggle-training.md), [`docs/findings/kaggle-quota-jobs.md`](docs/findings/kaggle-quota-jobs.md), [`docs/findings/gpu-probe.md`](docs/findings/gpu-probe.md), [`docs/findings/fineweb-plan.md`](docs/findings/fineweb-plan.md) |
| running or reporting an experiment | [`docs/experiments/`](docs/experiments/), [`docs/index.md`](docs/index.md) |
| auditing a RAT checkpoint (decay horizon, max absolute gamma, state deltas) | [`docs/findings/rat-state-probe.md`](docs/findings/rat-state-probe.md) |
| README, translations, docs, mascot, art, videos | [`docs/README.md`](docs/README.md), [`docs/assets/`](docs/assets/) |
| the release, versioning, tag, PyPI, Forgejo registry | issue [#3](https://git.pvlb.dev/ml-lab/tardits/issues/3) |
| a house rule or bootcamp change | the bootcamp link above; label `bootcamp` or `note` |

## Hard invariants (the ones that bite)

1. **RAT `forward` equals `step`.** Route every memory update through
   `Rattention._decay_add`; never extend the de Moivre scan to level >= 4.
   `tests/test_rat_equivalence.py` guards it; the trained-gamma L1 mismatch is a
   known pre-existing bug, do not paper over it by relaxing the test.
2. **Never quote decode speed without the KV-cached RealGPT baseline.** Fair
   numbers: RealGPT 392 to 215 tok/s over 16 to 2048; RAT L1 roughly flat 363 to
   317 (1.48x at 2048), and RAT is faster end-to-end beyond about 512 tokens
   (2.1x at 2048). Use `benchmarks/bench_inference.py --repeats 12` as the
   minimum.
3. **No `nn.Dropout` on complex tensors.** Use `HyperDropout`; the former raises
   for `ComplexFloat`.
4. **`block_size` is the training window, not a generation cap.** RAT decode is
   recurrent; RealGPT chunks through `prefill`. Beyond the window both models
   extrapolate, so only speed claims are fair.
5. **Level cost grows about 4x per level.** Prefer small `n_embd` and
   `block_size` at level >= 3; `HyperLinear` is BLAS-backed and `cd_mul` is the
   hot path.
6. **`model.arch` selects the family** (`real|hyper|rat`). A new config key is
   either a dataclass field or popped before `ModelConfig(**cfg)`.
7. **`tardits` is a namespace package** resolving to `src/tardits`; no model code
   at the repo root.

## Repo layout

```text
AGENTS.md            this map (tracked)
README.md            research narrative, architecture, benchmarks (+ .de/.cs/.zh/.ru)
docs/                model internals, repo map, findings/, experiments/, assets/
src/tardits/         package: train.py and gen.py CLIs, models/, modules/
tests/               pytest suite; test_registry validates every shipped config
benchmarks/          attention, inference, NIAH, trigger, memorization, degradation
examples/            configs/*.yaml and data/shakespeare.txt; saved/ is gitignored
devops/              LOCAL ONLY: workflow.md and the input drop box (gitignored)
GIT.md               LOCAL ONLY: identity and signing bootstrap (gitignored)
```

`devops/` and `GIT.md` are deliberately not committed, because the repo is
public. The map's local triggers resolve only in the resident checkout; in a
fresh clone, use the bootcamp and `docs/`.

## Session protocol

1. Read this map, then `git status`, `git fetch`, and the current branch.
2. Small reversible steps, verified with a real command (`pytest -q` for code,
   `bench_inference.py --repeats 12` for a speed claim).
3. Conventional signed commit (`feat:`, `fix:`, `docs:`, `test:`), then push.
4. Record the change: update `docs/index.md` or the matching finding or
   experiment report when a result changes, otherwise the Forgejo issue. A
   change is not finished until it is recorded.
5. Do not touch the release commit, `master` history, or another project's
   checkout.

## Where the truth lives

This map, then the `docs/` topic files, then the bootcamp wiki, then the issues
and experiment reports. Higher wins; fix the lower in the same pass.

- Experiment reports: [`docs/experiments/`](docs/experiments/)
- Results summary and backlog: [`docs/index.md`](docs/index.md), Forgejo issues
- Release state: issue [#3](https://git.pvlb.dev/ml-lab/tardits/issues/3)
- Session status: the open issues, not this file.

## House rule: keep this file small

The harness truncates past 65,536 bytes from the tail. New detail goes into a
`docs/` file with a trigger row, never inline here.
