# Repository map

The map is [`../AGENTS.md`](../AGENTS.md); model internals are in
[`model-internals.md`](model-internals.md); the local-only workflow notes are
gitignored and are not part of a fresh clone.

```text
tardits/
├── AGENTS.md                  # map: rules, quick start, trigger table
├── README.md                  # research narrative, architecture, benchmarks
├── README.{cs,de,ru,zh}.md    # translations (English is canonical)
├── LICENSE
├── pyproject.toml             # hatchling; deps; ttrain/tgen console scripts
├── uv.lock, .python-version, .gitignore
│
├── .forgejo/workflows/        # ci, publish (Forgejo registry), publish-pypi
│
├── src/tardits/               # the distributable package (namespace package)
│   ├── train.py               # `ttrain` CLI (real / hyper / rat)
│   ├── gen.py                 # `tgen` CLI (any arch; -f streams)
│   ├── models/
│   │   ├── base.py            # ModelConfig dataclass
│   │   ├── registry.py        # arch -> class, build_model(), aliases
│   │   ├── real.py            # RealGPT: RoPE, SwiGLU, RMSNorm, KV-cache/prefill
│   │   ├── hyper.py           # HyperFormer: Hermitian softmax hypercomplex
│   │   └── rat.py             # RAT: linear fast-weight memory (ex-LinA)
│   └── modules/
│       ├── core.py            # Cayley-Dickson algebra + hypercomplex layers
│       ├── base.py            # get_device(), LMCoreMixin
│       ├── trainer.py         # Trainer + TrainerConfig
│       └── logging.py         # Console/MLflow loggers
│
├── tests/                     # unittest/pytest suite (60 tests)
│   ├── test_core.py, test_config.py, test_dropout.py, test_gyrope.py
│   ├── test_hyperlinear.py, test_rat_equivalence.py, test_real.py
│   └── test_registry.py       # validates every shipped config
│
├── benchmarks/
│   ├── bench_attention.py     # n_head sweep (attention time vs head_size)
│   ├── bench_inference.py     # decode/prefill vs context 16-2048
│   └── bench_niah.py          # minimal needle-in-a-haystack pilot
│
├── examples/
│   ├── configs/*.yaml         # real, complex, quaternion, octonion, sedenion, rat, inter
│   ├── data/shakespeare.txt   # char-level corpus
│   └── saved/                 # checkpoints/GIFs (gitignored except .gitkeep)
│
├── docs/
│   ├── index.md               # entry point for README + Obsidian
│   ├── model-internals.md     # algebras, RAT, config, invariants
│   ├── repo-map.md            # this file
│   ├── experiments/           # dated reports with run tables and verdicts
│   ├── findings/              # audit, decay, speed, GPU, NIAH, degradation
│   ├── assets/                # curves, gamma, architecture diagrams
│   └── scripts/               # MLflow plots, gamma analysis, video helpers
│
└── local only (gitignored): devops/ (agent notes + input box), GIT.md, input/
```

## Commands

```bash
# setup
UV_CACHE_DIR="$PWD/.uv-cache" uv sync --extra mlflow --locked

# train / generate
.venv/bin/ttrain -c examples/configs/rat.yaml
.venv/bin/tgen -c examples/configs/rat.yaml -p "To be or not to be" -l 200

# tests (pytest is canonical; unittest discovery also works)
.venv/bin/pytest -q

# benchmarks
.venv/bin/python benchmarks/bench_attention.py -c examples/configs/rat.yaml
.venv/bin/python benchmarks/bench_inference.py        # --repeats 12 (default)
.venv/bin/python benchmarks/bench_niah.py

# MLflow + docs figures
mlflow ui --backend-store-uri sqlite:///mlflow.db
.venv/bin/python docs/scripts/plot_mlflow.py
.venv/bin/python docs/scripts/analyze_gamma.py \
  --checkpoint examples/saved/shakespeare/rat_l1_winner.pt
```

## Experiments

Reports live in [`experiments/`](experiments/), one dated file per sweep; the
headline results are summarised in [`index.md`](index.md). The research run
configs and the RAT2 state probe are not shipped in this demo repo; the reports
record the shapes, budgets and evidence.

## Assets

`docs/assets/curves/` and `docs/assets/gamma/` hold the committed figures and
their source CSVs; `docs/assets/art/` holds the two architecture diagrams. The
mascot art, videos and full-resolution originals stay out of the public repo.
