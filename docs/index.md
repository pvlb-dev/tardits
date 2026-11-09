# TARDITS docs

Working notes for the v0.3.0 release: the real-model audit, the shakespeare and
FineWeb experiments, decay/`gamma` analysis, and the generation-speed measurements.
Plain Markdown + PNGs, so it works both as a Forgejo directory and as an Obsidian
vault (open `docs/` as the vault, or link this folder into an existing vault).

## Start here

- [README](../README.md) - the research narrative, architecture and install/quickstart.
- [Model internals](model-internals.md) - algebras, RAT attention, config, invariants.
- [Repository map](repo-map.md) - tree, commands, entry points.
- [AGENTS.md](../AGENTS.md) - working map: contracts, trigger table, srvbtw workflow.
- [Experiments](experiments/2026-10-25-shakespeare.md) - the staged real-vs-RAT study.
- [FineWeb 22k pair](experiments/2026-10-30-fineweb-22k.md) - first matched 45M RAT-vs-RealGPT result.
- [FineWeb RAT 33M](experiments/2026-10-30-fineweb-rat33m.md) - 70% of RealGPT's params, still ahead at equal tokens and wall-clock.
- [FineWeb RAT 22M/11M](experiments/2026-10-31-fineweb-rat22m11m.md) - width scaling.
- [11M reruns](experiments/2026-11-01-reruns.md) - data-order fix and the original Real lucky draw.
- [LR + hyper controls](experiments/2026-11-02-lr-hyper.md) - Real48 higher LR and the HyperFormer control.
- [Tiny memorization drift](experiments/2026-11-04-memorize.md) - RAT reproduces a 512-char sequence from a 16/32-char prompt.
- [Cheap trigger mini-NIAH](experiments/2026-11-05-trigger.md) - the trigger probe and its false positives.
- [Real48 lower-LR bracket](experiments/2026-11-07-real48-lr-low.md) - the 48M anomaly was the optimizer.
- [RAT44 1e-4](experiments/2026-11-07-rat44-lr1e4.md) - lower LR helps the 44M RAT; a timeout cut at 26.5k.
- [RAT33/RAT22 1e-4](experiments/2026-11-08-rat33-22-lr1e4.md) - the Real LR does not transfer to narrower RAT.
- [RAT33 3e-4, 34k](experiments/2026-11-08-rat33-lr3e4-34k.md) - the tuned two-seed tie with Real48.
- [Big-model speed](experiments/2026-11-08-inference-speed-big.md) - T4/CPU decode, throughput and memory.
- [Real-model review](findings/real-model-review.md) - what was checked and changed.
- [RAT decay notes](findings/rat-decay.md) - what `gamma` learns and how to read it.
- [Generation speed](findings/inference-speed.md) - decode tokens/s vs context length.
- [GPU probe](findings/gpu-probe.md) - measured T4 throughput/memory.
- [RAT state probe](findings/rat-state-probe.md) - training-free checkpoint diagnostics (method notes; the probe implementation lives in the research tree).
- [NIAH pilot](findings/niah-pilot.md) - clean null: no in-context retrieval at 0.8M params.
- [NIAH at FineWeb scale](findings/niah-fineweb.md) - the adjacent-copy control fails at 11-48M.
- [Graceful degradation](findings/graceful-degradation.md) - random channel removal is architecture-neutral.

## Current results

The real GPT baseline finished at **val 1.6280** (0.90M params). RAT L2 (h16/5 layers,
0.62M) with dropout+weight decay is the best RAT result at **1.6585** (+0.0305, 31% fewer
parameters); the unregularised L2 was 1.6778 and L1 1.6849. Both are **ahead of the real
model through ~2.5k steps** (L2: 1.6678 vs 1.6967 at 2.5k) and then overfit. Dropout/WD
helped L2 (+0.019) but only marginally helped L1 (+0.006). On decode speed, the fair
KV-cached baseline is now measured from 16 to 2048 tokens: the two are tied up to ~128
tokens, RAT L1 is ~flat (363 to 317 tok/s) while the cached real model falls from ~392 to
~215 tok/s at 2048 (1.48x decode). With the vectorised chunked prefill (#21) RAT is also
faster end-to-end beyond ~512 tokens (2.1x at 2048 for a 16-token generation). Retrieval
beyond the window is a clean null at this scale (NIAH), so long-context claims stay
speed-only for now. On the first matched 1B-char FineWeb pair (90.1M chars each,
22k iters), RAT L1 44.8M reached **val 1.2911** vs RealGPT 48.1M **1.5523** at
3e-4, at ~1.6x the wall-clock; a 33.4M RAT (69.4% of RealGPT) reaches
**1.2540 / 1.2902** over two seeds (mean 1.2721), i.e. no quality tax at 70%.
The tuned comparison is now a tie: **RAT33 3e-4/34k mean 1.2361** vs
**Real48 1e-4/36k mean 1.2355**, with RAT using **31% fewer parameters** and a
far smaller seed spread (RAT 0.0007, Real 0.026). The earlier 22k RAT33 pair
(1.2540/1.2902) and the 44M RAT (1.2911) are shorter/older rows. See the
[RAT33 long run](experiments/2026-11-08-rat33-lr3e4-34k.md).
Lower-LR controls and the big-model speed run are in
[RAT44](experiments/2026-11-07-rat44-lr1e4.md),
[RAT33/RAT22](experiments/2026-11-08-rat33-22-lr1e4.md) and
[big-model speed](experiments/2026-11-08-inference-speed-big.md).
A first width-scaling run put **11.2M at 1.2684** and **11.1M RealGPT at
1.2903**; a data-order-fixed MLflow rerun gives **1.2845 vs 1.3253** on identical
batches. A **6.47M RAT** degrades to 1.3669. See
[the rerun report](experiments/2026-11-01-reruns.md), the
[48M LR correction](experiments/2026-11-07-real48-lr-low.md), the
[FineWeb 22k report](experiments/2026-10-30-fineweb-22k.md) and the
[33M follow-up](experiments/2026-10-30-fineweb-rat33m.md). The earlier "RAT 2.7x"
chart was an uncached-baseline artifact. See [generation speed](findings/inference-speed.md)
and the [NIAH pilot](findings/niah-pilot.md); see
[the experiment report](experiments/2026-10-25-shakespeare.md) for the honest verdict.
It tracks, for every run:

| field | why |
|---|---|
| arch / level / n_embd / n_head / n_layer | model shape |
| trainable params | parameter-efficiency claim |
| final train / val loss | quality |
| wall-clock minutes | CPU cost |
| MLflow run id | traceability |

## Videos and graphics

The committed figures live in `assets/`: the loss/parameter/speed curves under
`assets/curves/`, the decay plots under `assets/gamma/`, and the two architecture
diagrams under `assets/art/`. The SAIL and gamma videos are generated locally with
`docs/scripts/make_videos.sh` and are not committed; the README links the YouTube
versions. `docs/scripts/plot_mlflow.py` regenerates the curves and
`docs/scripts/analyze_gamma.py` regenerates the gamma plots.

## Reproduce

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv sync --extra mlflow --locked
export MPLCONFIGDIR="$PWD/.cache/matplotlib"

.venv/bin/ttrain -c examples/configs/rat.yaml
.venv/bin/tgen -c examples/configs/rat.yaml -p "To be or not to be" -l 200
.venv/bin/pytest -q

mlflow ui --backend-store-uri sqlite:///mlflow.db
.venv/bin/python docs/scripts/plot_mlflow.py
.venv/bin/python docs/scripts/analyze_gamma.py \
  --checkpoint examples/saved/shakespeare/rat_l1_winner.pt
```

The research run configs and the RAT2 state probe are not part of this demo repo;
the dated reports record what was run and where the evidence lives.

## Layout

```
docs/
├── index.md                    # this page
├── model-internals.md          # algebras, RAT attention, config, invariants
├── repo-map.md                 # tree, commands, entry points
├── experiments/                # dated reports with run tables and verdicts
├── findings/                   # audit, decay, speed, GPU, NIAH, degradation
├── assets/
│   ├── curves/                 # committed PNGs + source CSVs
│   ├── gamma/                  # decay plots + poster
│   └── art/                    # the two architecture diagrams
└── scripts/                    # plot_mlflow.py, analyze_gamma.py, make_videos.sh
```
