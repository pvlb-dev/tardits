# docs/ - how this folder is meant to be used

This is the docs folder for the v0.3.0 release. It is intentionally plain
Markdown with relative links so it works in:

- Forgejo/GitHub (start at `docs/index.md`),
- Obsidian (open this directory as a vault, or add it to an existing vault),
- a plain editor.

## Layout

```
docs/
├── index.md                       # start here
├── model-internals.md             # algebras, RAT attention, config, invariants
├── repo-map.md                    # tree, commands, entry points
├── experiments/<date>-<topic>.md  # dated reports with run tables and verdicts
├── findings/*.md                  # audit, decay, speed, GPU, NIAH, degradation
├── assets/curves/                 # committed PNGs + source CSVs
├── assets/gamma/                  # decay plots + poster
├── assets/art/                    # the two architecture diagrams
└── scripts/                       # reproducible plotting/video scripts
```

## Regenerating graphics

```bash
export MPLCONFIGDIR="$PWD/.cache/matplotlib"

.venv/bin/python docs/scripts/plot_mlflow.py \
  --experiment tardits-0.3.0-shakespeare \
  --out docs/assets/curves

.venv/bin/python docs/scripts/analyze_gamma.py \
  --checkpoint examples/saved/shakespeare/rat_l1_winner.pt \
  --out docs/assets/gamma

docs/scripts/make_videos.sh
```

The MP4s are gitignored; upload them to YouTube and put the URLs into the experiment
report / README. Poster PNGs and curves are small enough to commit.

## Obsidian conventions

- Use normal relative Markdown links (`[text](findings/rat-decay.md)`) so Forgejo and
  Obsidian both resolve them.
- Wikilinks (`[[rat-decay]]`) are fine inside the vault, but don't use them in files that
  must render on the forge; keep a relative link alongside.
- One dated report per sweep; link the MLflow run ids and the config files, not pasted
  metric dumps.
- Keep the "results" sections honest: include the lost runs and the gaps.

## What does not belong here

- Checkpoints, GIFs, raw MLflow artifacts (`examples/saved/`, `mlruns/` are gitignored).
- Secrets/tokens (never).
- The main README narrative - that is the release-repo overhaul, tracked separately.
