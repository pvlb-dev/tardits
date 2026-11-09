# docs/assets

Small, versioned figures used by the README and the docs. Videos are generated
locally and ignored by git; upload them to YouTube and link the URLs.

## Committed

- `curves/*.png` - the FineWeb loss/scaling curves, the small- and big-model
  inference charts, the memorization plots and the parameter-efficiency/runtime
  figures used by the reports.
- `curves/*.csv` - the source data behind those plots (including the new
  `throughput_*`, `inference_cpu_*` and `inference_big_*` runs).
- `gamma/*.png` - per-layer decay, memory horizon, complex-plane density and the
  gamma poster frame; `gamma/gamma_stats.csv` holds the per-layer numbers.
- `art/rat-hero.jpg`, `art/rat-architecture.jpg`, `art/hyper-architecture.jpg` -
  the README hero and the two architecture diagrams.
- `legacy/*.png` - the three older-run plots (complex vs real, real vs the
  hypercomplex rest, and the v0.2.0 centumduodevigintunions bonus), restored from
  the GitHub mirror because the original user-attachment links do not render on
  the forge.
- `README.md` - this guide.

The mascot variants, full-resolution originals and the video MP4s stay out of
the public repo; the hero and the two architecture diagrams are committed.

## Generated locally (gitignored)

- `sail/4-layer.mp4`, `sail/8-layer.mp4`, `sail/4-layer-3d.mp4`, `sail/8-layer-3d.mp4`
- `gamma/gamma_evolution.mp4`

Regenerate with `docs/scripts/make_videos.sh` (needs ffmpeg). The script defaults to the
latest gamma artifact directory in `mlruns/` and the `inter-4l`/`inter-8l` GIFs in
`examples/saved/`.

## YouTube

After upload, put the links in the generated-source section of `docs/index.md`,
`docs/experiments/2026-10-25-shakespeare.md`, and the README.

Suggested titles: "TARDITS - SAIL interference (4 layers)", "TARDITS - SAIL interference
(8 layers)", "TARDITS - learned complex decay gamma".
