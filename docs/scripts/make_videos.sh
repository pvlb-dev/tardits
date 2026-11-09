#!/usr/bin/env bash
# Convert the generated GIF/PNG artifacts into MP4s for YouTube/README/Obsidian.
#
# Usage:
#   docs/scripts/make_videos.sh
#   GAMMA_DIR=mlruns/2/<run-id>/artifacts/gamma_histograms \
#   SAIL_DIR=examples/saved docs/scripts/make_videos.sh
#
# Produces (gitignored, under docs/assets/):
#   gamma/gamma_evolution.mp4
#   sail/4-layer.mp4, sail/8-layer.mp4
#   sail/4-layer-3d.mp4, sail/8-layer-3d.mp4
#   */poster.png
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-docs/assets}"
SAIL_DIR="${SAIL_DIR:-examples/saved}"
GAMMA_DIR="${GAMMA_DIR:-$(find mlruns -type d -path '*artifacts/gamma_histograms' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)}"

mkdir -p "$OUT/gamma" "$OUT/sail"

if [[ -n "${GAMMA_DIR}" && -d "${GAMMA_DIR}" ]]; then
  echo "gamma frames: ${GAMMA_DIR}"
  # Build an ordered, step-500.. sequence. The step-0 frame is skipped: at init
  # every |gamma| is exactly 1, so its histogram is a single bin with an ugly
  # offset x-axis and says less than the first trained snapshot.
  TMP_GAMMA="$(mktemp -d)"
  i=0
  while IFS= read -r frame; do
    [[ "$frame" == *gamma_step_00000.png ]] && continue
    ln -s "$(realpath "$frame")" "$TMP_GAMMA/frame_$(printf '%03d' "$i").png"
    i=$((i + 1))
  done < <(find "$GAMMA_DIR" -name 'gamma_step_*.png' | sort)

  if [[ $i -eq 0 ]]; then
    echo "no trained gamma frames; falling back to all frames" >&2
    i=0
    while IFS= read -r frame; do
      ln -s "$(realpath "$frame")" "$TMP_GAMMA/frame_$(printf '%03d' "$i").png"
      i=$((i + 1))
    done < <(find "$GAMMA_DIR" -name 'gamma_step_*.png' | sort)
  fi

  # 1 fps input, 30 fps output: each histogram is held for one second.
  ffmpeg -y -loglevel error -framerate 1 -i "$TMP_GAMMA/frame_%03d.png" \
    -vf "scale=1400:-2:flags=lanczos,format=yuv420p" \
    -r 30 -movflags +faststart "$OUT/gamma/gamma_evolution.mp4"
  # Poster: last trained frame, not the init frame.
  ffmpeg -y -loglevel error -sseof -0.2 -i "$OUT/gamma/gamma_evolution.mp4" \
    -frames:v 1 -update 1 "$OUT/gamma/poster.png"
  rm -rf "$TMP_GAMMA"
else
  echo "no gamma frames found; set GAMMA_DIR" >&2
fi

# GIF -> H.264. Scale to an even width; -2 keeps the aspect ratio even.
for layers in 4 8; do
  gif="${SAIL_DIR}/inter-${layers}l_sail_all_layers_evolution.gif"
  if [[ -f "$gif" ]]; then
    ffmpeg -y -loglevel error -i "$gif" \
      -vf "scale=1000:-2:flags=lanczos,format=yuv420p" \
      -movflags +faststart "$OUT/sail/${layers}-layer.mp4"
    ffmpeg -y -loglevel error -i "$gif" -vf "select=eq(n\\,60)" \
      -frames:v 1 "$OUT/sail/${layers}-layer-poster.png"
  else
    echo "missing $gif" >&2
  fi

  gif3d="${SAIL_DIR}/inter-${layers}l_sail_3d_topography_evolution.gif"
  if [[ -f "$gif3d" ]]; then
    ffmpeg -y -loglevel error -i "$gif3d" \
      -vf "scale=1200:-2:flags=lanczos,format=yuv420p" \
      -movflags +faststart "$OUT/sail/${layers}-layer-3d.mp4"
  fi
done

echo "done:"
find "$OUT" -name '*.mp4' -o -name '*poster.png' | sort
