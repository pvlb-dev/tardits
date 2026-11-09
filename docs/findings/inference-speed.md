# Generation speed - KV-cached real GPT vs RAT

RAT's architectural selling point is O(1) work per generated token: the fast-weight state is
updated in place, while the real transformer keeps a growing KV cache and does O(T) attention
work for every token. `benchmarks/bench_inference.py` measures that on the trained shakespeare
checkpoints, from short prompts (16 tokens) to long ones (2048 tokens, 16× the 128-token
training window).

**Short version:** the models are tied up to ~256 tokens; beyond that RAT L1 is faster per
generated token (1.48× at 2048) and, with the vectorised prefill from
#21, faster **end-to-end** (2.1× at 2048 for a
16-token generation). Its decode speed is flat; the real model's falls by ~45% from 16 to
2048 tokens. At short context RAT is not faster.

## Method

- Checkpoints: `real_baseline.pt` (0.90M), `rat_l1_h21_5k.pt` (0.83M),
  `rat_l2_h16_l5_5k.pt` (0.62M); all trained on shakespeare with `block_size=128`.
- CPU only, batch 1, top-k sampling, `max_new_tokens 16`; **minimum of 12 repeats** (the
  minimum is a stable lower bound for a microbenchmark on this shared desktop host).
- **RealGPT uses a KV cache** (`RealGPT.prefill` + `generate_cached`): the prompt is
  prefilled in `block_size` chunks, the RoPE cache grows lazily, then each token is one
  cached attention step.
- **RAT prefill is the chunked parallel scan** (`RAT.prefill`, issue #21): the prompt is
  ingested in chunks whose size is bounded by the trained `|gamma|` decay (see
  `RAT._prefill_chunk_size`), with the fast-weight state carried between chunks. It is
  numerically equivalent to the token-by-token recurrence but runs the scan instead of one
  Python step per token (5.9 s → 0.18 s at 2048 tokens).
- **Decode timing excludes prefill.** Prefill time is measured and reported separately
  because it is a real part of latency and the two models have different prefill profiles.
- Context lengths: 16, 64, 128, 256, 512, 1024, 2048. The dashed line at 128 is the
  training window; everything to its right is **extrapolation**. This is a speed benchmark,
  not a quality claim - retrieval quality beyond the window is a clean null at this scale
  ([NIAH pilot](niah-pilot.md), #20).
- Equivalence is tested in `tests/test_real.py` (`TestRealGPTKVCache`) and
  `tests/test_rat_equivalence.py` (`TestRATPrefill`: scan prefill == sequential warmup, and
  chunked prefill == sequential across chunk boundaries for levels 1–4).

```bash
.venv/bin/python benchmarks/bench_inference.py
# -> docs/assets/curves/inference_scaling.csv
# -> docs/assets/curves/inference_scaling.png
```

## Results - decode speed (tokens/s, prefill excluded)

| context | RealGPT (KV-cached) | RAT L1 | RAT L2 |
|--------:|--------------------:|-------:|-------:|
| 16 | **392.3** | 362.7 | 108.0 |
| 64 | **377.8** | 352.7 | 110.2 |
| 128 | 331.5 | **350.7** | 106.8 |
| 256 | 319.1 | **332.4** | 106.6 |
| 512 | 286.8 | **324.9** | 103.9 |
| 1024 | 262.1 | **318.0** | 106.5 |
| 2048 | 215.1 | **317.4** | 105.0 |

![decode speed and prefill cost](../assets/curves/inference_scaling.png)

- **≤64 tokens:** the real model is slightly faster (~8%).
- **Cross-over between 128 and 256:** at 128 they are tied (331 vs 351), at 2048 RAT L1 is
  **1.48×** faster per token. RAT L1 is approximately flat (317–363 tok/s over the whole
  range, with the residual variation dominated by host noise); the real model loses ~45% of
  its throughput from 16 to 2048 as attention work grows.
- **RAT L2** is flat but ~3× slower than L1 because each fast-weight element is a quaternion
  product. It remains a parameter-efficiency result (0.62M vs 0.90M), not a speed result.

## Results - prefill time (ms)

| context | RealGPT (chunked) | RAT L1 | RAT L2 |
|--------:|------------------:|-------:|-------:|
| 16 | **3.4** | 6.4 | 13.2 |
| 64 | 4.4 | 10.0 | 20.8 |
| 128 | **6.1** | 11.9 | 29.3 |
| 256 | **13.2** | 22.1 | 56.2 |
| 512 | 40.3 | **44.5** | 110.4 |
| 1024 | 123.5 | **88.8** | 220.9 |
| 2048 | 413.2 | **181.9** | 457.7 |

RealGPT's chunked prefill is O(T²) attention work; RAT's chunked scan is O(T) with a small
constant, so **RAT prefill becomes cheaper than the real model beyond ~512 tokens** (at 2048:
182 ms vs 413 ms). Before #21 this was reversed: sequential token-by-token warmup took 5.9 s
at 2048 and was the reason the decode win did not translate into an end-to-end win.

## End-to-end latency (prefill + 16 generated tokens)

| context | RealGPT | RAT L1 | winner |
|--------:|--------:|-------:|-------:|
| 16 | 44.2 ms | 50.5 ms | real (1.14×) |
| 128 | 54.4 ms | 57.5 ms | real (1.06×) |
| 256 | 63.3 ms | 70.2 ms | real (1.11×) |
| 512 | 96.1 ms | 93.7 ms | tie |
| 1024 | 184.6 ms | 139.1 ms | RAT (1.33×) |
| 2048 | 487.6 ms | 232.3 ms | **RAT (2.10×)** |

So the end-to-end cross-over is around 512 tokens: for short prompts the real model is a few
percent faster, for long prompts RAT wins and keeps winning as the context grows. That is the
paper-grade claim: **fixed-size state gives constant decode cost and, thanks to linear
prefill, a real long-context latency win** - not a short-context speed-up.

## What this means for the paper / README

- **Do not** say “RAT is 2.7× faster”. That came from an **uncached** RealGPT baseline and is
  invalid.
- **Do** say: RAT L1 decode is flat at ~320–360 tok/s while the KV-cached real model falls
  from ~390 to ~215 tok/s (1.48× at 2048); with the chunked scan prefill, end-to-end latency
  at 2048 tokens is 2.1× lower for a 16-token generation. Cross-over is around 512 tokens of
  context.
- **Caveats to state:** CPU-only, batch 1, char-level 0.6–0.9M models; beyond the 128-token
  training window this is a speed measurement, and retrieval quality is a negative
  (NIAH null). RAT L2 is slower, so the speed story is specifically RAT L1.
- The marketing-safe framing: *constant-size state, O(1) decode and O(T) prefill → faster
  end-to-end generation once context exceeds ~512 tokens, with 31% fewer parameters at
  +0.03 val loss on shakespeare.*

## Caveats

- Batch 1, CPU, sampling included; not a GPU serving benchmark. The O(T) vs O(1) decode
  scaling is architectural; absolute throughput will differ on CUDA.
- CPU wall-clock is noisy (~5–10% even for the minimum of 12); do not compare runs measured
  under different host load. Absolute numbers move with the desktop, the ratios are stable.
- Beyond 128 tokens both models extrapolate: RoPE positions and RAT's decay horizon were
  never trained. Speed only - quality is not claimed.
- RAT L2's constant factor is heavy (quaternion products); it is included to show the
  parameter-efficiency trade-off, not as a speed candidate.
- Raw rows: `docs/assets/curves/inference_scaling.csv`.
