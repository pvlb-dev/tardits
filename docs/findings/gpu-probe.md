# GPU probe - T4 throughput/memory for the 50M pair (2026-10-29)

Measured on Kaggle `NvidiaTeslaT4` (2×T4 session), torch preinstalled, **fp32**,
via the research-tree probe notebook and the private code snapshot
`pavelburda/tardits-src`. Random char tokens (vocab 256), AdamW, `block_size
1024`, 10 timed fwd+bwd+step iterations after 2 warmups; peak is
`torch.cuda.max_memory_allocated()`.

## Results

| model | params | batch | ms/iter | chars/s | peak GB |
|---|---:|---:|---:|---:|---:|
| RealGPT 576×12 h16 | 48.09M | 4 | 838 | **4,888** | 10.47 |
| RealGPT 576×12 h16 | 48.09M | 2 | 425 | 4,815 | 5.59 |
| RAT L1 512×12 h64 | 44.76M | 4 | 1,098 | **3,732** | 9.49 |
| RAT L1 512×12 h64 | 44.76M | 2 | 595 | 3,443 | 5.65 |
| RAT L1 512×12 h32 | 44.86M | 2 | 834 | 2,456 | 9.44 |
| RAT L1 384×12 h32 | 25.35M | 4 | 911 | 4,499 | 9.41 |

- **Both 48M/45M configs fit a 16 GB T4 at batch 4** with real head-room
  (~10.5 / ~9.5 GB peak); batch 8 (Real) and batch 4 with RAT h32 OOM. The earlier
  CPU retained-tensor estimate (27 GB for RAT) over-predicted CUDA peak by ~3×;
  CUDA allocator reuse + freed scan temporaries explain the gap.
- **RAT's head count matters on GPU too:** h64 (head_size 8) is 1.5× faster than
  h32 (head_size 16) at the same params because the scan tensor is 4× smaller.
- RealGPT is **1.31× faster per char** than matched RAT (4,888 vs 3,732 chars/s):
  cuBLAS fp32 is efficient, RAT's complex elementwise ops are not, and `complex64`
  gets no T4 tensor cores. This flips the CPU decode story - GPU training favours
  dense attention.
- One 10.5 h session at these rates: **~185M chars RealGPT, ~141M RAT**; matched
  ~123M chars (30k iters at 4,096 chars/iter) gives RealGPT ~6.9 h and RAT
  ~9.1 h in one session.

## Implications

1. No gradient checkpointing is needed for the first run; batch 4 fits both.
2. Use RAT `n_head 64`, not 32, for the matched pair
   (`experiments/configs/fineweb/rat_l1_45m.yaml`).
3. A matched-token first run is **30k iters** (123M chars) in one 2×T4 session,
   both processes pinned to one GPU each; RealGPT finishes early, RAT sets the
   session length (~9–9.5 h with setup/evals).
4. Multi-week / >300M-char runs need checkpoint+optimizer resume (`ttrain` has
   none); see the Kaggle quota notes (research tree).
5. The probe writes `probe_results*.json`; the kernel is
   `pavelburda/tardits-gpu-probe` (v6 run).

## FineWeb smoke (2026-10-29, kernel `pavelburda/tardits-fineweb-smoke`)

200 steps of the matched pair on the prepared 1B-char FineWeb bins, two
processes pinned to the two T4s, `batch 4 × block 1024`, eval at 0/100,
`eval_iters 5`, then the 500-token sample generation:

| arm | wall incl. setup/eval/sample | final train | final val | status |
|---|---:|---:|---:|---|
| RealGPT 576×12 | 236 s | 2.449 | 2.382 | exit 0, checkpoint + sample OK |
| RAT L1 512×12 h64 | 296 s | ~ | **2.326** | exit 0, checkpoint + sample OK |

**Caveat:** this smoke ran before the #23 chunked training-scan fix, so the RAT
arm at block 1024 was numerically invalid beyond ~100 tokens. Treat the
losses as pipeline validation only, not a quality comparison. Data pipeline
validated end-to-end (memmap bins, vocab, `ttrain` on CUDA, checkpoints). The wall times include model/data startup (~30 s), evals and the
500-token generation, so they are **not** a throughput measurement; subtract
those before comparing with the probe rates. Under two-process load the RAT
iteration appears closer to ~1.2 s than the probe's 1.10 s, which is why the
first sized run uses **25k iters (102M chars)** instead of 30k - it keeps the
run inside the 10 h timeout. RAT is ahead at 200 steps (2.326 vs 2.382), the
same early-lead pattern as shakespeare.

## Update: chunked training scan (#23 fix), probe v7

After the training forward was made chunked (correct `forward == step` at
`block_size 1024`), the same probe was rerun:

| model | batch | ms/iter | chars/s | peak GB |
|---|---:|---:|---:|---:|
| RealGPT 576×12 h16 | 4 | 844 | 4,853 | 10.47 |
| RAT L1 512×12 h64 | 4 | **1,381** | **2,966** | 12.00 |
| RAT L1 512×12 h64 | 2 | 817 | 2,506 | 7.97 |
| RAT L1 512×12 h32 | 2 | 1,170 | 1,750 | 13.79 |

Chunking costs RAT ~25 % throughput but makes the training memory correct
(previous RAT-vs-step drift: up to 1.8 logits at t=127 on a trained L1
checkpoint; with chunking + forced de Moivre, ~3e-3 at t=1024). The first sized
run therefore uses **22k iters (90M chars)** rather than 30k: RAT ~8.4 h +
evals + setup ≈ 9.2 h, inside the 10 h timeout. RealGPT finishes at ~5.2 h and
idles; the session is RAT-bound.
