# NIAH at FineWeb scale - the adjacent-copy control fails

**Status: negative/instrumentation, 2026-10-31.** This is the first attempt to run
the needle-in-a-haystack (NIAH) protocol on the FineWeb char-level checkpoints
(11M RealGPT control, 11M RAT, 6.5M RAT, 33M RAT, 44M RAT, 48M RealGPT). The
result is **not** “RAT cannot retrieve.” It is that the protocol's own
**adjacent-copy sanity check fails at every tested scale**, so the long-range
rows are uninformative. Do not use this as a quality claim.

## Protocol

- Haystack: held-out FineWeb `sample-10BT` val text (19.8M chars, document-disjoint
  from training; 95 of 96 charset chars present - the vocab is the training vocab).
- Needle: `vornith` (7 chars, absent from the corpus; cue phrase also absent).
- Insertion: `\nthe secret word is vornith\n` replaces a same-length slice at
  depth `d`; depths 0.1/0.5/0.9/1.0. Depth 1.0 = needle immediately before the
  query: the **copy sanity check**.
- Query: append `\nthe secret word is ` and score the next 7 chars (greedy and
  teacher-forced; exact match, char accuracy, rank).
- Control: identical prompt with the needle absent (same total length), scored
  against the same answer.
- Lengths: 128/256/512/1024; samples 8 for 11M/6.5M; larger models copy-check at
  L=128 with 4 samples.
- Harness: `benchmarks/bench_niah.py` (vocab-set assertion relaxed for a held-out
  slice that misses one training char).

## Results

| model | params | no-needle tf char | depth 1.0 tf char | greedy exact | first rank |
|---|---:|---:|---:|---:|---:|
| RealGPT 11M | 11.13M | 0.138 | 0.187 | 0.00 | 17.8 |
| RAT 11M | 11.24M | 0.143 | 0.143 | 0.00 | 19.4 |
| RAT 6.5M | 6.47M | 0.000 | 0.000 | 0.00 | 19.1 |
| RAT 33M | 33.37M | 0.000 | 0.000 | 0.00 | 20.0 |
| RAT 44M | 44.43M | 0.143 | 0.143 | 0.00 | 18.8 |
| RealGPT 48M | 48.09M | 0.000 | 0.000 | 0.00 | 19.8 |

- **No model passes adjacent copy**: none spells any part of the needle even when
  it is immediately before the cue; teacher-forced char accuracy is at the
  no-needle/char-prior level, and greedy exact is 0 everywhere.
- Therefore the 11M full matrix (lengths/depths) is **not evidence about
  long-range retrieval**: if a model cannot copy an adjacent needle, a failure at
  depth 0.1 says nothing about its state/memory. The CSV is kept for the record:
  `docs/assets/curves/niah_fineweb_11m.csv`; the larger copy-check is
  `docs/assets/curves/niah_fineweb_big_copycheck.csv`.

## Why this happened

- The char-level base model has no instruction tuning and the cue
  `the secret word is` is OOD; NIAH relies on in-context copy induction, which a
  90M-char char-level model does not necessarily learn.
- The 7-char OOV proper noun is a hard copy target; a single-char or
  in-distribution target might pass, but would weaken the retrieval test.
- The earlier shakespeare pilot had the same adjacent-copy failure at 0.8M
  (`niah-pilot.md`); this run shows it persists at 33–48M on FineWeb chars.

## What to do instead

1. **Validate copy first**: use a cue/target that the model can copy at depth 1.0
   (e.g. an in-distribution repeated phrase or a shorter target) and *only then*
   sweep depth/length.
2. **Train a copy/induction task** or use a token-level model with instruction
   tuning before claiming NIAH capability.
3. Do not quote NIAH accuracy for these checkpoints; loss and speed results are
   unaffected.

## Provenance

- 11M/6.5M/Real11: `.cache/kaggle/{real11m-rat6m-out,rat22m11m-out}` checkpoints.
- 33M/44M/48M: `.cache/kaggle/{rat33m-out,rat22k-out,run-out}` checkpoints.
- Runs: `docs/assets/curves/niah_fineweb_11m.csv` (60 rows),
  `niah_fineweb_big_copycheck.csv` (copy check at L=128).
- Code: `benchmarks/bench_niah.py`; corpus `.cache/niah/fineweb_val.txt`.
