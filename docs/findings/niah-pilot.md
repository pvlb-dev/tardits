# NIAH pilot - retrieval beyond the 128-token training window

**Verdict (TL;DR): a clean null.** With the shipped 0.6–0.9M-parameter char-level
checkpoints (5000 iters, `block_size=128`) greedy exact-match on the needle is **0.000 in
every arm** - 3 models × haystack lengths 128/256/512 (+1024 for real & RAT L1) × depths
0.1/0.5/0.9/1.0, 10 samples each. The needle arms are statistically indistinguishable
from the needle-absent chance control, and the **adjacent-copy sanity check (depth 1.0)
also fails**. So the models do not have a measurable in-context copy/induction circuit at
all; the RAT-state-vs-RoPE long-context retrieval question cannot be answered with these
checkpoints. That is the finding, not a benchmark bug: at this scale there is nothing for
a longer window or a larger state to retrieve.

Benchmark: [`benchmarks/bench_niah.py`](../../benchmarks/bench_niah.py) ·
data: [`docs/assets/curves/niah_pilot.csv`](../assets/curves/niah_pilot.csv).

## Question

RAT carries a fixed-size recurrent fast-weight state and its generation path ignores
`block_size`; RealGPT's RoPE cache was trained only to position 128. Does either model
retrieve a fact placed 128–1024 characters back? NIAH is the cheapest standard probe:
insert one distinctive string, ask for it with a short cue, score the continuation.

## Protocol (the minimal version)

| component | choice | why |
|---|---|---|
| haystack | contiguous slice of the **training** split of `examples/data/shakespeare.txt`, length `L ∈ {128, 256, 512}` (+1024 for real/RAT L1), seed 1337 | in-distribution chars/word statistics, not random token noise |
| needle | `"okonkwo"` (7 chars, OOV proper noun, absent from the corpus) | distinctive but lower-case - all chars in the trained 65-char vocab; an upper-case + digit variant was tried and is worse (see below) |
| insertion | `"\nthe secret word is okonkwo\n"` (30 chars) replaces a same-length filler slice at `pos = round(depth · (L − 30))` | prompt length stays exactly `L + 20`; the needle cannot leak from corpus memorisation |
| depth | `0.1, 0.5, 0.9` (fraction of the haystack) and **`1.0` = adjacent-copy sanity** | 1.0 puts the needle immediately before the query cue: if the model cannot copy an adjacent 7-char string, long-range negatives are uninformative |
| query | append `"\nthe secret word is "` and evaluate the next 7 chars | a char LM only supports continuation; no prompt-format retraining needed |
| arms | needle vs **needle-absent control** (same haystack + query, same total length) | the control is the empirical chance level, not a theoretical guess |
| decode | **greedy argmax**, deterministic, no sampling; plus a **teacher-forced** pass that feeds the correct prefix | greedy exact-match is the literature metric; teacher forcing + answer-char rank decouples a weak retrieval signal from a cascading generation failure |
| metrics | exact match, char accuracy, first-char accuracy, mean matching prefix, and the **rank of the correct answer char** (0 = top-1; uniform chance over the 65-char vocab is 32) | |
| samples | 10 per `(model, length, depth)`, batched in one prefill per arm | |

Insertion examples (`L=128`, `depth=1.0`): prompt tail
`…the secret word is okonkwo⏎⏎the secret word is ` → target `okonkwo`.
The control prompt is `…haystack⏎the secret word is `.

### Why this is the smallest practical protocol

* **One script, one deterministic pass, no retraining.** 55 evaluated arms
  (44 needle + 11 control) in ~230 s of measured compute on CPU.
* **Continuation is the only query interface a char LM has**; anything richer (Q&A
  formatting, "answer:" instructions) would itself be OOD and confound the result.
* The cue is absent from the corpus and the needle is absent from the corpus, so a correct
  answer can only come from in-context retrieval, not memorisation.
* Depth 1.0 is the essential positive control. Without it, a null could be blamed on the
  prompt format; with it, a null says the capability itself is missing.
* 10 samples are enough to reject a *strong* retrieval effect (true retrieval at depth 1.0
  would show exact match near 10/10, not 0/10). They cannot rule out a weak (<20–30%)
  effect - a pilot limitation, stated in the caveats.
* The needle-absent control is the correct chance baseline. The rank of a frequent char is
  naturally ~10–20 without any retrieval (rank 32 is uniform-random, not chance for a
  language model), so theory cannot set the baseline - the control does.

### Deliberately not done

No synthetic retrieval/induction training, no multiple needles/distractors, no 1024 arm for
RAT L2, no per-head attribution, no RoPE-extrapolation tuning. All of these would only
matter *after* the depth-1.0 sanity check passes.

## Checkpoints and context handling

| name | checkpoint | class | level | layers | heads | embd | params |
|---|---|---|---:|---:|---:|---:|---:|
| real | `real_baseline.pt` | RealGPT | 0 | 8 | 6 | 96 | 899,681 |
| rat-l1 | `rat_l1_h21_5k.pt` | RAT | 1 | 8 | 21 | 84 | 828,324 |
| rat-l2 | `rat_l2_h16_l5_5k.pt` | RAT | 2 | 5 | 16 | 64 | 621,504 |

Both families were trained with batch 32 × `block_size` 128 × 5000 iters = ~20.5M char
predictions ≈ **23 epochs over the 0.89M-char training split**.

* **RAT** warms its recurrent state token by token over the whole prompt (`_warm_states`);
  by construction there is no position/window limit (`mode = recurrent`, context 148–1044).
* **RealGPT** is prefilled in `block_size` chunks via `RealGPT.prefill` with the lazily
  extended RoPE cache (`mode = chunked`, context 148–1044). Positions > 128 are pure RoPE
  **extrapolation** - the model never trained there. This long-context path is part of the
  in-flight long-context change in `src/tardits/models/real.py`; if it is not present, the
  benchmark falls back to the last 128 tokens and labels the rows `windowed(128)`.

## Cost

Measured with `.venv/bin/python benchmarks/bench_niah.py`, `samples=10`, CPU:

| model | 128 | 256 | 512 | 1024 | per-arm cost |
|---|---:|---:|---:|---:|---|
| real | 0.2 s | 0.3 s | 0.9 s | 2.5 s | 2 prefills + 14 teacher-forced/greedy decode steps |
| rat-l1 | 1.3 s | 2.6 s | 5.0 s | 8.9 s | same |
| rat-l2 | 3.9 s | 7.0 s | 13.5 s | - | same |

Sum over all 55 arms: **230 s ≈ 3.8 min** of measured compute; wall time with three
checkpoint loads and startup ≈ 5 min, well inside the 15-minute budget. RAT L2 dominates
and is optional (`--models real=…,rat-l1=…` drops it to ~1 min).

## Results

### Primary metric - greedy exact match

**0.0000 in all 55 arms** (44 needle, 11 control). Greedy char accuracy is 0.000–0.057
everywhere (0–4 matching chars out of 70 = 10 samples × 7 needle chars). A needle arm beats
its own control by at most **one** matching char (real `L=256`: 0.043 vs 0.029; real
`L=1024`: 0.029 vs 0.014; rat-l1 `L=128`: 0.014 vs 0.000), and the best needle cell (0.057,
real at `L=512`) ties its control. That is chance.

### Sensitivity metric - rank of the correct char under teacher forcing

Values are `mean rank (first-char rank)`; lower is better. The **control column is the
empirical chance level**; vocab is 65, so 32 would be uniform-random and ~15–18 is what a
language model gets for a frequent target without any retrieval.

| model | haystack | control | depth 0.1 | depth 0.5 | depth 0.9 | depth 1.0 (adjacent) |
|---|---:|---:|---:|---:|---:|---:|
| real | 128 | 16.4 (9.8) | 16.3 (9.5) | 16.4 (9.6) | 16.4 (9.6) | 16.5 (9.2) |
| real | 256 | 14.8 (10.7) | 14.7 (10.6) | 14.8 (10.0) | 14.8 (10.7) | 14.9 (11.0) |
| real | 512 | 14.7 (12.6) | 14.4 (12.1) | 14.7 (12.5) | 14.7 (12.6) | 14.8 (12.7) |
| real | 1024 | 16.0 (13.0) | 16.3 (12.5) | 15.9 (12.8) | 16.0 (12.9) | 16.1 (13.2) |
| rat-l1 | 128 | 15.8 (11.8) | 15.9 (11.9) | 16.0 (11.7) | 14.9 (9.5) | 16.4 (11.7) |
| rat-l1 | 256 | 16.1 (12.5) | 16.1 (12.5) | 16.1 (12.6) | 16.6 (12.1) | 16.5 (11.4) |
| rat-l1 | 512 | 16.0 (11.0) | 16.0 (11.0) | 16.0 (11.0) | 16.1 (10.9) | 16.5 (11.4) |
| rat-l1 | 1024 | 16.4 (12.0) | 16.4 (12.0) | 16.4 (12.0) | 16.4 (11.9) | 16.6 (11.5) |
| rat-l2 | 128 | 17.8 (12.0) | 17.8 (12.0) | 17.8 (12.1) | 17.5 (11.0) | 20.3 (14.1) |
| rat-l2 | 256 | 17.8 (12.6) | 17.8 (12.6) | 17.8 (12.6) | 18.3 (12.4) | 20.5 (14.5) |
| rat-l2 | 512 | 18.0 (11.9) | 18.0 (11.9) | 18.0 (11.9) | 18.1 (12.2) | 20.4 (14.1) |

The largest *improvement* over its own control anywhere is 0.9 rank (rat-l1, `L=128`,
depth 0.9) - one cell, 10 samples, i.e. noise. The largest *deviation* is the opposite
direction: RAT L2 at depth 1.0 is **2.4–2.7 ranks worse than control** at every length
(the adjacent needle block disrupts the local context and is not retrieved). Teacher-forced
char accuracy never exceeds 0.129 (RAT L1, `L=128`, depth 1.0) vs a 0.057 control maximum;
that is 9 correct chars out of 70 - not a systematic effect.

### Adjacent-copy sanity check fails (the key result)

At depth 1.0 the correct answer is literally the 7 chars before the query cue. Rank stays
at the control level for RealGPT (16.4 vs 16.5 at `L=128`) and RAT L1 (15.8 vs 16.4), and
*degrades* for RAT L2 (17.8 → 20.3). Greedy output is generic Shakespeare-like text
(`'the princ'`, `'the son o'`, `'the cou'`) identical to the needle-absent control. **The
models cannot copy an adjacent 7-char OOV string after a repeated 19-char cue.** The
failure mode is "no in-context copy/induction", not "state decayed over distance".

### Robustness: upper-case + digit needle (`"OKONKWO-3"`)

`samples=5`, `L ∈ {128, 512}`, real + RAT L1:
all greedy exact 0.000; teacher-forced mean rank 28.3–28.5 (RAT L1) and 34.8–37.9
(RealGPT) vs controls 28.3 and 35.3/37.9 - no signal, and the rank of the correct upper-case
`O` is *worse* than chance because `O` is rare in the corpus. Lower-case `"okonkwo"` is the
friendlier variant and is the one reported above; the null is not an artefact of the needle
being OOD upper-case.

## Feasibility verdict

The null is expected and is informative:

1. **Capacity.** 0.6–0.9M params, 5–8 layers, 65-char vocabulary, ~23 epochs on 0.89M
   chars. A NIAH needs an induction/copy circuit; a model this size trained this briefly
   does not form one. This matches the adjacent-copy failure, which is independent of
   context length and of RAT vs RealGPT.
2. **Training distribution.** The models were trained only on next-char prediction on
   shakespeare. Nothing in the objective favours copying an arbitrary string seen once.
   RoPE positions > 128 and RAT state horizons > 128 are both **unseen** at training time,
   so even a retrieval-capable model would be extrapolating.
3. **Therefore the headline question is unanswerable here.** We cannot say "RAT retrieves at
   512 where RealGPT fails", nor the reverse. The only honest reading is *neither retrieves
   at all at this scale*.

### What a feasible NIAH would need

* **A model that passes the depth-1.0 sanity check** (adjacent copy ≥ ~9/10) before any
  long-context claim. Practically: ≥10–50M params (or a few M with more layers/heads and a
  10–20× longer training run), and ideally long-context training data.
* **A training/evaluation window ≥ the tested context**, or explicit length-extrapolation
  training (RoPE scaling, state decay tuning). Testing 512 with a 128 window measures
  extrapolation, not retrieval; test 128/256 inside a 256+ trained window instead.
* **Synthetic retrieval/induction data** (NIAH-style needle–query pairs, or a copy/induction
  pretraining task) to install the circuit deliberately - this is the cheapest route and
  would make the RAT-vs-RealGPT comparison meaningful.
* **A larger evaluation**: ≥100 samples, multiple needles/distractors, several depths, and
  depth 1.0 as a mandatory positive control; report exact match with binomial CIs.
* Only then re-ask the architecture question (state size vs KV cache/RoPE).

The concrete path to such a model is #22: a
matched ~50M char-level RAT vs RealGPT pair trained on FineWeb via the Kaggle GPU quota
(#19), evaluated with this protocol.

## Caveats

* RealGPT's >128 context uses chunked prefill + RoPE extrapolation (landed with #18/#21);
  positions > 128 have no training signal.
  This does not change the null: depth 1.0 fails *inside* the 128 window too.
* One needle string, one cue, one haystack source, seed 1337, 10 samples. This rejects a
  strong effect, not a weak one.
* The needle block *replaces* 30 filler chars, so treated and control have identical total
  length but slightly different filler content; the control is a prompt-level chance
  baseline, not a one-character edit.
* RAT prefill is O(T) sequential (`_warm_states`), so benchmark wall time is dominated by
  state warm-up, not by the O(1) decode.
* CPU wall-clock is noisy; timings from a shared 12-core box.

## Reproduce

```bash
# primary matrix (real + RAT L1 + RAT L2, 128/256/512, all depths + chance control)
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python benchmarks/bench_niah.py
# -> docs/assets/curves/niah_pilot.csv

# 1024-token arm for the two long-context candidates (rows merged into the CSV)
.venv/bin/python benchmarks/bench_niah.py \
  --models real=examples/saved/shakespeare/real_baseline.pt,rat-l1=examples/saved/shakespeare/rat_l1_h21_5k.pt \
  --lengths 1024 --out .cache/niah_1024.csv

# upper-case robustness check (not in the committed CSV)
.venv/bin/python benchmarks/bench_niah.py --needle "OKONKWO-3" \
  --models real=examples/saved/shakespeare/real_baseline.pt,rat-l1=examples/saved/shakespeare/rat_l1_h21_5k.pt \
  --lengths 128,512 --depths 0.1,0.9,1.0 --samples 5 --out .cache/niah_upper.csv
```

## Proposed issue

Filed as #20. The copy-paste-ready body follows for reference.

**Title:** NIAH pilot: clean null at 0.8M params - drop the long-context retrieval claim or
scope a feasible benchmark

**Labels:** `type/task`, `area/ml`, `priority/next`

**Body:**

```markdown
## Context

The NIAH pilot (`docs/findings/niah-pilot.md`, `benchmarks/bench_niah.py`,
`docs/assets/curves/niah_pilot.csv`) measures whether the shipped char-level checkpoints
retrieve a needle beyond the 128-token training window. Question: can RAT's fixed-size
recurrent state retrieve where RealGPT's RoPE (trained to 128) cannot.

## Result - clean null

- Greedy exact-match **0.000 in all 44 needle arms and 11 chance arms**:
  3 models (RealGPT 0.90M, RAT L1 0.83M, RAT L2 0.62M) × haystack 128/256/512
  (+1024 for real & RAT L1) × depths 0.1/0.5/0.9/1.0 × 10 samples.
- Needle arms are indistinguishable from the needle-absent control; best needle char
  accuracy (0.057) ties its own control.
- **Adjacent-copy sanity (depth 1.0) fails**: the correct answer is the 7 chars immediately
  before the cue, yet both models emit generic Shakespeare-like text. Teacher-forced rank
  stays at the chance-control level (RealGPT 16.5 vs 16.4; RAT L1 16.4 vs 15.8; RAT L2
  degrades 17.8 → 20.3). No in-context copy/induction at this scale.
- Cost: 230 s measured (55 arms × greedy + teacher-forced), < 15 min budget.

## Interpretation

0.6–0.9M params, 5–8 layers, ~23 epochs on 0.89M chars, no retrieval training; positions
and state horizons > 128 are unseen. The benchmark is valid but the capability under test
is absent, so the RAT-vs-RoPE long-context retrieval question cannot be answered at this
scale. This is a documented negative, not a benchmark bug.

## Proposal

1. Do not claim long-context retrieval in the README/paper; keep issue #18's long-context
   story to O(1) decode / fixed-size state (speed), not retrieval quality.
2. Scope a feasible benchmark before any such claim:
   - positive control: adjacent copy ≥ 9/10 (implies ~10–50M params or much longer training);
   - training/eval window ≥ tested context (or length-extrapolation training);
   - synthetic NIAH/induction training data (cheapest route to install the circuit);
   - ≥100 samples, multiple needles, depth 1.0 mandatory, exact-match with CIs.
3. Keep `benchmarks/bench_niah.py` + the CSV as the reproducible null so the next
   checkpoint can be measured the same way.

## Acceptance criteria

- [ ] README/docs do not claim long-context retrieval; wording points to this negative
- [ ] decision recorded: drop the claim, or scope a feasible NIAH (model/window/data)
- [ ] the pilot script + CSV stay in-tree as the reproducible baseline

## References

- docs/findings/niah-pilot.md
- benchmarks/bench_niah.py
- docs/assets/curves/niah_pilot.csv
- issue #18 (long-context decode speed - speed only, no retrieval claim)
```

*Note for the Lead:* the `real` rows depend on the in-flight long-context change
(`RealGPT.prefill` + lazy RoPE extension) in `src/tardits/models/real.py`. If that change
is reverted, rerun the benchmark: it will fall back to `windowed(128)` and label the rows
accordingly.
