#!/usr/bin/env python
"""Cheap trigger-in-context / mini-NIAH probe (CPU).

Train a tiny char model to emit a fixed loop only when a rare trigger string is
present anywhere in the context. Then sweep trigger distance, no-trigger and
unseen-trigger controls. This is a trigger/retention probe, not a language task.
"""
import argparse, csv, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tardits.models import ModelConfig, RAT, RealGPT

CUE = "\n???\n"
LOOP = "NIAH" * 8
TRIGGER = "ZXQ"
VARIANTS = ("MWM", "KKK", "QQQ", "ZXP")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="rat,real")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--corpus", default="shakespeare", choices=("shakespeare", "fineweb"))
    ap.add_argument("--tokens", default=".cache/kaggle/prep-train/fineweb/train.bin")
    ap.add_argument("--vocab-file", default=".cache/kaggle/prep-out/fineweb/vocab.txt")
    ap.add_argument("--trigger-frac", type=float, default=0.2)
    ap.add_argument("--loop-len", type=int, default=8)
    ap.add_argument("--near-miss-frac", type=float, default=0.0)
    ap.add_argument("--near-misses", default="ZXP,ZXR")
    ap.add_argument("--special-trigger", action="store_true")
    ap.add_argument("--payload", action="store_true")
    ap.add_argument("--payload-len", type=int, default=32)
    ap.add_argument("--payload-seed", type=int, default=7)
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--stage1-steps", type=int, default=6000)
    ap.add_argument("--stage1-trigger-frac", type=float, default=0.5)
    ap.add_argument("--stage2-trigger-frac", type=float, default=0.25)
    ap.add_argument("--stage2-near-miss-frac", type=float, default=0.25)
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/assets/curves/trigger.csv")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if args.payload:
        prng = random.Random(args.payload_seed)
        alphabet = [chr(c) for c in range(32, 127) if chr(c) not in "¤§‡"]
        loop = ''.join(prng.choice(alphabet) for _ in range(args.payload_len))
    else:
        loop = "NIAH" * max(1, args.loop_len // 4)
        loop = loop[:args.loop_len]
    if args.corpus == "fineweb":
        import numpy as np
        arr = np.memmap(args.tokens, dtype=np.uint16, mode="r")
        vocab = Path(args.vocab_file).read_text()
        if args.special_trigger:
            vocab = vocab + "¤§‡"
            TRIGGER = "¤"; VARIANTS = ("§", "‡")
        stoi = {c:i for i,c in enumerate(vocab)}
        itos = {i:c for i,c in enumerate(vocab)}
        # use a 20M-char train slice and a 2M-char test slice
        train_end = min(len(arr) - args.context - 64, 20_000_000)
        test_start = max(train_end, len(arr) - 2_000_000)
        train_arr = arr[:train_end]; test_arr = arr[test_start:]
    else:
        text = Path("examples/data/shakespeare.txt").read_text()
        chars = sorted(set(text)); stoi = {c:i for i,c in enumerate(chars)}
        itos = {i:c for i,c in enumerate(chars)}
        split = int(len(text)*0.8); train_txt, test_txt = text[:split], text[split:]
        train_arr = train_txt; test_arr = test_txt

    def make_example(source, with_trigger, trig=None, pos=None, insert=False):
        if trig is None: trig = TRIGGER
        if args.corpus == "fineweb":
            start = rng.randrange(0, len(source)-args.context-len(loop)-2)
            ids = source[start:start+args.context+len(loop)]
            text_s = ''.join(itos[int(i)] for i in ids)
            filler, cont = text_s[:args.context], text_s[args.context:args.context+len(loop)]
        else:
            start = rng.randrange(0, len(source)-args.context-64)
            filler = source[start:start+args.context]
            cont = source[start+args.context:start+args.context+len(loop)]
        if with_trigger or insert:
            if pos is None: pos = rng.randrange(0, len(filler)-len(trig))
            filler = filler[:pos] + trig + filler[pos+len(trig):]
        prompt = filler + CUE
        target = loop if with_trigger else cont
        ids = torch.tensor([stoi[c] for c in prompt+target], dtype=torch.long)
        return ids, prompt, target

    for arch, level in (("rat",1),("real",0)):
        if arch not in args.archs.split(","): continue
        torch.manual_seed(args.seed)
        cfg = ModelConfig(level=level, block_size=args.block, vocab_size=len(stoi),
                          n_head=8, n_embd=64, n_layer=2, dropout=0.0)
        model = (RAT if level else RealGPT)(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        t0=time.time()
        rows=[]

        def run_cases(phase):
            def run_case(name, with_trigger, trig=None, pos=None):
                if trig is None: trig = TRIGGER
                correct=0; loops=0; n=20
                for _ in range(n):
                    ids, prompt, target = make_example(test_arr, with_trigger, trig, pos)
                    p = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0)
                    with torch.no_grad():
                        out = model.generate(p, max_new_tokens=len(loop), temp=1.0, top_k=1)[0].tolist()
                    gen = ''.join(itos[i] for i in out[-len(loop):])
                    ok = gen[:len(loop)] == loop
                    correct += ok; loops += gen.startswith(loop[:4])
                rows.append(dict(arch=arch, phase=phase, case=name, hit_rate=loops/n,
                                 exact=correct/n, secs=round(time.time()-t0,1)))
                print(rows[-1], flush=True)
            run_case("trigger_end", True, pos=args.context-len(TRIGGER))
            run_case("trigger_q3", True, pos=3*args.context//4)
            run_case("trigger_mid", True, pos=args.context//2)
            run_case("trigger_q1", True, pos=args.context//4)
            run_case("trigger_start", True, pos=0)
            run_case("no_trigger", False)
            for v in VARIANTS: run_case("variant_"+v, True, trig=v, pos=args.context//2)

        def train_stage(nsteps, trig_frac, near_frac):
            model.train()
            for _ in range(nsteps):
                r = rng.random(); insert = False
                if r < trig_frac:
                    with_trigger = True; trig = TRIGGER
                elif r < trig_frac + near_frac:
                    with_trigger = False; insert = True
                    trig = rng.choice(args.near_misses.split(","))
                else:
                    with_trigger = False; trig = TRIGGER
                ids, _, _ = make_example(train_arr, with_trigger, trig=trig, insert=insert)
                xb = ids[:-1].unsqueeze(0); yb = ids[1:].unsqueeze(0)
                _, loss = model(xb, yb)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        if args.curriculum:
            train_stage(args.stage1_steps, args.stage1_trigger_frac, 0.0)
            model.eval(); run_cases("stage1")
            train_stage(max(0, args.steps-args.stage1_steps),
                        args.stage2_trigger_frac, args.stage2_near_miss_frac)
            model.eval(); run_cases("stage2")
        else:
            train_stage(args.steps, args.trigger_frac, args.near_miss_frac)
            model.eval(); run_cases("final")

        out=Path(args.out)
        if out.exists():
            existing=list(csv.DictReader(open(out)))
        else:
            existing=[]
        with out.open("w", newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(existing+rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
