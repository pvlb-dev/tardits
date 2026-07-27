from argparse import ArgumentParser
import sys
import time
import os
import yaml

import torch
from tardits.models import HyperFormer, RealGPT
from tardits.modules import get_device


def main():
    # --- CLI Arguments ---
    parser = ArgumentParser(
        description="Generates text from a trained model using its YAML config"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="examples/configs/complex.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional override for model checkpoint path",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=None,
        help="Optional override for vocab path",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="\n",
        help="Text prompt to start generation",
    )
    parser.add_argument(
        "--length",
        "-l",
        type=int,
        default=500,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        "-t",
        type=float,
        default=1.0,
        help="Model sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=10,
        help="Top-k sampling threshold",
    )
    parser.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Endless stream of tokens",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.04,
        help="Delay between tokens in follow mode (seconds)",
    )

    args = parser.parse_args()

    # --- Read Config File ---
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg["model"]["level"] < 1:
        GPT = RealGPT
    else:
        GPT = HyperFormer

    # --- Load Config ---
    checkpoint_path = args.checkpoint or cfg["trainer"]["checkpoint"]
    vocab_path = args.vocab or cfg.get("data", {}).get(
        "vocab_save_path", "examples/saved/chars.txt"
    )
    device = get_device(cfg.get("device", "auto"))

    # --- Load Weights ---
    model = GPT.load_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    # --- Load Vocabulary ---
    with open(vocab_path, "r", encoding="utf-8") as ch:
        chars = list(ch.read())

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(si):
        return [stoi[c] for c in si]

    def decode(li):
        return "".join([itos[i] for i in li])

    prompt = args.prompt.replace("\\n", "\n")
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

    # --- Generation ---
    if args.follow:
        # 🧹 Clear terminal screen
        print("\033[2J\033[H", end="")
        print(
            f"⏳ Token-Stream (complexity-lvl: {cfg['model']['level']} @ {checkpoint_path}) — Ctrl+C to exit\n"
        )
        try:
            sys.stdout.write(prompt)
            sys.stdout.flush()

            for token_id in model.generate_stream(
                idx, temp=args.temp, top_k=args.top_k
            ):
                char = itos[int(token_id)]
                sys.stdout.write(char)
                sys.stdout.flush()

                if args.delay > 0:
                    time.sleep(args.delay)
        except KeyboardInterrupt:
            print("\n\n🛑 Token stream stopped.")

    else:
        print(
            decode(
                model.generate(
                    idx, max_new_tokens=args.length, temp=args.temp, top_k=args.top_k
                )[0].tolist()
            )
        )


if __name__ == "__main__":
    main()
