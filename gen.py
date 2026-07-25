from argparse import ArgumentParser
import importlib
import sys
import time
import os
import yaml

import torch
from modules.base import get_device

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
    help="Endless stream of tokens watching checkpoint updates in real-time",
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

# --- Load Config ---
model_type = cfg["model"]["type"]
checkpoint_path = args.checkpoint or cfg["trainer"]["checkpoint"]
vocab_path = args.vocab or cfg.get("data", {}).get(
    "vocab_save_path", "examples/saved/chars.txt"
)
device = get_device(cfg.get("device", "auto"))

class_name = f"{model_type}GPT"

# --- Import Model Architecture ---
models_module = importlib.import_module("models")
try:
    GPT_Class = getattr(models_module, class_name)
except AttributeError:
    raise ImportError(f"❌ Model {class_name} not found in models module")

# --- Load Weights ---
model = GPT_Class.load_from_checkpoint(checkpoint_path, device=device)
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
    print(f"⏳ Live token-stream ({class_name} @ {checkpoint_path}) — Ctrl+C to exit\n")
    try:
        sys.stdout.write(prompt)
        sys.stdout.flush()

        model_mtime = 0
        tk_step = 0

        for token_id in model.generate_stream(idx, temp=args.temp, top_k=args.top_k):
            char = itos[token_id]
            sys.stdout.write(char)
            sys.stdout.flush()

            # Hot-Reload Checkpoint
            if tk_step == 0 and os.path.exists(checkpoint_path):
                current_mtime = os.path.getmtime(checkpoint_path)
                if current_mtime > model_mtime:
                    model_mtime = current_mtime
                    model = GPT_Class.load_from_checkpoint(
                        checkpoint_path, device=device
                    )

            tk_step = (tk_step + 1) % 100

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
