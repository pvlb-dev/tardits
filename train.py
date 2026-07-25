from argparse import ArgumentParser
import importlib
import yaml

import torch

from modules import Trainer, ModelConfig, TrainerConfig, MLFlowConfig
from modules.logging import ConsoleLogger, MLFlowLogger
from modules.base import get_device

# --- Config & Arg Parsing ---
parser = ArgumentParser(
    description="Train hypercomplex transformers using YAML configs"
)
parser.add_argument(
    "--config",
    "-c",
    type=str,
    default="examples/configs/complex.yaml",
    help="Path to YAML configuration file",
)
args = parser.parse_args()

# --- Load YAML Config ---
with open(args.config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device = get_device(config.get("device", "auto"))
model_cfg_dict = config["model"]
trainer_cfg_dict = config["trainer"]
mlflow_cfg_dict = config.get("mlflow", {})
data_cfg_dict = config.get("data", {})

seed = config.get("seed", None)
if seed:
    torch.manual_seed(seed)

# --- Load Model Class ---
class_name = f"{model_cfg_dict.pop('type')}GPT"
models_module = importlib.import_module("models")
try:
    GPT_Class = getattr(models_module, class_name)
except AttributeError:
    raise ImportError(f"❌ Model {class_name} not found in models module")

# --- Data Preparation ---
data_path = data_cfg_dict.get("input_path", "examples/data/shakespeare.txt")
vocab_path = data_cfg_dict.get("vocab_save_path", "examples/saved/chars.txt")

with open(data_path, "r", encoding="utf-8") as f:
    full_text = f.read()

chars = sorted(list(set(full_text)))

with open(vocab_path, "w", encoding="utf-8") as ch:
    ch.write("".join(chars))

vocab_size = len(chars)

splitpos = int(len(full_text) * 0.8)
train_txt = full_text[:splitpos]
val_txt = full_text[splitpos:]

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(si):
    return [stoi[c] for c in si]


def decode(li):
    return "".join([itos[i] for i in li])


train_data = torch.tensor(encode(train_txt), dtype=torch.long).to(device)
val_data = torch.tensor(encode(val_txt), dtype=torch.long).to(device)

# --- Model & Configurations ---
modelconf = ModelConfig(vocab_size=vocab_size, **model_cfg_dict)

model = GPT_Class(modelconf).to(device)

trainconf = TrainerConfig(**trainer_cfg_dict)
mlflowconf = MLFlowConfig(**mlflow_cfg_dict)

loggers: list = [ConsoleLogger()]

if mlflowconf.experiment:
    loggers.append(MLFlowLogger(mlflowconf))

# --- Training ---
trainer = Trainer(trainconf, model, train_data, val_data, loggers)


trainer.run()

# --- Quick Test ---
print("\n\nTESTRUN (500 tokens):\n---------------------")
model.eval()
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
