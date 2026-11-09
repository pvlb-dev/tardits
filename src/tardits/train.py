from argparse import ArgumentParser
import yaml
import os

import numpy as np
import torch

from tardits.modules import Trainer, TrainerConfig, MLFlowConfig
from tardits.modules.logging import ConsoleLogger, MLFlowLogger
from tardits.modules.base import get_device
from tardits.models import ModelConfig, available_archs, build_model, resolve_arch


def main():
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
    parser.add_argument(
        "--arch",
        "-a",
        type=str,
        default=None,
        help=(
            "Override the model family selected in the config "
            f"({' | '.join(available_archs())})"
        ),
    )
    args = parser.parse_args()

    # --- Load YAML Config ---
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = get_device(config.get("device", "auto"))
    model_cfg_dict = config["model"]
    # `arch` selects the model class, so it is not part of ModelConfig
    config_arch = model_cfg_dict.pop("arch", None)
    try:
        arch = resolve_arch(args.arch or config_arch, model_cfg_dict.get("level", 1))
    except ValueError as e:
        parser.error(str(e))

    trainer_cfg_dict = config["trainer"]
    mlflow_cfg_dict = dict(config.get("mlflow", {}))
    data_cfg_dict = config.get("data", {})

    seed = config.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)

    # --- Data Preparation ---
    vocab_path = data_cfg_dict.get("vocab_save_path", "examples/saved/chars.txt")
    train_bin = data_cfg_dict.get("train_bin")
    val_bin = data_cfg_dict.get("val_bin")

    if train_bin:
        # Large-corpus path: pre-tokenised uint16/uint32 shards, memory-mapped
        # (the text path materialises the whole corpus and cannot scale to
        # FineWeb-sized data).
        val_bin = val_bin or train_bin
        dtype = np.dtype(data_cfg_dict.get("dtype", "uint16"))
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"vocab missing for bin data: {vocab_path} (write it from the "
                "pre-tokenisation meta/charset before training)"
            )
        with open(vocab_path, "r", encoding="utf-8") as f:
            chars = list(f.read())
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        def decode(li):
            return "".join([itos[i] for i in li])

        train_data = np.memmap(train_bin, dtype=dtype, mode="r")
        val_data = np.memmap(val_bin, dtype=dtype, mode="r")
        dataset_label = os.path.basename(train_bin)
        print(f"📚 memmap {train_bin} ({len(train_data):,} tokens), "
              f"val {len(val_data):,}, vocab {vocab_size}")
    else:
        data_path = data_cfg_dict.get("input_path", "examples/data/shakespeare.txt")
        with open(data_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        chars = sorted(list(set(full_text)))

        dirname = os.path.dirname(vocab_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

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
        dataset_label = os.path.basename(data_path)

    # Make runs self-describing in MLflow without adding trainer fields: the
    # dataset filename and seed are exactly what distinguishes these sweeps.
    mlflow_tags = dict(mlflow_cfg_dict.get("tags") or {})
    mlflow_tags.setdefault("dataset", dataset_label)
    mlflow_tags.setdefault("seed", str(seed))
    mlflow_cfg_dict["tags"] = mlflow_tags

    # --- Model & Configurations ---
    modelconf = ModelConfig(vocab_size=vocab_size, **model_cfg_dict)

    model = build_model(arch, modelconf).to(device)
    print(
        f"🧠 {type(model).__name__} | level {modelconf.level} "
        f"| {modelconf.n_layer} layers"
    )

    trainer_cfg_dict.setdefault("seed", seed if seed is not None else 1337)
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


if __name__ == "__main__":
    main()
