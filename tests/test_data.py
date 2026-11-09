"""Data-path tests: memmap shards must work like in-memory tensors.

Run:  python -m unittest tests.test_data
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from tardits.models import ModelConfig, RealGPT  # noqa: E402
from tardits.modules.trainer import Trainer, TrainerConfig  # noqa: E402


class TestMemmapData(unittest.TestCase):
    def test_get_batch_from_memmap(self):
        rng = np.random.default_rng(0)
        train = rng.integers(0, 16, size=5000, dtype=np.uint16)
        val = rng.integers(0, 16, size=1000, dtype=np.uint16)

        with tempfile.TemporaryDirectory() as tmp:
            train_path = os.path.join(tmp, "train.bin")
            val_path = os.path.join(tmp, "val.bin")
            train.tofile(train_path)
            val.tofile(val_path)

            cfg = ModelConfig(
                level=0,
                block_size=16,
                vocab_size=16,
                n_head=2,
                n_embd=8,
                n_layer=1,
                dropout=0.0,
            )
            model = RealGPT(cfg)
            trainer = Trainer(
                TrainerConfig(batch_size=4, max_iters=2, eval_interval=100, eval_iters=2),
                model,
                np.memmap(train_path, dtype=np.uint16, mode="r"),
                np.memmap(val_path, dtype=np.uint16, mode="r"),
            )

            xb, yb = trainer.get_batch("train")
            self.assertEqual(tuple(xb.shape), (4, 16))
            self.assertEqual(tuple(yb.shape), (4, 16))
            self.assertEqual(xb.dtype, torch.long)
            # Targets are the next token at every position.
            torch.testing.assert_close(yb[:, :-1], xb[:, 1:])
            self.assertTrue(int(xb.max()) < cfg.vocab_size)

            # eval path also exercises the memmap/value split
            losses = trainer.estimate_loss()
            self.assertTrue(torch.isfinite(losses["train"]))
            self.assertTrue(torch.isfinite(losses["val"]))


if __name__ == "__main__":
    unittest.main()
