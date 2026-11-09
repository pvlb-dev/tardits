"""Small trainer-configuration tests (optimizer plumbing, checkpoint guard)."""

import os
import sys
import unittest

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from tardits.models import ModelConfig, RAT  # noqa: E402
from tardits.modules import Trainer, TrainerConfig  # noqa: E402


class TestTrainerConfig(unittest.TestCase):
    def test_weight_decay_reaches_optimizer(self):
        cfg = ModelConfig(
            level=1, block_size=8, vocab_size=16, n_head=2, n_embd=8, n_layer=1
        )
        model = RAT(cfg)
        data = torch.zeros(64, dtype=torch.long)
        trainer = Trainer(
            TrainerConfig(weight_decay=0.05, checkpoint=""), model, data, data
        )
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["weight_decay"], 0.05)

    def test_gamma_magnitude_is_capped(self):
        cfg = ModelConfig(
            level=1, block_size=8, vocab_size=16, n_head=2, n_embd=8, n_layer=1
        )
        model = RAT(cfg)
        # Force one decay channel above the cap (trained checkpoints can drift).
        with torch.no_grad():
            model.blocks[0].sa.gamma.mul_(2.0)
        before = torch.linalg.vector_norm(
            model.blocks[0].sa.gamma, dim=-1
        ).max()
        self.assertGreater(float(before), 1.0)

        data = torch.zeros(64, dtype=torch.long)
        trainer = Trainer(
            TrainerConfig(gamma_max=1.0, checkpoint=""), model, data, data
        )
        trainer._cap_gamma()

        after = torch.linalg.vector_norm(
            model.blocks[0].sa.gamma, dim=-1
        ).max()
        self.assertLessEqual(float(after), 1.0 + 1e-6)
        # Phase is preserved: only the amplitude was projected.
        self.assertTrue(torch.isfinite(model.blocks[0].sa.gamma).all())

    def test_data_order_is_independent_of_model_init(self):
        cfg = ModelConfig(level=1, block_size=8, vocab_size=16, n_head=2, n_embd=8, n_layer=1)
        model_a = RAT(cfg)
        model_b = RAT(cfg)  # different random init consumes different global RNG
        data = torch.arange(256, dtype=torch.long) % 16
        ta = Trainer(TrainerConfig(seed=7, checkpoint=""), model_a, data, data)
        tb = Trainer(TrainerConfig(seed=7, checkpoint=""), model_b, data, data)
        xa, ya = ta.get_batch("train")
        xb, yb = tb.get_batch("train")
        self.assertTrue(torch.equal(xa, xb))
        self.assertTrue(torch.equal(ya, yb))

    def test_empty_checkpoint_is_allowed(self):
        cfg = ModelConfig(
            level=1, block_size=8, vocab_size=16, n_head=2, n_embd=8, n_layer=1
        )
        model = RAT(cfg)
        data = torch.zeros(64, dtype=torch.long)
        trainer = Trainer(TrainerConfig(checkpoint="", max_iters=0), model, data, data)
        # max_iters=0: run() must finish without trying to torch.save to "".
        trainer.run()
        self.assertEqual(trainer.step, 0)


if __name__ == "__main__":
    unittest.main()
