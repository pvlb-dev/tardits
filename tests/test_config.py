"""Validation tests for ModelConfig and model-shape preconditions.

These catch the shapes that used to fail later inside a ``view()`` or a RoPE
buffer slice, with a useful error instead of a cryptic runtime shape mismatch.

Run with:  python -m unittest tests.test_config
or:         python tests/test_config.py
"""

import os
import sys
import unittest

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from tardits.models import ModelConfig, RealGPT  # noqa: E402


class TestModelConfigValidation(unittest.TestCase):
    def test_valid_config_still_constructs(self):
        cfg = ModelConfig(
            level=1, block_size=8, vocab_size=16, n_head=2, n_embd=8, n_layer=1
        )
        self.assertEqual(cfg.n_head, 2)

    def test_indivisible_heads_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ModelConfig(
                level=1, block_size=8, vocab_size=16, n_head=3, n_embd=10, n_layer=1
            )
        self.assertIn("divisible", str(ctx.exception))

    def test_norm_layout_is_validated(self):
        with self.assertRaises(ValueError):
            ModelConfig(
                level=0,
                block_size=8,
                vocab_size=16,
                n_head=2,
                n_embd=8,
                n_layer=1,
                norm="sideways",
            )
        cfg = ModelConfig(
            level=0,
            block_size=8,
            vocab_size=16,
            n_head=2,
            n_embd=8,
            n_layer=1,
            norm="pre",
        )
        self.assertEqual(cfg.norm, "pre")

    def test_non_positive_dimensions_are_rejected(self):
        for kwargs in (
            dict(n_embd=0),
            dict(n_head=0),
            dict(n_layer=0),
            dict(block_size=0),
            dict(vocab_size=0),
            dict(level=-1),
        ):
            with self.subTest(**kwargs):
                base = dict(
                    level=1,
                    block_size=8,
                    vocab_size=16,
                    n_head=2,
                    n_embd=8,
                    n_layer=1,
                )
                base.update(kwargs)
                with self.assertRaises(ValueError):
                    ModelConfig(**base)


class TestRealGPTPreconditions(unittest.TestCase):
    def test_odd_head_size_is_rejected_for_rope(self):
        with self.assertRaises(ValueError) as ctx:
            RealGPT(
                ModelConfig(
                    level=0,
                    block_size=8,
                    vocab_size=16,
                    n_head=4,
                    n_embd=12,
                    n_layer=1,
                )
            )
        self.assertIn("even head_size", str(ctx.exception))

    def test_too_long_context_is_rejected(self):
        model = RealGPT(
            ModelConfig(
                level=0,
                block_size=4,
                vocab_size=16,
                n_head=2,
                n_embd=8,
                n_layer=1,
            )
        )
        idx = torch.zeros((1, 5), dtype=torch.long)

        with self.assertRaises(ValueError) as ctx:
            model(idx)
        self.assertIn("exceeds block_size", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
