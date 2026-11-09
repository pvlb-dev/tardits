"""Tests for HyperDropout, the complex-safe dropout used by RAT.

``nn.Dropout`` raises ``NotImplementedError`` for complex tensors, so RAT uses
``HyperDropout``: a single real mask shared across the real and imaginary parts,
with standard scaled-dropout semantics.

Run with:  python -m unittest tests.test_dropout
or:         python tests/test_dropout.py
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tardits.modules.core import HyperDropout  # noqa: E402


class TestHyperDropout(unittest.TestCase):
    def test_complex_drops_whole_entries(self):
        torch.manual_seed(0)
        p = 0.5
        layer = HyperDropout(p).train()
        x = torch.randn(20_000, dtype=torch.complex64)
        y = layer(x)

        self.assertEqual(y.dtype, torch.complex64)
        # Dropped fraction ~= p
        self.assertAlmostEqual((y == 0).float().mean().item(), p, delta=0.02)
        # Real and imaginary parts are dropped together (single shared mask)
        self.assertTrue(torch.equal(y.real == 0, y.imag == 0))
        # Survivors are scaled by 1 / (1 - p)
        kept = y != 0
        self.assertTrue(torch.allclose(y[kept], x[kept] / (1.0 - p)))

    def test_real_input_supported(self):
        torch.manual_seed(1)
        p = 0.5
        layer = HyperDropout(p).train()
        x = torch.randn(20_000)
        y = layer(x)

        self.assertEqual(y.dtype, torch.float32)
        self.assertAlmostEqual((y == 0).float().mean().item(), p, delta=0.02)

    def test_eval_and_zero_are_identity(self):
        x = torch.randn(64, dtype=torch.complex64)
        self.assertTrue(torch.equal(HyperDropout(0.5).eval()(x), x))
        self.assertTrue(torch.equal(HyperDropout(0.0).train()(x), x))

    def test_p_one_returns_zeros(self):
        x = torch.randn(64, dtype=torch.complex64)
        y = HyperDropout(1.0).train()(x)
        self.assertTrue((y == 0).all())

    def test_invalid_p_raises(self):
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                HyperDropout(bad)

    def test_has_no_parameters_or_buffers(self):
        layer = HyperDropout(0.2)
        self.assertEqual(len(list(layer.parameters())), 0)
        self.assertEqual(len(list(layer.buffers())), 0)
        self.assertEqual(len(layer.state_dict()), 0)


if __name__ == "__main__":
    unittest.main()
