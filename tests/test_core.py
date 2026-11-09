"""Tests for core.py primitives.

Covers the ``HyperRMSNorm`` amplitude optimization: it now computes ``|z|^2`` as
``real^2 + imag^2`` instead of ``abs() ** 2`` (which does a sqrt then a square).
The two are mathematically equal; this guards the numeric agreement.

Run with:  python -m unittest tests.test_core
or:         python tests/test_core.py
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tardits.modules.core import HyperRMSNorm  # noqa: E402


def _reference_norm(x, weight, eps, level):
    """The previous implementation (abs() ** 2)."""
    ampsq = (x.abs() ** 2).sum(dim=-1)
    ampsq = ampsq / (2 ** (level - 1))
    rms = torch.rsqrt(ampsq.mean(dim=-1, keepdim=True) + eps).unsqueeze(-1)
    return (x * rms) * weight


class TestHyperRMSNorm(unittest.TestCase):
    def test_matches_abs_squared_reference(self):
        torch.manual_seed(0)
        for level, hyper_dim in ((1, 1), (2, 2), (3, 4)):
            for shape in ((4, 10, 6, hyper_dim), (2, 5, 3, hyper_dim)):
                with self.subTest(level=level, shape=shape):
                    norm = HyperRMSNorm(shape[-2], level=level)
                    x = torch.randn(shape, dtype=torch.complex64)
                    got = norm(x)
                    ref = _reference_norm(x, norm.weight, norm.eps, level)
                    diff = (got - ref).abs().max().item()
                    self.assertLess(diff, 1e-5)

    def test_real_input_uses_real_part(self):
        norm = HyperRMSNorm(6, level=1)
        x = torch.randn(3, 6, 1)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
