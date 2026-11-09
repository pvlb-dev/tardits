"""Tests for the real-matrix HyperLinear implementation (Cayley-Dickson levels >= 2).

The optimized ``HyperLinear.forward`` spreads the hypercomplex weight into a real
``[out * n, in * n]`` matrix and applies it with a single ``F.linear`` (BLAS) call.
These tests verify it is numerically equivalent to the naive recursive
``cd_mul(...).sum(dim=-2)`` formulation and that its parameter layout (state_dict)
is unchanged, so existing checkpoints remain loadable.

Run with:  python -m unittest tests.test_hyperlinear
or:         python tests/test_hyperlinear.py
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tardits.modules.core import HyperLinear, cd_mul  # noqa: E402


def _reference_forward(weight, bias, x, level):
    """Naive recursive reference for the old HyperLinear level >= 2 path."""
    x_expanded = x.unsqueeze(-3)
    prod = cd_mul(x_expanded, weight, level=level)
    out = prod.sum(dim=-2)
    if bias is not None:
        out = out + bias
    return out


class TestHyperLinearEquivalence(unittest.TestCase):
    def test_matches_recursive_reference_levels_2_to_4(self):
        torch.manual_seed(0)
        for level in (2, 3, 4):
            for (in_f, out_f) in ((8, 8), (24, 24), (12, 7)):
                with self.subTest(level=level, in_f=in_f, out_f=out_f):
                    layer = HyperLinear(in_f, out_f, level=level)
                    x = torch.randn(
                        4, 6, in_f, 2 ** (level - 1), dtype=torch.complex64
                    )
                    y_new = layer(x)
                    y_ref = _reference_forward(layer.weight, layer.bias, x, level)
                    diff = (y_new - y_ref).abs().max().item()
                    self.assertLess(diff, 1e-5)

    def test_state_dict_layout_unchanged(self):
        layer = HyperLinear(8, 8, level=3)
        sd = layer.state_dict()
        self.assertIn("weight", sd)
        self.assertIn("bias", sd)
        self.assertEqual(tuple(sd["weight"].shape), (8, 8, 4))  # [out, in, hyper_dim]
        self.assertEqual(tuple(sd["bias"].shape), (8, 4))       # [out, hyper_dim]
        self.assertEqual(sd["weight"].dtype, torch.complex64)


if __name__ == "__main__":
    unittest.main()
