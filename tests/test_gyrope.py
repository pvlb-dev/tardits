"""GyRoPE must rotate the intended hypercomplex axis.

The original packing sliced ``real_components`` into first/second halves, which
interleaves components wrongly for level >= 2 (a quaternion i-rotation landed
on j). This test pins the component order used by ``cd_mul``.

Run with:  python -m unittest tests.test_gyrope
or:         python tests/test_gyrope.py
"""

import math
import os
import sys
import unittest

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from tardits.modules import GyRoPE  # noqa: E402


def _rotor(level, theta, axis_index=0):
    rope = GyRoPE(head_size=1, block_size=4, n_head=1, level=level)
    with torch.no_grad():
        rope.theta.fill_(theta)
        rope.axis_params.zero_()
        rope.axis_params[0, 0, axis_index] = 1.0
    return rope


class TestGyRoPEAxes(unittest.TestCase):
    def test_level1_rotates_into_imaginary_axis(self):
        rope = _rotor(level=1, theta=math.pi / 2)
        x = torch.zeros(1, 1, 1, 1, 1, dtype=torch.complex64)
        x[..., 0] = 1.0

        y = rope(x, start_pos=1)
        torch.testing.assert_close(
            y.flatten(), torch.tensor([1j], dtype=torch.complex64)
        )

    def test_level2_rotates_into_the_first_quaternion_axis(self):
        """Quaternion i, not j: component 0 = 1j, component 1 = 0."""
        rope = _rotor(level=2, theta=math.pi / 2, axis_index=0)
        x = torch.zeros(1, 1, 1, 1, 2, dtype=torch.complex64)
        x[..., 0] = 1.0

        y = rope(x, start_pos=1)
        expected = torch.tensor([1j, 0j], dtype=torch.complex64)
        torch.testing.assert_close(y.flatten(), expected)

    def test_rotor_preserves_hypercomplex_norm(self):
        torch.manual_seed(0)
        rope = GyRoPE(head_size=2, block_size=4, n_head=1, level=2)
        x = torch.randn(1, 1, 1, 2, 2, dtype=torch.complex64)

        y = rope(x, start_pos=1)
        norm_x = torch.linalg.vector_norm(x, dim=-1)
        norm_y = torch.linalg.vector_norm(y, dim=-1)
        torch.testing.assert_close(norm_y, norm_x, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
