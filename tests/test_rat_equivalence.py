"""Tests that RAT's parallel ``forward`` matches its recurrent ``step``.

This guards the train/inference consistency of the fast-weight memory. For
Cayley-Dickson level >= 4 (sedenion and beyond) the algebra is non-alternative,
so the de Moivre parallel scan is replaced by an exact sequential recurrence;
these tests assert ``forward`` and ``step`` agree for all supported levels.
``generate`` and ``generate_stream`` must also sample the same tokens, since
both drive ``step``.

Run with:  python -m unittest tests.test_rat_equivalence
or:         python tests/test_rat_equivalence.py
"""

import os
import sys
import unittest

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from tardits.models import ModelConfig, RAT  # noqa: E402


def _tiny_config(level=1, dropout=0.2):
    return ModelConfig(
        level=level,
        block_size=8,
        vocab_size=16,
        n_head=2,
        n_embd=8,
        n_layer=1,
        dropout=dropout,
    )


def _forward_vs_step_max_diff(level):
    cfg = _tiny_config(level=level)
    m = RAT(cfg).eval()
    B, T = 1, 5
    idx = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits_fwd, _ = m.forward(idx)

        states = m.init_states(B, torch.device("cpu"))
        logits_step = []
        for t in range(T):
            lo, states = m.step(idx[:, t : t + 1], states, pos=t)
            logits_step.append(lo)

    max_diff = 0.0
    for t in range(T):
        d = (logits_fwd[:, t, :] - logits_step[t][:, 0, :]).abs().max().item()
        max_diff = max(max_diff, d)
    return max_diff


class TestForwardStepEquivalence(unittest.TestCase):
    def test_forward_matches_step_all_levels(self):
        torch.manual_seed(0)
        for level in (1, 2, 3, 4):
            with self.subTest(level=level):
                diff = _forward_vs_step_max_diff(level)
                self.assertLess(diff, 1e-4)


class TestRATChunkedTrainingForward(unittest.TestCase):
    def test_forward_matches_step_beyond_chunk(self):
        """T > chunk must still equal the recurrence, including level 1."""
        torch.manual_seed(0)
        cfg = ModelConfig(
            level=1,
            block_size=128,
            vocab_size=16,
            n_head=2,
            n_embd=8,
            n_layer=1,
            dropout=0.0,
        )
        model = RAT(cfg).eval()
        # Trained decays are below 1; force the regime that broke the old scan.
        with torch.no_grad():
            for block in model.blocks:
                block.sa.gamma.mul_(0.8)

        idx = torch.randint(0, cfg.vocab_size, (1, 260))
        with torch.no_grad():
            logits_fwd, _ = model.forward(idx)
            states = model.init_states(1, torch.device("cpu"))
            logits_step = []
            for t in range(idx.shape[1]):
                lo, states = model.step(idx[:, t : t + 1], states, pos=t)
                logits_step.append(lo)

        max_diff = max(
            (logits_fwd[:, t, :] - logits_step[t][:, 0, :]).abs().max().item()
            for t in range(idx.shape[1])
        )
        self.assertLess(max_diff, 1e-3, f"chunked forward drift {max_diff}")

    def test_forward_matches_step_trained_gamma_at_block(self):
        """Regression for #23: trained |gamma|<1 must not saturate a scan inverse."""
        torch.manual_seed(0)
        cfg = ModelConfig(
            level=1,
            block_size=128,
            vocab_size=16,
            n_head=2,
            n_embd=8,
            n_layer=1,
            dropout=0.0,
        )
        model = RAT(cfg).eval()
        with torch.no_grad():
            for block in model.blocks:
                gamma = block.sa.gamma
                mag = torch.linalg.vector_norm(gamma, dim=-1, keepdim=True)
                gamma.mul_(0.83 / mag.clamp(min=1e-8))

        idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
        with torch.no_grad():
            logits_fwd, _ = model.forward(idx)
            states = model.init_states(1, torch.device("cpu"))
            logits_step = []
            for t in range(idx.shape[1]):
                lo, states = model.step(idx[:, t : t + 1], states, pos=t)
                logits_step.append(lo)

        max_diff = max(
            (logits_fwd[:, t, :] - logits_step[t][:, 0, :]).abs().max().item()
            for t in range(idx.shape[1])
        )
        self.assertLess(max_diff, 1e-4, f"trained-gamma forward drift {max_diff}")


class TestRATBidirectional(unittest.TestCase):
    def _model(self, level=1, bidirectional=True):
        cfg = ModelConfig(
            level=level,
            block_size=16,
            vocab_size=16,
            n_head=2,
            n_embd=8,
            n_layer=1,
            dropout=0.0,
            bidirectional=bidirectional,
        )
        return RAT(cfg).eval(), cfg

    def test_symmetric_kernel_is_reversal_equivariant(self):
        for level in (1, 2, 3, 4):
            with self.subTest(level=level):
                torch.manual_seed(0)
                model, cfg = self._model(level=level)
                seq = torch.randint(0, cfg.vocab_size, (1, 8))
                with torch.no_grad():
                    out = model(seq)[0]
                    out_rev = model(seq.flip(1))[0]
                torch.testing.assert_close(
                    out, out_rev.flip(1), rtol=1e-5, atol=1e-6
                )

    def test_generation_is_rejected(self):
        model, _ = self._model()
        seq = torch.randint(0, 16, (1, 8))
        with self.assertRaises(ValueError):
            model.generate(seq, max_new_tokens=1)
        with self.assertRaises(ValueError):
            next(model.generate_stream(seq))
        with self.assertRaises(ValueError):
            model.prefill(seq)
        with self.assertRaises(ValueError):
            model.step(seq[:, :1], model.init_states(1, torch.device("cpu")), 0)

    def test_window_must_fit_scan_chunk(self):
        model, cfg = self._model()
        seq = torch.randint(0, cfg.vocab_size, (1, 40))
        with self.assertRaises(ValueError):
            model(seq)


class TestRATPrefill(unittest.TestCase):
    def test_prefill_matches_sequential_warmup(self):
        """The scan-based prefill must produce the same memory as stepping."""
        torch.manual_seed(0)
        for level in (1, 2, 3, 4):
            with self.subTest(level=level):
                cfg = _tiny_config(level=level, dropout=0.0)
                model = RAT(cfg).eval()
                idx = torch.randint(0, cfg.vocab_size, (1, 6))

                with torch.no_grad():
                    logits_prefill, states_prefill = model.prefill(idx)
                    states_step = model.init_states(1, torch.device("cpu"))
                    logits_step, states_step = model._warm_states(idx, states_step)

                torch.testing.assert_close(
                    logits_prefill[:, -1, :],
                    logits_step[:, -1, :],
                    rtol=1e-4,
                    atol=1e-5,
                )
                for prefill_state, step_state in zip(states_prefill, states_step):
                    self.assertLess(
                        (prefill_state - step_state).abs().max().item(),
                        1e-4,
                        f"state drift at level {level}",
                    )

    def test_chunked_prefill_beyond_block_size(self):
        """Chunking the scan and carrying the state must match the recurrence."""
        torch.manual_seed(0)
        for level in (1, 2, 3, 4):
            with self.subTest(level=level):
                cfg = ModelConfig(
                    level=level,
                    block_size=4,
                    vocab_size=16,
                    n_head=2,
                    n_embd=8,
                    n_layer=1,
                    dropout=0.0,
                )
                model = RAT(cfg).eval()
                idx = torch.randint(0, cfg.vocab_size, (1, 10))
                self.assertGreater(idx.shape[1], cfg.block_size)

                with torch.no_grad():
                    logits_prefill, states_prefill = model.prefill(idx)
                    states_step = model.init_states(1, torch.device("cpu"))
                    logits_step, states_step = model._warm_states(idx, states_step)

                torch.testing.assert_close(
                    logits_prefill[:, -1, :],
                    logits_step[:, -1, :],
                    rtol=1e-3,
                    atol=1e-4,
                )
                for prefill_state, step_state in zip(states_prefill, states_step):
                    self.assertLess(
                        (prefill_state - step_state).abs().max().item(),
                        1e-3,
                        f"chunked carry drift at level {level}",
                    )

    def test_prefill_empty_prompt(self):
        model = RAT(_tiny_config()).eval()
        logits, states = model.prefill(torch.empty((1, 0), dtype=torch.long))
        self.assertIsNone(logits)
        self.assertEqual(len(states), len(model.blocks))


class TestRATDropout(unittest.TestCase):
    def test_active_in_train_identity_in_eval(self):
        torch.manual_seed(0)
        cfg = _tiny_config(dropout=0.5)
        m = RAT(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 8))

        m.train()
        with torch.no_grad():
            a = m.forward(idx)[0]
            b = m.forward(idx)[0]
        self.assertFalse(torch.allclose(a, b), "dropout should randomize train forwards")

        m.eval()
        with torch.no_grad():
            c = m.forward(idx)[0]
            d = m.forward(idx)[0]
        self.assertTrue(torch.equal(c, d), "eval forwards must be deterministic")


class TestRATGeneration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = RAT(_tiny_config()).eval()
        self.prompt = torch.tensor([[1, 2, 3, 4]])

    def test_generate_ignores_block_size(self):
        """RAT keeps memory across the whole context, it is not window-limited."""
        out = self.model.generate(self.prompt, max_new_tokens=6, temp=1.0, top_k=5)
        self.assertEqual(tuple(out.shape), (1, 4 + 6))
        self.assertTrue(torch.equal(out[:, :4], self.prompt))

    def test_stream_matches_generate(self):
        """Both paths sample from the same logits, so the seeds must line up."""
        tokens = 5
        torch.manual_seed(1234)
        sampled = self.model.generate(self.prompt, max_new_tokens=tokens, top_k=5)

        torch.manual_seed(1234)
        stream = []
        for token in self.model.generate_stream(self.prompt, top_k=5):
            stream.append(token)
            if len(stream) == tokens:
                break

        self.assertEqual(stream, sampled[0, -tokens:].tolist())

    def test_stream_rejects_empty_prompt(self):
        with self.assertRaises(ValueError):
            next(self.model.generate_stream(torch.empty((1, 0), dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
