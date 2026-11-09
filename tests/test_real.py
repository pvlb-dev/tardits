"""Behavioural tests for the RealGPT baseline (RoPE + causal mask + training).

The real model is the reference point for the RAT comparisons, so these tests
check the properties that matter for a fair benchmark: strict causality,
finite gradients, checkpoint round-trips, and that a tiny fixed batch can be
overfit (i.e. the optimisation path is actually learning).

Run with:  python -m unittest tests.test_real
or:         python tests/test_real.py
"""

import os
import sys
import tempfile
import unittest

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from tardits.models import KVCache, ModelConfig, RealGPT  # noqa: E402


def _tiny_config(**overrides):
    cfg = dict(
        level=0,
        block_size=8,
        vocab_size=16,
        n_head=2,
        n_embd=8,
        n_layer=2,
        dropout=0.0,
    )
    cfg.update(overrides)
    return ModelConfig(**cfg)


class TestRealGPTCausality(unittest.TestCase):
    def test_future_tokens_do_not_change_past_logits(self):
        torch.manual_seed(0)
        model = RealGPT(_tiny_config()).eval()
        idx = torch.randint(0, 16, (1, 8))
        logits_a, _ = model(idx)

        idx_b = idx.clone()
        idx_b[:, 5:] = (idx_b[:, 5:] + 1) % 16
        logits_b, _ = model(idx_b)

        torch.testing.assert_close(
            logits_a[:, :5], logits_b[:, :5], rtol=1e-5, atol=1e-6
        )


class TestRealGPTNormLayout(unittest.TestCase):
    def test_pre_norm_forward_and_block_flag(self):
        torch.manual_seed(0)
        model = RealGPT(_tiny_config(norm="pre")).eval()
        self.assertEqual(model.blocks[0].norm, "pre")
        logits, _ = model(torch.randint(0, 16, (1, 8)))
        self.assertTrue(torch.isfinite(logits).all())

    def test_branch_is_default(self):
        model = RealGPT(_tiny_config()).eval()
        self.assertEqual(model.blocks[0].norm, "branch")


class TestRealGPTKVCache(unittest.TestCase):
    def _model_and_prompt(self):
        torch.manual_seed(0)
        model = RealGPT(_tiny_config()).eval()
        prompt = torch.randint(0, 16, (1, 6))
        return model, prompt

    def test_cached_prefill_matches_full_forward(self):
        model, prompt = self._model_and_prompt()
        full, _ = model(prompt)

        caches = [KVCache() for _ in model.blocks]
        cached, _ = model(prompt, kv_cache=caches, start_pos=0)
        torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-6)

    def test_cached_step_matches_full_forward(self):
        model, prompt = self._model_and_prompt()
        caches = [KVCache() for _ in model.blocks]
        model(prompt, kv_cache=caches, start_pos=0)

        next_token = torch.tensor([[7]])
        step_logits, _ = model(next_token, kv_cache=caches, start_pos=prompt.shape[1])
        full_logits, _ = model(torch.cat((prompt, next_token), dim=1))

        torch.testing.assert_close(
            step_logits[:, -1, :], full_logits[:, -1, :], rtol=1e-5, atol=1e-6
        )

    def test_cached_generate_matches_generate(self):
        model, prompt = self._model_and_prompt()
        torch.manual_seed(1234)
        uncached = model.generate(prompt, max_new_tokens=2, top_k=5)
        torch.manual_seed(1234)
        cached = model.generate_cached(prompt, max_new_tokens=2, top_k=5)
        torch.testing.assert_close(cached, uncached)

    def test_long_cached_prefill_extends_rope_cache(self):
        model, _ = self._model_and_prompt()
        rope = model.blocks[0].sa.rope
        prompt = torch.randint(0, 16, (1, 20))
        self.assertGreater(prompt.shape[1], model.config.block_size)

        caches = [KVCache() for _ in model.blocks]
        logits, _ = model(prompt, kv_cache=caches, start_pos=0)

        self.assertTrue(torch.isfinite(logits).all())
        self.assertGreaterEqual(rope.cos_cached.shape[0], prompt.shape[1])
        # The extended rows must be the real RoPE values, not stale zeros.
        freqs = torch.outer(torch.tensor([19.0]), rope.inv_freq)
        expected = torch.cat((freqs, freqs), dim=-1)[0]
        torch.testing.assert_close(rope.cos_cached[19], expected.cos())
        torch.testing.assert_close(rope.sin_cached[19], expected.sin())

    def test_chunked_prefill_matches_single_call(self):
        model, _ = self._model_and_prompt()
        prompt = torch.randint(0, 16, (1, 20))

        single_caches = [KVCache() for _ in model.blocks]
        single, _ = model(prompt, kv_cache=single_caches, start_pos=0)

        chunked_caches = [KVCache() for _ in model.blocks]
        chunked = model.prefill(prompt, chunked_caches)

        # ``prefill`` returns the logits of the final chunk only.
        torch.testing.assert_close(
            chunked, single[:, -chunked.shape[1] :], rtol=1e-5, atol=1e-6
        )

    def test_cached_step_beyond_window_matches_wider_model_forward(self):
        torch.manual_seed(0)
        model = RealGPT(_tiny_config()).eval()
        # Same weights, larger RoPE cache: a reference for positions > block_size.
        wide = RealGPT(_tiny_config(block_size=32)).eval()
        state = {k: v for k, v in model.state_dict().items() if "tril" not in k}
        wide.load_state_dict(state, strict=False)

        prompt = torch.randint(0, 16, (1, 20))
        caches = [KVCache() for _ in model.blocks]
        model.prefill(prompt[:, :16], caches)
        step_logits, _ = model(prompt[:, 16:], kv_cache=caches, start_pos=16)

        full_logits, _ = wide(prompt)
        torch.testing.assert_close(
            step_logits, full_logits[:, 16:], rtol=1e-5, atol=1e-6
        )

    def test_generate_cached_beyond_block_size(self):
        model, _ = self._model_and_prompt()
        prompt = torch.randint(0, 16, (1, 12))
        self.assertGreater(prompt.shape[1], model.config.block_size)

        out = model.generate_cached(prompt, max_new_tokens=4, top_k=5)

        self.assertEqual(out.shape, (1, 16))
        self.assertTrue(torch.equal(out[:, :12], prompt))


class TestRealGPTTraining(unittest.TestCase):
    def test_forward_backward_is_finite(self):
        torch.manual_seed(0)
        model = RealGPT(_tiny_config()).train()
        x = torch.randint(0, 16, (2, 8))
        y = torch.randint(0, 16, (2, 8))

        _, loss = model(x, y)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertTrue(
                    torch.isfinite(param.grad).all().item(),
                    f"non-finite grad in {name}",
                )

    def test_tiny_batch_can_be_overfit(self):
        torch.manual_seed(0)
        model = RealGPT(_tiny_config(n_layer=1)).train()
        x = torch.randint(0, 16, (2, 8))
        y = torch.randint(0, 16, (2, 8))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        initial = None
        final = None
        for _ in range(60):
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            initial = loss.item() if initial is None else initial
            final = loss.item()

        self.assertLess(final, initial, "fixed batch loss did not decrease")


class TestRealGPTCheckpoint(unittest.TestCase):
    def test_save_load_roundtrip(self):
        torch.manual_seed(0)
        model = RealGPT(_tiny_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "real.pt")
            model.save_checkpoint(path)
            loaded = RealGPT.load_from_checkpoint(path, device="cpu")

        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[key])


if __name__ == "__main__":
    unittest.main()
