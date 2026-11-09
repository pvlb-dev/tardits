"""Tests for the model registry (the `arch` config option) and shipped configs.

`arch` selects the model family from a YAML config. It is popped out of the
`model` section before building :class:`ModelConfig`, so every config in
`examples/configs` must resolve to a registered class *and* only use real
ModelConfig fields. That is what the config sweep below checks.

Run with:  python -m unittest tests.test_registry
or:         python tests/test_registry.py
"""

import glob
import os
import sys
import unittest

import yaml
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from tardits.models import (  # noqa: E402
    MODEL_REGISTRY,
    ModelConfig,
    RAT,
    HyperFormer,
    RealGPT,
    available_archs,
    build_model,
    get_model_class,
    resolve_arch,
)

_TINY = dict(
    block_size=8, vocab_size=16, n_head=2, n_embd=8, n_layer=1, dropout=0.0
)


class TestResolveArch(unittest.TestCase):
    def test_canonical_names(self):
        self.assertEqual(resolve_arch("real"), "real")
        self.assertEqual(resolve_arch("hyper"), "hyper")
        self.assertEqual(resolve_arch("rat"), "rat")

    def test_case_and_separator_insensitive(self):
        self.assertEqual(resolve_arch("RAT"), "rat")
        self.assertEqual(resolve_arch(" Hyper "), "hyper")

    def test_aliases(self):
        self.assertEqual(resolve_arch("lina"), "rat")
        self.assertEqual(resolve_arch("HyperFormer"), "hyper")
        self.assertEqual(resolve_arch("gpt"), "real")

    def test_legacy_default_without_arch(self):
        """No `arch` key: keep the historical level-based rule."""
        self.assertEqual(resolve_arch(None, level=0), "real")
        self.assertEqual(resolve_arch(None, level=1), "hyper")
        self.assertEqual(resolve_arch("", level=2), "hyper")

    def test_unknown_arch_is_an_error(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_arch("quaternion")
        self.assertIn("rat", str(ctx.exception))

    def test_level_zero_rat_is_an_error(self):
        with self.assertRaises(ValueError):
            build_model("rat", ModelConfig(level=0, **_TINY))


class TestBuildModel(unittest.TestCase):
    def test_each_arch_builds_its_class(self):
        expected = {"real": RealGPT, "hyper": HyperFormer, "rat": RAT}
        for arch, cls in expected.items():
            with self.subTest(arch=arch):
                model = build_model(arch, ModelConfig(level=1, **_TINY))
                self.assertIsInstance(model, cls)

    def test_default_matches_get_model_class(self):
        cfg = ModelConfig(level=1, **_TINY)
        self.assertIsInstance(build_model(None, cfg), get_model_class(None, 1))

    def test_registry_is_the_source_of_truth(self):
        self.assertEqual(available_archs(), sorted(MODEL_REGISTRY))
        for cls in MODEL_REGISTRY.values():
            self.assertTrue(issubclass(cls, nn.Module))


class TestExampleConfigs(unittest.TestCase):
    """Every shipped config must be valid for `ttrain` / `tgen`."""

    def test_all_configs_resolve_to_a_model(self):
        paths = sorted(
            glob.glob(os.path.join(_REPO_ROOT, "examples", "configs", "*.yaml"))
        )
        paths += sorted(
            glob.glob(
                os.path.join(_REPO_ROOT, "experiments", "configs", "**", "*.yaml"),
                recursive=True,
            )
        )
        self.assertTrue(paths, "no example/experiment configs found")

        for path in paths:
            with self.subTest(config=os.path.basename(path)):
                with open(path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f)

                model_cfg = dict(raw["model"])
                arch = model_cfg.pop("arch", None)
                # raises TypeError on unknown/typo'd ModelConfig keys
                config = ModelConfig(**model_cfg)
                self.assertIn(
                    resolve_arch(arch, config.level),
                    MODEL_REGISTRY,
                    "configs must name a registered arch",
                )


if __name__ == "__main__":
    unittest.main()
