from dataclasses import dataclass

import numpy as np
import torch

from tardits.modules.logging import ConsoleLogger


@dataclass
class TrainerConfig:
    batch_size: int = 32  # how many independent sequences processed in parallel?
    max_iters: int = 5000
    # Data-order seed, independent of model initialisation. Two models with the
    # same trainer seed see exactly the same training and eval batches.
    seed: int = 1337
    eval_interval: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    # RAT only: project |gamma| back onto the unit ball after each step so a
    # learned decay can never become a gain. 1.0 keeps the recurrence stable at
    # any context length; existing checkpoints are unaffected until trained.
    gamma_max: float = 1.0
    eval_iters: int = 200
    checkpoint: str = ""
    save_interval: int = 0


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        model,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        loggers=None,
    ):
        self.config = config
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.device = next(model.parameters()).device
        self.step = 0

        # Separate RNGs: model init consumes the global RNG differently for
        # different architectures, so a global-seeded data sampler would give
        # RealGPT and RAT different token orders despite the same YAML seed.
        self._train_gen = torch.Generator().manual_seed(int(config.seed))
        self._val_gen = torch.Generator().manual_seed(int(config.seed) + 1)

        self.loggers = loggers if loggers is not None else [ConsoleLogger()]

    def _cap_gamma(self) -> None:
        """Project RAT decay magnitudes onto the ``gamma_max`` ball.

        ``gamma`` is a complex/hypercomplex rotor; only its magnitude may grow
        without bound. Clamping the magnitude (not the phase) keeps
        ``S_t = S_{t-1} (x) gamma + ...`` non-amplifying for long contexts. This
        is a no-op for every current checkpoint (all |gamma| <= 1 so far) and for
        non-RAT models.
        """
        cap = float(self.config.gamma_max)
        if cap <= 0:
            return
        with torch.no_grad():
            for module in self.model.modules():
                gamma = getattr(module, "gamma", None)
                if not isinstance(gamma, torch.nn.Parameter):
                    continue
                norm = torch.linalg.vector_norm(gamma, dim=-1, keepdim=True)
                scale = torch.clamp(norm, max=cap) / norm.clamp(min=1e-8)
                gamma.mul_(scale)

    def _as_tensor(self, data, start: int, length: int) -> torch.Tensor:
        """Slice a tensor or a memmap-like array into a long tensor."""
        if isinstance(data, torch.Tensor):
            return data[start : start + length]
        return torch.from_numpy(np.asarray(data[start : start + length]).astype(np.int64))

    def get_batch(self, split, generator=None):
        data = self.train_data if split == "train" else self.val_data
        mc = self.model.config
        gen = generator
        if gen is None:
            gen = self._train_gen if split == "train" else self._val_gen
        ix = torch.randint(
            len(data) - mc.block_size, (self.config.batch_size,), generator=gen
        )
        x = torch.stack([self._as_tensor(data, int(i), mc.block_size) for i in ix])
        y = torch.stack([self._as_tensor(data, int(i) + 1, mc.block_size) for i in ix])
        return x.to(self.device), y.to(self.device)

    @torch.no_grad()
    def estimate_loss(self):
        conf = self.config
        model = self.model
        out = {}
        model.eval()  # sets "evaluation phase"
        for si, split in enumerate(("train", "val")):
            # Fixed generator per (seed, step, split): every architecture sees
            # the same eval batches at the same step, independent of model init
            # or how many training batches were drawn.
            eval_gen = torch.Generator().manual_seed(
                int(conf.seed) + 1_000_003 + si * 10_007 + int(self.step)
            )
            losses = torch.zeros(conf.eval_iters)
            for k in range(conf.eval_iters):
                X, Y = self.get_batch(split, generator=eval_gen)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()  # sets "training phase"
        return out

    def run(self):
        model = self.model
        conf = self.config

        for logger in self.loggers:
            logger.on_train_start(self)

        for iter in range(conf.max_iters):
            if iter % conf.eval_interval == 0:  # or iter == 10:
                losses = self.estimate_loss()

                for logger in self.loggers:
                    logger.on_eval(self, losses)
                if conf.checkpoint:
                    model.save_checkpoint(conf.checkpoint)

            if (
                conf.checkpoint
                and conf.save_interval > 0
                and iter % conf.save_interval == 0
            ):
                model.save_checkpoint(conf.checkpoint)

            xb, yb = self.get_batch("train")

            # evaluate loss
            logits, loss = model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self._cap_gamma()

            for logger in self.loggers:
                logger.on_step(self, loss)

            self.step += 1

        model.eval()
        final_losses = self.estimate_loss()

        for logger in self.loggers:
            logger.on_train_end(self, final_losses)

        if conf.checkpoint:
            model.save_checkpoint(conf.checkpoint)
