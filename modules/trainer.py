import torch
from modules.conf import TrainerConfig
from modules.logging import ConsoleLogger


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
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        self.device = next(model.parameters()).device
        self.step = 0

        self.loggers = loggers if loggers is not None else [ConsoleLogger()]

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        mc = self.model.config
        ix = torch.randint(len(data) - mc.block_size, (self.config.batch_size,))
        x = torch.stack([data[i : i + mc.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + mc.block_size] for i in ix])
        return x.to(self.device), y.to(self.device)

    @torch.no_grad()
    def estimate_loss(self):
        conf = self.config
        model = self.model
        out = {}
        model.eval()  # sets "evaluation phase"
        for split in ["train", "val"]:
            losses = torch.zeros(conf.eval_iters)
            for k in range(conf.eval_iters):
                X, Y = self.get_batch(split)
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
                model.save_checkpoint(conf.checkpoint)

            if conf.save_interval > 0 and iter % conf.save_interval == 0:
                model.save_checkpoint(conf.checkpoint)

            xb, yb = self.get_batch("train")

            # evaluate loss
            logits, loss = model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            for logger in self.loggers:
                logger.on_step(self, loss)

            self.step += 1

        model.eval()
        final_losses = self.estimate_loss()

        for logger in self.loggers:
            logger.on_train_end(self, final_losses)

        if conf.checkpoint:
            model.save_checkpoint(conf.checkpoint)
