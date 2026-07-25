# pyright: reportAttributeAccessIssue=false
import torch
from torch.nn import functional as F
import os


def get_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "auto" or not requested_device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon GPU
        return torch.device("cpu")
    return torch.device(requested_device)


class LMCoreMixin:
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temp=1.0, top_k=10):
        device = next(self.parameters()).device
        idx = idx.to(device)
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size max_new_tokens
            idx_cond = idx[:, -self.config.block_size :]
            # get predictions
            logits, _ = self.forward(idx_cond)
            # focus only on the last timestep
            logits = logits[:, -1, :]  # becomes (B,C)
            # temp
            logits = logits / temp
            # top_k
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx

    @torch.no_grad()
    def generate_stream(self, idx, temp=1.0, top_k=10):
        while True:
            # slicing:
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]  # becomes (B,C)

            # temp/top_k
            logits = logits / temp
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

            # sampling
            probs = F.softmax(logits, dim=-1)  # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)

            # append
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)

            # yield single token
            yield idx_next.item()

    def save_checkpoint(self, filepath="checkpoint.pt"):
        """saves model weights and config to disk"""
        checkpoint = {"model_state_dict": self.state_dict(), "config": self.config}
        torch.save(checkpoint, filepath)
        # print(f"💾 checkpoint successfully saved: {filepath}")

    def get_num_params(self):
        """returns number of trainable weights"""
        total_params = 0
        for p in self.parameters():
            if p.requires_grad:
                if p.is_complex():
                    total_params += p.numel() * 2
                else:
                    total_params += p.numel()
        return total_params

    @classmethod
    def load_from_checkpoint(cls, filepath="checkpoint.pt", device="cpu"):
        """loads model state from checkpoint file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ file not found:\n{filepath}")

        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model = cls(checkpoint["config"])  # type: ignore
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        # print(f"🚀 checkpoint successfully restored from:\n'{filepath}'")
        return model
