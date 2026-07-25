import torch
import torch.nn as nn
import math


class ComplexEmbedding(nn.Module):
    """embedds tokens into complex space"""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb_r = nn.Embedding(vocab_size, d_model)
        self.emb_i = nn.Embedding(vocab_size, d_model)
        self._reset_parameters()
        # self._old_reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            for emb in [self.emb_r, self.emb_i]:
                nn.init.normal_(emb.weight, std=(1 / math.sqrt(2)))

    def forward(self, x):
        # x: (B, T)
        real = self.emb_r(x)
        imag = self.emb_i(x)
        return torch.complex(real, imag)


class CLMHead(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.lin = nn.Linear(d_model * 2, vocab_size, bias=False)

    def forward(self, z):
        # get real, touch gras
        real_imag = torch.cat([z.real, z.imag], dim=-1)
        return self.lin(real_imag)
