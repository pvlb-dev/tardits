import torch
import torch.nn as nn
import math


class QuaternionEmbedding(nn.Module):
    """embedds tokens into 4D space"""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb_r1 = nn.Embedding(vocab_size, d_model)
        self.emb_i1 = nn.Embedding(vocab_size, d_model)
        self.emb_r2 = nn.Embedding(vocab_size, d_model)
        self.emb_i2 = nn.Embedding(vocab_size, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            for emb in [self.emb_r1, self.emb_i1, self.emb_r2, self.emb_i2]:
                nn.init.normal_(
                    emb.weight, std=(1 / math.sqrt(4))
                )  # 4 * 0.5**2 = 1.0 varianz

    def forward(self, x):
        # x: (B, T)
        z1 = torch.complex(self.emb_r1(x), self.emb_i1(x))
        z2 = torch.complex(self.emb_r2(x), self.emb_i2(x))
        return z1, z2


class HLMHead(nn.Module):
    """4D HLMHead"""

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.lin = nn.Linear(d_model * 4, vocab_size, bias=False)

    def forward(self, z1, z2):
        # get real, touch gras
        features = torch.cat([z1.real, z1.imag, z2.real, z2.imag], dim=-1)
        return self.lin(features)
