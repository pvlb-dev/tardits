import torch
import torch.nn as nn
import math


class SedeEmbedding(nn.Module):
    """embedds tokens into 16D space"""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb_r1 = nn.Embedding(vocab_size, d_model)
        self.emb_i1 = nn.Embedding(vocab_size, d_model)
        self.emb_r2 = nn.Embedding(vocab_size, d_model)
        self.emb_i2 = nn.Embedding(vocab_size, d_model)
        self.emb_r3 = nn.Embedding(vocab_size, d_model)
        self.emb_i3 = nn.Embedding(vocab_size, d_model)
        self.emb_r4 = nn.Embedding(vocab_size, d_model)
        self.emb_i4 = nn.Embedding(vocab_size, d_model)
        self.emb_r5 = nn.Embedding(vocab_size, d_model)
        self.emb_i5 = nn.Embedding(vocab_size, d_model)
        self.emb_r6 = nn.Embedding(vocab_size, d_model)
        self.emb_i6 = nn.Embedding(vocab_size, d_model)
        self.emb_r7 = nn.Embedding(vocab_size, d_model)
        self.emb_i7 = nn.Embedding(vocab_size, d_model)
        self.emb_r8 = nn.Embedding(vocab_size, d_model)
        self.emb_i8 = nn.Embedding(vocab_size, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            for emb in [
                self.emb_r1,
                self.emb_i1,
                self.emb_r2,
                self.emb_i2,
                self.emb_r3,
                self.emb_i3,
                self.emb_r4,
                self.emb_i4,
                self.emb_r5,
                self.emb_i5,
                self.emb_r6,
                self.emb_i6,
                self.emb_r7,
                self.emb_i7,
                self.emb_r8,
                self.emb_i8,
            ]:
                nn.init.normal_(emb.weight, std=(1 / math.sqrt(16)))

    def forward(self, x):
        # x: (B, T)
        z1 = torch.complex(self.emb_r1(x), self.emb_i1(x))
        z2 = torch.complex(self.emb_r2(x), self.emb_i2(x))
        z3 = torch.complex(self.emb_r3(x), self.emb_i3(x))
        z4 = torch.complex(self.emb_r4(x), self.emb_i4(x))
        z5 = torch.complex(self.emb_r5(x), self.emb_i5(x))
        z6 = torch.complex(self.emb_r6(x), self.emb_i6(x))
        z7 = torch.complex(self.emb_r7(x), self.emb_i7(x))
        z8 = torch.complex(self.emb_r8(x), self.emb_i8(x))
        return z1, z2, z3, z4, z5, z6, z7, z8


class SLMHead(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.lin = nn.Linear(d_model * 16, vocab_size, bias=False)

    def forward(self, z1, z2, z3, z4, z5, z6, z7, z8):
        # get real, touch gras
        features = torch.cat(
            [
                z1.real,
                z1.imag,
                z2.real,
                z2.imag,
                z3.real,
                z3.imag,
                z4.real,
                z4.imag,
                z5.real,
                z5.imag,
                z6.real,
                z6.imag,
                z7.real,
                z7.imag,
                z8.real,
                z8.imag,
            ],
            dim=-1,
        )
        return self.lin(features)
