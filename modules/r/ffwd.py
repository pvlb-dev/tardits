import torch.nn as nn
from torch.nn import functional as F


class GELU(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SwiGLU(nn.Module):
    def __init__(self, n_embd, hidden_dim, dropout):
        super().__init__()
        self.w = nn.Linear(n_embd, hidden_dim, bias=False)
        self.v = nn.Linear(n_embd, hidden_dim, bias=False)

        self.out_proj = nn.Linear(hidden_dim, n_embd, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w(x))
        activated = gate * self.v(x)

        out = self.out_proj(activated)
        return self.dropout(out)
