import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),  # C^infinity smooth activation for stable autograd
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))  # Residual connection


class TracerPINN(nn.Module):
    def __init__(self, n_features, hidden_dim=256, n_blocks=4, dropout=0.15):
        super().__init__()
        # Input embedding without LayerNorm
        self.embed = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h).squeeze(-1)
