import torch.nn as nn

# ── 2. NETWORK ARCHITECTURE ────────────────────────────────────────────────────
#
# Residual MLP with MC-Dropout for uncertainty.
# Skip connections help gradients flow for the physics Jacobian computation.


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))  # residual connection


class TracerPINN(nn.Module):
    def __init__(self, n_features, hidden_dim=256, n_blocks=4, dropout=0.15):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
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
