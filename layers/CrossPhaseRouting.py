import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossPhaseRoutingLayer(nn.Module):
    """Router-based attention across phase positions (PhaseFormer style).

    Operates on ``Z`` of shape ``(B, C, L, D)`` where ``L`` is the number of
    phase positions within a period. A small set of learnable routers first
    aggregate information from all phase positions, then distribute it back,
    giving a cheap global mixing across phases without cross-channel leakage.
    """

    def __init__(self, latent_dim, num_routers=8, num_heads=4, dropout=0.0, mlp_ratio=4):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError('latent_dim must be divisible by num_heads')

        self.router = nn.Parameter(torch.randn(num_routers, latent_dim))
        nn.init.trunc_normal_(self.router, std=0.02)

        self.sender = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.receiver = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_ratio * latent_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * latent_dim, latent_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, Z):
        b, c, l, d = Z.shape
        x = Z.reshape(b * c, l, d)

        routers = self.router.unsqueeze(0).expand(b * c, -1, -1)
        router_buffer, _ = self.sender(routers, x, x)        # BC, R, D
        router_receive, _ = self.receiver(x, router_buffer, router_buffer)  # BC, L, D

        out = self.norm1(x + self.drop(router_receive))
        out = self.norm2(out + self.drop(self.mlp(out)))
        return out.reshape(b, c, l, d)


class PhaseRoutingBranch(nn.Module):
    """Retrieval-augmented cross-phase routing branch.

    Folds the (offset-normalized) lookback window and the retrieved future into
    phase tokens, concatenates them along the period axis, runs cross-phase
    routing, and predicts the future phase steps. The final predictor is
    zero-initialized so the branch outputs 0 at init (acts as a pure residual).
    """

    def __init__(
        self,
        period_len,
        seq_len,
        pred_len,
        latent_dim=64,
        n_layers=1,
        num_routers=8,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.period_len = period_len
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.p_in = (seq_len + period_len - 1) // period_len
        self.p_out = (pred_len + period_len - 1) // period_len

        self.embed = nn.Sequential(
            nn.Linear(self.p_in + self.p_out, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.layers = nn.ModuleList([
            CrossPhaseRoutingLayer(latent_dim, num_routers, num_heads, dropout)
            for _ in range(n_layers)
        ])
        self.predictor = nn.Linear(latent_dim, self.p_out)
        nn.init.zeros_(self.predictor.weight)
        nn.init.zeros_(self.predictor.bias)

    def _to_phase(self, x, n_periods):
        # x: (B, L, C) -> (B, C, period_len, n_periods)
        b, length, c = x.shape
        target_len = n_periods * self.period_len
        if target_len > length:
            x = x.permute(0, 2, 1)
            x = F.pad(x, (0, target_len - length), mode='replicate')
            x = x.permute(0, 2, 1)
        x = x.reshape(b, n_periods, self.period_len, c)
        return x.permute(0, 3, 2, 1).contiguous()

    def forward(self, x_norm, retrieved):
        x_phase = self._to_phase(x_norm, self.p_in)        # B, C, pl, p_in
        r_phase = self._to_phase(retrieved, self.p_out)    # B, C, pl, p_out
        tokens = torch.cat([x_phase, r_phase], dim=-1)     # B, C, pl, p_in + p_out

        z = self.embed(tokens)
        for layer in self.layers:
            z = layer(z)
        y = self.predictor(z)                              # B, C, pl, p_out

        y = y.permute(0, 3, 2, 1).contiguous()             # B, p_out, pl, C
        bsz, p_out, pl, c = y.shape
        y = y.reshape(bsz, p_out * pl, c)
        return y[:, :self.pred_len, :]
