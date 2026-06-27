import torch
import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    """Embedding used by iTransformer: variates are tokens, time is feature."""

    def __init__(self, seq_len, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        x = x.permute(0, 2, 1)
        if x_mark is not None:
            x_mark = x_mark.permute(0, 2, 1)
            x = torch.cat([x, x_mark], dim=1)
        return self.dropout(self.value_embedding(x))
