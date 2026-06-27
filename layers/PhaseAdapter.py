import torch
import torch.nn as nn

from layers.Retrieval import PhaseAlignedIdeaBlockRetrieval


class PhaseResidualAdapter(nn.Module):
    """Backbone-agnostic phase retrieval residual adapter."""

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        period_len,
        phase_radius=1,
        num_cycles=4,
        topk=20,
        temperature=0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.channels = channels
        self.retriever = PhaseAlignedIdeaBlockRetrieval(
            seq_len=seq_len,
            pred_len=pred_len,
            channels=channels,
            period_len=period_len,
            phase_radius=phase_radius,
            num_cycles=num_cycles,
            topk=topk,
            temperature=temperature,
        )
        self.residual_projector = nn.Linear(pred_len, pred_len)
        self.gate = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def prepare_dataset(self, train_data):
        self.retriever.prepare_dataset(train_data)

    def forward(self, x, index_abs, train=False):
        retrieved, info = self.retriever.retrieve(
            x,
            index_abs=index_abs,
            train=train,
            return_info=True,
        )
        residual = self.residual_projector(retrieved.permute(0, 2, 1)).permute(0, 2, 1)
        gate_features = torch.cat(
            [
                info['top1_sim'],
                info['sim_gap'],
                info['weight_max'],
                1.0 - info['weight_entropy'],
            ],
            dim=1,
        )
        gate = self.gate(gate_features).view(-1, 1, 1)
        return gate * residual, info
