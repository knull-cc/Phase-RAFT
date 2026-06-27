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
        fusion='phase_only',
        projector='identity',
        gate_init_bias=-4.0,
        residual_scale=1.0,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.channels = channels
        self.fusion = fusion
        self.residual_scale = residual_scale
        self.projector = projector
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
        if projector == 'identity':
            self.residual_projector = nn.Identity()
        elif projector == 'linear':
            self.residual_projector = nn.Linear(pred_len, pred_len)
        else:
            raise ValueError(f'unknown PIBR projector: {projector}')
        self.gate = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self._init_safe(gate_init_bias)

    def _init_safe(self, gate_init_bias):
        if isinstance(self.residual_projector, nn.Linear):
            nn.init.eye_(self.residual_projector.weight)
            nn.init.zeros_(self.residual_projector.bias)
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)
        self.gate[-2].bias.data.fill_(gate_init_bias)

    def prepare_dataset(self, train_data):
        self.retriever.prepare_dataset(train_data)

    def forward(self, x, index_abs, base_forecast=None, train=False):
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
        phase_forecast = x[:, -1:, :] + residual
        info['gate'] = gate.squeeze(-1).squeeze(-1)

        if base_forecast is None:
            return phase_forecast, info

        if self.fusion == 'phase_only':
            return phase_forecast, info

        if self.fusion == 'fixed_avg':
            return 0.5 * base_forecast + 0.5 * phase_forecast, info

        if self.fusion == 'residual_add':
            return base_forecast + self.residual_scale * residual, info

        if self.fusion == 'learned_gate':
            correction = phase_forecast - base_forecast
            return base_forecast + self.residual_scale * gate * correction, info

        raise ValueError(f'unknown PIBR fusion mode: {self.fusion}')
