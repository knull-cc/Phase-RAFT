import importlib

import torch
import torch.nn as nn

from layers.PhaseAdapter import PhaseResidualAdapter


class Model(nn.Module):
    """PIBR as a host-model wrapper with phase-aligned residual retrieval."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.host_name = getattr(configs, 'pibr_host', 'Linear')

        host_module = importlib.import_module(f'models.{self.host_name}')
        self.host = host_module.Model(configs)
        self.phase_adapter = PhaseResidualAdapter(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            period_len=configs.period_len,
            phase_radius=configs.idea_block_radius,
            num_cycles=configs.idea_block_cycles,
            topk=configs.topm,
            temperature=configs.temperature,
        )
        self.data_borders = {}

    def prepare_dataset(self, train_data, valid_data=None, test_data=None):
        self.phase_adapter.prepare_dataset(train_data)
        self.data_borders = {
            'train': int(getattr(train_data, 'border1', 0)),
            'valid': int(getattr(valid_data, 'border1', 0)) if valid_data is not None else 0,
            'test': int(getattr(test_data, 'border1', 0)) if test_data is not None else 0,
        }

    def _host_forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if self.host_name in ['Linear', 'DLinear', 'NLinear']:
            return self.host(x_enc)
        return self.host(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def forecast(self, x_enc, index, mode='train', x_mark_enc=None, x_dec=None, x_mark_dec=None):
        bsz, seq_len, channels = x_enc.shape
        if seq_len != self.seq_len or channels != self.channels:
            raise ValueError('PIBR input shape does not match configured seq_len/channels')

        base = self._host_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        border = self.data_borders.get(mode, 0)
        index_abs = index.to(x_enc.device).long() + int(border)
        residual, _ = self.phase_adapter(x_enc, index_abs=index_abs, train=mode == 'train')
        return base[:, -self.pred_len:, :] + residual

    def forward(
        self,
        x_enc,
        index,
        mode='train',
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
    ):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, index, mode, x_mark_enc, x_dec, x_mark_dec)
        raise NotImplementedError('PIBR currently supports forecasting only')
