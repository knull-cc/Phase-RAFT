import torch
import torch.nn as nn

from layers.Retrieval import PhaseAlignedIdeaBlockRetrieval


class Model(nn.Module):
    """Phase-aligned IdeaBlock Retrieval forecaster."""

    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        if configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{configs.gpu}')
        else:
            self.device = torch.device('cpu')

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len if self.task_name in [
            'classification',
            'anomaly_detection',
            'imputation',
        ] else configs.pred_len
        self.channels = configs.enc_in

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)

        self.retriever = PhaseAlignedIdeaBlockRetrieval(
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
        self.retriever.prepare_dataset(train_data)
        self.data_borders = {
            'train': int(getattr(train_data, 'border1', 0)),
            'valid': int(getattr(valid_data, 'border1', 0)) if valid_data is not None else 0,
            'test': int(getattr(test_data, 'border1', 0)) if test_data is not None else 0,
        }

    def encoder(self, x, index, mode):
        bsz, seq_len, channels = x.shape
        if seq_len != self.seq_len or channels != self.channels:
            raise ValueError('RAFT input shape does not match configured seq_len/channels')

        border = self.data_borders.get(mode, 0)
        index_abs = index.to(x.device).long() + int(border)

        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset
        backbone = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)
        backbone = backbone + x_offset

        retrieved_future = self.retriever.retrieve(
            x,
            index_abs=index_abs,
            train=mode == 'train',
        )

        pred = torch.cat([backbone, retrieved_future], dim=1)
        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1)
        return pred.reshape(bsz, self.pred_len, self.channels)

    def forecast(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def imputation(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def anomaly_detection(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def classification(self, x_enc, index, mode):
        enc_out = self.encoder(x_enc, index, mode)
        output = enc_out.reshape(enc_out.shape[0], -1)
        return self.projection(output)

    def forward(self, x_enc, index, mode='train'):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, index, mode)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, index, mode)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc, index, mode)
        if self.task_name == 'classification':
            return self.classification(x_enc, index, mode)
        return None
