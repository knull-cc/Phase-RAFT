import math

import torch
import torch.nn as nn

from layers.PhaseRetrieval import PhaseRetrievalTool
from layers.PhaseTokenizer import PhaseTokenizer


class ShallowPhasePredictor(nn.Module):
    def __init__(self, input_periods, output_periods, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_periods),
            nn.Linear(input_periods, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_periods),
        )

    def forward(self, tokens):
        bsz, channels, period_len, input_periods = tokens.shape
        y = self.net(tokens.reshape(-1, input_periods))
        return y.reshape(bsz, channels, period_len, -1)


class Model(nn.Module):
    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        if configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{configs.gpu}')
        else:
            self.device = torch.device('cpu')

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name in ['classification', 'anomaly_detection', 'imputation']:
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.channels = configs.enc_in
        self.n_period = configs.n_period
        self.topm = configs.topm
        self.no_retrieval = configs.no_retrieval
        self.period_len = configs.period_len

        self.tokenizer = PhaseTokenizer(self.period_len)
        self.input_phase_periods = self.tokenizer.n_periods(self.seq_len)
        self.output_phase_periods = self.tokenizer.n_periods(self.pred_len)

        self.rt = None
        if not self.no_retrieval:
            self.rt = PhaseRetrievalTool(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                channels=self.channels,
                n_period=self.n_period,
                temperature=configs.temperature,
                topm=self.topm,
            )
            self.period_num = self.rt.period_num[-1 * self.n_period:]
        else:
            period_num = [16, 8, 4, 2, 1]
            self.period_num = sorted(period_num[-1 * self.n_period:], reverse=True)

        module_list = [
            nn.Linear(math.ceil(self.pred_len / group), self.pred_len)
            for group in self.period_num
        ]
        self.retrieval_pred = nn.ModuleList(module_list)

        hidden_dim = max(1, configs.phase_hidden)
        self.phase_predictor = ShallowPhasePredictor(
            input_periods=self.input_phase_periods + self.output_phase_periods,
            output_periods=self.output_phase_periods,
            hidden_dim=hidden_dim,
            dropout=configs.dropout,
        )

        self.retrieval_dict = {}

    def prepare_dataset(self, train_data, valid_data, test_data):
        if self.no_retrieval:
            print('Phase-RAFT retrieval disabled (--no-retrieval).')
            return

        self.rt.prepare_dataset(train_data)

        print('Doing Phase-RAFT Train Retrieval')
        train_rt = self.rt.retrieve_all(train_data, train=True, device=self.device)

        print('Doing Phase-RAFT Valid Retrieval')
        valid_rt = self.rt.retrieve_all(valid_data, train=False, device=self.device)

        print('Doing Phase-RAFT Test Retrieval')
        test_rt = self.rt.retrieve_all(test_data, train=False, device=self.device)

        del self.rt
        self.rt = None
        torch.cuda.empty_cache()

        self.retrieval_dict['train'] = train_rt.detach()
        self.retrieval_dict['valid'] = valid_rt.detach()
        self.retrieval_dict['test'] = test_rt.detach()

    def _compress_retrieval_period(self, x, group):
        bsz, length, channels = x.shape
        target_len = math.ceil(length / group) * group
        pad_len = target_len - length

        if pad_len > 0:
            pad = x[:, -1:, :].repeat(1, pad_len, 1)
            x = torch.cat([x, pad], dim=1)

        x = x.reshape(bsz, target_len // group, group, channels)
        return x[:, :, 0, :]

    def _retrieved_future(self, bsz, channels, index, mode, device, dtype):
        if self.no_retrieval:
            return torch.zeros(bsz, self.pred_len, channels, device=device, dtype=dtype)

        index_cpu = index.detach().cpu().long()
        pred_from_retrieval = self.retrieval_dict[mode][:, index_cpu].to(device)

        retrieval_pred_list = []
        for i, pred_period in enumerate(pred_from_retrieval):
            assert (bsz, self.pred_len, channels) == pred_period.shape
            group = self.period_num[i]
            pred_period = self._compress_retrieval_period(pred_period, group)
            pred_period = self.retrieval_pred[i](pred_period.permute(0, 2, 1))
            pred_period = pred_period.permute(0, 2, 1).reshape(bsz, self.pred_len, channels)
            retrieval_pred_list.append(pred_period)

        retrieval_pred_list = torch.stack(retrieval_pred_list, dim=1)
        return retrieval_pred_list.sum(dim=1)

    def encoder(self, x, index, mode):
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        x_norm, x_offset = self.tokenizer.offset_normalize(x)
        retrieved_future = self._retrieved_future(
            bsz=bsz,
            channels=channels,
            index=index,
            mode=mode,
            device=x.device,
            dtype=x.dtype,
        )

        x_phase = self.tokenizer.to_phase(x_norm)
        retrieved_phase = self.tokenizer.to_phase(retrieved_future)
        phase_tokens = torch.cat([x_phase, retrieved_phase], dim=-1)

        y_phase = self.phase_predictor(phase_tokens)
        pred = self.tokenizer.to_time(y_phase, self.pred_len)
        return pred + x_offset

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
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, index, mode)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, index, mode)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc, index, mode)
        if self.task_name == 'classification':
            return self.classification(x_enc, index, mode)
        return None
