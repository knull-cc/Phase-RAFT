import torch
import torch.nn as nn

from layers.CrossPhaseRouting import MultiPeriodPhaseBranch
from layers.PhaseRetrieval import PhaseRetrievalTool


class Model(nn.Module):
    """RAFT with retrieval performed in the phase-token space.

    The accuracy backbone (DLinear direct path + learned fusion) is kept from
    RAFT; only the retrieval similarity is moved to a phase-aligned space (see
    ``layers/PhaseRetrieval.py``). The learned fusion guarantees the model can
    fall back to the DLinear prediction when retrieval is not helpful.
    """

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
        self.topm = configs.topm
        self.no_retrieval = configs.no_retrieval
        self.period_len = configs.period_len
        self.use_revin = getattr(configs, 'use_revin', False)
        self.revin_eps = 1e-5
        self.diversity_aware = getattr(configs, 'diversity_aware', False)
        self.rel_dim = 2 if (self.diversity_aware and not self.no_retrieval) else 0

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)

        self.rt = None
        if not self.no_retrieval:
            self.rt = PhaseRetrievalTool(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                channels=self.channels,
                period_len=self.period_len,
                temperature=configs.temperature,
                topm=self.topm,
                norm_mode='revin' if self.use_revin else 'offset',
                eps=self.revin_eps,
                diversity_aware=self.diversity_aware,
                retrieval_pool=getattr(configs, 'retrieval_pool', 100),
                nms_gap=getattr(configs, 'nms_gap', None),
                rel_topk=getattr(configs, 'rel_topk', 5),
            )

        self.retrieval_pred = nn.Linear(self.pred_len, self.pred_len)
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)

        self.use_phase_routing = getattr(configs, 'phase_routing', True)
        if self.use_phase_routing:
            self.phase_periods = self._parse_periods(
                getattr(configs, 'period_list', None), self.period_len
            )
            self.phase_branch = MultiPeriodPhaseBranch(
                periods=self.phase_periods,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                latent_dim=getattr(configs, 'latent_dim', 64),
                n_layers=getattr(configs, 'phase_layers', 1),
                num_routers=getattr(configs, 'phase_num_routers', 8),
                num_heads=getattr(configs, 'phase_heads', 4),
                dropout=getattr(configs, 'phase_attn_dropout', 0.1),
                rel_dim=self.rel_dim,
            )

        self.retrieval_dict = {}
        self.reliability_dict = {}

    @staticmethod
    def _parse_periods(period_list, period_len):
        if not period_list:
            return [period_len]
        if isinstance(period_list, (list, tuple)):
            periods = [int(p) for p in period_list]
        else:
            periods = [int(p) for p in str(period_list).split(',') if str(p).strip()]
        return periods or [period_len]

    def prepare_dataset(self, train_data, valid_data, test_data):
        if self.no_retrieval:
            print('Phase-RAFT retrieval disabled (--no-retrieval).')
            return

        self.rt.prepare_dataset(train_data)

        print('Doing Phase-RAFT Train Retrieval')
        train_rt, train_rel = self.rt.retrieve_all(train_data, train=True, device=self.device)

        print('Doing Phase-RAFT Valid Retrieval')
        valid_rt, valid_rel = self.rt.retrieve_all(valid_data, train=False, device=self.device)

        print('Doing Phase-RAFT Test Retrieval')
        test_rt, test_rel = self.rt.retrieve_all(test_data, train=False, device=self.device)

        del self.rt
        self.rt = None
        torch.cuda.empty_cache()

        self.retrieval_dict['train'] = train_rt.detach()
        self.retrieval_dict['valid'] = valid_rt.detach()
        self.retrieval_dict['test'] = test_rt.detach()

        if train_rel is not None:
            self.reliability_dict['train'] = train_rel.detach()
            self.reliability_dict['valid'] = valid_rel.detach()
            self.reliability_dict['test'] = test_rel.detach()

    def _retrieved_future(self, bsz, channels, index, mode, device, dtype):
        if self.no_retrieval:
            return torch.zeros(bsz, self.pred_len, channels, device=device, dtype=dtype)

        index_cpu = index.detach().cpu().long()
        pred_from_retrieval = self.retrieval_dict[mode][:, index_cpu].to(device)  # [1, B, P, C]
        pred_from_retrieval = pred_from_retrieval[0]  # B, P, C

        retrieval_pred = self.retrieval_pred(pred_from_retrieval.permute(0, 2, 1)).permute(0, 2, 1)
        return retrieval_pred.reshape(bsz, self.pred_len, channels)

    def _reliability(self, index, mode, device, dtype):
        if self.rel_dim == 0 or mode not in self.reliability_dict:
            return None
        index_cpu = index.detach().cpu().long()
        return self.reliability_dict[mode][index_cpu].to(device=device, dtype=dtype)  # B, rel_dim

    def encoder(self, x, index, mode):
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        if self.use_revin:
            mu = x.mean(dim=1, keepdim=True).detach()
            sigma = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.revin_eps).detach()
            x_norm = (x - mu) / sigma
        else:
            x_offset = x[:, -1:, :].detach()
            x_norm = x - x_offset

        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)  # B, P, C

        retrieval_pred = self._retrieved_future(
            bsz=bsz,
            channels=channels,
            index=index,
            mode=mode,
            device=x.device,
            dtype=x.dtype,
        )

        pred = torch.cat([x_pred_from_x, retrieval_pred], dim=1)
        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1)
        pred = pred.reshape(bsz, self.pred_len, channels)

        if self.use_phase_routing:
            rel = self._reliability(index, mode, x.device, x.dtype)
            pred = pred + self.phase_branch(x_norm, retrieval_pred, rel=rel)

        if self.use_revin:
            return pred * sigma + mu
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
