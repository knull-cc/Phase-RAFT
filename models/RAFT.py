import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.CrossPhaseRouting import MultiPeriodPhaseBranch
from layers.Retrieval import RetrievalTool

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        if configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{configs.gpu}')
        else:
            self.device = torch.device('cpu')
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
#         self.decompsition = series_decomp(configs.moving_avg)
#         self.individual = individual
        self.channels = configs.enc_in

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        
        self.n_period = configs.n_period
        self.topm = configs.topm
        
        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            temperature=configs.temperature,
            topm=self.topm,
            retrieval_variant=configs.retrieval_variant,
            phase_top_m=configs.phase_top_m,
            phase_lambda=configs.phase_lambda,
            phase_tau=configs.phase_tau,
            phase_period=configs.phase_period,
        )
        
        self.period_num = self.rt.period_num[-1 * self.n_period:]
        
        module_list = [
            nn.Linear(self.pred_len // g, self.pred_len)
            for g in self.period_num
        ]
        self.retrieval_pred = nn.ModuleList(module_list)
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)
        self.phase_fusion = configs.phase_fusion
        self.phase_branch = None
        if self.phase_fusion:
            self.phase_periods = self._parse_phase_periods(configs)
            self.phase_branch = MultiPeriodPhaseBranch(
                periods=self.phase_periods,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                latent_dim=configs.latent_dim,
                n_layers=configs.phase_layers,
                num_routers=configs.phase_num_routers,
                num_heads=configs.phase_heads,
                dropout=configs.phase_attn_dropout,
            )

#         if self.task_name == 'classification':
#             self.projection = nn.Linear(
#                 configs.enc_in * configs.seq_len, configs.num_class)

    @staticmethod
    def _parse_phase_periods(configs):
        if configs.period_list:
            periods = [int(item.strip()) for item in configs.period_list.split(',') if item.strip()]
        else:
            periods = [configs.period_len]
        periods = [period for period in periods if period > 0]
        if not periods:
            raise ValueError('phase fusion requires at least one positive period')
        return periods

    def prepare_dataset(self, train_data, valid_data, test_data):
        self.rt.prepare_dataset(train_data)
        
        self.retrieval_dict = {}
        
        print('Doing Train Retrieval')
        train_rt = self.rt.retrieve_all(train_data, train=True, device=self.device)

        print('Doing Valid Retrieval')
        valid_rt = self.rt.retrieve_all(valid_data, train=False, device=self.device)

        print('Doing Test Retrieval')
        test_rt = self.rt.retrieve_all(test_data, train=False, device=self.device)

        del self.rt
        torch.cuda.empty_cache()
            
        self.retrieval_dict['train'] = train_rt.detach()
        self.retrieval_dict['valid'] = valid_rt.detach()
        self.retrieval_dict['test'] = test_rt.detach()

    def encoder(self, x, index, mode):
        index_cpu = index.detach().cpu().long()
        
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels
        
        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset

        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1) # B, P, C
        
        pred_from_retrieval = self.retrieval_dict[mode][:, index_cpu] # G, B, P, C
        pred_from_retrieval = pred_from_retrieval.to(self.device)
        
        retrieval_pred_list = []
        
        # Compress repeating dimensions
        for i, pr in enumerate(pred_from_retrieval):
            assert((bsz, self.pred_len, channels) == pr.shape)
            g = self.period_num[i]
            pr = pr.reshape(bsz, self.pred_len // g, g, channels)
            pr = pr[:, :, 0, :]
            
            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            pr = pr.reshape(bsz, self.pred_len, self.channels)
            
            retrieval_pred_list.append(pr)

        retrieval_pred_list = torch.stack(retrieval_pred_list, dim=1)
        retrieved_future = retrieval_pred_list.sum(dim=1)
        
        pred = torch.cat([x_pred_from_x, retrieved_future], dim=1)
        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)
        if self.phase_branch is not None:
            pred = pred + self.phase_branch(x_norm, retrieved_future)
        
        pred = pred + x_offset
        
        return pred

    def forecast(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def imputation(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def anomaly_detection(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def classification(self, x_enc, index, mode):
        # Encoder
        enc_out = self.encoder(x_enc, index, mode)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, index, mode='train'):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, index, mode)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, index, mode)
            return dec_out  # [B, N]
        return None
