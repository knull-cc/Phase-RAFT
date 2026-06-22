import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class PhaseRetrievalTool:
    """Retrieval in the phase-token space.

    Instead of RAFT's multi-granularity downsampling, the lookback window is
    folded by ``period_len`` into phase tokens. Similarity is computed as the
    average of per-phase-position cosine similarities, so two windows match when
    they are aligned phase-by-phase. The retrieved value is the softmax-weighted
    average of the corresponding (offset-normalized) future windows.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        period_len=24,
        temperature=0.1,
        topm=20,
        norm_mode='offset',
        eps=1e-5,
    ):
        if period_len <= 0:
            raise ValueError('period_len must be positive')
        if norm_mode not in ('offset', 'revin'):
            raise ValueError("norm_mode must be 'offset' or 'revin'")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.period_len = period_len
        self.temperature = temperature
        self.topm = topm
        self.norm_mode = norm_mode
        self.eps = eps

        self.key_phase_norm = None  # [T, period_len, n_periods * C]
        self.value_all = None       # [T, pred_len, C]
        self.train_indices = None
        self.n_train = 0

    def _phase_feature(self, data):
        # data: [N, L, C] -> [N, period_len, n_periods * C], per-phase centered & L2-normalized
        n, length, c = data.shape
        n_periods = math.ceil(length / self.period_len)
        target_len = n_periods * self.period_len

        if target_len > length:
            pad = data[:, -1:, :].repeat(1, target_len - length, 1)
            data = torch.cat([data, pad], dim=1)

        folded = data.reshape(n, n_periods, self.period_len, c)
        folded = folded.permute(0, 2, 1, 3).contiguous()           # N, period_len, n_periods, C
        folded = folded.reshape(n, self.period_len, n_periods * c)  # N, period_len, n_periods*C

        # remove per-phase DC offset, then normalize so matching is on shape
        folded = folded - folded.mean(dim=2, keepdim=True)
        return F.normalize(folded, dim=2)

    def prepare_dataset(self, train_data):
        train_x_all = []
        train_y_all = []
        train_indices = []

        for i in range(len(train_data)):
            index, seq_x, seq_y, _, _ = train_data[i]
            seq_x = torch.as_tensor(seq_x).float()
            seq_y = torch.as_tensor(seq_y[-train_data.pred_len:]).float()

            if self.norm_mode == 'revin':
                mu = seq_x.mean(dim=0, keepdim=True)
                sigma = torch.sqrt(seq_x.var(dim=0, keepdim=True, unbiased=False) + self.eps)
                train_x_all.append((seq_x - mu) / sigma)
                train_y_all.append((seq_y - mu) / sigma)
            else:
                x_last = seq_x[-1:, :]
                train_x_all.append(seq_x - x_last)
                train_y_all.append(seq_y - x_last)
            train_indices.append(index)

        train_x_all = torch.stack(train_x_all, dim=0).float()
        train_y_all = torch.stack(train_y_all, dim=0).float()

        self.key_phase_norm = self._phase_feature(train_x_all)
        self.value_all = train_y_all
        self.train_indices = torch.tensor(train_indices).long()
        self.n_train = train_x_all.shape[0]

    def _phase_corr(self, query, in_bsz=512):
        # query: [B, period_len, F] -> sim: [B, T]
        # Averaging per-phase cosine similarity == dot product of the flattened
        # per-phase-normalized vectors divided by period_len. Use matmul instead
        # of einsum to avoid materializing a huge [B, K, period_len, F] tensor.
        bsz = query.shape[0]
        q_flat = query.reshape(bsz, -1)

        sims = []
        iters = math.ceil(self.n_train / in_bsz)
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, self.n_train)
            cur_key = self.key_phase_norm[start_idx:end_idx].to(query.device)
            cur_key = cur_key.reshape(cur_key.shape[0], -1)
            cur_sim = torch.matmul(q_flat, cur_key.transpose(0, 1)) / self.period_len
            sims.append(cur_sim)
        return torch.cat(sims, dim=1)

    def retrieve(self, x, index, train=True):
        index = index.to(x.device).long()

        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset

        query = self._phase_feature(x_norm)
        sim = self._phase_corr(query)

        if train:
            raw_sim = sim
            train_indices = self.train_indices.to(sim.device)
            overlap = (index.unsqueeze(1) - train_indices.unsqueeze(0)).abs()
            overlap = overlap < (self.seq_len + self.pred_len)
            sim = sim.masked_fill(overlap, float('-inf'))
            all_masked = torch.isinf(sim).all(dim=1, keepdim=True)
            sim = torch.where(all_masked, raw_sim, sim)

        topm = min(self.topm, self.n_train)
        topm_index = torch.topk(sim, topm, dim=1).indices
        ranking_sim = torch.full_like(sim, float('-inf'))

        rows = torch.arange(sim.size(0), device=sim.device).unsqueeze(-1)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]

        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=1)
        ranking_prob = ranking_prob.detach().cpu()

        value_flat = self.value_all.reshape(self.n_train, -1)
        pred_from_retrieval = torch.mm(ranking_prob, value_flat)
        pred_from_retrieval = pred_from_retrieval.reshape(bsz, self.pred_len, channels)

        return pred_from_retrieval.unsqueeze(0).to(x.device)  # [1, B, P, C]

    def retrieve_all(self, data, train=False, device=torch.device('cpu')):
        assert self.key_phase_norm is not None

        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )

        retrievals = []
        with torch.no_grad():
            for index, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(rt_loader):
                pred_from_retrieval = self.retrieve(batch_x.float().to(device), index, train=train)
                retrievals.append(pred_from_retrieval.cpu())

        return torch.cat(retrievals, dim=1)  # [1, T, P, C]
