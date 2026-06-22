import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class PhaseRetrievalTool:
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)
        self.temperature = temperature
        self.topm = topm

        self.key_mg = None
        self.key_mg_norm = None
        self.value_mg = None
        self.train_indices = None

    @staticmethod
    def _pad_to_group(data, group):
        length = data.shape[1]
        pad_len = math.ceil(length / group) * group - length
        if pad_len == 0:
            return data, length
        pad = data[:, -1:, :].repeat(1, pad_len, 1)
        return torch.cat([data, pad], dim=1), length

    def downsample(self, data_all):
        data_all = copy.deepcopy(data_all)
        mg = []

        for group in self.period_num:
            data_pad, orig_len = self._pad_to_group(data_all, group)
            cur = data_pad.unfold(dimension=1, size=group, step=group).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=group, dim=1)
            mg.append(cur[:, :orig_len, :])

        return torch.stack(mg, dim=0)

    def prepare_dataset(self, train_data):
        train_x_all = []
        train_y_all = []
        train_indices = []

        for i in range(len(train_data)):
            index, seq_x, seq_y, _, _ = train_data[i]
            x_last = seq_x[-1:, :]
            train_x_all.append(seq_x - x_last)
            train_y_all.append(seq_y[-train_data.pred_len:] - x_last)
            train_indices.append(index)

        train_x_all = torch.tensor(np.stack(train_x_all, axis=0)).float()
        train_y_all = torch.tensor(np.stack(train_y_all, axis=0)).float()

        self.key_mg = self.downsample(train_x_all)
        key_flat = self.key_mg.flatten(start_dim=2)
        key_centered = key_flat - key_flat.mean(dim=2, keepdim=True)
        self.key_mg_norm = F.normalize(key_centered, dim=2)

        self.value_mg = self.downsample(train_y_all)
        self.train_indices = torch.tensor(train_indices).long()
        self.n_train = train_x_all.shape[0]

    def periodic_batch_corr(self, key, in_bsz=512):
        _, _, features = key.shape
        _, train_len, _ = self.key_mg_norm.shape

        query = key - key.mean(dim=2, keepdim=True)
        query = F.normalize(query, dim=2)

        iters = math.ceil(train_len / in_bsz)
        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)
            cur_key = self.key_mg_norm[:, start_idx:end_idx].to(key.device)
            cur_sim = torch.bmm(query, cur_key.transpose(-1, -2))
            sim.append(cur_sim)

        return torch.cat(sim, dim=2)

    def retrieve(self, x, index, train=True):
        index = index.to(x.device).long()

        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset
        x_mg = self.downsample(x_norm.cpu()).to(x.device)

        sim = self.periodic_batch_corr(
            x_mg.flatten(start_dim=2),
        )

        if train:
            raw_sim = sim
            train_indices = self.train_indices.to(x.device)
            overlap = (index.unsqueeze(1) - train_indices.unsqueeze(0)).abs()
            overlap = overlap < (self.seq_len + self.pred_len)
            sim = sim.masked_fill(overlap.unsqueeze(0), float('-inf'))
            all_masked = torch.isinf(sim).all(dim=2, keepdim=True)
            sim = torch.where(all_masked, raw_sim, sim)

        flat_sim = sim.reshape(self.n_period * bsz, self.n_train)
        topm = min(self.topm, self.n_train)
        topm_index = torch.topk(flat_sim, topm, dim=1).indices
        ranking_sim = torch.ones_like(flat_sim) * float('-inf')

        rows = torch.arange(flat_sim.size(0)).unsqueeze(-1).to(flat_sim.device)
        ranking_sim[rows, topm_index] = flat_sim[rows, topm_index]

        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train)
        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)
        ranking_prob = ranking_prob.detach().cpu()

        value_flat = self.value_mg.flatten(start_dim=2)
        pred_from_retrieval = torch.bmm(ranking_prob, value_flat)
        pred_from_retrieval = pred_from_retrieval.reshape(self.n_period, bsz, -1, channels)

        return pred_from_retrieval.to(x.device)

    def retrieve_all(self, data, train=False, device=torch.device('cpu')):
        assert self.key_mg is not None

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

        return torch.cat(retrievals, dim=1)
