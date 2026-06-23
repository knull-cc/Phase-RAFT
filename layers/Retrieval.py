import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

class RetrievalTool():
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        retrieval_variant='A',
        phase_top_m=5,
        phase_lambda=0.1,
        phase_tau=2.0,
        phase_period=None,
        with_dec=False,
        return_key=False,
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
        self.retrieval_variant = retrieval_variant
        self.phase_top_m = phase_top_m
        self.phase_lambda = phase_lambda
        self.phase_tau = phase_tau
        self.phase_period = phase_period
        
        self.with_dec = with_dec
        self.return_key = return_key
        
    def prepare_dataset(self, train_data):
        train_data_all = []
        y_data_all = []
        train_indices = []
        train_abs_indices = []
        base_index = getattr(train_data, 'border1', 0)

        for i in range(len(train_data)):
            td = train_data[i]
            train_indices.append(td[0])
            train_abs_indices.append(td[0] + base_index)
            train_data_all.append(td[1])
            
            if self.with_dec:
                y_data_all.append(td[2][-(train_data.pred_len + train_data.label_len):])
            else:
                y_data_all.append(td[2][-train_data.pred_len:])
            
        self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float()
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)
        
        self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)

        self.n_train = self.train_data_all.shape[0]
        self.train_indices = torch.tensor(train_indices).long()
        self.train_abs_indices = torch.tensor(train_abs_indices).long()

    def phase_scores(self, query_abs_index, candidate_index):
        if self.phase_period is None or self.phase_period <= 0:
            raise ValueError('phase_period must be positive for phase-aware retrieval')

        train_abs_indices = self.train_abs_indices.to(candidate_index.device)
        candidate_abs_index = train_abs_indices[candidate_index]

        query_phase = (query_abs_index + self.seq_len - 1) % self.phase_period
        candidate_phase = (candidate_abs_index + self.seq_len - 1) % self.phase_period

        phase_dist = (candidate_phase - query_phase.unsqueeze(1)).abs().float()
        phase_dist = torch.minimum(phase_dist, self.phase_period - phase_dist)
        return torch.exp(-phase_dist / self.phase_tau)

    def select_candidates(self, sim, query_abs_index):
        bsz = query_abs_index.shape[0]
        flat_sim = sim.reshape(self.n_period * bsz, self.n_train)
        shape_top_k = min(self.topm, self.n_train)
        shape_score, shape_index = torch.topk(flat_sim, shape_top_k, dim=1)

        if self.retrieval_variant == 'A':
            selected_index = shape_index
            selected_score = shape_score
        else:
            phase_top_m = min(self.phase_top_m, shape_top_k)
            query_abs_index = query_abs_index.unsqueeze(0).repeat(self.n_period, 1).reshape(-1)
            score_phase = self.phase_scores(query_abs_index, shape_index)

            if self.retrieval_variant == 'B':
                selected_pos = torch.topk(score_phase, phase_top_m, dim=1).indices
                selected_score = torch.gather(shape_score, 1, selected_pos)
            elif self.retrieval_variant == 'C':
                score_final = shape_score + self.phase_lambda * score_phase
                selected_pos = torch.topk(score_final, phase_top_m, dim=1).indices
                selected_score = torch.gather(score_final, 1, selected_pos)
            else:
                raise ValueError(f'Unknown retrieval_variant: {self.retrieval_variant}')

            selected_index = torch.gather(shape_index, 1, selected_pos)

        ranking_sim = torch.ones_like(flat_sim) * float('-inf')
        rows = torch.arange(flat_sim.size(0)).unsqueeze(-1).to(flat_sim.device)
        ranking_sim[rows, selected_index] = selected_score
        return ranking_sim.reshape(self.n_period, bsz, self.n_train)

    def decompose_mg(self, data_all, remove_offset=True):
        data_all = copy.deepcopy(data_all) # T, S, C

        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)
            
            mg.append(cur)
#             data_all = data_all - cur
            
        mg = torch.stack(mg, dim=0) # G, T, S, C

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p[:,-1:,:]
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
        else:
            offset = None
            
        offset = torch.stack(offset, dim=0)
            
        return mg, offset
    
    def periodic_batch_corr(self, data_all, key, in_bsz = 512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape
        
        bx = key - torch.mean(key, dim=2, keepdim=True)
        
        iters = math.ceil(train_len / in_bsz)
        
        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)
            
            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)
            
            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)
            
        sim = torch.cat(sim, dim=2)
        
        return sim
        
    def retrieve(self, x, index, train=True, phase_index=None):
        index = index.to(x.device)
        if phase_index is None:
            phase_index = index
        phase_index = phase_index.to(x.device).long()
        
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels
        
        x_mg, mg_offset = self.decompose_mg(x) # G, B, S, C

        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2), # G, T, S * C
            x_mg.flatten(start_dim=2), # G, B, S * C
        ) # G, B, T
            
        if train:
            raw_sim = sim
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)
            
            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)
            
            sim = sim.masked_fill(self_mask.bool(), float('-inf')) # G, B, T
            all_masked = torch.isinf(sim).all(dim=2, keepdim=True)
            sim = torch.where(all_masked, raw_sim, sim)

        ranking_sim = self.select_candidates(sim, phase_index) # G, B, T

        data_len, seq_len, channels = self.train_data_all.shape
            
        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)
        ranking_prob = ranking_prob.detach().cpu() # G, B, T
        
        y_data_all = self.y_data_all_mg.flatten(start_dim=2) # G, T, P * C
        
        pred_from_retrieval = torch.bmm(ranking_prob, y_data_all).reshape(self.n_period, bsz, -1, channels)
        pred_from_retrieval = pred_from_retrieval.to(x.device)
        
        return pred_from_retrieval
    
    def retrieve_all(self, data, train=False, device=torch.device('cpu')):
        assert self.train_data_all_mg is not None
        
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
                phase_index = index + getattr(data, 'border1', 0)
                pred_from_retrieval = self.retrieve(
                    batch_x.float().to(device),
                    index,
                    train=train,
                    phase_index=phase_index,
                )
                pred_from_retrieval = pred_from_retrieval.cpu()
                retrievals.append(pred_from_retrieval)
                
        retrievals = torch.cat(retrievals, dim=1)
        
        return retrievals
