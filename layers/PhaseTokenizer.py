import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseTokenizer(nn.Module):
    def __init__(self, period_len):
        super().__init__()
        if period_len <= 0:
            raise ValueError('period_len must be positive')
        self.period_len = period_len

    @staticmethod
    def offset_normalize(x):
        offset = x[:, -1:, :].detach()
        return x - offset, offset

    def n_periods(self, length):
        return math.ceil(length / self.period_len)

    def to_phase(self, x):
        bsz, length, channels = x.shape
        n_periods = self.n_periods(length)
        target_len = n_periods * self.period_len
        pad_len = target_len - length

        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        x = x.reshape(bsz, n_periods, self.period_len, channels)
        return x.permute(0, 3, 2, 1).contiguous()

    @staticmethod
    def to_time(tokens, target_len):
        bsz, channels, period_len, n_periods = tokens.shape
        x = tokens.permute(0, 3, 2, 1).contiguous()
        x = x.reshape(bsz, n_periods * period_len, channels)
        return x[:, :target_len, :]
