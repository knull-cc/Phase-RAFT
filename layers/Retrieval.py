import torch
import torch.nn.functional as F


class PhaseAlignedIdeaBlockRetrieval:
    """Key-Value retrieval over Phase-aligned IdeaBlocks.

    Each memory key is built from the observed lookback window by selecting
    values whose absolute phases fall in ``[p-r, p+r]`` around a center phase
    ``p``. For forecasting, ``p`` is the last observed phase, and values are
    the true future sequence after that input window.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        period_len=24,
        phase_radius=1,
        num_cycles=4,
        topk=20,
        temperature=0.1,
        eps=1e-6,
    ):
        if period_len <= 0:
            raise ValueError('period_len must be positive')
        if phase_radius < 0:
            raise ValueError('phase_radius must be non-negative')
        if num_cycles <= 0:
            raise ValueError('num_cycles must be positive')
        if topk <= 0:
            raise ValueError('topk must be positive')
        if temperature <= 0:
            raise ValueError('temperature must be positive')

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.period_len = period_len
        self.phase_radius = phase_radius
        self.num_cycles = num_cycles
        self.topk = topk
        self.temperature = temperature
        self.eps = eps

        self.keys = None
        self.values = None
        self.train_abs_indices = None
        self.n_train = 0

    @property
    def block_width(self):
        return 2 * self.phase_radius + 1

    def prepare_dataset(self, train_data):
        keys = []
        values = []
        abs_indices = []
        base_index = int(getattr(train_data, 'border1', 0))

        for i in range(len(train_data)):
            index, seq_x, seq_y, _, _ = train_data[i]
            index_abs = int(index) + base_index
            x = torch.as_tensor(seq_x, dtype=torch.float32).unsqueeze(0)
            key = self.make_keys(
                x,
                torch.tensor([index_abs], dtype=torch.long),
            ).squeeze(0)
            keys.append(key.cpu())
            future = torch.as_tensor(seq_y[-self.pred_len:], dtype=torch.float32)
            last_observed = torch.as_tensor(seq_x[-1:], dtype=torch.float32)
            values.append(future - last_observed)
            abs_indices.append(index_abs)

        if not keys:
            raise ValueError('cannot build IdeaBlock memory from an empty training set')

        self.keys = torch.stack(keys, dim=0).float()
        self.values = torch.stack(values, dim=0).float()
        self.train_abs_indices = torch.tensor(abs_indices, dtype=torch.long)
        self.n_train = self.keys.shape[0]
        print(
            'Phase-aligned IdeaBlock memory: '
            f'{self.n_train} keys, P={self.period_len}, '
            f'r={self.phase_radius}, cycles={self.num_cycles}'
        )

    def make_keys(self, x, index_abs):
        """Build normalized Phase-aligned IdeaBlock keys.

        ``index_abs`` is the absolute start index of each lookback window.
        The center phase is the last observed timestamp:
        ``(index_abs + seq_len - 1) % period_len``.
        """

        x = x.float()
        bsz, length, channels = x.shape
        if length != self.seq_len or channels != self.channels:
            raise ValueError('IdeaBlock input shape does not match configured seq_len/channels')

        index_abs = index_abs.to(x.device).long()
        x_norm = x - x[:, -1:, :]

        deltas = torch.arange(
            -self.phase_radius,
            self.phase_radius + 1,
            device=x.device,
            dtype=torch.long,
        )
        cycles = torch.arange(self.num_cycles, device=x.device, dtype=torch.long)

        center_phase = (index_abs + self.seq_len - 1) % self.period_len
        end_phase = (index_abs + self.seq_len - 1) % self.period_len
        target_phase = (center_phase[:, None] + deltas[None, :]) % self.period_len
        steps_back = (end_phase[:, None] - target_phase) % self.period_len

        rel_pos = (
            self.seq_len
            - 1
            - steps_back[:, None, :]
            - cycles[None, :, None] * self.period_len
        )
        valid = (rel_pos >= 0) & (rel_pos < self.seq_len)

        flat_pos = rel_pos.reshape(bsz, -1).clamp(0, self.seq_len - 1)
        gather_index = flat_pos[:, :, None].expand(-1, -1, channels)
        block = torch.gather(x_norm, dim=1, index=gather_index)
        block = block * valid.reshape(bsz, -1, 1).float()

        key = block.reshape(bsz, -1)
        key = key - key.mean(dim=1, keepdim=True)
        return F.normalize(key, dim=1, eps=self.eps)

    def retrieve(self, x, index_abs, train=False):
        if self.keys is None or self.values is None:
            raise RuntimeError('IdeaBlock memory has not been prepared')

        query = self.make_keys(x, index_abs)
        keys = self.keys.to(query.device)
        sim = torch.matmul(query, keys.transpose(0, 1))

        if train:
            sim = self._mask_self_overlap(sim, index_abs)

        k = min(self.topk, self.n_train)
        top_sim, top_idx = torch.topk(sim, k, dim=1)
        weights = F.softmax(top_sim / self.temperature, dim=1)

        values = self.values.to(device=x.device, dtype=x.dtype)
        retrieved_trend = torch.zeros(
            x.shape[0],
            self.pred_len,
            self.channels,
            device=x.device,
            dtype=x.dtype,
        )
        for rank in range(k):
            retrieved_trend = retrieved_trend + weights[:, rank, None, None] * values[top_idx[:, rank]]
        return retrieved_trend

    def _mask_self_overlap(self, sim, query_abs_index):
        train_abs = self.train_abs_indices.to(sim.device)
        query_abs_index = query_abs_index.to(sim.device).long()
        gap = self.seq_len + self.pred_len
        overlap = (train_abs[None, :] - query_abs_index[:, None]).abs() < gap
        masked = sim.masked_fill(overlap, float('-inf'))
        all_masked = torch.isinf(masked).all(dim=1, keepdim=True)
        return torch.where(all_masked, sim, masked)

    def describe_block(self):
        phases = list(range(-self.phase_radius, self.phase_radius + 1))
        return {
            'period_len': self.period_len,
            'phase_radius': self.phase_radius,
            'num_cycles': self.num_cycles,
            'relative_phases': phases,
            'key_slots': self.num_cycles * self.block_width,
        }


# Compatibility name used by older imports in this repository.
RetrievalTool = PhaseAlignedIdeaBlockRetrieval
