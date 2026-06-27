import torch
import torch.nn.functional as F


class PhaseAlignedIdeaBlockRetrieval:
    """Key-Value retrieval over Phase-aligned IdeaBlocks."""

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
        self._clear_device_cache()

    @property
    def block_width(self):
        return 2 * self.phase_radius + 1

    def prepare_dataset(self, train_data):
        self._clear_device_cache()
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
        self._clear_device_cache()
        print(
            'Phase-aligned IdeaBlock memory: '
            f'{self.n_train} keys, P={self.period_len}, '
            f'r={self.phase_radius}, cycles={self.num_cycles}'
        )

    def make_keys(self, x, index_abs):
        """Build normalized Phase-aligned IdeaBlock keys around the last observed phase."""

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
        target_phase = (center_phase[:, None] + deltas[None, :]) % self.period_len
        steps_back = (center_phase[:, None] - target_phase) % self.period_len

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

        block = block.reshape(bsz, -1)
        mask = valid.float().reshape(bsz, -1)
        key = torch.cat([block, mask], dim=1)
        key = key - key.mean(dim=1, keepdim=True)
        return F.normalize(key, dim=1, eps=self.eps)

    def _clear_device_cache(self):
        self._keys_cache = None
        self._keys_cache_device = None
        self._values_cache = None
        self._values_cache_device = None
        self._values_cache_dtype = None
        self._train_abs_cache = None
        self._train_abs_cache_device = None

    def _keys_on(self, device):
        if self._keys_cache is None or self._keys_cache_device != device:
            self._keys_cache = self.keys.to(device)
            self._keys_cache_device = device
        return self._keys_cache

    def _values_on(self, device, dtype):
        if (
            self._values_cache is None
            or self._values_cache_device != device
            or self._values_cache_dtype != dtype
        ):
            self._values_cache = self.values.to(device=device, dtype=dtype)
            self._values_cache_device = device
            self._values_cache_dtype = dtype
        return self._values_cache

    def _train_abs_on(self, device):
        if self._train_abs_cache is None or self._train_abs_cache_device != device:
            self._train_abs_cache = self.train_abs_indices.to(device)
            self._train_abs_cache_device = device
        return self._train_abs_cache

    def retrieve(self, x, index_abs, train=False, return_info=False):
        if self.keys is None or self.values is None:
            raise RuntimeError('IdeaBlock memory has not been prepared')

        query = self.make_keys(x, index_abs)
        keys = self._keys_on(query.device)
        sim = torch.matmul(query, keys.transpose(0, 1))

        if train:
            sim = self._mask_self_overlap(sim, index_abs)

        k = min(self.topk, self.n_train)
        top_sim, top_idx = torch.topk(sim, k, dim=1)
        weights = F.softmax(top_sim / self.temperature, dim=1)

        values = self._values_on(x.device, x.dtype)
        retrieved = (weights[:, :, None, None] * values[top_idx]).sum(dim=1)
        if not return_info:
            return retrieved

        log_k = torch.log(torch.tensor(float(k), device=x.device, dtype=x.dtype)).clamp_min(self.eps)
        entropy = -(weights * torch.log(weights.clamp_min(self.eps))).sum(dim=1, keepdim=True) / log_k
        top1 = top_sim[:, :1]
        mean_sim = top_sim.mean(dim=1, keepdim=True)
        info = {
            'top1_sim': top1.to(dtype=x.dtype),
            'mean_sim': mean_sim.to(dtype=x.dtype),
            'sim_gap': (top1 - mean_sim).to(dtype=x.dtype),
            'weight_entropy': entropy.to(dtype=x.dtype),
            'weight_max': weights.max(dim=1, keepdim=True).values.to(dtype=x.dtype),
        }
        return retrieved, info

    def _mask_self_overlap(self, sim, query_abs_index):
        train_abs = self._train_abs_on(sim.device)
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


RetrievalTool = PhaseAlignedIdeaBlockRetrieval
