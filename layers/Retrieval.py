import torch
import torch.nn.functional as F


class PhaseAlignedIdeaBlockRetrieval:
    """Key-Value retrieval over Phase-aligned IdeaBlocks.

    Each memory key is built from the observed lookback window by selecting
    values whose absolute phases fall in ``[p-r, p+r]`` around a center phase
    ``p``. Values are stored as future residuals against either the last
    observed point or the latest observed point with the same phase as each
    future horizon.
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
        horizon_wise_phase=False,
        value_anchor='phase',
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
        if value_anchor not in ['phase', 'last']:
            raise ValueError("value_anchor must be 'phase' or 'last'")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.period_len = period_len
        self.phase_radius = phase_radius
        self.num_cycles = num_cycles
        self.topk = topk
        self.temperature = temperature
        self.horizon_wise_phase = horizon_wise_phase
        self.value_anchor = value_anchor
        self.eps = eps

        self.keys = None
        self.phase_keys = None
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
        phase_key_chunks = [[] for _ in range(self.period_len)] if self.horizon_wise_phase else None
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
                horizon_offset=None,
            ).squeeze(0)
            keys.append(key.cpu())
            if self.horizon_wise_phase:
                for phase in range(self.period_len):
                    horizon_offset = (phase - (index_abs + self.seq_len)) % self.period_len
                    phase_key = self.make_keys(
                        x,
                        torch.tensor([index_abs], dtype=torch.long),
                        horizon_offset=torch.tensor([horizon_offset], dtype=torch.long),
                    ).squeeze(0)
                    phase_key_chunks[phase].append(phase_key.cpu())
            future = torch.as_tensor(seq_y[-self.pred_len:], dtype=torch.float32)
            anchors = self.make_future_anchors(
                x,
                torch.tensor([index_abs], dtype=torch.long),
            ).squeeze(0)
            values.append(future - anchors)
            abs_indices.append(index_abs)

        if not keys:
            raise ValueError('cannot build IdeaBlock memory from an empty training set')

        self.keys = torch.stack(keys, dim=0).float()
        if self.horizon_wise_phase:
            self.phase_keys = [
                torch.stack(phase_keys, dim=0).float()
                for phase_keys in phase_key_chunks
            ]
        self.values = torch.stack(values, dim=0).float()
        self.train_abs_indices = torch.tensor(abs_indices, dtype=torch.long)
        self.n_train = self.keys.shape[0]
        self._clear_device_cache()
        mode = 'horizon-wise' if self.horizon_wise_phase else 'sequence-wise'
        print(
            'Phase-aligned IdeaBlock memory: '
            f'{self.n_train} keys, P={self.period_len}, '
            f'r={self.phase_radius}, cycles={self.num_cycles}, '
            f'mode={mode}, value_anchor={self.value_anchor}'
        )

    def make_keys(self, x, index_abs, horizon_offset=None):
        """Build normalized Phase-aligned IdeaBlock keys.

        ``index_abs`` is the absolute start index of each lookback window.
        By default, the center phase is the last observed timestamp:
        ``(index_abs + seq_len - 1) % period_len``. When ``horizon_offset`` is
        given, the center phase is the corresponding future timestamp:
        ``(index_abs + seq_len + horizon_offset) % period_len``.
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

        if horizon_offset is None:
            center_phase = (index_abs + self.seq_len - 1) % self.period_len
        else:
            horizon_offset = horizon_offset.to(x.device).long()
            center_phase = (index_abs + self.seq_len + horizon_offset) % self.period_len
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

        block = block.reshape(bsz, -1)
        mask = valid.float().reshape(bsz, -1)
        key = torch.cat([block, mask], dim=1)
        key = key - key.mean(dim=1, keepdim=True)
        return F.normalize(key, dim=1, eps=self.eps)

    def make_future_anchors(self, x, index_abs):
        """Return the observed anchor used to express each future horizon."""

        x = x.float()
        bsz, length, channels = x.shape
        if length != self.seq_len or channels != self.channels:
            raise ValueError('IdeaBlock input shape does not match configured seq_len/channels')

        if self.value_anchor == 'last':
            return x[:, -1:, :].expand(-1, self.pred_len, -1)

        index_abs = index_abs.to(x.device).long()
        horizons = torch.arange(self.pred_len, device=x.device, dtype=torch.long)
        future_phase = (index_abs[:, None] + self.seq_len + horizons[None, :]) % self.period_len
        end_phase = (index_abs + self.seq_len - 1) % self.period_len
        steps_back = (end_phase[:, None] - future_phase) % self.period_len
        rel_pos = self.seq_len - 1 - steps_back
        valid = rel_pos >= 0

        gather_pos = rel_pos.clamp(0, self.seq_len - 1)
        gather_index = gather_pos[:, :, None].expand(-1, -1, channels)
        anchors = torch.gather(x, dim=1, index=gather_index)
        last_observed = x[:, -1:, :].expand(-1, self.pred_len, -1)
        return torch.where(valid[:, :, None], anchors, last_observed)

    def _clear_device_cache(self):
        self._keys_cache = None
        self._keys_cache_device = None
        self._phase_keys_cache = None
        self._phase_keys_cache_device = None
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

    def _phase_keys_on(self, device):
        if self.phase_keys is None:
            raise RuntimeError('horizon-wise phase memory has not been prepared')
        if self._phase_keys_cache is None or self._phase_keys_cache_device != device:
            self._phase_keys_cache = [keys.to(device) for keys in self.phase_keys]
            self._phase_keys_cache_device = device
        return self._phase_keys_cache

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

    def set_values(self, values):
        values = values.detach().cpu().float()
        expected = (self.n_train, self.pred_len, self.channels)
        if tuple(values.shape) != expected:
            raise ValueError(f'IdeaBlock values must have shape {expected}, got {tuple(values.shape)}')
        self.values = values
        self._values_cache = None
        self._values_cache_device = None
        self._values_cache_dtype = None

    def retrieve_values(self, x, index_abs, train=False):
        if self.keys is None or self.values is None:
            raise RuntimeError('IdeaBlock memory has not been prepared')

        if self.horizon_wise_phase:
            return self._retrieve_horizon_wise_values(x, index_abs, train=train)

        query = self.make_keys(x, index_abs, horizon_offset=None)
        keys = self._keys_on(query.device)
        sim = torch.matmul(query, keys.transpose(0, 1))

        if train:
            sim = self._mask_self_overlap(sim, index_abs)

        k = min(self.topk, self.n_train)
        top_sim, top_idx = torch.topk(sim, k, dim=1)
        weights = F.softmax(top_sim / self.temperature, dim=1)

        values = self._values_on(x.device, x.dtype)
        return (weights[:, :, None, None] * values[top_idx]).sum(dim=1)

    def retrieve(self, x, index_abs, train=False):
        residual = self.retrieve_values(x, index_abs, train=train)
        return self.make_future_anchors(x, index_abs).to(dtype=x.dtype) + residual

    def _retrieve_horizon_wise_values(self, x, index_abs, train=False):
        if self.phase_keys is None:
            raise RuntimeError('horizon-wise phase memory has not been prepared')

        bsz = x.shape[0]
        phase_keys = self._phase_keys_on(x.device)
        values = self._values_on(x.device, x.dtype)
        index_abs_device = index_abs.to(x.device).long()
        horizons = torch.arange(self.pred_len, device=x.device, dtype=torch.long)
        offset_count = min(self.period_len, self.pred_len)
        offsets = torch.arange(offset_count, device=x.device, dtype=torch.long)
        k = min(self.topk, self.n_train)

        # Query IdeaBlocks only depend on future phase, so h and h + P reuse the same key.
        query_chunks = []
        for offset in range(offset_count):
            horizon_offset = torch.full(
                (bsz,),
                offset,
                device=x.device,
                dtype=torch.long,
            )
            query_chunks.append(self.make_keys(x, index_abs, horizon_offset=horizon_offset))

        query = torch.stack(query_chunks, dim=1)
        query_flat = query.reshape(bsz * offset_count, -1)
        query_phase = (index_abs_device[:, None] + self.seq_len + offsets[None, :]) % self.period_len
        query_phase_flat = query_phase.reshape(-1)
        query_abs_flat = index_abs_device[:, None].expand(-1, offset_count).reshape(-1)
        pair_top_idx = torch.empty(
            bsz * offset_count,
            k,
            device=x.device,
            dtype=torch.long,
        )
        pair_weights = torch.empty(
            bsz * offset_count,
            k,
            device=x.device,
            dtype=query.dtype,
        )

        for phase in torch.unique(query_phase_flat):
            phase_id = int(phase.item())
            flat_idx = torch.nonzero(query_phase_flat == phase, as_tuple=False).squeeze(1)
            keys = phase_keys[phase_id]
            sim = torch.matmul(query_flat[flat_idx], keys.transpose(0, 1))
            if train:
                sim = self._mask_self_overlap(sim, query_abs_flat[flat_idx])

            top_sim, top_idx = torch.topk(sim, k, dim=1)
            weights = F.softmax(top_sim / self.temperature, dim=1)
            pair_top_idx[flat_idx] = top_idx
            pair_weights[flat_idx] = weights

        horizon_offsets = horizons % self.period_len
        pair_ids = (
            torch.arange(bsz, device=x.device, dtype=torch.long)[:, None] * offset_count
            + horizon_offsets[None, :]
        )
        retrieved = torch.empty(
            bsz,
            self.pred_len,
            self.channels,
            device=x.device,
            dtype=x.dtype,
        )
        max_gather_values = 8_000_000
        horizon_chunk = max(1, max_gather_values // (bsz * k * self.channels))
        # Chunk the final value gather so high-channel datasets do not allocate B*H*K*C at once.
        for start in range(0, self.pred_len, horizon_chunk):
            end = min(start + horizon_chunk, self.pred_len)
            chunk_pair_ids = pair_ids[:, start:end]
            chunk_top_idx = pair_top_idx[chunk_pair_ids]
            chunk_weights = pair_weights[chunk_pair_ids]
            chunk_horizons = horizons[start:end][None, :, None].expand(bsz, end - start, k)
            gathered_values = values[chunk_top_idx, chunk_horizons, :]
            retrieved[:, start:end, :] = (
                chunk_weights[:, :, :, None] * gathered_values
            ).sum(dim=2)

        return retrieved

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
            'value_anchor': self.value_anchor,
            'relative_phases': phases,
            'key_slots': self.num_cycles * self.block_width,
        }


# Compatibility name used by older imports in this repository.
RetrievalTool = PhaseAlignedIdeaBlockRetrieval
