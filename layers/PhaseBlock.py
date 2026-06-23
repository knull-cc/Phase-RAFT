import numpy as np
import torch
import torch.nn.functional as F


class PhaseBlockResidualMemory:
    """Phase-conditioned residual corrector memory.

    ``full`` mode builds a global PhaseBlock bank: every memory sample
    contributes one key/value pair for each forecast step. The key is a
    phase-aligned local time block built only from the observed lookback window;
    the value is the base-model residual at that forecast step.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        periods=None,
        max_periods=2,
        min_period_strength=0.05,
        patch_width=3,
        num_cycles=4,
        topk=5,
        temperature=0.1,
        alpha=0.2,
        sim_threshold=0.3,
        residual_var_scale=1.0,
        stride=1,
        bank_mode='full',
        query_chunk=256,
        eps=1e-6,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.user_periods = self._parse_periods(periods)
        self.periods = []
        self.max_periods = max_periods
        self.min_period_strength = min_period_strength
        self.patch_width = patch_width if patch_width % 2 == 1 else patch_width + 1
        self.num_cycles = num_cycles
        self.topk = topk
        self.temperature = temperature
        self.alpha = alpha
        self.sim_threshold = sim_threshold
        self.residual_var_scale = residual_var_scale
        self.stride = stride
        self.bank_mode = bank_mode
        self.query_chunk = query_chunk
        self.eps = eps

        self.keys = None
        self.values = None
        self._key_chunks = []
        self._value_chunks = []
        self._phase_key_chunks = {}
        self._phase_value_chunks = {}
        self.phase_banks = {}
        self.enabled = False
        self.memory_ready = False

    @staticmethod
    def _parse_periods(periods):
        if periods is None or periods == '':
            return None
        if isinstance(periods, str):
            return [int(item.strip()) for item in periods.split(',') if item.strip()]
        return [int(period) for period in periods]

    def estimate_periods(self, data):
        if self.user_periods is not None:
            periods = self.user_periods
        else:
            periods = self._estimate_periods_fft(data)

        periods = [
            int(period)
            for period in periods
            if int(period) > self.patch_width and int(period) <= self.seq_len
        ]
        periods = periods[: self.max_periods]
        self.periods = periods
        self.enabled = len(self.periods) > 0
        if self.enabled:
            print(f'PhaseBlock periods: {self.periods}')
        else:
            print('PhaseBlock disabled: no reliable period no longer than seq_len')
        return self.periods

    def _estimate_periods_fft(self, data):
        values = np.asarray(data, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if len(values) < 4:
            return []

        values = values - np.nanmean(values, axis=0, keepdims=True)
        spectrum = np.fft.rfft(values, axis=0)
        power = np.mean(np.abs(spectrum) ** 2, axis=1)
        if len(power) <= 2:
            return []

        power[0] = 0.0
        total = float(np.sum(power) + self.eps)
        ranked = np.argsort(power)[::-1]

        periods = []
        strengths = []
        for freq_idx in ranked:
            if freq_idx <= 0:
                continue
            strength = float(power[freq_idx] / total)
            if strength < self.min_period_strength:
                break
            period = int(round(len(values) / freq_idx))
            if period <= self.patch_width or period > self.seq_len:
                continue
            if any(abs(period - seen) <= 1 for seen in periods):
                continue
            periods.append(period)
            strengths.append(strength)
            if len(periods) >= self.max_periods:
                break

        if periods:
            msg = ', '.join([f'{p}(strength={s:.3f})' for p, s in zip(periods, strengths)])
            print(f'PhaseBlock FFT periods: {msg}')
        return periods

    def reset_memory(self):
        self.keys = None
        self.values = None
        self._key_chunks = []
        self._value_chunks = []
        self._phase_key_chunks = {}
        self._phase_value_chunks = {}
        self.phase_banks = {}
        self.memory_ready = False

    def add_batch(self, x, residual, index_abs=None):
        if not self.enabled:
            return
        if self.stride > 1:
            x = x[:: self.stride]
            residual = residual[:: self.stride]
            if index_abs is not None:
                index_abs = index_abs[:: self.stride]
        if x.numel() == 0:
            return

        if self.bank_mode == 'single':
            keys = self._make_single_keys(x.detach())
            values = residual.detach().cpu().float()
            self._key_chunks.append(keys.cpu())
            self._value_chunks.append(values)
            return

        keys, phases = self._make_full_keys(x.detach(), index_abs=index_abs)
        flat_keys = keys.reshape(-1, keys.shape[-1]).cpu()
        flat_values = residual.detach().reshape(-1, self.channels).cpu().float()
        flat_phases = phases.reshape(-1).cpu()

        for phase in torch.unique(flat_phases):
            phase_id = int(phase.item())
            mask = flat_phases == phase_id
            self._phase_key_chunks.setdefault(phase_id, []).append(flat_keys[mask])
            self._phase_value_chunks.setdefault(phase_id, []).append(flat_values[mask])

    def finalize_memory(self):
        if not self.enabled:
            self.memory_ready = False
            print('PhaseBlock memory: disabled')
            return

        if self.bank_mode == 'single':
            self._finalize_single_memory()
            return

        if not self._phase_key_chunks:
            self.memory_ready = False
            print('PhaseBlock memory: empty')
            return

        total = 0
        self.phase_banks = {}
        for phase_id in sorted(self._phase_key_chunks):
            keys = torch.cat(self._phase_key_chunks[phase_id], dim=0).float()
            values = torch.cat(self._phase_value_chunks[phase_id], dim=0).float()
            self.phase_banks[phase_id] = (keys, values)
            total += keys.shape[0]

        self._phase_key_chunks = {}
        self._phase_value_chunks = {}
        self.memory_ready = total > 0
        print(f'PhaseBlock full bank size: {total} entries across {len(self.phase_banks)} phase buckets')

    def _finalize_single_memory(self):
        if not self._key_chunks:
            self.memory_ready = False
            print('PhaseBlock memory: empty')
            return
        self.keys = torch.cat(self._key_chunks, dim=0).float()
        self.values = torch.cat(self._value_chunks, dim=0).float()
        self._key_chunks = []
        self._value_chunks = []
        self.memory_ready = True
        print(f'PhaseBlock single bank size: {self.keys.shape[0]}')

    def correct(self, x, pred, index_abs=None):
        if not self.enabled or not self.memory_ready:
            return pred
        if self.bank_mode == 'single':
            return self._correct_single(x, pred)
        return self._correct_full(x, pred, index_abs=index_abs)

    def _correct_single(self, x, pred):
        keys = self.keys.to(x.device)
        values = self.values.to(x.device)
        query = self._make_single_keys(x)
        delta = self._retrieve_delta(query, keys, values)
        return pred + delta

    def _correct_full(self, x, pred, index_abs=None):
        query, phases = self._make_full_keys(x, index_abs=index_abs)
        bsz, horizon, dim = query.shape
        flat_query = query.reshape(bsz * horizon, dim)
        flat_phases = phases.reshape(-1)
        flat_delta = torch.zeros(
            bsz * horizon,
            self.channels,
            device=x.device,
            dtype=pred.dtype,
        )

        for phase in torch.unique(flat_phases):
            phase_id = int(phase.item())
            if phase_id not in self.phase_banks:
                continue
            row_idx = torch.nonzero(flat_phases == phase_id, as_tuple=False).squeeze(1)
            bank_keys, bank_values = self.phase_banks[phase_id]
            flat_delta[row_idx] = self._retrieve_delta(
                flat_query[row_idx],
                bank_keys.to(x.device),
                bank_values.to(x.device),
            ).to(pred.dtype)

        return pred + flat_delta.reshape(bsz, horizon, self.channels)

    def _retrieve_delta(self, query, keys, values):
        k = min(self.topk, keys.shape[0])
        deltas = []
        for start in range(0, query.shape[0], self.query_chunk):
            q = query[start:start + self.query_chunk]
            sim = torch.matmul(q, keys.transpose(0, 1))
            top_sim, top_idx = torch.topk(sim, k, dim=1)
            weights = F.softmax(top_sim / max(self.temperature, self.eps), dim=1)

            top_values = values[top_idx]
            if top_values.dim() == 3:
                residual = torch.sum(weights[:, :, None] * top_values, dim=1)
                residual_mean = residual[:, None, :]
                reduce_dims = (1, 2)
                gate_shape = (-1, 1)
            else:
                residual = torch.sum(weights[:, :, None, None] * top_values, dim=1)
                residual_mean = residual[:, None, :, :]
                reduce_dims = (1, 2, 3)
                gate_shape = (-1, 1, 1)

            best_sim = top_sim[:, 0]
            sim_gate = torch.clamp(
                (best_sim - self.sim_threshold) / max(1.0 - self.sim_threshold, self.eps),
                min=0.0,
                max=1.0,
            )

            residual_var = torch.mean((top_values - residual_mean) ** 2, dim=reduce_dims)
            residual_energy = torch.mean(top_values ** 2, dim=reduce_dims)
            agreement_gate = 1.0 / (
                1.0 + self.residual_var_scale * residual_var / (residual_energy + self.eps)
            )

            gate = (self.alpha * sim_gate * agreement_gate).view(*gate_shape)
            deltas.append(gate * residual)

        return torch.cat(deltas, dim=0)

    def _make_single_keys(self, x):
        full_keys, _ = self._make_full_keys(x, index_abs=None)
        return full_keys[:, 0, :]

    def _make_full_keys(self, x, index_abs=None):
        # x: B, seq_len, C. Keys use only observed lookback values.
        x = x.float()
        x = x - x[:, -1:, :]
        bsz, length, channels = x.shape
        if length != self.seq_len or channels != self.channels:
            raise ValueError('PhaseBlock input shape does not match configured seq_len/channels')

        if index_abs is None:
            index_abs = torch.zeros(bsz, device=x.device, dtype=torch.long)
        else:
            index_abs = index_abs.to(x.device).long()

        primary_period = self.periods[0]
        offsets = torch.arange(self.pred_len, device=x.device).long()
        phases = (index_abs[:, None] + self.seq_len + offsets[None, :]) % primary_period

        keys = []
        for h in range(self.pred_len):
            blocks = []
            for period in self.periods:
                first_back = (h + period) // period
                for cycle in range(self.num_cycles):
                    anchor = self.seq_len + h - (first_back + cycle) * period
                    patch = self._extract_patch(x, anchor)
                    blocks.append(patch.reshape(bsz, -1))
            keys.append(torch.cat(blocks, dim=1))

        keys = torch.stack(keys, dim=1)
        keys = keys - keys.mean(dim=2, keepdim=True)
        keys = F.normalize(keys, dim=2, eps=self.eps)
        return keys, phases

    def _extract_patch(self, x, anchor):
        bsz, _, channels = x.shape
        if anchor < 0 or anchor >= self.seq_len:
            return torch.zeros(
                bsz,
                self.patch_width,
                channels,
                device=x.device,
                dtype=x.dtype,
            )

        half = self.patch_width // 2
        positions = torch.arange(
            anchor - half,
            anchor + half + 1,
            device=x.device,
        )
        positions = positions.clamp(0, self.seq_len - 1)
        return x.index_select(1, positions)
