import math
import os
from typing import Dict, Optional

import torch
from torch import nn

try:
    from PIL import Image
except Exception:
    Image = None


class GaussianSoftScatter(nn.Module):
    """Gaussian soft scatter from sparse pillars to dense BEV."""

    def __init__(
            self,
            in_channels,
            output_shape,
            sigma=1.0,
            radius=1,
            normalize=True,
            min_weight=0.0,
            debug: Optional[Dict] = None):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels
        self.fp16_enabled = False

        self.sigma = float(sigma)
        self.radius = int(radius)
        self.normalize = bool(normalize)
        self.min_weight = float(min_weight)
        self.eps = 1e-6

        offsets, kernel_weights = self._build_kernel()
        self.offsets = offsets
        self.register_buffer('kernel_weights', kernel_weights, persistent=False)

        debug = {} if debug is None else debug
        self.debug_enabled = bool(debug.get('enabled', False))
        self.debug_log_every_n_steps = max(1, int(debug.get('log_every_n_steps', 200)))
        self.debug_save_visualization = bool(debug.get('save_visualization', False))
        self.debug_max_visualizations = int(debug.get('max_visualizations', 20))
        self.debug_out_dir = str(debug.get('out_dir', 'outputs/soft_scatter_debug'))
        self.debug_print_to_stdout = bool(debug.get('print_to_stdout', False))

        self._forward_calls = 0
        self._saved_visualizations = 0
        self._last_debug_stats = None

    def _build_kernel(self):
        offsets = []
        weights = []
        sigma_sq = max(self.sigma * self.sigma, self.eps)
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                weight = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma_sq))
                if weight < self.min_weight:
                    continue
                offsets.append((dy, dx))
                weights.append(weight)

        if not offsets:
            offsets = [(0, 0)]
            weights = [1.0]
        return offsets, torch.tensor(weights, dtype=torch.float32)

    def pop_debug_stats(self):
        stats = self._last_debug_stats
        self._last_debug_stats = None
        return stats

    def _save_map(self, map_tensor, fname):
        if Image is None:
            return
        map_tensor = map_tensor.detach().float().cpu()
        map_tensor = map_tensor - map_tensor.min()
        map_tensor = map_tensor / map_tensor.max().clamp(min=self.eps)
        map_uint8 = (map_tensor * 255.0).to(torch.uint8).numpy()
        Image.fromarray(map_uint8, mode='L').save(fname)

    def _collect_debug(self, bev_features, weight_accum, batch_size):
        if not self.debug_enabled:
            return
        if self._forward_calls % self.debug_log_every_n_steps != 0 and self._forward_calls != 1:
            return

        weight_map = weight_accum.view(batch_size, self.ny, self.nx)
        occupancy = (weight_map > 0).float()
        stats = {
            'occupancy_ratio': occupancy.mean().item(),
            'mean_support_weight': weight_map[occupancy.bool()].mean().item() if occupancy.any() else 0.0,
            'max_support_weight': weight_map.max().item(),
            'kernel_points': float(len(self.offsets)),
            'sigma': self.sigma,
            'radius': float(self.radius),
        }
        self._last_debug_stats = stats

        if self.debug_print_to_stdout:
            print(f'[GaussianSoftScatter] stats: {stats}')

        if (not self.debug_save_visualization) or self._saved_visualizations >= self.debug_max_visualizations:
            return

        os.makedirs(self.debug_out_dir, exist_ok=True)
        bev_energy = bev_features[0].norm(dim=0)
        support_map = weight_map[0]
        self._save_map(
            bev_energy,
            os.path.join(self.debug_out_dir, f'bev_energy_step_{self._forward_calls:06d}.png'))
        self._save_map(
            support_map,
            os.path.join(self.debug_out_dir, f'support_weight_step_{self._forward_calls:06d}.png'))
        self._saved_visualizations += 1

    def forward_single(self, voxel_features, coors):
        if coors.size(-1) == 3:
            coors = torch.cat((torch.zeros_like(coors[:, :1]), coors), dim=1)
        batch_canvas = self.forward_batch(voxel_features, coors, batch_size=1)
        return [batch_canvas]

    def forward_batch(self, voxel_features, coors, batch_size):
        assert coors.size(-1) == 4, f'Expected coors shape [N, 4], got {tuple(coors.shape)}'
        coors = coors.long()

        flat_size = batch_size * self.ny * self.nx
        canvas = voxel_features.new_zeros((flat_size, self.in_channels))
        weight_accum = voxel_features.new_zeros((flat_size,))

        batch_inds = coors[:, 0]
        ys = coors[:, 2]
        xs = coors[:, 3]
        weights = self.kernel_weights.to(device=voxel_features.device, dtype=voxel_features.dtype)

        for i, (dy, dx) in enumerate(self.offsets):
            tgt_y = ys + dy
            tgt_x = xs + dx
            valid = (tgt_y >= 0) & (tgt_y < self.ny) & (tgt_x >= 0) & (tgt_x < self.nx)
            if not valid.any():
                continue

            valid_batch = batch_inds[valid]
            valid_y = tgt_y[valid]
            valid_x = tgt_x[valid]
            flat_idx = valid_batch * (self.ny * self.nx) + valid_y * self.nx + valid_x

            weight = float(weights[i].item())
            canvas.index_add_(0, flat_idx, voxel_features[valid] * weight)
            if self.normalize:
                weight_accum.index_add_(
                    0,
                    flat_idx,
                    torch.full(
                        (flat_idx.numel(),),
                        weight,
                        dtype=voxel_features.dtype,
                        device=flat_idx.device))

        if self.normalize:
            safe_norm = weight_accum.clamp(min=self.eps).unsqueeze(1)
            canvas = canvas / safe_norm
            canvas = canvas * (weight_accum > 0).unsqueeze(1)
        else:
            weight_accum = (canvas.abs().sum(dim=1) > 0).to(canvas.dtype)

        batch_canvas = canvas.view(batch_size, self.ny, self.nx, self.in_channels)
        batch_canvas = batch_canvas.permute(0, 3, 1, 2).contiguous()

        self._forward_calls += 1
        self._collect_debug(batch_canvas, weight_accum, batch_size)
        return batch_canvas

    def forward(self, voxel_features, coors, batch_size=None):
        if batch_size is None:
            return self.forward_single(voxel_features, coors)
        return self.forward_batch(voxel_features, coors, batch_size)
