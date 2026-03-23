from typing import Optional

import torch
from torch import nn


class BEVFeatureFusion(nn.Module):
    """Fuse radar BEV and camera BEV while keeping detector head unchanged."""

    def __init__(self, channels: int, fusion_type: str = 'concat_1x1'):
        super().__init__()
        self.channels = int(channels)
        self.fusion_type = str(fusion_type)
        self._last_debug_stats = None

        if self.fusion_type == 'add':
            self.alpha = nn.Parameter(torch.tensor(1.0))
        elif self.fusion_type == 'concat_1x1':
            self.proj = nn.Sequential(
                nn.Conv2d(self.channels * 2, self.channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
            )
        elif self.fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Conv2d(self.channels * 2, self.channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            self.refine = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
            )
        elif self.fusion_type == 'none':
            pass
        else:
            raise ValueError(
                f"Unsupported fusion type '{self.fusion_type}'. "
                "Supported types: ['none', 'add', 'concat_1x1', 'gated']")

    def pop_debug_stats(self):
        stats = self._last_debug_stats
        self._last_debug_stats = None
        return stats

    def forward(self, radar_bev: torch.Tensor, camera_bev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if camera_bev is None or self.fusion_type == 'none':
            return radar_bev

        if radar_bev.shape != camera_bev.shape:
            raise ValueError(
                f'Radar and camera BEV shapes must match, got '
                f'{tuple(radar_bev.shape)} vs {tuple(camera_bev.shape)}')

        if self.fusion_type == 'add':
            fused = radar_bev + self.alpha * camera_bev
            self._last_debug_stats = {'alpha': float(self.alpha.detach().item())}
            return fused

        if self.fusion_type == 'concat_1x1':
            fused = self.proj(torch.cat((radar_bev, camera_bev), dim=1))
            return fused

        gate = self.gate(torch.cat((radar_bev, camera_bev), dim=1))
        fused = radar_bev + gate * camera_bev
        fused = self.refine(fused)
        self._last_debug_stats = {
            'gate_mean': float(gate.detach().mean().item()),
            'gate_max': float(gate.detach().max().item()),
            'gate_min': float(gate.detach().min().item()),
        }
        return fused

