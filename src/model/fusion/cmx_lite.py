from typing import Sequence

import torch
import torch.nn as nn


class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = int(dim)
        hidden = max(1, (self.dim * 4) // int(reduction))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.dim * 2),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        bsz = x1.shape[0]
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(bsz, self.dim * 2)
        max_v = self.max_pool(x).view(bsz, self.dim * 2)
        y = torch.cat((avg, max_v), dim=1)
        y = self.mlp(y).view(bsz, self.dim * 2, 1)
        return y.reshape(bsz, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = int(dim)
        hidden = max(1, self.dim // int(reduction))
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        bsz, _, h, w = x1.shape
        x = torch.cat((x1, x2), dim=1)
        return self.mlp(x).reshape(bsz, 2, 1, h, w).permute(1, 0, 2, 3, 4)


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super().__init__()
        self.lambda_c = float(lambda_c)
        self.lambda_s = float(lambda_s)
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        assert dim % num_heads == 0, f'dim {dim} should be divisible by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        bsz, n_tokens, channels = x1.shape
        q1 = x1.reshape(bsz, -1, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(bsz, -1, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(bsz, -1, 2, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(bsz, -1, 2, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        out_x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(bsz, n_tokens, channels).contiguous()
        out_x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(bsz, n_tokens, channels).contiguous()
        return out_x1, out_x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=4):
        super().__init__()
        reduced_dim = max(1, int(dim) // int(reduction))
        self.channel_proj1 = nn.Linear(dim, reduced_dim * 2)
        self.channel_proj2 = nn.Linear(dim, reduced_dim * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(reduced_dim, num_heads=num_heads)
        self.end_proj1 = nn.Linear(reduced_dim * 2, dim)
        self.end_proj2 = nn.Linear(reduced_dim * 2, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1):
        super().__init__()
        reduced = max(1, int(out_channels) // int(reduction))
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1, bias=True),
            nn.Conv2d(reduced, reduced, kernel_size=3, stride=1, padding=1, bias=True, groups=reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, h, w):
        bsz, n_tokens, channels = x.shape
        x = x.permute(0, 2, 1).reshape(bsz, channels, h, w).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        return self.norm(residual + x)


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=4):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction)

    def forward(self, x1, x2):
        bsz, channels, h, w = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merged = torch.cat((x1, x2), dim=-1)
        return self.channel_emb(merged, h, w)


class CMXLiteFuser(nn.Module):
    """CMX-inspired feature rectification + fusion across backbone scales."""

    def __init__(self, in_channels: Sequence[int], num_heads: Sequence[int] = (2, 4, 8), reduction: int = 1):
        super().__init__()
        in_channels = list(in_channels)
        assert len(in_channels) > 0, 'in_channels must be non-empty'
        if len(num_heads) < len(in_channels):
            num_heads = list(num_heads) + [num_heads[-1]] * (len(in_channels) - len(num_heads))

        self.frm_blocks = nn.ModuleList()
        self.ffm_blocks = nn.ModuleList()
        for ch, heads in zip(in_channels, num_heads):
            self.frm_blocks.append(FeatureRectifyModule(dim=ch, reduction=reduction))
            self.ffm_blocks.append(FeatureFusionModule(dim=ch, reduction=reduction, num_heads=heads))

    def pop_debug_stats(self):
        return None

    def forward(self, radar_feats, camera_feats):
        assert len(radar_feats) == len(camera_feats) == len(self.frm_blocks)
        fused = []
        for stage, (r_feat, c_feat) in enumerate(zip(radar_feats, camera_feats)):
            c_rect, r_rect = self.frm_blocks[stage](c_feat, r_feat)
            fused_feat = self.ffm_blocks[stage](c_rect, r_rect)
            fused.append(fused_feat)
        return tuple(fused)

