import math
from typing import Dict, Optional

import torch
from torch import nn


class ImageBEVGaussianEncoder(nn.Module):
    """Lift monocular image features to BEV via Gaussian soft scatter."""

    def __init__(
            self,
            out_channels: int,
            output_shape,
            point_cloud_range,
            voxel_size,
            depth_num_bins: int = 16,
            depth_min: float = 1.0,
            depth_max: float = 60.0,
            sigma: float = 0.8,
            radius: int = 1,
            normalize: bool = True,
            min_weight: float = 0.0,
            min_opacity: float = 0.05,
            base_channels: int = 32,
            lift_mode: str = 'expected',
            depth_topk: int = 3,
            chunk_size: int = 50000,
            use_depth_variance_sigma: bool = False,
            sigma_beta: float = 0.5,
            sigma_min: float = 0.4,
            sigma_max: float = 1.6,
            debug: Optional[Dict] = None):
        super().__init__()
        self.out_channels = int(out_channels)
        self.ny = int(output_shape[0])
        self.nx = int(output_shape[1])
        self.pc_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)

        self.depth_num_bins = int(depth_num_bins)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.min_opacity = float(min_opacity)
        self.normalize = bool(normalize)
        self.lift_mode = str(lift_mode)
        self.depth_topk = int(depth_topk)
        self.chunk_size = int(chunk_size)
        self.use_depth_variance_sigma = bool(use_depth_variance_sigma)
        self.sigma_beta = float(sigma_beta)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.eps = 1e-6

        self.sigma = float(sigma)
        self.radius = int(radius)
        self.min_weight = float(min_weight)
        offsets, kernel_weights = self._build_kernel()
        self.offsets = offsets
        self.register_buffer('kernel_weights', kernel_weights, persistent=False)

        depth_values = torch.linspace(
            self.depth_min,
            self.depth_max,
            self.depth_num_bins,
            dtype=torch.float32)
        self.register_buffer('depth_values', depth_values, persistent=False)

        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

        self.feature_head = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, self.out_channels, kernel_size=1, padding=0),
        )
        self.depth_head = nn.Conv2d(c4, self.depth_num_bins, kernel_size=1, padding=0)
        self.opacity_head = nn.Conv2d(c4, 1, kernel_size=1, padding=0)

        debug = {} if debug is None else debug
        self.debug_enabled = bool(debug.get('enabled', False))
        self.debug_log_memory = bool(debug.get('log_memory', False))
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

    def _backproject_to_lidar(self, depth_hypotheses, camera_projection, t_lidar_camera, image_shape):
        """
        Backproject image pixels with one or more depth hypotheses to lidar frame.

        Args:
            depth_hypotheses: (B, K, Hf, Wf)
        """
        bsz, num_hyp, feat_h, feat_w = depth_hypotheses.shape
        img_h, img_w = image_shape
        device = depth_hypotheses.device
        dtype = depth_hypotheses.dtype

        ys = (torch.arange(feat_h, device=device, dtype=dtype) + 0.5) * (float(img_h) / float(feat_h))
        xs = (torch.arange(feat_w, device=device, dtype=dtype) + 0.5) * (float(img_w) / float(feat_w))
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        yy = yy.view(1, 1, feat_h, feat_w)
        xx = xx.view(1, 1, feat_h, feat_w)

        fx = camera_projection[:, 0, 0].view(bsz, 1, 1, 1).clamp(min=self.eps)
        fy = camera_projection[:, 1, 1].view(bsz, 1, 1, 1).clamp(min=self.eps)
        cx = camera_projection[:, 0, 2].view(bsz, 1, 1, 1)
        cy = camera_projection[:, 1, 2].view(bsz, 1, 1, 1)

        z = depth_hypotheses
        x_cam = (xx - cx) * z / fx
        y_cam = (yy - cy) * z / fy

        cam_homo = torch.stack(
            (x_cam, y_cam, z, torch.ones_like(z)),
            dim=-1
        ).reshape(bsz, -1, 4)
        lidar_points = torch.bmm(
            t_lidar_camera.to(dtype=dtype),
            cam_homo.transpose(1, 2)).transpose(1, 2)
        return lidar_points[..., :3].reshape(bsz, num_hyp, feat_h, feat_w, 3)

    def _compute_point_sigma(self, depth_var_flat):
        if not self.use_depth_variance_sigma:
            return None
        depth_std = torch.sqrt(depth_var_flat.clamp(min=0.0))
        sigma = self.sigma * (1.0 + self.sigma_beta * depth_std)
        return sigma.clamp(min=self.sigma_min, max=self.sigma_max)

    def _scatter_chunk(
            self,
            canvas,
            weight_accum,
            features_chunk,
            batch_chunk,
            x_chunk,
            y_chunk,
            base_weights_chunk,
            point_sigma_chunk):
        spatial_size = self.ny * self.nx
        kernel_weights = self.kernel_weights.to(device=canvas.device, dtype=canvas.dtype)
        for idx, (dy, dx) in enumerate(self.offsets):
            tx = x_chunk + dx
            ty = y_chunk + dy
            valid = (tx >= 0) & (tx < self.nx) & (ty >= 0) & (ty < self.ny)
            if not valid.any():
                continue

            flat_idx = batch_chunk[valid] * spatial_size + ty[valid] * self.nx + tx[valid]
            if point_sigma_chunk is None:
                scatter_weight = base_weights_chunk[valid] * float(kernel_weights[idx].item())
            else:
                sigma_local = point_sigma_chunk[valid].clamp(min=self.eps)
                dist2 = float(dx * dx + dy * dy)
                offset_weight = torch.exp(-dist2 / (2.0 * sigma_local * sigma_local))
                if self.min_weight > 0.0:
                    offset_weight = offset_weight * (offset_weight >= self.min_weight).to(offset_weight.dtype)
                scatter_weight = base_weights_chunk[valid] * offset_weight

            if scatter_weight.numel() == 0:
                continue
            canvas.index_add_(0, flat_idx, features_chunk[valid] * scatter_weight.unsqueeze(1))
            if self.normalize:
                weight_accum.index_add_(0, flat_idx, scatter_weight)

    def _scatter_to_bev(
            self,
            features_flat,
            batch_inds,
            x_inds,
            y_inds,
            point_weights,
            point_sigma,
            batch_size,
            chunk_size=None):
        flat_size = batch_size * self.ny * self.nx
        canvas = features_flat.new_zeros((flat_size, self.out_channels))
        weight_accum = features_flat.new_zeros((flat_size,))

        num_points = int(features_flat.shape[0])
        if num_points == 0:
            bev = canvas.view(batch_size, self.ny, self.nx, self.out_channels)
            bev = bev.permute(0, 3, 1, 2).contiguous()
            return bev, weight_accum, 0

        if chunk_size is None or int(chunk_size) <= 0:
            chunk_size = num_points
        chunk_size = int(chunk_size)
        chunk_count = 0
        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            chunk_count += 1
            self._scatter_chunk(
                canvas=canvas,
                weight_accum=weight_accum,
                features_chunk=features_flat[start:end],
                batch_chunk=batch_inds[start:end],
                x_chunk=x_inds[start:end],
                y_chunk=y_inds[start:end],
                base_weights_chunk=point_weights[start:end],
                point_sigma_chunk=None if point_sigma is None else point_sigma[start:end])

        if self.normalize:
            safe_norm = weight_accum.clamp(min=self.eps).unsqueeze(1)
            canvas = canvas / safe_norm
            canvas = canvas * (weight_accum > 0).unsqueeze(1)
        else:
            weight_accum = (canvas.abs().sum(dim=1) > 0).to(canvas.dtype)

        bev = canvas.view(batch_size, self.ny, self.nx, self.out_channels)
        bev = bev.permute(0, 3, 1, 2).contiguous()
        return bev, weight_accum, chunk_count

    def _build_debug_stats(
            self,
            weight_accum,
            total_points,
            valid_points,
            effective_bins_per_pixel,
            chunk_count,
            device):
        if not self.debug_enabled:
            return
        occupied = (weight_accum > 0).float()
        stats = {
            'occupancy_ratio': float(occupied.mean().item()),
            'mean_support_weight': float(weight_accum[weight_accum > 0].mean().item()) if (weight_accum > 0).any() else 0.0,
            'max_support_weight': float(weight_accum.max().item()),
            'valid_points_ratio': float(valid_points / max(total_points, 1)),
            'num_points_total': float(total_points),
            'num_points_valid': float(valid_points),
            'effective_bins_per_pixel': float(effective_bins_per_pixel),
            'kernel_points': float(len(self.offsets)),
            'chunk_count': float(chunk_count),
        }
        if self.debug_log_memory and device.type == 'cuda':
            stats['cuda_memory_allocated_mb'] = float(torch.cuda.memory_allocated(device=device) / (1024.0 ** 2))
            stats['cuda_max_memory_allocated_mb'] = float(torch.cuda.max_memory_allocated(device=device) / (1024.0 ** 2))
        self._last_debug_stats = stats

    def _prepare_bev_indices(self, lidar_flat):
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]
        vx, vy = self.voxel_size[0], self.voxel_size[1]

        x_inds = torch.floor((lidar_flat[:, 0] - x_min) / vx).long()
        y_inds = torch.floor((lidar_flat[:, 1] - y_min) / vy).long()
        in_range = (
            (lidar_flat[:, 0] >= x_min) & (lidar_flat[:, 0] < x_max) &
            (lidar_flat[:, 1] >= y_min) & (lidar_flat[:, 1] < y_max) &
            (x_inds >= 0) & (x_inds < self.nx) &
            (y_inds >= 0) & (y_inds < self.ny)
        )
        return x_inds, y_inds, in_range

    def _lift_expected(self, pixel_features, expected_depth, opacity, depth_var, camera_projection, t_lidar_camera, image_shape):
        bsz, channels, feat_h, feat_w = pixel_features.shape
        lidar_points = self._backproject_to_lidar(
            expected_depth.unsqueeze(1),
            camera_projection,
            t_lidar_camera,
            image_shape=image_shape)
        lidar_flat = lidar_points.reshape(-1, 3)
        feat_flat = pixel_features.permute(0, 2, 3, 1).reshape(-1, channels)
        opacity_flat = opacity.reshape(-1)
        depth_var_flat = depth_var.reshape(-1)
        batch_inds = torch.arange(bsz, device=pixel_features.device).view(bsz, 1, 1).expand(bsz, feat_h, feat_w).reshape(-1)

        x_inds, y_inds, in_range = self._prepare_bev_indices(lidar_flat)
        valid = (opacity_flat > self.min_opacity) & in_range

        point_sigma = self._compute_point_sigma(depth_var_flat)
        if point_sigma is not None:
            point_sigma = point_sigma[valid]
        bev, weight_accum, chunk_count = self._scatter_to_bev(
            features_flat=feat_flat[valid],
            batch_inds=batch_inds[valid],
            x_inds=x_inds[valid],
            y_inds=y_inds[valid],
            point_weights=opacity_flat[valid],
            point_sigma=point_sigma,
            batch_size=bsz,
            chunk_size=None)
        return bev, weight_accum, int(valid.numel()), int(valid.sum().item()), chunk_count

    def _lift_naive_dense(self, pixel_features, depth_prob, opacity, depth_var, camera_projection, t_lidar_camera, image_shape):
        bsz, channels, feat_h, feat_w = pixel_features.shape
        num_bins = int(depth_prob.shape[1])
        depth_hyp = self.depth_values.view(1, num_bins, 1, 1).to(dtype=depth_prob.dtype, device=depth_prob.device)
        depth_hyp = depth_hyp.expand(bsz, num_bins, feat_h, feat_w)
        lidar_points = self._backproject_to_lidar(
            depth_hyp,
            camera_projection,
            t_lidar_camera,
            image_shape=image_shape).permute(0, 2, 3, 1, 4).reshape(-1, 3)

        feat_flat = pixel_features.permute(0, 2, 3, 1).unsqueeze(3)
        feat_flat = feat_flat.expand(bsz, feat_h, feat_w, num_bins, channels).reshape(-1, channels)
        opacity_flat = opacity.reshape(-1).repeat_interleave(num_bins)
        depth_prob_flat = depth_prob.permute(0, 2, 3, 1).reshape(-1)
        point_weight = opacity_flat * depth_prob_flat
        depth_var_flat = depth_var.reshape(-1).repeat_interleave(num_bins)
        batch_inds = torch.arange(bsz, device=pixel_features.device).view(bsz, 1, 1, 1)
        batch_inds = batch_inds.expand(bsz, feat_h, feat_w, num_bins).reshape(-1)

        x_inds, y_inds, in_range = self._prepare_bev_indices(lidar_points)
        valid = (opacity_flat > self.min_opacity) & (point_weight > 0) & in_range

        point_sigma = self._compute_point_sigma(depth_var_flat)
        if point_sigma is not None:
            point_sigma = point_sigma[valid]
        bev, weight_accum, chunk_count = self._scatter_to_bev(
            features_flat=feat_flat[valid],
            batch_inds=batch_inds[valid],
            x_inds=x_inds[valid],
            y_inds=y_inds[valid],
            point_weights=point_weight[valid],
            point_sigma=point_sigma,
            batch_size=bsz,
            chunk_size=None)
        return bev, weight_accum, int(valid.numel()), int(valid.sum().item()), chunk_count

    def _lift_topk_chunked(self, pixel_features, depth_prob, opacity, depth_var, camera_projection, t_lidar_camera, image_shape):
        bsz, channels, feat_h, feat_w = pixel_features.shape
        topk = max(1, min(int(self.depth_topk), int(depth_prob.shape[1])))
        prob_topk, idx_topk = torch.topk(depth_prob, k=topk, dim=1)
        depth_values = self.depth_values.to(device=depth_prob.device, dtype=depth_prob.dtype)
        depth_topk = depth_values[idx_topk]

        lidar_points = self._backproject_to_lidar(
            depth_topk,
            camera_projection,
            t_lidar_camera,
            image_shape=image_shape).permute(0, 2, 3, 1, 4).reshape(-1, 3)

        feat_flat = pixel_features.permute(0, 2, 3, 1).reshape(-1, channels)
        feat_flat = feat_flat.unsqueeze(1).expand(-1, topk, -1).reshape(-1, channels)
        opacity_flat = opacity.reshape(-1).repeat_interleave(topk)
        prob_flat = prob_topk.permute(0, 2, 3, 1).reshape(-1)
        point_weight = opacity_flat * prob_flat
        depth_var_flat = depth_var.reshape(-1).repeat_interleave(topk)
        batch_inds = torch.arange(bsz, device=pixel_features.device).view(bsz, 1, 1, 1)
        batch_inds = batch_inds.expand(bsz, feat_h, feat_w, topk).reshape(-1)

        x_inds, y_inds, in_range = self._prepare_bev_indices(lidar_points)
        valid = (opacity_flat > self.min_opacity) & (point_weight > 0) & in_range

        point_sigma = self._compute_point_sigma(depth_var_flat)
        if point_sigma is not None:
            point_sigma = point_sigma[valid]
        bev, weight_accum, chunk_count = self._scatter_to_bev(
            features_flat=feat_flat[valid],
            batch_inds=batch_inds[valid],
            x_inds=x_inds[valid],
            y_inds=y_inds[valid],
            point_weights=point_weight[valid],
            point_sigma=point_sigma,
            batch_size=bsz,
            chunk_size=self.chunk_size)
        return bev, weight_accum, int(valid.numel()), int(valid.sum().item()), chunk_count

    def forward(self, image, camera_projection, t_lidar_camera):
        """
        Args:
            image: (B, 3, H, W)
            camera_projection: (B, 3, 4)
            t_lidar_camera: (B, 4, 4)
        """
        feats = self.encoder(image)
        pixel_features = self.feature_head(feats)
        depth_logits = self.depth_head(feats)
        opacity_logits = self.opacity_head(feats)

        depth_prob = torch.softmax(depth_logits, dim=1)
        depth_values = self.depth_values.to(device=depth_prob.device, dtype=depth_prob.dtype).view(1, -1, 1, 1)
        expected_depth = (depth_prob * depth_values).sum(dim=1)
        depth_var = ((depth_values - expected_depth.unsqueeze(1)) ** 2 * depth_prob).sum(dim=1)
        uncertainty_weight = 1.0 / (1.0 + depth_var)
        opacity = torch.sigmoid(opacity_logits).squeeze(1) * uncertainty_weight

        image_shape = (int(image.shape[-2]), int(image.shape[-1]))
        if self.lift_mode == 'expected':
            bev, weight_accum, total_points, valid_points, chunk_count = self._lift_expected(
                pixel_features=pixel_features,
                expected_depth=expected_depth,
                opacity=opacity,
                depth_var=depth_var,
                camera_projection=camera_projection,
                t_lidar_camera=t_lidar_camera,
                image_shape=image_shape)
            effective_bins_per_pixel = 1.0
        elif self.lift_mode == 'naive_dense':
            bev, weight_accum, total_points, valid_points, chunk_count = self._lift_naive_dense(
                pixel_features=pixel_features,
                depth_prob=depth_prob,
                opacity=opacity,
                depth_var=depth_var,
                camera_projection=camera_projection,
                t_lidar_camera=t_lidar_camera,
                image_shape=image_shape)
            effective_bins_per_pixel = float(self.depth_num_bins)
        elif self.lift_mode == 'topk_chunked':
            bev, weight_accum, total_points, valid_points, chunk_count = self._lift_topk_chunked(
                pixel_features=pixel_features,
                depth_prob=depth_prob,
                opacity=opacity,
                depth_var=depth_var,
                camera_projection=camera_projection,
                t_lidar_camera=t_lidar_camera,
                image_shape=image_shape)
            effective_bins_per_pixel = float(max(1, min(self.depth_topk, self.depth_num_bins)))
        else:
            raise ValueError(
                f"Unsupported camera lift mode '{self.lift_mode}'. "
                "Supported modes: ['expected', 'naive_dense', 'topk_chunked']")

        self._build_debug_stats(
            weight_accum=weight_accum,
            total_points=total_points,
            valid_points=valid_points,
            effective_bins_per_pixel=effective_bins_per_pixel,
            chunk_count=chunk_count,
            device=image.device)
        return bev
