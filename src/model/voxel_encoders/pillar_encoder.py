import torch
from torch import nn

from .utils import PFNLayer, get_paddings_indicator

class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        extra_feat_channels (tuple, optional): Additional intermediate PFN
            widths prepended before feat_channels. Useful for increasing
            encoder capacity while keeping the final output width fixed.
            Defaults to ().
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 extra_feat_channels=(),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 mode='max',
                 legacy=True,
                 doppler_feature_aug=False,
                 doppler_vr_index=4,
                 doppler_time_index=6,
                 doppler_add_abs=True,
                 doppler_add_range_norm=True,
                 doppler_add_time_interaction=True,
                 doppler_range_eps=1e-2,
                 recency_weighted_pooling=False,
                 recency_time_index=-1,
                 recency_beta=2.0,
                 recency_min_weight=0.0,
                 range_feature_aug=False,
                 range_add_inverse=False,
                 range_add_log=False,
                 range_eps=1e-2,
                 feature_scale_norm_enabled=False,
                 feature_scale_norm_include_distance=True,
                 feature_scale_norm_eps=1e-3,
                 feature_scale_norm_clip=5.0,
                 quality_weighted_pooling=False,
                 quality_rcs_index=3,
                 quality_vr_index=4,
                 quality_time_index=6,
                 quality_bias=-1.2,
                 quality_w_rcs=1.0,
                 quality_w_vr=0.6,
                 quality_w_time=0.8,
                 quality_w_range=0.4,
                 quality_min_weight=0.1,
                 quality_range_scale=35.0):
        super().__init__()
        extra_feat_channels = tuple(extra_feat_channels)
        feat_channels = tuple(extra_feat_channels) + tuple(feat_channels)
        assert len(feat_channels) > 0
        self.legacy = legacy
        self.raw_in_channels = int(in_channels)

        self.doppler_feature_aug = bool(doppler_feature_aug)
        self.doppler_vr_index = int(doppler_vr_index)
        self.doppler_time_index = int(doppler_time_index)
        self.doppler_add_abs = bool(doppler_add_abs)
        self.doppler_add_range_norm = bool(doppler_add_range_norm)
        self.doppler_add_time_interaction = bool(doppler_add_time_interaction)
        self.doppler_range_eps = float(max(1e-6, doppler_range_eps))
        self.doppler_extra_channels = self._count_doppler_extra_channels(self.raw_in_channels)
        if self.doppler_feature_aug:
            in_channels += self.doppler_extra_channels

        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 2
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.recency_weighted_pooling = bool(recency_weighted_pooling)
        self.recency_time_index = int(recency_time_index)
        self.recency_beta = float(max(0.0, recency_beta))
        self.recency_min_weight = float(max(0.0, min(1.0, recency_min_weight)))
        self.range_feature_aug = bool(range_feature_aug)
        self.range_add_inverse = bool(range_add_inverse)
        self.range_add_log = bool(range_add_log)
        self.range_eps = float(max(1e-6, range_eps))
        self.range_extra_channels = self._count_range_extra_channels()
        if self.range_feature_aug:
            in_channels += self.range_extra_channels
        self.feature_scale_norm_enabled = bool(feature_scale_norm_enabled)
        self.feature_scale_norm_include_distance = bool(feature_scale_norm_include_distance)
        self.feature_scale_norm_eps = float(max(1e-6, feature_scale_norm_eps))
        self.feature_scale_norm_clip = float(max(0.0, feature_scale_norm_clip))
        self.quality_weighted_pooling = bool(quality_weighted_pooling)
        self.quality_rcs_index = int(quality_rcs_index)
        self.quality_vr_index = int(quality_vr_index)
        self.quality_time_index = int(quality_time_index)
        self.quality_bias = float(quality_bias)
        self.quality_w_rcs = float(quality_w_rcs)
        self.quality_w_vr = float(quality_w_vr)
        self.quality_w_time = float(quality_w_time)
        self.quality_w_range = float(quality_w_range)
        self.quality_min_weight = float(max(0.0, min(1.0, quality_min_weight)))
        self.quality_range_scale = float(max(1e-3, quality_range_scale))
        self.eps = 1e-6
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    def _resolve_index(self, num_raw_channels, idx):
        index = int(idx)
        if index < 0:
            index = num_raw_channels + index
        if index < 0 or index >= num_raw_channels:
            return None
        return index

    def _count_doppler_extra_channels(self, num_raw_channels):
        if not self.doppler_feature_aug:
            return 0

        vr_idx = self._resolve_index(num_raw_channels, self.doppler_vr_index)
        if vr_idx is None:
            return 0

        count = 0
        if self.doppler_add_abs:
            count += 1
        if self.doppler_add_range_norm:
            count += 1
        if self.doppler_add_time_interaction:
            time_idx = self._resolve_index(num_raw_channels, self.doppler_time_index)
            if time_idx is not None:
                count += 1
        return count

    def _build_valid_mask(self, features, num_points):
        voxel_count = features.shape[1]
        return get_paddings_indicator(num_points, voxel_count, axis=0).type_as(features)

    def _count_range_extra_channels(self):
        if not self.range_feature_aug:
            return 0
        count = 1  # planar range sqrt(x^2 + y^2)
        if self.range_add_inverse:
            count += 1
        if self.range_add_log:
            count += 1
        return count

    def _minmax_normalize(self, values, valid):
        large = torch.finfo(values.dtype).max
        v_min = torch.where(valid > 0, values, torch.full_like(values, large)).min(dim=1, keepdim=True)[0]
        v_max = torch.where(valid > 0, values, torch.full_like(values, -large)).max(dim=1, keepdim=True)[0]
        v_range = (v_max - v_min).clamp(min=self.eps)
        norm = ((values - v_min) / v_range).clamp(min=0.0, max=1.0)
        return norm * valid

    def _normalize_appended_features(self, augmented_features, valid):
        """Normalize appended range/doppler/distance channels per voxel.

        This stabilizes feature scales before PFN processing when extra radar
        channels are concatenated onto raw point features.
        """
        if augmented_features is None:
            return None
        if not self.feature_scale_norm_enabled:
            return augmented_features

        valid = valid.unsqueeze(-1).type_as(augmented_features)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)

        mean = (augmented_features * valid).sum(dim=1, keepdim=True) / denom
        centered = (augmented_features - mean) * valid
        var = (centered * centered).sum(dim=1, keepdim=True) / denom
        std = torch.sqrt(var + self.feature_scale_norm_eps)

        normalized = centered / std
        if self.feature_scale_norm_clip > 0.0:
            normalized = normalized.clamp(
                min=-self.feature_scale_norm_clip,
                max=self.feature_scale_norm_clip)
        return normalized * valid

    def _build_doppler_features(self, raw_features, num_points):
        if not self.doppler_feature_aug:
            return None

        num_raw_channels = int(raw_features.shape[-1])
        vr_idx = self._resolve_index(num_raw_channels, self.doppler_vr_index)
        if vr_idx is None:
            return None

        valid = self._build_valid_mask(raw_features, num_points)
        vr = raw_features[:, :, vr_idx]
        extra = []

        if self.doppler_add_abs:
            extra.append(vr.abs().unsqueeze(-1))

        if self.doppler_add_range_norm:
            xy_range = torch.sqrt(raw_features[:, :, 0].pow(2) + raw_features[:, :, 1].pow(2) + self.doppler_range_eps)
            vr_over_range = (vr / (xy_range + self.doppler_range_eps)).clamp(min=-5.0, max=5.0)
            extra.append(vr_over_range.unsqueeze(-1))

        if self.doppler_add_time_interaction:
            time_idx = self._resolve_index(num_raw_channels, self.doppler_time_index)
            if time_idx is not None:
                time_values = raw_features[:, :, time_idx]
                abs_t_max = torch.where(
                    valid > 0,
                    time_values.abs(),
                    torch.zeros_like(time_values)).max(dim=1, keepdim=True)[0].clamp(min=1.0)
                t_norm = time_values / abs_t_max
                extra.append((vr * t_norm).unsqueeze(-1))

        if not extra:
            return None

        doppler = torch.cat(extra, dim=-1)
        return doppler * valid.unsqueeze(-1)

    def _build_range_features(self, raw_features, num_points):
        if not self.range_feature_aug:
            return None

        num_raw_channels = int(raw_features.shape[-1])
        x_idx = self._resolve_index(num_raw_channels, 0)
        y_idx = self._resolve_index(num_raw_channels, 1)
        if x_idx is None or y_idx is None:
            return None

        valid = self._build_valid_mask(raw_features, num_points)
        x = raw_features[:, :, x_idx]
        y = raw_features[:, :, y_idx]
        planar_range = torch.sqrt(x.pow(2) + y.pow(2) + self.range_eps)

        feats = [planar_range.unsqueeze(-1)]
        if self.range_add_inverse:
            feats.append((1.0 / (planar_range + self.range_eps)).unsqueeze(-1))
        if self.range_add_log:
            feats.append(torch.log(planar_range + self.range_eps).unsqueeze(-1))

        out = torch.cat(feats, dim=-1)
        return out * valid.unsqueeze(-1)

    def _build_recency_point_weights(self, raw_features, num_points):
        if not self.recency_weighted_pooling or self.recency_beta <= 0.0:
            return None

        num_raw_channels = int(raw_features.shape[-1])
        time_idx = self._resolve_index(num_raw_channels, self.recency_time_index)
        if time_idx is None:
            return None

        valid = self._build_valid_mask(raw_features, num_points)
        time_values = raw_features[:, :, time_idx]

        t_norm = self._minmax_normalize(time_values, valid)

        weights = torch.exp(self.recency_beta * t_norm) * valid
        weights = weights / weights.max(dim=1, keepdim=True)[0].clamp(min=self.eps)
        if self.recency_min_weight > 0.0:
            weights = torch.where(valid > 0, weights.clamp(min=self.recency_min_weight), weights)
        return weights * valid

    def _build_quality_point_weights(self, raw_features, num_points):
        if not self.quality_weighted_pooling:
            return None

        num_raw_channels = int(raw_features.shape[-1])
        rcs_idx = self._resolve_index(num_raw_channels, self.quality_rcs_index)
        vr_idx = self._resolve_index(num_raw_channels, self.quality_vr_index)
        if rcs_idx is None or vr_idx is None:
            return None

        valid = self._build_valid_mask(raw_features, num_points)
        rcs_values = raw_features[:, :, rcs_idx]
        vr_values = raw_features[:, :, vr_idx]
        xy_range = torch.sqrt(raw_features[:, :, 0].pow(2) + raw_features[:, :, 1].pow(2) + self.eps)

        rcs_norm = self._minmax_normalize(rcs_values, valid)
        vr_norm = self._minmax_normalize(vr_values.abs(), valid)
        range_term = torch.exp(-xy_range / self.quality_range_scale) * valid

        quality_logits = (
            self.quality_bias
            + self.quality_w_rcs * rcs_norm
            + self.quality_w_vr * vr_norm
            + self.quality_w_range * range_term)

        time_idx = self._resolve_index(num_raw_channels, self.quality_time_index)
        if time_idx is not None and self.quality_w_time != 0.0:
            time_norm = self._minmax_normalize(raw_features[:, :, time_idx], valid)
            quality_logits = quality_logits + self.quality_w_time * time_norm

        quality = torch.sigmoid(quality_logits) * valid
        if self.quality_min_weight > 0.0:
            quality = torch.where(valid > 0, quality.clamp(min=self.quality_min_weight), quality)
        return quality * valid

    def _combine_point_weights(self, raw_features, num_points, recency_weights, quality_weights):
        if recency_weights is None and quality_weights is None:
            return None

        valid = self._build_valid_mask(raw_features, num_points)
        if recency_weights is None:
            combined = quality_weights
        elif quality_weights is None:
            combined = recency_weights
        else:
            combined = recency_weights * quality_weights

        combined = combined * valid
        combined = combined / combined.max(dim=1, keepdim=True)[0].clamp(min=self.eps)
        return combined * valid

    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        raw_features = features
        valid = self._build_valid_mask(raw_features, num_points)
        recency_weights = self._build_recency_point_weights(raw_features, num_points)
        quality_weights = self._build_quality_point_weights(raw_features, num_points)
        point_weights = self._combine_point_weights(raw_features, num_points, recency_weights, quality_weights)
        doppler_features = self._build_doppler_features(raw_features, num_points)
        range_features = self._build_range_features(raw_features, num_points)
        doppler_features = self._normalize_appended_features(doppler_features, valid)
        range_features = self._normalize_appended_features(range_features, valid)
        if doppler_features is not None:
            features = torch.cat([features, doppler_features], dim=-1)
        if range_features is not None:
            features = torch.cat([features, range_features], dim=-1)

        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :2])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
            else:
                f_center = features[:, :, :2]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            if self.feature_scale_norm_include_distance:
                points_dist = self._normalize_appended_features(points_dist, valid)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        if point_weights is not None:
            point_weights = point_weights * mask.squeeze(-1)

        for pfn in self.pfn_layers:
            features = pfn(features, num_points, point_weights=point_weights)

        return features.squeeze(1)
