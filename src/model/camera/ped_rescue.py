import torch
from torch import nn

from src.model.utils import draw_heatmap_gaussian


class CameraPedRescue(nn.Module):
    """Monocular Pedestrian proposal branch with BEV projection.

    The branch predicts image-plane pedestrian proposal heatmap + depth,
    decodes top-k proposals, then projects them to BEV.
    """

    def __init__(
            self,
            bev_shape,
            point_cloud_range,
            voxel_size,
            out_size_factor=2,
            topk=200,
            score_threshold=0.15,
            proposal_radius=2,
            depth_min=1.0,
            depth_max=80.0,
            image_heatmap_radius=2,
            feat_channels=(32, 64, 96),
            eps=1e-4):
        super().__init__()
        self.bev_h = int(bev_shape[0])
        self.bev_w = int(bev_shape[1])
        self.pc_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.out_size_factor = int(out_size_factor)

        self.topk = int(topk)
        self.score_threshold = float(score_threshold)
        self.proposal_radius = int(proposal_radius)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.image_heatmap_radius = int(image_heatmap_radius)
        self.eps = float(eps)

        c1, c2, c3 = [int(v) for v in feat_channels]
        self.backbone = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, 1, kernel_size=1, padding=0),
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, 1, kernel_size=1, padding=0),
        )

    def _decode_depth(self, depth_logits):
        depth = torch.sigmoid(depth_logits)
        depth = self.depth_min + depth * (self.depth_max - self.depth_min)
        return depth

    @staticmethod
    def _project_cam_to_img(cam_points_homo, camera_projection):
        img_homo = torch.matmul(cam_points_homo, camera_projection.transpose(0, 1))
        z = img_homo[:, 2].clamp(min=1e-6)
        u = img_homo[:, 0] / z
        v = img_homo[:, 1] / z
        return u, v, z

    def _topk_to_bev(self, heatmap_logits, depth_map, camera_projection, t_lidar_camera, image_shape):
        device = heatmap_logits.device
        batch_size, _, feat_h, feat_w = heatmap_logits.shape
        img_h, img_w = int(image_shape[0]), int(image_shape[1])

        heatmap_prob = torch.sigmoid(heatmap_logits)
        bev_prob = heatmap_logits.new_zeros((batch_size, 1, self.bev_h, self.bev_w))

        flat = heatmap_prob.view(batch_size, -1)
        topk = max(1, min(self.topk, flat.shape[1]))
        scores, inds = torch.topk(flat, k=topk, dim=1)

        scale_x = float(img_w) / float(feat_w)
        scale_y = float(img_h) / float(feat_h)

        for b in range(batch_size):
            score_b = scores[b]
            ind_b = inds[b]
            keep = score_b >= self.score_threshold
            if not torch.any(keep):
                continue

            score_b = score_b[keep]
            ind_b = ind_b[keep]

            ys = torch.div(ind_b, feat_w, rounding_mode='floor')
            xs = ind_b % feat_w

            depth_vals = depth_map[b, 0, ys, xs]
            u = (xs.to(torch.float32) + 0.5) * scale_x
            v = (ys.to(torch.float32) + 0.5) * scale_y

            p = camera_projection[b]
            fx = p[0, 0].clamp(min=self.eps)
            fy = p[1, 1].clamp(min=self.eps)
            cx = p[0, 2]
            cy = p[1, 2]
            tx = p[0, 3] if p.shape[1] >= 4 else torch.tensor(0.0, device=device)
            ty = p[1, 3] if p.shape[1] >= 4 else torch.tensor(0.0, device=device)

            x_cam = (u * depth_vals - cx * depth_vals - tx) / fx
            y_cam = (v * depth_vals - cy * depth_vals - ty) / fy
            cam_homo = torch.stack(
                [x_cam, y_cam, depth_vals, torch.ones_like(depth_vals)],
                dim=1)

            lidar_homo = torch.matmul(t_lidar_camera[b], cam_homo.transpose(0, 1)).transpose(0, 1)
            x_l = lidar_homo[:, 0]
            y_l = lidar_homo[:, 1]

            gx = torch.div((x_l - self.pc_range[0]), self.voxel_size[0] * self.out_size_factor, rounding_mode='floor').long()
            gy = torch.div((y_l - self.pc_range[1]), self.voxel_size[1] * self.out_size_factor, rounding_mode='floor').long()
            valid = (gx >= 0) & (gx < self.bev_w) & (gy >= 0) & (gy < self.bev_h)
            if not torch.any(valid):
                continue

            gx = gx[valid]
            gy = gy[valid]
            sv = score_b[valid].to(bev_prob.dtype)

            for cx_i, cy_i, score_i in zip(gx, gy, sv):
                center = torch.tensor([cx_i.item(), cy_i.item()], dtype=torch.int64, device=device)
                draw_heatmap_gaussian(
                    bev_prob[b, 0],
                    center=center,
                    radius=self.proposal_radius,
                    k=float(score_i.item()))

        bev_prob = bev_prob.clamp(min=self.eps, max=1.0 - self.eps)
        bev_logits = torch.log(bev_prob) - torch.log1p(-bev_prob)
        return bev_logits, bev_prob

    def forward(self, image, camera_projection, t_lidar_camera):
        feats = self.backbone(image)
        heatmap_logits = self.heatmap_head(feats)
        depth_logits = self.depth_head(feats)
        depth_map = self._decode_depth(depth_logits)

        bev_logits, bev_prob = self._topk_to_bev(
            heatmap_logits=heatmap_logits,
            depth_map=depth_map,
            camera_projection=camera_projection,
            t_lidar_camera=t_lidar_camera,
            image_shape=image.shape[-2:])

        return {
            'image_heatmap_logits': heatmap_logits,
            'depth_map': depth_map,
            'bev_logits': bev_logits,
            'bev_prob': bev_prob,
            'feat_shape': tuple(feats.shape[-2:]),
            'image_shape': tuple(image.shape[-2:]),
        }
