import os
import tempfile
import pickle
import math
from datetime import datetime

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from vod.evaluation import Evaluation
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation


import lightning as L
import torch.distributed as dist

from src.ops import Voxelization
from src.model.voxel_encoders import PillarFeatureNet
from src.model.middle_encoders import PointPillarsScatter, GaussianSoftScatter
from src.model.backbones import SECOND
from src.model.necks import SECONDFPN
from src.model.heads import CenterHead
from src.model.camera import CameraPedRescue
from src.model.losses import GaussianFocalLoss
from src.model.utils import draw_heatmap_gaussian


def _largest_divisor_at_most(value, upper):
    divisor = max(1, min(int(upper), int(value)))
    while value % divisor != 0 and divisor > 1:
        divisor -= 1
    return divisor


class CenterPoint(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        self.img_shape = torch.tensor([1936 , 1216])
        self.data_root = config.get('data_root', None)
        self.class_names = config.get('class_names', None)
        self.output_dir = config.get('output_dir', None)
        self.pc_range = torch.tensor(config.get('point_cloud_range', None))
        
        voxel_layer_config = config.get('pts_voxel_layer', None)
        voxel_encoder_config = config.get('voxel_encoder', None)
        middle_encoder_config = config.get('middle_encoder', None)
        fusion_config = config.get('fusion', {})
        backbone_config = config.get('backbone', None)
        neck_config = config.get('neck', None)
        head_config = config.get('head', None)

        regularization_cfg = config.get('regularization', {})
        regularization_cfg = dict(regularization_cfg or {})
        freeze_bn_cfg = dict(regularization_cfg.get('freeze_bn', {}) or {})
        self.norm_mode = str(regularization_cfg.get('norm_mode', 'batchnorm')).lower()
        self.group_norm_groups = int(regularization_cfg.get('group_norm_groups', 16))
        self.freeze_bn_enabled = bool(freeze_bn_cfg.get('enabled', False))
        self.freeze_bn_epoch = int(freeze_bn_cfg.get('freeze_epoch', 0))
        self.freeze_bn_affine = bool(freeze_bn_cfg.get('freeze_affine', False))
        self._bn_freeze_notice_printed = False
        
        self.voxel_layer = Voxelization(**voxel_layer_config)
        self.voxel_encoder = PillarFeatureNet(**voxel_encoder_config)
        middle_encoder_cfg = dict(middle_encoder_config)
        self.middle_output_shape = tuple(int(v) for v in middle_encoder_cfg.get('output_shape', [160, 160]))
        middle_encoder_type = middle_encoder_cfg.pop('type', 'hard')
        if middle_encoder_type == 'gaussian_soft':
            self.middle_encoder = GaussianSoftScatter(**middle_encoder_cfg)
        elif middle_encoder_type == 'hard':
            hard_scatter_cfg = {
                'in_channels': middle_encoder_cfg['in_channels'],
                'output_shape': middle_encoder_cfg['output_shape'],
            }
            self.middle_encoder = PointPillarsScatter(**hard_scatter_cfg)
        else:
            raise ValueError(
                f"Unsupported middle encoder type '{middle_encoder_type}'. "
                "Supported types: ['hard', 'gaussian_soft']")

        requested_fusion = bool(fusion_config.get('enabled', False))
        if requested_fusion:
            raise ValueError(
                "Fusion has been removed from this codebase. "
                "Please run radar-only profiles/configs.")

        self.backbone = SECOND(**backbone_config)
        self.neck = SECONDFPN(**neck_config)
        self.head = CenterHead(**head_config)

        self.camera_rescue_cfg = dict(config.get('camera_rescue', {}) or {})
        self.camera_rescue_enabled = bool(self.camera_rescue_cfg.get('enabled', False))
        self._camera_missing_warned = False
        self.camera_rescue = None
        self.camera_rescue_loss_heatmap = None
        self.camera_rescue_loss_weight_heatmap = float(self.camera_rescue_cfg.get('loss_weight_heatmap', 1.0))
        self.camera_rescue_loss_weight_depth = float(self.camera_rescue_cfg.get('loss_weight_depth', 0.2))
        self.camera_rescue_support_dilation = int(self.camera_rescue_cfg.get('support_dilation', 2))
        self.camera_rescue_ped_task_id_cfg = int(self.camera_rescue_cfg.get('ped_task_id', 1))
        self.camera_fusion_gate_bias = nn.Parameter(torch.tensor(
            float((self.camera_rescue_cfg.get('fusion_gate', {}) or {}).get('bias', -2.0)),
            dtype=torch.float32), requires_grad=self.camera_rescue_enabled)
        self.camera_fusion_gate_no_support = nn.Parameter(torch.tensor(
            float((self.camera_rescue_cfg.get('fusion_gate', {}) or {}).get('no_support_weight', 4.0)),
            dtype=torch.float32), requires_grad=self.camera_rescue_enabled)
        self.camera_fusion_gate_cam_conf = nn.Parameter(torch.tensor(
            float((self.camera_rescue_cfg.get('fusion_gate', {}) or {}).get('cam_conf_weight', 1.5)),
            dtype=torch.float32), requires_grad=self.camera_rescue_enabled)
        self.camera_fusion_gate_radar_unc = nn.Parameter(torch.tensor(
            float((self.camera_rescue_cfg.get('fusion_gate', {}) or {}).get('radar_uncertainty_weight', 1.0)),
            dtype=torch.float32), requires_grad=self.camera_rescue_enabled)
        self.camera_fusion_scale = nn.Parameter(torch.tensor(
            float((self.camera_rescue_cfg.get('fusion_gate', {}) or {}).get('camera_scale', 1.0)),
            dtype=torch.float32), requires_grad=self.camera_rescue_enabled)
        if self.camera_rescue_enabled:
            train_cfg = dict((head_config or {}).get('train_cfg', {}) or {})
            out_size_factor = int(train_cfg.get('out_size_factor', 2))
            grid_size = train_cfg.get('grid_size', [self.middle_output_shape[1], self.middle_output_shape[0], 1])
            bev_w = int(grid_size[0]) // out_size_factor
            bev_h = int(grid_size[1]) // out_size_factor
            self.camera_rescue = CameraPedRescue(
                bev_shape=(bev_h, bev_w),
                point_cloud_range=config.get('point_cloud_range', [0, -25.6, -3, 51.2, 25.6, 2]),
                voxel_size=(config.get('voxel_size', [0.2, 0.2, 5])[:2]),
                out_size_factor=out_size_factor,
                topk=int(self.camera_rescue_cfg.get('topk', 200)),
                score_threshold=float(self.camera_rescue_cfg.get('score_threshold', 0.15)),
                proposal_radius=int(self.camera_rescue_cfg.get('proposal_radius', 2)),
                depth_min=float(self.camera_rescue_cfg.get('depth_min', 1.0)),
                depth_max=float(self.camera_rescue_cfg.get('depth_max', 80.0)),
                image_heatmap_radius=int(self.camera_rescue_cfg.get('image_heatmap_radius', 2)),
                feat_channels=tuple(self.camera_rescue_cfg.get('feat_channels', [32, 64, 96])),
            )
            self.camera_rescue_loss_heatmap = GaussianFocalLoss(
                reduction='mean',
                loss_weight=1.0)

        if self.norm_mode == 'groupnorm':
            replaced = self._convert_batchnorm_to_groupnorm()
            print(f"[regularization] Converted {replaced} BatchNorm layers to GroupNorm.")
        elif self.norm_mode != 'batchnorm':
            raise ValueError(f"Unsupported regularization.norm_mode '{self.norm_mode}'; use 'batchnorm' or 'groupnorm'.")

        if self.freeze_bn_enabled and self.norm_mode == 'groupnorm':
            print("[regularization] freeze_bn.enabled=true has no effect with norm_mode=groupnorm.")

        self.optimizer_config = config.get('optimizer', None)
        
        self.vod_kitti_locations = KittiLocations(root_dir = self.data_root, 
                                     output_dir= self.output_dir,
                                     frame_set_path='',
                                     pred_dir='',)
        self.inference_mode = config.get('inference_mode', 'val')
        self.save_results = config.get('save_preds_results', False)
        self.val_results_list =[]
        self.freeze_modules_cfg = dict(config.get('freeze_modules', {}) or {})
        self._apply_module_freeze_cfg()

    def _resolve_ped_label_index(self):
        if not isinstance(self.class_names, (list, tuple)):
            return 1
        if 'Pedestrian' in self.class_names:
            return int(self.class_names.index('Pedestrian'))
        return 1

    def _resolve_ped_task_id(self):
        if self.head is None or not hasattr(self.head, 'class_names'):
            return int(self.camera_rescue_ped_task_id_cfg)
        task_id = int(self.camera_rescue_ped_task_id_cfg)
        if 0 <= task_id < len(self.head.class_names):
            task_cls = self.head.class_names[task_id]
            if isinstance(task_cls, (list, tuple)) and 'Pedestrian' in task_cls:
                return task_id
        for idx, cls_names in enumerate(self.head.class_names):
            if isinstance(cls_names, (list, tuple)) and 'Pedestrian' in cls_names:
                return int(idx)
        return max(0, min(task_id, len(self.head.class_names) - 1))

    def _build_radar_support_map(self, coors, batch_size, target_hw):
        full_h, full_w = int(self.middle_output_shape[0]), int(self.middle_output_shape[1])
        support = coors.new_zeros((batch_size, 1, full_h, full_w), dtype=torch.float32)
        if coors.numel() > 0:
            b = coors[:, 0].long()
            y = coors[:, 2].long()
            x = coors[:, 3].long()
            valid = (b >= 0) & (b < batch_size) & (y >= 0) & (y < full_h) & (x >= 0) & (x < full_w)
            if torch.any(valid):
                support[b[valid], 0, y[valid], x[valid]] = 1.0

        tgt_h, tgt_w = int(target_hw[0]), int(target_hw[1])
        if full_h != tgt_h or full_w != tgt_w:
            stride_h = full_h // max(1, tgt_h)
            stride_w = full_w // max(1, tgt_w)
            if stride_h > 1 and stride_w > 1 and full_h % tgt_h == 0 and full_w % tgt_w == 0:
                support = F.max_pool2d(support, kernel_size=(stride_h, stride_w), stride=(stride_h, stride_w))
            else:
                support = F.interpolate(support, size=(tgt_h, tgt_w), mode='nearest')

        dilation = int(max(0, self.camera_rescue_support_dilation))
        if dilation > 0:
            kernel = 2 * dilation + 1
            support = F.max_pool2d(support, kernel_size=kernel, stride=1, padding=dilation)
        return support.clamp(min=0.0, max=1.0)

    def _build_camera_image_targets(self, batch, feat_shape):
        device = batch['image'].device
        batch_size = batch['image'].shape[0]
        feat_h, feat_w = int(feat_shape[0]), int(feat_shape[1])
        img_h, img_w = int(batch['image'].shape[-2]), int(batch['image'].shape[-1])

        heatmap_target = torch.zeros((batch_size, 1, feat_h, feat_w), device=device, dtype=torch.float32)
        depth_target = torch.zeros((batch_size, 1, feat_h, feat_w), device=device, dtype=torch.float32)
        depth_mask = torch.zeros((batch_size, 1, feat_h, feat_w), device=device, dtype=torch.float32)

        ped_label = self._resolve_ped_label_index()
        scale_x = float(feat_w) / float(img_w)
        scale_y = float(feat_h) / float(img_h)
        radius = int(max(1, getattr(self.camera_rescue, 'image_heatmap_radius', 2)))

        for b in range(batch_size):
            labels = batch['gt_labels_3d'][b].to(device=device)
            if labels.numel() == 0:
                continue
            ped_mask = labels == ped_label
            if not torch.any(ped_mask):
                continue

            boxes_obj = batch['gt_bboxes_3d'][b]
            boxes_tensor = boxes_obj.tensor.to(device=device)
            ped_boxes = boxes_tensor[ped_mask][:, :7]
            if ped_boxes.numel() == 0:
                continue

            centers_lidar = ped_boxes[:, :3]
            lidar_homo = torch.cat(
                [centers_lidar, torch.ones((centers_lidar.shape[0], 1), device=device, dtype=centers_lidar.dtype)],
                dim=1)

            t_lidar_camera = batch['t_lidar_camera'][b].to(device=device, dtype=torch.float32)
            t_camera_lidar = torch.linalg.inv(t_lidar_camera)
            cam_homo = torch.matmul(t_camera_lidar, lidar_homo.transpose(0, 1)).transpose(0, 1)
            z_cam = cam_homo[:, 2]
            valid_z = z_cam > 1e-3
            if not torch.any(valid_z):
                continue

            cam_homo = cam_homo[valid_z]
            z_cam = z_cam[valid_z]

            proj = batch['camera_projection'][b].to(device=device, dtype=torch.float32)
            img_homo = torch.matmul(cam_homo, proj.transpose(0, 1))
            u = img_homo[:, 0] / img_homo[:, 2].clamp(min=1e-6)
            v = img_homo[:, 1] / img_homo[:, 2].clamp(min=1e-6)
            valid_img = (u >= 0.0) & (u < float(img_w)) & (v >= 0.0) & (v < float(img_h))
            if not torch.any(valid_img):
                continue

            u = u[valid_img]
            v = v[valid_img]
            z_cam = z_cam[valid_img]
            fx = u * scale_x
            fy = v * scale_y
            cx_int = fx.long()
            cy_int = fy.long()

            valid_feat = (cx_int >= 0) & (cx_int < feat_w) & (cy_int >= 0) & (cy_int < feat_h)
            if not torch.any(valid_feat):
                continue
            cx_int = cx_int[valid_feat]
            cy_int = cy_int[valid_feat]
            z_cam = z_cam[valid_feat]

            for cx_i, cy_i, z_i in zip(cx_int, cy_int, z_cam):
                center = torch.tensor([int(cx_i.item()), int(cy_i.item())], dtype=torch.int64, device=device)
                draw_heatmap_gaussian(heatmap_target[b, 0], center, radius=radius, k=1.0)
                y_idx = int(cy_i.item())
                x_idx = int(cx_i.item())
                if depth_mask[b, 0, y_idx, x_idx] == 0 or z_i < depth_target[b, 0, y_idx, x_idx]:
                    depth_target[b, 0, y_idx, x_idx] = z_i
                    depth_mask[b, 0, y_idx, x_idx] = 1.0

        return heatmap_target, depth_target, depth_mask

    def _compute_camera_rescue_losses(self, batch, camera_out):
        if (not self.camera_rescue_enabled) or camera_out is None:
            return {}
        if self.camera_rescue_loss_heatmap is None:
            return {}

        heatmap_target, depth_target, depth_mask = self._build_camera_image_targets(
            batch=batch,
            feat_shape=camera_out['feat_shape'])
        image_heatmap_logits = camera_out['image_heatmap_logits']
        image_heatmap_prob = image_heatmap_logits.sigmoid().clamp(min=1e-4, max=1.0 - 1e-4)
        num_pos = float(heatmap_target.eq(1).float().sum().item())
        loss_heatmap = self.camera_rescue_loss_heatmap(
            image_heatmap_prob,
            heatmap_target,
            avg_factor=max(num_pos, 1.0))

        depth_pred = camera_out['depth_map']
        valid = depth_mask > 0
        if torch.any(valid):
            loss_depth = F.l1_loss(depth_pred[valid], depth_target[valid], reduction='mean')
        else:
            loss_depth = depth_pred.sum() * 0.0

        return {
            'camera.loss_heatmap': loss_heatmap * self.camera_rescue_loss_weight_heatmap,
            'camera.loss_depth': loss_depth * self.camera_rescue_loss_weight_depth,
        }

    def _apply_camera_rescue(self, batch, ret_dict, coors, batch_size):
        if not self.camera_rescue_enabled or self.camera_rescue is None:
            return None
        has_camera_inputs = (
            ('image' in batch) and
            ('camera_projection' in batch) and
            ('t_lidar_camera' in batch))
        if not has_camera_inputs:
            if self.training:
                raise RuntimeError('camera_rescue.enabled=true but camera tensors are missing in batch.')
            if not self._camera_missing_warned:
                print('[camera_rescue] Missing camera tensors in eval batch; skipping camera rescue path.')
                self._camera_missing_warned = True
            return None

        camera_out = self.camera_rescue(
            image=batch['image'],
            camera_projection=batch['camera_projection'],
            t_lidar_camera=batch['t_lidar_camera'])

        ped_task_id = self._resolve_ped_task_id()
        if ped_task_id < 0 or ped_task_id >= len(ret_dict):
            return camera_out
        if len(ret_dict[ped_task_id]) == 0:
            return camera_out

        radar_heatmap_logits = ret_dict[ped_task_id][0]['heatmap']
        cam_bev_logits = camera_out['bev_logits'].to(radar_heatmap_logits.dtype)
        if cam_bev_logits.shape[-2:] != radar_heatmap_logits.shape[-2:]:
            cam_bev_logits = F.interpolate(
                cam_bev_logits,
                size=radar_heatmap_logits.shape[-2:],
                mode='bilinear',
                align_corners=False)
        support = self._build_radar_support_map(
            coors=coors,
            batch_size=batch_size,
            target_hw=radar_heatmap_logits.shape[-2:]).to(radar_heatmap_logits.dtype)

        radar_prob = radar_heatmap_logits.sigmoid()
        cam_prob = cam_bev_logits.sigmoid()
        gate_input = (
            self.camera_fusion_gate_bias
            + self.camera_fusion_gate_no_support * (1.0 - support)
            + self.camera_fusion_gate_cam_conf * cam_prob
            + self.camera_fusion_gate_radar_unc * (1.0 - radar_prob))
        gate = gate_input.sigmoid()
        fused_prob = radar_prob + gate * self.camera_fusion_scale.abs() * cam_prob * (1.0 - radar_prob)
        fused_prob = fused_prob.clamp(min=1e-4, max=1.0 - 1e-4)
        fused_logits = torch.log(fused_prob) - torch.log1p(-fused_prob)
        ret_dict[ped_task_id][0]['heatmap'] = fused_logits

        camera_out['gate'] = gate
        camera_out['support'] = support
        camera_out['ped_task_id'] = ped_task_id
        return camera_out

    def _log_module_debug_stats(self, stage, module, module_name):
        if module is None or not hasattr(module, 'pop_debug_stats'):
            return
        stats = module.pop_debug_stats()
        if not stats:
            return
        for name, value in stats.items():
            if not isinstance(value, (float, int)):
                continue
            self.log(f'{stage}/{module_name}/{name}',
                     float(value),
                     batch_size=1,
                     sync_dist=(stage == 'validation'))

    def _set_module_trainable(self, module, trainable):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = bool(trainable)

    def _apply_module_freeze_cfg(self):
        cfg = dict(self.freeze_modules_cfg or {})
        if not cfg:
            return

        freeze_radar_encoder = bool(cfg.get('radar_encoder', False))
        freeze_radar_backbone = bool(cfg.get('radar_backbone', False))
        freeze_radar_head = bool(cfg.get('radar_head', False))

        if freeze_radar_encoder:
            self._set_module_trainable(self.voxel_encoder, False)
            self._set_module_trainable(self.middle_encoder, False)
        if freeze_radar_backbone:
            self._set_module_trainable(self.backbone, False)
            self._set_module_trainable(self.neck, False)
        if freeze_radar_head:
            self._set_module_trainable(self.head, False)

        print(
            '[freeze_modules] '
            f'radar_encoder={freeze_radar_encoder} '
            f'radar_backbone={freeze_radar_backbone} '
            f'radar_head={freeze_radar_head}')

    def _convert_batchnorm_to_groupnorm(self):
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

        def _convert(module):
            replaced = 0
            for name, child in list(module.named_children()):
                if isinstance(child, bn_types):
                    groups = _largest_divisor_at_most(child.num_features, self.group_norm_groups)
                    gn = nn.GroupNorm(
                        num_groups=groups,
                        num_channels=child.num_features,
                        eps=child.eps,
                        affine=child.affine)
                    if child.affine:
                        with torch.no_grad():
                            gn.weight.copy_(child.weight.detach())
                            gn.bias.copy_(child.bias.detach())
                    setattr(module, name, gn)
                    replaced += 1
                else:
                    replaced += _convert(child)
            return replaced

        return _convert(self)

    def _freeze_batchnorm_layers(self):
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        for module in self.modules():
            if isinstance(module, bn_types):
                module.eval()
                if self.freeze_bn_affine:
                    if module.weight is not None:
                        module.weight.requires_grad = False
                    if module.bias is not None:
                        module.bias.requires_grad = False

    def _apply_batchnorm_freeze_if_needed(self):
        if not self.freeze_bn_enabled:
            return
        if self.current_epoch < self.freeze_bn_epoch:
            return
        self._freeze_batchnorm_layers()
        if not self._bn_freeze_notice_printed:
            print(
                f"[regularization] BatchNorm freeze active from epoch {self.freeze_bn_epoch} "
                f"(current_epoch={self.current_epoch}, freeze_affine={self.freeze_bn_affine}).")
            self._bn_freeze_notice_printed = True

    def on_train_start(self):
        self._apply_batchnorm_freeze_if_needed()

    def on_train_epoch_start(self):
        self._apply_batchnorm_freeze_if_needed()

    ## Voxelization
    def voxelize(self, points):
        voxel_dict = dict()
        voxels, coors, num_points = [], [], []
        for i, res in enumerate(points):
            res_voxels, res_coors, res_num_points = self.voxel_layer(res.cuda())
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)

        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors = torch.cat(coors, dim=0)

        voxel_dict['voxels'] = voxels
        voxel_dict['num_points'] = num_points
        voxel_dict['coors'] = coors

        return voxel_dict
    
    def _model_forward(self, batch, return_aux=False):
        pts_data = batch['pts']

        voxel_dict = self.voxelize(pts_data)
    
        voxels = voxel_dict['voxels']
        num_points = voxel_dict['num_points']
        coors = voxel_dict['coors']
    
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        # Use dataloader batch size directly.
        # Some samples can have zero remaining voxels (e.g., after stronger
        # geometric augmentation), in which case inferring batch size from
        # `coors[:, 0].max()+1` underestimates the true batch size.
        bs = len(pts_data)
        radar_bev_feats = self.middle_encoder(voxel_features, coors, bs)

        backbone_feats = self.backbone(radar_bev_feats)
        neck_feats = self.neck(backbone_feats)
        ret_dict = self.head(neck_feats)
        camera_out = self._apply_camera_rescue(
            batch=batch,
            ret_dict=ret_dict,
            coors=coors,
            batch_size=bs)
        if not return_aux:
            return ret_dict

        aux_losses = self._compute_camera_rescue_losses(batch=batch, camera_out=camera_out)
        aux = {
            'losses': aux_losses,
            'camera_out': camera_out,
        }
        return ret_dict, aux
    
    def training_step(self, batch, batch_idx):
        gt_label_3d = batch['gt_labels_3d']
        gt_bboxes_3d = batch['gt_bboxes_3d']
        
        ret_dict, aux = self._model_forward(batch, return_aux=True)
        self._log_module_debug_stats(stage='train', module=self.middle_encoder, module_name='middle_encoder')
        loss_input = [gt_bboxes_3d, gt_label_3d, ret_dict]
        
        losses = self.head.loss(*loss_input)
        for loss_name, loss_value in (aux.get('losses', {}) or {}).items():
            losses[loss_name] = loss_value
        
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            log_vars[loss_name] = loss_value.mean()

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
            self.log(f'train/{loss_name}', loss_value, batch_size=1)
        camera_out = aux.get('camera_out', None)
        if camera_out is not None and ('gate' in camera_out) and ('support' in camera_out):
            self.log('train/camera_rescue/gate_mean', camera_out['gate'].mean(), batch_size=1)
            self.log('train/camera_rescue/support_ratio', camera_out['support'].mean(), batch_size=1)

        return loss
    
    def configure_optimizers(self):
        optimizer_cfg = dict(self.optimizer_config or {})
        scheduler_cfg = optimizer_cfg.pop('scheduler', None)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError('No trainable parameters found. Check freeze_modules configuration.')
        optimizer = torch.optim.AdamW(trainable_params, **optimizer_cfg)

        if not scheduler_cfg or not bool(scheduler_cfg.get('enabled', False)):
            return optimizer

        total_steps = int(getattr(self.trainer, 'estimated_stepping_batches', 0) or 0)
        if total_steps <= 0:
            return optimizer

        scheduler_type = str(scheduler_cfg.get('type', 'flat_step')).lower()
        warmup_steps = int(scheduler_cfg.get('warmup_steps', 0) or 0)
        if warmup_steps <= 0:
            warmup_epochs = float(scheduler_cfg.get('warmup_epochs', 0.0) or 0.0)
            if warmup_epochs > 0:
                steps_per_epoch = int(getattr(self.trainer, 'num_training_batches', 0) or 0)
                warmup_steps = int(round(warmup_epochs * steps_per_epoch))

        warmup_steps = max(0, min(warmup_steps, max(0, total_steps - 1)))
        decay_steps = max(1, total_steps - warmup_steps)

        min_lr_ratio = float(scheduler_cfg.get('min_lr_ratio', 0.02))
        min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

        def _with_warmup(post_warmup_scale_fn):
            def _lr_lambda(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return max(float(step + 1) / float(warmup_steps), 1e-8)
                return post_warmup_scale_fn(step)
            return _lr_lambda

        if scheduler_type in ('cosine', 'cosine_warmup'):
            def _post_warmup_cosine(step):
                progress = float(step - warmup_steps) / float(decay_steps)
                progress = max(0.0, min(1.0, progress))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=_with_warmup(_post_warmup_cosine))
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'name': 'lr-cosine-warmup',
                },
            }

        if scheduler_type in ('flat_step', 'multistep', 'piecewise'):
            milestones = scheduler_cfg.get('milestones', [0.45, 0.70, 0.85])
            if milestones is None:
                milestones = []
            milestones = sorted(
                max(0.0, min(1.0, float(v)))
                for v in milestones)
            gamma = float(scheduler_cfg.get('gamma', 0.3))
            gamma = max(1e-8, min(1.0, gamma))

            def _post_warmup_flat_step(step):
                progress = float(step + 1) / float(total_steps)
                num_drops = sum(progress >= milestone for milestone in milestones)
                scale = gamma ** int(num_drops)
                return max(scale, min_lr_ratio)

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=_with_warmup(_post_warmup_flat_step))
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'name': 'lr-flat-step',
                },
            }

        if scheduler_type in ('plateau', 'reduce_on_plateau'):
            base_lrs = [group['lr'] for group in optimizer.param_groups]
            min_lr = [max(lr * min_lr_ratio, 1e-8) for lr in base_lrs]
            factor = float(scheduler_cfg.get('factor', 0.5))
            factor = max(1e-8, min(1.0, factor))
            patience = int(scheduler_cfg.get('patience', 2))
            threshold = float(scheduler_cfg.get('threshold', 1e-3))
            monitor_metric = str(scheduler_cfg.get('monitor', 'validation/ROI/mAP'))
            mode = str(scheduler_cfg.get('mode', 'max')).lower()
            if mode not in ('min', 'max'):
                mode = 'max'

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=mode,
                factor=factor,
                patience=max(0, patience),
                threshold=threshold,
                min_lr=min_lr)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'monitor': monitor_metric,
                    'name': 'lr-plateau',
                },
            }

        raise ValueError(
            f"Unsupported optimizer.scheduler.type='{scheduler_type}'. "
            "Supported: ['flat_step', 'cosine', 'plateau']"
        )
    
    
    def validation_step(self, batch, batch_idx):
        assert len(batch['pts']) == 1, 'Batch size should be 1 for validation'
        metas = batch['metas']
        gt_label_3d = batch['gt_labels_3d']
        gt_bboxes_3d = batch['gt_bboxes_3d']
        
        ret_dict, aux = self._model_forward(batch, return_aux=True)
        self._log_module_debug_stats(stage='validation', module=self.middle_encoder, module_name='middle_encoder')
        loss_input = [gt_bboxes_3d, gt_label_3d, ret_dict]
        
        bbox_list = self.head.get_bboxes(ret_dict, img_metas=metas)
        
        bbox_results = [
            dict(bboxes_3d = bboxes, 
                 scores_3d = scores, 
                 labels_3d = labels)
            for bboxes, scores, labels in bbox_list
        ]

        losses = self.head.loss(*loss_input)
        for loss_name, loss_value in (aux.get('losses', {}) or {}).items():
            losses[loss_name] = loss_value
        
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            log_vars[loss_name] = loss_value.mean()
        
        val_loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = val_loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
            self.log(f'validation/{loss_name}', loss_value, batch_size=1, sync_dist=True)
        camera_out = aux.get('camera_out', None)
        if camera_out is not None and ('gate' in camera_out) and ('support' in camera_out):
            self.log('validation/camera_rescue/gate_mean', camera_out['gate'].mean(), batch_size=1, sync_dist=True)
            self.log('validation/camera_rescue/support_ratio', camera_out['support'].mean(), batch_size=1, sync_dist=True)
        # task0.loss_heatmap', 'task0.loss_bbox', 'task1.loss_heatmap', 'task1.loss_bbox', 'task2.loss_heatmap', 'task2.loss_bbox', 'loss'
        sample_idx = batch['metas'][0]['num_frame']
        # Convert to a compact CPU representation immediately to avoid
        # retaining full batch tensors/GPU objects for the whole val epoch.
        box_dict = self.convert_valid_bboxes(bbox_results[0], sample_idx)
        self.val_results_list.append(dict(
            sample_idx=sample_idx,
            box_dict=box_dict,
        ))
    
    def on_validation_epoch_end(self):
        if (not self.save_results) or self.training: 
            tmp_dir = tempfile.TemporaryDirectory()
            working_dir = tmp_dir.name
        else:
            tmp_dir = None
            working_dir = self.output_dir

        preds_dst = os.path.join(working_dir, f'{self.inference_mode}_preds')
        os.makedirs(preds_dst, exist_ok=True)
        
        outputs = self.val_results_list
        self.val_results_list = []
        results = self.format_results(outputs, results_save_path=preds_dst)
        
        if self.inference_mode =='val': 
            gt_dst = os.path.join(self.data_root, 'lidar', 'training', 'label_2')
            
            evaluation = Evaluation(test_annotation_file=gt_dst)
            results = evaluation.evaluate(result_path=preds_dst, current_class=[0, 1, 2])
            
            self.log('validation/entire_area/Car_3d', results['entire_area']['Car_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/entire_area/Pedestrian_3d', results['entire_area']['Pedestrian_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/entire_area/Cyclist_3d', results['entire_area']['Cyclist_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/entire_area/mAP', (results['entire_area']['Car_3d_all'] + results['entire_area']['Pedestrian_3d_all'] + results['entire_area']['Cyclist_3d_all']) / 3, batch_size=1, sync_dist=True)
            self.log('validation/ROI/Car_3d', results['roi']['Car_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/ROI/Pedestrian_3d', results['roi']['Pedestrian_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/ROI/Cyclist_3d', results['roi']['Cyclist_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/ROI/mAP', (results['roi']['Car_3d_all'] + results['roi']['Pedestrian_3d_all'] + results['roi']['Cyclist_3d_all']) / 3, batch_size=1, sync_dist=True)
        
            print("Results: \n"
                f"Entire annotated area: \n"
                f"Car: {results['entire_area']['Car_3d_all']} \n"
                f"Pedestrian: {results['entire_area']['Pedestrian_3d_all']} \n"
                f"Cyclist: {results['entire_area']['Cyclist_3d_all']} \n"
                f"mAP: {(results['entire_area']['Car_3d_all'] + results['entire_area']['Pedestrian_3d_all'] + results['entire_area']['Cyclist_3d_all']) / 3} \n"
                f"Driving corridor area: \n"
                f"Car: {results['roi']['Car_3d_all']} \n"
                f"Pedestrian: {results['roi']['Pedestrian_3d_all']} \n"
                f"Cyclist: {results['roi']['Cyclist_3d_all']} \n"
                f"mAP: {(results['roi']['Car_3d_all'] + results['roi']['Pedestrian_3d_all'] + results['roi']['Cyclist_3d_all']) / 3} \n"
                )
            
        if isinstance(tmp_dir, tempfile.TemporaryDirectory):
            tmp_dir.cleanup() 
        return results
            
        # detection_annotation_file = results_path
        
    def format_results(self, 
                       outputs, 
                       results_save_path=None,
                       pklfile_prefix=None):
        
        det_annos = []
        print('\nConverting prediction to KITTI format')
        print(f'Writing results to {results_save_path}')
        for result in outputs:
            sample_idx = result['sample_idx']
            box_dict = result.get('box_dict', None)
            
            annos = []
            if box_dict is None:
                continue
            
            anno = {                 
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': [],
            }
            
            if len(box_dict['box2d']) > 0:
                box2d_preds = box_dict['box2d']
                box3d_preds_lidar = box_dict['box3d_lidar']
                box3d_location_cam = box_dict['location_cam']
                scores = box_dict['scores']
                label_preds = box_dict['label_preds']
                
                for box3d_lidar, location_cam, box2d, score, label in zip(box3d_preds_lidar, box3d_location_cam, box2d_preds, scores, label_preds):                                      
                    box2d[2:] = np.minimum(box2d[2:], self.img_shape.cpu().numpy()[:2])
                    box2d[:2] = np.maximum(box2d[:2], [0, 0])
                    anno['name'].append(self.class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    #anno['alpha'].append(limit_period(np.arctan2(location_cam[2], location_cam[0]) + box3d_lidar[6] - np.pi/2, offset=0.5, period=2*np.pi))
                    anno['alpha'].append(np.arctan2(location_cam[2], location_cam[0]) + box3d_lidar[6] - np.pi/2)
                    anno['bbox'].append(box2d)
                    anno['dimensions'].append(box3d_lidar[3:6])
                    anno['location'].append(location_cam[:3])
                    anno['rotation_y'].append(box3d_lidar[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)
            
            if results_save_path is not None:
                curr_file = f'{results_save_path}/{sample_idx}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lwh -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][2], dims[idx][1],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)
                
            annos[-1]['sample_idx'] = np.array([sample_idx] * len(annos[-1]['score']), dtype=np.int64)
            det_annos += annos
        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            with open(out, "wb") as f:
                pickle.dump(det_annos, f)
            print(f'Result is saved to {out}.')
        return det_annos
        
    def convert_valid_bboxes(self, box_dict, sample_idx):
        # Convert the predicted bounding boxes to the format required by the evaluation metric
        # This function should be implemented based on the specific requirements of your dataset
        box_preds = box_dict['bboxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        vod_frame_data = FrameDataLoader(kitti_locations=self.vod_kitti_locations, frame_number=sample_idx)
        local_transforms = FrameTransformMatrix(vod_frame_data)
        
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)
        device = box_preds.tensor.device
                
        box_preds_corners_lidar = box_preds.corners
        box_preds_bottom_center_lidar = box_preds.bottom_center # box_preds.gravity_center
        # box_preds_gravity_center_lidar = box_preds.gravity_center
        
        box_preds_corners_img_list = [] 
        box_preds_bottom_center_cam_list =[]
        
        for box_pred_corners, box_pred_bottom_center in zip(box_preds_corners_lidar, box_preds_bottom_center_lidar):
            
            box_pred_corners_lidar_homo= torch.ones((8,4))
            box_pred_corners_lidar_homo[:, :3] = box_pred_corners
            box_pred_corners_cam_homo = homogeneous_transformation(box_pred_corners_lidar_homo, local_transforms.t_camera_lidar)
            box_pred_corners_img = np.dot(box_pred_corners_cam_homo, local_transforms.camera_projection_matrix.T)
            box_pred_corners_img = torch.tensor((box_pred_corners_img[:, :2].T / box_pred_corners_img[:, 2]).T, device=device)
            box_preds_corners_img_list.append(box_pred_corners_img)

            box_pred_bottom_center_lidar_homo = torch.ones((1,4))
            box_pred_bottom_center_lidar_homo[:, :3] = box_pred_bottom_center
            box_pred_bottom_center_cam_homo = homogeneous_transformation(box_pred_bottom_center_lidar_homo, local_transforms.t_camera_lidar)
            box_pred_bottom_center_cam = torch.tensor(box_pred_bottom_center_cam_homo[:,:3])
            box_preds_bottom_center_cam_list.append(box_pred_bottom_center_cam)

        if box_preds_corners_img_list != []:
            box_preds_corners_img = torch.stack(box_preds_corners_img_list, dim=0)
            assert box_preds_bottom_center_cam_list != []
            box_preds_bottom_center_cam = torch.cat(box_preds_bottom_center_cam_list, dim=0).to(device)
        
            minxy = torch.min(box_preds_corners_img, dim=1)[0]
            maxxy = torch.max(box_preds_corners_img, dim=1)[0]
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)

            self.img_shape = self.img_shape.to(device)
            self.pc_range = self.pc_range.to(device)
            
            valid_cam_inds = ((box_2d_preds[:, 0] < self.img_shape[0]) & (box_2d_preds[:, 1] < self.img_shape[1]) & (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
            valid_pcd_inds = ((box_preds.center > self.pc_range[:3]) & (box_preds.center < self.pc_range[3:]))
            valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)
            
            if valid_inds.sum() > 0:
                return dict(
                    box2d=box_2d_preds[valid_inds, :].float().cpu().numpy(),
                    location_cam=box_preds_bottom_center_cam[valid_inds].float().cpu().numpy(),
                    box3d_lidar=box_preds[valid_inds].tensor.float().cpu().numpy(),
                    scores=scores[valid_inds].float().cpu().numpy(),
                    label_preds=labels[valid_inds].cpu().numpy(),
                    sample_idx=sample_idx)
            else:
                return dict(
                    box2d=np.zeros([0, 4]),
                    location_cam=np.zeros([0, 3]),
                    # box3d_camera_corners=np.zeros([0, 7]),
                    box3d_lidar=np.zeros([0, 7]),
                    scores=np.zeros([0]),
                    label_preds=np.zeros([0, 4]),
                    sample_idx=sample_idx)
        else:
            return dict(
                box2d=np.zeros([0, 4]),
                location_cam=np.zeros([0, 3]),
                # box3d_camera_corners=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)
        
        
        
        
        
