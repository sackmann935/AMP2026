import os
import tempfile
import pickle
import math
from datetime import datetime

import numpy as np 
import torch
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
from src.model.camera import ImageBEVGaussianEncoder
from src.model.fusion import BEVFeatureFusion, CMXLiteFuser

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
        camera_encoder_config = config.get('camera', {})
        fusion_config = config.get('fusion', {})
        backbone_config = config.get('backbone', None)
        neck_config = config.get('neck', None)
        head_config = config.get('head', None)
        
        self.voxel_layer = Voxelization(**voxel_layer_config)
        self.voxel_encoder = PillarFeatureNet(**voxel_encoder_config)
        middle_encoder_cfg = dict(middle_encoder_config)
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

        self.fusion_enabled = bool(fusion_config.get('enabled', False))
        self.fusion_type = str(fusion_config.get('type', 'none'))
        self.camera_encoder = None
        self.camera_backbone = None
        self.bev_fuser = None
        if self.fusion_enabled:
            camera_cfg = dict(camera_encoder_config)
            camera_debug_cfg = camera_cfg.pop('debug', None)
            self.camera_encoder = ImageBEVGaussianEncoder(
                out_channels=middle_encoder_cfg['in_channels'],
                output_shape=middle_encoder_cfg['output_shape'],
                point_cloud_range=config.get('point_cloud_range', None),
                voxel_size=config.get('voxel_size', None),
                debug=camera_debug_cfg,
                **camera_cfg,
            )
            if self.fusion_type == 'cmx_lite':
                self.camera_backbone = SECOND(**backbone_config)
                self.bev_fuser = CMXLiteFuser(
                    in_channels=backbone_config['out_channels'],
                    num_heads=fusion_config.get('cmx_num_heads', [2, 4, 8]),
                    reduction=fusion_config.get('cmx_reduction', 1))
            else:
                self.bev_fuser = BEVFeatureFusion(
                    channels=middle_encoder_cfg['in_channels'],
                    fusion_type=self.fusion_type)

        self.backbone = SECOND(**backbone_config)
        self.neck = SECONDFPN(**neck_config)
        self.head = CenterHead(**head_config)
        
        self.optimizer_config = config.get('optimizer', None)
        
        self.vod_kitti_locations = KittiLocations(root_dir = self.data_root, 
                                     output_dir= self.output_dir,
                                     frame_set_path='',
                                     pred_dir='',)
        self.inference_mode = config.get('inference_mode', 'val')
        self.save_results = config.get('save_preds_results', False)
        self.val_results_list =[]

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
    
    def _model_forward(self, batch):
        pts_data = batch['pts']

        voxel_dict = self.voxelize(pts_data)
    
        voxels = voxel_dict['voxels']
        num_points = voxel_dict['num_points']
        coors = voxel_dict['coors']
    
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        bs = coors[-1,0].item() + 1
        radar_bev_feats = self.middle_encoder(voxel_features, coors, bs)

        camera_bev_feats = None
        if self.fusion_enabled:
            required_keys = ('image', 'camera_projection', 't_lidar_camera')
            missing = [k for k in required_keys if k not in batch]
            if missing:
                raise ValueError(
                    f'Fusion is enabled but camera inputs are missing: {missing}. '
                    'Ensure dataset is created with include_camera=True.')
            camera_bev_feats = self.camera_encoder(
                batch['image'].to(device=radar_bev_feats.device),
                batch['camera_projection'].to(device=radar_bev_feats.device),
                batch['t_lidar_camera'].to(device=radar_bev_feats.device))

        if self.fusion_enabled and self.fusion_type == 'cmx_lite':
            radar_feats = self.backbone(radar_bev_feats)
            camera_feats = self.camera_backbone(camera_bev_feats)
            backbone_feats = self.bev_fuser(radar_feats, camera_feats)
        else:
            if self.fusion_enabled:
                bev_feats = self.bev_fuser(radar_bev_feats, camera_bev_feats)
            else:
                bev_feats = radar_bev_feats
            backbone_feats = self.backbone(bev_feats)

        neck_feats = self.neck(backbone_feats)
        ret_dict = self.head(neck_feats)
        return ret_dict
    
    def training_step(self, batch, batch_idx):
        gt_label_3d = batch['gt_labels_3d']
        gt_bboxes_3d = batch['gt_bboxes_3d']
        
        ret_dict = self._model_forward(batch)
        self._log_module_debug_stats(stage='train', module=self.middle_encoder, module_name='middle_encoder')
        self._log_module_debug_stats(stage='train', module=self.camera_encoder, module_name='camera_encoder')
        self._log_module_debug_stats(stage='train', module=self.bev_fuser, module_name='fusion')
        loss_input = [gt_bboxes_3d, gt_label_3d, ret_dict]
        
        losses = self.head.loss(*loss_input)
        
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

        return loss
    
    def configure_optimizers(self):
        optimizer_cfg = dict(self.optimizer_config or {})
        scheduler_cfg = optimizer_cfg.pop('scheduler', None)
        optimizer = torch.optim.AdamW(self.parameters(), **optimizer_cfg)

        if not scheduler_cfg or not bool(scheduler_cfg.get('enabled', False)):
            return optimizer

        total_steps = int(getattr(self.trainer, 'estimated_stepping_batches', 0) or 0)
        if total_steps <= 0:
            return optimizer

        warmup_steps = int(scheduler_cfg.get('warmup_steps', 0) or 0)
        if warmup_steps <= 0:
            warmup_epochs = float(scheduler_cfg.get('warmup_epochs', 0.0) or 0.0)
            if warmup_epochs > 0:
                steps_per_epoch = int(getattr(self.trainer, 'num_training_batches', 0) or 0)
                warmup_steps = int(round(warmup_epochs * steps_per_epoch))

        warmup_steps = max(0, min(warmup_steps, max(0, total_steps - 1)))
        decay_steps = max(1, total_steps - warmup_steps)

        min_lr_ratio = float(scheduler_cfg.get('min_lr_ratio', 0.05))
        min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return max(float(step + 1) / float(warmup_steps), 1e-8)

            progress = float(step - warmup_steps) / float(decay_steps)
            progress = max(0.0, min(1.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'name': 'lr-cosine-warmup',
            },
        }
    
    
    def validation_step(self, batch, batch_idx):
        assert len(batch['pts']) == 1, 'Batch size should be 1 for validation'
        metas = batch['metas']
        gt_label_3d = batch['gt_labels_3d']
        gt_bboxes_3d = batch['gt_bboxes_3d']
        
        ret_dict = self._model_forward(batch)
        self._log_module_debug_stats(stage='validation', module=self.middle_encoder, module_name='middle_encoder')
        self._log_module_debug_stats(stage='validation', module=self.camera_encoder, module_name='camera_encoder')
        self._log_module_debug_stats(stage='validation', module=self.bev_fuser, module_name='fusion')
        loss_input = [gt_bboxes_3d, gt_label_3d, ret_dict]
        
        bbox_list = self.head.get_bboxes(ret_dict, img_metas=metas)
        
        bbox_results = [
            dict(bboxes_3d = bboxes, 
                 scores_3d = scores, 
                 labels_3d = labels)
            for bboxes, scores, labels in bbox_list
        ]

        losses = self.head.loss(*loss_input)
        
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
                    box2d=box_2d_preds[valid_inds, :].cpu().numpy(),
                    location_cam=box_preds_bottom_center_cam[valid_inds].cpu().numpy(),
                    box3d_lidar=box_preds[valid_inds].tensor.cpu().numpy(),
                    scores=scores[valid_inds].cpu().numpy(),
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
        
        
        
        
        
