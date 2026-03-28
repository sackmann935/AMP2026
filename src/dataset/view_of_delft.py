import os
import numpy as np
from src.model.utils import LiDARInstance3DBoxes

import torch
from torch.utils.data import Dataset

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation

class ViewOfDelft(Dataset):
    CLASSES = ['Car', 
               'Pedestrian', 
               'Cyclist',]
            #    'rider', 
            #    'unused_bicycle', 
            #    'bicycle_rack', 
            #    'human_depiction', 
            #    'moped_or_scooter', 
            #    'motor',
            #    'truck',
            #    'other_ride',
            #    'other_vehicle',
            #    'uncertain_ride'
    
    LABEL_MAPPING = {
        'class': 0, # Describes the type of object: 'Car', 'Pedestrian', 'Cyclist', etc.
        'truncated': 1, # Not used, only there to be compatible with KITTI format.
        'occluded': 2, # Integer (0,1,2) indicating occlusion state 0 = fully visible, 1 = partly occluded 2 = largely occluded.
        'alpha': 3, # Observation angle of object, ranging [-pi..pi]
        'bbox2d': slice(4,8),
        'bbox3d_dimensions': slice(8,11), # 3D object dimensions: height, width, length (in meters).
        'bbox3d_location': slice(11,14), # 3D object location x,y,z in camera coordinates (in meters).
        'bbox3d_rotation': 14, # Rotation around -Z-axis in LiDAR coordinates [-pi..pi].
    }

    RADAR_SOURCE_ALIASES = {
        'radar': 'radar',
        'single': 'radar',
        'single_scan': 'radar',
        'radar_3frames': 'radar_3frames',
        'radar_3_scans': 'radar_3frames',
        '3frames': 'radar_3frames',
        '3scans': 'radar_3frames',
        'radar_5frames': 'radar_5frames',
        'radar_5_scans': 'radar_5frames',
        '5frames': 'radar_5frames',
        '5scans': 'radar_5frames',
    }
    
    def __init__(self, 
                 data_root = 'data/view_of_delft', 
                 sequential_loading=False,
                 split = 'train',
                 radar_source='radar',
                 radar_prioritize_recent=True,
                 radar_z_clip_enabled=False,
                 radar_z_clip_min=-2.5,
                 radar_z_clip_max=2.0,
                 radar_z_jitter_std=0.0,
                 radar_geo_aug_enabled=False,
                 radar_geo_aug_flip_x_prob=0.0,
                 radar_geo_aug_flip_y_prob=0.5,
                 radar_geo_aug_rotation_deg=0.0,
                 radar_geo_aug_scale_min=0.98,
                 radar_geo_aug_scale_max=1.02,
                 include_camera=False,
                 image_mean=(0.485, 0.456, 0.406),
                 image_std=(0.229, 0.224, 0.225)):
        super().__init__()
        
        self.data_root = data_root
        self.include_camera = bool(include_camera)
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"
        self.split = split
        split_file = os.path.join(data_root, 'lidar', 'ImageSets', f'{split}.txt')

        with open(split_file, 'r') as f:
            lines = f.readlines()
            self.sample_list = [line.strip() for line in lines]
        
        self.vod_kitti_locations = KittiLocations(root_dir = data_root)
        self.radar_source = self._normalize_radar_source(radar_source)
        self.radar_prioritize_recent = bool(radar_prioritize_recent)
        self.radar_z_clip_enabled = bool(radar_z_clip_enabled)
        self.radar_z_clip_min = float(radar_z_clip_min)
        self.radar_z_clip_max = float(radar_z_clip_max)
        self.radar_z_jitter_std = float(max(0.0, radar_z_jitter_std))
        self.radar_geo_aug_enabled = bool(radar_geo_aug_enabled)
        self.radar_geo_aug_flip_x_prob = float(np.clip(radar_geo_aug_flip_x_prob, 0.0, 1.0))
        self.radar_geo_aug_flip_y_prob = float(np.clip(radar_geo_aug_flip_y_prob, 0.0, 1.0))
        self.radar_geo_aug_rotation_deg = float(max(0.0, radar_geo_aug_rotation_deg))
        self.radar_geo_aug_scale_min = float(radar_geo_aug_scale_min)
        self.radar_geo_aug_scale_max = float(radar_geo_aug_scale_max)
        self._override_radar_dir()

    def __len__(self):
        return len(self.sample_list)

    def _normalize_radar_source(self, radar_source):
        key = str(radar_source).strip().lower()
        if key not in self.RADAR_SOURCE_ALIASES:
            supported = sorted(self.RADAR_SOURCE_ALIASES.keys())
            raise ValueError(
                f"Unsupported radar_source '{radar_source}'. Supported values: {supported}")
        return self.RADAR_SOURCE_ALIASES[key]

    def _override_radar_dir(self):
        if self.radar_source == 'radar':
            return

        radar_candidates = {
            'radar_3frames': [
                os.path.join(self.data_root, 'radar_3frames', 'training', 'velodyne'),
                os.path.join(self.data_root, 'radar_3_scans', 'training', 'velodyne'),
            ],
            'radar_5frames': [
                os.path.join(self.data_root, 'radar_5frames', 'training', 'velodyne'),
                os.path.join(self.data_root, 'radar_5_scans', 'training', 'velodyne'),
            ],
        }

        for candidate in radar_candidates[self.radar_source]:
            if os.path.isdir(candidate):
                self.vod_kitti_locations.radar_dir = candidate
                return

        searched = radar_candidates[self.radar_source]
        raise FileNotFoundError(
            f"Could not find directory for radar_source='{self.radar_source}'. "
            f"Searched: {searched}")

    def __getitem__(self, idx):
        num_frame = self.sample_list[idx]
        vod_frame_data = FrameDataLoader(kitti_locations=self.vod_kitti_locations,
                                         frame_number=num_frame)
        local_transforms = FrameTransformMatrix(vod_frame_data)
        
        radar_data = vod_frame_data.radar_data
        if radar_data is None:
            raise RuntimeError(
                f"Radar data missing for frame '{num_frame}' using radar_source='{self.radar_source}'.")
        radar_data = np.asarray(radar_data, dtype=np.float32)

        # Temporal radar folders store sweep-id in the last feature column
        # (e.g., -4 ... 0). Keep newest points first so hard voxelization
        # retains recent measurements when max points per voxel is reached.
        if self.radar_prioritize_recent and self.radar_source != 'radar':
            if radar_data.ndim == 2 and radar_data.shape[1] >= 7 and radar_data.shape[0] > 1:
                order = np.argsort(radar_data[:, 6], kind='stable')[::-1]
                radar_data = radar_data[order]

        if radar_data.ndim == 2 and radar_data.shape[1] >= 3 and radar_data.shape[0] > 0:
            if self.split == 'train' and self.radar_z_jitter_std > 0.0:
                radar_data[:, 2] = radar_data[:, 2] + np.random.normal(
                    loc=0.0,
                    scale=self.radar_z_jitter_std,
                    size=radar_data.shape[0]).astype(np.float32)
            if self.radar_z_clip_enabled:
                radar_data[:, 2] = np.clip(
                    radar_data[:, 2],
                    self.radar_z_clip_min,
                    self.radar_z_clip_max)

        
        gt_labels_3d_list = []
        gt_bboxes_3d_list = []
        if self.split != 'test':
            raw_labels = vod_frame_data.raw_labels
            for idx, label in enumerate(raw_labels):
                label = label.split(' ')
                
                if label[self.LABEL_MAPPING['class']] in self.CLASSES: 

                    gt_labels_3d_list.append(int(self.CLASSES.index(label[self.LABEL_MAPPING['class']])))

                    bbox3d_loc_camera = np.array(label[self.LABEL_MAPPING['bbox3d_location']])
                    trans_homo_cam = np.ones((1,4))
                    trans_homo_cam[:, :3] = bbox3d_loc_camera
                    bbox3d_loc_lidar = homogeneous_transformation(trans_homo_cam, local_transforms.t_lidar_camera)
                    
                    bbox3d_locs = np.array(bbox3d_loc_lidar[0,:3], dtype=np.float32)         
                    bbox3d_dims = np.array(label[self.LABEL_MAPPING['bbox3d_dimensions']], dtype=np.float32)[[2, 1, 0]] # hwl -> lwh
                    bbox3d_rot = np.array([label[self.LABEL_MAPPING['bbox3d_rotation']]], dtype=np.float32)
                
                    gt_bboxes_3d_list.append(np.concatenate([bbox3d_locs, bbox3d_dims, bbox3d_rot], axis=0))

        if gt_bboxes_3d_list == []:
            gt_labels_3d = np.zeros((0,), dtype=np.int64)
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
        else:
            gt_labels_3d = np.array(gt_labels_3d_list, dtype=np.int64)
            gt_bboxes_3d = np.stack(gt_bboxes_3d_list, axis=0)

        if self.split == 'train' and self.radar_geo_aug_enabled:
            radar_data, gt_bboxes_3d = self._apply_radar_geo_aug(radar_data, gt_bboxes_3d)

        radar_data = torch.from_numpy(np.ascontiguousarray(radar_data)).to(torch.float32)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0))
        
        gt_labels_3d = torch.tensor(gt_labels_3d)
        
        return dict(
            lidar_data = radar_data,
            gt_labels_3d = gt_labels_3d,
            gt_bboxes_3d = gt_bboxes_3d,
            meta = dict(
                num_frame = num_frame 
            ),
            **self._get_camera_dict(vod_frame_data, local_transforms)
        )

    def _get_camera_dict(self, vod_frame_data, local_transforms):
        if not self.include_camera:
            return {}

        image = vod_frame_data.image
        if image is None:
            raise RuntimeError('Camera image could not be loaded for sample.')

        image = np.array(image, copy=True)
        image = torch.from_numpy(image)
        if image.ndim == 2:
            image = image.unsqueeze(-1).repeat(1, 1, 3)
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = image.to(torch.float32)
        if image.max() > 1.0:
            image = image / 255.0
        image = image.permute(2, 0, 1).contiguous()
        image = (image - self.image_mean) / self.image_std

        camera_projection = torch.tensor(local_transforms.camera_projection_matrix, dtype=torch.float32)
        t_lidar_camera = torch.tensor(local_transforms.t_lidar_camera, dtype=torch.float32)
        return {
            'image': image,
            'camera_projection': camera_projection,
            't_lidar_camera': t_lidar_camera,
        }

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _apply_radar_geo_aug(self, radar_data, gt_bboxes_3d):
        if isinstance(radar_data, torch.Tensor):
            radar_data = radar_data.detach().cpu().numpy()
        if radar_data.ndim != 2 or radar_data.shape[0] == 0 or radar_data.shape[1] < 3:
            return radar_data, gt_bboxes_3d

        boxes = gt_bboxes_3d.copy()
        points = radar_data.copy()

        scale_min = self.radar_geo_aug_scale_min
        scale_max = self.radar_geo_aug_scale_max
        if scale_max < scale_min:
            scale_min, scale_max = scale_max, scale_min
        if scale_max > 0.0 and scale_max > scale_min:
            scale = float(np.random.uniform(scale_min, scale_max))
            points[:, :3] *= scale
            if boxes.shape[0] > 0:
                boxes[:, :6] *= scale

        rot_deg = self.radar_geo_aug_rotation_deg
        if rot_deg > 0.0:
            angle = float(np.deg2rad(np.random.uniform(-rot_deg, rot_deg)))
            sin_a, cos_a = np.sin(angle), np.cos(angle)
            x = points[:, 0].copy()
            y = points[:, 1].copy()
            points[:, 0] = cos_a * x - sin_a * y
            points[:, 1] = sin_a * x + cos_a * y
            if boxes.shape[0] > 0:
                bx = boxes[:, 0].copy()
                by = boxes[:, 1].copy()
                boxes[:, 0] = cos_a * bx - sin_a * by
                boxes[:, 1] = sin_a * bx + cos_a * by
                boxes[:, 6] = self._wrap_to_pi(boxes[:, 6] - angle)

        if np.random.rand() < self.radar_geo_aug_flip_x_prob:
            points[:, 0] = -points[:, 0]
            if boxes.shape[0] > 0:
                boxes[:, 0] = -boxes[:, 0]
                boxes[:, 6] = self._wrap_to_pi(-boxes[:, 6])

        if np.random.rand() < self.radar_geo_aug_flip_y_prob:
            points[:, 1] = -points[:, 1]
            if boxes.shape[0] > 0:
                boxes[:, 1] = -boxes[:, 1]
                boxes[:, 6] = self._wrap_to_pi(-boxes[:, 6] - np.pi)

        return points.astype(np.float32, copy=False), boxes.astype(np.float32, copy=False)
    

        
