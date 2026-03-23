def collate_vod_batch(batch):
    pts_list = []
    gt_labels_3d_list = []
    gt_bboxes_3d_list = []
    meta_list = []
    image_list = []
    camera_projection_list = []
    t_lidar_camera_list = []
    has_camera = 'image' in batch[0]
    for idx, sample in enumerate(batch):
        pts_list.append(sample['lidar_data'])
        gt_labels_3d_list.append(sample['gt_labels_3d'])
        gt_bboxes_3d_list.append(sample['gt_bboxes_3d'])
        meta_list.append(sample['meta'])
        if has_camera:
            image_list.append(sample['image'])
            camera_projection_list.append(sample['camera_projection'])
            t_lidar_camera_list.append(sample['t_lidar_camera'])

    batch_dict = dict(
        pts = pts_list,
        gt_labels_3d = gt_labels_3d_list,
        gt_bboxes_3d = gt_bboxes_3d_list,
        metas = meta_list
    )
    if has_camera:
        import torch
        batch_dict['image'] = torch.stack(image_list, dim=0)
        batch_dict['camera_projection'] = torch.stack(camera_projection_list, dim=0)
        batch_dict['t_lidar_camera'] = torch.stack(t_lidar_camera_list, dim=0)
    return batch_dict
    
