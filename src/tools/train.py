import os
import sys
import os.path as osp
root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)
    
import hydra
import wandb
from  omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import torch
from torch.utils.data import DataLoader
from src.model.detector import CenterPoint
from src.dataset import ViewOfDelft, collate_vod_batch


def _load_warm_start_weights(model, checkpoint_path, strict=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=bool(strict))
    missing_count = len(missing)
    unexpected_count = len(unexpected)
    print(
        f"[warm-start] Loaded weights from {checkpoint_path} "
        f"(strict={bool(strict)}). missing={missing_count}, unexpected={unexpected_count}")
    if missing_count > 0:
        preview = ', '.join(missing[:8])
        if missing_count > 8:
            preview = preview + ', ...'
        print(f"[warm-start] missing keys preview: {preview}")
    if unexpected_count > 0:
        preview = ', '.join(unexpected[:8])
        if unexpected_count > 8:
            preview = preview + ', ...'
        print(f"[warm-start] unexpected keys preview: {preview}")


@hydra.main(version_base=None, config_path='../config', config_name='train')    
def train(cfg: DictConfig)-> None:
    L.seed_everything(cfg.seed, workers=True)
    matmul_precision = str(OmegaConf.select(cfg, 'matmul_precision', default='high')).lower()
    if matmul_precision in ('highest', 'high', 'medium'):
        torch.set_float32_matmul_precision(matmul_precision)
    benchmark = bool(OmegaConf.select(cfg, 'benchmark', default=True))
    torch.backends.cudnn.benchmark = benchmark

    fusion_enabled = bool(OmegaConf.select(cfg, 'model.fusion.enabled', default=False))
    if fusion_enabled:
        raise ValueError('Fusion has been removed. Use radar-only configuration.')
    camera_rescue_enabled = bool(OmegaConf.select(cfg, 'model.camera_rescue.enabled', default=False))
    radar_source = str(OmegaConf.select(cfg, 'radar_source', default='radar'))
    radar_prioritize_recent = bool(OmegaConf.select(cfg, 'radar_prioritize_recent', default=True))
    radar_z_clip_enabled = bool(OmegaConf.select(cfg, 'radar_z_clip.enabled', default=False))
    radar_z_clip_min = float(OmegaConf.select(cfg, 'radar_z_clip.min', default=-2.5))
    radar_z_clip_max = float(OmegaConf.select(cfg, 'radar_z_clip.max', default=2.0))
    radar_z_jitter_std = float(OmegaConf.select(cfg, 'radar_z_jitter_std', default=0.0))
    radar_geo_aug_enabled = bool(OmegaConf.select(cfg, 'radar_geo_aug.enabled', default=False))
    radar_geo_aug_flip_x_prob = float(OmegaConf.select(cfg, 'radar_geo_aug.flip_x_prob', default=0.5))
    radar_geo_aug_flip_y_prob = float(OmegaConf.select(cfg, 'radar_geo_aug.flip_y_prob', default=0.0))
    radar_geo_aug_rotation_deg = float(OmegaConf.select(cfg, 'radar_geo_aug.rotation_deg', default=0.0))
    radar_geo_aug_scale_min = float(OmegaConf.select(cfg, 'radar_geo_aug.scale_min', default=0.95))
    radar_geo_aug_scale_max = float(OmegaConf.select(cfg, 'radar_geo_aug.scale_max', default=1.05))
    temporal_max_num_points = OmegaConf.select(cfg, 'temporal_max_num_points', default=10)
    checkpoint_monitor = str(OmegaConf.select(cfg, 'checkpoint_monitor', default='validation/ROI/mAP'))
    checkpoint_mode = str(OmegaConf.select(cfg, 'checkpoint_mode', default='max'))
    eval_score_threshold = OmegaConf.select(cfg, 'eval_score_threshold', default=None)
    if eval_score_threshold is not None:
        cfg.model.head.test_cfg.score_threshold = float(eval_score_threshold)
    if radar_source != 'radar' and temporal_max_num_points is not None:
        cfg.model.pts_voxel_layer.max_num_points = int(temporal_max_num_points)
        print(
            f"[temporal-radar] radar_source={radar_source} -> "
            f"pts_voxel_layer.max_num_points={int(temporal_max_num_points)}")
    
    train_dataset = ViewOfDelft(
        data_root=cfg.data_root,
        split='train',
        radar_source=radar_source,
        radar_prioritize_recent=radar_prioritize_recent,
        radar_z_clip_enabled=radar_z_clip_enabled,
        radar_z_clip_min=radar_z_clip_min,
        radar_z_clip_max=radar_z_clip_max,
        radar_z_jitter_std=radar_z_jitter_std,
        radar_geo_aug_enabled=radar_geo_aug_enabled and (not fusion_enabled),
        radar_geo_aug_flip_x_prob=radar_geo_aug_flip_x_prob,
        radar_geo_aug_flip_y_prob=radar_geo_aug_flip_y_prob,
        radar_geo_aug_rotation_deg=radar_geo_aug_rotation_deg,
        radar_geo_aug_scale_min=radar_geo_aug_scale_min,
        radar_geo_aug_scale_max=radar_geo_aug_scale_max,
        include_camera=camera_rescue_enabled)
    val_dataset = ViewOfDelft(
        data_root=cfg.data_root,
        split='val',
        radar_source=radar_source,
        radar_prioritize_recent=radar_prioritize_recent,
        radar_z_clip_enabled=radar_z_clip_enabled,
        radar_z_clip_min=radar_z_clip_min,
        radar_z_clip_max=radar_z_clip_max,
        radar_z_jitter_std=0.0,
        include_camera=camera_rescue_enabled)
    
    num_workers = int(OmegaConf.select(cfg, 'num_workers', default=0))
    val_num_workers = int(OmegaConf.select(cfg, 'val_num_workers', default=0))
    pin_memory = bool(OmegaConf.select(cfg, 'pin_memory', default=True))
    persistent_workers = bool(OmegaConf.select(cfg, 'persistent_workers', default=True))
    prefetch_factor = OmegaConf.select(cfg, 'prefetch_factor', default=2)
    accumulate_grad_batches = int(OmegaConf.select(cfg, 'accumulate_grad_batches', default=1))
    train_loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_vod_batch,
        pin_memory=pin_memory)
    if num_workers > 0:
        train_loader_kwargs['persistent_workers'] = persistent_workers
        if prefetch_factor is not None:
            train_loader_kwargs['prefetch_factor'] = int(prefetch_factor)
    train_dataloader = DataLoader(train_dataset, **train_loader_kwargs)

    val_loader_kwargs = dict(
        batch_size=1,
        num_workers=val_num_workers,
        shuffle=False,
        collate_fn=collate_vod_batch,
        pin_memory=pin_memory)
    if val_num_workers > 0:
        val_loader_kwargs['persistent_workers'] = persistent_workers
        if prefetch_factor is not None:
            val_loader_kwargs['prefetch_factor'] = int(prefetch_factor)
    val_dataloader = DataLoader(val_dataset, **val_loader_kwargs)
    model = CenterPoint(cfg.model)
    warm_start_checkpoint_path = str(OmegaConf.select(cfg, 'warm_start_checkpoint_path', default='') or '').strip()
    warm_start_strict = bool(OmegaConf.select(cfg, 'warm_start_strict', default=False))
    if warm_start_checkpoint_path:
        if not osp.isfile(warm_start_checkpoint_path):
            raise FileNotFoundError(f'warm_start_checkpoint_path not found: {warm_start_checkpoint_path}')
        _load_warm_start_weights(
            model=model,
            checkpoint_path=warm_start_checkpoint_path,
            strict=warm_start_strict)

    callbacks = [
        ModelCheckpoint(
            dirpath=osp.join(cfg.output_dir, "checkpoints"),
            filename='ep{epoch}-'+cfg.exp_id,
            save_last=True,
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
            auto_insert_metric_name=False,
            save_top_k=cfg.save_top_model,
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]
    logger = WandbLogger(
        save_dir=osp.join(cfg.output_dir, 'wandb_logs'),
        project='amp',
        name=cfg.exp_id,
        log_model=False,
    )
    logger.watch(model, log_graph=False)
        

    
    trainer = L.Trainer(
        logger=logger,
        log_every_n_steps=cfg.log_every,
        accelerator="gpu",
        devices=cfg.gpus,
        check_val_every_n_epoch=cfg.val_every,
        strategy="auto",
        callbacks=callbacks,
        max_epochs=cfg.epochs,
        sync_batchnorm=cfg.sync_bn,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=OmegaConf.select(cfg, 'precision', default='32-true'),
        num_sanity_val_steps=int(OmegaConf.select(cfg, 'num_sanity_val_steps', default=0)),
        benchmark=benchmark,
        enable_model_summary= True,
    )
    
    resume_ckpt_path = OmegaConf.select(cfg, 'checkpoint_path', default=None)
    if resume_ckpt_path in ('', None):
        resume_ckpt_path = None

    trainer.fit(model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                    ckpt_path=resume_ckpt_path)
    
if __name__ == '__main__':
    train()
