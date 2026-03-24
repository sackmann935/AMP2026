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

@hydra.main(version_base=None, config_path='../config', config_name='train')    
def train(cfg: DictConfig)-> None:
    L.seed_everything(cfg.seed, workers=True)

    fusion_enabled = bool(OmegaConf.select(cfg, 'model.fusion.enabled', default=False))
    
    train_dataset = ViewOfDelft(data_root=cfg.data_root, split='train', include_camera=fusion_enabled)
    val_dataset = ViewOfDelft(data_root=cfg.data_root, split='val', include_camera=fusion_enabled)
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=cfg.batch_size, 
                                  num_workers=cfg.num_workers, 
                                  shuffle=True,
                                  collate_fn=collate_vod_batch)
    val_num_workers = int(OmegaConf.select(cfg, 'val_num_workers', default=0))
    accumulate_grad_batches = int(OmegaConf.select(cfg, 'accumulate_grad_batches', default=1))
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=1, 
                                num_workers=val_num_workers,
                                shuffle=False,
                                collate_fn=collate_vod_batch)
    model = CenterPoint(cfg.model)
    callbacks = [
        ModelCheckpoint(
            dirpath=osp.join(cfg.output_dir, "checkpoints"),
            filename='ep{epoch}-'+cfg.exp_id,
            save_last=True,
            monitor='validation/entire_area/mAP',
            mode='max',
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
        enable_model_summary= True,
    )
    
    trainer.fit(model, 
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=cfg.checkpoint_path)
    # wandb.finish()
    
if __name__ == '__main__':
    train()
