# Advanced Machine Perception Final Assignment

## Task
This project targets the **AMP final assignment**: improve a baseline 3D detector on the **View of Delft** dataset using **radar and/or monocular camera** inputs.

Constraints from the assignment:
- Improve over the provided radar-based CenterPoint baseline.
- Use only radar and camera as perception inputs (no LiDAR input to the model).
- Evaluate on KITTI-style 3D metrics (Car, Pedestrian, Cyclist) and report mAP improvements.
- Stay within limited resources (single GPU, strict VRAM/CPU RAM caps).

## Current Solution
The current codebase is a CenterPoint-style detector with radar backbone plus optional camera fusion.

Implemented components:
- Radar pipeline:
  - Voxelization + pillar feature encoder.
  - Two middle encoders: `hard` scatter and custom `gaussian_soft` scatter.
- Camera BEV pipeline:
  - `ImageBEVGaussianEncoder` lifts monocular image features to BEV.
  - Lift modes: `expected`, `naive_dense`, `topk_chunked`.
- Fusion options:
  - `none`, `add`, `concat_1x1`, `gated`.
  - `cmx_lite` for multi-scale cross-modal fusion (radar + camera backbone features).
- Detector head:
  - CenterPoint head for 3 classes (`Car`, `Pedestrian`, `Cyclist`).
- Training stack:
  - Hydra configs + PyTorch Lightning training loop.
  - AdamW optimizer with step-wise linear warmup + cosine decay scheduler.
  - Validation logging and configurable checkpoint monitor (default: `validation/ROI/mAP`).

## Current Experiment Profile
Main cluster profile in `src/tools/slurm_train.sh`:
- `PROFILE=ped_bnfreeze_no_cosine`

This profile currently uses:
- `model.middle_encoder.type=gaussian_soft`
- `model.fusion.enabled=true`
- `model.fusion.type=cmx_lite`
- `model.camera.lift_mode=topk_chunked`
- `model.regularization.freeze_bn.enabled=true`
- `model.optimizer.scheduler.enabled=false`
- `epochs=20`, `batch_size=2`

## Regularization And Memory-Constrained Training Switches
Current regularization implemented in code/config:
- Optimizer: `AdamW` with non-zero `weight_decay` (`src/config/model/centerpoint_radar.yaml`).
- LR schedule: linear warmup + cosine decay (`src/model/detector/centerpoint.py`, configured under `optimizer.scheduler`).
- Checkpoint selection: top-k checkpoints tracked by configurable monitor (default `validation/ROI/mAP`).

New ablation switches (all optional and independently toggleable):
- `sync_bn=false` by default in single-GPU training (`src/config/train.yaml`).
- `accumulate_grad_batches` to emulate larger effective batch sizes without higher VRAM (`src/config/train.yaml`, `src/tools/train.py`).
- `checkpoint_monitor` / `checkpoint_mode` to select checkpoints by the target metric (default monitor is ROI mAP).
- `eval_score_threshold` to override evaluation-time score filtering without deep Hydra path overrides.
- `model.regularization.norm_mode`:
  - `batchnorm` (default)
  - `groupnorm` (replaces BatchNorm layers at model init)
- `model.regularization.freeze_bn`:
  - `enabled`
  - `freeze_epoch`
  - `freeze_affine`

Defaults preserve the current behavior unless explicitly overridden:
- `norm_mode=batchnorm`
- `freeze_bn.enabled=false`
- `accumulate_grad_batches=1`
- `checkpoint_monitor=validation/ROI/mAP`

Data handling fix applied:
- Empty-label frames no longer inject fake Car ground-truth boxes; they now keep empty GT tensors in `src/dataset/view_of_delft.py`.

## Project Layout
- `src/tools/train.py` - training entrypoint
- `src/tools/eval.py` - validation from checkpoint
- `src/tools/test.py` - test inference / prediction export
- `src/tools/slurm_train.sh` - DelftBlue SLURM launcher with profiles
- `src/tools/smoke_train.sh` - short GPU smoke test
- `src/config/` - Hydra train/eval/test/model configs
- `src/model/` - detector, middle encoders, camera lift, fusion modules
- `src/dataset/` - View of Delft dataset loader and collate function

## Data and Environment
Expected data location:
- `data/view_of_delft`

Expected environment:
- `conda activate amp`

Note:
- Large artifacts are intentionally excluded from Git (`data/`, `outputs/`, caches, logs, checkpoints).

## How To Run
From repository root:

```bash
# quick sanity check on GPU
bash src/tools/smoke_train.sh

# train (default Hydra train config)
python -u src/tools/train.py

# train with overrides (example: current multimodal setup)
python -u src/tools/train.py \
  exp_id=gaussian_topk_chunked_regularized_cosine \
  epochs=20 batch_size=2 num_workers=2 \
  model.middle_encoder.type=gaussian_soft \
  model.fusion.enabled=true model.fusion.type=cmx_lite \
  model.camera.lift_mode=topk_chunked

# evaluate validation split from checkpoint
python src/tools/eval.py checkpoint_path=/path/to/checkpoint.ckpt

# run test split inference for leaderboard export
python src/tools/test.py checkpoint_path=/path/to/checkpoint.ckpt
```

Ablation examples for the new switches:

```bash
# 1) Only gradient accumulation
python -u src/tools/train.py accumulate_grad_batches=2

# 2) Only BN freeze (keep BatchNorm)
python -u src/tools/train.py \
  model.regularization.freeze_bn.enabled=true \
  model.regularization.freeze_bn.freeze_epoch=1

# 3) Only GroupNorm switch
python -u src/tools/train.py \
  model.regularization.norm_mode=groupnorm \
  model.regularization.group_norm_groups=16

# 4) Combine GroupNorm + accumulation
python -u src/tools/train.py \
  model.regularization.norm_mode=groupnorm \
  accumulate_grad_batches=2

# 5) ROI-oriented checkpointing + lower eval threshold
python -u src/tools/train.py \
  checkpoint_monitor=validation/ROI/mAP \
  eval_score_threshold=0.03

# 6) Ped-focused target sharpening
python -u src/tools/train.py \
  model.head.train_cfg.min_radius=1 \
  eval_score_threshold=0.03
```

For cluster runs, use:

```bash
PROFILE=ped_bnfreeze_no_cosine bash src/tools/slurm_train.sh
PROFILE=ped_bnfreeze_no_cosine_thr003 bash src/tools/slurm_train.sh
PROFILE=ped_bnfreeze_no_cosine_minr1_thr003 bash src/tools/slurm_train.sh
PROFILE=ped_accum2_no_cosine_minr1_thr003 bash src/tools/slurm_train.sh
```

## Current TODOs
Primary TODO: **improve generalization under strict memory constraints**.

Planned next steps:
- Tune warmup+cosine scheduler hyperparameters (`warmup_epochs`, `min_lr_ratio`) together with `lr` and `weight_decay`.
- Run clean ablations for:
  - BN freeze only
  - GroupNorm only
  - gradient accumulation only
  - combinations of the above
- Add stronger data augmentation/perturbation for radar and camera branches.
- Add early stopping based on `validation/ROI/mAP` to stop after the peak epoch.
- Track and compare: training loss vs validation loss, per-class validation AP, ROI mAP vs entire-area mAP.

Optional TODOs (not implemented yet):
- Add mixed precision mode toggle (`bf16-mixed` or `16-mixed`) in trainer config.
- Add optional dropout/drop-path blocks in camera/fusion/head modules.
- Add configurable BN freeze policy by global step (not only epoch).
- Add optional EMA of model weights for validation/evaluation.
- Add automatic best-checkpoint evaluation script to avoid accidental `last.ckpt` reporting.

## Short Status
- Baseline training/eval/test scripts are in place and working.
- Multimodal Gaussian + CMX-lite path is implemented.
- Norm/BN/accumulation toggles are now available for memory-constrained ablations.
