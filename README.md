# Advanced Machine Perception Final Assignment

## Task
This project targets the **AMP final assignment**: improve a baseline 3D detector on the **View of Delft** dataset using **radar and/or monocular camera** inputs.

Constraints from the assignment:
- Improve over the provided radar-based CenterPoint baseline.
- Use only radar and camera as perception inputs (no LiDAR input to the model).
- Evaluate on KITTI-style 3D metrics (Car, Pedestrian, Cyclist) and report mAP improvements.

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
  - Validation logging and checkpointing based on `validation/entire_area/mAP`.

## Current Experiment Profile
Main cluster profile in `src/tools/slurm_train.sh`:
- `PROFILE=gaussian_topk_chunked_e20_s06_regularized`

This profile currently uses:
- `model.middle_encoder.type=gaussian_soft`
- `model.fusion.enabled=true`
- `model.fusion.type=cmx_lite`
- `model.camera.lift_mode=topk_chunked`
- `epochs=20`, `batch_size=2`

## Current Regularization
Current regularization implemented in code/config:
- Optimizer: `AdamW` with non-zero `weight_decay` (`src/config/model/centerpoint_radar.yaml`).
- LR schedule: linear warmup + cosine decay (implemented in `src/model/detector/centerpoint.py`, configured under `optimizer.scheduler`).
- Checkpoint regularization-by-selection: top-k model checkpoints are tracked by `validation/entire_area/mAP`.

Current gaps (not yet implemented):
- No explicit dropout/drop-path in camera/fusion/head blocks.
- No data augmentation pipeline in `src/dataset/view_of_delft.py`.
- No early stopping callback yet.

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
  exp_id=gaussian_topk_chunked_regularized \
  epochs=20 batch_size=2 num_workers=2 \
  model.middle_encoder.type=gaussian_soft \
  model.fusion.enabled=true model.fusion.type=cmx_lite \
  model.camera.lift_mode=topk_chunked

# evaluate validation split from checkpoint
python src/tools/eval.py checkpoint_path=/path/to/checkpoint.ckpt

# run test split inference for leaderboard export
python src/tools/test.py checkpoint_path=/path/to/checkpoint.ckpt
```

For cluster runs, use:

```bash
PROFILE=gaussian_topk_chunked_e20_s06_regularized bash src/tools/slurm_train.sh
```

## Current TODOs
Primary TODO: **improve generalization after adding scheduler regularization**.

Planned next steps:
- Keep warmup+cosine scheduler and tune its hyperparameters (`warmup_epochs`, `min_lr_ratio`) together with `lr` and `weight_decay`.
- Add explicit model regularization (dropout/drop-path candidates in camera encoder, fusion blocks, and head).
- Add stronger data augmentation/perturbation for radar and camera branches.
- Add early-stopping based on `validation/entire_area/mAP` to stop after the peak epoch.
- Track and compare: training loss vs validation loss, per-class validation AP, ROI mAP vs entire-area mAP.

## Short Status
- Baseline training/eval/test scripts are in place and working.
- Multimodal Gaussian + CMX-lite path is implemented.
- Next milestone is robust regularization and ablation to improve generalization.
