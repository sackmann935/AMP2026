# Advanced Machine Perception Final Assignment

## Goal
Improve CenterPoint-style 3D detection on View of Delft under strict resource limits (single 10GB MIG GPU, low CPU RAM), targeting higher ROI mAP for `Car`, `Pedestrian`, and `Cyclist`.

## Current Implementation

### Radar branch
- Hard voxelization (`src/ops/voxelize.py`) + PillarFeatureNet (`src/model/voxel_encoders/pillar_encoder.py`).
- Middle encoder options:
  - `hard` scatter
  - `gaussian_soft` scatter (`src/model/middle_encoders/gaussian_soft_scatter.py`)
- Radar input source switch:
  - `radar` (single scan)
  - `radar_3frames`
  - `radar_5frames`

### Temporal radar handling (new)
Implemented to make multi-frame aggregation effective:
- **Recency-first point ordering** for temporal radar sources in dataset loader (`src/dataset/view_of_delft.py`):
  - For `radar_3frames` / `radar_5frames`, points are sorted by sweep-id descending (newest first).
  - This prevents old sweeps from dominating voxel slots.
- **Higher voxel point cap for temporal runs** in training (`src/tools/train.py`):
  - If `radar_source != radar`, `model.pts_voxel_layer.max_num_points` is set from `temporal_max_num_points` (default `10`).
  - Base model config still keeps `5` for single-scan baseline.

### Camera + fusion branch
- Camera BEV lift encoder (`src/model/camera/image_bev_gaussian.py`).
- Lift mode options include `topk_chunked`.
- Fusion options include `cmx_lite` (`src/model/fusion/cmx_lite.py`).

### Head / optimization
- CenterPoint head with 3 tasks (Car/Pedestrian/Cyclist).
- AdamW optimizer.
- Optional cosine warmup scheduler.
- Optional regularization toggles:
  - `groupnorm`
  - `freeze_bn`
  - `accumulate_grad_batches`
- Checkpoint monitor defaults to `validation/ROI/mAP`.

## Config Surface (important)

### Train config (`src/config/train.yaml`)
- `radar_source`: `radar | radar_3frames | radar_5frames`
- `radar_prioritize_recent`: default `true`
- `temporal_max_num_points`: default `10` (used for temporal radar sources)
- `accumulate_grad_batches`, `sync_bn`, checkpoint/eval threshold options

### Eval/test configs
- `src/config/eval.yaml` and `src/config/test.yaml` include:
  - `radar_source`
  - `radar_prioritize_recent`

## SLURM Profiles
Main launcher: `src/tools/slurm_train.sh`

Useful profiles:
- Reference / ablation baselines:
  - `ablate_accum2_no_cosine`
  - `ablate_bnfreeze_e_no_cosine`
  - `ablate_groupnorm_g16`
  - `ped_accum2_no_cosine_minr1_thr003`
- Temporal radar ablations (ped-best settings):
  - `radar_single_pedbest`
  - `radar_3f_pedbest`
  - `radar_5f_pedbest`
- Temporal radar ablations (groupnorm settings):
  - `radar_single_gn`
  - `radar_3f_gn`
  - `radar_5f_gn`

For temporal profiles (`radar_3f_*`, `radar_5f_*`), SLURM now explicitly sets:
- `radar_prioritize_recent=true`
- `temporal_max_num_points=10`

## How To Run

### Local train
```bash
python -u src/tools/train.py \
  exp_id=my_run \
  epochs=20 batch_size=2 num_workers=2 \
  radar_source=radar_5frames \
  model.middle_encoder.type=gaussian_soft \
  model.fusion.enabled=true model.fusion.type=cmx_lite \
  model.camera.lift_mode=topk_chunked
```

### SLURM
```bash
PROFILE=radar_5f_pedbest bash src/tools/slurm_train.sh
```

### Eval / test
```bash
python src/tools/eval.py checkpoint_path=/path/to.ckpt radar_source=radar_5frames
python src/tools/test.py checkpoint_path=/path/to.ckpt radar_source=radar_5frames
```

## Current Known Issues / Risks
- Some historical runs were interrupted by SLURM `SIGTERM`, so compare best checkpoints, not only final epoch lines.
- W&B `output.log` in some runs can be truncated; prefer SLURM logs or full run artifacts when analyzing failures.
- Camera image loading emits a non-writable array warning in older logs; dataset now defensively copies image arrays.

## TODO (Prioritized)
1. Run clean temporal A/B/C with fixed seed:
   - `radar` vs `radar_3frames` vs `radar_5frames`
   - keep all other hyperparameters identical.
2. Add an ablation toggle for temporal point ordering:
   - compare `radar_prioritize_recent=true` vs `false` on 5-frame.
3. Sweep temporal voxel cap on 5-frame:
   - `temporal_max_num_points`: `8`, `10`, `12`, `15`.
4. Separate radar-only and radar+camera temporal tests:
   - detect whether gains are lost inside fusion.
5. Add temporal-aware point sampling (optional):
   - reserve per-voxel quota for `t=0` before filling with older sweeps.
6. Stabilize Cyclist class:
   - evaluate class-aware thresholds or class-aware NMS post-processing.

## Next Ideas
- Add lightweight temporal feature channel engineering (e.g., normalized age embedding) before PFN.
- Consider dynamic voxelization for temporal runs to reduce order-sensitive clipping effects.
- Add automated summary script that reads all run logs and reports best ROI mAP + config deltas.
