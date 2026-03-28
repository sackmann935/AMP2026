# Radar Ablation Learnings And Next Fusion Redesign

Last updated: 2026-03-28

## Implementation Status (current branch)

Implemented:
- Pedestrian proposal refinement module:
  - `src/model/fusion/radar_proposal_refinement.py`
- Detector integration for:
  - radar-first proposal-guided refinement,
  - fusion modality dropout,
  - optional module freezing via config.
- Config surface additions in:
  - `src/config/model/centerpoint_radar.yaml`
- New launch profiles in:
  - `src/tools/slurm_train.sh`
  - `fusion_rebuild_stageA`, `fusion_rebuild_a`, `fusion_rebuild_b`, `fusion_rebuild_c`

Quick launch:
- Stage A warm-start (2 epochs, radar frozen):
  - `PROFILE=fusion_rebuild_stageA sbatch src/tools/slurm_train.sh`
- Main rebuild from `ablate_bev_res_016` checkpoint:
  - `PROFILE=fusion_rebuild_a sbatch src/tools/slurm_train.sh`
- Ped-upweighted variant:
  - `PROFILE=fusion_rebuild_b sbatch src/tools/slurm_train.sh`
- No-modality-dropout control:
  - `PROFILE=fusion_rebuild_c sbatch src/tools/slurm_train.sh`
- To resume from a different checkpoint:
  - `BASE_RADAR_CKPT=/abs/path/to/ckpt PROFILE=fusion_rebuild_a sbatch src/tools/slurm_train.sh`

Checkpoint loading note:
- Fusion rebuild profiles use `warm_start_checkpoint_path` (non-strict weight init), not strict Lightning resume.
- This is required when loading radar-only checkpoints into a larger radar+camera+refinement model.
- Use `checkpoint_path=...` only for same-architecture resume.

## 1) What We Learned From Current Debug Runs

Source artifacts:
- `outputs/ped_debug_aggregate.tsv`
- `outputs/*/ped_debug_all_*/ped_debug_summary.json`
- local W&B summaries in `outputs/*/wandb_logs/wandb/run-*/files/wandb-summary.json`

### 1.1 Strongest radar-only baseline
- Best overall baseline to build from: `ablate_bev_res_016`
  - Best observed ROI mAP: `64.39`
  - Best observed ROI Pedestrian AP: `35.92`
  - Best pedestrian debug recall (@ eval threshold): `0.331`

### 1.2 Core pedestrian bottleneck (not solved by small tuning)
- In ped-debug bins, most pedestrian GT boxes contain no strict in-box radar points:
  - `0 points`: `3555 / 3749` (`94.83%`)
  - `1-2 points`: `185 / 3749` (`4.93%`)
  - `3-5 points`: `9 / 3749` (`0.24%`)
- This explains why many tweaks only shift precision/recall tradeoff without giving a large AP jump.

### 1.3 Observed behavior from ablations
- Higher BEV resolution (`0.16`) gives the most reliable gain.
- Adding distance+doppler often improves confidence/precision but does not consistently improve pedestrian recall.
- `min_radius=2` globally increases conservatism (precision up, recall down) and hurts pedestrian coverage.
- Temporal taming helps stability in some distance bins, but not enough to break the ped ceiling.

### 1.4 Calibration symptom
- In multiple ped-debug summaries, low score thresholds (`0.03`, `0.05`, `0.1`) produce identical P/R/F1.
- This indicates poor score separation in the low-score region for pedestrian predictions.

## 2) Why Previous Fusion Attempts Likely Underperformed

Main hypothesis from run behavior:
- Fusion injected camera features too uniformly (or too weakly where needed), instead of focusing on radar-uncertain proposals.
- With sparse pedestrian radar support, camera must act as a targeted refiner, not a broad dense replacement.
- Without modality dropout, one branch can dominate and reduce complementary learning.

## 3) Bottom-Up Redesign (Recommended)

Goal:
- Keep radar as proposal generator.
- Use camera only to refine pedestrian proposals.
- Force real complementary behavior with modality dropout.

### 3.1 Architecture concept
1. Radar proposal stage:
   - Run radar path to produce proposal heatmaps/boxes (especially Pedestrian task).
2. Camera-guided refinement stage:
   - Use fused radar+camera features to predict residual corrections (ped score and optionally center/size residuals).
   - Apply refinement mainly around radar proposal regions.
3. Output:
   - Final detections come from refined predictions.

### 3.2 Modality dropout (training only)
- Apply stochastic branch dropout during training:
  - `drop_camera_prob`: recommended `0.25-0.35`
  - `drop_radar_prob`: recommended `0.05-0.15`
- Never drop both simultaneously in the same sample.
- Effect:
  - prevents camera-overwrite failure mode,
  - improves robustness to weak camera/radar conditions,
  - encourages each branch to provide useful independent signal.

## 4) Concrete Implementation Blueprint

This section is the exact implementation target for the next code pass.

### 4.1 Config additions
Add to model config (radar model YAML):
- `fusion.mode: radar_proposal_refine`
- `fusion.proposal_refine.enabled: true`
- `fusion.proposal_refine.ped_task_id: 1`
- `fusion.proposal_refine.topk: 300`
- `fusion.proposal_refine.score_thresh: 0.05`
- `fusion.proposal_refine.refine_heatmap: true`
- `fusion.proposal_refine.refine_box: false` (start simple)
- `fusion.modality_dropout.enabled: true`
- `fusion.modality_dropout.drop_camera_prob: 0.30`
- `fusion.modality_dropout.drop_radar_prob: 0.10`

### 4.2 Detector flow changes
In `src/model/detector/centerpoint.py`:
1. Compute radar features and radar proposals first.
2. Compute camera features.
3. Apply modality dropout masks to branches (training only).
4. Fuse radar+camera features.
5. Predict refined outputs.
6. Apply pedestrian refinement using radar proposal map + fused features.

### 4.3 Refinement module (new file)
Create:
- `src/model/fusion/radar_proposal_refinement.py`

Minimal first version:
- Inputs:
  - radar pedestrian proposal heatmap logits,
  - fused neck feature map.
- Predict:
  - residual pedestrian heatmap logits (masked to proposal neighborhoods).
- Later extension:
  - residual bbox offsets for pedestrian branch.

Why start with heatmap-only refinement:
- lower risk,
- cheaper training,
- isolates score/recall calibration issue first.

## 5) Training Strategy Under Current Compute Constraints

Given your hardware/time constraints and 6-epoch pattern:

### Stage A (stability warm start, 2 epochs)
- Initialize from `ablate_bev_res_016` checkpoint.
- Train camera encoder + fusion + refinement modules.
- Keep main radar backbone/head frozen.
- LR: `1e-3`, no scheduler.

### Stage B (joint adaptation, 4 epochs)
- Unfreeze all modules.
- Continue from Stage A checkpoint.
- LR: `5e-4` to `8e-4`, no scheduler.

### Recommended first run set
1. `fusion_rebuild_a`: proposal-refine + modality dropout (heatmap refine only)
2. `fusion_rebuild_b`: same + ped task weight bump (`task_loss_weights=[1.0,1.35,1.0]`)
3. `fusion_rebuild_c`: same as (2) but without modality dropout (ablation for attribution)

## 6) Guardrails To Avoid Old Failure Modes

- Keep `bev016` geometry as default reference.
- Avoid global `min_radius=2` as default for ped-focused work.
- Do not mix too many new knobs in one run; keep attribution clean.
- Always compare against `ablate_bev_res_016` checkpoint with same eval setup.

## 7) Decision Criteria

Promote the redesign only if it shows:
- pedestrian ROI AP gain with minimal degradation in Car/Cyclist,
- improved ped-debug recall in `15-30m` and `30m+`,
- better threshold-sweep separation (0.03/0.05/0.1 not all identical).

## 8) Failed Run Learnings (2026-03-28 Cleanup)

Archived artifacts:
- `outputs/failed_runs_archive/2026-03-28_fusion_stageA_cleanup`
- See manifest: `outputs/failed_runs_archive/2026-03-28_fusion_stageA_cleanup/MANIFEST.md`

Runs/logs archived:
- Slurm jobs: `9463638`, `9463677`, `9463696`
- Local W&B run dirs: `ksqhxqma`, `gixn97yp`, `h1iy5t94`

### 8.1 Root causes and fixes

1. Fusion checkpoint restore mismatch (job `9463638`)
- Failure: strict Lightning resume from radar-only checkpoint into larger radar+camera+fusion model.
- Symptom: many missing state_dict keys for camera/fusion/refiner.
- Fix: use warm-start path (`warm_start_checkpoint_path` + `warm_start_strict=false`) instead of strict `checkpoint_path` resume for architecture changes.

2. BF16 validation conversion crash (job `9463696`)
- Failure: conversion to NumPy from BF16 tensors in validation export path.
- Symptom: `TypeError: Got unsupported ScalarType BFloat16`.
- Fix implemented: cast eval tensors to float32 before `.cpu().numpy()` in `convert_valid_bboxes` (`box2d`, `location_cam`, `box3d_lidar`, `scores`).

3. Intermittent train-loss explosions in StageA
- Failure mode: `train/loss_fusion_ratio_reg` dominated total loss on spike steps.
- Observed pattern: spikes happened when `drop_radar=1`; pre-cap assist ratios exploded and reg term blew up.
- Fix implemented: mask ratio-regularization loss when radar branch is dropped by modality dropout (`train/fusion/ratio_reg_masked_drop_radar` logged for traceability).

4. Slurm CPU request incompatibility (submission issue)
- Failure: `--cpus-per-task=4` rejected on `gpu-a100-small` (max `2`).
- Fix: reverted to `--cpus-per-task=2` and default `NUM_WORKERS=2`.

### 8.2 Practical guardrails for next runs
- For fusion warm-starts, never use strict `checkpoint_path` across architecture changes.
- For mixed precision (`bf16-mixed`), ensure all eval-export NumPy conversions cast to float32 first.
- Keep modality dropout, but prevent ratio-reg from being computed on dropped-radar batches.
- Keep slurm profile resource requests compatible with partition limits.
