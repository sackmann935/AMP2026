#!/bin/bash
#SBATCH --job-name="centperpoint_ro47020"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --account=education-me-courses-ro47020
#SBATCH --mail-type=END
#SBATCH --chdir=/home/lsackmann/final_assignment
#SBATCH --output=/home/lsackmann/final_assignment/outputs/slurm_centerpoint_ro47020_%j.out
#SBATCH --error=/home/lsackmann/final_assignment/outputs/slurm_centerpoint_ro47020_%j.err

set -euo pipefail

module load 2024r1 miniconda3/4.12.0 cuda/12.5

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amp

if command -v nvidia-smi >/dev/null 2>&1; then
  previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
  nvidia-smi
else
  previous=""
  echo "warning: nvidia-smi not found; skipping GPU usage snapshot."
fi

# Use one of:
#   PROFILE=baseline
#   PROFILE=gauss_s06_r1_gated_expected
#   PROFILE=gauss_s06_r1_gated_naive
#   PROFILE=gauss_s06_r1_lift_mode_naive
#   PROFILE=gauss_s08_r1_cmxlite
#   PROFILE=gauss_s06_r1_cmxlite_long
#   PROFILE=naive_smoke
#   PROFILE=topk_chunked_smoke
#   PROFILE=gaussian_topk_chunked_e20_s06_regularized_cosine
#
# Ablation profiles for memory/generalization fixes:
#   PROFILE=ablate_accum2
#   PROFILE=ablate_bnfreeze_e3
#   PROFILE=ablate_groupnorm_g16
#   PROFILE=ablate_groupnorm_g16_accum2
#   PROFILE=ablate_bnfreeze_e1_accum2
#
# Ped-focused ROI profiles (recommended for today's runs):
#   PROFILE=ped_bnfreeze_no_cosine
#   PROFILE=ped_bnfreeze_no_cosine_thr003
#   PROFILE=ped_bnfreeze_no_cosine_thr001
#   PROFILE=ped_bnfreeze_no_cosine_minr1
#   PROFILE=ped_bnfreeze_no_cosine_minr1_thr003
#   PROFILE=ped_bnfreeze_no_cosine_minr1_thr003_nmsped02
#   PROFILE=ped_accum2_no_cosine_minr1_thr003
PROFILE=${PROFILE:-ped_accum2_no_cosine_minr1_thr003}

case "$PROFILE" in
  baseline)
    CMD=(python -u src/tools/train.py
      exp_id=centerpoint_baseline_db_try_slurm
      batch_size=4 num_workers=2 epochs=12)
    ;;
  gauss_s06_r1_gated_expected)
    CMD=(python -u src/tools/train.py
      exp_id=gauss_s06_r1_gated
      epochs=6 batch_size=2 num_workers=2
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true
      model.fusion.type=cmx_lite
      model.camera.lift_mode=expected)
    ;;
  gauss_s06_r1_gated_naive)
    CMD=(python -u src/tools/train.py
      exp_id=gauss_s06_r1_gated
      epochs=6 batch_size=2 num_workers=2
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true
      model.fusion.type=cmx_lite
      model.camera.lift_mode=naive_dense)
    ;;
  gauss_s06_r1_lift_mode_naive)
    CMD=(python -u src/tools/train.py
      exp_id=gauss_s06_r1_lift_mode_naive
      epochs=6 batch_size=2 num_workers=2
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true
      model.fusion.type=cmx_lite
      model.camera.lift_mode=naive_dense)
    ;;
  gauss_s08_r1_cmxlite)
    CMD=(python -u src/tools/train.py
      exp_id=gauss_s08_r1_cmxlite
      epochs=6 batch_size=4 num_workers=2
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true
      model.fusion.type=cmx_lite)
    ;;
  gauss_s06_r1_cmxlite_long)
    CMD=(python -u src/tools/train.py
      exp_id=gauss_s06_r1_cmxlite_long
      epochs=24 batch_size=2 num_workers=2
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true
      model.fusion.type=cmx_lite)
    ;;
  naive_smoke)
    CMD=(python -u src/tools/train.py
      exp_id=naive_smoke
      epochs=1 batch_size=1 num_workers=0
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=naive_dense
      model.camera.debug.enabled=true model.camera.debug.log_memory=true)
    ;;
  topk_chunked_smoke)
    CMD=(python -u src/tools/train.py
      exp_id=topk_chunked_smoke
      epochs=1 batch_size=2 num_workers=2
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.camera.debug.enabled=true model.camera.debug.log_memory=true)
    ;;
  gaussian_topk_chunked_e20_s06_regularized_cosine)
    CMD=(python -u src/tools/train.py
      exp_id=gaussian_topk_chunked_regularized_cosine
      epochs=20 batch_size=2 num_workers=2
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked)
    ;;

  # Fix 1: gradient accumulation only (effective larger batch)
  ablate_accum2)
    CMD=(python -u src/tools/train.py
      exp_id=ablate_accum2
      epochs=20 batch_size=2 num_workers=2
      accumulate_grad_batches=2
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked)
    ;;

  ablate_accum2_no_cosine)
    CMD=(python -u src/tools/train.py
      exp_id=ablate_accum2_no_cosine
      epochs=20 batch_size=2 num_workers=2
      accumulate_grad_batches=2
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.optimizer.scheduler.enabled=false)
    ;;

  # Fix 2: freeze BatchNorm stats from epoch 1
  ablate_bnfreeze_e3)
    CMD=(python -u src/tools/train.py
      exp_id=ablate_bnfreeze_e3
      epochs=20 batch_size=2 num_workers=2
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=batchnorm
      model.regularization.freeze_bn.enabled=true
      model.regularization.freeze_bn.freeze_epoch=3
      model.regularization.freeze_bn.freeze_affine=false)
    ;;

  ablate_bnfreeze_e_no_cosine)
    CMD=(python -u src/tools/train.py
      exp_id=ablate_bnfreeze_e_no_cosine
      epochs=20 batch_size=2 num_workers=2
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=batchnorm
      model.regularization.freeze_bn.enabled=true
      model.regularization.freeze_bn.freeze_epoch=3
      model.regularization.freeze_bn.freeze_affine=false
      model.optimizer.scheduler.enabled=false)
    ;;

  # Fix 3: replace BatchNorm with GroupNorm
  ablate_groupnorm_g16)
    CMD=(python -u src/tools/train.py
      exp_id=ablate_groupnorm_g16
      epochs=20 batch_size=2 num_workers=2
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=groupnorm
      model.regularization.group_norm_groups=16
      model.optimizer.scheduler.enabled=false)
    ;;

  # Combination: GroupNorm + gradient accumulation
  ablate_groupnorm_g16_accum2)
    CMD=(python -u src/tools/train.py
      exp_id=ablate_groupnorm_g16_accum2
      epochs=20 batch_size=2 num_workers=2
      accumulate_grad_batches=2
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=groupnorm
      model.regularization.group_norm_groups=16)
    ;;

  # Combination: BN freeze + gradient accumulation
  ablate_bnfreeze_e1_accum2)
    CMD=(python -u src/tools/train.py
      exp_id=ablate_bnfreeze_e1_accum2
      epochs=20 batch_size=2 num_workers=2
      accumulate_grad_batches=2
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=batchnorm
      model.regularization.freeze_bn.enabled=true
      model.regularization.freeze_bn.freeze_epoch=1
      model.regularization.freeze_bn.freeze_affine=false)
    ;;

  # Ped-focused base: best stable setting from recent runs.
  ped_bnfreeze_no_cosine)
    CMD=(python -u src/tools/train.py
      exp_id=ped_bnfreeze_no_cosine
      epochs=20 batch_size=2 num_workers=2
      checkpoint_monitor=validation/ROI/mAP checkpoint_mode=max
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=batchnorm
      model.regularization.freeze_bn.enabled=true
      model.regularization.freeze_bn.freeze_epoch=3
      model.regularization.freeze_bn.freeze_affine=false
      model.optimizer.scheduler.enabled=false)
    ;;

  # Lower ROI eval threshold to recover low-confidence Ped/Cyclist detections.
  ped_bnfreeze_no_cosine_thr003)
    CMD=(python -u src/tools/train.py
      exp_id=ped_bnfreeze_no_cosine_thr003
      epochs=20 batch_size=2 num_workers=2
      checkpoint_monitor=validation/ROI/mAP checkpoint_mode=max
      eval_score_threshold=0.03
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=batchnorm
      model.regularization.freeze_bn.enabled=true
      model.regularization.freeze_bn.freeze_epoch=3
      model.regularization.freeze_bn.freeze_affine=false
      model.optimizer.scheduler.enabled=false)
    ;;

  ped_bnfreeze_no_cosine_thr001)
    CMD=(python -u src/tools/train.py
      exp_id=ped_bnfreeze_no_cosine_thr001
      epochs=20 batch_size=2 num_workers=2
      checkpoint_monitor=validation/ROI/mAP checkpoint_mode=max
      eval_score_threshold=0.01
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=batchnorm
      model.regularization.freeze_bn.enabled=true
      model.regularization.freeze_bn.freeze_epoch=3
      model.regularization.freeze_bn.freeze_affine=false
      model.optimizer.scheduler.enabled=false)
    ;;

  # Sharper heatmap targets for small objects.
  ped_bnfreeze_no_cosine_minr1)
    CMD=(python -u src/tools/train.py
      exp_id=ped_bnfreeze_no_cosine_minr1
      epochs=20 batch_size=2 num_workers=2
      checkpoint_monitor=validation/ROI/mAP checkpoint_mode=max
      model.head.train_cfg.min_radius=1
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=batchnorm
      model.regularization.freeze_bn.enabled=true
      model.regularization.freeze_bn.freeze_epoch=3
      model.regularization.freeze_bn.freeze_affine=false
      model.optimizer.scheduler.enabled=false)
    ;;

  ped_bnfreeze_no_cosine_minr1_thr003)
    CMD=(python -u src/tools/train.py
      exp_id=ped_bnfreeze_no_cosine_minr1_thr003
      epochs=20 batch_size=2 num_workers=2
      checkpoint_monitor=validation/ROI/mAP checkpoint_mode=max
      eval_score_threshold=0.03
      model.head.train_cfg.min_radius=1
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=batchnorm
      model.regularization.freeze_bn.enabled=true
      model.regularization.freeze_bn.freeze_epoch=3
      model.regularization.freeze_bn.freeze_affine=false
      model.optimizer.scheduler.enabled=false)
    ;;

  # Less aggressive Ped circle-NMS suppression.
  ped_bnfreeze_no_cosine_minr1_thr003_nmsped02)
    CMD=(python -u src/tools/train.py
      exp_id=ped_bnfreeze_no_cosine_minr1_thr003_nmsped02
      epochs=20 batch_size=2 num_workers=2
      checkpoint_monitor=validation/ROI/mAP checkpoint_mode=max
      eval_score_threshold=0.03
      model.head.train_cfg.min_radius=1
      model.head.test_cfg.min_radius=[4,0.2,0.85]
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=batchnorm
      model.regularization.freeze_bn.enabled=true
      model.regularization.freeze_bn.freeze_epoch=3
      model.regularization.freeze_bn.freeze_affine=false
      model.optimizer.scheduler.enabled=false)
    ;;

  # Ped-focused combo with larger effective batch.
  ped_accum2_no_cosine_minr1_thr003)
    CMD=(python -u src/tools/train.py
      exp_id=ped_accum2_no_cosine_minr1_thr003
      epochs=20 batch_size=2 num_workers=2
      accumulate_grad_batches=2
      checkpoint_monitor=validation/ROI/mAP checkpoint_mode=max
      eval_score_threshold=0.03
      model.head.train_cfg.min_radius=1
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=batchnorm
      model.regularization.freeze_bn.enabled=true
      model.regularization.freeze_bn.freeze_epoch=3
      model.regularization.freeze_bn.freeze_affine=false
      model.optimizer.scheduler.enabled=false)
    ;;

  *)
    echo "Unknown PROFILE=$PROFILE"
    exit 2
    ;;
esac

echo "Running PROFILE=$PROFILE"
printf 'Command: %q ' "${CMD[@]}"; echo
srun "${CMD[@]}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous" || true
fi
