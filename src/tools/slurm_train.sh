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

# Profiles kept after cleanup + new radar-source ablations.
#
# Core tested profiles:
#   PROFILE=baseline
#   PROFILE=gaussian_topk_chunked_regularized_cosine
#   PROFILE=ablate_accum2
#   PROFILE=ablate_accum2_no_cosine
#   PROFILE=ablate_bnfreeze_e3
#   PROFILE=ablate_bnfreeze_e_no_cosine
#   PROFILE=ablate_groupnorm_g16
#   PROFILE=ped_accum2_no_cosine_minr1_thr003
#
# Radar-source ablations (ped-best settings):
#   PROFILE=radar_single_pedbest
#   PROFILE=radar_3f_pedbest
#   PROFILE=radar_5f_pedbest
#
# Radar-source ablations (groupnorm settings):
#   PROFILE=radar_single_gn
#   PROFILE=radar_3f_gn
#   PROFILE=radar_5f_gn
#
# Smoke test:
#   PROFILE=topk_chunked_smoke
PROFILE=${PROFILE:-radar_5f_pedbest_tempmax10}

case "$PROFILE" in
  baseline)
    CMD=(python -u src/tools/train.py
      exp_id=centerpoint_radar_baseline
      epochs=12 batch_size=4 num_workers=2
      radar_source=radar)
    ;;

  topk_chunked_smoke)
    CMD=(python -u src/tools/train.py
      exp_id=topk_chunked_smoke
      epochs=1 batch_size=2 num_workers=2
      radar_source=radar
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.camera.debug.enabled=true model.camera.debug.log_memory=true)
    ;;

  gaussian_topk_chunked_regularized_cosine)
    CMD=(python -u src/tools/train.py
      exp_id=gaussian_topk_chunked_regularized_cosine
      epochs=20 batch_size=2 num_workers=2
      radar_source=radar
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked)
    ;;

  ablate_accum2)
    CMD=(python -u src/tools/train.py
      exp_id=ablate_accum2
      epochs=20 batch_size=2 num_workers=2
      accumulate_grad_batches=2
      radar_source=radar
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked)
    ;;

  ablate_accum2_no_cosine)
    CMD=(python -u src/tools/train.py
      exp_id=ablate_accum2_no_cosine
      epochs=20 batch_size=2 num_workers=2
      accumulate_grad_batches=2
      radar_source=radar
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.optimizer.scheduler.enabled=false)
    ;;

  ablate_bnfreeze_e3)
    CMD=(python -u src/tools/train.py
      exp_id=ablate_bnfreeze_e3
      epochs=20 batch_size=2 num_workers=2
      radar_source=radar
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
      radar_source=radar
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=batchnorm
      model.regularization.freeze_bn.enabled=true
      model.regularization.freeze_bn.freeze_epoch=3
      model.regularization.freeze_bn.freeze_affine=false
      model.optimizer.scheduler.enabled=false)
    ;;

  ablate_groupnorm_g16)
    CMD=(python -u src/tools/train.py
      exp_id=ablate_groupnorm_g16
      epochs=20 batch_size=2 num_workers=2
      radar_source=radar
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=groupnorm
      model.regularization.group_norm_groups=16
      model.optimizer.scheduler.enabled=false)
    ;;

  ped_accum2_no_cosine_minr1_thr003)
    CMD=(python -u src/tools/train.py
      exp_id=ped_accum2_no_cosine_minr1_thr003
      epochs=20 batch_size=2 num_workers=2
      accumulate_grad_batches=2
      checkpoint_monitor=validation/ROI/mAP checkpoint_mode=max
      eval_score_threshold=0.03
      radar_source=radar
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

  # Radar-source ablations on ped-best setup.
  radar_single_pedbest)
    CMD=(python -u src/tools/train.py
      exp_id=radar_single_pedbest
      epochs=20 batch_size=2 num_workers=2
      accumulate_grad_batches=2
      checkpoint_monitor=validation/ROI/mAP checkpoint_mode=max
      eval_score_threshold=0.03
      radar_source=radar
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

  radar_3f_pedbest)
    CMD=(python -u src/tools/train.py
      exp_id=radar_3f_pedbest
      epochs=20 batch_size=2 num_workers=2
      accumulate_grad_batches=2
      checkpoint_monitor=validation/ROI/mAP checkpoint_mode=max
      eval_score_threshold=0.03
      radar_source=radar_3frames
      radar_prioritize_recent=true
      temporal_max_num_points=10
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

  radar_5f_pedbest_tempmax10)
    CMD=(python -u src/tools/train.py
      exp_id=radar_5f_pedbest_tempmax10
      epochs=20 batch_size=2 num_workers=2
      accumulate_grad_batches=2
      checkpoint_monitor=validation/ROI/mAP checkpoint_mode=max
      eval_score_threshold=0.03
      radar_source=radar_5frames
      radar_prioritize_recent=true
      temporal_max_num_points=10
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

  # Radar-source ablations on groupnorm setup.
  radar_single_gn)
    CMD=(python -u src/tools/train.py
      exp_id=radar_single_gn
      epochs=20 batch_size=2 num_workers=2
      accumulate_grad_batches=2
      checkpoint_monitor=validation/ROI/mAP checkpoint_mode=max
      eval_score_threshold=0.03
      radar_source=radar
      model.head.train_cfg.min_radius=1
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=groupnorm
      model.regularization.group_norm_groups=16
      model.optimizer.scheduler.enabled=false)
    ;;

  radar_3f_gn)
    CMD=(python -u src/tools/train.py
      exp_id=radar_3f_gn
      epochs=20 batch_size=2 num_workers=2
      accumulate_grad_batches=2
      checkpoint_monitor=validation/ROI/mAP checkpoint_mode=max
      eval_score_threshold=0.03
      radar_source=radar_3frames
      radar_prioritize_recent=true
      temporal_max_num_points=10
      model.head.train_cfg.min_radius=1
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=groupnorm
      model.regularization.group_norm_groups=16
      model.optimizer.scheduler.enabled=false)
    ;;

  radar_5f_gn)
    CMD=(python -u src/tools/train.py
      exp_id=radar_5f_gn
      epochs=20 batch_size=2 num_workers=2
      accumulate_grad_batches=2
      checkpoint_monitor=validation/ROI/mAP checkpoint_mode=max
      eval_score_threshold=0.03
      radar_source=radar_5frames
      radar_prioritize_recent=true
      temporal_max_num_points=10
      model.head.train_cfg.min_radius=1
      model.middle_encoder.type=gaussian_soft
      model.fusion.enabled=true model.fusion.type=cmx_lite
      model.camera.lift_mode=topk_chunked
      model.regularization.norm_mode=groupnorm
      model.regularization.group_norm_groups=16
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
