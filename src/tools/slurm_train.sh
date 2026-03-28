#!/bin/bash
#SBATCH --job-name="centerpoint_ro47020"
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

print_profiles() {
  cat <<'TXT'
Available PROFILE values:

  radar_5f_recency_hard
  ablate_bev_res_016
  ablate_bev_res_020
  bev020_cam_ped_rescue_smoke
  bev020_cam_ped_rescue_dist_dopp
  ablate_distance_doppler_hi_capacity
  ablate_bev016_dist_only
  ablate_bev016_dopp_only
  ablate_bev016_dist_dopp_hicap
  ablate_bev016_dist_dopp_hicap_pfnplus
  ablate_bev016_dist_dopp_hicap_temporal_tamed
  ablate_bev016_dist_dopp_hicap_no_bnfreeze
  ablate_bev016_dist_dopp_hicap_no_bnfreeze_tmp6
  ablate_bev016_minr2_global
  ablate_bev016_minr_task_ped2
  ablate_bev016_minr_task_ped3
  ablate_bev016_dist_dopp_hicap_featnorm
  ablate_bev016_dist_dopp_hicap_temporal_tamed_minr_task_ped2
  ablate_bev016_dist_dopp_hicap_featnorm_temporal_tamed
  ablate_bev016_dist_dopp_hicap_featnorm_temporal_tamed_minr_task_ped2
  list

Common env overrides:
  PROFILE=ablate_bev_res_016
  EPOCHS=6 BATCH_SIZE=2 NUM_WORKERS=2 ACCUMULATE_GRAD_BATCHES=2
  EVAL_SCORE_THRESHOLD=0.03 TEMPORAL_MAX_NUM_POINTS=10
TXT
}

add_bev016_common() {
  CMD+=(
    radar_source=radar_5frames
    model.middle_encoder.type=hard
    model.voxel_encoder.recency_beta=2.0
    'model.voxel_size=[0.16,0.16,5]'
    'model.middle_encoder.output_shape=[320,320]'
    'model.head.train_cfg.grid_size=[320,320,1]'
    'model.head.bbox_coder.voxel_size=[0.16,0.16]'
    'model.pts_voxel_layer.max_voxels=[18000,50000]'
  )
}

add_bev020_common() {
  CMD+=(
    radar_source=radar_5frames
    model.middle_encoder.type=hard
    model.voxel_encoder.recency_beta=2.0
    'model.voxel_size=[0.20,0.20,5]'
    'model.middle_encoder.output_shape=[256,256]'
    'model.head.train_cfg.grid_size=[256,256,1]'
    'model.head.bbox_coder.voxel_size=[0.20,0.20]'
    'model.pts_voxel_layer.max_voxels=[12000,32000]'
  )
}

add_camera_rescue_common() {
  CMD+=(
    model.camera_rescue.enabled=true
    model.camera_rescue.ped_task_id=1
    model.camera_rescue.topk=220
    model.camera_rescue.score_threshold=0.15
    model.camera_rescue.proposal_radius=2
    model.camera_rescue.support_dilation=2
    model.camera_rescue.depth_min=1.0
    model.camera_rescue.depth_max=80.0
    model.camera_rescue.image_heatmap_radius=2
    'model.camera_rescue.feat_channels=[32,64,96]'
    model.camera_rescue.loss_weight_heatmap=1.0
    model.camera_rescue.loss_weight_depth=0.2
    model.camera_rescue.fusion_gate.bias=-2.0
    model.camera_rescue.fusion_gate.no_support_weight=4.0
    model.camera_rescue.fusion_gate.cam_conf_weight=1.5
    model.camera_rescue.fusion_gate.radar_uncertainty_weight=1.0
    model.camera_rescue.fusion_gate.camera_scale=1.0
  )
}

add_dist_range() {
  CMD+=(
    model.voxel_encoder.with_distance=true
    model.voxel_encoder.range_feature_aug=true
    model.voxel_encoder.range_add_inverse=true
    model.voxel_encoder.range_add_log=true
    model.voxel_encoder.range_eps=0.01
  )
}

add_doppler() {
  CMD+=(
    model.voxel_encoder.doppler_feature_aug=true
    model.voxel_encoder.doppler_add_abs=true
    model.voxel_encoder.doppler_add_range_norm=true
    model.voxel_encoder.doppler_add_time_interaction=true
  )
}

add_hicap() {
  CMD+=(
    'model.voxel_encoder.extra_feat_channels=[128]'
  )
}

add_hicap_pfnplus() {
  CMD+=(
    'model.voxel_encoder.extra_feat_channels=[128,96]'
  )
}

add_temporal_tamed() {
  CMD+=(
    model.voxel_encoder.recency_beta=1.0
    model.voxel_encoder.doppler_add_time_interaction=false
  )
}

add_feature_norm() {
  CMD+=(
    model.voxel_encoder.feature_scale_norm_enabled=true
    model.voxel_encoder.feature_scale_norm_include_distance=true
    model.voxel_encoder.feature_scale_norm_eps=0.001
    model.voxel_encoder.feature_scale_norm_clip=5.0
  )
}

add_no_bnfreeze() {
  CMD+=(
    model.regularization.freeze_bn.enabled=false
  )
}

PROFILE=${PROFILE:-ablate_bev_res_020}
EPOCHS=${EPOCHS:-6}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-2}
ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES:-2}
EVAL_SCORE_THRESHOLD=${EVAL_SCORE_THRESHOLD:-0.03}
TEMPORAL_MAX_NUM_POINTS=${TEMPORAL_MAX_NUM_POINTS:-10}
TRAIN_PRECISION=${TRAIN_PRECISION:-32-true}
NUM_SANITY_VAL_STEPS=${NUM_SANITY_VAL_STEPS:-0}
VAL_EVERY=${VAL_EVERY:-1}

CMD=(
  python -u src/tools/train.py
  epochs=${EPOCHS}
  batch_size=${BATCH_SIZE}
  num_workers=${NUM_WORKERS}
  precision=${TRAIN_PRECISION}
  num_sanity_val_steps=${NUM_SANITY_VAL_STEPS}
  accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}
  checkpoint_monitor=validation/ROI/mAP
  checkpoint_mode=max
  val_every=${VAL_EVERY}
  eval_score_threshold=${EVAL_SCORE_THRESHOLD}
  radar_source=radar_5frames
  radar_prioritize_recent=true
  temporal_max_num_points=${TEMPORAL_MAX_NUM_POINTS}
  model.head.train_cfg.min_radius=1
  model.middle_encoder.type=hard
  model.voxel_encoder.recency_weighted_pooling=true
  model.voxel_encoder.recency_time_index=6
  model.voxel_encoder.recency_beta=2.0
  model.voxel_encoder.recency_min_weight=0.2
  model.regularization.norm_mode=batchnorm
  model.regularization.freeze_bn.enabled=true
  model.regularization.freeze_bn.freeze_epoch=3
  model.regularization.freeze_bn.freeze_affine=false
  model.optimizer.scheduler.enabled=false
)

case "$PROFILE" in
  list)
    print_profiles
    exit 0
    ;;

  radar_5f_recency_hard)
    CMD+=(
      exp_id=radar_5f_recency_hard
      radar_source=radar_5frames
      model.middle_encoder.type=hard
      model.voxel_encoder.recency_beta=2.0
    )
    ;;

  ablate_bev_res_016)
    CMD+=(exp_id=ablate_bev_res_016)
    add_bev016_common
    ;;

  ablate_bev_res_020)
    CMD+=(exp_id=ablate_bev_res_020)
    add_bev020_common
    ;;

  bev020_cam_ped_rescue_smoke)
    CMD+=(
      exp_id=bev020_cam_ped_rescue_smoke
      epochs=3
      num_workers=0
      val_num_workers=0
      batch_size=1
      accumulate_grad_batches=1
    )
    add_bev020_common
    add_camera_rescue_common
    ;;

  bev020_cam_ped_rescue_dist_dopp)
    CMD+=(exp_id=bev020_cam_ped_rescue_dist_dopp)
    add_bev020_common
    add_dist_range
    add_doppler
    add_camera_rescue_common
    ;;

  ablate_distance_doppler_hi_capacity)
    CMD+=(
      exp_id=ablate_distance_doppler_hi_capacity
      radar_source=radar_5frames
      model.middle_encoder.type=hard
      model.voxel_encoder.recency_beta=2.0
    )
    add_dist_range
    add_doppler
    add_hicap
    ;;

  ablate_bev016_dist_only)
    CMD+=(exp_id=ablate_bev016_dist_only)
    add_bev016_common
    add_dist_range
    ;;

  ablate_bev016_dopp_only)
    CMD+=(exp_id=ablate_bev016_dopp_only)
    add_bev016_common
    add_doppler
    ;;

  ablate_bev016_dist_dopp_hicap)
    CMD+=(exp_id=ablate_bev016_dist_dopp_hicap)
    add_bev016_common
    add_dist_range
    add_doppler
    add_hicap
    ;;

  ablate_bev016_dist_dopp_hicap_pfnplus)
    CMD+=(exp_id=ablate_bev016_dist_dopp_hicap_pfnplus)
    add_bev016_common
    add_dist_range
    add_doppler
    add_hicap_pfnplus
    ;;

  ablate_bev016_dist_dopp_hicap_temporal_tamed)
    CMD+=(exp_id=ablate_bev016_dist_dopp_hicap_temporal_tamed)
    add_bev016_common
    add_dist_range
    add_doppler
    add_hicap
    add_temporal_tamed
    ;;

  ablate_bev016_dist_dopp_hicap_no_bnfreeze)
    CMD+=(exp_id=ablate_bev016_dist_dopp_hicap_no_bnfreeze)
    add_bev016_common
    add_dist_range
    add_doppler
    add_hicap
    add_no_bnfreeze
    ;;

  ablate_bev016_dist_dopp_hicap_no_bnfreeze_tmp6)
    CMD+=(
      exp_id=ablate_bev016_dist_dopp_hicap_no_bnfreeze_tmp6
      temporal_max_num_points=6
    )
    add_bev016_common
    add_dist_range
    add_doppler
    add_hicap
    add_no_bnfreeze
    ;;

  ablate_bev016_minr2_global)
    CMD+=(exp_id=ablate_bev016_minr2_global)
    add_bev016_common
    CMD+=(model.head.train_cfg.min_radius=2)
    ;;

  ablate_bev016_minr_task_ped2)
    CMD+=(exp_id=ablate_bev016_minr_task_ped2)
    add_bev016_common
    CMD+=('model.head.train_cfg.min_radius=[1,2,1]')
    ;;

  ablate_bev016_minr_task_ped3)
    CMD+=(exp_id=ablate_bev016_minr_task_ped3)
    add_bev016_common
    CMD+=('model.head.train_cfg.min_radius=[1,3,1]')
    ;;

  ablate_bev016_dist_dopp_hicap_featnorm)
    CMD+=(exp_id=ablate_bev016_dist_dopp_hicap_featnorm)
    add_bev016_common
    add_dist_range
    add_doppler
    add_hicap
    add_feature_norm
    ;;

  ablate_bev016_dist_dopp_hicap_temporal_tamed_minr_task_ped2)
    CMD+=(exp_id=ablate_bev016_dist_dopp_hicap_temporal_tamed_minr_task_ped2)
    add_bev016_common
    add_dist_range
    add_doppler
    add_hicap
    add_temporal_tamed
    CMD+=('model.head.train_cfg.min_radius=[1,2,1]')
    ;;

  ablate_bev016_dist_dopp_hicap_featnorm_temporal_tamed)
    CMD+=(exp_id=ablate_bev016_dist_dopp_hicap_featnorm_temporal_tamed)
    add_bev016_common
    add_dist_range
    add_doppler
    add_hicap
    add_feature_norm
    add_temporal_tamed
    ;;

  ablate_bev016_dist_dopp_hicap_featnorm_temporal_tamed_minr_task_ped2)
    CMD+=(exp_id=ablate_bev016_dist_dopp_hicap_featnorm_temporal_tamed_minr_task_ped2)
    add_bev016_common
    add_dist_range
    add_doppler
    add_hicap
    add_feature_norm
    add_temporal_tamed
    CMD+=('model.head.train_cfg.min_radius=[1,2,1]')
    ;;

  *)
    echo "Unknown PROFILE=$PROFILE"
    print_profiles
    exit 2
    ;;
esac

echo "Running PROFILE=$PROFILE"
printf 'Command: %q ' "${CMD[@]}"; echo
srun "${CMD[@]}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous" || true
fi
