#!/bin/bash
#SBATCH --job-name="ped_debug_ro47020"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --account=education-me-courses-ro47020
#SBATCH --mail-type=END
#SBATCH --chdir=/home/lsackmann/final_assignment
#SBATCH --output=/home/lsackmann/final_assignment/outputs/slurm_ped_debug_%j.out
#SBATCH --error=/home/lsackmann/final_assignment/outputs/slurm_ped_debug_%j.err

set -euo pipefail

module load 2024r1 miniconda3/4.12.0 cuda/12.5

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amp

# Ranked checkpoint presets (validation/ROI/mAP from checkpoint metadata)
# Copy-paste one block before sbatch:
#
# RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev_res_016
# CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev_res_016/checkpoints/ep4-ablate_bev_res_016-v1.ckpt
#
# RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_temporal_tamed
# CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_temporal_tamed/checkpoints/ep2-ablate_bev016_dist_dopp_hicap_temporal_tamed.ckpt
#
# RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_minr_task_ped2
# CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_minr_task_ped2/checkpoints/ep4-ablate_bev016_minr_task_ped2.ckpt
#
# RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_minr2_global
# CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_minr2_global/checkpoints/ep5-ablate_bev016_minr2_global.ckpt
#
# RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev020_dist_dopp_hicap_no_bnfreeze
# CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev020_dist_dopp_hicap_no_bnfreeze/checkpoints/ep3-ablate_bev020_dist_dopp_hicap_no_bnfreeze.ckpt
#
# RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_no_bnfreeze_tmp6
# CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_no_bnfreeze_tmp6/checkpoints/ep3-ablate_bev016_dist_dopp_hicap_no_bnfreeze_tmp6.ckpt
#
# RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap
# CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap/checkpoints/ep3-ablate_bev016_dist_dopp_hicap.ckpt
#
# RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev_res_020
# CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev_res_020/checkpoints/ep4-ablate_bev_res_020.ckpt
#
# RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dopp_only
# CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dopp_only/checkpoints/ep3-ablate_bev016_dopp_only.ckpt
#
# RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_pfnplus
# CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_pfnplus/checkpoints/ep3-ablate_bev016_dist_dopp_hicap_pfnplus.ckpt
#
# RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_featnorm
# CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_featnorm/checkpoints/ep5-ablate_bev016_dist_dopp_hicap_featnorm.ckpt
#
# RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_distance_doppler_hi_capacity
# CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_distance_doppler_hi_capacity/checkpoints/ep4-ablate_distance_doppler_hi_capacity.ckpt
#
# RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_no_bnfreeze
# CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_no_bnfreeze/checkpoints/ep5-ablate_bev016_dist_dopp_hicap_no_bnfreeze.ckpt
#
# RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_only
# CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_only/checkpoints/ep3-ablate_bev016_dist_only.ckpt
#
# Then run:
# sbatch src/tools/slurm_ped_debug.sh

RUN_DIR=${RUN_DIR:-/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_pfnplus}
CHECKPOINT=${CHECKPOINT:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
SCORE_THRESHOLDS=${SCORE_THRESHOLDS:-0.03,0.05,0.1,0.2,0.3,0.5}
EVAL_SCORE_THRESHOLD=${EVAL_SCORE_THRESHOLD:-0.1}
IOU_THRESHOLD=${IOU_THRESHOLD:-0.25}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_SAMPLES=${MAX_SAMPLES:-0}
PED_LABEL_INDEX=${PED_LABEL_INDEX:--1}

if [[ "${CHECKPOINT}" == "list" ]]; then
  cat <<'TXT'
Checkpoint presets (copy-paste):
RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev_res_016
CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev_res_016/checkpoints/ep4-ablate_bev_res_016-v1.ckpt

RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_temporal_tamed
CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_temporal_tamed/checkpoints/ep2-ablate_bev016_dist_dopp_hicap_temporal_tamed.ckpt

RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_minr_task_ped2
CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_minr_task_ped2/checkpoints/ep4-ablate_bev016_minr_task_ped2.ckpt

RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_minr2_global
CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_minr2_global/checkpoints/ep5-ablate_bev016_minr2_global.ckpt

RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev020_dist_dopp_hicap_no_bnfreeze
CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev020_dist_dopp_hicap_no_bnfreeze/checkpoints/ep3-ablate_bev020_dist_dopp_hicap_no_bnfreeze.ckpt

RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_no_bnfreeze_tmp6
CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_no_bnfreeze_tmp6/checkpoints/ep3-ablate_bev016_dist_dopp_hicap_no_bnfreeze_tmp6.ckpt

RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap
CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap/checkpoints/ep3-ablate_bev016_dist_dopp_hicap.ckpt

RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev_res_020
CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev_res_020/checkpoints/ep4-ablate_bev_res_020.ckpt

RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dopp_only
CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dopp_only/checkpoints/ep3-ablate_bev016_dopp_only.ckpt

RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_pfnplus
CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_pfnplus/checkpoints/ep3-ablate_bev016_dist_dopp_hicap_pfnplus.ckpt

RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_featnorm
CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_featnorm/checkpoints/ep5-ablate_bev016_dist_dopp_hicap_featnorm.ckpt

RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_distance_doppler_hi_capacity
CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_distance_doppler_hi_capacity/checkpoints/ep4-ablate_distance_doppler_hi_capacity.ckpt

RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_no_bnfreeze
CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_no_bnfreeze/checkpoints/ep5-ablate_bev016_dist_dopp_hicap_no_bnfreeze.ckpt

RUN_DIR=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_only
CHECKPOINT=/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_only/checkpoints/ep3-ablate_bev016_dist_only.ckpt
TXT
  exit 0
fi

CMD=(
  python -u src/tools/ped_debug.py
  --run-dir "${RUN_DIR}"
  --score-thresholds "${SCORE_THRESHOLDS}"
  --eval-score-threshold "${EVAL_SCORE_THRESHOLD}"
  --iou-threshold "${IOU_THRESHOLD}"
  --num-workers "${NUM_WORKERS}"
  --max-samples "${MAX_SAMPLES}"
  --ped-label-index "${PED_LABEL_INDEX}"
  --device cuda
)

if [[ -n "${CHECKPOINT}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output-dir "${OUTPUT_DIR}")
fi

echo "Running pedestrian debug diagnostics"
printf 'Command: %q ' "${CMD[@]}"; echo
srun "${CMD[@]}"
