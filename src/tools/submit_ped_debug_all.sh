#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SLURM_SCRIPT="${REPO_ROOT}/src/tools/slurm_ped_debug.sh"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "Could not find ${SLURM_SCRIPT}"
  exit 1
fi

# name|run_dir|checkpoint
PRESETS=(
  "bev016_best|/home/lsackmann/final_assignment/outputs/ablate_bev_res_016|/home/lsackmann/final_assignment/outputs/ablate_bev_res_016/checkpoints/ep4-ablate_bev_res_016-v1.ckpt"
  "dist_dopp_tamed|/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_temporal_tamed|/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_temporal_tamed/checkpoints/ep2-ablate_bev016_dist_dopp_hicap_temporal_tamed.ckpt"
  "minr_task_ped2|/home/lsackmann/final_assignment/outputs/ablate_bev016_minr_task_ped2|/home/lsackmann/final_assignment/outputs/ablate_bev016_minr_task_ped2/checkpoints/ep4-ablate_bev016_minr_task_ped2.ckpt"
  "minr2_global|/home/lsackmann/final_assignment/outputs/ablate_bev016_minr2_global|/home/lsackmann/final_assignment/outputs/ablate_bev016_minr2_global/checkpoints/ep5-ablate_bev016_minr2_global.ckpt"
  "bev020_no_bnfreeze|/home/lsackmann/final_assignment/outputs/ablate_bev020_dist_dopp_hicap_no_bnfreeze|/home/lsackmann/final_assignment/outputs/ablate_bev020_dist_dopp_hicap_no_bnfreeze/checkpoints/ep3-ablate_bev020_dist_dopp_hicap_no_bnfreeze.ckpt"
  "no_bnfreeze_tmp6|/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_no_bnfreeze_tmp6|/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_no_bnfreeze_tmp6/checkpoints/ep3-ablate_bev016_dist_dopp_hicap_no_bnfreeze_tmp6.ckpt"
  "dist_dopp_hicap|/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap|/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap/checkpoints/ep3-ablate_bev016_dist_dopp_hicap.ckpt"
  "bev020_res|/home/lsackmann/final_assignment/outputs/ablate_bev_res_020|/home/lsackmann/final_assignment/outputs/ablate_bev_res_020/checkpoints/ep4-ablate_bev_res_020.ckpt"
  "dopp_only|/home/lsackmann/final_assignment/outputs/ablate_bev016_dopp_only|/home/lsackmann/final_assignment/outputs/ablate_bev016_dopp_only/checkpoints/ep3-ablate_bev016_dopp_only.ckpt"
  "pfnplus|/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_pfnplus|/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_pfnplus/checkpoints/ep3-ablate_bev016_dist_dopp_hicap_pfnplus.ckpt"
  "featnorm|/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_featnorm|/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_featnorm/checkpoints/ep5-ablate_bev016_dist_dopp_hicap_featnorm.ckpt"
  "dist_dopp_hi_capacity|/home/lsackmann/final_assignment/outputs/ablate_distance_doppler_hi_capacity|/home/lsackmann/final_assignment/outputs/ablate_distance_doppler_hi_capacity/checkpoints/ep4-ablate_distance_doppler_hi_capacity.ckpt"
  "no_bnfreeze|/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_no_bnfreeze|/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_dopp_hicap_no_bnfreeze/checkpoints/ep5-ablate_bev016_dist_dopp_hicap_no_bnfreeze.ckpt"
  "dist_only|/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_only|/home/lsackmann/final_assignment/outputs/ablate_bev016_dist_only/checkpoints/ep3-ablate_bev016_dist_only.ckpt"
)

usage() {
  cat <<'TXT'
Submit pedestrian debug jobs for all ranked checkpoint presets.

Usage:
  bash src/tools/submit_ped_debug_all.sh [all|top5|top10|list]

Optional env overrides:
  MAX_SAMPLES=0                 # 0 = full val set, >0 = quick debug
  SCORE_THRESHOLDS=0.03,0.05,...
  EVAL_SCORE_THRESHOLD=0.1
  IOU_THRESHOLD=0.25
  NUM_WORKERS=2
  PED_LABEL_INDEX=-1
  SLEEP_BETWEEN_SUBMITS=0.2
  DRY_RUN=0                     # 1 = print sbatch commands only
  OUTPUT_TAG=all                # output folder suffix

Examples:
  bash src/tools/submit_ped_debug_all.sh list
  MAX_SAMPLES=400 bash src/tools/submit_ped_debug_all.sh top5
  bash src/tools/submit_ped_debug_all.sh all
TXT
}

MODE="${1:-all}"
case "${MODE}" in
  list|all|top5|top10) ;;
  -h|--help|help) usage; exit 0 ;;
  *)
    echo "Unknown mode: ${MODE}"
    usage
    exit 2
    ;;
esac

select_count=${#PRESETS[@]}
if [[ "${MODE}" == "top5" ]]; then
  select_count=5
elif [[ "${MODE}" == "top10" ]]; then
  select_count=10
fi

if [[ "${MODE}" == "list" ]]; then
  echo "Available presets:"
  for i in "${!PRESETS[@]}"; do
    IFS='|' read -r name run_dir ckpt <<< "${PRESETS[$i]}"
    printf "%2d. %-28s | %s\n" "$((i+1))" "${name}" "${ckpt}"
  done
  exit 0
fi

MAX_SAMPLES="${MAX_SAMPLES:-0}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-0.03,0.05,0.1,0.2,0.3,0.5}"
EVAL_SCORE_THRESHOLD="${EVAL_SCORE_THRESHOLD:-0.1}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.25}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PED_LABEL_INDEX="${PED_LABEL_INDEX:--1}"
SLEEP_BETWEEN_SUBMITS="${SLEEP_BETWEEN_SUBMITS:-0.2}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_TAG="${OUTPUT_TAG:-all}"

echo "Submitting ${select_count} ped_debug jobs (mode=${MODE})"
echo "MAX_SAMPLES=${MAX_SAMPLES} | IOU_THRESHOLD=${IOU_THRESHOLD} | SCORE_THRESHOLDS=${SCORE_THRESHOLDS}"
echo

for i in $(seq 0 $((select_count - 1))); do
  IFS='|' read -r name run_dir ckpt <<< "${PRESETS[$i]}"
  out_dir="${run_dir}/ped_debug_${OUTPUT_TAG}_${name}"
  echo "[$((i+1))/${select_count}] ${name}"
  echo "  run_dir: ${run_dir}"
  echo "  ckpt:    ${ckpt}"
  echo "  out:     ${out_dir}"

  cmd=(
    env
    RUN_DIR="${run_dir}"
    CHECKPOINT="${ckpt}"
    OUTPUT_DIR="${out_dir}"
    MAX_SAMPLES="${MAX_SAMPLES}"
    SCORE_THRESHOLDS="${SCORE_THRESHOLDS}"
    EVAL_SCORE_THRESHOLD="${EVAL_SCORE_THRESHOLD}"
    IOU_THRESHOLD="${IOU_THRESHOLD}"
    NUM_WORKERS="${NUM_WORKERS}"
    PED_LABEL_INDEX="${PED_LABEL_INDEX}"
    sbatch
    "${SLURM_SCRIPT}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  DRY_RUN cmd: %q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
  sleep "${SLEEP_BETWEEN_SUBMITS}"
done

echo
if [[ "${MAX_SAMPLES}" == "0" ]]; then
  echo "Submitted full-val diagnostics. Feasible runtime depends on queue parallelism."
  echo "Rough walltime per job: <= 1.5h (from slurm_ped_debug.sh limit)."
else
  echo "Submitted quick diagnostics with MAX_SAMPLES=${MAX_SAMPLES}."
fi
