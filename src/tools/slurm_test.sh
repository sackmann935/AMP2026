#!/bin/bash
#SBATCH --job-name="centerpoint_test_ro47020"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --account=education-me-courses-ro47020
#SBATCH --mail-type=END
#SBATCH --chdir=/home/lsackmann/final_assignment
#SBATCH --output=/home/lsackmann/final_assignment/outputs/slurm_test_ro47020_%j.out
#SBATCH --error=/home/lsackmann/final_assignment/outputs/slurm_test_ro47020_%j.err

set -euo pipefail

if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  echo "Missing CHECKPOINT_PATH."
  echo "Usage:"
  echo "  CHECKPOINT_PATH=/abs/path/to/model.ckpt sbatch src/tools/slurm_test.sh"
  exit 2
fi

module load 2024r1 miniconda3/4.12.0 cuda/12.5

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amp

if [[ -z "${EXP_ID:-}" ]]; then
  # expected checkpoint path layout: outputs/<exp_id>/checkpoints/*.ckpt
  EXP_ID="$(basename "$(dirname "$(dirname "$CHECKPOINT_PATH")")")"
fi

RADAR_SOURCE=${RADAR_SOURCE:-radar_5frames}
NUM_WORKERS=${NUM_WORKERS:-2}
MAKE_ZIP=${MAKE_ZIP:-true}

CMD=(
  python -u src/tools/test.py
  model=centerpoint_radar
  checkpoint_path="${CHECKPOINT_PATH}"
  exp_id="${EXP_ID}"
  output_dir="outputs/${EXP_ID}"
  radar_source="${RADAR_SOURCE}"
  num_workers="${NUM_WORKERS}"
)

echo "Running test for EXP_ID=${EXP_ID}"
printf 'Command: %q ' "${CMD[@]}"; echo
srun "${CMD[@]}"

if [[ "${MAKE_ZIP}" == "true" ]]; then
  RES_FOLDER="outputs/${EXP_ID}/test_preds"
  ZIP_PATH="outputs/${EXP_ID}/submission.zip"
  echo "Creating submission zip: ${ZIP_PATH}"
  python -u src/tools/zip_files.py --res_folder "${RES_FOLDER}" --output_path "${ZIP_PATH}"
fi
