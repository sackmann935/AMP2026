#!/usr/bin/env bash
set -euo pipefail

# Quick GPU smoke test:
# - initializes config/model/dataset
# - runs only a few training iterations
# - verifies the first optimization steps succeed
#
# Usage (from anywhere):
#   bash src/tools/smoke_train.sh
#   bash src/tools/smoke_train.sh hard
#   SMOKE_BATCHES=4 SMOKE_BATCH_SIZE=2 bash src/tools/smoke_train.sh gaussian_soft
#   bash src/tools/smoke_train.sh gaussian_soft cmx_lite
#   bash src/tools/smoke_train.sh hard off

MIDDLE_ENCODER_TYPE="${1:-}" # optional: hard | gaussian_soft
FUSION_TYPE="${2:-}"         # optional: off | add | concat_1x1 | gated | cmx_lite

SMOKE_BATCHES="${SMOKE_BATCHES:-2}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
SMOKE_SUBSET_SIZE="${SMOKE_SUBSET_SIZE:-8}"
SMOKE_SEED="${SMOKE_SEED:-47020}"
export MIDDLE_ENCODER_TYPE
export SMOKE_BATCHES
export SMOKE_BATCH_SIZE
export SMOKE_NUM_WORKERS
export SMOKE_SUBSET_SIZE
export SMOKE_SEED
export FUSION_TYPE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[smoke] repo root: ${REPO_ROOT}"
echo "[smoke] python: ${PYTHON_BIN}"
echo "[smoke] middle_encoder.type override: ${MIDDLE_ENCODER_TYPE:-<none>}"
echo "[smoke] fusion.type override: ${FUSION_TYPE:-<none>}"
echo "[smoke] batches=${SMOKE_BATCHES}, batch_size=${SMOKE_BATCH_SIZE}, workers=${SMOKE_NUM_WORKERS}, subset=${SMOKE_SUBSET_SIZE}"

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import lightning as L
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, Subset

from src.dataset import ViewOfDelft, collate_vod_batch
from src.model.detector import CenterPoint

middle_encoder_type = os.environ.get("MIDDLE_ENCODER_TYPE", "").strip()
fusion_type = os.environ.get("FUSION_TYPE", "").strip()
smoke_batches = int(os.environ.get("SMOKE_BATCHES", "2"))
smoke_batch_size = int(os.environ.get("SMOKE_BATCH_SIZE", "1"))
smoke_num_workers = int(os.environ.get("SMOKE_NUM_WORKERS", "0"))
smoke_subset_size = int(os.environ.get("SMOKE_SUBSET_SIZE", "8"))
smoke_seed = int(os.environ.get("SMOKE_SEED", "47020"))

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available. Run this inside a GPU interactive session.")

L.seed_everything(smoke_seed, workers=True)

config_dir = str((Path.cwd() / "src" / "config").resolve())
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name="train")

if middle_encoder_type:
    cfg.model.middle_encoder.type = middle_encoder_type
if fusion_type:
    if fusion_type.lower() in {"off", "false", "0", "disabled"}:
        cfg.model.fusion.enabled = False
    else:
        cfg.model.fusion.enabled = True
        cfg.model.fusion.type = fusion_type

cfg.batch_size = smoke_batch_size
cfg.num_workers = smoke_num_workers
fusion_enabled = bool(getattr(getattr(cfg.model, "fusion", {}), "enabled", False))

train_dataset = ViewOfDelft(
    data_root=cfg.data_root,
    split="train",
    include_camera=fusion_enabled,
)
subset_len = min(smoke_subset_size, len(train_dataset))
train_subset = Subset(train_dataset, range(subset_len))
train_loader = DataLoader(
    train_subset,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    shuffle=False,
    collate_fn=collate_vod_batch,
)

model = CenterPoint(cfg.model)

trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=1,
    limit_train_batches=smoke_batches,
    limit_val_batches=0,
    num_sanity_val_steps=0,
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    log_every_n_steps=1,
)

trainer.fit(model, train_dataloaders=train_loader)
print("SMOKE_TRAIN_OK")
PY
