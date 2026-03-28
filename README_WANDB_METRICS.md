# W&B Metrics Extraction Playbook

This file is the canonical guide for extracting ablation metrics quickly and consistently.

## Scope
- Project: `/home/lsackmann/final_assignment`
- Primary score: `validation/ROI/mAP`
- Class-wise ROI metrics:
  - `validation/ROI/Car_3d`
  - `validation/ROI/Pedestrian_3d`
  - `validation/ROI/Cyclist_3d`

## Environment (important)
`wandb` is available in the `amp` env python binary:

```bash
/home/lsackmann/.conda/envs/amp/bin/python
```

Use that directly. Do not rely on `conda` being on `PATH`.

## 1) Find the W&B run id(s)
From a run output directory:

```bash
ls -1 outputs/<exp_id>/wandb_logs/wandb
# format: run-YYYYMMDD_HHMMSS-<run_id>
```

Or from slurm logs:

```bash
rg -n "View run at https://wandb.ai/.*/runs/" outputs/<exp_id>/slurm_*.err outputs/<exp_id>/slurm_*.out
```

## 2) Pull final summary metrics from W&B
Replace run ids as needed.

```bash
/home/lsackmann/.conda/envs/amp/bin/python - <<'PY'
import wandb

RUN_IDS = ["io1j80l2", "u96ytsyl"]  # edit
PROJECT = "l-v-sackmann-tu-delft/amp"
KEYS = [
    "validation/ROI/mAP",
    "validation/ROI/Car_3d",
    "validation/ROI/Pedestrian_3d",
    "validation/ROI/Cyclist_3d",
    "validation/loss",
    "train/loss",
]

api = wandb.Api(timeout=40)
for rid in RUN_IDS:
    run = api.run(f"{PROJECT}/{rid}")
    print(f"\n===== {run.name} ({rid}) =====")
    for k in KEYS:
        print(f"{k}: {run.summary.get(k)}")
PY
```

## 3) Pull per-epoch ROI trajectory (recommended)
Use `scan_history` to get all logged epoch points.

```bash
/home/lsackmann/.conda/envs/amp/bin/python - <<'PY'
import wandb

RUN_ID = "io1j80l2"  # edit
PROJECT = "l-v-sackmann-tu-delft/amp"
KEYS = [
    "epoch",
    "validation/ROI/mAP",
    "validation/ROI/Car_3d",
    "validation/ROI/Pedestrian_3d",
    "validation/ROI/Cyclist_3d",
    "validation/loss",
    "train/loss",
]

api = wandb.Api(timeout=40)
run = api.run(f"{PROJECT}/{RUN_ID}")
print(f"===== {run.name} ({RUN_ID}) =====")

for row in run.scan_history(keys=KEYS):
    if "validation/ROI/mAP" not in row:
        continue
    print(
        f"epoch={int(row['epoch'])} "
        f"mAP={row['validation/ROI/mAP']:.3f} "
        f"Car={row['validation/ROI/Car_3d']:.3f} "
        f"Ped={row['validation/ROI/Pedestrian_3d']:.3f} "
        f"Cyc={row['validation/ROI/Cyclist_3d']:.3f}"
    )
PY
```

## 4) Compare runs by peak vs final ROI/mAP
Useful for diagnosing instability.

```bash
/home/lsackmann/.conda/envs/amp/bin/python - <<'PY'
import wandb

RUN_IDS = ["io1j80l2", "u96ytsyl", "fpdv2ruj", "v60ys2dk", "brsw04rd"]  # edit
PROJECT = "l-v-sackmann-tu-delft/amp"

api = wandb.Api(timeout=40)
for rid in RUN_IDS:
    run = api.run(f"{PROJECT}/{rid}")
    rows = list(run.scan_history(keys=[
        "epoch",
        "validation/ROI/mAP",
        "validation/ROI/Car_3d",
        "validation/ROI/Pedestrian_3d",
        "validation/ROI/Cyclist_3d",
    ]))
    rows = [r for r in rows if "validation/ROI/mAP" in r]
    if not rows:
        continue
    best = max(rows, key=lambda r: r["validation/ROI/mAP"])
    final = rows[-1]
    print(f"\n===== {run.name} ({rid}) =====")
    print(
        f"best: epoch={int(best['epoch'])} mAP={best['validation/ROI/mAP']:.3f} "
        f"(Car={best['validation/ROI/Car_3d']:.3f}, "
        f"Ped={best['validation/ROI/Pedestrian_3d']:.3f}, "
        f"Cyc={best['validation/ROI/Cyclist_3d']:.3f})"
    )
    print(
        f"final: epoch={int(final['epoch'])} mAP={final['validation/ROI/mAP']:.3f} "
        f"(Car={final['validation/ROI/Car_3d']:.3f}, "
        f"Ped={final['validation/ROI/Pedestrian_3d']:.3f}, "
        f"Cyc={final['validation/ROI/Cyclist_3d']:.3f})"
    )
    print(f"drop(best->final): {best['validation/ROI/mAP'] - final['validation/ROI/mAP']:.3f}")
PY
```

## 5) Fallback: parse local slurm log only (no W&B)
Extract all ROI evaluation blocks from an `.out` log:

```bash
awk '
  /Driving corridor area:/ {dc=1; next}
  dc==1 && /^Car:/ {car=$2; next}
  dc==1 && /^Pedestrian:/ {ped=$2; next}
  dc==1 && /^Cyclist:/ {cyc=$2; next}
  dc==1 && /^mAP:/ {map=$2; i++; printf("eval_%d ROI Car=%.3f Ped=%.3f Cyc=%.3f mAP=%.3f\n", i, car, ped, cyc, map); dc=0}
' outputs/<exp_id>/slurm_*.out
```

## 6) Known interpretation rules for this project
- Ignore the initial sanity-check metric block with all zeros.
- For `epochs=6`, expect 6 real validation points (`epoch=0..5`).
- `src/tools/slurm_train.sh` sets `model.head.train_cfg.min_radius=1` globally unless a profile overrides it.
- Late slurm cancellation during W&B upload can still happen after training completed; metrics can still be valid if final validation was logged.

## 7) Quick checklist before analysis
- Confirm run ids.
- Pull per-epoch ROI trajectory (not just summary).
- Report both:
  - best ROI/mAP epoch
  - final ROI/mAP epoch
- Include class-wise ROI values for Car/Ped/Cyclist.
