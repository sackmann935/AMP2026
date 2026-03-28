# Advanced Machine Perception Final Assignment

This repository is now configured as a **radar-only** 3D detection pipeline.

## Current Scope
- Model: CenterPoint-style radar detector on View of Delft.
- Input: radar point clouds only.
- Fusion: removed from runtime/model code.

## Main Entry Points
- Train: `src/tools/train.py`
- Validate: `src/tools/eval.py`
- Test/submission export: `src/tools/test.py`
- Slurm launcher: `src/tools/slurm_train.sh`

## Recommended Training Profiles
Run from repository root:

```bash
PROFILE=ablate_bev_res_016 sbatch src/tools/slurm_train.sh
```

Other radar-only profiles are listed by:

```bash
PROFILE=list bash src/tools/slurm_train.sh
```

## Notes
- Any attempt to enable `model.fusion.enabled=true` now raises an error.
- Historical fusion experimentation docs are kept for record only.
