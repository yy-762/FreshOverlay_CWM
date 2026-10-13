# Conditional World Model for Passive Overlay Control

This repository contains the Python implementation of a conditional world model
(CWM) for freshness-aware control in a passive overlay network. The model
predicts how candidate communication actions affect throughput and Age of
Information (AoI). A short-horizon planner uses those predictions to select
actions subject to feasibility constraints.

The code includes model training, prediction evaluation, action planning, and
closed-loop comparisons with fixed, MCS-only, and masked-DQN controllers. It is
a **code-only release**: datasets, PHY lookup tables, checkpoints, and generated
figures are not included.

## Repository contents

| Path | Purpose |
| --- | --- |
| `cwm/` | World model, data loading, action planner, and PHY lookup utilities |
| `scripts/` | Training, evaluation, controller comparisons, and checks |
| `configs/` | Quick and full experiment configurations |
| `tests/` | Unit tests and a data-independent smoke test |
| `data/` | Locations for external experiment inputs |
| `outputs/` | Generated checkpoints and results (ignored by Git) |

## Quick start

Use Python 3.10 or newer. Install a PyTorch build appropriate for your machine,
then run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/smoke_test.py
python -m pytest -q
```

The smoke test does not need experiment data. Two LUT-dependent tests are
skipped until the PHY lookup table is available.

## Experiment inputs

Place the externally generated files at these paths:

```text
data/stage2_cwm_dataset/train/train_dataset.mat
data/stage2_cwm_dataset/val/val_dataset.mat
data/stage2_cwm_dataset/test/test_dataset.mat
data/stage2_cwm_dataset/unseen_density/unseen_density_dataset.mat
data/stage1_phy_lut/phy_lut_4d.csv
```

The MAT files contain the trajectory windows used to train and evaluate the
world model; the CSV contains the PHY lookup table used by the planner and
closed-loop simulator. See `DATA_FORMAT.md` for the expected data format. The
MATLAB generators for these inputs are not part of this Python release.

## Run the experiments

```bash
python scripts/check_environment.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
python scripts/plan_actions.py --config configs/default.yaml --split test --num-states 200 --horizon 3 --lambda-a 0.5
python scripts/run_closed_loop_evaluation.py --quick
```

The last command expects a trained checkpoint at
`outputs/default_run/checkpoints/best.pt` and the PHY lookup table above. Its
`--quick` option is for a small validation run; omit it for the full evaluation.
All generated files are written under `outputs/`.

Reported paper results cannot be reproduced from this repository alone without
the original experiment inputs and trained checkpoints. No illustrative or
manually adjusted plots are included as measured results.
