# FreshOverlay CWM — minimal Python code

This is a code-only package for the conditional world model, its training and
evaluation, and closed-loop controller comparisons. It does **not** include
datasets, PHY lookup tables, trained checkpoints, generated results, or the
illustrative `SAMPLE` figures. Those inputs must be supplied separately.

The MATLAB Stage 1 PHY simulator and Stage 2 trajectory generator are separate
projects and are **not** included here. See `DATA_FORMAT.md` for
the expected Stage 2 arrays.

## Layout

```text
cwm/       model, data loader, dynamics, planner, and LUT reader
scripts/   training, evaluation, closed-loop baselines, and checks
configs/   quick and full experiment settings
tests/     small code tests
data/      empty input locations (large files are ignored by Git)
outputs/   generated files (ignored by Git)
```

## Install and check

Use Python 3.10 or newer. Install a PyTorch build suitable for your CPU/GPU,
then run from this folder:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/smoke_test.py
python -m pytest -q
```

The smoke test and data-independent unit tests do not need external inputs;
the two LUT-dependent tests are skipped until the LUT is supplied. For the
environment/data check, run this after placing the inputs below:

```bash
python scripts/check_environment.py --config configs/default.yaml
```

## External inputs

```text
data/stage2_cwm_dataset/train/train_dataset.mat
data/stage2_cwm_dataset/val/val_dataset.mat
data/stage2_cwm_dataset/test/test_dataset.mat
data/stage2_cwm_dataset/unseen_density/unseen_density_dataset.mat
data/stage1_phy_lut/phy_lut_4d.csv
```

The four MAT files are Stage 2 MATLAB v7.3/HDF5 outputs. The CSV is the Stage 1
PHY LUT used for action masking and closed-loop evaluation. Do not commit
private or large inputs unless you have permission to publish them.

## Basic run order

```bash
python scripts/train.py --config configs/quick.yaml
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
python scripts/plan_actions.py --config configs/default.yaml --split test --num-states 200 --horizon 3 --lambda-a 0.5
python scripts/run_closed_loop_evaluation.py --quick
```

The final command also needs the trained checkpoint at
`outputs/default_run/checkpoints/best.pt` and the Stage 1 LUT CSV. `--quick` is
only a smoke-sized evaluation; omit it for the full run. Results are written
under `outputs/` and are not tracked by Git.

This package does not claim to reproduce reported paper metrics without the
original datasets, configuration, and checkpoints. In particular, the
previously prepared `SAMPLE` plots are visual examples, not measured results.
