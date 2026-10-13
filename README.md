# Freshness-Aware Passive Overlay Control with a Conditional World Model

This repository contains the Python implementation of a conditional world model
(CWM) and predictive controller for a Web-connected robotic swarm. It is the
**control and evaluation component** of a larger cross-layer study; the
lower-layer MATLAB simulators, generated datasets, PHY lookup table, trained
weights, and reported figures are not included in this code-only release.

## Problem and approach

Robots already send Wi-Fi/WoT packets carrying telemetry and sensing data.
Low-power peripherals can embed status bits in those occupied packets by
attenuating part of the host waveform, avoiding a separate channel-access
attempt. The controller chooses three coupled variables at each decision epoch:

| Action | Meaning | Main trade-off |
| --- | --- | --- |
| Attenuation depth `delta_dB` | Separation between passive symbols | Passive detection versus host distortion |
| Passive rate `rb_kbps` | Embedded status-bit rate | More bits versus shorter detection time |
| Host MCS `mcs` | Host modulation and coding scheme | Nominal rate versus reliability and packet duration |

An action is considered only if it passes host/passive packet-error and
packet-duration constraints from the PHY lookup table. Among feasible actions,
the controller balances delivered throughput against **Age of Information
(AoI)**, the time since the freshest delivered status update was generated.

Instead of learning a value function for one reward setting, the CWM learns
*action-conditioned service dynamics*. The same trained predictor can then be
used with different throughput-freshness weights without retraining.

## Model and control pipeline

1. **Offline traces.** A cross-layer simulator generates consecutive
   transitions containing local wireless state, shared resource context,
   action, next state, delivered rates, and AoI. The external train/validation/
   test files should be split by trajectory, not by adjacent packet epoch.
2. **Shared CWM.** A state encoder, action encoder, GRU-based latent transition,
   and decoder predict the next service state and resource context. All robots
   share the same model parameters; resource identity and contention enter as
   inputs.
3. **Multi-step training.** Continuous features are normalized with training
   statistics. After a one-step warm-up, the model is trained on autoregressive
   windows with a discounted, state-and-context prediction loss. The full
   configuration uses a five-step training rollout.
4. **Local predictive control.** Each robot masks infeasible actions, rolls out
   candidate actions through the CWM, and scores predicted delivery and AoI
   using a configurable freshness weight `lambda_a`. Only the first action of
   the best sequence is executed; the robot observes the new state and replans.
   The default planning horizon is three steps with discount `beta = 0.95`.

The paper presents the normalized throughput-minus-AoI objective; the formal
closed-loop script uses a scale-aware utility with logarithmic delivered-rate
terms and a passive-delivery-quality term. Consult the script when comparing
its numerical results with the paper formulation.

The Python data interface is `X` (9 current state/context features), `A` (3
action features), and `Y` (6 next-state/context targets). `X` includes SINR,
host/passive rates, AoI, three resource indicators, channel-busy ratio, and
contender load. `Y` predicts the next SINR, rates, AoI, busy ratio, and load.
See `DATA_FORMAT.md` for column order and trajectory metadata.

## Repository map

| Path | Contents |
| --- | --- |
| `cwm/model.py`, `cwm/data.py`, `cwm/losses.py` | CWM architecture, trajectory windows, rollout objective |
| `cwm/planner.py`, `cwm/phy_lut.py` | Counterfactual beam search and action feasibility |
| `scripts/train.py`, `scripts/evaluate.py` | Training and held-out prediction evaluation |
| `scripts/run_closed_loop_evaluation.py` | Fixed, MCS-only, CWM-H1/H3, and freshness-weight sweep |
| `scripts/train_masked_dqn.py`, `scripts/run_additional_baselines.py` | Masked-DQN and exact one-step reference |
| `configs/`, `tests/` | Experiment settings and code checks |

## Installation and data

Use Python 3.10 or newer. Install a PyTorch build appropriate for the target
CPU/GPU, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/smoke_test.py
python -m pytest -q
```

The smoke test is data-independent. Two LUT-dependent unit tests are skipped
until the external PHY table is present. Place the experiment inputs at:

```text
data/stage2_cwm_dataset/train/train_dataset.mat
data/stage2_cwm_dataset/val/val_dataset.mat
data/stage2_cwm_dataset/test/test_dataset.mat
data/stage2_cwm_dataset/unseen_density/unseen_density_dataset.mat
data/stage1_phy_lut/phy_lut_4d.csv
```

The MAT files contain Stage-2 trajectory data; the CSV is the Stage-1 PHY
lookup table. The MATLAB generators are separate from this Python repository.
Check the inputs before training:

```bash
python scripts/check_environment.py --config configs/default.yaml
```

## Training and evaluation

```bash
# Offline world-model training and held-out prediction tests
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml

# Inspect model-based action selection on held-out states
python scripts/plan_actions.py --config configs/default.yaml --split test --num-states 200 --horizon 3 --lambda-a 0.5

# Small closed-loop check; omit --quick for the full experiment
python scripts/run_closed_loop_evaluation.py --quick
```

Training writes the best checkpoint to
`outputs/default_run/checkpoints/best.pt`; subsequent CWM commands expect it.
The closed-loop evaluation also requires the PHY table. The full comparison
uses five robot densities (`5, 10, 20, 30, 40` per resource), three resources,
five seeds, and 300 decision epochs per episode, with periodic and bursty
status traffic. The script additionally sweeps `lambda_a` over
`0, 0.1, 0.3, 1, 3, 10` while reusing the trained CWM. These are **evaluation
settings**, not performance claims.

For optional additional baselines, first train the masked-DQN policy, then
run the comparison script:

```bash
python scripts/train_masked_dqn.py
python scripts/run_additional_baselines.py --quick
```

Here `exact_h1` uses the simulator for **one-step** action evaluation. It is an
evaluation reference, not the deployed learned controller or an implementation
of the paper's full multi-step exact-model MPC. The MATLAB communication-only
optimizer is not part of this repository.

## What to compare

- **Prediction fidelity:** one- to five-step NRMSE and R² for host/passive
  rate, AoI, channel-busy ratio, and contender load on held-out trajectories.
- **Closed-loop service:** aggregate throughput, mean and P95 AoI, freshness
  probability, constraint-violation rate, idle rate, and decision latency.
- **Controls:** fixed overlay, MCS-only, CWM-H1, CWM-H3, masked-DQN, and the
  exact one-step reference. Vary `lambda_a` to inspect the throughput-AoI
  trade-off; use density scaling and model ablations to inspect robustness.

Generated checkpoints, metrics, and plots are written under `outputs/`, which
is intentionally ignored by Git. This repository does not include the original
experiment inputs or weights and therefore cannot reproduce numerical paper
results on its own. Illustrative or manually adjusted figures must not be
interpreted as measured outcomes.
