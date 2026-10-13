# Stage-2 Data Format Expected by the Python CWM

The loader directly reads the MATLAB `-v7.3` HDF5 MAT files produced by Stage 2.

## X — current state/context (9 columns)

```text
0 sinr_t_dB
1 host_rate_t_Mbps
2 passive_rate_t_kbps
3 aoi_t_ms
4 resource_1
5 resource_2
6 resource_3
7 cbr_t
8 contender_load_t
```

## A — current action (3 columns)

```text
0 delta_dB
1 rb_kbps
2 mcs
```

## Y — next dynamic state/context (6 columns)

```text
0 sinr_next_dB
1 host_rate_next_Mbps
2 passive_rate_next_kbps
3 aoi_next_ms
4 cbr_next
5 contender_load_next
```

## Meta — trajectory bookkeeping

Only these fields are needed to build non-crossing rollout windows:

```text
0 episode_id
3 resource_id
4 robot_id
5 time_step
```

The Python loader sorts by `(episode_id, resource_id, robot_id, time_step)` if necessary and constructs H-step windows only when time indices are consecutive.

## Required directory layout

```text
data/stage2_cwm_dataset/
├── train/train_dataset.mat
├── val/val_dataset.mat
├── test/test_dataset.mat
└── unseen_density/unseen_density_dataset.mat
```

The Stage-1 LUT CSV is optional during CWM training but required for the formal feasibility-mask controller:

```text
data/stage1_phy_lut/phy_lut_4d.csv
```
