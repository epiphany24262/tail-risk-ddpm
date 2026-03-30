# Interface Spec

## prices.parquet
Columns:
- trade_date
- asset
- open
- high
- low
- close
- volume
- amount

Conventions:
- sorted by (`trade_date`, `asset`)
- `trade_date` dtype: datetime64[ns]
- numeric fields are parsed as numeric types
- default output may be unbalanced across assets before their listing dates

## dataset_*.npz
Contains:
- X: shape [num_samples, 20, N]
- C: shape [num_samples, K]
- y_tail: shape [num_samples]
- state_label: shape [num_samples]
- trade_date: shape [num_samples], dtype datetime64[D]
- asset_names: shape [N]
- condition_names: shape [K]
- split: shape [1]

Current values in this repository:
- N = 5 assets
- K = 4 conditions (`cumret_5d`, `vol_20d`, `amount_change_5d`, `high_vol`)
- window length = 20
- files:
  - `data/processed/dataset_train.npz`
  - `data/processed/dataset_valid.npz`
  - `data/processed/dataset_test.npz`

Related files:
- `data/processed/features.parquet`
- `data/processed/tail_stats.json`
- `outputs/tables/b_dataset_composition.csv`
- `outputs/tables/b_feature_descriptive_stats.csv`

## sample outputs
- file name format: sample_{condition}_{ckpt}.npy
- generated paths shape: [num_samples, 20, N]
- dtype: float32

Additional sample artifacts:
- `sample_{condition}_{ckpt}_price.npy`:
  - explicit baseline `t0=1` included
  - shape: [num_samples, 21, N]
  - dtype: float32
- `sample_{condition}_{ckpt}_traj.npy` (when `save_trajectory=true`):
  - sparse reverse-diffusion trajectory
  - dtype: float32
- `sample_{condition}_{ckpt}_traj_steps.npy` (when `save_trajectory=true`):
  - integer step indices aligned with `traj` first dimension
  - order: reverse sampling order from `T` to `0`
  - dtype: int32

Notes:
- `sample_{condition}_{ckpt}.npy` stores 20-step log-return paths.
- `_price.npy` is derived from log-returns via cumulative sum + exponential.
