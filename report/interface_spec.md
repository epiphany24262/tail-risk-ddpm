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

## sample outputs
- file name format: sample_{condition}_{ckpt}.npy
- generated paths shape: [num_samples, 20, N]

Note:
- Sampling outputs are not available yet because `src/sample.py` is currently a placeholder.
