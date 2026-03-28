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

## sample outputs
- file name format: sample_{condition}_{ckpt}.npy
- generated paths shape: [num_samples, 20, N]
