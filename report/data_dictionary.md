# Data Dictionary

## `data/processed/prices.parquet`

| Column | Type | Meaning |
|---|---|---|
| `trade_date` | datetime64[ns] | Trading date of the asset quote |
| `asset` | string | Asset identifier (e.g., `hs300_etf`) |
| `open` | float | Open price of the day |
| `high` | float | High price of the day |
| `low` | float | Low price of the day |
| `close` | float | Close price of the day |
| `volume` | numeric | Trading volume |
| `amount` | float | Trading amount/turnover |

Notes:
- `prices.parquet` is sorted by (`trade_date`, `asset`).
- Default preprocessing keeps full history by asset; optional common-date alignment is available.

## `data/processed/features.parquet`

| Column group | Meaning |
|---|---|
| `portfolio_return` | Equal-weight portfolio log return |
| `tail_loss` | `-portfolio_return` |
| `cumret_5d` | 5-day cumulative portfolio return |
| `vol_20d` | 20-day rolling volatility of portfolio return |
| `amount_change_5d` | 5-day log change of portfolio amount |
| `high_vol` | Binary high-vol regime label (median split on train vol) |
| `y_tail` | Binary tail-event label (`tail_loss` >= train 95% quantile) |
| `ret_*` | Per-asset log return columns |
| `split` | Time split label (`train`/`valid`/`test`) |

## `data/processed/dataset_*.npz`

| Key | Shape | Meaning |
|---|---|---|
| `X` | `[num_samples, 20, N]` | Log-return paths |
| `C` | `[num_samples, 4]` | Condition vectors (`cumret_5d`, `vol_20d`, `amount_change_5d`, `high_vol`) |
| `y_tail` | `[num_samples]` | Tail-event label |
| `state_label` | `[num_samples]` | High-vol regime label |
| `trade_date` | `[num_samples]` | Sample end dates |
| `asset_names` | `[N]` | Asset channel names |
| `condition_names` | `[4]` | Condition names |
| `split` | `[1]` | Split name |
