# Data Cleaning Notes

## A Module Scope

The A module builds a reproducible base price table for the selected asset pool:
- download raw daily data by configured assets
- standardize columns
- clean invalid rows
- export `data/processed/prices.parquet`

## Raw-to-Standard Mapping

Raw CSV Chinese columns are mapped to standard names:
- `日期` -> `trade_date`
- `开盘` -> `open`
- `最高` -> `high`
- `最低` -> `low`
- `收盘` -> `close`
- `成交量` -> `volume`
- `成交额` -> `amount`

## Cleaning Rules

For each asset file:
1. Keep only required columns listed above.
2. Parse `trade_date` as datetime, numeric columns as numeric.
3. Drop rows missing any key fields (`trade_date`, `open`, `high`, `low`, `close`).
4. Drop rows with non-positive prices (`open/high/low/close <= 0`).
5. Drop duplicate rows by (`trade_date`, `asset`) keeping first.
6. Sort by date.

After concatenation:
- sort by (`trade_date`, `asset`)
- optional `--align-common-dates` keeps only dates available for all selected assets

## Quality Outputs

`src/preprocess.py` also saves:
- `outputs/tables/a_data_quality_summary.csv`

This file contains:
- per-asset source file, date range, trading-day count
- raw rows vs final rows
- key missing rows, dropped NA rows
- invalid-price rows, duplicate rows removed
- overall missing rate

## Impact of Common-Date Alignment

- Without alignment: keeps full per-asset history (unbalanced panel).
- With alignment: returns a balanced panel for multi-asset modeling but may shorten the start date because of newer listings.
