 # tail-risk-ddpm

MVP for financial tail-risk scenario generation and attribution with conditional DDPM.

## Goal
This repository implements the midterm MVP of our project:
- collect daily data for 5 A-share ETFs/indices
- build return paths and 4 conditional factors
- train a minimal conditional DDPM with tail-weighted loss
- generate 20-day return paths under different market conditions
- evaluate VaR/ES
- perform what-if single-factor attribution

## Repository Structure
- `configs/`: configuration files
- `data/`: raw and processed data
- `notebooks/`: quick inspection and plotting notebooks
- `src/`: core scripts
- `outputs/`: checkpoints, figures, tables, logs
- `report/`: project notes and interface specs

## Team Roles
- A: data engineering (`download_data.py`, `preprocess.py`)
- B: features and dataset (`make_dataset.py`)
- C: model training and sampling (`model.py`, `diffusion.py`, `train.py`, `sample.py`)
- D: evaluation and attribution (`evaluate.py`, `attribution.py`)

## Implementation Status (Current)
- A module: implemented and runnable
  - scripts: `src/download_data.py`, `src/preprocess.py`, `src/check_dates.py`
  - outputs: `data/raw/*.csv`, `data/processed/prices.parquet`
- B module: implemented and runnable
  - script: `src/make_dataset.py`
  - config: `configs/data.yaml`
  - outputs: `data/processed/features.parquet`, `dataset_train.npz`, `dataset_valid.npz`,
    `dataset_test.npz`, `tail_stats.json`
- C module: placeholder files only (not implemented yet)
- D module: placeholder files only (not implemented yet)

## Environment
- Python: 3.10+
- Dependency install:
  - `pip install -r requirements.txt`
  - or `conda install --file requirements.txt` (if using conda env)

## Data Pipeline (A)
1. Download raw data (defaults: 2015-01-01 to today):
   - `python src/download_data.py`
2. Build `prices.parquet` (keep full history by default):
   - `python src/preprocess.py`
3. Optional: enforce common trading dates across all assets:
   - `python src/preprocess.py --align-common-dates`
4. Validate date coverage and schema:
   - `python src/check_dates.py`

## Feature Pipeline (B)
1. Build features and split datasets:
   - `python src/make_dataset.py --config configs/data.yaml`
2. Generated files:
   - `data/processed/features.parquet`
   - `data/processed/dataset_train.npz`
   - `data/processed/dataset_valid.npz`
   - `data/processed/dataset_test.npz`
   - `data/processed/tail_stats.json`

Notes:
- `make_dataset.py` builds a balanced panel using common dates across selected assets.
- With current asset pool, common-date range is `2020-11-16` to latest trading day.

## Current Scope
Midterm only:
- 5 assets
- daily data from 2015 onward
- 4 conditional factors
- 20-day windows
- conditional DDPM
- VaR/ES + what-if attribution
