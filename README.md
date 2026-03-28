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

## Current Scope
Midterm only:
- 5 assets
- daily data from 2015 onward
- 4 conditional factors
- 20-day windows
- conditional DDPM
- VaR/ES + what-if attribution
