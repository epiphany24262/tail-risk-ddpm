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
- `notebooks/`: runnable analysis notebooks for A/B/D diagnostics
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
- C module: implemented and runnable
  - scripts: `src/model.py`, `src/diffusion.py`, `src/train.py`, `src/sample.py`
  - outputs: `outputs/checkpoints/latest.pt`, `outputs/checkpoints/best.pt`,
    `outputs/logs/train_log.csv`, `outputs/samples/sample_*.npy`
- D module: implemented and runnable
  - scripts: `src/evaluate.py`, `src/attribution.py`
  - outputs: `outputs/tables/d_*.csv`, `outputs/figures/d_*.png`

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
5. Generate quality summary table automatically during preprocessing:
   - `outputs/tables/a_data_quality_summary.csv`

## Feature Pipeline (B)
1. Build features and split datasets:
   - `python src/make_dataset.py --config configs/data.yaml`
2. Generated files:
  - `data/processed/features.parquet`
  - `data/processed/dataset_train.npz`
  - `data/processed/dataset_valid.npz`
  - `data/processed/dataset_test.npz`
  - `data/processed/tail_stats.json`
  - `outputs/tables/b_dataset_composition.csv`
  - `outputs/tables/b_feature_descriptive_stats.csv`

Notes:
- `make_dataset.py` builds a balanced panel using common dates across selected assets.
- `configs/data.yaml` `start_date` / `end_date` are applied before panel alignment.
- With current asset pool, common-date range is `2020-11-16` to latest trading day.

Data documentation:
- `report/data_dictionary.md`
- `report/data_cleaning_notes.md`

## Model Pipeline (C)
1. Train conditional DDPM:
   - `python src/train.py`
2. Generate conditional samples:
   - `python src/sample.py`
3. Generated files:
   - `outputs/checkpoints/latest.pt`
   - `outputs/checkpoints/best.pt`
   - `outputs/logs/train_log.csv`
   - `outputs/samples/sample_{condition}_{ckpt}.npy`
   - `outputs/samples/sample_{condition}_{ckpt}_price.npy`
- `outputs/samples/sample_{condition}_{ckpt}_traj.npy` (when `save_trajectory: true`)
- `outputs/samples/sample_{condition}_{ckpt}_traj_steps.npy` (when `save_trajectory: true`)

Notes:
- `sample_{condition}_{ckpt}.npy`: 20-step log-return paths `[num_samples, 20, N]`.
- `{condition}` comes from keys under `conditions` in `configs/sample.yaml` (safe-sanitized).
- `_price.npy`: includes explicit baseline `t0=1`, shape `[num_samples, 21, N]`.
- `_traj.npy` is the full reverse-diffusion trajectory in `float32`,
  with aligned step index file `_traj_steps.npy` in `int32`, ordered from `T` to `0`.
- `train.py` supports optional CLI overrides `--config`, `--seed`, `--tail-weight`.
- `sample.py` supports optional CLI overrides `--config`, `--checkpoint`, `--n-samples`.

## Evaluation & Attribution Pipeline (D)
1. Evaluate real vs generated risk metrics:
   - `python src/evaluate.py --checkpoint-filter latest --seed 42`
2. Run single-factor what-if attribution:
   - `python src/attribution.py --seed 42`
3. Generated files:
   - `outputs/tables/d_real_risk_metrics.csv`
   - `outputs/tables/d_generated_risk_metrics.csv`
   - `outputs/tables/d_distribution_compare.csv`
   - `outputs/tables/d_factor_sensitivity.csv`
   - `outputs/tables/d_factor_sensitivity_meta.json`
   - `outputs/figures/d_return_dist_last_day.png`
   - `outputs/figures/d_left_tail_compare_last_day.png`
   - `outputs/figures/d_es_by_condition_last_day.png`
   - `outputs/figures/d_return_dist_cum_20d.png`
   - `outputs/figures/d_left_tail_compare_cum_20d.png`
   - `outputs/figures/d_es_by_condition_cum_20d.png`
   - `outputs/figures/d_factor_sensitivity_es_last_day.png`
   - `outputs/figures/d_factor_sensitivity_es_cum_20d.png`

Notes:
- `evaluate.py` compares real test-set windows with generated samples on two horizons:
  `last_day` and `cum_20d`.
- `evaluate.py` supports `--checkpoint-filter latest|best|all` (default `latest`) to avoid
  mixing checkpoints in one run.
- D risk tables include both legacy columns (`var_5pct`, `es_5pct`) and generic columns
  (`alpha`, `var_alpha`, `es_alpha`) for compatibility and clarity.
- `attribution.py` perturbs one condition factor at a time and reports
  risk-metric deltas (`delta_var_5pct`, `delta_es_5pct`) relative to the base condition.
- `d_factor_sensitivity_meta.json` records the run configuration
  (`checkpoint`, `base_condition`, `n_samples`, factor grids, `timesteps`, `seed`).

## Calibration Workflow
1. Run the calibration grid:
   - `python src/calibrate.py`
2. Calibration policy:
   - fixed seeds: `42`, `52`, `62`
   - fixed tail weights: `1.0`, `3.0`, `5.0`
   - generation/evaluation use `best` checkpoint only
3. Generated files:
   - `outputs/calibration/calibration_summary.csv`
   - `outputs/calibration/seed_{seed}_tailw_{tail_weight}/...`

Notes:
- This stage prioritizes training/sampling calibration before condition expansion or backbone upgrades.
- `calibration_summary.csv` is the main experiment ledger and ranks runs by:
  `mean_wasserstein`, then `mean_abs_es_gap`, then `mean_ks_stat`.
- Formal presentation should prefer the top-ranked `best` checkpoint run rather than `latest`.

## Notebooks
- `notebooks/01_data_check.ipynb`
  - A-module presentation notebook on `prices.parquet` and `a_data_quality_summary.csv`.
  - Focus: data coverage, listing-date heterogeneity, common-date interpretation, and cross-asset correlation.
- `notebooks/02_tail_label.ipynb`
  - B-module presentation notebook on `features.parquet`, `dataset_*.npz`, and B tables.
  - Focus: tail-label balance, split regime differences, and model-ready tensor interface.
- `notebooks/03_figures.ipynb`
  - D-module presentation notebook from `outputs/tables/d_*.csv` and `outputs/figures/d_*.png`.
  - Focus: real-vs-generated risk gaps, distribution mismatch, and condition-level sensitivity interpretation.
  - All three notebooks are presentation-oriented materials with explicit summaries, limitations, and reproducibility context.

Notebook usage:
- Run from project root with Jupyter/Lab.
- Notebook paths auto-detect the project root and work when launched from either the repository root or `notebooks/`.

## Current Scope
Midterm only:
- 5 assets
- daily data from 2015 onward
- 4 conditional factors
- 20-day windows
- conditional DDPM
- VaR/ES + what-if attribution
- current optimization priority: calibration before model expansion
