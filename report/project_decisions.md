# Project Decisions

## Data
- Asset pool: Scheme A
- Time range: 2015-01-01 to latest trading day
- Frequency: daily
- Return type: log return
- Note: because assets have different listing dates, default `prices.parquet`
  keeps full history (unbalanced panel). Use common-date alignment only when a
  balanced panel is explicitly required.

## Portfolio
- Equal-weight portfolio
- Rationale: MVP baseline with minimal assumptions and easy reproducibility.
- Future extension: add weighting sensitivity checks (e.g., value-weighted / risk-parity) in final stage.

## Tail Definition
- Loss = - portfolio return
- Tail flag if loss >= 95th percentile
- Threshold is estimated on train split only to avoid look-ahead bias.

## Factors
- 5-day cumulative return
- 20-day rolling volatility
- 5-day amount change
- high_vol state label
- Continuous factors are standardized by train-split mean/std (high_vol remains binary).

## Window
- length = 20

## Split
- Configured boundary:
  - train_end: 2022-12-31
  - valid_end: 2023-12-31
- Effective split on current common-date panel:
  - train: 2020-11-16 to 2022-12-30
  - valid: 2023-01-03 to 2023-12-29
  - test: 2024-01-02 to latest

## Model
- minimal conditional DDPM
- MLP backbone first
- tail sample weight = 3
- Implementation status: complete for C module (`model.py`, `diffusion.py`, `train.py`, `sample.py`)

## Calibration
- Calibration grid completed on `seed in {42, 52, 62}` and `tail_weight in {1, 3, 5}`
- Two runs are retained as reporting anchors:
  - `seed_42_tailw_1.0`: preferred when overall distribution distance is the main objective
  - `seed_42_tailw_3.0`: preferred when ES alignment to the real test set is the main objective
- Decision boundary for the next stage:
  - if distribution fit is the priority, continue from `tail_weight=1.0`
  - if tail-risk metric alignment is the priority, continue from `tail_weight=3.0`
  - if both are required at once, move to condition expansion or backbone upgrade rather than more seed/weight tuning

## Evaluation
- Compare real test windows vs generated samples on two horizons:
  - `last_day`
  - `cum_20d`
- Primary risk metrics: mean, std, VaR(5%), ES(5%), worst return
- Distribution diagnostics: KS statistic/p-value and histogram Spearman correlation

## Attribution
- Single-factor what-if perturbation around a base condition
- Factors: `cumret_5d`, `vol_20d`, `amount_change_5d`, `high_vol`
- Output deltas relative to baseline:
  - `delta_var_5pct`
  - `delta_es_5pct`
- Implementation status: complete for D module (`evaluate.py`, `attribution.py`)
