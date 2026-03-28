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

## Tail Definition
- Loss = - portfolio return
- Tail flag if loss >= 95th percentile

## Factors
- 5-day cumulative return
- 20-day rolling volatility
- 5-day amount change
- high_vol state label

## Window
- length = 20

## Split
- train: 2015-2022
- valid: 2023
- test: 2024-latest

## Model
- minimal conditional DDPM
- MLP backbone first
- tail sample weight = 3
