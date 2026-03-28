"""
preprocess.py
-------------
Load raw price CSVs, compute log-returns, apply rolling z-score
normalisation, create binary tail labels, and persist processed data.

Usage
-----
    python src/preprocess.py
    python src/preprocess.py --cfg configs/data.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def compute_returns(prices: pd.DataFrame, return_type: str = "log") -> pd.DataFrame:
    """Compute log or simple returns from a price DataFrame."""
    if return_type == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    return returns.dropna()


def rolling_zscore(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Apply rolling window z-score normalisation column-wise."""
    mu = returns.rolling(window, min_periods=window).mean()
    sigma = returns.rolling(window, min_periods=window).std(ddof=1)
    normalised = (returns - mu) / sigma
    return normalised.dropna()


def make_tail_label(returns: pd.DataFrame, quantile: float) -> pd.Series:
    """
    Binary tail label: 1 if *any* asset's return falls below the
    cross-sectional quantile threshold on that day, else 0.
    """
    threshold = returns.quantile(quantile)
    tail_flag = (returns <= threshold).any(axis=1).astype(int)
    return tail_flag


def preprocess(cfg_path: str = "configs/data.yaml") -> None:
    cfg = OmegaConf.load(cfg_path)
    raw_dir = Path(cfg.raw_dir)
    processed_dir = Path(cfg.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    panel_path = raw_dir / "prices.csv"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} not found. Run download_data.py first."
        )

    prices = pd.read_csv(panel_path, index_col=0, parse_dates=True)
    log.info("Loaded price panel: %s rows × %s cols", *prices.shape)

    returns = compute_returns(prices, cfg.return_type)
    log.info("Returns computed: %s rows", len(returns))

    normed = rolling_zscore(returns, cfg.window_size)
    log.info("Normalised returns: %s rows (after rolling window drop)", len(normed))

    tail_label = make_tail_label(returns.loc[normed.index], cfg.tail_quantile)

    # Save artefacts
    normed.to_csv(processed_dir / "returns_normalised.csv")
    tail_label.to_csv(processed_dir / "tail_labels.csv", header=["tail"])
    returns.loc[normed.index].to_csv(processed_dir / "returns_raw.csv")
    log.info("Saved processed data to %s", processed_dir)

    # Summary statistics
    tail_rate = tail_label.mean()
    log.info(
        "Tail-event rate (q=%.2f): %.2f%% of days",
        cfg.tail_quantile,
        tail_rate * 100,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw price data.")
    parser.add_argument("--cfg", default="configs/data.yaml", help="Path to data config.")
    args = parser.parse_args()
    preprocess(args.cfg)


if __name__ == "__main__":
    main()
