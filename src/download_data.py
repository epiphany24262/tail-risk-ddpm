"""
download_data.py
----------------
Download historical OHLCV data for the configured tickers and save the
adjusted-close price series to data/raw/ as CSV files.

Usage
-----
    python src/download_data.py
    python src/download_data.py --cfg configs/data.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import yfinance as yf
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def download(cfg_path: str = "configs/data.yaml") -> None:
    cfg = OmegaConf.load(cfg_path)
    raw_dir = Path(cfg.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    tickers = list(cfg.tickers)
    log.info("Downloading %d tickers: %s", len(tickers), tickers)

    data: dict[str, pd.Series] = {}
    for ticker in tickers:
        log.info("  Fetching %s …", ticker)
        df = yf.download(
            ticker,
            start=cfg.start_date,
            end=cfg.end_date,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            log.warning("  No data returned for %s — skipping.", ticker)
            continue
        close = df["Close"].rename(ticker)
        data[ticker] = close
        out_path = raw_dir / f"{ticker}.csv"
        close.to_csv(out_path)
        log.info("  Saved %d rows → %s", len(close), out_path)

    # Also save a combined panel
    panel = pd.DataFrame(data)
    panel_path = raw_dir / "prices.csv"
    panel.to_csv(panel_path)
    log.info("Combined panel (%d rows × %d cols) → %s", *panel.shape, panel_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw price data.")
    parser.add_argument("--cfg", default="configs/data.yaml", help="Path to data config.")
    args = parser.parse_args()
    download(args.cfg)


if __name__ == "__main__":
    main()
