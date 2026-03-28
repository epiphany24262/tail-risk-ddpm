from __future__ import annotations

import argparse

import pandas as pd


REQUIRED_COLUMNS = {
    "trade_date",
    "asset",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check date coverage and alignment in prices.parquet.")
    parser.add_argument(
        "--input-path",
        default="data/processed/prices.parquet",
        help="Parquet file path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = pd.read_parquet(args.input_path)

    missing_cols = REQUIRED_COLUMNS - set(prices.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    prices["trade_date"] = pd.to_datetime(prices["trade_date"], errors="coerce")
    prices = prices.dropna(subset=["trade_date", "asset"]).copy()

    print(f"[shape] {prices.shape}")
    print(f"[assets] {prices['asset'].nunique()} -> {sorted(prices['asset'].unique().tolist())}")
    print(f"[range] {prices['trade_date'].min().date()} -> {prices['trade_date'].max().date()}")

    print("[per_asset]")
    grouped = prices.groupby("asset")
    for asset, group in grouped:
        print(
            f"  {asset}: rows={len(group)}, days={group['trade_date'].nunique()}, "
            f"range={group['trade_date'].min().date()}->{group['trade_date'].max().date()}"
        )

    asset_dates = {asset: set(group["trade_date"]) for asset, group in grouped}
    common_dates = set.intersection(*asset_dates.values())
    if common_dates:
        print(
            f"[common_dates] {len(common_dates)} "
            f"range={min(common_dates).date()}->{max(common_dates).date()}"
        )
    else:
        print("[common_dates] 0")

    daily_asset_count = prices.groupby("trade_date")["asset"].nunique()
    print("[daily_asset_count_distribution]")
    print(daily_asset_count.value_counts().sort_index())


if __name__ == "__main__":
    main()
