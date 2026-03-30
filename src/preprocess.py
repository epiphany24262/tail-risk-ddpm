from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FILE_ASSET_MAP = {
    "hs300_etf_510300.csv": "hs300_etf",
    "zz500_etf_510500.csv": "zz500_etf",
    "cyb_etf_159915.csv": "cyb_etf",
    "cyb50_etf_159949.csv": "cyb50_etf",
    "kc50_etf_588000.csv": "kc50_etf",
}

RENAME_MAP = {
    "日期": "trade_date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
}

OUTPUT_COLUMNS = [
    "trade_date",
    "asset",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw csv files into prices.parquet.")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw csv folder")
    parser.add_argument("--out-path", default="data/processed/prices.parquet", help="Output parquet")
    parser.add_argument(
        "--align-common-dates",
        action="store_true",
        help="Keep only common trading dates across all assets",
    )
    return parser.parse_args()


def _load_one_csv(file_path: Path, asset_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    missing_cols = [col for col in RENAME_MAP.keys() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{file_path.name} missing columns: {missing_cols}")

    df = df[list(RENAME_MAP.keys())].rename(columns=RENAME_MAP)
    df["asset"] = asset_name
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["trade_date", "open", "high", "low", "close"])
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df.drop_duplicates(subset=["trade_date", "asset"], keep="first")
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def _summarize(df: pd.DataFrame, title: str) -> None:
    print(f"[summary] {title}")
    print(f"  shape={df.shape}")

    if df.empty or df["trade_date"].dropna().empty:
        date_range_str = "N/A -> N/A"
    else:
        min_date = df["trade_date"].min()
        max_date = df["trade_date"].max()
        if pd.isna(min_date) or pd.isna(max_date):
            date_range_str = "N/A -> N/A"
        else:
            date_range_str = f"{min_date.date()} -> {max_date.date()}"

    print(f"  date_range={date_range_str}")
    per_asset = df.groupby("asset")["trade_date"].nunique().sort_index()
    print("  per_asset_trading_days:")
    for asset, n_days in per_asset.items():
        print(f"    {asset}: {n_days}")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for file_name, asset_name in FILE_ASSET_MAP.items():
        file_path = raw_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = _load_one_csv(file_path, asset_name)
        print(f"[loaded] {file_name} -> rows={len(df)}")
        all_dfs.append(df)

    prices = pd.concat(all_dfs, ignore_index=True)
    prices = prices[OUTPUT_COLUMNS].sort_values(["trade_date", "asset"]).reset_index(drop=True)
    _summarize(prices, "before alignment")

    if args.align_common_dates:
        asset_dates = {
            asset: set(group["trade_date"]) for asset, group in prices.groupby("asset")
        }
        common_dates = set.intersection(*asset_dates.values())
        prices = prices[prices["trade_date"].isin(common_dates)].copy()
        prices = prices.sort_values(["trade_date", "asset"]).reset_index(drop=True)
        _summarize(prices, "after common-date alignment")

    prices.to_parquet(out_path, index=False)
    print(f"[saved] {out_path}")
    print(f"[dtypes]\n{prices.dtypes}")


if __name__ == "__main__":
    main()
