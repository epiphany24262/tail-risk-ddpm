from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_FILE_ASSET_MAP = {
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
    parser.add_argument("--config", default="configs/data.yaml", help="Data config path")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw csv folder")
    parser.add_argument("--out-path", default="data/processed/prices.parquet", help="Output parquet")
    parser.add_argument(
        "--qc-out-path",
        default="outputs/tables/a_data_quality_summary.csv",
        help="A module data quality summary output",
    )
    parser.add_argument(
        "--align-common-dates",
        action="store_true",
        help="Keep only common trading dates across all assets",
    )
    return parser.parse_args()


def load_file_asset_map(config_path: Path) -> dict[str, str]:
    if not config_path.exists():
        return dict(DEFAULT_FILE_ASSET_MAP)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    asset_pool = cfg.get("asset_pool")
    if isinstance(asset_pool, list) and asset_pool:
        mapped: dict[str, str] = {}
        for item in asset_pool:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            raw_file = str(item.get("raw_file", "")).strip()
            code = str(item.get("code", "")).strip()
            if not name:
                continue
            if raw_file:
                mapped[raw_file] = name
            elif code:
                mapped[f"{name}_{code}.csv"] = name
        if mapped:
            return mapped

    return dict(DEFAULT_FILE_ASSET_MAP)


def _load_one_csv(file_path: Path, asset_name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    df_raw = pd.read_csv(file_path, encoding="utf-8-sig")
    raw_rows = int(len(df_raw))
    missing_cols = [col for col in RENAME_MAP.keys() if col not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"{file_path.name} missing columns: {missing_cols}")

    key_missing_rows = int(df_raw[list(RENAME_MAP.keys())].isna().any(axis=1).sum())

    df = df_raw[list(RENAME_MAP.keys())].rename(columns=RENAME_MAP).copy()
    df["asset"] = asset_name
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_drop = len(df)
    df = df.dropna(subset=["trade_date", "open", "high", "low", "close"])
    dropped_na_rows = int(before_drop - len(df))

    before_price_filter = len(df)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    invalid_price_rows = int(before_price_filter - len(df))

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["trade_date", "asset"], keep="first")
    duplicate_rows_removed = int(before_dedup - len(df))

    df = df.sort_values("trade_date").reset_index(drop=True)
    final_rows = int(len(df))
    missing_rate = float((raw_rows - final_rows) / raw_rows) if raw_rows else 0.0

    if final_rows:
        date_start = str(df["trade_date"].min().date())
        date_end = str(df["trade_date"].max().date())
        trading_days = int(df["trade_date"].nunique())
    else:
        date_start = None
        date_end = None
        trading_days = 0

    stat = {
        "asset": asset_name,
        "source_file": file_path.name,
        "raw_rows": raw_rows,
        "final_rows": final_rows,
        "trading_days": trading_days,
        "date_start": date_start,
        "date_end": date_end,
        "key_missing_rows_raw": key_missing_rows,
        "dropped_na_rows": dropped_na_rows,
        "invalid_price_rows": invalid_price_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "missing_rate": round(missing_rate, 6),
    }
    return df, stat


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
    qc_out_path = Path(args.qc_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qc_out_path.parent.mkdir(parents=True, exist_ok=True)

    file_asset_map = load_file_asset_map(Path(args.config))

    all_dfs: list[pd.DataFrame] = []
    qc_stats: list[dict[str, object]] = []
    for file_name, asset_name in file_asset_map.items():
        file_path = raw_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df, stat = _load_one_csv(file_path, asset_name)
        print(f"[loaded] {file_name} -> rows={len(df)}")
        all_dfs.append(df)
        qc_stats.append(stat)

    prices = pd.concat(all_dfs, ignore_index=True)
    prices = prices[OUTPUT_COLUMNS].sort_values(["trade_date", "asset"]).reset_index(drop=True)
    _summarize(prices, "before alignment")

    if args.align_common_dates:
        asset_dates = {asset: set(group["trade_date"]) for asset, group in prices.groupby("asset")}
        common_dates = set.intersection(*asset_dates.values())
        prices = prices[prices["trade_date"].isin(common_dates)].copy()
        prices = prices.sort_values(["trade_date", "asset"]).reset_index(drop=True)
        _summarize(prices, "after common-date alignment")

    prices.to_parquet(out_path, index=False)
    print(f"[saved] {out_path}")
    print(f"[dtypes]\n{prices.dtypes}")

    qc_df = pd.DataFrame(qc_stats).sort_values("asset").reset_index(drop=True)
    qc_df.to_csv(qc_out_path, index=False, encoding="utf-8-sig")
    print(f"[saved] {qc_out_path}")


if __name__ == "__main__":
    main()
