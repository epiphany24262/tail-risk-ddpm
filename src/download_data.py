from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import akshare as ak
import pandas as pd
import yaml


DEFAULT_SYMBOLS = {
    "510300": "hs300_etf",
    "510500": "zz500_etf",
    "159915": "cyb_etf",
    "159949": "cyb50_etf",
    "588000": "kc50_etf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download daily ETF data for Scheme A assets."
    )
    parser.add_argument("--config", default="configs/data.yaml", help="Data config path")
    parser.add_argument("--start-date", default="20150101", help="YYYYMMDD")
    parser.add_argument(
        "--end-date",
        default=date.today().strftime("%Y%m%d"),
        help="YYYYMMDD, default=today",
    )
    parser.add_argument("--out-dir", default="data/raw", help="Raw csv output folder")
    return parser.parse_args()


def load_asset_pool(config_path: Path) -> list[tuple[str, str]]:
    if not config_path.exists():
        return list(DEFAULT_SYMBOLS.items())

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    asset_pool = cfg.get("asset_pool")
    if isinstance(asset_pool, list) and asset_pool:
        resolved: list[tuple[str, str]] = []
        for item in asset_pool:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code", "")).strip()
            name = str(item.get("name", "")).strip()
            if code and name:
                resolved.append((code, name))
        if resolved:
            return resolved

    return list(DEFAULT_SYMBOLS.items())


def _sina_symbol(code: str) -> str:
    # ETFs on SH usually start with 5/6/9; SZ usually starts with 0/1/2/3.
    return f"sh{code}" if code.startswith(("5", "6", "9")) else f"sz{code}"


def _normalize_sina_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "date": "日期",
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "volume": "成交量",
        "amount": "成交额",
    }
    missing = [c for c in rename.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"sina response missing columns: {missing}")
    return df.rename(columns=rename)[list(rename.values())].copy()


def _filter_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    out = df.copy()
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    start_ts = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
    end_ts = pd.to_datetime(end_date, format="%Y%m%d", errors="coerce")
    if pd.notna(start_ts):
        out = out[out["日期"] >= start_ts]
    if pd.notna(end_ts):
        out = out[out["日期"] <= end_ts]
    out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def fetch_with_fallback(code: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, str]:
    try:
        df_em = ak.fund_etf_hist_em(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="",
        )
        if df_em is not None and not df_em.empty:
            return df_em, "eastmoney"
    except Exception as exc:
        print(f"[warn] eastmoney failed for {code}: {exc}")

    # fallback: sina
    sina_symbol = _sina_symbol(code)
    df_sina = ak.fund_etf_hist_sina(symbol=sina_symbol)
    if df_sina is None or df_sina.empty:
        raise RuntimeError(f"sina fallback returned empty data for {code} ({sina_symbol})")
    df_sina = _normalize_sina_columns(df_sina)
    df_sina = _filter_by_date(df_sina, start_date=start_date, end_date=end_date)
    if df_sina.empty:
        raise RuntimeError(f"sina fallback has no rows in date range for {code} ({sina_symbol})")
    return df_sina, "sina"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = load_asset_pool(Path(args.config))

    failed: list[str] = []
    for code, name in assets:
        print(f"[download] {name} ({code}) {args.start_date} -> {args.end_date}")
        try:
            df, source = fetch_with_fallback(
                code=code,
                start_date=args.start_date,
                end_date=args.end_date,
            )
        except Exception as exc:  # pragma: no cover
            print(f"[error] failed to fetch {name} ({code}): {exc}")
            failed.append(code)
            continue

        if df is None or df.empty:
            print(f"[error] empty data for {name} ({code})")
            failed.append(code)
            continue

        file_path = out_dir / f"{name}_{code}.csv"
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"[saved] {file_path} shape={df.shape} source={source}")

    print("-" * 60)
    if failed:
        raise RuntimeError(f"Download failed for symbols: {failed}")
    print("[done] all symbols downloaded successfully.")


if __name__ == "__main__":
    main()
