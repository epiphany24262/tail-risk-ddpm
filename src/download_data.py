from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import akshare as ak


SYMBOLS = {
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
    parser.add_argument("--start-date", default="20150101", help="YYYYMMDD")
    parser.add_argument(
        "--end-date",
        default=date.today().strftime("%Y%m%d"),
        help="YYYYMMDD, default=today",
    )
    parser.add_argument("--out-dir", default="data/raw", help="Raw csv output folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    for code, name in SYMBOLS.items():
        print(f"[download] {name} ({code}) {args.start_date} -> {args.end_date}")
        try:
            df = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date=args.start_date,
                end_date=args.end_date,
                adjust="",
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
        print(f"[saved] {file_path} shape={df.shape}")

    print("-" * 60)
    if failed:
        raise RuntimeError(f"Download failed for symbols: {failed}")
    print("[done] all symbols downloaded successfully.")


if __name__ == "__main__":
    main()
