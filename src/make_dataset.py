from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


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
CONTINUOUS_CONDITIONS = ["cumret_5d", "vol_20d", "amount_change_5d"]
ALL_CONDITIONS = CONTINUOUS_CONDITIONS + ["high_vol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build features.parquet and split datasets from prices.parquet."
    )
    parser.add_argument("--config", default="configs/data.yaml", help="Data config path")
    parser.add_argument(
        "--prices-path",
        default="data/processed/prices.parquet",
        help="Input parquet path from preprocess.py",
    )
    parser.add_argument(
        "--features-path",
        default="data/processed/features.parquet",
        help="Output daily features parquet",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Output directory for split datasets and stats",
    )
    parser.add_argument(
        "--strict-assets",
        action="store_true",
        help="Fail if configured assets are not fully available in prices.parquet",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def load_prices(prices_path: Path) -> pd.DataFrame:
    prices = pd.read_parquet(prices_path)
    missing_cols = REQUIRED_COLUMNS - set(prices.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    prices = prices.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], errors="coerce")
    prices = prices.dropna(subset=["trade_date", "asset", "close", "amount"])
    prices = prices.sort_values(["trade_date", "asset"]).reset_index(drop=True)
    return prices


def resolve_assets(
    prices: pd.DataFrame,
    configured_assets: list[str] | None,
    strict_assets: bool,
) -> list[str]:
    available_assets = sorted(prices["asset"].unique().tolist())
    if not configured_assets:
        return available_assets

    missing_assets = [asset for asset in configured_assets if asset not in available_assets]
    if missing_assets:
        message = (
            "Configured assets missing from prices.parquet: "
            f"{missing_assets}. Available assets: {available_assets}."
        )
        if strict_assets:
            raise ValueError(message)
        print(f"[warn] {message}")
        print("[warn] Falling back to all available assets in prices.parquet.")
        return available_assets

    return configured_assets


def build_balanced_panels(
    prices: pd.DataFrame,
    assets: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = prices[prices["asset"].isin(assets)].copy()
    close_panel = (
        subset.pivot(index="trade_date", columns="asset", values="close")
        .reindex(columns=assets)
        .sort_index()
    )
    amount_panel = (
        subset.pivot(index="trade_date", columns="asset", values="amount")
        .reindex(columns=assets)
        .sort_index()
    )

    valid_dates = close_panel.dropna().index.intersection(amount_panel.dropna().index)
    close_panel = close_panel.loc[valid_dates].copy()
    amount_panel = amount_panel.loc[valid_dates].copy()

    if close_panel.empty:
        raise ValueError("No common dates remain after aligning selected assets.")

    return close_panel, amount_panel


def assign_split(
    dates: pd.Series,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> pd.Series:
    split = np.where(
        dates <= train_end,
        "train",
        np.where(dates <= valid_end, "valid", "test"),
    )
    return pd.Series(split, index=dates.index, dtype="object")


def build_features(
    close_panel: pd.DataFrame,
    amount_panel: pd.DataFrame,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
    tail_quantile: float,
) -> tuple[pd.DataFrame, float, float]:
    asset_returns = np.log(close_panel / close_panel.shift(1))
    asset_returns.columns = [f"ret_{asset}" for asset in close_panel.columns]

    portfolio_return = asset_returns.mean(axis=1)
    portfolio_amount = amount_panel.sum(axis=1)

    features = pd.DataFrame(index=close_panel.index)
    features["portfolio_return"] = portfolio_return
    features["tail_loss"] = -portfolio_return
    features["cumret_5d"] = portfolio_return.rolling(5, min_periods=5).sum()
    features["vol_20d"] = portfolio_return.rolling(20, min_periods=20).std()
    features["amount_change_5d"] = np.log(portfolio_amount / portfolio_amount.shift(5))
    features["portfolio_amount"] = portfolio_amount
    features["asset_count"] = close_panel.shape[1]
    features["split"] = assign_split(
        pd.Series(features.index, index=features.index),
        train_end=train_end,
        valid_end=valid_end,
    )
    features = pd.concat([features, asset_returns], axis=1)

    train_mask = features["split"] == "train"
    train_losses = features.loc[train_mask, "tail_loss"].dropna()
    if train_losses.empty:
        raise ValueError("No training rows available to compute tail threshold.")
    tail_threshold = float(train_losses.quantile(tail_quantile))

    features["y_tail"] = (features["tail_loss"] >= tail_threshold).astype("int64")

    train_vol = features.loc[train_mask, "vol_20d"].dropna()
    if train_vol.empty:
        raise ValueError("No training rows available to compute high-vol threshold.")
    high_vol_threshold = float(train_vol.median())
    features["high_vol"] = (features["vol_20d"] >= high_vol_threshold).astype("int64")

    features = features.reset_index().rename(columns={"index": "trade_date"})
    return features, tail_threshold, high_vol_threshold


def build_condition_frame(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    train_mask = features["split"] == "train"
    train_stats = features.loc[train_mask, CONTINUOUS_CONDITIONS].agg(["mean", "std"])

    means = train_stats.loc["mean"].to_dict()
    stds = train_stats.loc["std"].replace(0.0, 1.0).fillna(1.0).to_dict()

    condition_frame = pd.DataFrame(index=features.index)
    for name in CONTINUOUS_CONDITIONS:
        condition_frame[name] = (features[name] - means[name]) / stds[name]
    condition_frame["high_vol"] = features["high_vol"].astype("float32")

    scale_stats: dict[str, float] = {}
    for name in CONTINUOUS_CONDITIONS:
        scale_stats[f"{name}_mean"] = float(means[name])
        scale_stats[f"{name}_std"] = float(stds[name])

    return condition_frame, scale_stats


def build_samples(
    features: pd.DataFrame,
    condition_frame: pd.DataFrame,
    window_length: int,
) -> dict[str, dict[str, np.ndarray]]:
    return_columns = [col for col in features.columns if col.startswith("ret_")]
    if not return_columns:
        raise ValueError("No return columns found in features table.")

    valid_row_mask = features[return_columns + CONTINUOUS_CONDITIONS].notna().all(axis=1)

    split_samples: dict[str, dict[str, list[Any]]] = {
        "train": {"X": [], "C": [], "y_tail": [], "state_label": [], "trade_date": []},
        "valid": {"X": [], "C": [], "y_tail": [], "state_label": [], "trade_date": []},
        "test": {"X": [], "C": [], "y_tail": [], "state_label": [], "trade_date": []},
    }

    for end_idx in range(window_length - 1, len(features)):
        start_idx = end_idx - window_length + 1
        window_returns = features.iloc[start_idx : end_idx + 1][return_columns]
        end_row = features.iloc[end_idx]

        if not valid_row_mask.iloc[end_idx]:
            continue
        if window_returns.isna().any().any():
            continue

        split_name = str(end_row["split"])
        sample_bucket = split_samples[split_name]
        sample_bucket["X"].append(window_returns.to_numpy(dtype=np.float32))
        sample_bucket["C"].append(condition_frame.iloc[end_idx][ALL_CONDITIONS].to_numpy(dtype=np.float32))
        sample_bucket["y_tail"].append(np.int64(end_row["y_tail"]))
        sample_bucket["state_label"].append(np.int64(end_row["high_vol"]))
        sample_bucket["trade_date"].append(np.datetime64(end_row["trade_date"].date()))

    finalized: dict[str, dict[str, np.ndarray]] = {}
    n_assets = len(return_columns)
    n_conditions = len(ALL_CONDITIONS)
    for split_name, values in split_samples.items():
        n_samples = len(values["X"])
        finalized[split_name] = {
            "X": (
                np.stack(values["X"]).astype(np.float32)
                if values["X"]
                else np.empty((0, window_length, n_assets), dtype=np.float32)
            ),
            "C": (
                np.stack(values["C"]).astype(np.float32)
                if values["C"]
                else np.empty((0, n_conditions), dtype=np.float32)
            ),
            "y_tail": np.asarray(values["y_tail"], dtype=np.int64),
            "state_label": np.asarray(values["state_label"], dtype=np.int64),
            "trade_date": np.asarray(values["trade_date"], dtype="datetime64[D]"),
        }

        if n_samples == 0:
            print(f"[warn] split={split_name} has zero samples after windowing.")

    return finalized


def save_split_dataset(
    path: Path,
    split_name: str,
    dataset: dict[str, np.ndarray],
    assets: list[str],
) -> None:
    np.savez_compressed(
        path,
        X=dataset["X"],
        C=dataset["C"],
        y_tail=dataset["y_tail"],
        state_label=dataset["state_label"],
        trade_date=dataset["trade_date"],
        asset_names=np.asarray(assets),
        condition_names=np.asarray(ALL_CONDITIONS),
        split=np.asarray([split_name]),
    )


def summarize_tail_stats(
    features: pd.DataFrame,
    split_datasets: dict[str, dict[str, np.ndarray]],
    assets: list[str],
    window_length: int,
    tail_quantile: float,
    tail_threshold: float,
    high_vol_threshold: float,
    scale_stats: dict[str, float],
) -> dict[str, Any]:
    split_summary: dict[str, Any] = {}
    for split_name, dataset in split_datasets.items():
        n_samples = int(dataset["X"].shape[0])
        n_tail = int(dataset["y_tail"].sum())
        split_summary[split_name] = {
            "n_samples": n_samples,
            "n_tail": n_tail,
            "tail_ratio": float(n_tail / n_samples) if n_samples else 0.0,
            "date_start": (
                str(features.loc[features["split"] == split_name, "trade_date"].min().date())
                if (features["split"] == split_name).any()
                else None
            ),
            "date_end": (
                str(features.loc[features["split"] == split_name, "trade_date"].max().date())
                if (features["split"] == split_name).any()
                else None
            ),
        }

    usable_rows = features.dropna(subset=CONTINUOUS_CONDITIONS + ["portfolio_return"])
    return {
        "assets": assets,
        "n_assets": len(assets),
        "window_length": window_length,
        "tail_quantile": tail_quantile,
        "tail_threshold": tail_threshold,
        "high_vol_threshold": high_vol_threshold,
        "n_feature_rows": int(len(usable_rows)),
        "feature_date_range": {
            "start": str(usable_rows["trade_date"].min().date()) if not usable_rows.empty else None,
            "end": str(usable_rows["trade_date"].max().date()) if not usable_rows.empty else None,
        },
        "feature_scaling": scale_stats,
        "splits": split_summary,
    }


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    prices_path = Path(args.prices_path)
    features_path = Path(args.features_path)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    prices = load_prices(prices_path)

    configured_assets = config.get("assets")
    assets = resolve_assets(prices, configured_assets, strict_assets=args.strict_assets)

    window_length = int(config.get("window_length", 20))
    tail_quantile = float(config.get("tail_quantile", 0.95))
    train_end = pd.Timestamp(config.get("train_end", "2022-12-31"))
    valid_end = pd.Timestamp(config.get("valid_end", "2023-12-31"))

    close_panel, amount_panel = build_balanced_panels(prices, assets)
    features, tail_threshold, high_vol_threshold = build_features(
        close_panel=close_panel,
        amount_panel=amount_panel,
        train_end=train_end,
        valid_end=valid_end,
        tail_quantile=tail_quantile,
    )
    condition_frame, scale_stats = build_condition_frame(features)
    split_datasets = build_samples(
        features=features,
        condition_frame=condition_frame,
        window_length=window_length,
    )

    features.to_parquet(features_path, index=False)
    print(f"[saved] {features_path}")

    for split_name, dataset in split_datasets.items():
        dataset_path = out_dir / f"dataset_{split_name}.npz"
        save_split_dataset(dataset_path, split_name, dataset, assets)
        print(
            f"[saved] {dataset_path} "
            f"X={dataset['X'].shape} C={dataset['C'].shape} tail={int(dataset['y_tail'].sum())}"
        )

    stats = summarize_tail_stats(
        features=features,
        split_datasets=split_datasets,
        assets=assets,
        window_length=window_length,
        tail_quantile=tail_quantile,
        tail_threshold=tail_threshold,
        high_vol_threshold=high_vol_threshold,
        scale_stats=scale_stats,
    )
    stats_path = out_dir / "tail_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[saved] {stats_path}")

    print("[summary]")
    print(
        f"  assets={assets} common_dates={len(close_panel)} "
        f"range={close_panel.index.min().date()}->{close_panel.index.max().date()}"
    )
    print(
        f"  thresholds tail={tail_threshold:.6f} "
        f"high_vol={high_vol_threshold:.6f}"
    )
    for split_name, split_info in stats["splits"].items():
        print(
            f"  split={split_name} samples={split_info['n_samples']} "
            f"tail={split_info['n_tail']} ratio={split_info['tail_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
