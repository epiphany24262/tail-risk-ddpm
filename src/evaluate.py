from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


TEST_DATASET_PATH = Path("data/processed/dataset_test.npz")
SAMPLE_DIR = Path("outputs/samples")
TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")
ALPHA = 0.05


@dataclass
class SamplePack:
    condition: str
    checkpoint: str
    path: Path
    log_returns: np.ndarray  # [B, 20, N]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate real vs generated return distributions for D module."
    )
    parser.add_argument("--test-dataset", default=str(TEST_DATASET_PATH), help="Path to dataset_test.npz")
    parser.add_argument("--sample-dir", default=str(SAMPLE_DIR), help="Directory with sample_*.npy")
    parser.add_argument("--tables-dir", default=str(TABLES_DIR), help="Output directory for CSV tables")
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR), help="Output directory for figures")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Tail probability (default=0.05)")
    return parser.parse_args()


def compute_portfolio_returns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 3:
        raise ValueError(f"Expected shape [B, T, N], got {x.shape}")
    portfolio_step = x.mean(axis=2)  # [B, T]
    last_day = portfolio_step[:, -1]
    cum_20d = portfolio_step.sum(axis=1)
    return last_day.astype(np.float64), cum_20d.astype(np.float64)


def risk_metrics(returns: np.ndarray, alpha: float) -> dict[str, float]:
    if returns.size == 0:
        raise ValueError("Empty return array.")
    var_alpha = float(np.quantile(returns, alpha))
    tail_mask = returns <= var_alpha
    es_alpha = float(returns[tail_mask].mean()) if np.any(tail_mask) else var_alpha
    return {
        "count": int(returns.size),
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0,
        "var_5pct": var_alpha,
        "es_5pct": es_alpha,
        "worst": float(np.min(returns)),
    }


def parse_sample_file(path: Path) -> tuple[str, str]:
    stem = path.stem  # sample_{condition}_{ckpt}
    if not stem.startswith("sample_"):
        raise ValueError(f"Unexpected sample file name: {path.name}")
    suffix = stem[len("sample_") :]
    if "_" not in suffix:
        raise ValueError(f"Cannot parse condition/checkpoint from: {path.name}")
    condition, checkpoint = suffix.rsplit("_", 1)
    return condition, checkpoint


def load_generated_samples(sample_dir: Path) -> list[SamplePack]:
    packs: list[SamplePack] = []
    for path in sorted(sample_dir.glob("sample_*.npy")):
        name = path.name
        if name.endswith("_price.npy") or name.endswith("_traj.npy") or name.endswith("_traj_steps.npy"):
            continue
        condition, checkpoint = parse_sample_file(path)
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError(f"Sample file must be [B, 20, N], got {path}: {arr.shape}")
        packs.append(
            SamplePack(
                condition=condition,
                checkpoint=checkpoint,
                path=path,
                log_returns=arr.astype(np.float32),
            )
        )
    return packs


def histogram_spearman(x: np.ndarray, y: np.ndarray, bins: int = 40) -> float:
    lo = float(min(np.quantile(x, 0.005), np.quantile(y, 0.005)))
    hi = float(max(np.quantile(x, 0.995), np.quantile(y, 0.995)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return float("nan")

    edges = np.linspace(lo, hi, bins + 1)
    hx, _ = np.histogram(x, bins=edges, density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    rho = stats.spearmanr(hx, hy).correlation
    return float(rho) if rho is not None else float("nan")


def distribution_compare(real: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    ks_stat, ks_pvalue = stats.ks_2samp(real, generated, alternative="two-sided", method="auto")
    return {
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "spearman_hist": histogram_spearman(real, generated),
    }


def save_tables(
    real_rows: list[dict[str, object]],
    gen_rows: list[dict[str, object]],
    cmp_rows: list[dict[str, object]],
    tables_dir: Path,
) -> tuple[Path, Path, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)

    real_df = pd.DataFrame(real_rows)
    gen_df = pd.DataFrame(gen_rows)
    cmp_df = pd.DataFrame(cmp_rows)

    real_path = tables_dir / "d_real_risk_metrics.csv"
    gen_path = tables_dir / "d_generated_risk_metrics.csv"
    cmp_path = tables_dir / "d_distribution_compare.csv"

    real_df.to_csv(real_path, index=False, encoding="utf-8-sig")
    gen_df.to_csv(gen_path, index=False, encoding="utf-8-sig")
    cmp_df.to_csv(cmp_path, index=False, encoding="utf-8-sig")
    return real_path, gen_path, cmp_path


def plot_return_distribution(
    real: np.ndarray,
    generated_last: Iterable[tuple[str, np.ndarray]],
    out_path: Path,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(real, bins=60, density=True, alpha=0.45, label="real_test_last_day")
    for label, arr in generated_last:
        plt.hist(arr, bins=60, density=True, alpha=0.35, label=f"gen_{label}")
    plt.title("Return Distribution (Last Day)")
    plt.xlabel("log return")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_left_tail(
    real: np.ndarray,
    generated_last: Iterable[tuple[str, np.ndarray]],
    out_path: Path,
) -> None:
    all_sets = [real] + [arr for _, arr in generated_last]
    left = float(min(np.quantile(a, 0.01) for a in all_sets))
    right = float(max(np.quantile(a, 0.10) for a in all_sets))

    plt.figure(figsize=(9, 5))
    plt.hist(real, bins=50, range=(left, right), density=True, alpha=0.5, label="real_test_last_day")
    for label, arr in generated_last:
        plt.hist(arr, bins=50, range=(left, right), density=True, alpha=0.35, label=f"gen_{label}")
    plt.title("Left Tail Zoom (Last Day)")
    plt.xlabel("log return")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_es_bar(
    real_es: float,
    gen_last_metrics: list[tuple[str, float]],
    out_path: Path,
) -> None:
    labels = [item[0] for item in gen_last_metrics]
    es_vals = [item[1] for item in gen_last_metrics]
    xs = np.arange(len(labels))

    plt.figure(figsize=(8, 4.5))
    plt.bar(xs, es_vals, alpha=0.8)
    plt.axhline(real_es, color="red", linestyle="--", linewidth=1.5, label=f"real ES={real_es:.5f}")
    plt.xticks(xs, labels, rotation=15)
    plt.title("ES(5%) by Condition (Last Day)")
    plt.ylabel("ES(5%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    test_dataset = Path(args.test_dataset)
    sample_dir = Path(args.sample_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    alpha = float(args.alpha)

    figures_dir.mkdir(parents=True, exist_ok=True)

    test_data = np.load(test_dataset, allow_pickle=True)
    x_test = test_data["X"].astype(np.float32)
    real_last, real_cum = compute_portfolio_returns(x_test)

    packs = load_generated_samples(sample_dir)
    if not packs:
        raise FileNotFoundError(f"No sample files found in {sample_dir} (expected sample_*.npy)")

    real_rows: list[dict[str, object]] = []
    gen_rows: list[dict[str, object]] = []
    cmp_rows: list[dict[str, object]] = []

    real_last_metrics = risk_metrics(real_last, alpha=alpha)
    real_cum_metrics = risk_metrics(real_cum, alpha=alpha)
    real_rows.append({"horizon": "last_day", **real_last_metrics})
    real_rows.append({"horizon": "cum_20d", **real_cum_metrics})

    generated_for_plot: list[tuple[str, np.ndarray]] = []
    gen_last_es: list[tuple[str, float]] = []

    for pack in packs:
        gen_last, gen_cum = compute_portfolio_returns(pack.log_returns)
        label = f"{pack.condition}_{pack.checkpoint}"

        m_last = risk_metrics(gen_last, alpha=alpha)
        m_cum = risk_metrics(gen_cum, alpha=alpha)
        gen_rows.append(
            {
                "condition": pack.condition,
                "checkpoint": pack.checkpoint,
                "horizon": "last_day",
                **m_last,
                "source_file": str(pack.path),
            }
        )
        gen_rows.append(
            {
                "condition": pack.condition,
                "checkpoint": pack.checkpoint,
                "horizon": "cum_20d",
                **m_cum,
                "source_file": str(pack.path),
            }
        )

        c_last = distribution_compare(real_last, gen_last)
        c_cum = distribution_compare(real_cum, gen_cum)
        cmp_rows.append(
            {
                "condition": pack.condition,
                "checkpoint": pack.checkpoint,
                "horizon": "last_day",
                **c_last,
            }
        )
        cmp_rows.append(
            {
                "condition": pack.condition,
                "checkpoint": pack.checkpoint,
                "horizon": "cum_20d",
                **c_cum,
            }
        )

        generated_for_plot.append((label, gen_last))
        gen_last_es.append((label, m_last["es_5pct"]))

    real_path, gen_path, cmp_path = save_tables(real_rows, gen_rows, cmp_rows, tables_dir)

    dist_fig = figures_dir / "d_return_dist_last_day.png"
    tail_fig = figures_dir / "d_left_tail_compare_last_day.png"
    es_fig = figures_dir / "d_es_by_condition_last_day.png"

    plot_return_distribution(real_last, generated_for_plot, dist_fig)
    plot_left_tail(real_last, generated_for_plot, tail_fig)
    plot_es_bar(real_last_metrics["es_5pct"], gen_last_es, es_fig)

    print(f"[saved] {real_path}")
    print(f"[saved] {gen_path}")
    print(f"[saved] {cmp_path}")
    print(f"[saved] {dist_fig}")
    print(f"[saved] {tail_fig}")
    print(f"[saved] {es_fig}")
    print("[summary]")
    print(
        f"  real_last_day: mean={real_last_metrics['mean']:.6f}, "
        f"VaR5={real_last_metrics['var_5pct']:.6f}, ES5={real_last_metrics['es_5pct']:.6f}"
    )
    print(
        f"  real_cum20d: mean={real_cum_metrics['mean']:.6f}, "
        f"VaR5={real_cum_metrics['var_5pct']:.6f}, ES5={real_cum_metrics['es_5pct']:.6f}"
    )


if __name__ == "__main__":
    main()
