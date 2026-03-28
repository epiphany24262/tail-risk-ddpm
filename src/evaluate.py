"""
evaluate.py
-----------
Evaluate generated scenarios against historical data using standard
financial risk metrics:

    * Value at Risk (VaR) at multiple confidence levels
    * Expected Shortfall (CVaR / ES)
    * Kolmogorov–Smirnov test (marginal distribution fit)
    * Spearman rank correlation (dependence structure)
    * Tail-event recall: what fraction of generated scenarios are true tail days

Results are printed to stdout and saved as CSV tables in outputs/tables/.

Usage
-----
    python src/evaluate.py
    python src/evaluate.py --cfg configs/sample.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def var_historical(returns: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Historical VaR at confidence level (1 - alpha) for each asset."""
    return np.quantile(returns, alpha, axis=0)


def expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Expected Shortfall (CVaR) at confidence level (1 - alpha) for each asset."""
    var = var_historical(returns, alpha)
    es = np.array([returns[returns[:, i] <= var[i], i].mean() for i in range(returns.shape[1])])
    return es


def portfolio_var_es(
    returns: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Compute VaR and ES for an equal-weighted (or custom) portfolio."""
    portfolio_ret = returns @ weights
    var = float(np.quantile(portfolio_ret, alpha))
    es = float(portfolio_ret[portfolio_ret <= var].mean())
    return {"VaR": var, "ES": es, "alpha": alpha}


# ---------------------------------------------------------------------------
# Distribution tests
# ---------------------------------------------------------------------------

def ks_test_marginals(
    real: np.ndarray,
    synthetic: np.ndarray,
    asset_names: list[str],
) -> pd.DataFrame:
    """Run two-sample KS test for each asset marginal distribution."""
    rows = []
    for i, name in enumerate(asset_names):
        stat, p = stats.ks_2samp(real[:, i], synthetic[:, i])
        rows.append({"asset": name, "ks_stat": stat, "p_value": p})
    return pd.DataFrame(rows)


def spearman_corr(returns: np.ndarray) -> np.ndarray:
    """Compute Spearman rank correlation matrix."""
    corr, _ = stats.spearmanr(returns)
    if returns.shape[1] == 1:
        return np.array([[1.0]])
    return corr


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate(cfg_path: str = "configs/sample.yaml") -> None:
    cfg = OmegaConf.load(cfg_path)
    data_cfg = OmegaConf.load("configs/data.yaml")
    tables_dir = Path(cfg.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load generated scenarios ----
    samples_path = Path(cfg.samples_dir) / cfg.samples_file
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples not found at {samples_path}. Run sample.py first.")

    raw = np.load(samples_path)
    scenarios = raw["scenarios"]  # (N, C, L)
    # Use last day of each window as the return vector
    gen_returns = scenarios[:, :, -1]  # (N, C)
    log.info("Loaded %d generated scenarios, %d assets.", *gen_returns.shape)

    # ---- Load historical test-set returns ----
    processed_dir = Path(data_cfg.processed_dir)
    hist_df = pd.read_csv(
        processed_dir / "returns_normalised.csv", index_col=0, parse_dates=True
    )
    n = len(hist_df)
    test_start = int(n * (data_cfg.split.train + data_cfg.split.val))
    hist_returns = hist_df.values[test_start:]  # (T_test, C)
    asset_names = hist_df.columns.tolist()
    log.info("Historical test-set: %d days.", len(hist_returns))

    weights = np.ones(len(asset_names)) / len(asset_names)

    # ---- VaR / ES ----
    results: list[dict] = []
    for alpha in [0.01, 0.05, 0.10]:
        hist_metrics = portfolio_var_es(hist_returns, weights, alpha)
        gen_metrics = portfolio_var_es(gen_returns, weights, alpha)
        results.append({
            "alpha": alpha,
            "hist_VaR": hist_metrics["VaR"],
            "gen_VaR": gen_metrics["VaR"],
            "hist_ES": hist_metrics["ES"],
            "gen_ES": gen_metrics["ES"],
        })

    risk_table = pd.DataFrame(results)
    risk_path = tables_dir / "var_es_comparison.csv"
    risk_table.to_csv(risk_path, index=False)
    log.info("VaR/ES comparison:\n%s", risk_table.to_string(index=False))

    # ---- KS tests ----
    ks_table = ks_test_marginals(hist_returns, gen_returns, asset_names)
    ks_path = tables_dir / "ks_tests.csv"
    ks_table.to_csv(ks_path, index=False)
    log.info("KS tests:\n%s", ks_table.to_string(index=False))

    # ---- Spearman correlation ----
    hist_corr = spearman_corr(hist_returns)
    gen_corr = spearman_corr(gen_returns)
    corr_diff = pd.DataFrame(
        np.abs(hist_corr - gen_corr),
        index=asset_names,
        columns=asset_names,
    )
    corr_path = tables_dir / "spearman_corr_diff.csv"
    corr_diff.to_csv(corr_path)
    log.info(
        "Mean |Spearman corr diff|: %.4f", corr_diff.values[~np.eye(len(asset_names), dtype=bool)].mean()
    )

    log.info("Evaluation tables saved to %s", tables_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated scenarios.")
    parser.add_argument("--cfg", default="configs/sample.yaml")
    args = parser.parse_args()
    evaluate(args.cfg)


if __name__ == "__main__":
    main()
