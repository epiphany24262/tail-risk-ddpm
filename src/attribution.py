"""
attribution.py
--------------
What-if attribution analysis: quantify how much each asset (or feature)
contributes to extreme portfolio losses under the generated tail scenarios.

Two attribution methods are implemented:

1. **Marginal contribution** – shift each asset's return by ε and measure
   the change in portfolio VaR / ES.

2. **Shapley-inspired sampling** – randomly ablate subsets of assets (replace
   with their unconditional mean) and compute the marginal value of adding
   each asset to an empty coalition.

Usage
-----
    python src/attribution.py
    python src/attribution.py --cfg configs/sample.yaml
"""
from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from evaluate import expected_shortfall, portfolio_var_es

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Marginal attribution
# ---------------------------------------------------------------------------

def marginal_var_attribution(
    returns: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
    epsilon: float = 0.01,
) -> pd.Series:
    """
    Estimate each asset's marginal contribution to portfolio VaR via
    finite-difference perturbation.

    Returns
    -------
    pd.Series indexed by asset index with attribution values.
    """
    n_assets = returns.shape[1]
    base = portfolio_var_es(returns, weights, alpha)["VaR"]
    attributions = np.zeros(n_assets)
    for i in range(n_assets):
        perturbed = returns.copy()
        perturbed[:, i] += epsilon
        w_copy = weights.copy()
        delta = portfolio_var_es(perturbed, w_copy, alpha)["VaR"] - base
        attributions[i] = delta / epsilon  # approximate derivative
    return pd.Series(attributions)


def marginal_es_attribution(
    returns: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
    epsilon: float = 0.01,
) -> pd.Series:
    """Marginal contribution to ES via finite-difference."""
    n_assets = returns.shape[1]
    base = portfolio_var_es(returns, weights, alpha)["ES"]
    attributions = np.zeros(n_assets)
    for i in range(n_assets):
        perturbed = returns.copy()
        perturbed[:, i] += epsilon
        delta = portfolio_var_es(perturbed, weights, alpha)["ES"] - base
        attributions[i] = delta / epsilon
    return pd.Series(attributions)


# ---------------------------------------------------------------------------
# Approximated Shapley attribution
# ---------------------------------------------------------------------------

def approx_shapley_es(
    returns: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
    n_permutations: int = 200,
    seed: int = 0,
) -> pd.Series:
    """
    Estimate Shapley values for ES attribution by Monte Carlo permutation
    sampling over asset coalitions.

    Parameters
    ----------
    n_permutations : number of random permutations to average over
    """
    rng = np.random.default_rng(seed)
    n_assets = returns.shape[1]
    asset_mean = returns.mean(axis=0)
    shapley = np.zeros(n_assets)

    for _ in range(n_permutations):
        perm = rng.permutation(n_assets)
        coalition = np.zeros(n_assets, dtype=bool)
        prev_val = 0.0
        for asset in perm:
            coalition[asset] = True
            r_sub = returns.copy()
            r_sub[:, ~coalition] = asset_mean[~coalition]
            cur_val = portfolio_var_es(r_sub, weights, alpha)["ES"]
            shapley[asset] += cur_val - prev_val
            prev_val = cur_val

    shapley /= n_permutations
    return pd.Series(shapley)


# ---------------------------------------------------------------------------
# Main attribution routine
# ---------------------------------------------------------------------------

def run_attribution(cfg_path: str = "configs/sample.yaml") -> None:
    cfg = OmegaConf.load(cfg_path)
    data_cfg = OmegaConf.load("configs/data.yaml")
    tables_dir = Path(cfg.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load generated tail scenarios ----
    samples_path = Path(cfg.samples_dir) / cfg.samples_file
    if not samples_path.exists():
        raise FileNotFoundError(f"{samples_path} not found. Run sample.py first.")

    raw = np.load(samples_path)
    scenarios = raw["scenarios"]  # (N, C, L)
    gen_returns = scenarios[:, :, -1]  # last-day return for each scenario
    log.info("Loaded %d scenarios for attribution.", len(gen_returns))

    asset_names: List[str] = list(data_cfg.tickers)
    n_assets = gen_returns.shape[1]
    weights = np.ones(n_assets) / n_assets

    results: dict[str, pd.Series] = {}

    # ---- Marginal VaR attribution ----
    log.info("Computing marginal VaR attribution …")
    var_attr = marginal_var_attribution(gen_returns, weights, alpha=0.05)
    var_attr.index = asset_names
    results["marginal_VaR_attribution"] = var_attr

    # ---- Marginal ES attribution ----
    log.info("Computing marginal ES attribution …")
    es_attr = marginal_es_attribution(gen_returns, weights, alpha=0.05)
    es_attr.index = asset_names
    results["marginal_ES_attribution"] = es_attr

    # ---- Shapley ES attribution ----
    log.info("Computing Shapley ES attribution (may take a moment) …")
    shapley_attr = approx_shapley_es(gen_returns, weights, alpha=0.05, n_permutations=200)
    shapley_attr.index = asset_names
    results["shapley_ES_attribution"] = shapley_attr

    # ---- Save ----
    attr_table = pd.DataFrame(results)
    attr_path = tables_dir / "attribution.csv"
    attr_table.to_csv(attr_path)
    log.info("Attribution table:\n%s", attr_table.to_string())
    log.info("Saved → %s", attr_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run what-if attribution analysis.")
    parser.add_argument("--cfg", default="configs/sample.yaml")
    args = parser.parse_args()
    run_attribution(args.cfg)


if __name__ == "__main__":
    main()
