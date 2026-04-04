from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from diffusion import DDPM
from model import ConditionalMLPDenoiser


SAMPLE_CONFIG_PATH = Path("configs/sample.yaml")
TRAIN_CONFIG_PATH = Path("configs/train.yaml")
DATASET_REF_PATH = Path("data/processed/dataset_test.npz")
CHECKPOINT_DIR = Path("outputs/checkpoints")
TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")
WINDOW_LENGTH = 20
COND_DIM = 4
CONTINUOUS_FACTORS = ["cumret_5d", "vol_20d", "amount_change_5d"]
ALL_FACTORS = CONTINUOUS_FACTORS + ["high_vol"]


@dataclass
class AttributionRunConfig:
    checkpoint_label: str
    checkpoint_path: Path
    n_samples: int
    base_condition_name: str
    base_condition: np.ndarray
    values_cont: list[float]
    values_high_vol: list[float]
    device: torch.device
    timesteps_override: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-factor what-if attribution for conditional DDPM samples."
    )
    parser.add_argument("--sample-config", default=str(SAMPLE_CONFIG_PATH), help="Path to configs/sample.yaml")
    parser.add_argument("--train-config", default=str(TRAIN_CONFIG_PATH), help="Path to configs/train.yaml")
    parser.add_argument("--dataset-ref", default=str(DATASET_REF_PATH), help="Path to dataset npz for dimensions")
    parser.add_argument("--checkpoint", default="latest", help="latest|best|path/to/checkpoint.pt")
    parser.add_argument("--n-samples", type=int, default=256, help="Generated samples per factor value")
    parser.add_argument("--base-condition", default="normal_market", help="Base condition key from sample.yaml")
    parser.add_argument(
        "--factor-values",
        default="-1.0,-0.5,0.0,0.5,1.0",
        help="Comma-separated values for continuous factors",
    )
    parser.add_argument(
        "--high-vol-values",
        default="0,1",
        help="Comma-separated values for high_vol factor",
    )
    parser.add_argument("--tables-dir", default=str(TABLES_DIR), help="Output directory for tables")
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR), help="Output directory for figures")
    parser.add_argument(
        "--timesteps-override",
        type=int,
        default=None,
        help="Optional override DDPM timesteps for quick debugging",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_float_values(raw: str) -> list[float]:
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise ValueError("No numeric values provided.")
    return vals


def resolve_checkpoint(checkpoint_name: str) -> tuple[str, Path]:
    ckpt = str(checkpoint_name)
    if ckpt == "latest":
        path = CHECKPOINT_DIR / "latest.pt"
    elif ckpt == "best":
        path = CHECKPOINT_DIR / "best.pt"
    else:
        candidate = Path(ckpt)
        if candidate.exists():
            path = candidate
        else:
            if candidate.suffix != ".pt":
                candidate = Path(f"{ckpt}.pt")
            path = CHECKPOINT_DIR / candidate.name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return ckpt, path


def load_reference_dims(path: Path) -> int:
    data = np.load(path, allow_pickle=True)
    x = data["X"]
    if x.ndim != 3 or x.shape[1] != WINDOW_LENGTH:
        raise ValueError(f"Reference X must be [B,{WINDOW_LENGTH},N], got {x.shape}")
    return int(x.shape[-1])


def condition_vector_from_mapping(mapping: dict) -> np.ndarray:
    return np.asarray(
        [
            float(mapping.get("cumret_5d", 0.0)),
            float(mapping.get("vol_20d", 0.0)),
            float(mapping.get("amount_change_5d", 0.0)),
            float(mapping.get("high_vol", 0.0)),
        ],
        dtype=np.float32,
    )


def risk_metrics(returns: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    q = float(np.quantile(returns, alpha))
    tail = returns[returns <= q]
    es = float(tail.mean()) if tail.size else q
    return {
        "count": int(returns.size),
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0,
        "var_5pct": q,
        "es_5pct": es,
        "worst": float(np.min(returns)),
    }


def portfolio_returns(log_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = log_returns.mean(axis=2)  # [B,T]
    return p[:, -1].astype(np.float64), p.sum(axis=1).astype(np.float64)


def run_generation(
    model: ConditionalMLPDenoiser,
    ddpm: DDPM,
    cond_vec: np.ndarray,
    n_samples: int,
    n_assets: int,
    device: torch.device,
) -> np.ndarray:
    cond = np.repeat(cond_vec.reshape(1, COND_DIM), repeats=n_samples, axis=0)
    cond_t = torch.from_numpy(cond).to(device=device, dtype=torch.float32)
    result = ddpm.sample(
        model=model,
        condition=cond_t,
        window_length=WINDOW_LENGTH,
        n_assets=n_assets,
        n_samples=n_samples,
        return_trajectory=False,
    )
    return result.samples.detach().cpu().numpy().astype(np.float32)


def build_run_config(args: argparse.Namespace) -> AttributionRunConfig:
    sample_cfg = load_yaml(Path(args.sample_config))
    train_cfg = load_yaml(Path(args.train_config))

    ckpt_label, ckpt_path = resolve_checkpoint(args.checkpoint)
    cond_cfg = sample_cfg.get("conditions", {})
    if not isinstance(cond_cfg, dict) or not cond_cfg:
        raise ValueError("configs/sample.yaml must contain non-empty mapping under `conditions`.")
    if args.base_condition not in cond_cfg:
        available = sorted(cond_cfg.keys())
        raise ValueError(f"base-condition '{args.base_condition}' not found. Available: {available}")

    base_vec = condition_vector_from_mapping(cond_cfg[args.base_condition])
    values_cont = parse_float_values(args.factor_values)
    values_high_vol = parse_float_values(args.high_vol_values)

    cfg_device = str(train_cfg.get("device", "cpu")).lower()
    if cfg_device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable. Falling back to cpu.")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg_device)

    return AttributionRunConfig(
        checkpoint_label=ckpt_label,
        checkpoint_path=ckpt_path,
        n_samples=int(args.n_samples),
        base_condition_name=str(args.base_condition),
        base_condition=base_vec,
        values_cont=values_cont,
        values_high_vol=values_high_vol,
        device=device,
        timesteps_override=args.timesteps_override,
    )


def plot_factor_sensitivity(df: pd.DataFrame, out_path: Path, horizon: str) -> None:
    plot_df = df[df["horizon"] == horizon].copy()
    if plot_df.empty:
        return
    plt.figure(figsize=(9, 5))
    for factor in ALL_FACTORS:
        sub = plot_df[plot_df["factor"] == factor].sort_values("factor_value")
        plt.plot(sub["factor_value"].values, sub["es_5pct"].values, marker="o", label=factor)
    plt.title(f"Factor Sensitivity on ES(5%) [{horizon}]")
    plt.xlabel("factor value")
    plt.ylabel("ES(5%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    run_cfg = build_run_config(args)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_assets = load_reference_dims(Path(args.dataset_ref))
    checkpoint = torch.load(run_cfg.checkpoint_path, map_location=run_cfg.device)
    n_assets_ckpt = int(checkpoint.get("n_assets", n_assets))
    if n_assets_ckpt != n_assets:
        raise ValueError(f"Checkpoint n_assets={n_assets_ckpt} mismatches dataset n_assets={n_assets}")

    model = ConditionalMLPDenoiser(
        window_length=WINDOW_LENGTH,
        n_assets=n_assets,
        cond_dim=COND_DIM,
        hidden_dim=128,
    ).to(run_cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    timesteps = int(checkpoint.get("timesteps", 1000))
    if run_cfg.timesteps_override is not None:
        timesteps = int(run_cfg.timesteps_override)
    ddpm = DDPM(timesteps=timesteps, device=str(run_cfg.device))

    baseline_samples = run_generation(
        model=model,
        ddpm=ddpm,
        cond_vec=run_cfg.base_condition,
        n_samples=run_cfg.n_samples,
        n_assets=n_assets,
        device=run_cfg.device,
    )
    base_last, base_cum = portfolio_returns(baseline_samples)
    base_last_m = risk_metrics(base_last, alpha=0.05)
    base_cum_m = risk_metrics(base_cum, alpha=0.05)

    rows: list[dict[str, object]] = []
    baseline_row = {
        "checkpoint": run_cfg.checkpoint_label,
        "base_condition": run_cfg.base_condition_name,
        "factor": "baseline",
        "factor_value": 0.0,
        "horizon": "last_day",
        **base_last_m,
        "delta_var_5pct": 0.0,
        "delta_es_5pct": 0.0,
    }
    rows.append(baseline_row)
    rows.append(
        {
            **baseline_row,
            "horizon": "cum_20d",
            **base_cum_m,
        }
    )

    factor_to_index = {name: i for i, name in enumerate(ALL_FACTORS)}

    for factor in ALL_FACTORS:
        idx = factor_to_index[factor]
        values = run_cfg.values_cont if factor != "high_vol" else run_cfg.values_high_vol
        for value in values:
            cond = run_cfg.base_condition.copy()
            cond[idx] = float(value)
            samples = run_generation(
                model=model,
                ddpm=ddpm,
                cond_vec=cond,
                n_samples=run_cfg.n_samples,
                n_assets=n_assets,
                device=run_cfg.device,
            )
            last_day, cum20 = portfolio_returns(samples)
            m_last = risk_metrics(last_day, alpha=0.05)
            m_cum = risk_metrics(cum20, alpha=0.05)

            rows.append(
                {
                    "checkpoint": run_cfg.checkpoint_label,
                    "base_condition": run_cfg.base_condition_name,
                    "factor": factor,
                    "factor_value": float(value),
                    "horizon": "last_day",
                    **m_last,
                    "delta_var_5pct": m_last["var_5pct"] - base_last_m["var_5pct"],
                    "delta_es_5pct": m_last["es_5pct"] - base_last_m["es_5pct"],
                }
            )
            rows.append(
                {
                    "checkpoint": run_cfg.checkpoint_label,
                    "base_condition": run_cfg.base_condition_name,
                    "factor": factor,
                    "factor_value": float(value),
                    "horizon": "cum_20d",
                    **m_cum,
                    "delta_var_5pct": m_cum["var_5pct"] - base_cum_m["var_5pct"],
                    "delta_es_5pct": m_cum["es_5pct"] - base_cum_m["es_5pct"],
                }
            )
            print(
                f"[done] factor={factor} value={value:.4f} "
                f"last_es={m_last['es_5pct']:.6f} cum_es={m_cum['es_5pct']:.6f}"
            )

    out_csv = tables_dir / "d_factor_sensitivity.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    fig_last = figures_dir / "d_factor_sensitivity_es_last_day.png"
    fig_cum = figures_dir / "d_factor_sensitivity_es_cum_20d.png"
    plot_factor_sensitivity(df[df["factor"] != "baseline"], fig_last, horizon="last_day")
    plot_factor_sensitivity(df[df["factor"] != "baseline"], fig_cum, horizon="cum_20d")

    print(f"[saved] {out_csv}")
    print(f"[saved] {fig_last}")
    print(f"[saved] {fig_cum}")
    print("[summary]")
    print(
        f"  base_condition={run_cfg.base_condition_name} checkpoint={run_cfg.checkpoint_label} "
        f"n_samples={run_cfg.n_samples} timesteps={timesteps}"
    )
    print(
        f"  baseline last_day: VaR5={base_last_m['var_5pct']:.6f}, ES5={base_last_m['es_5pct']:.6f}"
    )
    print(
        f"  baseline cum_20d: VaR5={base_cum_m['var_5pct']:.6f}, ES5={base_cum_m['es_5pct']:.6f}"
    )


if __name__ == "__main__":
    main()
