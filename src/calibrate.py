from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG_PATH = ROOT / "configs/train.yaml"
SAMPLE_CONFIG_PATH = ROOT / "configs/sample.yaml"
OUTPUTS_DIR = ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
SAMPLES_DIR = OUTPUTS_DIR / "samples"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CALIBRATION_DIR = OUTPUTS_DIR / "calibration"
SUMMARY_PATH = CALIBRATION_DIR / "calibration_summary.csv"
DEFAULT_SEEDS = [42, 52, 62]
DEFAULT_TAIL_WEIGHTS = [1.0, 3.0, 5.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run calibration grid over seeds and tail weights.")
    parser.add_argument("--train-config", default=str(TRAIN_CONFIG_PATH), help="Path to train yaml config")
    parser.add_argument("--sample-config", default=str(SAMPLE_CONFIG_PATH), help="Path to sample yaml config")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_command(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def reset_runtime_dirs() -> None:
    for path in [CHECKPOINTS_DIR, LOGS_DIR, SAMPLES_DIR, TABLES_DIR, FIGURES_DIR]:
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)


def copy_dir_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def snapshot_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    copy_dir_if_exists(CHECKPOINTS_DIR, run_dir / "checkpoints")
    copy_dir_if_exists(LOGS_DIR, run_dir / "logs")
    copy_dir_if_exists(SAMPLES_DIR, run_dir / "samples")
    copy_dir_if_exists(TABLES_DIR, run_dir / "tables")
    copy_dir_if_exists(FIGURES_DIR, run_dir / "figures")


def load_run_metrics(run_dir: Path) -> dict[str, float]:
    tables_dir = run_dir / "tables"
    cmp_df = pd.read_csv(tables_dir / "d_distribution_compare.csv")
    gen_df = pd.read_csv(tables_dir / "d_generated_risk_metrics.csv")
    real_df = pd.read_csv(tables_dir / "d_real_risk_metrics.csv")

    real_ref = real_df[["horizon", "es_alpha", "var_alpha"]].rename(
        columns={"es_alpha": "real_es_alpha", "var_alpha": "real_var_alpha"}
    )
    merged = gen_df.merge(real_ref, on="horizon", how="left", validate="many_to_one")
    merged["abs_es_gap"] = (merged["es_alpha"] - merged["real_es_alpha"]).abs()
    merged["abs_var_gap"] = (merged["var_alpha"] - merged["real_var_alpha"]).abs()

    return {
        "mean_wasserstein": float(cmp_df["wasserstein_distance"].mean()),
        "mean_ks_stat": float(cmp_df["ks_stat"].mean()),
        "mean_abs_es_gap": float(merged["abs_es_gap"].mean()),
        "mean_abs_var_gap": float(merged["abs_var_gap"].mean()),
    }


def build_run_dir(seed: int, tail_weight: float) -> Path:
    return CALIBRATION_DIR / f"seed_{seed}_tailw_{tail_weight}"


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml(Path(args.train_config))
    sample_cfg = load_yaml(Path(args.sample_config))
    sample_n = int(sample_cfg.get("n_samples", 1000))

    python = sys.executable
    script_train = str(ROOT / "src/train.py")
    script_sample = str(ROOT / "src/sample.py")
    script_evaluate = str(ROOT / "src/evaluate.py")

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for seed in DEFAULT_SEEDS:
        for tail_weight in DEFAULT_TAIL_WEIGHTS:
            run_dir = build_run_dir(seed, tail_weight)
            print("=" * 72)
            print(f"[calibration] seed={seed} tail_weight={tail_weight:.1f} run_dir={run_dir}")

            reset_runtime_dirs()

            run_command(
                [
                    python,
                    script_train,
                    "--config",
                    str(Path(args.train_config)),
                    "--seed",
                    str(seed),
                    "--tail-weight",
                    str(tail_weight),
                ]
            )
            run_command(
                [
                    python,
                    script_sample,
                    "--config",
                    str(Path(args.sample_config)),
                    "--checkpoint",
                    "best",
                    "--n-samples",
                    str(sample_n),
                ]
            )
            run_command(
                [
                    python,
                    script_evaluate,
                    "--checkpoint-filter",
                    "best",
                    "--seed",
                    str(seed),
                ]
            )

            snapshot_run(run_dir)
            metrics = load_run_metrics(run_dir)
            rows.append(
                {
                    "seed": int(seed),
                    "tail_weight": float(tail_weight),
                    "checkpoint_used": "best",
                    **metrics,
                    "run_dir": str(run_dir.relative_to(ROOT)),
                }
            )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        by=["mean_wasserstein", "mean_abs_es_gap", "mean_ks_stat"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary["rank"] = summary.index + 1
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    best_row = summary.iloc[0]
    baseline_mask = (summary["seed"] == 42) & (summary["tail_weight"] == 3.0)
    if baseline_mask.any():
        baseline = summary.loc[baseline_mask].iloc[0]
        print("[baseline]")
        print(
            f"  seed=42 tail_weight=3.0 mean_wasserstein={baseline['mean_wasserstein']:.6f} "
            f"mean_abs_es_gap={baseline['mean_abs_es_gap']:.6f}"
        )

    print(f"[saved] {SUMMARY_PATH}")
    print("[best_run]")
    print(
        f"  rank={int(best_row['rank'])} seed={int(best_row['seed'])} "
        f"tail_weight={float(best_row['tail_weight']):.1f} "
        f"mean_wasserstein={float(best_row['mean_wasserstein']):.6f} "
        f"mean_abs_es_gap={float(best_row['mean_abs_es_gap']):.6f}"
    )


if __name__ == "__main__":
    main()
