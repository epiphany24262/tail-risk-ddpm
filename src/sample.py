from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import yaml

from diffusion import DDPM
from model import ConditionalMLPDenoiser


SAMPLE_CONFIG_PATH = Path("configs/sample.yaml")
TRAIN_CONFIG_PATH = Path("configs/train.yaml")
DATASET_REF_PATH = Path("data/processed/dataset_test.npz")
CHECKPOINT_DIR = Path("outputs/checkpoints")
SAMPLE_OUT_DIR = Path("outputs/samples")
WINDOW_LENGTH = 20
COND_DIM = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate conditional DDPM samples.")
    parser.add_argument("--config", default=str(SAMPLE_CONFIG_PATH), help="Path to sample yaml config")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint override: latest, best, or explicit .pt path",
    )
    parser.add_argument("--n-samples", type=int, default=None, help="Optional override for config n_samples")
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "cond"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def load_reference_dims() -> tuple[int, np.ndarray]:
    data = np.load(DATASET_REF_PATH, allow_pickle=True)
    x = data["X"]
    if x.ndim != 3 or x.shape[1] != WINDOW_LENGTH:
        raise ValueError(f"Reference dataset X must be [B, {WINDOW_LENGTH}, N], got {x.shape}")
    n_assets = int(x.shape[-1])
    cond_names = data["condition_names"] if "condition_names" in data.files else np.asarray([], dtype=object)
    return n_assets, cond_names


def condition_items_from_config(sample_cfg: dict, cond_names: np.ndarray) -> list[tuple[str, np.ndarray]]:
    cond_cfg = sample_cfg.get("conditions", {})
    items: list[tuple[str, np.ndarray]] = []

    if isinstance(cond_cfg, dict):
        for i, (key, value) in enumerate(cond_cfg.items()):
            if not isinstance(value, dict):
                raise ValueError(f"Condition '{key}' must be a mapping of 4 factors.")
            vector = np.asarray(
                [
                    float(value.get("cumret_5d", 0.0)),
                    float(value.get("vol_20d", 0.0)),
                    float(value.get("amount_change_5d", 0.0)),
                    float(value.get("high_vol", 0.0)),
                ],
                dtype=np.float32,
            )
            if vector.shape[0] != COND_DIM:
                raise ValueError(f"Condition '{key}' vector length must be {COND_DIM}.")
            label = sanitize_label(str(key))
            items.append((label, vector))
    elif isinstance(cond_cfg, list):
        for i, raw_vec in enumerate(cond_cfg):
            vector = np.asarray(raw_vec, dtype=np.float32).reshape(-1)
            if vector.shape[0] != COND_DIM:
                raise ValueError(f"Condition index {i} vector length must be {COND_DIM}.")
            label = sanitize_label(str(cond_names[i])) if i < len(cond_names) else f"cond{i}"
            items.append((label, vector))
    else:
        raise ValueError("conditions must be a mapping or a list.")

    if not items:
        raise ValueError("No conditions configured in configs/sample.yaml.")
    return items


def to_price_path_from_log_returns(log_returns: np.ndarray) -> np.ndarray:
    # log_returns: [B, 20, N] -> price: [B, 21, N], with t0=1
    price_tail = np.exp(np.cumsum(log_returns, axis=1, dtype=np.float64)).astype(np.float32)
    init = np.ones((log_returns.shape[0], 1, log_returns.shape[2]), dtype=np.float32)
    return np.concatenate([init, price_tail], axis=1)


def main() -> None:
    args = parse_args()
    sample_cfg = load_yaml(Path(args.config))
    train_cfg = load_yaml(TRAIN_CONFIG_PATH)
    n_samples = int(args.n_samples) if args.n_samples is not None else int(sample_cfg.get("n_samples", 1000))
    save_trajectory = bool(sample_cfg.get("save_trajectory", True))

    cfg_device = str(train_cfg.get("device", "cpu")).lower()
    if cfg_device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable. Falling back to cpu.")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg_device)

    checkpoint_name = str(args.checkpoint) if args.checkpoint is not None else str(sample_cfg.get("checkpoint", "latest"))
    ckpt_label, ckpt_path = resolve_checkpoint(checkpoint_name)
    checkpoint = torch.load(ckpt_path, map_location=device)

    n_assets_runtime, cond_names = load_reference_dims()
    n_assets = int(checkpoint.get("n_assets", n_assets_runtime))
    if n_assets != n_assets_runtime:
        raise ValueError(
            f"Checkpoint n_assets={n_assets} mismatches reference dataset n_assets={n_assets_runtime}."
        )

    model = ConditionalMLPDenoiser(
        window_length=WINDOW_LENGTH,
        n_assets=n_assets,
        cond_dim=COND_DIM,
        hidden_dim=128,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ddpm = DDPM(timesteps=int(checkpoint.get("timesteps", 1000)), device=str(device))
    cond_items = condition_items_from_config(sample_cfg, cond_names)

    print(
        f"[config] config={Path(args.config)} checkpoint={ckpt_label} "
        f"n_samples={n_samples} save_trajectory={save_trajectory}"
    )

    SAMPLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, (label, cond_vec) in enumerate(cond_items):
        cond = np.repeat(cond_vec.reshape(1, COND_DIM), repeats=n_samples, axis=0)
        cond_t = torch.from_numpy(cond).to(device=device, dtype=torch.float32)

        result = ddpm.sample(
            model=model,
            condition=cond_t,
            window_length=WINDOW_LENGTH,
            n_assets=n_assets,
            n_samples=n_samples,
            return_trajectory=save_trajectory,
        )
        log_returns = result.samples.detach().cpu().numpy().astype(np.float32)

        sample_path = SAMPLE_OUT_DIR / f"sample_{label}_{ckpt_label}.npy"
        np.save(sample_path, log_returns.astype(np.float32))

        price_path = to_price_path_from_log_returns(log_returns)
        price_out = SAMPLE_OUT_DIR / f"sample_{label}_{ckpt_label}_price.npy"
        np.save(price_out, price_path.astype(np.float32))

        print(f"[saved] {sample_path} shape={log_returns.shape} dtype={log_returns.dtype}")
        print(f"[saved] {price_out} shape={price_path.shape} dtype={price_path.dtype}")

        if save_trajectory:
            if result.trajectory is None or result.trajectory_steps is None:
                raise RuntimeError("save_trajectory=true but trajectory not returned.")
            full_traj = result.trajectory.detach().cpu().numpy().astype(np.float32)
            full_steps = result.trajectory_steps.astype(np.int32)
            if full_traj.shape[0] != full_steps.shape[0]:
                raise ValueError("trajectory and trajectory_steps are not aligned.")

            traj_out = SAMPLE_OUT_DIR / f"sample_{label}_{ckpt_label}_traj.npy"
            step_out = SAMPLE_OUT_DIR / f"sample_{label}_{ckpt_label}_traj_steps.npy"
            np.save(traj_out, full_traj)
            np.save(step_out, full_steps)
            print(f"[saved] {traj_out} shape={full_traj.shape} dtype={full_traj.dtype}")
            print(f"[saved] {step_out} shape={full_steps.shape} dtype={full_steps.dtype}")


if __name__ == "__main__":
    main()
