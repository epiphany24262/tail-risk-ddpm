from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from diffusion import DDPM
from model import ConditionalMLPDenoiser


DATA_DIR = Path("data/processed")
TRAIN_PATH = DATA_DIR / "dataset_train.npz"
VALID_PATH = DATA_DIR / "dataset_valid.npz"
CHECKPOINT_DIR = Path("outputs/checkpoints")
LOG_DIR = Path("outputs/logs")
TRAIN_CONFIG_PATH = Path("configs/train.yaml")
WINDOW_LENGTH = 20
COND_DIM = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conditional DDPM with tail-weighted loss.")
    parser.add_argument("--config", default=str(TRAIN_CONFIG_PATH), help="Path to train yaml config")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for config seed")
    parser.add_argument(
        "--tail-weight",
        type=float,
        default=None,
        help="Optional override for config tail_weight",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def load_split_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    x = data["X"].astype(np.float32)
    c = data["C"].astype(np.float32)
    y_tail = data["y_tail"].astype(np.int64)
    return x, c, y_tail


def make_loader(x: np.ndarray, c: np.ndarray, y_tail: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(c),
        torch.from_numpy(y_tail),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def run_epoch(
    model: ConditionalMLPDenoiser,
    ddpm: DDPM,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    tail_weight: float,
    normal_weight: float,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses: list[float] = []

    for x, c, y_tail in loader:
        x = x.to(device)
        c = c.to(device)
        y_tail = y_tail.to(device)

        loss = ddpm.loss(
            model=model,
            x_start=x,
            c=c,
            y_tail=y_tail,
            tail_weight=tail_weight,
            normal_weight=normal_weight,
        )

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


def save_checkpoint(
    path: Path,
    model: ConditionalMLPDenoiser,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    n_assets: int,
    best_valid_loss: float,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "n_assets": int(n_assets),
        "cond_dim": int(COND_DIM),
        "window_length": int(WINDOW_LENGTH),
        "timesteps": 1000,
        "best_valid_loss": float(best_valid_loss),
    }
    torch.save(ckpt, path)


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    model_name = str(config.get("model_name", "conditional_ddpm_mlp"))
    if model_name != "conditional_ddpm_mlp":
        print(
            f"[warn] Unsupported model_name='{model_name}'. "
            "This implementation uses conditional_ddpm_mlp."
        )
    epochs = int(config.get("epochs", 50))
    batch_size = int(config.get("batch_size", 64))
    learning_rate = float(config.get("learning_rate", 1e-3))
    tail_weight = float(args.tail_weight) if args.tail_weight is not None else float(config.get("tail_weight", 3.0))
    normal_weight = float(config.get("normal_weight", 1.0))
    seed = int(args.seed) if args.seed is not None else int(config.get("seed", 42))

    cfg_device = str(config.get("device", "cpu")).lower()
    if cfg_device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable. Falling back to cpu.")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg_device)

    set_seed(seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"[config] config={Path(args.config)} seed={seed} tail_weight={tail_weight:.4f} "
        f"epochs={epochs} batch_size={batch_size} lr={learning_rate}"
    )

    x_train, c_train, y_train = load_split_dataset(TRAIN_PATH)
    x_valid, c_valid, y_valid = load_split_dataset(VALID_PATH)
    if x_train.ndim != 3 or x_train.shape[1] != WINDOW_LENGTH:
        raise ValueError(f"X train shape must be [B, {WINDOW_LENGTH}, N], got {x_train.shape}")

    n_assets = int(x_train.shape[-1])
    if c_train.shape[1] != COND_DIM:
        raise ValueError(f"C train condition dim must be {COND_DIM}, got {c_train.shape[1]}")
    if c_valid.shape[1] != COND_DIM:
        raise ValueError(f"C valid condition dim must be {COND_DIM}, got {c_valid.shape[1]}")

    train_loader = make_loader(x_train, c_train, y_train, batch_size=batch_size, shuffle=True)
    valid_loader = make_loader(x_valid, c_valid, y_valid, batch_size=batch_size, shuffle=False)

    model = ConditionalMLPDenoiser(
        window_length=WINDOW_LENGTH,
        n_assets=n_assets,
        cond_dim=COND_DIM,
        hidden_dim=128,
    ).to(device)
    ddpm = DDPM(timesteps=1000, device=str(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    latest_path = CHECKPOINT_DIR / "latest.pt"
    best_path = CHECKPOINT_DIR / "best.pt"
    log_path = LOG_DIR / "train_log.csv"
    best_valid = float("inf")

    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "valid_loss", "lr"])

        for epoch in range(1, epochs + 1):
            train_loss = run_epoch(
                model=model,
                ddpm=ddpm,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                tail_weight=tail_weight,
                normal_weight=normal_weight,
            )
            with torch.no_grad():
                valid_loss = run_epoch(
                    model=model,
                    ddpm=ddpm,
                    loader=valid_loader,
                    device=device,
                    optimizer=None,
                    tail_weight=tail_weight,
                    normal_weight=normal_weight,
                )

            writer.writerow([epoch, f"{train_loss:.8f}", f"{valid_loss:.8f}", learning_rate])
            f.flush()

            save_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                n_assets=n_assets,
                best_valid_loss=best_valid,
            )
            if valid_loss < best_valid:
                best_valid = valid_loss
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    n_assets=n_assets,
                    best_valid_loss=best_valid,
                )

            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
                f"valid_loss={valid_loss:.6f} best_valid={best_valid:.6f}"
            )

    print(f"[saved] {latest_path}")
    print(f"[saved] {best_path}")
    print(f"[saved] {log_path}")


if __name__ == "__main__":
    main()
