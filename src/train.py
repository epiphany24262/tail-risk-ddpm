"""
train.py
--------
Training loop for the conditional DDPM.

Usage
-----
    python src/train.py
    python src/train.py --train_cfg configs/train.yaml --data_cfg configs/data.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from make_dataset import get_dataloaders
from model import build_model
from diffusion import build_diffusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(train_cfg_path: str = "configs/train.yaml",
          data_cfg_path: str = "configs/data.yaml") -> None:
    train_cfg = OmegaConf.load(train_cfg_path)
    set_seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # ---------- data ----------
    train_dl, val_dl, _ = get_dataloaders(data_cfg_path, train_cfg_path)

    # ---------- model & diffusion ----------
    model = build_model(train_cfg.model).to(device)
    diffusion = build_diffusion(train_cfg.diffusion)
    diffusion._to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{n_params:,}")

    # ---------- optimiser ----------
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.optimizer.lr,
        weight_decay=train_cfg.optimizer.weight_decay,
        betas=tuple(train_cfg.optimizer.betas),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.scheduler.T_max,
        eta_min=train_cfg.scheduler.eta_min,
    )

    # ---------- output dirs ----------
    ckpt_dir = Path(train_cfg.checkpoint_dir)
    log_dir = Path(train_cfg.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Optional wandb
    if train_cfg.use_wandb:
        import wandb
        wandb.init(project=train_cfg.wandb_project, config=OmegaConf.to_container(train_cfg))

    best_val_loss = float("inf")

    for epoch in range(1, train_cfg.num_epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for x, label in tqdm(train_dl, desc=f"Epoch {epoch} [train]", leave=False):
            x = x.permute(0, 2, 1).to(device)  # (B, C, L)
            label = label.to(device)

            t = torch.randint(0, diffusion.T, (x.size(0),), device=device)
            loss = diffusion.p_losses(model, x, t, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_dl.dataset)
        scheduler.step()

        # ---- validate ----
        if epoch % train_cfg.eval_every == 0 or epoch == train_cfg.num_epochs:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, label in val_dl:
                    x = x.permute(0, 2, 1).to(device)
                    label = label.to(device)
                    t = torch.randint(0, diffusion.T, (x.size(0),), device=device)
                    loss = diffusion.p_losses(model, x, t, label)
                    val_loss += loss.item() * x.size(0)
            val_loss /= len(val_dl.dataset)
            log.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | lr=%.2e",
                epoch, train_cfg.num_epochs, train_loss, val_loss,
                scheduler.get_last_lr()[0],
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict(),
                     "val_loss": val_loss},
                    ckpt_dir / "best.pt",
                )
                log.info("  ✓ New best checkpoint saved (val_loss=%.4f)", val_loss)
            if train_cfg.use_wandb:
                import wandb
                wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                           "lr": scheduler.get_last_lr()[0], "epoch": epoch})
        else:
            log.info(
                "Epoch %d/%d | train_loss=%.4f | lr=%.2e",
                epoch, train_cfg.num_epochs, train_loss,
                scheduler.get_last_lr()[0],
            )

        # ---- periodic checkpoint ----
        if epoch % train_cfg.save_every == 0:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                ckpt_dir / f"epoch_{epoch:04d}.pt",
            )

    log.info("Training complete. Best val_loss=%.4f", best_val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train conditional DDPM.")
    parser.add_argument("--train_cfg", default="configs/train.yaml")
    parser.add_argument("--data_cfg", default="configs/data.yaml")
    args = parser.parse_args()
    train(args.train_cfg, args.data_cfg)


if __name__ == "__main__":
    main()
