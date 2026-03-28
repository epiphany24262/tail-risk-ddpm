"""
sample.py
---------
Load a trained checkpoint and generate synthetic return scenarios using
either the DDPM or DDIM sampler.

Usage
-----
    python src/sample.py
    python src/sample.py --cfg configs/sample.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from model import build_model
from diffusion import build_diffusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def sample(cfg_path: str = "configs/sample.yaml") -> np.ndarray:
    cfg = OmegaConf.load(cfg_path)
    train_cfg = OmegaConf.load(cfg.train_cfg)
    data_cfg = OmegaConf.load("configs/data.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # ---------- model ----------
    model = build_model(train_cfg.model).to(device)
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Loaded checkpoint from %s (epoch %s)", cfg.checkpoint, ckpt.get("epoch", "?"))

    # ---------- diffusion ----------
    diffusion = build_diffusion(train_cfg.diffusion)
    diffusion._to(device)

    # ---------- sampling ----------
    seq_len = data_cfg.seq_len
    n_assets = train_cfg.model.in_channels
    shape_per_batch = (cfg.batch_size, n_assets, seq_len)

    all_samples: list[np.ndarray] = []
    remaining = cfg.num_samples
    torch.manual_seed(cfg.seed)

    while remaining > 0:
        batch_n = min(cfg.batch_size, remaining)
        shape = (batch_n, n_assets, seq_len)
        label_val = cfg.tail_label if cfg.condition_on_tail else 0
        label = torch.full((batch_n,), label_val, dtype=torch.long, device=device)

        if cfg.sampler.method == "ddpm":
            samples = diffusion.p_sample_loop(model, shape, label, device)
        elif cfg.sampler.method == "ddim":
            samples = diffusion.ddim_sample_loop(
                model, shape, label, device,
                steps=cfg.sampler.ddim_steps,
                eta=cfg.sampler.eta,
            )
        else:
            raise ValueError(f"Unknown sampler method: {cfg.sampler.method!r}")

        all_samples.append(samples.cpu().numpy())
        remaining -= batch_n
        log.info("Generated %d / %d scenarios …", cfg.num_samples - remaining, cfg.num_samples)

    scenarios = np.concatenate(all_samples, axis=0)  # (N, C, L)

    # ---------- save ----------
    out_dir = Path(cfg.samples_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.samples_file
    np.savez_compressed(out_path, scenarios=scenarios)
    log.info("Saved %d scenarios → %s", len(scenarios), out_path)

    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scenarios with trained DDPM.")
    parser.add_argument("--cfg", default="configs/sample.yaml")
    args = parser.parse_args()
    sample(args.cfg)


if __name__ == "__main__":
    main()
