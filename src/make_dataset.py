"""
make_dataset.py
---------------
Build train/val/test PyTorch Dataset objects from preprocessed CSV files
and expose a helper that returns DataLoaders for each split.

Usage
-----
    python src/make_dataset.py          # quick sanity check
    python src/make_dataset.py --cfg configs/data.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class ReturnSequenceDataset(Dataset):
    """
    Sliding-window dataset of normalised return sequences with tail labels.

    Each sample is a tuple (x, label) where:
        x     : float32 tensor of shape (seq_len, n_assets)
        label : int64 scalar tensor (0 = normal, 1 = tail event)
    The label corresponds to the *last* day in the window.
    """

    def __init__(
        self,
        returns: np.ndarray,
        labels: np.ndarray,
        seq_len: int,
    ) -> None:
        self.returns = returns.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.seq_len = seq_len
        assert len(returns) == len(labels), "returns and labels length mismatch"

    def __len__(self) -> int:
        return max(0, len(self.returns) - self.seq_len + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.returns[idx : idx + self.seq_len]  # (seq_len, n_assets)
        label = self.labels[idx + self.seq_len - 1]
        return torch.from_numpy(x), torch.tensor(label)


def load_splits(
    cfg_path: str = "configs/data.yaml",
) -> Tuple[ReturnSequenceDataset, ReturnSequenceDataset, ReturnSequenceDataset]:
    cfg = OmegaConf.load(cfg_path)
    processed_dir = Path(cfg.processed_dir)

    returns = pd.read_csv(
        processed_dir / "returns_normalised.csv", index_col=0, parse_dates=True
    ).values
    labels = pd.read_csv(
        processed_dir / "tail_labels.csv", index_col=0, parse_dates=True
    ).values.squeeze()

    n = len(returns)
    train_end = int(n * cfg.split.train)
    val_end = train_end + int(n * cfg.split.val)

    datasets = {}
    for split, (start, end) in zip(
        ["train", "val", "test"],
        [(0, train_end), (train_end, val_end), (val_end, n)],
    ):
        datasets[split] = ReturnSequenceDataset(
            returns[start:end], labels[start:end], cfg.seq_len
        )
        log.info("%5s split: %d windows", split, len(datasets[split]))

    return datasets["train"], datasets["val"], datasets["test"]


def get_dataloaders(
    cfg_path: str = "configs/data.yaml",
    train_cfg_path: str = "configs/train.yaml",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_cfg = OmegaConf.load(train_cfg_path)
    train_ds, val_ds, test_ds = load_splits(cfg_path)

    train_dl = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_dl, val_dl, test_dl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build datasets and loaders.")
    parser.add_argument("--cfg", default="configs/data.yaml")
    args = parser.parse_args()

    train_ds, val_ds, test_ds = load_splits(args.cfg)
    x, y = train_ds[0]
    log.info("Sample shape: x=%s, label=%s", x.shape, y)


if __name__ == "__main__":
    main()
