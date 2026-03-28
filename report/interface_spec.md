# Interface Specification

## Module interfaces

This document describes the public interfaces for each Python module in `src/`.

---

### `download_data.py`

```python
def download(cfg_path: str = "configs/data.yaml") -> None:
    """
    Download adjusted-close price data for all tickers specified in the
    data config and save per-ticker CSVs plus a combined panel to data/raw/.
    """
```

**Config keys used**: `tickers`, `start_date`, `end_date`, `raw_dir`

**Outputs**:
- `data/raw/{TICKER}.csv`  — single-column price series
- `data/raw/prices.csv`    — combined panel (date × ticker)

---

### `preprocess.py`

```python
def compute_returns(prices: pd.DataFrame, return_type: str = "log") -> pd.DataFrame: ...
def rolling_zscore(returns: pd.DataFrame, window: int) -> pd.DataFrame: ...
def make_tail_label(returns: pd.DataFrame, quantile: float) -> pd.Series: ...
def preprocess(cfg_path: str = "configs/data.yaml") -> None: ...
```

**Config keys used**: `raw_dir`, `processed_dir`, `return_type`, `window_size`,
`tail_quantile`

**Outputs**:
- `data/processed/returns_raw.csv`         — unscaled log-returns
- `data/processed/returns_normalised.csv`  — rolling z-score normalised returns
- `data/processed/tail_labels.csv`         — binary tail-event labels

---

### `make_dataset.py`

```python
class ReturnSequenceDataset(Dataset):
    """__getitem__ returns (x: Tensor[seq_len, n_assets], label: Tensor[])"""

def load_splits(cfg_path: str) -> Tuple[Dataset, Dataset, Dataset]: ...
def get_dataloaders(cfg_path: str, train_cfg_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]: ...
```

**Config keys used**: `processed_dir`, `split`, `seq_len`, `batch_size`

---

### `model.py`

```python
class ConditionalUNet1D(nn.Module):
    def forward(self, x: Tensor, t: Tensor, label: Tensor) -> Tensor:
        """
        x     : (B, C, L)  noisy return sequence
        t     : (B,)       diffusion timestep
        label : (B,)       binary tail label
        Returns predicted noise of shape (B, C, L)
        """

def build_model(cfg) -> ConditionalUNet1D: ...
```

**Config keys used** (under `model` sub-key): `in_channels`, `hidden_dim`,
`num_res_blocks`, `attn_resolutions`, `dropout`, `cond_dim`

---

### `diffusion.py`

```python
class GaussianDiffusion:
    def q_sample(self, x0: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor: ...
    def p_losses(self, model, x0, t, label, noise=None, loss_type="l2") -> Tensor: ...
    def p_sample_loop(self, model, shape, label, device) -> Tensor: ...
    def ddim_sample_loop(self, model, shape, label, device, steps=50, eta=0.0) -> Tensor: ...

def build_diffusion(cfg) -> GaussianDiffusion: ...
```

**Config keys used** (under `diffusion` sub-key): `timesteps`, `beta_schedule`,
`beta_start`, `beta_end`

---

### `train.py`

```python
def train(train_cfg_path: str, data_cfg_path: str) -> None: ...
```

**Config keys used**: all keys in `configs/train.yaml` and `configs/data.yaml`

**Outputs**:
- `outputs/checkpoints/best.pt`          — best validation checkpoint
- `outputs/checkpoints/epoch_{N:04d}.pt` — periodic checkpoints

---

### `sample.py`

```python
def sample(cfg_path: str = "configs/sample.yaml") -> np.ndarray:
    """Returns array of shape (num_samples, n_assets, seq_len)."""
```

**Config keys used**: all keys in `configs/sample.yaml`

**Outputs**:
- `outputs/samples/generated_scenarios.npz` — compressed numpy array

---

### `evaluate.py`

```python
def var_historical(returns: np.ndarray, alpha: float) -> np.ndarray: ...
def expected_shortfall(returns: np.ndarray, alpha: float) -> np.ndarray: ...
def portfolio_var_es(returns: np.ndarray, weights: np.ndarray, alpha: float) -> dict: ...
def ks_test_marginals(real, synthetic, asset_names) -> pd.DataFrame: ...
def spearman_corr(returns: np.ndarray) -> np.ndarray: ...
def evaluate(cfg_path: str) -> None: ...
```

**Outputs**:
- `outputs/tables/var_es_comparison.csv`
- `outputs/tables/ks_tests.csv`
- `outputs/tables/spearman_corr_diff.csv`

---

### `attribution.py`

```python
def marginal_var_attribution(returns, weights, alpha, epsilon) -> pd.Series: ...
def marginal_es_attribution(returns, weights, alpha, epsilon) -> pd.Series: ...
def approx_shapley_es(returns, weights, alpha, n_permutations, seed) -> pd.Series: ...
def run_attribution(cfg_path: str) -> None: ...
```

**Outputs**:
- `outputs/tables/attribution.csv`

---

## End-to-end pipeline

```
download_data.py   →  data/raw/prices.csv
preprocess.py      →  data/processed/{returns_normalised,tail_labels}.csv
train.py           →  outputs/checkpoints/best.pt
sample.py          →  outputs/samples/generated_scenarios.npz
evaluate.py        →  outputs/tables/{var_es_comparison,ks_tests,spearman_corr_diff}.csv
attribution.py     →  outputs/tables/attribution.csv
notebooks/03_figures.ipynb  →  outputs/figures/*.png
```
