# Midterm Notes

## Status Summary

| Milestone | Status |
|-----------|--------|
| Data download & preprocessing | Complete |
| Dataset / DataLoader | Complete |
| Model skeleton (MLP backbone) | Complete |
| Diffusion forward/reverse process | Complete |
| Training loop | Complete |
| DDPM sampler | Complete |
| Evaluation (VaR, ES, KS, Spearman) | Complete |
| Attribution (single-factor what-if) | Complete |
| Exploratory notebooks | Complete (refreshed and aligned with current A/B/D outputs) |
| End-to-end run on real data | Complete (A->B->C->D) |
| Hyperparameter tuning | Pending |
| Report write-up | Pending |

---

## Open Questions

1. **Classifier-free guidance**: Should we implement CFG to strengthen conditioning on the tail label?
2. **Multi-step sequence generation**: Current evaluation focuses on the last day in each window; should we extend to multi-day risk metrics?
3. **Exogenous volatility signal**: Should volatility regime be provided as an explicit conditioning scalar?
4. **Calibration checks**: Should we add rank-histogram / PIT coverage tests?

---

## Key Findings So Far

- A module produces `prices.parquet` with full-history coverage from 2015 onward.
- B module produces balanced-panel datasets (`dataset_train/valid/test.npz`) from common dates.
- With current asset pool, common-date range starts at 2020-11-16 due to asset listing dates.
- C module now trains and samples successfully on current dataset with saved checkpoints and sample artifacts.
- D module now generates evaluation and attribution tables/figures in `outputs/tables` and `outputs/figures`.
- Notebooks are now runnable and mapped to current outputs:
  `01_data_check.ipynb`, `02_tail_label.ipynb`, `03_figures.ipynb`.
- Current B split caveat: validation tail density is much lower than train/test, so validation is a weak stress-regime proxy.
- Current D finding: the evaluation layer is working, but generated distributions remain materially miscalibrated relative to real test data.

---

## Next Steps

- [x] Run `download_data.py` -> `preprocess.py` end-to-end.
- [x] Run `make_dataset.py` and generate `dataset_*.npz`.
- [x] Implement `model.py`, `diffusion.py`, `train.py`, and `sample.py`.
- [x] Implement `evaluate.py` and `attribution.py`.
- [x] Run end-to-end training, sampling, and evaluation.
- [ ] Finalize report section with quantitative tables and figures.
