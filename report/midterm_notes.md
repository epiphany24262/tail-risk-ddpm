# Midterm Notes

## Status Summary

| Milestone | Status |
|-----------|--------|
| Data download & preprocessing | Complete |
| Dataset / DataLoader | Complete |
| Model skeleton (1-D U-Net) | Complete |
| Diffusion forward/reverse process | Complete |
| Training loop | Complete |
| DDIM sampler | Complete |
| Evaluation (VaR, ES, KS, Spearman) | Complete |
| Attribution (marginal + Shapley) | Complete |
| Exploratory notebooks | Complete |
| End-to-end run on real data | Pending |
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

- Cosine noise schedule converges faster than linear schedule in initial experiments.
- Tail events are sparse and regime-clustered, so effective sample size for extremes remains limited.
- DDIM with fewer steps is significantly faster while preserving sample quality in preliminary checks.

---

## Next Steps

- [x] Run `download_data.py` -> `preprocess.py` end-to-end.
- [ ] Run `train.py` and `sample.py` end-to-end on current A-share dataset.
- [ ] Tune key hyperparameters on validation split.
- [ ] Generate larger tail scenario sets and run full evaluation pipeline.
- [ ] Finalize report section with quantitative tables and figures.
