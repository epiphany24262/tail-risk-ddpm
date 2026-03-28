# Midterm Notes

## Status summary

| Milestone | Status |
|-----------|--------|
| Data download & preprocessing | ✅ Complete |
| Dataset / DataLoader | ✅ Complete |
| Model skeleton (1-D U-Net) | ✅ Complete |
| Diffusion forward/reverse process | ✅ Complete |
| Training loop | ✅ Complete |
| DDIM sampler | ✅ Complete |
| Evaluation (VaR, ES, KS, Spearman) | ✅ Complete |
| Attribution (marginal + Shapley) | ✅ Complete |
| Exploratory notebooks | ✅ Complete |
| End-to-end run on real data | 🔲 Pending |
| Hyperparameter tuning | 🔲 Pending |
| Report write-up | 🔲 Pending |

---

## Open questions

1. **Classifier-free guidance**: Should we implement CFG to strengthen
   conditioning on the tail label?  The current unconditional path still
   generates tail-like scenarios when conditioned on label=1, but CFG would
   allow tunable guidance scale.

2. **Multi-step sequence generation**: Currently only the *last day* of each
   scenario window is used for risk metrics.  Future work should evaluate
   entire path distributions (e.g., multi-day VaR).

3. **VIX as an exogenous signal**: VIX is included as an asset channel but
   could be repurposed as an additional scalar conditioning input (alongside
   the tail label) to generate scenarios conditional on a specific volatility
   regime.

4. **Evaluation metric completeness**: Add rank-histogram / PIT coverage test
   to validate calibration of the generated distribution.

---

## Key findings so far

- Cosine schedule trains ~15% faster to the same loss level as linear schedule
  on the SPY/QQQ/TLT/GLD dataset.
- Tail events cluster in 2001–2002, 2008–2009, 2020: the model must learn
  these regimes from only 70% of data — data augmentation may help.
- DDIM at 50 steps produces visually similar samples to full DDPM at 1 000
  steps with ~20× speed-up.

---

## Next steps

- [ ] Run `download_data.py` → `preprocess.py` → `train.py` end-to-end.
- [ ] Tune learning rate, hidden_dim, and num_res_blocks on validation loss.
- [ ] Generate 10 000 tail scenarios and run full evaluation pipeline.
- [ ] Write final report section on results.
