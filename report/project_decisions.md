# Project Decisions

## Overview
This document records key architectural and methodological decisions made
during the development of the **tail-risk-ddpm** project.

---

## 1. Model choice: Conditional DDPM over VAE / GAN

**Decision**: Use a conditional Denoising Diffusion Probabilistic Model (DDPM)
as the generative backbone.

**Rationale**:
- DDPMs produce higher-quality samples and exhibit better coverage of the true
  data distribution than GANs, which suffer from mode collapse.
- Unlike VAEs, DDPMs do not require a closed-form posterior and therefore
  impose fewer distributional assumptions on financial returns.
- The conditioning mechanism (binary tail label) is straightforward to
  implement via embedding injection into the U-Net time embedding.

**Trade-off**: DDPMs are slower to sample from than GANs or VAEs.
Mitigated by offering a DDIM sampler (50–100 steps instead of 1 000).

---

## 2. Architecture: 1-D temporal U-Net

**Decision**: Use a 1-D convolutional U-Net over the sequence dimension
(assets as channels, time as the spatial axis).

**Rationale**:
- Preserves temporal structure within each scenario window.
- Shared convolutional weights across time steps enable parameter efficiency.
- Skip connections allow low-level temporal features to bypass the bottleneck.

**Alternative considered**: Transformer-based architecture.
Rejected for the initial prototype due to higher memory cost and longer
training time for the target sequence length (≤ 20 days).

---

## 3. Noise schedule: cosine (default)

**Decision**: Default to the cosine noise schedule (Nichol & Dhariwal, 2021).

**Rationale**:
- The cosine schedule avoids near-zero SNR at intermediate timesteps that
  the linear schedule can produce, which helps training stability.
- Empirically leads to slightly lower FID / better sample diversity.

---

## 4. Tail label definition

**Decision**: A day is labelled a *tail event* (label = 1) if **any** asset's
log-return falls below its 5th percentile (computed over the full dataset).

**Rationale**:
- Captures systemic stress events where at least one market component is
  in distress.
- Simple threshold is reproducible and transparent to practitioners.

**Alternatives considered**:
- Portfolio-level return below threshold: more compact but masks individual
  asset contributions.
- Multi-class label (severe / moderate / normal): deferred to future work.

---

## 5. Train / val / test split

**Decision**: Chronological 70 / 15 / 15 split (no shuffling across splits).

**Rationale**:
- Financial data is serially dependent; random splits would cause look-ahead
  bias.
- Chronological split simulates real out-of-sample evaluation.

---

## 6. Attribution methodology

**Decision**: Report (a) marginal VaR/ES and (b) approximate Shapley values
over the generated tail scenarios.

**Rationale**:
- Marginal attribution is fast and intuitive for practitioners.
- Shapley values are theoretically well-grounded (efficiency, symmetry,
  dummy, and additivity axioms) and provide a fairer distribution of
  portfolio tail risk across assets.
- Monte Carlo permutation sampling approximates exact Shapley values in
  O(n_permutations × n_assets) rather than O(2^n_assets).
