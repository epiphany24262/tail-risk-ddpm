"""
diffusion.py
------------
Implements the DDPM (Ho et al., 2020) forward / reverse diffusion process
with support for both the original linear noise schedule and the improved
cosine schedule (Nichol & Dhariwal, 2021).

Key objects
-----------
GaussianDiffusion
    .q_sample(x0, t, noise)   – forward process
    .p_losses(model, x0, t, label) – training loss
    .p_sample_loop(model, shape, label) – full reverse sampling
    .ddim_sample_loop(model, shape, label, steps, eta) – DDIM sampling
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 8e-3) -> Tensor:
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.9999)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _extract(a: Tensor, t: Tensor, x_shape: Tuple) -> Tensor:
    """Gather values from 1-D tensor `a` at indices `t` and reshape."""
    b = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(b, *((1,) * (len(x_shape) - 1))).to(t.device)


# ---------------------------------------------------------------------------
# Core diffusion class
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    """
    Gaussian DDPM forward/reverse process.

    Parameters
    ----------
    timesteps    : total diffusion steps T
    beta_schedule: "linear" or "cosine"
    beta_start   : start value for linear schedule
    beta_end     : end value for linear schedule
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        self.T = timesteps

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule!r}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register("betas", betas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register("log_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).log())
        self.register("sqrt_recip_alphas_cumprod", (1.0 / alphas_cumprod).sqrt())
        self.register(
            "sqrt_recipm1_alphas_cumprod", (1.0 / alphas_cumprod - 1).sqrt()
        )

        # Posterior q(x_{t-1} | x_t, x_0) variance
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register("posterior_variance", posterior_variance)
        self.register(
            "posterior_log_variance_clipped",
            posterior_variance.clamp(min=1e-20).log(),
        )
        self.register(
            "posterior_mean_coef1",
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
        )
        self.register(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def register(self, name: str, value: Tensor) -> None:
        """Store a tensor as a plain attribute (no nn.Module needed)."""
        setattr(self, name, value)

    def _to(self, device: torch.device) -> "GaussianDiffusion":
        """Move all schedule tensors to `device`."""
        for attr in [
            "betas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "log_one_minus_alphas_cumprod", "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod", "posterior_variance",
            "posterior_log_variance_clipped",
            "posterior_mean_coef1", "posterior_mean_coef2",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample x_t ~ q(x_t | x_0) = N(sqrt_bar_alpha * x0, (1-bar_alpha)*I)."""
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    # ------------------------------------------------------------------
    # Training objective
    # ------------------------------------------------------------------

    def p_losses(
        self,
        model: torch.nn.Module,
        x0: Tensor,
        t: Tensor,
        label: Tensor,
        noise: Optional[Tensor] = None,
        loss_type: str = "l2",
    ) -> Tensor:
        """
        Compute the simplified DDPM training loss L_simple = ||ε - ε_θ||².

        Parameters
        ----------
        model     : denoising network
        x0        : clean input (B, C, L)
        t         : diffusion step  (B,)
        label     : conditioning label (B,)
        noise     : optional pre-sampled noise
        loss_type : "l2" (MSE) or "l1"
        """
        if noise is None:
            noise = torch.randn_like(x0)

        x_noisy = self.q_sample(x0, t, noise)
        predicted = model(x_noisy, t, label)

        if loss_type == "l2":
            return F.mse_loss(predicted, noise)
        elif loss_type == "l1":
            return F.l1_loss(predicted, noise)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type!r}")

    # ------------------------------------------------------------------
    # Reverse process – DDPM sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        x_t: Tensor,
        t: Tensor,
        label: Tensor,
    ) -> Tensor:
        """One reverse step: sample x_{t-1} ~ p_θ(x_{t-1} | x_t)."""
        betas_t = _extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = _extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )

        # Predict x_0 mean from ε_θ
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * model(x_t, t, label) / sqrt_one_minus_alphas_cumprod_t
        )
        if (t == 0).all():
            return model_mean
        posterior_var = _extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + posterior_var.sqrt() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: torch.nn.Module,
        shape: Tuple,
        label: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Full DDPM reverse diffusion from x_T ~ N(0, I) to x_0."""
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, dtype=torch.long, device=device)
            x = self.p_sample(model, x, t, label)
        return x

    # ------------------------------------------------------------------
    # Reverse process – DDIM sampler (faster, deterministic/stochastic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: torch.nn.Module,
        shape: Tuple,
        label: Tensor,
        device: torch.device,
        steps: int = 50,
        eta: float = 0.0,
    ) -> Tensor:
        """
        DDIM sampling (Song et al., 2020).

        Parameters
        ----------
        steps : number of denoising steps (can be << T)
        eta   : 0 = deterministic DDIM, 1 = stochastic (≈ DDPM)
        """
        # Uniformly spaced subset of timesteps
        step_seq = np.linspace(0, self.T - 1, steps, dtype=int)[::-1]
        x = torch.randn(shape, device=device)

        for i, t_val in enumerate(step_seq):
            t = torch.full((shape[0],), t_val, dtype=torch.long, device=device)
            t_prev = torch.full(
                (shape[0],),
                step_seq[i + 1] if i + 1 < len(step_seq) else 0,
                dtype=torch.long,
                device=device,
            )
            alpha_bar = _extract(self.alphas_cumprod, t, x.shape)
            alpha_bar_prev = _extract(self.alphas_cumprod, t_prev, x.shape)

            eps = model(x, t, label)
            x0_pred = (x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
            x0_pred = x0_pred.clamp(-5, 5)

            sigma = (
                eta
                * ((1 - alpha_bar_prev) / (1 - alpha_bar)).sqrt()
                * (1 - alpha_bar / alpha_bar_prev).sqrt()
            )
            dir_xt = (1 - alpha_bar_prev - sigma ** 2).sqrt() * eps
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = alpha_bar_prev.sqrt() * x0_pred + dir_xt + sigma * noise

        return x


def build_diffusion(cfg) -> GaussianDiffusion:
    """Instantiate GaussianDiffusion from OmegaConf diffusion sub-config."""
    return GaussianDiffusion(
        timesteps=cfg.timesteps,
        beta_schedule=cfg.beta_schedule,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
    )
