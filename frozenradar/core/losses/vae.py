"""KL term of the beta-VAE objective."""

from __future__ import annotations

import torch


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence between a diagonal Gaussian posterior and ``N(0, I)``.

    The closed form is averaged over batch and latent dimensions, i.e. it
    returns the per-sample KL divided by the latent dimensionality ``d_z``.

    Args:
        mu: Latent mean ``(B, latent_dim)``.
        logvar: Latent log-variance ``(B, latent_dim)``.

    Returns:
        Scalar KL divergence averaged over batch and latent dimensions.
    """
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
