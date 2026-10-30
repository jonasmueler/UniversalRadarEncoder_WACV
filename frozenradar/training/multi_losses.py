"""Beta-VAE objective over the 2D log-power projection reconstructions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from frozenradar.core.losses import kl_divergence


def multi_projection_loss(
    projections: dict[str, Tensor],
    targets: dict[str, Tensor],
    mu: Tensor,
    logvar: Tensor,
    beta: float,
    projection_weights: dict[str, float] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Beta-VAE loss with one MSE reconstruction term per projection.

    Each projection's MSE is a mean over its elements, so large projections
    do not dominate the sum.  The total is
    ``sum_p w_p * MSE_p + beta * KL / d_z``.

    Args:
        projections: Dict of decoder outputs ``{name: (B, H, W)}``.
        targets: Dict of target projections ``{name: (B, H, W)}``.
        mu: Latent mean ``(B, latent_dim)``.
        logvar: Latent log-variance ``(B, latent_dim)``.
        beta: KL divergence weight.
        projection_weights: Optional per-projection loss weights.

    Returns:
        Tuple of (total_loss, metrics_dict) where metrics_dict contains
        per-projection losses and totals as floats.
    """
    if projection_weights is None:
        projection_weights = {k: 1.0 for k in projections}

    recon_loss = torch.tensor(0.0, device=mu.device)
    metrics: dict[str, float] = {}

    for name, pred in projections.items():
        target = targets[name]
        proj_loss = F.mse_loss(pred, target)
        weight = projection_weights.get(name, 1.0)
        recon_loss = recon_loss + weight * proj_loss
        metrics[f"recon/{name}"] = proj_loss.item()

    kl = kl_divergence(mu, logvar)
    total = recon_loss + beta * kl

    metrics["recon_total"] = recon_loss.item()
    metrics["kl"] = kl.item()
    metrics["loss"] = total.item()
    return total, metrics
