"""Tests for the beta-VAE loss terms."""

from __future__ import annotations

import pytest
import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as torch_kl

from frozenradar.core.losses import kl_divergence
from frozenradar.training.multi_losses import multi_projection_loss


class TestKLDivergence:
    """Tests for :func:`kl_divergence`."""

    def test_zero_at_prior(self) -> None:
        """The posterior N(0, I) has zero KL to the prior."""
        mu = torch.zeros(4, 8)
        logvar = torch.zeros(4, 8)
        assert kl_divergence(mu, logvar).item() == pytest.approx(0.0, abs=1e-7)

    def test_matches_closed_form_per_dimension(self) -> None:
        """The value equals the analytic KL averaged over batch and latent dims."""
        torch.manual_seed(0)
        mu = torch.randn(4, 8)
        logvar = torch.randn(4, 8)
        ref = torch_kl(Normal(mu, (0.5 * logvar).exp()), Normal(0.0, 1.0)).mean()
        assert kl_divergence(mu, logvar).item() == pytest.approx(ref.item(), rel=1e-5)


class TestMultiProjectionLoss:
    """Tests for :func:`multi_projection_loss`."""

    def test_total_is_recon_plus_beta_kl(self) -> None:
        """The total is the summed projection MSE plus beta times the KL."""
        torch.manual_seed(0)
        projections = {"a": torch.randn(2, 4, 5), "b": torch.randn(2, 3, 3)}
        targets = {k: torch.randn_like(v) for k, v in projections.items()}
        mu = torch.randn(2, 6)
        logvar = torch.randn(2, 6)
        total, metrics = multi_projection_loss(projections, targets, mu, logvar, beta=0.1)
        recon = sum(
            torch.nn.functional.mse_loss(projections[k], targets[k]) for k in projections
        )
        expected = recon + 0.1 * kl_divergence(mu, logvar)
        assert total.item() == pytest.approx(expected.item(), rel=1e-6)
        assert metrics["recon_total"] == pytest.approx(recon.item(), rel=1e-6)
        assert set(metrics) == {"recon/a", "recon/b", "recon_total", "kl", "loss"}

    def test_projection_weights(self) -> None:
        """A zero weight removes a projection from the reconstruction term."""
        torch.manual_seed(0)
        projections = {"a": torch.randn(2, 4, 5), "b": torch.randn(2, 3, 3)}
        targets = {k: torch.randn_like(v) for k, v in projections.items()}
        mu = torch.zeros(2, 6)
        logvar = torch.zeros(2, 6)
        _, metrics = multi_projection_loss(
            projections, targets, mu, logvar, beta=1.0, projection_weights={"a": 1.0, "b": 0.0}
        )
        assert metrics["recon_total"] == pytest.approx(metrics["recon/a"], rel=1e-6)
