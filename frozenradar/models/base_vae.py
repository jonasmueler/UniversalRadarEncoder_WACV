"""Abstract beta-VAE with a Gaussian bottleneck."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BetaVAE(nn.Module, ABC):
    """Abstract beta-VAE for radar data.

    Subclasses implement :meth:`_encode` and :meth:`_decode`.  The bottleneck
    (mu / logvar heads, reparameterization) is handled here.

    Args:
        latent_dim: Dimensionality of the latent space.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # Heads are created by _init_bottleneck once the encoder output dim is known.
        self.mu_head: nn.Linear
        self.logvar_head: nn.Linear

    @abstractmethod
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map model-formatted input to a flat feature vector.

        Args:
            x: Model-formatted input tensor.

        Returns:
            Flat feature vector ``(B, enc_dim)``.
        """
        ...

    @abstractmethod
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent vector to reconstructed input.

        Args:
            z: Latent vector ``(B, latent_dim)``.

        Returns:
            Reconstructed input matching the original input shape.
        """
        ...

    def _init_bottleneck(self, encoder_output_dim: int) -> None:
        """Create mu / logvar projection heads.

        Must be called by the subclass ``__init__`` after the encoder is built.

        Args:
            encoder_output_dim: Flat feature size produced by ``_encode``.
        """
        self.mu_head = nn.Linear(encoder_output_dim, self.latent_dim)
        self.logvar_head = nn.Linear(encoder_output_dim, self.latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample *z* via the reparameterization trick.

        Args:
            mu: Mean of the latent distribution ``(B, latent_dim)``.
            logvar: Log-variance of the latent distribution ``(B, latent_dim)``.

        Returns:
            Sampled latent vector ``(B, latent_dim)``.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encode, bottleneck, decode pass.

        Args:
            x: Model-formatted input tensor.

        Returns:
            ``(x_recon, mu, logvar)``
        """
        features = self._encode(x)
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)
        return x_recon, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to the latent mean (deterministic, for downstream use).

        Args:
            x: Model-formatted input tensor.

        Returns:
            Latent mean ``(B, latent_dim)``.
        """
        features = self._encode(x)
        return self.mu_head(features)
