"""Temporal transformer head classifying sequences of frozen-encoder embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn

ALPHA_SCHEDULES = ("fixed",)


class TemporalClassifier(nn.Module):
    """Temporal transformer classifier for sequences of VAE latents.

    During training the input is ``mu + alpha_max * std * eps`` (with
    ``alpha_max = 0`` the head sees the posterior mean only);
    during evaluation it is ``mu``. A CLS token aggregates the sequence through
    self-attention and a linear head produces the class logits.

    Args:
        embed_dim: Latent embedding dimension (VAE ``latent_dim``).
        temporal_layers: Number of transformer encoder layers.
        max_seq_len: Maximum sequence length for the positional embedding.
        num_heads: Number of attention heads.
        ff_ratio: Feed-forward expansion ratio.
        dropout: Dropout rate.
        num_classes: Number of classes.
        alpha_schedule: Noise schedule; only ``"fixed"`` is supported.
        alpha_max: Scale of the posterior noise added during training.

    Raises:
        ValueError: On an unsupported ``alpha_schedule``.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        temporal_layers: int = 4,
        max_seq_len: int = 1024,
        num_heads: int = 8,
        ff_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 12,
        alpha_schedule: str = "fixed",
        alpha_max: float = 0.0,
    ) -> None:
        super().__init__()
        if alpha_schedule not in ALPHA_SCHEDULES:
            raise ValueError(
                f"alpha_schedule must be one of {ALPHA_SCHEDULES}, got {alpha_schedule!r}"
            )
        self.embed_dim = embed_dim
        self.alpha_schedule = alpha_schedule
        self.alpha_max = alpha_max
        self._epoch_progress: float = 0.0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(ff_ratio * embed_dim),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_layers,
            norm=nn.LayerNorm(embed_dim),
        )
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise the CLS token and the positional embedding."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        """Record the training progress (the fixed schedule does not depend on it).

        Args:
            epoch: Current epoch (1-indexed).
            total_epochs: Total number of training epochs.
        """
        self._epoch_progress = epoch / total_epochs

    def _sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Return the head input: ``mu`` plus scaled posterior noise in training.

        Args:
            mu: Mean ``(B, T, D)``.
            logvar: Log-variance ``(B, T, D)``.

        Returns:
            Latent ``(B, T, D)``.
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (std * eps) * self.alpha_max

    def forward(
        self,
        mu: Tensor,
        logvar: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Classify a batch of variable-length embedding sequences.

        Args:
            mu: Latent means ``(B, T, D)``.
            logvar: Latent log-variances ``(B, T, D)``.
            padding_mask: Bool mask ``(B, T)``, True for padded positions.

        Returns:
            Logits ``(B, num_classes)``.
        """
        B, T, D = mu.shape

        z = self._sample(mu, logvar)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, z], dim=1)
        x = x + self.pos_embed[:, : T + 1]

        # The CLS position is never masked.
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mu.device)
            src_mask = torch.cat([cls_mask, padding_mask], dim=1)
        else:
            src_mask = None

        x = self.encoder(x, src_key_padding_mask=src_mask)
        return self.head(x[:, 0])
