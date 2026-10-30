"""Beta-VAE with a 3D HRNet encoder, transformer read-out and projection decoders."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from torch import Tensor

from .base_vae import BetaVAE
from .hrnet_backbone import HRNET_CONFIGS, HighResolution3DNet
from .projections import PROJECTIONS, build_decoders


class HRNetMultiPhaseBetaVAE(BetaVAE):
    """3D HRNet encoder with a transformer read-out and 2D projection decoders.

    The format channels are folded into the Doppler axis, so the 3D HRNet
    convolves over (A, E, R) with ``D * C`` input channels.  Its
    multi-resolution branch outputs are flattened into tokens, projected to a
    common width, and aggregated by a small transformer encoder; the output
    at a learned CLS token feeds the VAE bottleneck.  Decoders reconstruct
    2D log-power projections of the input cube.

    Args:
        latent_dim: Dimensionality of the latent space.
        input_shape: Spatial shape ``(D, A, E, R)`` of one radar frame.
        n_input_channels: Number of format channels C (2 for I/Q, 7 for the
            magnitude + phase-difference format).
        hrnet_config: Key into ``HRNET_CONFIGS``.
        transformer_dim: Token embedding dimension for the transformer.
        transformer_heads: Number of attention heads.
        transformer_layers: Number of transformer encoder layers.
        decoder_hidden: Hidden dimension for projection decoders.
        doppler_gate: Unsupported option; must be False.
        doppler_se: Unsupported option; must be False.
        projection_set: Subset of projection names to reconstruct
            (``None`` = all 6).
        reconstruction_target: Must be ``"projections"`` (2D log-power views).
        backbone_type: Backbone family; only ``"hrnet"`` is supported.

    Raises:
        ValueError: On an unsupported option or an unknown projection name.
    """

    def __init__(
        self,
        latent_dim: int,
        input_shape: tuple[int, int, int, int],
        n_input_channels: int = 7,
        hrnet_config: str = "hr_tiny_feat64_zyx_l4_in64",
        transformer_dim: int = 1024,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        decoder_hidden: int = 256,
        doppler_gate: bool = False,
        doppler_se: bool = False,
        projection_set: list[str] | None = None,
        reconstruction_target: str = "projections",
        backbone_type: str = "hrnet",
    ) -> None:
        super().__init__(latent_dim=latent_dim)
        if doppler_gate or doppler_se:
            raise ValueError("doppler_gate and doppler_se are not supported; set both to false")
        if reconstruction_target != "projections":
            raise ValueError(
                f"unsupported reconstruction_target {reconstruction_target!r}; "
                "only 'projections' is supported"
            )
        if backbone_type != "hrnet":
            raise ValueError(
                f"unsupported backbone_type {backbone_type!r}; only 'hrnet' is supported"
            )
        self.input_dims = input_shape
        self.n_input_channels = n_input_channels
        self.projection_set = projection_set
        self.reconstruction_target = reconstruction_target
        D = input_shape[0]

        # layer1 takes the D * C folded Doppler/format channels directly.
        cfg = copy.deepcopy(HRNET_CONFIGS[hrnet_config])
        cfg["LAYER1"]["INPLANES"] = D * n_input_channels
        self.backbone: nn.Module = HighResolution3DNet(cfg)

        # Branch output shapes (1, C_i, A_i, E_i, R_i) set the token counts.
        with torch.no_grad():
            dummy = torch.zeros(1, D * n_input_channels, *input_shape[1:])
            feat_list = self.backbone(dummy)
            branch_shapes = [f.shape for f in feat_list]

        self.branch_projs = nn.ModuleList()
        self.pos_embeddings = nn.ParameterList()
        for shape in branch_shapes:
            c_branch = shape[1]
            n_tokens = math.prod(shape[2:])
            self.branch_projs.append(nn.Linear(c_branch, transformer_dim))
            self.pos_embeddings.append(
                nn.Parameter(torch.randn(1, n_tokens, transformer_dim) * 0.02),
            )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, transformer_dim) * 0.02,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers,
        )
        self.post_norm = nn.LayerNorm(transformer_dim)

        self._init_bottleneck(transformer_dim)

        self.decoders = build_decoders(
            latent_dim, input_shape, decoder_hidden,
            projection_set=projection_set,
        )

    def _encode(self, x: Tensor) -> Tensor:
        """Encode formatted radar input via HRNet and transformer aggregation.

        Args:
            x: Float input ``(B, D, A, E, R, C)`` with C format channels.

        Returns:
            CLS token output ``(B, transformer_dim)``.
        """
        B, D, A, E, R, C = x.shape
        # Fold format channels into the Doppler axis: (B, D*C, A, E, R), d-major.
        x = x.permute(0, 1, 5, 2, 3, 4).reshape(B, D * C, A, E, R)
        feat_list = self.backbone(x)

        tokens = []
        for i, feat in enumerate(feat_list):
            # (B, C_i, A_i, E_i, R_i) -> (B, N_i, C_i)
            t = feat.float().flatten(2).transpose(1, 2)
            t = self.branch_projs[i](t) + self.pos_embeddings[i]
            tokens.append(t)

        # Prepend CLS token: (B, 1 + total_tokens, transformer_dim)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, *tokens], dim=1)

        tokens = self.transformer(tokens)
        return self.post_norm(tokens[:, 0])

    def _decode(self, z: Tensor) -> Tensor:
        """Single-tensor decoding is not defined for this model.

        Args:
            z: Latent vector ``(B, latent_dim)``.

        Raises:
            NotImplementedError: Always; use :meth:`_decode_projections`.
        """
        raise NotImplementedError("Use forward for multi-projection decoding")

    def _decode_projections(self, z: Tensor) -> dict[str, Tensor]:
        """Decode a latent vector to every configured 2D projection.

        Args:
            z: Latent vector ``(B, latent_dim)``.

        Returns:
            Dict mapping projection name to ``(B, H, W)`` log-power map.
        """
        return {name: dec(z) for name, dec in self.decoders.items()}

    def forward(
        self,
        x: Tensor,
        masks: dict[str, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        """Full encode, bottleneck, multi-decode pass.

        Args:
            x: Input ``(B, D, A, E, R, C)`` float tensor.
            masks: Optional dict with ``"doppler"`` key mapping to a boolean
                tensor ``(B, D)`` where True = keep.  Masked chirps are
                zeroed out before encoding.

        Returns:
            ``(projections_dict, mu, logvar)``
        """
        if masks is not None and masks.get("doppler") is not None:
            dmask = masks["doppler"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x = x * dmask

        features = self._encode(x)
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        z = self.reparameterize(mu, logvar)
        projections = self._decode_projections(z)
        return projections, mu, logvar

    def encode(
        self, x: Tensor, masks: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Encode to latent mean (deterministic, for downstream use).

        Args:
            x: Input ``(B, D, A, E, R, C)`` float tensor.
            masks: Accepted for interface compatibility, ignored.

        Returns:
            Latent mean ``(B, latent_dim)``.
        """
        features = self._encode(x)
        return self.mu_head(features)

    def compute_targets(self, x: Tensor) -> dict[str, Tensor]:
        """Compute reconstruction targets from the formatted input.

        Power is ``|X|^2``: ``I^2 + Q^2`` for the 2-channel I/Q format, or the
        squared magnitude channel (channel 0) for the 7-channel phase format.
        Each target is ``log1p`` of the power summed over the axes that the
        projection removes.

        Args:
            x: Formatted input ``(B, D, A, E, R, C)`` with C=2 (I/Q) or C=7
                (channel 0 = magnitude).

        Returns:
            Dict mapping projection name to its log-power map ``(B, H, W)``.
        """
        if x.shape[-1] == 2:
            power = x[..., 0].pow(2) + x[..., 1].pow(2)
        else:
            power = x[..., 0].pow(2)
        names = (
            list(PROJECTIONS) if self.projection_set is None
            else self.projection_set
        )
        targets = {}
        for name in names:
            proj = power
            for dim in sorted(PROJECTIONS[name], reverse=True):
                proj = proj.sum(dim=dim)
            targets[name] = torch.log1p(proj)
        return targets
