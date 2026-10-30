"""2D log-power projections of the radar cube and their decoders."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

# Projection name -> tensor dims summed out of a batched (B, D, A, E, R) power cube.
PROJECTIONS: dict[str, tuple[int, ...]] = {
    "range_doppler": (2, 3),       # sum over A, E → (B, D, R)
    "range_azimuth": (1, 3),       # sum over D, E → (B, A, R)
    "range_elevation": (1, 2),     # sum over D, A → (B, E, R)
    "doppler_azimuth": (3, 4),     # sum over E, R → (B, D, A)
    "azimuth_elevation": (1, 4),   # sum over D, R → (B, A, E)
    "doppler_elevation": (2, 4),   # sum over A, R → (B, D, E)
}


def get_projection_shape(
    name: str,
    input_dims: tuple[int, int, int, int],
) -> tuple[int, int]:
    """Get output shape for a projection given input dimensions.

    Args:
        name: Projection name (must be in PROJECTIONS).
        input_dims: ``(D, A, E, R)`` input dimensions.

    Returns:
        ``(H, W)`` output shape.
    """
    sum_dims = PROJECTIONS[name]
    # PROJECTIONS indexes batched tensors, so DAER axis i is tensor dim i + 1.
    keep = [input_dims[i] for i in range(4) if (i + 1) not in sum_dims]
    return tuple(keep)


class ProjectionDecoder(nn.Module):
    """Decode latent vector to a 2D log-power radar projection.

    The architecture depends on the output size: an MLP for at most 64
    elements, a linear layer with a residual conv refinement for at most 512,
    and a linear layer followed by a ConvTranspose2d upsampler otherwise.

    Args:
        latent_dim: Input latent dimension.
        output_shape: ``(H, W)`` target shape.
        hidden_dim: Hidden dimension for MLP/conv layers.
    """

    def __init__(
        self,
        latent_dim: int,
        output_shape: tuple[int, ...],
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.output_shape = output_shape
        n_elements = 1
        for s in output_shape:
            n_elements *= s

        if n_elements <= 64:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_elements),
            )
            self.mode = "mlp"
        elif n_elements <= 512:
            self.fc = nn.Linear(latent_dim, n_elements)
            self.refine = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 1, 3, padding=1),
            )
            self.mode = "medium"
        else:
            init_h = max(output_shape[0] // 8, 2)
            init_w = max(output_shape[1] // 8, 2)
            self.init_shape = (init_h, init_w)
            ch = 64

            self.fc = nn.Linear(latent_dim, ch * init_h * init_w)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(ch, 32, 4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            )
            self.mode = "conv"

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent to projection.

        Args:
            z: Latent vector ``(B, latent_dim)``.

        Returns:
            Projection ``(B, H, W)``.
        """
        if self.mode == "mlp":
            return self.net(z).reshape(z.shape[0], *self.output_shape)

        if self.mode == "medium":
            h = self.fc(z).reshape(z.shape[0], 1, *self.output_shape)
            h = h + self.refine(h)
            return h.squeeze(1)

        h = self.fc(z).reshape(z.shape[0], 64, *self.init_shape)
        h = self.decoder(h)
        h = F.interpolate(
            h, size=self.output_shape, mode="bilinear", align_corners=False
        )
        return h.squeeze(1)


def build_decoders(
    latent_dim: int,
    input_dims: tuple[int, int, int, int],
    hidden_dim: int = 256,
    projection_set: list[str] | None = None,
) -> nn.ModuleDict:
    """Build projection decoders (all 6 by default).

    Args:
        latent_dim: Latent space dimension.
        input_dims: ``(D, A, E, R)`` input dimensions.
        hidden_dim: Hidden dimension for decoder networks.
        projection_set: Subset of projection names to build (``None`` = all).

    Returns:
        ModuleDict mapping projection name to decoder module.

    Raises:
        ValueError: If ``projection_set`` contains an unknown name.
    """
    names = list(PROJECTIONS) if projection_set is None else projection_set
    unknown = set(names) - set(PROJECTIONS)
    if unknown:
        raise ValueError(
            f"Unknown projection names {sorted(unknown)}; "
            f"valid: {sorted(PROJECTIONS)}"
        )
    decoders = {}
    for name in names:
        shape = get_projection_shape(name, input_dims)
        decoders[name] = ProjectionDecoder(latent_dim, shape, hidden_dim)
    return nn.ModuleDict(decoders)
