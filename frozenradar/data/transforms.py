"""Conversion of complex radar frames into the encoder input format."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from frozenradar.core.preprocessing import complex_to_iq

_ENCODER_TYPES = ("hrnet_multi_phase",)


def _phase_diff(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cos/sin of the forward phase difference along ``dim``.

    Uses ``x[i+1] * conj(x[i])`` so the angle is the forward phase difference
    without wrapping. The last position along ``dim`` is zero-padded.

    Args:
        x: Complex tensor.
        dim: Dimension along which to take the difference.

    Returns:
        ``(cos_dphi, sin_dphi)`` float tensors with the shape of ``x``.
    """
    size = x.shape[dim]
    x_curr = x.narrow(dim, 0, size - 1)
    x_next = x.narrow(dim, 1, size - 1)
    product = x_next * x_curr.conj()
    product_abs = product.abs().clamp(min=1e-8)
    cos_dphi = product.real / product_abs
    sin_dphi = product.imag / product_abs

    # F.pad takes (before, after) pairs from the last dimension backwards.
    ndim = x.ndim
    pad = [0, 0] * ndim
    pad[2 * (ndim - 1 - dim % ndim) + 1] = 1
    return F.pad(cos_dphi, pad), F.pad(sin_dphi, pad)


def format_phase_diff(x: torch.Tensor, fft_norm: str = "forward") -> torch.Tensor:
    """Convert pre-FFT complex radar frames to magnitude plus phase differences.

    Applies the range FFT only, then computes the magnitude and the phase
    differences along the Doppler (chirp), azimuth and elevation axes.

    Args:
        x: Batched complex radar frames ``(B, D, A, E, R)`` after clutter
            removal and before any FFT.
        fft_norm: Normalisation mode of the range FFT.

    Returns:
        Float tensor ``(B, D, A, E, R, 7)`` with channels
        ``[mag, cos_dD, sin_dD, cos_dA, sin_dA, cos_dE, sin_dE]``.
    """
    x = torch.fft.fft(x, dim=-1, norm=fft_norm)
    x = torch.fft.fftshift(x, dim=-1)

    channels = [x.abs()]
    for dim in (-4, -3, -2):
        cos_dp, sin_dp = _phase_diff(x, dim)
        channels.append(cos_dp)
        channels.append(sin_dp)
    return torch.stack(channels, dim=-1)


def normalize_magnitude_channel(x: torch.Tensor) -> torch.Tensor:
    """Rescale the magnitude channel (index 0) to unit per-frame RMS.

    The cos/sin phase channels (indices 1-6) are bounded in ``[-1, 1]``; only
    the magnitude carries a sensor-specific gain, which this removes.

    Args:
        x: Phase-difference tensor ``(B, D, A, E, R, 7)``.

    Returns:
        A copy of ``x`` with channel 0 rescaled to unit RMS per batch element.
    """
    mag = x[..., 0]
    dims = tuple(range(1, mag.dim()))
    rms = mag.pow(2).mean(dim=dims, keepdim=True).sqrt()
    x = x.clone()
    x[..., 0] = mag / (rms + 1e-8)
    return x


def normalize_complex_rms(x: torch.Tensor) -> torch.Tensor:
    """Rescale complex frames to unit per-frame RMS magnitude.

    Removes the sensor-specific gain difference between datasets while
    preserving relative phase.

    Args:
        x: Batched complex radar frames ``(B, D, A, E, R)``.

    Returns:
        The complex tensor rescaled to unit RMS per batch element.
    """
    dims = tuple(range(1, x.dim()))
    rms = x.abs().pow(2).mean(dim=dims, keepdim=True).sqrt()
    return x / rms.clamp(min=1e-8)


def format_for_model(
    x: torch.Tensor,
    encoder_type: str,
    format_mode: str = "iq",
    normalize_magnitude: bool = False,
) -> torch.Tensor:
    """Convert complex radar frames into the encoder's input tensor.

    Args:
        x: Batched complex radar frames ``(B, D, A, E, R)``.
        encoder_type: ``model.encoder_type`` of the config.
        format_mode: ``"iq"`` stacks real and imaginary parts; ``"phase_diff"``
            produces magnitude plus phase-difference channels.
        normalize_magnitude: Rescale to unit per-frame RMS (the magnitude
            channel for ``"phase_diff"``, the complex signal for ``"iq"``).

    Returns:
        Float tensor with the format channels as the last dimension.

    Raises:
        ValueError: On an unknown encoder type.
    """
    if format_mode == "phase_diff":
        out = format_phase_diff(x)
        if normalize_magnitude:
            out = normalize_magnitude_channel(out)
        return out

    if normalize_magnitude:
        x = normalize_complex_rms(x)
    if encoder_type not in _ENCODER_TYPES:
        raise ValueError(f"Unknown encoder type: {encoder_type!r}")
    return complex_to_iq(x)
