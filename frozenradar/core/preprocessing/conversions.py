"""Conversions between complex tensors and real I/Q tensors."""

from __future__ import annotations

import torch


def complex_to_iq(x: torch.Tensor) -> torch.Tensor:
    """Stack real and imaginary parts as the last dimension.

    Args:
        x: Complex tensor of arbitrary shape.

    Returns:
        Float tensor with an extra trailing dimension of size 2 (I, Q).
    """
    return torch.stack([x.real, x.imag], dim=-1)


def iq_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert I/Q float tensor back to complex.

    Args:
        x: Float tensor ``(..., 2)`` with real and imaginary as last dim.

    Returns:
        Complex tensor with the trailing dimension removed.
    """
    return torch.complex(x[..., 0], x[..., 1])
