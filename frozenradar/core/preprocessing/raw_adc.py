"""Clutter removal and 4D FFT chain for raw ADC radar frames."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RawAdcPreprocessor:
    """Preprocessing pipeline for raw ADC radar data.

    Optionally removes static clutter and applies a 4D FFT chain (range,
    elevation, azimuth, Doppler) to complex-valued radar frames.

    Args:
        apply_fft: Run the 4D FFT pipeline.
        fft_norm: Normalisation mode passed to ``torch.fft.fft``
            (``"forward"`` divides by N).
        clutter_removal: Subtract the mean across the Doppler (chirp)
            dimension before the FFT, which removes static clutter.
    """

    apply_fft: bool = True
    fft_norm: str = "forward"
    clutter_removal: bool = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Process a raw radar frame or sequence.

        Args:
            x: Complex tensor.  Single frame ``(D, A, E, R)`` or temporal
                sequence ``(T, D, A, E, R)``.

        Returns:
            Processed complex tensor with the same shape as ``x``.
        """
        if self.clutter_removal:
            x = x - x.mean(dim=-4, keepdim=True)

        if self.apply_fft:
            x = self._fft_chain(x)

        return x

    def _fft_chain(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 1D FFTs along all four spatial axes, then centre each spectrum.

        Args:
            x: Complex tensor whose last four dims are (Doppler, Azimuth,
                Elevation, Range).

        Returns:
            FFT-processed complex tensor with zero frequency at the centre
            of every axis.
        """
        x = torch.fft.fft(x, dim=-1, norm=self.fft_norm)
        x = torch.fft.fft(x, dim=-2, norm=self.fft_norm)
        x = torch.fft.fft(x, dim=-3, norm=self.fft_norm)
        x = torch.fft.fft(x, dim=-4, norm=self.fft_norm)

        x = torch.fft.fftshift(x, dim=-1)
        x = torch.fft.fftshift(x, dim=-2)
        x = torch.fft.fftshift(x, dim=-3)
        x = torch.fft.fftshift(x, dim=-4)

        return x
