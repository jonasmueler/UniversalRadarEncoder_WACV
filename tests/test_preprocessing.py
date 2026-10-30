"""Tests for the raw ADC preprocessing pipeline."""

import torch

from frozenradar.core.preprocessing import (
    RawAdcPreprocessor,
    complex_to_iq,
    iq_to_complex,
)


class TestRawAdcPreprocessor:
    """Tests for :class:`RawAdcPreprocessor`."""

    def test_fft_chain(self) -> None:
        """The FFT chain keeps shape and complex dtype."""
        pp = RawAdcPreprocessor(apply_fft=True, clutter_removal=True)
        x = torch.randn(8, 4, 4, 32, dtype=torch.complex64)
        out = pp(x)
        assert out.shape == x.shape
        assert out.dtype == torch.complex64

    def test_no_fft(self) -> None:
        """Without FFT and clutter removal the input passes through unchanged."""
        pp = RawAdcPreprocessor(apply_fft=False, clutter_removal=False)
        x = torch.randn(8, 4, 4, 32, dtype=torch.complex64)
        out = pp(x)
        assert torch.allclose(out, x)


def test_iq_round_trip() -> None:
    """Complex -> I/Q -> complex is the identity."""
    x = torch.randn(3, 5, dtype=torch.complex64)
    iq = complex_to_iq(x)
    assert iq.shape == (3, 5, 2)
    assert torch.equal(iq_to_complex(iq), x)
