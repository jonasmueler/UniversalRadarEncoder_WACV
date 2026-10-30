"""mmRadPose single-frame dataset for beta-VAE pretraining and embedding extraction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from frozenradar.core.mmradpose import MmRadPoseRawADC
from frozenradar.core.preprocessing import RawAdcPreprocessor


class RadarVAEDataset(Dataset):
    """Single-frame radar dataset for unsupervised beta-VAE training.

    Wraps ``MmRadPoseRawADC`` with ``time_window=1`` so each sample is one
    radar frame. Ground-truth poses are discarded; only the radar tensor is
    returned.

    Args:
        root: Path to mmRadPose raw ADC directory.
        participants: Participant IDs to include.
        angles: Angle IDs to include (``None`` = all).
        apply_fft: Whether to apply the 4-D FFT chain.
        clutter_removal: Whether to subtract Doppler-mean clutter.
        target_range_bins: If set, zero-pad the range axis to this many bins.
    """

    def __init__(
        self,
        root: str,
        participants: list[str],
        angles: list[str] | None = None,
        apply_fft: bool = True,
        clutter_removal: bool = True,
        target_range_bins: int | None = None,
    ) -> None:
        self._inner = MmRadPoseRawADC(
            root=root,
            participants=participants,
            angles=angles,
            time_window=1,
            target_frame="last",
            preload_gt=False,
            return_meta=False,
        )
        self._preprocessor = RawAdcPreprocessor(
            apply_fft=apply_fft,
            clutter_removal=clutter_removal,
        )
        self._target_range_bins = target_range_bins

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a single preprocessed radar frame.

        Args:
            idx: Frame index.

        Returns:
            Complex64 tensor of shape ``(D, A, E, R)``.
        """
        radar, _pose = self._inner[idx]
        radar = radar.squeeze(0)

        if self._target_range_bins and radar.shape[-1] < self._target_range_bins:
            pad_size = self._target_range_bins - radar.shape[-1]
            radar = F.pad(radar, (0, pad_size))

        return self._preprocessor(radar)
