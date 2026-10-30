"""Tests of the mmRadPose raw-ADC frame dataset on a synthetic directory."""

from pathlib import Path

import torch

from frozenradar.core.mmradpose import MmRadPoseRawADC


class TestRawAdcDataset:
    """Windowing, target selection and participant filtering of ``MmRadPoseRawADC``."""

    def test_basic_load(self, tmp_raw_adc_dataset: Path) -> None:
        """A window has the requested length and the expected dtypes.

        Args:
            tmp_raw_adc_dataset: Synthetic dataset root.
        """
        ds = MmRadPoseRawADC(
            root=tmp_raw_adc_dataset,
            time_window=5,
        )
        assert len(ds) > 0
        X, Y = ds[0]
        assert X.dtype == torch.complex64
        assert Y.dtype == torch.float32
        assert X.shape[0] == 5

    def test_target_last(self, tmp_raw_adc_dataset: Path) -> None:
        """``target_frame="last"`` returns the pose of one frame, ``(J, 3)``.

        Args:
            tmp_raw_adc_dataset: Synthetic dataset root.
        """
        ds = MmRadPoseRawADC(
            root=tmp_raw_adc_dataset,
            time_window=5,
            target_frame="last",
        )
        _, Y = ds[0]
        assert Y.dim() == 2

    def test_target_all(self, tmp_raw_adc_dataset: Path) -> None:
        """``target_frame="all"`` returns one pose per window frame, ``(T, J, 3)``.

        Args:
            tmp_raw_adc_dataset: Synthetic dataset root.
        """
        ds = MmRadPoseRawADC(
            root=tmp_raw_adc_dataset,
            time_window=5,
            target_frame="all",
        )
        _, Y = ds[0]
        assert Y.shape[0] == 5

    def test_participant_filter(self, tmp_raw_adc_dataset: Path) -> None:
        """Only the requested participant's recordings are indexed.

        Args:
            tmp_raw_adc_dataset: Synthetic dataset root.
        """
        ds = MmRadPoseRawADC(
            root=tmp_raw_adc_dataset,
            time_window=5,
            participants=["p1"],
        )
        for i in range(len(ds)):
            meta = ds._get_meta(i)
            assert meta.participant == "p1"
