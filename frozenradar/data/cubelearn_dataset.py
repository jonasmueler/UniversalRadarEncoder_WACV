"""CubeLearn single-frame dataset for pretraining and embedding extraction."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from frozenradar.core.preprocessing import RawAdcPreprocessor

logger = logging.getLogger(__name__)


class CubeLearnVAEDataset(Dataset):
    """Single-frame CubeLearn dataset for VAE training/extraction.

    Each CubeLearn recording file has shape ``(2, T, 128, 12, 256)`` (real and
    imaginary part, T frames, 128 chirps, 12 virtual antennas, 256 samples).
    Frames are returned as ``(D=128, A=4, E=4, R=256)`` complex64. The IWR6843
    uses 3 TX x 4 RX TDM-MIMO; the 12 virtual antennas are laid out as TX0 x
    RX0-3 (indices 0-3), TX2 x RX0-3 (4-7) and TX1 x RX0-3 (8-11) and are placed
    on the ``(4, 4)`` grid as rows TX0, TX1, TX2, with row 3 zero.

    Args:
        root: Directory containing ``{user}_{gesture}_{sample}.npy`` files.
        users: List of user IDs to include.
        sample_ids: List of sample IDs to include.
        n_gestures: Number of gesture/activity classes.
        frames_per_recording: Frames per file (10 for HGR/AGR, 20 for HAR).
        apply_fft: Whether to apply the 4-D FFT chain.
        clutter_removal: Whether to subtract Doppler-mean clutter.
    """

    def __init__(
        self,
        root: str | Path,
        users: list[int],
        sample_ids: list[int],
        n_gestures: int = 12,
        frames_per_recording: int = 10,
        apply_fft: bool = False,
        clutter_removal: bool = True,
    ) -> None:
        self._root = Path(root)

        self._preprocessor = RawAdcPreprocessor(
            apply_fft=apply_fft,
            clutter_removal=clutter_removal,
        )

        self._index: list[tuple[int, int, int, int]] = []
        for user in users:
            for gesture in range(n_gestures):
                for sample in sample_ids:
                    path = self._root / f"{user}_{gesture}_{sample}.npy"
                    if path.exists():
                        for frame in range(frames_per_recording):
                            self._index.append((user, gesture, sample, frame))

        logger.info(
            "CubeLearn dataset: %d frames from %d recordings (%d users, %d gestures)",
            len(self._index),
            len(self._index) // frames_per_recording,
            len(users),
            n_gestures,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a single preprocessed radar frame.

        Args:
            idx: Frame index.

        Returns:
            Complex64 tensor of shape ``(D=128, A=4, E=4, R=256)``.
        """
        user, gesture, sample, frame_idx = self._index[idx]
        path = self._root / f"{user}_{gesture}_{sample}.npy"

        # Memory-mapped: only the requested frame is read, not the whole recording.
        data = np.load(path, mmap_mode="r")
        frame = np.asarray(data[0, frame_idx]) + 1j * np.asarray(data[1, frame_idx])
        frame = torch.from_numpy(frame.astype(np.complex64))  # (D=128, VA=12, R=256)

        grid = torch.zeros(128, 4, 4, 256, dtype=torch.complex64)
        grid[:, 0, :, :] = frame[:, 0:4, :]    # TX0 x RX0-3
        grid[:, 1, :, :] = frame[:, 8:12, :]   # TX1 x RX0-3
        grid[:, 2, :, :] = frame[:, 4:8, :]    # TX2 x RX0-3
        return self._preprocessor(grid)
