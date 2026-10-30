"""CI4R-Gait raw-capture parsing and the single-frame extraction dataset."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from frozenradar.core.preprocessing import RawAdcPreprocessor

logger = logging.getLogger(__name__)

# Radar configuration (the dataset's datToImage_77.m):
#   - 256 ADC samples per chirp (the native range length of the encoder)
#   - 4 RX, SINGLE TX (no MIMO): the reference script extracts 4 RX lanes and
#     treats consecutive chirps as pure Doppler, so there is no TX demux.
#   - 128 chirps per frame -> 128 Doppler samples
#   - DCA1000 4-lane complex layout: every 8 int16 hold one range sample across
#     the 4 RX as [I0 I1 I2 I3 Q0 Q1 Q2 Q3] (datToImage_77.m: reshape to 8 rows,
#     rows 1-4 real, rows 5-8 imaginary).
NTS = 256
NUM_RX = 4
NUM_LANES = 4
CHIRP_LOOPS = 128

# One raw chirp slot: NTS range samples x NUM_RX channels, complex -> 2 int16.
_INT16_PER_CHIRP = NTS * NUM_RX * 2
# One frame consumes CHIRP_LOOPS chirp slots (single TX, no MIMO multiplexing).
_INT16_PER_FRAME = _INT16_PER_CHIRP * CHIRP_LOOPS

_RAW_BIN_RE = re.compile(r"(.+?)_?Raw_?(\d+)\.bin$")

# Activity folder prefix -> class index (11 activities).
ACTIVITY_CODES = ["05", "06", "07", "08", "09", "10", "11", "16", "17", "18", "19"]
ACTIVITY_NAMES = [
    "walking_towards",
    "walking_away",
    "picking_up",
    "bending",
    "sitting",
    "kneeling",
    "crawling",
    "walking_on_toes",
    "limping",
    "short_steps",
    "scissors_gait",
]
_CODE_TO_IDX = {c: i for i, c in enumerate(ACTIVITY_CODES)}


def _lvds_to_rx(raw: np.ndarray) -> np.ndarray:
    """Decode the DCA1000 4-lane int16 stream into per-RX complex samples.

    Every 8 int16 hold one range sample across the 4 RX channels as
    ``[I0 I1 I2 I3 Q0 Q1 Q2 Q3]`` (datToImage_77.m). Trailing values that do
    not fill a group of 8 are dropped.

    Args:
        raw: Flat int16 array from a ``.bin`` capture.

    Returns:
        Complex64 array ``(n_time_samples, NUM_RX)`` where time samples run
        range-fastest within each chirp.
    """
    usable = (raw.size // (NUM_LANES * 2)) * (NUM_LANES * 2)
    m = raw[:usable].reshape(-1, NUM_LANES * 2).astype(np.float32)
    return (m[:, 0:NUM_LANES] + 1j * m[:, NUM_LANES:]).astype(np.complex64)


def _rx_to_frames(rx: np.ndarray, n_frames: int) -> torch.Tensor:
    """Assemble per-RX streams into VAE input frames.

    The 4 virtual antennas fill row 0 of the ``(4, 4)`` grid (TX0 x RX0-3);
    rows 1-3 stay zero, the same zero-padded-TX-row convention as CubeLearn,
    here with a single physical TX.

    Args:
        rx: ``(n_time_samples, NUM_RX)`` complex per-RX samples.
        n_frames: Number of full frames to assemble.

    Returns:
        Complex64 tensor ``(n_frames, D=128, A=4, E=4, R=256)``.
    """
    frames = torch.zeros(n_frames, CHIRP_LOOPS, 4, 4, NTS, dtype=torch.complex64)
    n_used = n_frames * CHIRP_LOOPS * NTS
    for rx_ch in range(NUM_RX):
        # (n_frames, D=128, R=256): time samples run range-fastest within a
        # chirp, chirps run within a frame.
        cube = rx[:n_used, rx_ch].reshape(n_frames, CHIRP_LOOPS, NTS)
        frames[:, :, 0, rx_ch, :] = torch.from_numpy(np.ascontiguousarray(cube))
    return frames


def group_recordings(root: str | Path) -> dict[str, list[Path]]:
    """Group the .bin parts of each recording, ordered by part index.

    DCA1000 may split a capture into ``<name>_Raw_0.bin, _Raw_1.bin, ...``;
    the CI4R-Gait recordings are single-part, but the pattern is handled. The
    recording key keeps the activity folder so the label can be recovered.
    Empty files are skipped.

    Args:
        root: Directory searched recursively for ``.bin`` files.

    Returns:
        Mapping of recording key (part path without the ``_Raw_<n>.bin`` suffix,
        activity folder included) to its ordered part paths.
    """
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(Path(root).rglob("*.bin")):
        if path.stat().st_size == 0:
            continue
        match = _RAW_BIN_RE.search(path.name)
        if match:
            key = str(path.parent / match.group(1))
            groups.setdefault(key, []).append((int(match.group(2)), path))
        else:
            groups.setdefault(str(path.with_suffix("")), []).append((0, path))
    return {key: [p for _, p in sorted(parts)] for key, parts in groups.items()}


def activity_from_key(key: str) -> int | None:
    """Return the activity class index for a recording key, or None.

    The activity is the numeric prefix of the parent folder name
    (e.g. ``05_Walking_Towards_radar`` -> code ``05``).

    Args:
        key: Recording key from :func:`group_recordings`.

    Returns:
        Class index 0-10, or ``None`` if the folder is not a known activity.
    """
    folder = Path(key).parent.name
    return _CODE_TO_IDX.get(folder.split("_", 1)[0])


def _read_int16_slice(
    paths: list[Path], sizes: list[int], start: int, end: int
) -> np.ndarray:
    """Read a global int16 range from a recording's concatenated .bin parts.

    Args:
        paths: Ordered .bin parts.
        sizes: int16 count of each part.
        start: Global start index (inclusive).
        end: Global end index (exclusive).

    Returns:
        int16 array of length ``end - start`` (zero-padded past the stream).
    """
    out = np.zeros(end - start, dtype=np.int16)
    file_start = 0
    for path, n in zip(paths, sizes):
        file_end = file_start + n
        lo, hi = max(start, file_start), min(end, file_end)
        if lo < hi:
            mm = np.memmap(path, dtype=np.int16, mode="r")
            out[lo - start : hi - start] = mm[lo - file_start : hi - file_start]
        file_start = file_end
    return out


class Multi3VAEDataset(Dataset):
    """Single-frame CI4R-Gait dataset for frozen-encoder embedding extraction.

    Iterates the frames of every recording under ``root`` in a fixed,
    documented order (recordings sorted by key, frames in temporal order).
    Frames are read lazily via memmap so a worker never holds a full parsed
    recording in memory.

    Args:
        root: Directory containing the ``.bin`` files in activity subdirectories.
        apply_fft: Whether to apply the 4-D FFT chain. Must match the
            encoder's training config.
        clutter_removal: Whether to subtract Doppler-mean clutter.
    """

    def __init__(
        self,
        root: str | Path,
        apply_fft: bool = False,
        clutter_removal: bool = True,
    ) -> None:
        self._preprocessor = RawAdcPreprocessor(
            apply_fft=apply_fft,
            clutter_removal=clutter_removal,
        )
        self._recordings = group_recordings(root)
        self._keys = sorted(self._recordings)

        self._index: list[tuple[str, int]] = []
        self._frame_counts: dict[str, int] = {}
        self._part_sizes: dict[str, list[int]] = {}
        for key in self._keys:
            sizes = [p.stat().st_size // 2 for p in self._recordings[key]]
            self._part_sizes[key] = sizes
            n_frames = sum(sizes) // _INT16_PER_FRAME
            self._frame_counts[key] = n_frames
            self._index.extend((key, f) for f in range(n_frames))

        logger.info(
            "CI4R-Gait dataset: %d frames from %d recordings under %s",
            len(self._index), len(self._keys), root,
        )

    @property
    def index(self) -> list[tuple[str, int]]:
        """Flat (recording_key, frame_idx) index in iteration order."""
        return self._index

    @property
    def frame_counts(self) -> dict[str, int]:
        """Number of frames per recording key."""
        return self._frame_counts

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a single preprocessed radar frame.

        Args:
            idx: Frame index.

        Returns:
            Complex64 tensor of shape ``(D=128, A=4, E=4, R=256)``.
        """
        key, frame_idx = self._index[idx]
        raw = _read_int16_slice(
            self._recordings[key],
            self._part_sizes[key],
            frame_idx * _INT16_PER_FRAME,
            (frame_idx + 1) * _INT16_PER_FRAME,
        )
        rx = _lvds_to_rx(raw)
        frame = _rx_to_frames(rx, 1)[0]
        return self._preprocessor(frame)
