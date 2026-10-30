"""CI4R-ASL raw-capture parsing and the single-frame extraction dataset."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from frozenradar.core.preprocessing import RawAdcPreprocessor

logger = logging.getLogger(__name__)

# Radar configuration (the dataset's RDC_extract.m / RDC_to_rangeDopp.m):
#   - 256 ADC samples per chirp (the native range length of the encoder)
#   - 2 TX x 4 RX with BPM MIMO: chirp pairs carry TX0+TX1 and TX0-TX1
#   - 128 chirp loops per 40 ms frame -> 128 decoded Doppler samples per TX
NTS = 256
NUM_RX = 4
NUM_TX = 2
CHIRP_LOOPS = 128

# One raw chirp slot in the .bin stream: NTS samples x NUM_RX channels, complex
# int16 pairs -> 2 int16 per complex sample.
_INT16_PER_CHIRP = NTS * NUM_RX * 2
# One decoded frame consumes NUM_TX * CHIRP_LOOPS raw chirp slots.
_INT16_PER_FRAME = _INT16_PER_CHIRP * NUM_TX * CHIRP_LOOPS

_RAW_BIN_RE = re.compile(r"(.+?)_?Raw_?(\d+)\.bin$")


def bpm_decode(lvds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate the two BPM-coded transmitters via sum/difference.

    Consecutive chirps carry ``TX0 + TX1`` and ``TX0 - TX1``; the per-TX
    streams are recovered as ``TX0 = (a + b) / 2`` and ``TX1 = (a - b) / 2``.
    Unlike the reference MATLAB (which duplicates decoded chirps to keep the
    raw chirp count), each decoded stream keeps only its unique chirps, so a
    40 ms frame yields exactly ``CHIRP_LOOPS`` Doppler samples per TX.

    Args:
        lvds: ``(NTS * NUM_RX, n_chirps)`` complex64 raw chirp matrix.

    Returns:
        Tuple ``(tx0, tx1)``, each ``(NTS * NUM_RX, n_chirps // 2)``.
    """
    n_pairs = lvds.shape[1] // 2
    a = lvds[:, 0 : 2 * n_pairs : 2]
    b = lvds[:, 1 : 2 * n_pairs : 2]
    return 0.5 * (a + b), 0.5 * (a - b)


def to_frames(tx0: np.ndarray, tx1: np.ndarray) -> torch.Tensor:
    """Assemble decoded per-TX streams into VAE input frames.

    The 8 virtual antennas fill the first two rows of the ``(4, 4)`` grid
    (row 0 = TX0 x RX0-3, row 1 = TX1 x RX0-3); rows 2-3 stay zero — the same
    zero-padded-TX-row convention as the CubeLearn remap.

    Args:
        tx0: ``(NTS * NUM_RX, n_chirps)`` decoded TX0 stream.
        tx1: Same for TX1.

    Returns:
        Complex64 tensor ``(n_frames, D=128, A=4, E=4, R=256)``.
    """
    n_frames = tx0.shape[1] // CHIRP_LOOPS
    frames = torch.zeros(
        n_frames, CHIRP_LOOPS, 4, 4, NTS, dtype=torch.complex64
    )
    for tx_row, stream in ((0, tx0), (1, tx1)):
        for rx in range(NUM_RX):
            # (NTS, n_chirps) for this virtual antenna.
            chan = stream[rx * NTS : (rx + 1) * NTS, : n_frames * CHIRP_LOOPS]
            cube = chan.T.reshape(n_frames, CHIRP_LOOPS, NTS)
            frames[:, :, tx_row, rx, :] = torch.from_numpy(
                np.ascontiguousarray(cube)
            )
    return frames


def decode_frame(raw: np.ndarray) -> torch.Tensor:
    """Decode the int16 samples of one frame into a ``(4, 4)`` antenna-grid frame.

    The DCA1000 writes int16 in the 2-lane interleaved layout: every four values
    are ``I1, I2, Q1, Q2`` of two consecutive complex samples. A frame starts on
    an even chirp, so the BPM chirp pairs are contained in it.

    Args:
        raw: ``_INT16_PER_FRAME`` int16 values of one frame.

    Returns:
        Complex64 tensor ``(D=128, A=4, E=4, R=256)``.
    """
    lvds = np.empty(raw.size // 2, dtype=np.complex64)
    lvds[0::2] = raw[0::4] + 1j * raw[2::4].astype(np.float32)
    lvds[1::2] = raw[1::4] + 1j * raw[3::4].astype(np.float32)
    matrix = lvds.reshape(NUM_TX * CHIRP_LOOPS, NTS * NUM_RX).T
    tx0, tx1 = bpm_decode(matrix)
    return to_frames(tx0, tx1)[0]


def group_recordings(root: str | Path) -> dict[str, list[Path]]:
    """Group the .bin parts of each recording, ordered by part index.

    DCA1000 splits one capture into ``<name>_Raw_0.bin, _Raw_1.bin, ...``;
    parts are byte-level continuations and must be concatenated in order.
    Files not matching the pattern are treated as single-part recordings.

    Args:
        root: Directory searched recursively for ``.bin`` files.

    Returns:
        Mapping of recording key (part path without the ``_Raw_<n>.bin`` suffix)
        to its ordered part paths.
    """
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(Path(root).rglob("*.bin")):
        match = _RAW_BIN_RE.search(path.name)
        if match:
            key = str(path.parent / match.group(1))
            groups.setdefault(key, []).append((int(match.group(2)), path))
        else:
            groups.setdefault(str(path.with_suffix("")), []).append((0, path))
    return {
        key: [p for _, p in sorted(parts)] for key, parts in groups.items()
    }


def _read_int16_slice(
    paths: list[Path], sizes: list[int], start: int, end: int
) -> np.ndarray:
    """Read a global int16 range from a recording's concatenated .bin parts.

    Parts are byte-level continuations of one capture; the requested range
    may straddle part boundaries. Only the needed bytes are touched (memmap).

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


class CI4RVAEDataset(Dataset):
    """Single-frame CI4R-ASL dataset for frozen-encoder embedding extraction.

    Iterates the frames of every recording under ``root`` in a fixed,
    documented order (recordings sorted by key, frames in temporal order).
    Frames are read lazily via memmap: one frame touches ~1 MB of raw data, so
    workers never hold a parsed recording (~13 GB) in memory.

    Args:
        root: Directory containing the subject's raw ``.bin`` files.
        apply_fft: Whether to apply the 4-D FFT chain. Must match the
            encoder's training config (``cfg.dataset.apply_fft``).
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

        # Flat index: (recording_key, frame_idx). Frame counts and per-part
        # int16 sizes are derived from file sizes without parsing.
        self._index: list[tuple[str, int]] = []
        self._frame_counts: dict[str, int] = {}
        self._part_sizes: dict[str, list[int]] = {}
        for key in self._keys:
            sizes = [p.stat().st_size // 2 for p in self._recordings[key]]
            self._part_sizes[key] = sizes
            n_chirps = sum(sizes) // _INT16_PER_CHIRP
            n_frames = (n_chirps // NUM_TX) // CHIRP_LOOPS
            self._frame_counts[key] = n_frames
            self._index.extend((key, f) for f in range(n_frames))

        logger.info(
            "CI4R-ASL dataset: %d frames from %d recordings under %s",
            len(self._index),
            len(self._keys),
            root,
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
        return self._preprocessor(decode_frame(raw))
