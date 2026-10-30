"""CI4R-ASL and CI4R-Gait pretraining datasets read from train-only stores."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from frozenradar.core.preprocessing import RawAdcPreprocessor

from .ci4r_dataset import _RAW_BIN_RE, decode_frame
from .ci4r_splits import verify_store
from .multi3_dataset import _INT16_PER_FRAME as _M3_INT16_PER_FRAME
from .multi3_dataset import _lvds_to_rx, _rx_to_frames

logger = logging.getLogger(__name__)

# The VAE validation slice is carved deterministically from the benchmark's train
# units; benchmark val and test units never enter either VAE split.
_VAE_VAL_FRACTION = 0.1
_VAE_VAL_SEED = 0


def _partition(items: list, split: str) -> list:
    """Split a unit list into VAE-train / VAE-val deterministically.

    Args:
        items: Units already restricted to the benchmark's train portion.
        split: ``"train"`` or ``"val"``.

    Returns:
        The requested subset, stable across processes and runs.

    Raises:
        ValueError: On an unknown split.
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    perm = np.random.RandomState(_VAE_VAL_SEED).permutation(len(items))
    n_val = max(1, int(round(len(items) * _VAE_VAL_FRACTION))) if items else 0
    chosen = perm[:n_val] if split == "val" else perm[n_val:]
    return [items[i] for i in sorted(chosen)]


def _multi3_base(path: Path) -> str:
    """Return the recording base name of a CI4R-Gait ``.bin`` part.

    Derived exactly as ``extract_ci4r_reps.link_multi3_train`` derives it, so a
    store's files map onto the keys in its stamp.

    Args:
        path: Path of a ``.bin`` part.

    Returns:
        The file name without the ``_Raw_<n>.bin`` suffix (or the stem).
    """
    return _RAW_BIN_RE.sub(r"\1", path.name) if _RAW_BIN_RE.search(path.name) else path.stem


def _even_indices(n: int, keep: int | None) -> list[int]:
    """Return ``keep`` indices spread evenly over ``range(n)``.

    Args:
        n: Number of available indices.
        keep: Number to keep; ``None`` keeps all, ``0`` keeps none.

    Returns:
        Sorted indices.
    """
    if keep is None or keep >= n:
        return list(range(n))
    if keep <= 0:
        return []
    if keep == 1:
        return [0]
    return [round(i * (n - 1) / (keep - 1)) for i in range(keep)]


class CI4RPretrainDataset(Dataset):
    """CI4R-ASL frames for VAE pretraining, read from a train-only store.

    The benchmark splits at the repetition level, so a raw ``.bin`` recording
    carries both train and test frames and can never be handed to pretraining
    whole. This dataset therefore reads the store written by
    ``extract_ci4r_reps.py``, which contains only train-assigned repetitions.

    Each store file is one repetition, an int16 array with one raw frame per
    row, decoded exactly as the extraction path decodes raw captures (2-lane
    LVDS de-interleave, BPM decode, 4x4 virtual-antenna grid, rows 0-1 populated).

    Args:
        root: Store root (``subject_<s>/<base>_rep<r>.npy``).
        split: ``"train"`` or ``"val"`` (a VAE-side slice of the stored units).
        apply_fft: Whether to apply the 4-D FFT chain.
        clutter_removal: Whether to subtract Doppler-mean clutter.
        frames_per_recording: Optional per-repetition cap, evenly spaced.

    Raises:
        ValueError: If the store is empty, holds foreign files, or fails the
            stamp verification of :func:`~frozenradar.data.ci4r_splits.verify_store`.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        apply_fft: bool = False,
        clutter_removal: bool = True,
        frames_per_recording: int | None = None,
    ) -> None:
        self._preprocessor = RawAdcPreprocessor(
            apply_fft=apply_fft, clutter_removal=clutter_removal,
        )
        self._root = Path(root)
        all_files = sorted(self._root.rglob("*.npy"))
        if not all_files:
            raise ValueError(
                f"no repetition files under {root} — run extract_ci4r_reps.py first"
            )
        stray = [f for f in all_files if "_rep" not in f.stem]
        if stray:
            raise ValueError(
                f"{len(stray)} file(s) under {root} are not named <base>_rep<n>.npy "
                f"(e.g. {stray[0].name}); this is not a CI4R-ASL repetition store."
            )
        verify_store(self._root, "ci4r",
                     [f"{f.parent.name.removeprefix('subject_')}|"
                      f"{f.stem.rsplit('_rep', 1)[0]}|{f.stem.rsplit('_rep', 1)[1]}"
                      for f in all_files])
        self._files = _partition(all_files, split)

        # Frame counts come from the .npy header, so indexing loads no sample data.
        self._index: list[tuple[int, int]] = []
        for fi, path in enumerate(self._files):
            n = int(np.load(path, mmap_mode="r").shape[0])
            self._index.extend((fi, f) for f in _even_indices(n, frames_per_recording))

        logger.info(
            "CI4R-ASL pretrain [%s]: %d frames from %d of %d train repetitions",
            split, len(self._index), len(self._files), len(all_files),
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
        fi, frame = self._index[idx]
        # Memory-mapped: a worker reads one 1 MiB frame, not the whole repetition.
        raw = np.asarray(np.load(self._files[fi], mmap_mode="r")[frame])
        return self._preprocessor(decode_frame(raw))


class Multi3PretrainDataset(Dataset):
    """CI4R-Gait frames for VAE pretraining, read from a train-only store.

    CI4R-Gait splits at the recording level, so a whole ``.bin`` belongs to one
    split and the store holds whole train recordings. The decode mirrors
    ``Multi3VAEDataset`` (4-lane LVDS, single TX, row 0 of the grid); the
    VAE-side train/val partition is added on top.

    Args:
        root: Store root holding the train-assigned ``.bin`` recordings.
        split: ``"train"`` or ``"val"``.
        apply_fft: Whether to apply the 4-D FFT chain.
        clutter_removal: Whether to subtract Doppler-mean clutter.
        frames_per_recording: Optional per-recording cap, evenly spaced.

    Raises:
        ValueError: If the store is empty or fails the stamp verification.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        apply_fft: bool = False,
        clutter_removal: bool = True,
        frames_per_recording: int | None = None,
    ) -> None:
        self._preprocessor = RawAdcPreprocessor(
            apply_fft=apply_fft, clutter_removal=clutter_removal,
        )
        self._root = Path(root)
        all_files = sorted(self._root.rglob("*.bin"))
        if not all_files:
            raise ValueError(f"no .bin recordings under {root} — run extract_ci4r_reps.py first")
        # The store has the same layout as the raw benchmark tree; only the stamp
        # tells a train store from the raw tree, which contains the test split.
        verify_store(self._root, "multi3", [
            f"{f.parent.name}|{_multi3_base(f)}" for f in all_files
        ])
        self._files = _partition(all_files, split)

        self._index: list[tuple[int, int]] = []
        for fi, path in enumerate(self._files):
            n = path.stat().st_size // 2 // _M3_INT16_PER_FRAME
            if n == 0:
                continue
            self._index.extend((fi, f) for f in _even_indices(n, frames_per_recording))

        logger.info(
            "CI4R-Gait pretrain [%s]: %d frames from %d of %d train recordings",
            split, len(self._index), len(self._files), len(all_files),
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
        fi, frame = self._index[idx]
        offset = frame * _M3_INT16_PER_FRAME
        raw = np.fromfile(
            self._files[fi], dtype=np.int16, count=_M3_INT16_PER_FRAME, offset=offset * 2,
        )
        return self._preprocessor(_rx_to_frames(_lvds_to_rx(raw), 1)[0])
