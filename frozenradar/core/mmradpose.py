"""Loader for the mmRadPose raw ADC radar cubes and skeleton labels."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from frozenradar.core.dataset_base import RadarPoseDataset, SampleMeta

logger = logging.getLogger(__name__)


class MmRadPoseRawADC(RadarPoseDataset):
    """mmRadPose dataset loader for raw ADC / preprocessed radar cubes.

    Loads complex-valued 4D radar tensors (Doppler, Azimuth, Elevation, Range)
    paired with skeleton ground truth from ``.npz`` archives.

    File naming convention::

        data_cube_parsed_{participant}_{angle}_{action}_{recording}.npz
        skeleton_{participant}_{angle}_{action}_{recording}.npy

    Args:
        root: Path to directory containing ``.npz`` and ``.npy`` files.
        participants: Include-list of participant IDs (e.g. ``["p1", "p3"]``).
            ``None`` loads all.
        angles: Include-list of angle IDs (e.g. ``["an0"]``).  ``None`` loads all.
        time_window: Number of frames per temporal window.
        target_frame: ``"last"`` returns GT for the last frame in the window;
            ``"all"`` returns GT for every frame.
        unit_scale: Multiplicative factor for ground truth.
        center_to_root: Subtract root joint from all keypoints.
        preload_gt: Cache ground truth arrays in RAM.
        return_meta: Include :class:`SampleMeta` in ``__getitem__`` output.

    Raises:
        RuntimeError: If no recordings or no complete windows are found.
    """

    def __init__(
        self,
        root: str | Path,
        participants: list[str] | None = None,
        angles: list[str] | None = None,
        time_window: int = 8,
        target_frame: str = "last",
        unit_scale: float = 100.0,
        center_to_root: bool = True,
        preload_gt: bool = True,
        return_meta: bool = False,
    ) -> None:
        super().__init__()
        self.return_meta = return_meta
        self.time_window = time_window
        self.target_frame = target_frame
        self.unit_scale = unit_scale
        self.center_to_root = center_to_root

        self._root = Path(root).expanduser().resolve()
        self._recordings: list[_RecordingMeta] = []
        self._windows: list[_WindowEntry] = []
        self._gt_cache: dict[int, np.ndarray] = {}

        self._index_data(
            participants=set(participants) if participants else None,
            angles=set(angles) if angles else None,
        )

        if preload_gt:
            self._preload_ground_truth()

    def _index_data(
        self,
        participants: set[str] | None = None,
        angles: set[str] | None = None,
    ) -> None:
        """Find all radar/skeleton file pairs and build window index.

        Args:
            participants: Participant IDs to include (``None`` = all).
            angles: Angle IDs to include (``None`` = all).

        Raises:
            RuntimeError: If no window fits into any recording.
        """
        self._recordings = _discover_recordings(self._root, participants, angles)

        for rec_idx, rec in enumerate(self._recordings):
            prep_dir = self._root / "preprocessed" / rec.stem / "radar"
            if prep_dir.exists():
                T = len(list(prep_dir.glob("*.npy")))
            else:
                gt = np.load(rec.gt_path)
                if gt.ndim == 2 and gt.shape[1] != 3:
                    num_kp = gt.shape[1] // 3
                    gt = gt.reshape(gt.shape[0], num_kp, 3)
                T = gt.shape[0]

            for start in range(T - self.time_window + 1):
                self._windows.append(_WindowEntry(rec_idx=rec_idx, start=start))

        if not self._windows:
            raise RuntimeError(
                f"No valid windows found under {self._root}. "
                "Check that .npz and .npy files exist and time_window is appropriate."
            )
        logger.info(
            "Indexed %d windows from %d recordings", len(self._windows), len(self._recordings)
        )

    def _load_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a windowed (radar, pose) sample.

        Reads the per-frame ``preprocessed/`` files when present, otherwise
        slices the original ``.npz`` / ``.npy`` recording.

        Args:
            idx: Window index.

        Returns:
            Complex radar window ``(T, D, A, E, R)`` and pose ``(J, 3)``
            (``target_frame="last"``) or ``(T, J, 3)`` (``"all"``).
        """
        entry = self._windows[idx]
        rec = self._recordings[entry.rec_idx]
        end = entry.start + self.time_window

        prep_dir = self._root / "preprocessed" / rec.stem

        if prep_dir.exists():
            frames = [
                torch.from_numpy(np.load(prep_dir / "radar" / f"{t}.npy"))
                for t in range(entry.start, end)
            ]
            X_t = torch.stack(frames)
        else:
            radar_data = np.load(rec.radar_path)
            radar_arr = radar_data[radar_data.files[0]]
            X = radar_arr[entry.start : end]
            # On-disk layout (T, Az, El, Range, Doppler) -> (T, Doppler, Az, El, Range).
            X_t = torch.from_numpy(X.copy()).to(torch.complex64)
            X_t = X_t.permute(0, 4, 1, 2, 3)

        skel_dir = prep_dir / "skeleton"
        if skel_dir.exists():
            if self.target_frame == "last":
                Y_t = torch.from_numpy(np.load(skel_dir / f"{end - 1}.npy"))
            else:
                Y_t = torch.stack([
                    torch.from_numpy(np.load(skel_dir / f"{t}.npy"))
                    for t in range(entry.start, end)
                ])
        else:
            gt = self._get_gt(entry.rec_idx)
            if self.target_frame == "last":
                Y = gt[end - 1]
            else:
                Y = gt[entry.start : end]
            Y_t = torch.from_numpy(Y.copy()).float()

        Y_t = Y_t * self.unit_scale
        if self.center_to_root:
            root = Y_t[..., 0:1, :]
            Y_t = Y_t - root

        return X_t, Y_t

    def _get_meta(self, idx: int) -> SampleMeta:
        """Return metadata for a sample.

        Args:
            idx: Window index.

        Returns:
            Participant, angle, activity and recording of the window.
        """
        entry = self._windows[idx]
        rec = self._recordings[entry.rec_idx]
        return SampleMeta(
            participant=rec.participant,
            angle=rec.angle,
            activity=rec.action,
            recording=rec.recording,
        )

    def __len__(self) -> int:
        """Return the number of temporal windows."""
        return len(self._windows)

    def _get_gt(self, rec_idx: int) -> np.ndarray:
        """Get the ground truth of a recording from the cache or disk.

        Args:
            rec_idx: Recording index.

        Returns:
            Keypoints ``(T, J, 3)``.
        """
        if rec_idx in self._gt_cache:
            return self._gt_cache[rec_idx]
        gt = np.load(self._recordings[rec_idx].gt_path)
        if gt.ndim == 2 and gt.shape[1] != 3:
            gt = gt.reshape(gt.shape[0], gt.shape[1] // 3, 3)
        return gt

    def _preload_ground_truth(self) -> None:
        """Cache all GT arrays in RAM."""
        for i, rec in enumerate(self._recordings):
            gt = np.load(rec.gt_path)
            if gt.ndim == 2 and gt.shape[1] != 3:
                gt = gt.reshape(gt.shape[0], gt.shape[1] // 3, 3)
            self._gt_cache[i] = gt
        logger.info("Preloaded GT for %d recordings", len(self._recordings))


class _RecordingMeta:
    """Internal bookkeeping for one radar/GT file pair."""

    __slots__ = ("participant", "angle", "action", "recording", "radar_path", "gt_path", "stem")

    def __init__(
        self,
        participant: str,
        angle: str,
        action: str,
        recording: str,
        radar_path: Path,
        gt_path: Path,
        stem: str,
    ) -> None:
        self.participant = participant
        self.angle = angle
        self.action = action
        self.recording = recording
        self.radar_path = radar_path
        self.gt_path = gt_path
        self.stem = stem


class _WindowEntry:
    """Lightweight index entry for one temporal window."""

    __slots__ = ("rec_idx", "start")

    def __init__(self, rec_idx: int, start: int) -> None:
        self.rec_idx = rec_idx
        self.start = start


def _discover_recordings(
    root: Path,
    participants: set[str] | None = None,
    angles: set[str] | None = None,
) -> list[_RecordingMeta]:
    """Find all data_cube_parsed_*.npz / skeleton_*.npy pairs.

    Supports two directory layouts: ``root/radar/*.npz`` with
    ``root/skeletons/*.npy``, or both flat in ``root``.

    Args:
        root: Directory to search.
        participants: Participant filter.
        angles: Angle filter.

    Returns:
        Sorted list of recording metadata.

    Raises:
        RuntimeError: If no recording is found.
    """
    radar_dir = root / "radar"
    skeleton_dir = root / "skeletons"
    if radar_dir.is_dir():
        search_dir = radar_dir
    else:
        search_dir = root
        skeleton_dir = root

    recordings: list[_RecordingMeta] = []

    for npz in sorted(search_dir.glob("data_cube_parsed_*.npz")):
        stem = npz.stem.replace("data_cube_parsed_", "")
        parts = stem.split("_")
        if len(parts) < 4:
            logger.warning("Cannot parse filename %s — skipping", npz.name)
            continue

        participant, angle, action, recording = (
            parts[0],
            parts[1],
            parts[2],
            parts[3],
        )

        if participants and participant not in participants:
            continue
        if angles and angle not in angles:
            continue

        gt_path = skeleton_dir / f"skeleton_{stem}.npy"
        if not gt_path.exists():
            logger.warning("Missing GT file %s — skipping", gt_path.name)
            continue

        recordings.append(
            _RecordingMeta(
                participant=participant,
                angle=angle,
                action=action,
                recording=recording,
                radar_path=npz,
                gt_path=gt_path,
                stem=stem,
            )
        )

    if not recordings:
        raise RuntimeError(
            f"No recordings found under {root}. "
            "Expected files: data_cube_parsed_<p>_<ang>_<act>_<rec>.npz"
        )

    logger.info("Discovered %d recordings under %s", len(recordings), root)
    return recordings
