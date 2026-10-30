"""Convert mmRadPose recordings to per-frame ``.npy`` files."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _process_recording(
    npz_path: Path,
    skeleton_dir: Path,
    prep_dir: Path,
    index: int,
    total: int,
) -> str:
    """Convert one recording from .npz/.npy to per-frame .npy files.

    Existing complete output for the recording is kept; incomplete output
    is deleted and rewritten.

    Args:
        npz_path: Path to the radar .npz file.
        skeleton_dir: Directory containing skeleton .npy files.
        prep_dir: Output directory for preprocessed data.
        index: 1-based index for progress logging.
        total: Total number of recordings.

    Returns:
        Status message string.

    Raises:
        RuntimeError: If radar and skeleton have different frame counts.
    """
    stem = npz_path.stem.replace("data_cube_parsed_", "")
    rec_dir = prep_dir / stem

    if rec_dir.exists():
        radar_out = rec_dir / "radar"
        skel_out = rec_dir / "skeleton"
        if radar_out.exists() and skel_out.exists():
            n_radar = len(list(radar_out.glob("*.npy")))
            n_skel = len(list(skel_out.glob("*.npy")))
            if n_radar > 0 and n_radar == n_skel:
                return f"[{index}/{total}] {stem} — skipped (exists, {n_radar} frames)"
        shutil.rmtree(rec_dir)

    radar_data = np.load(npz_path)
    radar_arr = radar_data[radar_data.files[0]].astype(np.complex64)
    # On-disk layout (T, Az, El, Range, Doppler) -> (T, Doppler, Az, El, Range).
    radar_arr = np.transpose(radar_arr, (0, 4, 1, 2, 3))
    T_radar = radar_arr.shape[0]

    gt_path = skeleton_dir / f"skeleton_{stem}.npy"
    if not gt_path.exists():
        raise RuntimeError(f"Missing skeleton file: {gt_path}")

    gt = np.load(gt_path)
    if gt.ndim == 2 and gt.shape[1] != 3:
        gt = gt.reshape(gt.shape[0], gt.shape[1] // 3, 3)
    gt = gt.astype(np.float32)
    T_skel = gt.shape[0]

    if T_radar != T_skel:
        raise RuntimeError(
            f"{stem}: radar has {T_radar} frames but skeleton has {T_skel}"
        )

    radar_out = rec_dir / "radar"
    radar_out.mkdir(parents=True)
    for t in range(T_radar):
        np.save(radar_out / f"{t}.npy", radar_arr[t])

    skel_out = rec_dir / "skeleton"
    skel_out.mkdir()
    for t in range(T_skel):
        np.save(skel_out / f"{t}.npy", gt[t])

    return f"[{index}/{total}] {stem} — {T_radar} frames"


def preprocess_dataset(root: str | Path, workers: int | None = None) -> None:
    """Convert mmRadPose .npz/.npy recordings to per-frame .npy files.

    For each recording, creates a subdirectory under ``root/preprocessed/``
    containing individual radar and skeleton frames as NumPy arrays.
    Radar frames are stored as complex64 with axes permuted from the on-disk
    layout ``(Az, El, Range, Doppler)`` to ``(Doppler, Az, El, Range)``.
    Skeleton frames are stored as float32 with shape ``(K, 3)``.

    Existing output directories are skipped (idempotent).

    Args:
        root: Path to the mmRadPose dataset root directory.
        workers: Number of parallel workers.  Defaults to CPU count.

    Raises:
        RuntimeError: If no ``data_cube_parsed_*.npz`` file is found.
    """
    root = Path(root).expanduser().resolve()

    radar_dir = root / "radar"
    skeleton_dir = root / "skeletons"
    if not radar_dir.is_dir():
        radar_dir = root
        skeleton_dir = root

    prep_dir = root / "preprocessed"
    prep_dir.mkdir(exist_ok=True)

    npz_files = sorted(radar_dir.glob("data_cube_parsed_*.npz"))
    if not npz_files:
        raise RuntimeError(f"No data_cube_parsed_*.npz files found under {radar_dir}")

    if workers is None:
        workers = os.cpu_count() or 4

    total = len(npz_files)
    logger.info("Preprocessing %d recordings with %d workers", total, workers)

    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_recording, npz_path, skeleton_dir, prep_dir, i + 1, total
            ): npz_path
            for i, npz_path in enumerate(npz_files)
        }
        for future in as_completed(futures):
            try:
                logger.info("%s", future.result())
            except RuntimeError as exc:
                errors.append(str(exc))
                logger.error("%s", exc)

    logger.info("Verifying %d recordings", total)
    mismatches = 0
    for npz_path in npz_files:
        stem = npz_path.stem.replace("data_cube_parsed_", "")
        rec_dir = prep_dir / stem
        radar_dir_ = rec_dir / "radar"
        skel_dir_ = rec_dir / "skeleton"
        if not radar_dir_.exists() or not skel_dir_.exists():
            logger.warning("Missing: %s", stem)
            mismatches += 1
            continue
        n_r = len(list(radar_dir_.glob("*.npy")))
        n_s = len(list(skel_dir_.glob("*.npy")))
        if n_r != n_s:
            logger.warning("Mismatch: %s radar=%d skeleton=%d", stem, n_r, n_s)
            mismatches += 1

    if errors:
        logger.error("%d recording(s) failed during processing", len(errors))
    if mismatches:
        logger.error("%d recording(s) have missing or mismatched frames", mismatches)
    else:
        logger.info("All recordings verified: radar and skeleton frame counts match")

    logger.info("Done. Output: %s", prep_dir)


def main() -> None:
    """Parse command-line arguments and preprocess the dataset."""
    parser = argparse.ArgumentParser(
        description="Preprocess mmRadPose dataset to per-frame .npy files.",
    )
    parser.add_argument("--root", type=str, required=True, help="Path to mmRadPose dataset root.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers.")
    args = parser.parse_args()

    preprocess_dataset(args.root, workers=args.workers)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
