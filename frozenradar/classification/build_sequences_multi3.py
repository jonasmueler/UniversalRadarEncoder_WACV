"""Build the CI4R-Gait sequence caches from frame embeddings."""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from frozenradar.classification.build_sequences import RecordingMeta
from frozenradar.data.multi3_dataset import (
    ACTIVITY_NAMES,
    activity_from_key,
)

logger = logging.getLogger(__name__)

N_CLASSES = len(ACTIVITY_NAMES)
# Recording file names start with <2 digits><2 digits><4-digit subject code>_.
_SUBJECT_RE = re.compile(r"\d{2}\d{2}(\d{4})_")


def _load_embeddings(embedded_dir: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Load concatenated embeddings and per-recording frame counts.

    Args:
        embedded_dir: Directory with ``batch_NNNN.pt`` + ``index.pt``.

    Returns:
        Tuple of (mu, logvar, frame_counts) in extraction order.

    Raises:
        FileNotFoundError: If there are no batch files.
        ValueError: If the embedding count does not match the index.
    """
    batch_files = sorted(
        embedded_dir.glob("batch_*.pt"), key=lambda p: int(p.stem.split("_")[1]),
    )
    if not batch_files:
        raise FileNotFoundError(f"No batch files in {embedded_dir}")
    mu = torch.cat([torch.load(f, weights_only=True)["mu"] for f in batch_files])
    logvar = torch.cat([torch.load(f, weights_only=True)["logvar"] for f in batch_files])
    frame_counts: dict[str, int] = torch.load(
        embedded_dir / "index.pt", weights_only=True,
    )["frame_counts"]
    if mu.shape[0] != sum(frame_counts.values()):
        raise ValueError(
            f"{embedded_dir}: {mu.shape[0]} embeddings vs "
            f"{sum(frame_counts.values())} expected"
        )
    return mu, logvar, frame_counts


def _subject_of(key: str) -> str:
    """Extract the subject code from a recording key's file name.

    Args:
        key: Recording key.

    Returns:
        The 4-digit subject code, or ``"unknown"``.
    """
    match = _SUBJECT_RE.match(Path(key).name)
    return match.group(1) if match else "unknown"


def _shuffle_split(
    n: int, test_size: float, val_size: float, seed: int,
) -> dict[int, str]:
    """Assign each recording index to train/val/test by a shuffled split.

    Replicates the release's protocol: shuffle all recordings, then take an
    80/20 train/test split (``create_dataset.ipynb`` with ``shuffle_data=True``).
    A ``val_size`` slice is carved from the training portion for checkpoint
    selection.

    Args:
        n: Number of recordings.
        test_size: Test fraction of recordings (0.2 = benchmark).
        val_size: Validation fraction, carved from the training portion.
        seed: Shuffle seed.

    Returns:
        Mapping recording index -> split name.
    """
    perm = np.random.RandomState(seed).permutation(n)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    assign: dict[int, str] = {}
    for j, i in enumerate(perm):
        assign[int(i)] = (
            "test" if j < n_test else "val" if j < n_test + n_val else "train"
        )
    return assign


def build_sequences_multi3(
    embedded_dir: Path,
    output_dir: Path,
    max_frames: int = 512,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 1,
) -> None:
    """Build the CI4R-Gait train/val/test sequence caches.

    Each labelled recording is one sample: its frame sequence (one performed
    activity, capped at ``max_frames``) is classified into 11 activities.
    Recordings are split with :func:`_shuffle_split`, so no recording straddles
    two splits.

    Args:
        embedded_dir: Directory of extracted embeddings (``batch_*.pt`` + ``index.pt``).
        output_dir: Sequence-cache root; ``train``/``val``/``test`` subdirs are written.
        max_frames: Cap on frames per recording (fits the positional embedding).
        test_size: Test fraction of the recordings.
        val_size: Validation fraction of the recordings.
        seed: Split seed.
    """
    mu, logvar, frame_counts = _load_embeddings(embedded_dir)

    # Recording slices in extraction order (sorted keys, cumulative offset).
    records: list[tuple[str, int, torch.Tensor, torch.Tensor]] = []
    offset = 0
    n_unlabeled = 0
    for key in sorted(frame_counts):
        n = frame_counts[key]
        lab = activity_from_key(key)
        if lab is not None:
            end = offset + min(n, max_frames)
            records.append((key, lab, mu[offset:end].clone(), logvar[offset:end].clone()))
        else:
            n_unlabeled += 1
        offset += n

    assign = _shuffle_split(len(records), test_size, val_size, seed)
    logger.info(
        "CI4R-Gait: %d labelled recordings (%d unlabeled skipped); "
        "train=%d val=%d test=%d",
        len(records), n_unlabeled,
        sum(v == "train" for v in assign.values()),
        sum(v == "val" for v in assign.values()),
        sum(v == "test" for v in assign.values()),
    )

    rec_idx = {"train": 0, "val": 0, "test": 0}
    metas: dict[str, list[RecordingMeta]] = {"train": [], "val": [], "test": []}
    for name in ("train", "val", "test"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    for i, (key, lab, mu_s, logvar_s) in enumerate(records):
        split = assign[i]
        torch.save(
            {"mu": mu_s, "logvar": logvar_s},
            output_dir / split / f"rec_{rec_idx[split]:04d}.pt",
        )
        metas[split].append(
            RecordingMeta(
                activity=ACTIVITY_NAMES[lab],
                activity_idx=lab,
                participant=_subject_of(key),
                angle="Front",
                recording=Path(key).name,
                n_frames=mu_s.shape[0],
            )
        )
        rec_idx[split] += 1

    for name in ("train", "val", "test"):
        torch.save([asdict(m) for m in metas[name]], output_dir / name / "meta.pt")
        per_class: dict[int, int] = defaultdict(int)
        for m in metas[name]:
            per_class[m.activity_idx] += 1
        logger.info(
            "%s: %d recordings; classes present %d/%d",
            name, rec_idx[name], len(per_class), N_CLASSES,
        )


def main() -> None:
    """Build the CI4R-Gait sequence caches from extracted embeddings."""
    parser = argparse.ArgumentParser(
        description="Build the CI4R-Gait train/val/test sequence caches.",
    )
    parser.add_argument("--embedded-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    build_sequences_multi3(
        embedded_dir=args.embedded_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
