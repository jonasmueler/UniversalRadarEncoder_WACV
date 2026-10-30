"""Build the mmRadPose sequence caches from frame embeddings."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from frozenradar.core.mmradpose import MmRadPoseRawADC
from frozenradar.data.splits import MMRADPOSE_PARTICIPANTS

logger = logging.getLogger(__name__)

ACTIVITIES = [f"ac{i}" for i in range(12)]
ACTIVITY_TO_IDX = {a: i for i, a in enumerate(ACTIVITIES)}
ACTIVITY_NAMES = [
    "T-pose",
    "Left upper limb ext.",
    "Right upper limb ext.",
    "Bilateral upper limb ext.",
    "Bicep curls",
    "Front arm rotation",
    "Torso forward bending",
    "Left front lunge",
    "Right front lunge",
    "Squats",
    "Side lower limb ext.",
    "Front lower limb ext.",
]


@dataclass(frozen=True)
class RecordingMeta:
    """Metadata for one reconstructed recording sequence.

    Args:
        activity: Activity label (e.g. ``"ac0"``).
        activity_idx: Integer class index (0-11).
        participant: Participant ID (e.g. ``"p1"``).
        angle: Angle ID (e.g. ``"an0"``).
        recording: Recording ID (e.g. ``"r0"``).
        n_frames: Number of frames in the sequence.
    """

    activity: str
    activity_idx: int
    participant: str
    angle: str
    recording: str
    n_frames: int


def build_sequences(
    embedded_dir: Path,
    data_root: Path,
    output_dir: Path,
    participants: list[str],
) -> list[RecordingMeta]:
    """Reconstruct per-recording sequences from batch embedding files.

    Loads all batch files into memory and slices them into per-recording
    sequences using the dataset's internal index order.

    Args:
        embedded_dir: Directory containing ``batch_NNNN.pt`` files.
        data_root: Root of mmRadPose raw ADC data (for metadata).
        output_dir: Where to save per-recording ``.pt`` files and metadata.
        participants: Participant IDs matching the embedding extraction split.

    Returns:
        List of ``RecordingMeta`` for each saved sequence.

    Raises:
        FileNotFoundError: If there are no batch files.
        ValueError: If there are fewer embeddings than dataset frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = MmRadPoseRawADC(
        root=data_root,
        participants=participants,
        time_window=1,
        preload_gt=False,
        return_meta=False,
    )

    # Numeric sort: a lexical sort misorders the files once the index has 7 digits.
    batch_files = sorted(
        embedded_dir.glob("batch_*.pt"), key=lambda p: int(p.stem.split("_")[1]),
    )
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {embedded_dir}")

    logger.info("Found %d batch files in %s", len(batch_files), embedded_dir)

    all_mu: list[torch.Tensor] = []
    all_logvar: list[torch.Tensor] = []
    for bf in batch_files:
        data = torch.load(bf, weights_only=True)
        all_mu.append(data["mu"])
        all_logvar.append(data["logvar"])
    all_mu_cat = torch.cat(all_mu)
    all_logvar_cat = torch.cat(all_logvar)
    logger.info(
        "Loaded %d embeddings (dataset size: %d)",
        all_mu_cat.shape[0], len(ds),
    )
    if all_mu_cat.shape[0] < len(ds):
        raise ValueError(
            f"Not enough embeddings ({all_mu_cat.shape[0]}) "
            f"for dataset size ({len(ds)})"
        )

    # Group consecutive frame indices by recording.
    recordings: list[tuple[int, int, RecordingMeta]] = []
    current_rec_idx = -1
    start = 0

    for i in range(len(ds)):
        entry = ds._windows[i]
        if entry.rec_idx != current_rec_idx:
            if current_rec_idx >= 0:
                rec = ds._recordings[current_rec_idx]
                n = i - start
                recordings.append((start, i, RecordingMeta(
                    activity=rec.action,
                    activity_idx=ACTIVITY_TO_IDX[rec.action],
                    participant=rec.participant,
                    angle=rec.angle,
                    recording=rec.recording,
                    n_frames=n,
                )))
            current_rec_idx = entry.rec_idx
            start = i

    rec = ds._recordings[current_rec_idx]
    n = len(ds) - start
    recordings.append((start, len(ds), RecordingMeta(
        activity=rec.action,
        activity_idx=ACTIVITY_TO_IDX[rec.action],
        participant=rec.participant,
        angle=rec.angle,
        recording=rec.recording,
        n_frames=n,
    )))

    meta_list: list[RecordingMeta] = []

    for idx, (s, e, meta) in enumerate(recordings):
        torch.save(
            {"mu": all_mu_cat[s:e].clone(), "logvar": all_logvar_cat[s:e].clone()},
            output_dir / f"rec_{idx:04d}.pt",
        )
        meta_list.append(meta)

    torch.save([asdict(m) for m in meta_list], output_dir / "meta.pt")
    logger.info(
        "Saved %d recording sequences to %s (activities: %s)",
        len(meta_list),
        output_dir,
        sorted({m.activity for m in meta_list}),
    )
    return meta_list


def main() -> None:
    """Build recording sequences for the train, val, and test splits."""
    parser = argparse.ArgumentParser(
        description="Reconstruct per-recording sequences from VAE batch embeddings.",
    )
    parser.add_argument(
        "--embedded-dir", type=str, required=True,
        help="Root directory containing train/ val/ test/ batch files.",
    )
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="Path to mmRadPose raw ADC data.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for sequences.",
    )
    # Defaults: the participant split of data/splits.py.
    parser.add_argument(
        "--participants-train", nargs="+", default=MMRADPOSE_PARTICIPANTS["train"],
    )
    parser.add_argument(
        "--participants-val", nargs="+", default=MMRADPOSE_PARTICIPANTS["val"],
    )
    parser.add_argument(
        "--participants-test", nargs="+", default=MMRADPOSE_PARTICIPANTS["test"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    embedded_dir = Path(args.embedded_dir)
    output_dir = Path(args.output_dir)

    for split, participants in [
        ("train", args.participants_train),
        ("val", args.participants_val),
        ("test", args.participants_test),
    ]:
        if not (embedded_dir / split).exists():
            logger.warning("Skipping %s split (no %s/ dir)", split, split)
            continue
        logger.info("Building %s sequences...", split)
        build_sequences(
            embedded_dir=embedded_dir / split,
            data_root=Path(args.data_root),
            output_dir=output_dir / split,
            participants=participants,
        )


if __name__ == "__main__":
    main()
