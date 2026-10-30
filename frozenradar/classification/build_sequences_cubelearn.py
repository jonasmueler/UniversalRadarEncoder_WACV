"""Build the CubeLearn sequence caches from frame embeddings."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import torch

from frozenradar.classification.build_sequences import RecordingMeta
from frozenradar.data.splits import (
    CUBELEARN_OUTOFSET_USERS,
    CUBELEARN_USERS,
    cubelearn_sample_ids,
)

logger = logging.getLogger(__name__)


def build_sequences_cubelearn(
    embedded_dir: Path,
    output_dir: Path,
    n_gestures: int = 6,
    n_samples: int = 30,
    frames_per_recording: int = 20,
    split_dir: str | Path | None = None,
    split_mode: str = "inset",
) -> None:
    """Group CubeLearn frame embeddings into per-recording sequences.

    ``inset`` uses CubeLearn's sample-index split (users 0-5, disjoint
    train/val/test sample indices), the same split the encoder was pretrained
    on. ``outofset`` regroups the held-out users 6-7 and routes all samples to
    ``test/``. The extraction iterated user -> gesture -> sample -> frame, which
    this function reproduces to regroup the embeddings.

    Args:
        embedded_dir: Directory containing ``batch_NNNN.pt`` files.
        output_dir: Where to save per-split sequence directories.
        n_gestures: Number of classes (6 for HAR, 12 for HGR/AGR).
        n_samples: Samples per (user, gesture) in the extraction.
        frames_per_recording: Frames per recording (10 or 20).
        split_dir: Directory of the official split CSVs (``None`` = repo copy).
        split_mode: ``"inset"`` (official sample split) or ``"outofset"``
            (users 6-7, everything to ``test/``).

    Raises:
        FileNotFoundError: If no batch files are present.
        ValueError: On an unknown split mode or an embedding count that does
            not match the expected total.
    """
    if split_mode == "inset":
        sample_split = cubelearn_sample_ids(split_dir)
        users = CUBELEARN_USERS
    elif split_mode == "outofset":
        sample_split = {"test": list(range(n_samples))}
        users = CUBELEARN_OUTOFSET_USERS
    else:
        raise ValueError(f"Unknown split_mode: {split_mode!r}")
    sample_to_split = {s: name for name, ids in sample_split.items() for s in ids}
    n_users = len(users)

    # Numeric sort: a lexical sort misorders the files once the index has 7 digits.
    batch_files = sorted(
        embedded_dir.glob("batch_*.pt"), key=lambda p: int(p.stem.split("_")[1]),
    )
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {embedded_dir}")
    logger.info("Loading %d batch files from %s", len(batch_files), embedded_dir)

    all_mu = torch.cat([torch.load(bf, weights_only=True)["mu"] for bf in batch_files])
    all_logvar = torch.cat([torch.load(bf, weights_only=True)["logvar"] for bf in batch_files])

    expected = n_users * n_gestures * n_samples * frames_per_recording
    if all_mu.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} frames "
            f"({n_users}u * {n_gestures}g * {n_samples}s * {frames_per_recording}f), "
            f"got {all_mu.shape[0]}"
        )
    logger.info("Total embeddings: %d (latent_dim=%d)", all_mu.shape[0], all_mu.shape[1])

    # Regroup in extraction order and route each recording to its split by sample.
    split_recordings: dict[str, list] = {name: [] for name in sample_split}
    idx = 0
    for user in users:
        for gesture in range(n_gestures):
            for sample in range(n_samples):
                mu_seq = all_mu[idx : idx + frames_per_recording]
                logvar_seq = all_logvar[idx : idx + frames_per_recording]
                idx += frames_per_recording
                split = sample_to_split.get(sample)
                if split is not None:
                    split_recordings[split].append((user, gesture, sample, mu_seq, logvar_seq))

    for split_name, recs in split_recordings.items():
        split_out = output_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        meta_list: list[RecordingMeta] = []
        for rec_idx, (user, gesture, sample, mu_seq, logvar_seq) in enumerate(recs):
            torch.save(
                {"mu": mu_seq.clone(), "logvar": logvar_seq.clone()},
                split_out / f"rec_{rec_idx:04d}.pt",
            )
            meta_list.append(
                RecordingMeta(
                    activity=f"g{gesture}",
                    activity_idx=gesture,
                    participant=f"u{user}",
                    angle="a0",
                    recording=f"s{sample}",
                    n_frames=frames_per_recording,
                )
            )
        torch.save([asdict(m) for m in meta_list], split_out / "meta.pt")
        logger.info("Saved %d %s recordings to %s", len(meta_list), split_name, split_out)


def main() -> None:
    """Build CubeLearn recording sequences from batch embeddings."""
    parser = argparse.ArgumentParser(
        description="Group CubeLearn frame embeddings into per-recording sequences.",
    )
    parser.add_argument("--embedded-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--n-gestures", type=int, default=6,
        help="Number of classes (6 for HAR, 12 for HGR/AGR).",
    )
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument(
        "--frames-per-recording", type=int, default=20,
        help="Frames per recording (10 for HGR/AGR, 20 for HAR).",
    )
    parser.add_argument(
        "--split-dir", type=str, default=None,
        help="Directory with the official split CSVs (default: repo copy).",
    )
    parser.add_argument(
        "--split-mode", type=str, default="inset", choices=["inset", "outofset"],
        help="inset: users 0-5, official sample split; outofset: users 6-7, all to test/.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    build_sequences_cubelearn(
        embedded_dir=Path(args.embedded_dir),
        output_dir=Path(args.output_dir),
        n_gestures=args.n_gestures,
        n_samples=args.n_samples,
        frames_per_recording=args.frames_per_recording,
        split_dir=args.split_dir,
        split_mode=args.split_mode,
    )


if __name__ == "__main__":
    main()
