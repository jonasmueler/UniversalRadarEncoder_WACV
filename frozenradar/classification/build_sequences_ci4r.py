"""Build the CI4R-ASL sequence caches from per-subject frame embeddings."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from frozenradar.classification.build_sequences import RecordingMeta

logger = logging.getLogger(__name__)

# Integer label -> class name (dataset README): 3 gross activities + 15 ASL signs.
# Label 0 ("No motion") separates motion intervals and is not a class. Label 19
# ("TEACH") never occurs in the label files, so the labels are 1..18. Labels 9
# and 10 are both "HOT" in the README and are kept distinct as in the label files.
CI4R_CLASS_NAMES = [
    "walking",       # 1
    "sitting",       # 2
    "standing up",   # 3
    "TIRED",         # 4
    "BOOK",          # 5
    "SLEEP",         # 6
    "EVENING",       # 7
    "READY",         # 8
    "HOT",           # 9
    "HOT(2)",        # 10
    "MONTH",         # 11
    "COOK",          # 12
    "AGAIN",         # 13
    "SUMMON",        # 14
    "MAYBE",         # 15
    "NIGHT",         # 16
    "SOMETHING",     # 17
    "TEACHER",       # 18
]
N_CLASSES = len(CI4R_CLASS_NAMES)  # activity_idx = label - 1
# Repetitions per recording; repetition r covers the r-th fifth of the frames.
N_REPS = 5

# Subject directories in canonical order. The split is assigned positionally over
# the units enumerated in this order, so the order is part of the split
# definition. "6_continued" has raw recordings but no label files (0 units).
CANONICAL_SUBJECTS = [
    "6",
    "7",
    "8",
    "9",
    "12",
    "14-15",
    "1",
    "3",
    "17",
    "5",
    "16",
    "2",
    "4",
    "12_continued",
    "16_continued",
    "5_continued",
    "6-10-11-12-13",
    "6-7",
    "6-7-8",
    "6-7-8-9",
    "6_continued",
]

# Hearing participants: the canonical order without the four deaf/ASL users. Used
# as a filter over the canonical assignment, never to redraw the split.
HEARING_SUBJECTS = [s for s in CANONICAL_SUBJECTS if s not in ("1", "3", "2", "4")]


def _load_subject(embedded_dir: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Load a subject's concatenated embeddings and recording frame counts.

    Args:
        embedded_dir: Directory with ``batch_NNNN.pt`` + ``index.pt``.

    Returns:
        Tuple of (mu, logvar, frame_counts) with frames in extraction order.

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
    logvar = torch.cat(
        [torch.load(f, weights_only=True)["logvar"] for f in batch_files]
    )
    meta = torch.load(embedded_dir / "index.pt", weights_only=True)
    frame_counts: dict[str, int] = meta["frame_counts"]
    if mu.shape[0] != sum(frame_counts.values()):
        raise ValueError(
            f"{embedded_dir}: {mu.shape[0]} embeddings but index expects "
            f"{sum(frame_counts.values())}"
        )
    return mu, logvar, frame_counts


def _find_label_file(labels_root: Path, subject: str, base: str, rep: int) -> Path | None:
    """Locate the per-frame rangeDoppler label file for one repetition.

    Args:
        labels_root: Root of the unzipped ``ASL Sequential Dataset`` tree.
        subject: Subject identifier (e.g. ``"5"`` or the combined ``"14-15"``).
        base: Recording base name, e.g. ``10010000_1617641906``.
        rep: Repetition number (1-5).

    Returns:
        The label path, or ``None`` if this repetition is unlabeled.
    """
    matches = list(
        labels_root.glob(
            f"**/subject {subject}/labels/rangeDoppler/{base}_{rep}.txt"
        )
    )
    return matches[0] if matches else None


def _enumerate_units(
    embedded_root: Path, labels_root: Path, subjects: list[str],
) -> list[dict]:
    """List every labelled repetition unit across all subjects.

    A unit is one repetition of one recording, the split element of the
    release's per-repetition split. Only ``index.pt`` and the label files are
    read, so the split is decided before any embedding tensor is loaded.

    Args:
        embedded_root: Directory containing ``subject_<n>/`` embedding dirs.
        labels_root: Unzipped ``ASL Sequential Dataset`` directory.
        subjects: Subject identifiers to pool.

    Returns:
        List of unit dicts with subject, base, rep, subject-relative frame
        start, usable length and the integer label sequence.
    """
    units: list[dict] = []
    for subject in subjects:
        index = torch.load(
            embedded_root / f"subject_{subject}" / "index.pt", weights_only=True,
        )
        frame_counts: dict[str, int] = index["frame_counts"]
        offset = 0
        for key in sorted(frame_counts):
            n_frames = frame_counts[key]
            base = Path(key).name
            rep_len = n_frames // N_REPS
            for rep in range(N_REPS):
                label_path = _find_label_file(labels_root, subject, base, rep + 1)
                if label_path is None:
                    continue
                labels = [int(v) for v in label_path.read_text().split()]
                units.append({
                    "subject": subject,
                    "base": base,
                    "rep": rep + 1,
                    "rep_start": offset + rep * rep_len,
                    "usable": min(len(labels), rep_len),
                    "labels": labels,
                })
            offset += n_frames
    return units


def assign_unit_splits(
    n_units: int, test_size: float, val_size: float, seed: int,
) -> dict[int, str]:
    """Assign repetition-unit indices to train/val/test deterministically.

    One shuffled permutation with a fixed seed, then test / val / train
    slices (the release's 80/20 random split plus a val slice). Shared by the
    sequence builder and the split manifests.

    Args:
        n_units: Number of repetition units (in canonical enumeration order).
        test_size: Test fraction.
        val_size: Validation fraction, carved from the training portion.
        seed: Split random state.

    Returns:
        Mapping unit index -> split name.
    """
    perm = np.random.RandomState(seed).permutation(n_units)
    n_test = int(round(n_units * test_size))
    n_val = int(round(n_units * val_size))
    assign = {int(i): "test" for i in perm[:n_test]}
    assign.update({int(i): "val" for i in perm[n_test : n_test + n_val]})
    assign.update({int(i): "train" for i in perm[n_test + n_val :]})
    return assign


def _motion_segments(labels: list[int], usable: int) -> list[tuple[int, int, int]]:
    """Split a per-frame label sequence into contiguous motion segments.

    A segment is a maximal run of one identical non-zero label, the release's
    ground-truth motion interval (one performed activity or sign). Runs of
    label 0 ("no motion") separate segments and are discarded.

    Args:
        labels: Per-frame integer labels for one repetition.
        usable: Number of leading frames that align with embeddings.

    Returns:
        List of ``(label, start, end)`` half-open intervals, non-zero label only.
    """
    segments: list[tuple[int, int, int]] = []
    i = 0
    n = min(len(labels), usable)
    while i < n:
        lab = labels[i]
        if lab == 0:
            i += 1
            continue
        j = i
        while j < n and labels[j] == lab:
            j += 1
        segments.append((lab, i, j))
        i = j
    return segments


def _carve_unit(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    unit: dict,
    split_out: Path,
    rec_idx: int,
    min_seg_len: int,
    max_seg_len: int,
) -> tuple[list[RecordingMeta], int, int]:
    """Emit one sample per ground-truth motion segment of a repetition.

    Each contiguous non-zero-label run becomes one variable-length sample that
    the temporal head classifies as a whole. Segments shorter than
    ``min_seg_len`` are dropped; longer ones are truncated to ``max_seg_len``
    to fit the positional embedding.

    Args:
        mu: The unit's subject concatenated means ``(T, D)``.
        logvar: The unit's subject concatenated log-variances ``(T, D)``.
        unit: A unit dict from :func:`_enumerate_units`.
        split_out: Output directory for this split.
        rec_idx: Next free ``rec_NNNN`` index in this split.
        min_seg_len: Minimum segment length (frames) to keep.
        max_seg_len: Maximum segment length (frames); longer are truncated.

    Returns:
        Tuple of (new metas, updated rec_idx, dropped count).
    """
    metas: list[RecordingMeta] = []
    dropped = 0
    rep_start = unit["rep_start"]
    for seg_k, (lab, s, e) in enumerate(_motion_segments(unit["labels"], unit["usable"])):
        if e - s < min_seg_len:
            dropped += 1
            continue
        e = min(e, s + max_seg_len)
        frame_lo, frame_hi = rep_start + s, rep_start + e
        torch.save(
            {
                "mu": mu[frame_lo:frame_hi].clone(),
                "logvar": logvar[frame_lo:frame_hi].clone(),
            },
            split_out / f"rec_{rec_idx:04d}.pt",
        )
        metas.append(
            RecordingMeta(
                activity=CI4R_CLASS_NAMES[lab - 1],
                activity_idx=lab - 1,
                participant=f"s{unit['subject']}",
                angle="Front",
                recording=f"s{unit['subject']}_{unit['base']}_rep{unit['rep']}_seg{seg_k}",
                n_frames=e - s,
            )
        )
        rec_idx += 1
    return metas, rec_idx, dropped


def build_sequences_ci4r_mixed(
    embedded_root: Path,
    labels_root: Path,
    output_dir: Path,
    subjects: list[str],
    min_seg_len: int = 5,
    max_seg_len: int = 512,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 1,
    subject_filter: list[str] | None = None,
) -> None:
    """Build the CI4R-ASL train/val/test sequence caches.

    Replicates the release's protocol: one random split of the repetition files,
    ``test_size=0.2`` at ``random_state=1``, pooling every subject. Each
    ground-truth motion segment becomes one sample classified into 18 classes.
    A ``val_size`` slice is carved from the training units for checkpoint
    selection. All segments of a repetition land in the same split.

    Args:
        embedded_root: Directory containing ``subject_<n>/`` embedding dirs.
        labels_root: Unzipped ``ASL Sequential Dataset`` directory.
        output_dir: Sequence-cache root; ``train``/``val``/``test`` subdirs are written.
        subjects: Subject identifiers pooled into the split.
        min_seg_len: Minimum motion-segment length (frames) to keep.
        max_seg_len: Maximum segment length (frames); longer are truncated.
        test_size: Held-out test fraction of repetition files (0.2 = benchmark).
        val_size: Validation fraction of all files, carved from the train portion.
        seed: Split random state (1 = benchmark default).
        subject_filter: If given, drop the units of every other subject after
            the canonical assignment, so each kept unit keeps its split and a
            filtered test set is a subset of the pooled one.

    Raises:
        ValueError: If no unit is found or ``subject_filter`` names an unknown
            subject.
    """
    units = _enumerate_units(embedded_root, labels_root, subjects)
    if not units:
        raise ValueError(f"No labeled units found under {embedded_root}")

    assign = assign_unit_splits(len(units), test_size, val_size, seed)
    logger.info(
        "Split %d units (seed %d): train=%d val=%d test=%d",
        len(units), seed,
        sum(v == "train" for v in assign.values()),
        sum(v == "val" for v in assign.values()),
        sum(v == "test" for v in assign.values()),
    )

    keep = set(subject_filter) if subject_filter is not None else None
    if keep is not None:
        # Validated against the subject list, not the units: some canonical
        # subjects ('6', '6_continued') contribute zero labelled units.
        unknown = keep - set(subjects)
        if unknown:
            raise ValueError(f"subject_filter names unknown subjects: {sorted(unknown)}")
        empty = keep - {u["subject"] for u in units}
        if empty:
            logger.warning("subject_filter entries with zero labelled units: %s", sorted(empty))
        kept = [assign[i] for i, u in enumerate(units) if u["subject"] in keep]
        logger.info(
            "Subject filter keeps %d/%d units: train=%d val=%d test=%d",
            len(kept), len(units),
            sum(v == "train" for v in kept),
            sum(v == "val" for v in kept),
            sum(v == "test" for v in kept),
        )

    for name in ("train", "val", "test"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    rec_idx = {"train": 0, "val": 0, "test": 0}
    metas: dict[str, list[RecordingMeta]] = {"train": [], "val": [], "test": []}
    dropped = {"train": 0, "val": 0, "test": 0}

    by_subject: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(units):
        by_subject[u["subject"]].append(i)

    for subject, unit_ids in by_subject.items():
        if keep is not None and subject not in keep:
            continue
        mu, logvar, _ = _load_subject(embedded_root / f"subject_{subject}")
        for i in unit_ids:
            split = assign[i]
            new_metas, rec_idx[split], n_drop = _carve_unit(
                mu, logvar, units[i], output_dir / split, rec_idx[split],
                min_seg_len, max_seg_len,
            )
            metas[split].extend(new_metas)
            dropped[split] += n_drop

    for name in ("train", "val", "test"):
        torch.save([asdict(m) for m in metas[name]], output_dir / name / "meta.pt")
        per_class: dict[int, int] = {}
        for m in metas[name]:
            per_class[m.activity_idx] = per_class.get(m.activity_idx, 0) + 1
        logger.info(
            "%s: %d windows (%d dropped); classes present: %d/%d, "
            "min/max per class: %s/%s",
            name, rec_idx[name], dropped[name], len(per_class), N_CLASSES,
            min(per_class.values()) if per_class else 0,
            max(per_class.values()) if per_class else 0,
        )


def main() -> None:
    """Build the CI4R-ASL sequence caches from per-subject embeddings."""
    parser = argparse.ArgumentParser(
        description="Carve CI4R-ASL motion segments into subject-mixed train/val/test caches.",
    )
    parser.add_argument("--embedded-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", type=str, nargs="+", required=True)
    parser.add_argument("--min-seg-len", type=int, default=5)
    parser.add_argument("--max-seg-len", type=int, default=512)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--subject-filter", type=str, nargs="+", default=None)
    parser.add_argument(
        "--subject-group", choices=("all", "hearing"), default="all",
        help="'hearing' keeps only the hearing participants (a filter over the "
             "canonical assignment); shorthand for --subject-filter.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    subject_filter = args.subject_filter
    if args.subject_group == "hearing":
        if subject_filter is not None:
            parser.error("--subject-group hearing and --subject-filter are mutually exclusive")
        subject_filter = HEARING_SUBJECTS
    build_sequences_ci4r_mixed(
        embedded_root=args.embedded_root,
        labels_root=args.labels_root,
        output_dir=args.output_dir,
        subjects=args.subjects,
        min_seg_len=args.min_seg_len,
        max_seg_len=args.max_seg_len,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        subject_filter=subject_filter,
    )


if __name__ == "__main__":
    main()
