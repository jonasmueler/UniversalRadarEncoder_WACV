"""Build the leave-one-subject-out sequence caches for CI4R-Gait."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from frozenradar.classification.build_sequences import RecordingMeta
from frozenradar.classification.build_sequences_multi3 import (
    N_CLASSES,
    _load_embeddings,
    _subject_of,
)
from frozenradar.data.multi3_dataset import ACTIVITY_NAMES, activity_from_key

logger = logging.getLogger(__name__)

# Subject codes encoded in the release's recording file names: 0010 (116
# recordings), 0014 (91), 0029 (221), 0030 (214). The dataset publication reports
# six participants; the release exposes these four codes. Code 0000 (10
# recordings of a single activity) cannot be a test fold and is always train.
FOLD_SUBJECTS = ["0010", "0014", "0029", "0030"]
ALWAYS_TRAIN_SUBJECTS = {"0000"}

# Written into every fold manifest.
BUILDER_VERSION = "loso1"


def loso_assign(
    subjects: list[str], fold_subject: str, val_size: float, seed: int,
) -> dict[int, str]:
    """Assign recording indices to train/val/test for one held-out subject.

    The single definition of a LOSO fold. Test is every recording of
    ``fold_subject``. Val is drawn only from the remaining subjects, excluding
    ``ALWAYS_TRAIN_SUBJECTS``, so neither training nor model selection sees the
    held-out subject.

    Args:
        subjects: Subject code per recording, in enumeration order.
        fold_subject: The held-out subject code.
        val_size: Validation fraction of the val-eligible recordings.
        seed: Val-draw seed.

    Returns:
        Mapping recording index -> ``"train"`` / ``"val"`` / ``"test"``.

    Raises:
        AssertionError: If the result is not subject-disjoint or incomplete.
    """
    test_idx = {i for i, s in enumerate(subjects) if s == fold_subject}
    rest_idx = [i for i, s in enumerate(subjects) if s != fold_subject]
    val_pool = [i for i in rest_idx if subjects[i] not in ALWAYS_TRAIN_SUBJECTS]
    perm = np.random.RandomState(seed).permutation(len(val_pool))
    n_val = int(round(len(val_pool) * val_size))
    val_idx = {val_pool[i] for i in perm[:n_val]}
    assign = {
        i: ("test" if i in test_idx else "val" if i in val_idx else "train")
        for i in range(len(subjects))
    }

    test_subjects = {subjects[i] for i, v in assign.items() if v == "test"}
    trainval_subjects = {subjects[i] for i, v in assign.items() if v != "test"}
    if test_subjects != {fold_subject}:
        raise AssertionError(f"test contains {test_subjects}, expected {{{fold_subject!r}}}")
    if fold_subject in trainval_subjects:
        raise AssertionError(
            f"subject {fold_subject!r} appears in train/val — the fold is not "
            "subject-disjoint; refusing to proceed"
        )
    if len(assign) != len(subjects):
        raise AssertionError("split does not cover every recording exactly once")
    return assign


def build_sequences_multi3_loso(
    embedded_dir: Path,
    output_dir: Path,
    fold_subject: str,
    max_frames: int = 512,
    val_size: float = 0.1,
    seed: int = 1,
) -> None:
    """Build the subject-disjoint caches of one CI4R-Gait LOSO fold.

    Every recording of ``fold_subject`` becomes the test set; the remaining
    subjects supply train and val. Recordings are enumerated as in the
    subject-mixed builder (sorted keys, labelled only, ``max_frames`` cap) and
    written in the same layout (``train/ val/ test/`` with ``rec_NNNN.pt`` and
    ``meta.pt``). A ``fold_manifest.json`` records the per-split recording keys.

    Args:
        embedded_dir: Directory of extracted embeddings (``batch_*.pt`` + ``index.pt``).
        output_dir: Sequence-cache root for THIS fold.
        fold_subject: Subject code whose recordings form the test set.
        max_frames: Cap on frames per recording (fits the positional embedding).
        val_size: Validation fraction, carved from the non-fold subjects only.
        seed: Val-carving seed.

    Raises:
        ValueError: On an unusable fold subject, no labelled recordings, or
            unknown subject codes.
        AssertionError: If the training split misses a class.
    """
    if fold_subject not in FOLD_SUBJECTS:
        raise ValueError(
            f"fold_subject {fold_subject!r} is not a usable fold; choose from "
            f"{FOLD_SUBJECTS} (code 0000 has a single class and cannot be tested on)"
        )

    mu, logvar, frame_counts = _load_embeddings(embedded_dir)

    # Recording slices in extraction order, as in the subject-mixed builder.
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
    if not records:
        raise ValueError(f"no labelled recordings under {embedded_dir}")

    subjects = [_subject_of(key) for key, _, _, _ in records]
    known = set(FOLD_SUBJECTS) | ALWAYS_TRAIN_SUBJECTS
    unknown = {s for s in subjects if s not in known}
    if unknown:
        raise ValueError(
            f"unrecognised subject codes {sorted(unknown)}: the fold definition "
            "no longer covers the data — refusing to build a split with "
            "unattributed recordings"
        )
    if fold_subject not in subjects:
        raise ValueError(f"no recordings for fold subject {fold_subject!r}")

    assign = loso_assign(subjects, fold_subject, val_size, seed)
    counts = {s: sum(v == s for v in assign.values()) for s in ("train", "val", "test")}
    train_classes = {records[i][1] for i, v in assign.items() if v == "train"}
    if len(train_classes) != N_CLASSES:
        raise AssertionError(
            f"train covers {len(train_classes)}/{N_CLASSES} classes for fold "
            f"{fold_subject!r} — the head could not learn the missing ones"
        )

    logger.info(
        "CI4R-Gait LOSO fold %s: %d labelled recordings (%d unlabeled skipped); "
        "train=%d val=%d test=%d",
        fold_subject, len(records), n_unlabeled,
        counts["train"], counts["val"], counts["test"],
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
                participant=subjects[i],
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

    (output_dir / "fold_manifest.json").write_text(json.dumps({
        "builder_version": BUILDER_VERSION,
        "fold_subject": fold_subject,
        "val_size": val_size,
        "seed": seed,
        "max_frames": max_frames,
        "counts": counts,
        "keys": {
            s: sorted(records[i][0] for i, v in assign.items() if v == s)
            for s in ("train", "val", "test")
        },
    }, indent=2), encoding="utf8")
    logger.info("fold manifest written to %s", output_dir / "fold_manifest.json")


def main() -> None:
    """Build one LOSO fold's sequence caches from extracted embeddings."""
    parser = argparse.ArgumentParser(
        description="Build subject-disjoint (LOSO) CI4R-Gait caches.",
    )
    parser.add_argument("--embedded-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-subject", required=True, choices=FOLD_SUBJECTS)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    build_sequences_multi3_loso(
        embedded_dir=args.embedded_dir,
        output_dir=args.output_dir,
        fold_subject=args.fold_subject,
        max_frames=args.max_frames,
        val_size=args.val_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
