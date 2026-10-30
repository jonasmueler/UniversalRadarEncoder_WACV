"""Build the train-only CI4R pretraining stores from a split manifest."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from frozenradar.data.ci4r_dataset import (
    _INT16_PER_FRAME,
    _RAW_BIN_RE,
    _read_int16_slice,
    group_recordings,
)
from frozenradar.data.ci4r_splits import (
    N_REPS,
    read_manifest,
    require_pinned_digest,
    test_keys,
    train_entries,
    write_store_manifest,
)

logger = logging.getLogger(__name__)


def _select_frames(rep_len: int, keep: int | None) -> list[int]:
    """Pick ``keep`` frame indices spread evenly across one repetition.

    Adjacent frames are highly correlated (a repetition is ~24 s at ~25 fps),
    so an even spread covers more variation than the first ``keep`` frames.

    Args:
        rep_len: Frames in the repetition.
        keep: Frames to keep; ``None`` keeps all.

    Returns:
        Sorted frame indices.
    """
    if keep is None or keep >= rep_len:
        return list(range(rep_len))
    return [round(i * (rep_len - 1) / (keep - 1)) for i in range(keep)] if keep > 1 else [0]


def extract(
    manifest_path: Path, raw_root: Path, out_root: Path, frames_per_rep: int | None,
) -> None:
    """Write the frames of train-assigned repetitions to a pretraining store.

    CI4R-ASL splits at the repetition level, so a raw ``.bin`` recording holds
    both train and test frames. This writes one ``.npy`` per train repetition,
    so the store contains no benchmark val or test frame.

    Args:
        manifest_path: Manifest from ``ci4r_splits.build_ci4r_manifest``.
        raw_root: Root of the raw ``.bin`` tree.
        out_root: Destination store.
        frames_per_rep: Frames kept per repetition (``None`` = all).

    Raises:
        ValueError: On a non-CI4R-ASL or unpinned manifest, ambiguous recording
            names, a non-empty ``out_root``, or recordings whose length differs
            from the manifest.
        AssertionError: If a train entry is also marked test.
        RuntimeError: If a train repetition has no recording or nothing is written.
    """
    manifest = read_manifest(manifest_path)
    if manifest["dataset"] != "ci4r":
        raise ValueError(f"expected a ci4r manifest, got {manifest['dataset']!r}")
    require_pinned_digest("ci4r", manifest["order_digest"])
    train = train_entries(manifest)
    forbidden = test_keys(manifest)
    logger.info(
        "manifest %s: %d units, %d train, digest %s",
        manifest_path, manifest["n_units"], len(train), manifest["order_digest"],
    )

    # Index the raw tree once: base name -> ordered part paths.
    by_base: dict[str, list[list[Path]]] = {}
    for key, parts in group_recordings(raw_root).items():
        by_base.setdefault(Path(key).name, []).append(parts)
    recordings = {b: v[0] for b, v in by_base.items() if len(v) == 1}
    ambiguous = {b for b, v in by_base.items() if len(v) > 1}
    if ambiguous:
        raise ValueError(
            f"{len(ambiguous)} recording base names occur in more than one directory "
            f"(e.g. {sorted(ambiguous)[:3]}); the manifest cannot address them unambiguously"
        )

    # A store is always built from scratch, so no unit of another split survives.
    if out_root.exists() and any(out_root.iterdir()):
        raise ValueError(
            f"{out_root} is not empty. A store must be rebuilt from scratch, never "
            "added to: leftovers from an earlier manifest would silently survive and "
            "may be test repetitions. Delete it and re-run."
        )
    out_root.mkdir(parents=True, exist_ok=True)
    written_keys: list[str] = []
    written = frames_out = skipped = 0
    for entry in train:
        subject, base, rep = entry["subject"], entry["base"], entry["rep"]
        key = f"{subject}|{base}|{rep}"
        if key in forbidden:
            raise AssertionError(f"train entry {key} is also marked test")
        parts = recordings.get(base)
        if parts is None:
            skipped += 1
            continue

        sizes = [p.stat().st_size // 2 for p in parts]
        n_frames = sum(sizes) // _INT16_PER_FRAME
        rep_len = n_frames // N_REPS
        if rep_len == 0:
            skipped += 1
            continue
        # If the recording length differs from the one the split was defined
        # against, every repetition window shifts into its neighbour.
        want_n = entry.get("n_frames")
        if want_n is None:
            raise ValueError(
                f"{base}: the manifest records no n_frames, so it cannot be checked "
                "against this raw tree; rebuild it."
            )
        if want_n != n_frames:
            raise ValueError(
                f"{base}: raw tree has {n_frames} frames but the split was defined "
                f"against {want_n}. The repetition boundaries have moved, so the "
                "manifest no longer describes this data — rebuild the manifest."
            )
        offset = (rep - 1) * rep_len               # rep is 1-based
        idx = _select_frames(rep_len, frames_per_rep)

        buf = np.empty((len(idx), _INT16_PER_FRAME), dtype=np.int16)
        for j, f in enumerate(idx):
            start = (offset + f) * _INT16_PER_FRAME
            buf[j] = _read_int16_slice(parts, sizes, start, start + _INT16_PER_FRAME)

        dest = out_root / f"subject_{subject}" / f"{base}_rep{rep}.npy"
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, buf)
        written_keys.append(key)
        written += 1
        frames_out += len(idx)
        if written % 50 == 0:
            logger.info("  %d/%d reps written (%d frames)", written, len(train), frames_out)

    logger.info(
        "done: %d reps, %d frames, %.1f GB -> %s (%d reps skipped: recording not in raw tree)",
        written, frames_out, frames_out * _INT16_PER_FRAME * 2 / 1e9, out_root, skipped,
    )
    if skipped:
        raise RuntimeError(
            f"{skipped} of {len(train)} train repetitions have no recording under "
            f"{raw_root}. The manifest describes data this tree does not contain — "
            "extract the missing archives first, or rebuild the manifest against "
            "the tree you actually have."
        )
    if written == 0:
        raise RuntimeError("no repetitions written — raw_root probably does not match the manifest")
    write_store_manifest(manifest, out_root, written_keys)


def link_multi3_train(manifest_path: Path, raw_root: Path, out_root: Path) -> None:
    """Populate a train-only CI4R-Gait store by hard-linking whole recordings.

    CI4R-Gait splits at the recording level, so a whole ``.bin`` belongs to
    exactly one split. Hard links (or copies across devices) produce a tree that
    contains train recordings only.

    Args:
        manifest_path: Manifest from ``ci4r_splits.build_multi3_manifest``.
        raw_root: Root of the raw CI4R-Gait ``.bin`` tree.
        out_root: Destination store.

    Raises:
        ValueError: On a non-CI4R-Gait or unpinned manifest or a non-empty
            ``out_root``.
        AssertionError: If a recording is marked both train and non-train.
        RuntimeError: If train recordings are missing from ``raw_root``.
    """
    manifest = read_manifest(manifest_path)
    if manifest["dataset"] != "multi3":
        raise ValueError(f"expected a multi3 manifest, got {manifest['dataset']!r}")
    require_pinned_digest("multi3", manifest["order_digest"])
    # Matched on (activity folder, base): the same base occurs in several folders.
    train = {(Path(e["key"]).parent.name, e["base"]) for e in train_entries(manifest)}
    non_train = {
        (Path(e["key"]).parent.name, e["base"])
        for e in manifest["entries"] if e["split"] != "train"
    }
    if train & non_train:
        raise AssertionError("a recording is marked both train and non-train")

    if out_root.exists() and any(out_root.iterdir()):
        raise ValueError(
            f"{out_root} is not empty. A store must be rebuilt from scratch, never "
            "added to: leftovers from an earlier manifest may be test recordings."
        )
    out_root.mkdir(parents=True, exist_ok=True)
    linked_keys: list[str] = []
    linked = 0
    for path in sorted(Path(raw_root).rglob("*.bin")):
        base = _RAW_BIN_RE.sub(r"\1", path.name) if _RAW_BIN_RE.search(path.name) else path.stem
        ident = (path.parent.name, base)
        if ident not in train:
            continue
        dest = out_root / path.parent.name / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            try:
                dest.hardlink_to(path)
            except OSError:  # cross-device: copy instead
                dest.write_bytes(path.read_bytes())
        linked += 1
        linked_keys.append(f"{ident[0]}|{ident[1]}")
    logger.info("CI4R-Gait store: %d train parts under %s", linked, out_root)
    if linked == 0:
        raise RuntimeError("no train recordings linked — raw_root does not match the manifest")
    linked_idents = {tuple(k.split("|", 1)) for k in linked_keys}
    missing = train - linked_idents
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(train)} train recordings are absent from {raw_root} "
            f"(e.g. {sorted(missing)[:3]}). Extract the missing archives, or rebuild the "
            "manifest against the tree you actually have."
        )
    write_store_manifest(manifest, out_root, sorted(set(linked_keys)))


def main() -> None:
    """Build one train-only store from a manifest."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("ci4r", "multi3"), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    # 120 of ~605 frames per repetition: ~126k train frames from the 1047 train
    # repetitions, comparable to mmRadPose's 136,600, in a ~126 GB store.
    p.add_argument("--frames-per-rep", type=int, default=120,
                   help="CI4R-ASL only; frames kept per repetition, 0 = keep all")
    a = p.parse_args()
    if a.dataset == "ci4r":
        extract(a.manifest, a.raw_root, a.out_root,
                None if a.frames_per_rep == 0 else a.frames_per_rep)
    else:
        link_multi3_train(a.manifest, a.raw_root, a.out_root)


if __name__ == "__main__":
    main()
