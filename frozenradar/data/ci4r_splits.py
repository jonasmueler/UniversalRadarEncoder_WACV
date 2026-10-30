"""Split manifests and store stamps for CI4R-ASL and CI4R-Gait."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import torch

from frozenradar.classification.build_sequences_ci4r import (
    CANONICAL_SUBJECTS,
    HEARING_SUBJECTS,
    N_REPS,
    _enumerate_units,
    assign_unit_splits,
)
from frozenradar.classification.build_sequences_multi3 import (
    _load_embeddings,
    _shuffle_split,
    activity_from_key,
)
from frozenradar.data.ci4r_dataset import _INT16_PER_FRAME
from frozenradar.data.ci4r_dataset import group_recordings as ci4r_group_recordings
from frozenradar.data.multi3_dataset import group_recordings as multi3_group_recordings

__all__ = [
    "CANONICAL_DIGEST",
    "CANONICAL_SUBJECTS",
    "CANONICAL_UNIT_COUNT",
    "HEARING_SUBJECTS",
    "STORE_MANIFEST_NAME",
    "build_ci4r_manifest",
    "build_multi3_manifest",
    "enumerate_units_from_raw",
    "read_manifest",
    "require_pinned_digest",
    "test_keys",
    "train_entries",
    "verify_store",
    "write_manifest",
    "write_store_manifest",
]

logger = logging.getLogger(__name__)

# Benchmark protocol, identical to the build_sequences_ci4r/_multi3 defaults.
# The test slice is sized round(n * test_size) (sklearn's train_test_split uses
# ceil): 299 instead of 300 CI4R-ASL units and 130 instead of 131 CI4R-Gait
# recordings, with the borderline unit in val.
TEST_SIZE = 0.2
VAL_SIZE = 0.1
SEED = 1

STORE_MANIFEST_NAME = "_store_manifest.json"

# Digests of the published unit orderings (see _digest). The split is assigned
# positionally over the unit list, so a different subject order or unit set
# yields a different partition and a different digest.
#   ci4r    1496 repetition units (21 subject dirs) -> train 1047 / val 150 / test 299
#   multi3   652 recordings                         -> train  457 / val  65 / test 130
CANONICAL_DIGEST: dict[str, str] = {
    "ci4r": "d684d5a9125a4bd1",
    "multi3": "a8d6e1becf5193ac",
}

CANONICAL_UNIT_COUNT: dict[str, int] = {
    "ci4r": 1496,
    "multi3": 652,
}


def _digest(keys: list[str]) -> str:
    """Fingerprint a unit ordering.

    Args:
        keys: Unit identifiers in enumeration order.

    Returns:
        The first 16 hex digits of the SHA-256 of the newline-joined keys.
    """
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def enumerate_units_from_raw(
    raw_root: Path, labels_root: Path, subjects: list[str],
) -> list[dict]:
    """Enumerate CI4R-ASL repetition units from the raw tree, without embeddings.

    Mirrors ``build_sequences_ci4r._enumerate_units`` exactly (subjects in the
    given order, then ``sorted(keys)`` within each subject, then repetitions
    1..N_REPS, skipping repetitions without a label file) but takes the frame
    counts from the ``.bin`` sizes instead of an embedding index. The split is
    assigned positionally over this list, so the order defines the partition.

    Args:
        raw_root: Root of the raw ``.bin`` tree (``subject <s>/`` directories).
        labels_root: Unzipped ``ASL Sequential Dataset`` root.
        subjects: Subject identifiers, in the canonical order.

    Returns:
        Unit dicts with subject, base, rep and n_frames.

    Raises:
        ValueError: If a subject has no raw directory.
    """
    units: list[dict] = []
    for subject in subjects:
        subject_dir = Path(raw_root) / f"subject {subject}"
        if not subject_dir.is_dir():
            raise ValueError(f"no raw directory for subject {subject!r} at {subject_dir}")

        # One listing per subject instead of a recursive glob per repetition.
        labels = {
            f.name
            for d in Path(labels_root).glob(f"*/subject {subject}/labels/rangeDoppler")
            for f in d.iterdir() if f.suffix == ".txt"
        }

        recordings = ci4r_group_recordings(subject_dir)
        for key in sorted(recordings):
            n_frames = sum(p.stat().st_size // 2 for p in recordings[key]) // _INT16_PER_FRAME
            base = Path(key).name
            for rep in range(N_REPS):
                if f"{base}_{rep + 1}.txt" not in labels:
                    continue
                units.append({
                    "subject": subject, "base": base, "rep": rep + 1, "n_frames": n_frames,
                })
    return units


def build_ci4r_manifest(
    embedded_root: Path | None,
    labels_root: Path,
    subjects: list[str] | None = None,
    expected_digest: str | None = None,
    raw_root: Path | None = None,
) -> dict:
    """Assign every CI4R-ASL repetition to train/val/test.

    Uses ``assign_unit_splits`` from the sequence builder over the same unit
    enumeration, so the split is the one the benchmark uses. The unit order is
    what the permutation indexes, so it is never re-sorted.

    CI4R-ASL splits at the repetition level: each recording holds ``N_REPS``
    repetitions assigned independently, so one ``.bin`` carries both train and
    test frames. Repetition ``r`` (1-based) covers frames
    ``[(r-1)*rep_len, r*rep_len)`` of its recording, ``rep_len = n_frames // N_REPS``.

    Args:
        embedded_root: Directory of ``subject_<n>/`` embedding dirs (only
            ``index.pt`` is read), or None when ``raw_root`` is given.
        labels_root: Unzipped ``ASL Sequential Dataset`` root.
        subjects: Subject identifiers in canonical order (default
            :data:`CANONICAL_SUBJECTS`).
        expected_digest: Required unit-order digest (default: the pinned one).
        raw_root: Raw ``.bin`` tree to enumerate instead of the embedding index.

    Returns:
        Manifest dict with protocol parameters, the order digest, and one entry
        per repetition carrying its within-recording frame range and split.

    Raises:
        ValueError: If neither root is given, no unit is found, units are
            duplicated, or the digest differs from ``expected_digest``.
    """
    subjects = list(CANONICAL_SUBJECTS if subjects is None else subjects)
    if expected_digest is None:
        expected_digest = CANONICAL_DIGEST["ci4r"]
    if raw_root is not None:
        units = enumerate_units_from_raw(raw_root, labels_root, subjects)
        source = raw_root
    elif embedded_root is not None:
        units = _enumerate_units(embedded_root, labels_root, subjects)
        source = embedded_root
    else:
        raise ValueError("build_ci4r_manifest needs one of raw_root or embedded_root")
    if not units:
        raise ValueError(f"no labelled units found under {source}")
    assign = assign_unit_splits(len(units), TEST_SIZE, VAL_SIZE, SEED)

    # The frame geometry is recorded so the store extractor can assert that the
    # raw tree still has the recording lengths the repetition windows were cut from.
    frames_by_base: dict[str, int] = {}
    if raw_root is not None:
        frames_by_base = {u["base"]: u["n_frames"] for u in units}
    else:
        for subject in subjects:
            index = torch.load(
                Path(embedded_root) / f"subject_{subject}" / "index.pt", weights_only=True,
            )
            for key, n in index["frame_counts"].items():
                frames_by_base[Path(key).name] = int(n)

    entries = []
    for i, unit in enumerate(units):
        n_frames = frames_by_base.get(unit["base"])
        rep_len = None if n_frames is None else n_frames // N_REPS
        entries.append({
            "subject": unit["subject"],
            "base": unit["base"],
            "rep": unit["rep"],
            "split": assign[i],
            "n_frames": n_frames,
            "rep_len": rep_len,
            "frame_start": None if rep_len is None else (unit["rep"] - 1) * rep_len,
        })
    keys = [f'{e["subject"]}|{e["base"]}|{e["rep"]}' for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (subject, base, rep) units — ambiguous split")

    digest = _digest(keys)
    if expected_digest is not None and digest != expected_digest:
        raise ValueError(
            f"unit-order digest {digest} != expected {expected_digest}. The split is "
            "assigned positionally, so a different subject order or a different set of "
            "labelled units produces a DIFFERENT train/test partition. Refusing to "
            "build a manifest that does not match the published split."
        )

    counts = {s: sum(e["split"] == s for e in entries) for s in ("train", "val", "test")}
    logger.info(
        "CI4R-ASL: %d units from %d subjects — train=%d val=%d test=%d (digest %s)",
        len(entries), len(subjects), counts["train"], counts["val"], counts["test"], digest,
    )
    return {
        "dataset": "ci4r",
        "protocol": {
            "test_size": TEST_SIZE, "val_size": VAL_SIZE, "seed": SEED,
            "n_reps": N_REPS, "split_unit": "repetition",
        },
        "subjects": list(subjects),
        "n_units": len(entries),
        "order_digest": digest,
        "counts": counts,
        "entries": entries,
    }


def _multi3_keys_from_raw(raw_root: Path) -> list[str]:
    """Enumerate CI4R-Gait recording keys from the raw ``.bin`` tree.

    Produces exactly the keys the embedding extractor records (recording stem
    relative to the root, activity folder included), so a manifest built here
    hashes to the same digest as one built from an embedding cache.

    Args:
        raw_root: Directory holding ``<NN>_<activity>/*.bin``.

    Returns:
        Recording keys relative to ``raw_root``.
    """
    root_str = str(Path(raw_root).resolve())
    return [
        k.removeprefix(root_str).lstrip("/")
        for k in multi3_group_recordings(Path(raw_root).resolve())
    ]


def build_multi3_manifest(embedded_dir: Path | None, raw_root: Path | None = None) -> dict:
    """Assign every CI4R-Gait recording to train/val/test.

    Uses ``_shuffle_split`` and ``activity_from_key`` from the sequence
    builder. Unlike CI4R-ASL this benchmark splits at the recording level, so a
    whole ``.bin`` belongs to exactly one split.

    Args:
        embedded_dir: Directory holding the CI4R-Gait embedding cache, or None
            when ``raw_root`` is given.
        raw_root: Raw ``.bin`` tree to enumerate instead of an embedding cache.

    Returns:
        Manifest dict, one entry per labelled recording.

    Raises:
        ValueError: If neither root is given or no labelled recording is found.
    """
    if raw_root is not None:
        all_keys = _multi3_keys_from_raw(raw_root)
        source = raw_root
    elif embedded_dir is not None:
        _, _, frame_counts = _load_embeddings(embedded_dir)
        all_keys = list(frame_counts)
        source = embedded_dir
    else:
        raise ValueError("build_multi3_manifest needs embedded_dir or raw_root")
    # Same enumeration as build_sequences_multi3: sorted keys, labelled only.
    keys = [k for k in sorted(all_keys) if activity_from_key(k) is not None]
    if not keys:
        raise ValueError(f"no labelled recordings found under {source}")
    assign = _shuffle_split(len(keys), TEST_SIZE, VAL_SIZE, SEED)

    entries = [
        {"key": k, "base": Path(k).name, "split": assign[i]}
        for i, k in enumerate(keys)
    ]
    counts = {s: sum(e["split"] == s for e in entries) for s in ("train", "val", "test")}
    logger.info(
        "CI4R-Gait: %d labelled recordings — train=%d val=%d test=%d",
        len(entries), counts["train"], counts["val"], counts["test"],
    )
    return {
        "dataset": "multi3",
        "protocol": {
            "test_size": TEST_SIZE, "val_size": VAL_SIZE, "seed": SEED,
            "split_unit": "recording",
        },
        "n_units": len(entries),
        "order_digest": _digest(keys),
        "counts": counts,
        "entries": entries,
    }


def train_entries(manifest: dict) -> list[dict]:
    """Return the train-assigned entries (the only ones pretraining may use).

    Args:
        manifest: A split manifest.

    Returns:
        Entries whose split is ``"train"``.
    """
    return [e for e in manifest["entries"] if e["split"] == "train"]


def test_keys(manifest: dict) -> set[str]:
    """Return the identifiers of every test-assigned unit.

    The keys use the store-stamp key space: ``<subject>|<base>|<rep>`` for
    CI4R-ASL and ``<activity folder>|<base>`` for CI4R-Gait.

    Args:
        manifest: A split manifest.

    Returns:
        Test unit identifiers.
    """
    if manifest["dataset"] == "ci4r":
        return {
            f'{e["subject"]}|{e["base"]}|{e["rep"]}'
            for e in manifest["entries"] if e["split"] == "test"
        }
    return {
        f'{Path(e["key"]).parent.name}|{e["base"]}'
        for e in manifest["entries"] if e["split"] == "test"
    }


def write_manifest(manifest: dict, path: Path) -> None:
    """Write a manifest to JSON, creating parent directories.

    Args:
        manifest: A split manifest.
        path: Output file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    logger.info("wrote %s (%d units, digest %s)",
                path, manifest["n_units"], manifest["order_digest"])


def read_manifest(path: Path) -> dict:
    """Read a manifest from JSON.

    Args:
        path: Manifest file.

    Returns:
        The manifest dict.
    """
    return json.loads(Path(path).read_text(encoding="utf8"))


def require_pinned_digest(dataset: str, digest: str) -> None:
    """Check a manifest digest against the pinned one before building a store.

    ``verify_store`` applies the same check at read time; checking at write time
    avoids building a store no reader would accept.

    Args:
        dataset: ``"ci4r"`` or ``"multi3"``.
        digest: The manifest's order digest.

    Raises:
        ValueError: If the dataset has no pinned digest or the digests differ.
    """
    pinned = CANONICAL_DIGEST.get(dataset)
    if pinned is None:
        raise ValueError(f"no pinned digest for dataset {dataset!r}")
    if digest != pinned:
        raise ValueError(
            f"manifest digest {digest} != pinned {pinned}; this is not the published split"
        )


def write_store_manifest(manifest: dict, store_root: Path, unit_keys: list[str]) -> None:
    """Stamp a store with the split it was built from.

    Readers verify the stamp, which distinguishes a store from a stale one built
    from another manifest and from a raw benchmark tree with the same layout.

    Args:
        manifest: The manifest the store was built from.
        store_root: Store directory.
        unit_keys: Identifiers of the units written to the store.
    """
    (store_root / STORE_MANIFEST_NAME).write_text(json.dumps({
        "dataset": manifest["dataset"],
        "protocol": manifest["protocol"],
        "order_digest": manifest["order_digest"],
        "n_units_total": manifest["n_units"],
        "unit_keys": sorted(unit_keys),
    }, indent=2), encoding="utf8")
    logger.info("stamped %s with %d train units (digest %s)",
                store_root, len(unit_keys), manifest["order_digest"])


def verify_store(store_root: Path, dataset: str, present_keys: list[str]) -> dict:
    """Check a store's stamp before any of its data is used.

    Args:
        store_root: Store to verify.
        dataset: ``"ci4r"`` or ``"multi3"``.
        present_keys: Unit identifiers derived from the files actually found.

    Returns:
        The store manifest.

    Raises:
        ValueError: If the stamp is missing, is for another dataset or another
            split, or if the files on disk are not exactly the units it lists.
    """
    path = Path(store_root) / STORE_MANIFEST_NAME
    if not path.exists():
        raise ValueError(
            f"{store_root} has no {STORE_MANIFEST_NAME}: it was not produced by "
            "extract_ci4r_reps.py and may be a raw benchmark tree containing test "
            "frames. Refusing to pretrain on unverified data."
        )
    stamp = json.loads(path.read_text(encoding="utf8"))
    if stamp["dataset"] != dataset:
        raise ValueError(f"{store_root} is a {stamp['dataset']!r} store, expected {dataset!r}")

    # The stamp alone only proves internal consistency; the digest pinned in
    # this module proves the store was built from the published split.
    pinned = CANONICAL_DIGEST.get(dataset)
    if pinned is None:
        raise ValueError(f"no pinned digest for dataset {dataset!r}")
    if stamp.get("order_digest") != pinned:
        raise ValueError(
            f"{store_root} was built from split {stamp.get('order_digest')} but the "
            f"published split is {pinned}. The unit ordering or count differs, so this "
            "store's 'train' units are a DIFFERENT partition — it will contain "
            "benchmark test repetitions. Rebuild the manifest and the store."
        )
    listed, found = set(stamp["unit_keys"]), set(present_keys)
    if found - listed:
        raise ValueError(
            f"{store_root} holds {len(found - listed)} unit(s) the manifest does not list "
            f"(e.g. {sorted(found - listed)[:3]}). The store is stale or was written under "
            "a different split; rebuild it from scratch rather than adding to it."
        )
    if listed - found:
        raise ValueError(
            f"{store_root} is missing {len(listed - found)} unit(s) the manifest lists "
            "— the store is incomplete (interrupted extraction)."
        )
    return stamp


def main() -> None:
    """Build and write a split manifest.

    ``--expected-digest`` makes the build fail unless the unit ordering hashes
    to the given value.

    Raises:
        ValueError: If a CI4R-Gait manifest does not match ``--expected-digest``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build and write a CI4R split manifest.")
    p.add_argument("--dataset", choices=("ci4r", "multi3"), required=True)
    p.add_argument("--embedded-root", type=Path,
                   help="ci4r: dir of subject_<n>/ embedding dirs; multi3: the embedding cache")
    p.add_argument("--raw-root", type=Path,
                   help="enumerate from the raw .bin tree instead of an embedding cache")
    p.add_argument("--labels-root", type=Path,
                   help="ci4r only: unzipped 'ASL Sequential Dataset' root")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--subjects", nargs="*", default=None,
                   help=f"ci4r only; defaults to the canonical order {CANONICAL_SUBJECTS}")
    p.add_argument("--expected-digest", default=None,
                   help="fail unless the unit ordering hashes to this value")
    a = p.parse_args()

    if a.dataset == "ci4r":
        if a.labels_root is None:
            p.error("--labels-root is required for --dataset ci4r")
        if a.embedded_root is None and a.raw_root is None:
            p.error("--dataset ci4r needs --raw-root or --embedded-root")
        manifest = build_ci4r_manifest(
            a.embedded_root, a.labels_root, a.subjects, a.expected_digest, a.raw_root,
        )
    else:
        if a.embedded_root is None and a.raw_root is None:
            p.error("--dataset multi3 needs --raw-root or --embedded-root")
        manifest = build_multi3_manifest(a.embedded_root, a.raw_root)
        if a.expected_digest and manifest["order_digest"] != a.expected_digest:
            raise ValueError(
                f"digest {manifest['order_digest']} != expected {a.expected_digest}"
            )
    write_manifest(manifest, a.out)


if __name__ == "__main__":
    main()
