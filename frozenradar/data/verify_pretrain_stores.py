"""Check the CI4R pretraining stores, their split manifests and the sampler."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from frozenradar.data.ci4r_dataset import decode_frame
from frozenradar.data.ci4r_pretrain_dataset import (
    _M3_INT16_PER_FRAME,
    CI4RPretrainDataset,
    Multi3PretrainDataset,
    _multi3_base,
)
from frozenradar.data.ci4r_splits import (
    CANONICAL_DIGEST,
    CANONICAL_SUBJECTS,
    build_ci4r_manifest,
    build_multi3_manifest,
    read_manifest,
    test_keys,
)
from frozenradar.data.multi3_dataset import _lvds_to_rx, _rx_to_frames
from frozenradar.data.samplers import build_balanced_sampler

logger = logging.getLogger(__name__)

# Cells and rows of the 4x4 antenna grid each sensor occupies before the angle
# FFT; a wrong TX/RX mapping keeps shape, dtype and plausible magnitudes.
EXPECTED_OCCUPANCY = {"ci4r": (8, [1, 1, 0, 0]), "multi3": (4, [1, 0, 0, 0])}


class Check:
    """Collects pass/fail results so every check runs before exiting."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> None:
        """Report and record one check.

        Args:
            name: Check name.
            ok: Whether the check passed.
            detail: Optional detail appended to the report line.
        """
        logger.info("  [%s] %s%s", "PASS" if ok else "FAIL", name, f" — {detail}" if detail else "")
        if not ok:
            self.failures.append(name)


class _FakeMultiSource:
    """Stand-in for a ``MultiRadarDataset`` exposing only what the sampler reads.

    Args:
        lengths: Frame count per source name.
    """

    def __init__(self, lengths: dict[str, int]) -> None:
        self.source_names = list(lengths)
        self.source_lengths = dict(lengths)
        ids: list[int] = []
        for i, n in enumerate(lengths.values()):
            ids += [i] * n
        self.source_ids = torch.tensor(ids)

    def __len__(self) -> int:
        return int(self.source_ids.numel())


def check_split_integrity(check: Check, manifest_path: Path, store_root: Path) -> None:
    """Check that no unit in the store is a benchmark test unit.

    Args:
        check: Result collector.
        manifest_path: Split manifest of the store.
        store_root: Store directory.
    """
    manifest = read_manifest(manifest_path)
    forbidden = test_keys(manifest)
    if manifest["dataset"] == "ci4r":
        present = {
            f"{f.parent.name.removeprefix('subject_')}|"
            f"{f.stem.rsplit('_rep', 1)[0]}|{f.stem.rsplit('_rep', 1)[1]}"
            for f in store_root.rglob("*.npy")
        }
    else:
        present = {f"{f.parent.name}|{_multi3_base(f)}" for f in store_root.rglob("*.bin")}
    leaked = present & forbidden
    check(
        f"{manifest['dataset']}: store holds no test unit",
        not leaked,
        f"{len(present)} units, {len(leaked)} leaked"
        + (f" e.g. {sorted(leaked)[:3]}" if leaked else ""),
    )


def check_manifest_reproducible(
    check: Check, manifest_path: Path, embedded_root: Path, labels_root: Path | None,
) -> None:
    """Check that rebuilding the manifest gives an identical assignment.

    Catches a changed subject order, seed or unit count, each of which yields a
    different partition while every other check still passes.

    Args:
        check: Result collector.
        manifest_path: Stored manifest.
        embedded_root: Embedding cache to rebuild the manifest from.
        labels_root: CI4R-ASL label root (unused for CI4R-Gait).
    """
    stored = read_manifest(manifest_path)
    try:
        _check_manifest_reproducible(check, stored, embedded_root, labels_root)
    except Exception as exc:
        # A digest mismatch raises inside the builder; record it as a failure so
        # the remaining checks still run.
        check(f"{stored['dataset']}: manifest reproduces exactly", False,
              f"{type(exc).__name__}: {exc}")


def _check_manifest_reproducible(
    check: Check, stored: dict, embedded_root: Path, labels_root: Path | None,
) -> None:
    """Rebuild the manifest and compare it with the stored one (may raise).

    Args:
        check: Result collector.
        stored: Stored manifest.
        embedded_root: Embedding cache to rebuild the manifest from.
        labels_root: CI4R-ASL label root (unused for CI4R-Gait).
    """
    if stored["dataset"] == "ci4r":
        # Rebuilt from CANONICAL_SUBJECTS, not stored["subjects"], so a manifest
        # built in the wrong order cannot pass.
        rebuilt = build_ci4r_manifest(embedded_root, labels_root, CANONICAL_SUBJECTS)
        if stored["subjects"] != CANONICAL_SUBJECTS:
            check("ci4r: manifest used the canonical subject order", False,
                  f"{stored['subjects']} != {CANONICAL_SUBJECTS}")
    else:
        rebuilt = build_multi3_manifest(embedded_root)
    same_digest = rebuilt["order_digest"] == stored["order_digest"]
    same_splits = [e["split"] for e in rebuilt["entries"]] == [
        e["split"] for e in stored["entries"]
    ]
    check(
        f"{stored['dataset']}: manifest reproduces exactly",
        same_digest and same_splits,
        f"digest {stored['order_digest']} vs {rebuilt['order_digest']}",
    )


def check_frames(
    check: Check, name: str, ds: CI4RPretrainDataset | Multi3PretrainDataset,
) -> None:
    """Check shape, dtype and pre-FFT antenna occupancy of one stored frame.

    Args:
        check: Result collector.
        name: ``"ci4r"`` or ``"multi3"``.
        ds: The store's pretraining dataset.
    """
    x = ds[0]
    check(f"{name}: frame shape/dtype", tuple(x.shape) == (128, 4, 4, 256)
          and x.dtype == torch.complex64, f"{tuple(x.shape)} {x.dtype}")

    want_cells, want_rows = EXPECTED_OCCUPANCY[name]
    if name == "ci4r":
        raw = np.asarray(np.load(ds._files[0], mmap_mode="r")[0])
        grid = decode_frame(raw)
    else:
        raw = np.fromfile(ds._files[0], dtype=np.int16, count=_M3_INT16_PER_FRAME)
        grid = _rx_to_frames(_lvds_to_rx(raw), 1)[0]
    occ = grid.abs().sum(dim=(0, -1)) > 0
    check(
        f"{name}: virtual-array occupancy (pre-FFT)",
        int(occ.sum()) == want_cells and occ.any(dim=1).int().tolist() == want_rows,
        f"{int(occ.sum())}/16 cells, rows {occ.any(dim=1).int().tolist()}",
    )


def check_sampler(check: Check, n_sources: int = 4, budget: int = 169_000) -> None:
    """Check that sources are balanced and each epoch draws a fresh subset.

    Args:
        check: Result collector.
        n_sources: Number of fake sources.
        budget: Draws per epoch.
    """
    ds = _FakeMultiSource({"a": 380_000, "b": 136_600, "c": 114_000, "d": 32_400})
    sampler = build_balanced_sampler(ds, None, num_samples=budget)
    e1, e2 = torch.tensor(list(sampler)), torch.tensor(list(sampler))
    shares = [(ds.source_ids[e1] == i).float().mean().item() for i in range(n_sources)]
    check("sampler: sources within 2% of equal share",
          max(abs(s - 1 / n_sources) for s in shares) < 0.02,
          " ".join(f"{s:.3f}" for s in shares))
    check("sampler: epochs draw different subsets", not torch.equal(e1, e2))
    check("sampler: default reproduces len(dataset)",
          build_balanced_sampler(ds, None).num_samples == len(ds))


def main() -> None:
    """Run every store and split check; exit non-zero if any fails."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.INFO)
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--ci4r-store", type=Path)
    p.add_argument("--ci4r-manifest", type=Path)
    p.add_argument("--ci4r-embedded-root", type=Path)
    p.add_argument("--ci4r-labels-root", type=Path)
    p.add_argument("--multi3-store", type=Path)
    p.add_argument("--multi3-manifest", type=Path)
    p.add_argument("--multi3-embedded-root", type=Path)
    a = p.parse_args()
    if a.ci4r_embedded_root is not None and a.ci4r_labels_root is None:
        p.error("--ci4r-labels-root is required alongside --ci4r-embedded-root")

    check = Check()
    logger.info("pinned split digests")
    for ds_name, dig in CANONICAL_DIGEST.items():
        check(f"{ds_name}: canonical digest pinned", bool(dig), dig or "unpinned")
    logger.info("sampler")
    check_sampler(check)

    for name, store, manifest, embedded, labels in (
        ("ci4r", a.ci4r_store, a.ci4r_manifest, a.ci4r_embedded_root, a.ci4r_labels_root),
        ("multi3", a.multi3_store, a.multi3_manifest, a.multi3_embedded_root, None),
    ):
        logger.info(name)
        if store is None:
            logger.info("  [SKIP] no --%s-store given", name)
            continue
        cls = CI4RPretrainDataset if name == "ci4r" else Multi3PretrainDataset
        # Constructing the dataset verifies the store stamp.
        ds = cls(store, split="train")
        check(f"{name}: store stamp verified on load", True, f"{len(ds)} frames")
        check_frames(check, name, ds)
        if manifest is not None:
            check_split_integrity(check, manifest, store)
            if embedded is not None:
                check_manifest_reproducible(check, manifest, embedded, labels)

    logger.info("")
    if check.failures:
        logger.info("FAILED: %d check(s): %s", len(check.failures), ", ".join(check.failures))
        sys.exit(1)
    logger.info("all checks passed")


if __name__ == "__main__":
    main()
