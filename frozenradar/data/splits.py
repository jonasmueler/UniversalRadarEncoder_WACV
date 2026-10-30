"""Train/val/test splits for mmRadPose and CubeLearn."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

SplitName = Literal["train", "val", "test"]

MMRADPOSE_PARTICIPANTS: dict[SplitName, list[str]] = {
    "train": ["p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11"],
    "val": ["p3"],
    "test": ["p1", "p2", "p12"],
}

# Train, val and test all come from the same six users; no sample index is shared.
CUBELEARN_USERS: list[int] = [0, 1, 2, 3, 4, 5]

# Out-of-set users (CubeLearn paper, Sec. 6.2): never trained on; all 30 samples of
# each gesture/activity are used for cross-user testing.
CUBELEARN_OUTOFSET_USERS: list[int] = [6, 7]

DEFAULT_CUBELEARN_SPLIT_DIR: Path = (
    Path(__file__).resolve().parents[2] / "splits" / "cubelearn"
)

_CUBELEARN_SPLIT_FILES: dict[SplitName, str] = {
    "train": "train_samples.csv",
    "val": "validation_samples.csv",
    "test": "test_samples.csv",
}


def cubelearn_sample_ids(split_dir: str | Path | None = None) -> dict[SplitName, list[int]]:
    """Load CubeLearn's official train/val/test sample-index split.

    Args:
        split_dir: Directory holding ``{train,validation,test}_samples.csv``.
            Defaults to the copy shipped in the repo.

    Returns:
        Mapping of split name to its list of sample indices.

    Raises:
        FileNotFoundError: If a split CSV is missing.
        ValueError: If the loaded splits are not pairwise disjoint.
    """
    root = Path(split_dir) if split_dir is not None else DEFAULT_CUBELEARN_SPLIT_DIR
    sample_ids: dict[SplitName, list[int]] = {}
    for split, fname in _CUBELEARN_SPLIT_FILES.items():
        path = root / fname
        if not path.exists():
            raise FileNotFoundError(f"CubeLearn split file not found: {path}")
        ids = np.genfromtxt(path, dtype=int).reshape(-1).tolist()
        sample_ids[split] = [int(i) for i in ids]
    assert_disjoint(sample_ids, label="CubeLearn sample-ids")
    return sample_ids


def assert_disjoint(splits: dict[SplitName, list], label: str = "splits") -> None:
    """Check that the train/val/test lists share no element.

    Args:
        splits: Mapping of split name to a list of subject/sample identifiers.
        label: Human-readable name used in the error message.

    Raises:
        ValueError: If any two splits overlap.
    """
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(splits.get(a, [])) & set(splits.get(b, []))
        if overlap:
            raise ValueError(f"{label}: {a}/{b} splits overlap on {sorted(overlap)}")


assert_disjoint(MMRADPOSE_PARTICIPANTS, label="mmRadPose participants")
