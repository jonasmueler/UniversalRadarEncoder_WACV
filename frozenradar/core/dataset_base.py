"""Abstract base class and sample metadata for radar datasets with pose labels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class SampleMeta:
    """Metadata attached to each sample for grouping and analysis.

    Args:
        participant: Participant ID (e.g. ``"p1"``).
        angle: Recording angle (e.g. ``"an0"``).
        activity: Activity/exercise ID (e.g. ``"ex3"``).
        recording: Optional sub-recording identifier.
    """

    participant: str
    angle: str
    activity: str
    recording: str = ""


class RadarPoseDataset(Dataset, ABC):
    """Abstract base class for radar-based pose estimation datasets.

    Subclasses must implement ``_index_data``, ``_load_sample``, ``_get_meta``,
    and ``__len__``.  The base class provides grouping and splitting utilities.

    Args:
        return_meta: If True, ``__getitem__`` returns ``(radar, pose, meta)``
            instead of ``(radar, pose)``.
    """

    return_meta: bool

    @abstractmethod
    def _index_data(self) -> None:
        """Discover all data files and build the internal sample index."""
        ...

    @abstractmethod
    def _load_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and return a single (radar, pose) pair.

        Args:
            idx: Sample index.

        Returns:
            Tuple ``(radar_tensor, pose_tensor)``.
        """
        ...

    @abstractmethod
    def _get_meta(self, idx: int) -> SampleMeta:
        """Return metadata for the sample at *idx*.

        Args:
            idx: Sample index.

        Returns:
            Metadata of this sample.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples."""
        ...

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, SampleMeta]:
        """Return a sample, optionally with metadata.

        Args:
            idx: Sample index.

        Returns:
            ``(radar, pose)``, or ``(radar, pose, meta)`` when ``return_meta`` is True.
        """
        radar, pose = self._load_sample(idx)
        if self.return_meta:
            return radar, pose, self._get_meta(idx)
        return radar, pose

    def groups(self, by: str = "participant") -> dict[str, list[int]]:
        """Group sample indices by a metadata field.

        Args:
            by: One of ``"participant"``, ``"angle"``, ``"activity"``.

        Returns:
            Mapping from group name to list of sample indices.

        Raises:
            ValueError: If ``by`` is not a supported metadata field.
        """
        valid = {"participant", "angle", "activity"}
        if by not in valid:
            raise ValueError(f"`by` must be one of {valid}, got '{by}'")

        mapping: dict[str, list[int]] = {}
        for i in range(len(self)):
            key = getattr(self._get_meta(i), by)
            mapping.setdefault(key, []).append(i)
        return mapping

    def split_by_groups(
        self,
        train_groups: Iterable[str],
        val_groups: Iterable[str] | None = None,
        test_groups: Iterable[str] | None = None,
        by: str = "participant",
    ) -> tuple[Subset, Subset | None, Subset | None]:
        """Split the dataset into train / val / test subsets by metadata groups.

        Args:
            train_groups: Group names for the training set.
            val_groups: Group names for the validation set (may be None).
            test_groups: Group names for the test set (may be None).
            by: Metadata field to group by.

        Returns:
            ``(train_subset, val_subset, test_subset)``; empty val/test splits are None.

        Raises:
            KeyError: If a requested group name does not exist.
        """
        group_map = self.groups(by=by)

        def _collect(names: Iterable[str] | None) -> list[int]:
            """Return the sorted sample indices of the given groups.

            Args:
                names: Group names, or None for none.

            Returns:
                Sorted sample indices.

            Raises:
                KeyError: If a group name does not exist.
            """
            if names is None:
                return []
            indices: list[int] = []
            for name in names:
                if name not in group_map:
                    raise KeyError(
                        f"Group '{name}' not found for by='{by}'. Available: {sorted(group_map)}"
                    )
                indices.extend(group_map[name])
            return sorted(indices)

        tr = _collect(train_groups)
        va = _collect(val_groups)
        te = _collect(test_groups)
        return (
            Subset(self, tr),
            Subset(self, va) if va else None,
            Subset(self, te) if te else None,
        )
