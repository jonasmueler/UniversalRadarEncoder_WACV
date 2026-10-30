"""Concatenation of several single-frame radar datasets for multi-source pretraining."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class MultiRadarDataset(Dataset):
    """Concatenate several radar-frame datasets into one unsupervised dataset.

    Each source dataset must return a single complex radar frame ``(D, A, E, R)``
    of the same shape, so frames from different sensors batch together. The
    per-index source id is exposed so a source-balanced sampler can equalise
    the mixture regardless of the sources' sizes.

    Args:
        sources: List of ``(name, dataset)`` pairs.

    Raises:
        ValueError: If ``sources`` is empty.
    """

    def __init__(self, sources: list[tuple[str, Dataset]]) -> None:
        if not sources:
            raise ValueError("MultiRadarDataset requires at least one source")
        self._names = [name for name, _ in sources]
        self._datasets = [ds for _, ds in sources]
        self._lengths = [len(ds) for ds in self._datasets]

        self._offsets = [0]
        for length in self._lengths:
            self._offsets.append(self._offsets[-1] + length)

        source_ids: list[int] = []
        for src, length in enumerate(self._lengths):
            source_ids.extend([src] * length)
        self._source_ids = torch.tensor(source_ids, dtype=torch.long)

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return frame ``idx`` of the concatenation.

        Args:
            idx: Global frame index.

        Returns:
            The source dataset's frame.
        """
        src = int(self._source_ids[idx])
        local = idx - self._offsets[src]
        return self._datasets[src][local]

    @property
    def source_names(self) -> list[str]:
        """Ordered source names."""
        return list(self._names)

    @property
    def source_lengths(self) -> dict[str, int]:
        """Frame count per source name."""
        return dict(zip(self._names, self._lengths))

    @property
    def source_ids(self) -> torch.Tensor:
        """Per-index source id (index into :attr:`source_names`)."""
        return self._source_ids
