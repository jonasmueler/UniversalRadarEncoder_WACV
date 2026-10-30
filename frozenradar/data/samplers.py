"""Source-balanced sampling over a :class:`MultiRadarDataset`."""

from __future__ import annotations

import torch
from torch.utils.data import WeightedRandomSampler

from .multi import MultiRadarDataset


def build_balanced_sampler(
    dataset: MultiRadarDataset,
    weights: dict[str, float] | None = None,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Sampler that draws sources with balanced (or configured) probability.

    Each frame is weighted by ``w_source / len(source)`` so that, per epoch,
    each source is drawn with probability proportional to ``w_source``
    regardless of its frame count (equal by default).

    Args:
        dataset: The combined multi-source dataset.
        weights: Optional per-source weight keyed by source name (default equal).
        num_samples: Draws per epoch (default: ``len(dataset)``).

    Returns:
        A ``WeightedRandomSampler`` (with replacement).

    Raises:
        ValueError: If a source has a zero or negative weight.
    """
    names = dataset.source_names
    lengths = dataset.source_lengths
    w = weights if weights is not None else {name: 1.0 for name in names}

    missing = [name for name in names if w.get(name, 0.0) <= 0.0]
    if missing:
        raise ValueError(
            f"sources {missing} have zero/negative sampling weight and would never be "
            "drawn; give every source a positive weight or drop it from dataset.sources"
        )

    per_source = torch.tensor(
        [w[name] / max(lengths[name], 1) for name in names],
        dtype=torch.double,
    )
    sample_weights = per_source[dataset.source_ids]
    n = num_samples if num_samples is not None else len(dataset)
    return WeightedRandomSampler(sample_weights, num_samples=n, replacement=True)
