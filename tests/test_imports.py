"""Import every documented entry point (catches circular imports and broken references)."""

from __future__ import annotations

import importlib

import pytest

ENTRY_POINTS = [
    "frozenradar.train",
    "frozenradar.extract_embeddings",
    "frozenradar.extract_embeddings_cubelearn",
    "frozenradar.extract_embeddings_ci4r",
    "frozenradar.extract_embeddings_multi3",
    "frozenradar.classification.train",
    "frozenradar.classification.build_sequences",
    "frozenradar.classification.build_sequences_cubelearn",
    "frozenradar.classification.build_sequences_ci4r",
    "frozenradar.classification.build_sequences_multi3",
    "frozenradar.classification.build_sequences_multi3_loso",
    "frozenradar.data.ci4r_splits",
    "frozenradar.data.extract_ci4r_reps",
    "frozenradar.data.verify_pretrain_stores",
    "frozenradar.data.download_cubelearn",
    "frozenradar.data.download_ci4r_asl",
    "frozenradar.data.download_ci4r_gait",
    "frozenradar.aggregate",
    "frozenradar.aggregate_seeds",
    "frozenradar.reproduce",
    "frozenradar.download_weights",
]

@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_entry_point_imports(module: str) -> None:
    """The module imports without error.

    Args:
        module: Dotted module name.
    """
    importlib.import_module(module)
