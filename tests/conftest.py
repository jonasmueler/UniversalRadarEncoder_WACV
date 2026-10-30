"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_raw_adc_dataset(tmp_path: Path) -> Path:
    """Create a minimal synthetic mmRadPose raw-ADC directory.

    Layout (participants ``p1`` and ``p2``)::

        tmp_path/data_cube_parsed_p1_an0_ac0_r0.npz
        tmp_path/skeleton_p1_an0_ac0_r0.npy

    Args:
        tmp_path: Temporary directory.

    Returns:
        The dataset root.
    """
    rng = np.random.default_rng(42)
    T, D, A, E, R = 30, 8, 4, 4, 32
    for p in ["p1", "p2"]:
        stem = f"{p}_an0_ac0_r0"
        radar = (
            rng.standard_normal((T, D, A, E, R)) + 1j * rng.standard_normal((T, D, A, E, R))
        ).astype(np.complex64)
        gt = rng.standard_normal((T, 26, 3)).astype(np.float32)
        np.savez_compressed(tmp_path / f"data_cube_parsed_{stem}.npz", data=radar)
        np.save(tmp_path / f"skeleton_{stem}.npy", gt)
    return tmp_path
