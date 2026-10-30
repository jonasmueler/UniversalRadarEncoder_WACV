"""The shipped per-seed records reproduce every printed cell of our encoder."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest

from frozenradar.aggregate import RESULTS_DIR, compare, load_records

POOLS = {"pool_mcag", "pool_cag", "pool_mag", "pool_mcg", "pool_mca"}


@pytest.fixture(scope="module")
def records() -> list[dict[str, Any]]:
    """Load every shipped result record.

    Returns:
        The records of ``results/``.
    """
    return load_records(RESULTS_DIR)


def test_only_our_encoders(records: list[dict[str, Any]]) -> None:
    """Every record is one of our five encoders with seed 0, 1 or 2."""
    assert {r["model"] for r in records} == {"ours"}
    assert {r["variant"] for r in records} == POOLS
    assert {r["seed"] for r in records} == {0, 1, 2}
    keys = Counter((r["variant"], r["benchmark"], r["split"], r["fold"], r["seed"])
                   for r in records)
    assert max(keys.values()) == 1


def test_tables_reproduce(records: list[dict[str, Any]]) -> None:
    """All 43 printed cells are recomputed from the records without a difference or tie."""
    tables = json.loads((RESULTS_DIR / "paper_tables.json").read_text())
    lines, mismatches = compare(records, tables)
    assert sum(len(row["cells"]) for rows in tables.values() for row in rows) == 43
    assert mismatches == 0, [line for line in lines if "DIFF" in line][:5]
    assert not [line for line in lines if line.lstrip().startswith("tie")]
