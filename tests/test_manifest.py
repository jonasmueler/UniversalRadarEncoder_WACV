"""The weights manifest holds exactly one released head per reported record."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "weights_manifest.json").read_text())
POOLS = ("pool_mcag", "pool_cag", "pool_mag", "pool_mcg", "pool_mca")


def _head_dir(record: dict) -> str:
    """Directory of the head that produced one of our result records.

    Args:
        record: One entry of ``results/ours/<benchmark>.json``.

    Returns:
        The head directory as laid out by ``scripts/eval_*.sh``.
    """
    fold = f"_f{record['fold']}" if record["fold"] else ""
    return (f"ours/{record['variant']}/classification/"
            f"{record['benchmark']}{fold}_s{record['seed']}")


def test_only_our_weights_are_listed() -> None:
    """Every manifest entry is one of our encoders or heads."""
    assert {e["model"] for e in MANIFEST["files"]} == {"ours"}


def test_entries_are_well_formed() -> None:
    """Drive ids, checksums and sizes are present and plausible, paths unique."""
    files = [e["file"] for e in MANIFEST["files"]]
    assert len(files) == len(set(files))
    for e in MANIFEST["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", e["sha256"]), e["file"]
        assert re.fullmatch(r"[A-Za-z0-9_-]{20,}", e["id"]), e["file"]
        assert e["bytes"] > 0, e["file"]


def test_encoders_are_complete() -> None:
    """Each of the five encoders ships its checkpoint and config."""
    files = {e["file"] for e in MANIFEST["files"]}
    for pool in POOLS:
        assert {f"ours/{pool}/best.pt", f"ours/{pool}/config.json"} <= files, pool


def test_one_head_per_record() -> None:
    """The released heads are exactly the heads behind the shipped records."""
    heads = {str(Path(e["file"]).parent) for e in MANIFEST["files"]
             if "/classification/" in e["file"]}
    expected = set()
    for path in (ROOT / "results" / "ours").glob("*.json"):
        expected |= {_head_dir(r) for r in json.loads(path.read_text())}
    assert heads == expected
    for head in heads:
        names = {Path(e["file"]).name for e in MANIFEST["files"]
                 if str(Path(e["file"]).parent) == head}
        assert names == {"best.pt", "config.json"}, head
