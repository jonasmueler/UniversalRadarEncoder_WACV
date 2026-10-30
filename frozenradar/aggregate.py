"""Rebuild the result tables from the per-seed result records."""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEEDS = (0, 1, 2)
LOSO_FOLDS = ("0010", "0014", "0029", "0030")


def load_records(results_dir: Path) -> list[dict]:
    """Read every ``<model>/<benchmark>.json`` record file below a directory.

    Args:
        results_dir: Directory holding one sub-directory per model.

    Returns:
        All per-seed records.
    """
    records = []
    for path in sorted(results_dir.glob("*/*.json")):
        records.extend(json.loads(path.read_text()))
    return records


def _per_seed(records: list[dict], cell: dict, model: str, variant: str) -> list[float]:
    """Collect one metric per seed for a table cell.

    Args:
        records: Per-seed records.
        cell: Cell spec with ``benchmark``, ``split``, ``fold`` and ``metric``.
        model: Model name of the row.
        variant: Variant of the row.

    Returns:
        One value per seed in ``SEEDS`` that has a record (fold mean for ``fold="mean"``).
    """
    folds = LOSO_FOLDS if cell["fold"] == "mean" else (cell["fold"],)
    index = {
        (r["fold"], r["seed"]): r[cell["metric"]] for r in records
        if (r["model"], r["variant"], r["benchmark"], r["split"])
        == (model, variant, cell["benchmark"], cell["split"])
    }
    values = []
    for seed in SEEDS:
        per_fold = [index[(f, seed)] for f in folds if (f, seed) in index]
        if len(per_fold) == len(folds):
            values.append(statistics.mean(per_fold))
    return values


def cell_value(
    records: list[dict], cell: dict, model: str, variant: str,
) -> tuple[float, float] | None:
    """Compute one table cell as (mean, std) over seeds.

    Args:
        records: Per-seed records.
        cell: Cell spec.
        model: Model name of the row.
        variant: Variant of the row.

    Returns:
        ``(mean, std)`` or ``None`` if a seed is missing.
    """
    values = _per_seed(records, cell, model, variant)
    if len(values) != len(SEEDS):
        return None
    return statistics.mean(values), statistics.stdev(values)


def compare(records: list[dict], tables: dict) -> tuple[list[str], int]:
    """Format every table cell next to its printed paper value.

    Args:
        records: Per-seed records.
        tables: Paper tables from ``results/paper_tables.json``.

    Returns:
        Report lines and the number of cells that differ at three decimals. A cell
        whose unrounded mean or std lies on a rounding boundary (within 1e-9 of
        ``x.xxx5``) and prints one unit away from the paper is reported as ``tie``
        and not counted: the logs carry three decimals, so such a cell can round
        either way.
    """
    lines, mismatches = [], 0
    for label, rows in tables.items():
        lines.append(f"\n{label}")
        for row in rows:
            for cell in row["cells"]:
                paper = "{:.3f}±{:.3f}".format(*cell["paper"])
                got = cell_value(records, cell, row["model"], row["variant"])
                ours = "missing" if got is None else "{:.3f}±{:.3f}".format(*got)
                ok = ours == paper
                tie = not ok and got is not None and _rounding_tie(got, cell["paper"])
                mismatches += not (ok or tie)
                where = cell["benchmark"] + (f" fold {cell['fold']}" if cell["fold"] else "")
                lines.append(
                    f"  {'ok ' if ok else 'tie ' if tie else 'DIFF'} {row['model']:15s} "
                    f"{row['variant']:16s} {where:28s} {cell['split']:9s} {cell['metric']:4s} "
                    f"paper {paper}  records {ours}"
                )
    return lines, mismatches


def _rounding_tie(got: tuple[float, float], paper: list[float], eps: float = 1e-9) -> bool:
    """Whether ``got`` differs from the printed value only by a half-unit rounding tie.

    Args:
        got: Unrounded ``(mean, std)`` from the records.
        paper: Printed ``[mean, std]``.
        eps: Tolerance around the ``x.xxx5`` boundary.

    Returns:
        ``True`` if every component either rounds to the paper value or sits on a
        boundary adjacent to it.
    """
    for value, printed in zip(got, paper):
        if round(value, 3) == printed:
            continue
        on_boundary = abs(value * 1000 - math.floor(value * 1000) - 0.5) < eps * 1000
        if not (on_boundary and abs(value - printed) < 0.0005 + eps):
            return False
    return True


def main() -> None:
    """Print every paper table cell from the records and optionally check it."""
    parser = argparse.ArgumentParser(description="Rebuild the paper tables from result records.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--records", type=Path, default=None,
        help="JSON-lines file of your own per-seed records (default: the shipped ones)",
    )
    parser.add_argument("--check", action="store_true", help="exit 1 if any cell differs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.records is None:
        records = load_records(args.results_dir)
    else:
        records = [json.loads(line) for line in args.records.read_text().splitlines() if line]
    tables = json.loads((args.results_dir / "paper_tables.json").read_text())
    lines, mismatches = compare(records, tables)
    for line in lines:
        logger.info(line)
    n_cells = sum(len(row["cells"]) for rows in tables.values() for row in rows)
    ties = sum(line.startswith("  tie ") for line in lines)
    logger.info("\n%d of %d cells match the paper at three decimals%s",
                n_cells - mismatches - ties, n_cells,
                f", {ties} on a rounding tie (see compare())" if ties else "")
    if args.check and mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
