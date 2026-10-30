"""Aggregate the FINAL scores of run logs into mean ± std tables over seeds."""

from __future__ import annotations

import argparse
import logging
import re
import statistics
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_FINAL_RE = re.compile(
    r"=== FINAL ([A-Z][A-Z \-]*?):\s+acc=([\d.]+)\s+f1=([\d.]+)"
    r"(?:\s+prec=([\d.]+)\s+rec=([\d.]+))?\s*==="
)
_RUN_MARKER_RE = re.compile(r"=== RUN (\S+) ===")
_SEED_SUFFIX_RE = re.compile(r"_s(\d+)$")

METRICS = ["acc", "f1", "prec", "rec"]


def collect(log_dirs: list[Path]) -> dict[tuple[str, str], dict[int, dict[str, float]]]:
    """Collect per-(condition, split) metrics keyed by seed.

    Args:
        log_dirs: Directories containing ``*.log`` files.

    Returns:
        Mapping ``(condition, split) -> {seed -> {metric -> value}}``. When the
        same (condition, split, seed) appears in several logs, the last one by
        file modification time wins.
    """
    results: dict[tuple[str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    log_files: list[Path] = []
    for d in log_dirs:
        log_files.extend(sorted(Path(d).glob("*.log")))
    log_files.sort(key=lambda p: p.stat().st_mtime)

    for log in log_files:
        current = log.stem
        for line in log.read_text(errors="ignore").splitlines():
            marker = _RUN_MARKER_RE.search(line)
            if marker:
                current = marker.group(1)
                continue
            final = _FINAL_RE.search(line)
            if not final or final.group(4) is None:
                continue
            seed_match = _SEED_SUFFIX_RE.search(current)
            if not seed_match:
                continue
            seed = int(seed_match.group(1))
            condition = _SEED_SUFFIX_RE.sub("", current)
            split = final.group(1).strip()
            results[(condition, split)][seed] = {
                "acc": float(final.group(2)),
                "f1": float(final.group(3)),
                "prec": float(final.group(4)),
                "rec": float(final.group(5)),
            }
    return results


def _fmt(values: list[float], latex: bool) -> str:
    """Format mean±std of a metric at three decimals.

    Args:
        values: One value per seed.
        latex: Use the LaTeX plus-minus macro instead of ``±``.

    Returns:
        The formatted cell.
    """
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    sep = r"$\pm$" if latex else "±"
    return f"{mean:.3f}{sep}{std:.3f}"


def main() -> None:
    """Aggregate and print the mean±std table.

    Raises:
        SystemExit: If no FINAL line is found or a condition misses seeds.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = collect(args.log_dirs)
    if not results:
        raise SystemExit("No parseable FINAL lines with prec/rec found.")

    incomplete = []
    for (condition, split), by_seed in sorted(results.items()):
        seeds = sorted(by_seed)
        cells = []
        for metric in METRICS:
            cells.append(f"{metric}={_fmt([by_seed[s][metric] for s in seeds], args.latex)}")
        flag = ""
        if len(seeds) != args.expected_seeds:
            flag = f"  [INCOMPLETE: seeds {seeds}]"
            incomplete.append((condition, split, seeds))
        logger.info("%-45s %-14s n=%d  %s%s", condition, split, len(seeds), "  ".join(cells), flag)

    if incomplete:
        logger.info("")
        for condition, split, seeds in incomplete:
            logger.info("INCOMPLETE %s / %s: seeds %s", condition, split, seeds)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
