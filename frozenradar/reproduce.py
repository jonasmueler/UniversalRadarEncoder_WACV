"""Evaluate the released encoders and heads on every benchmark and compare with the tables."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
from pathlib import Path

from frozenradar.aggregate import LOSO_FOLDS, RESULTS_DIR, SEEDS, compare, load_records

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
FINAL_RE = re.compile(r"=== FINAL (TEST|OUTOFSET):\s+acc=([\d.]+)\s+f1=([\d.]+)"
                      r"\s+prec=([\d.]+)\s+rec=([\d.]+)")
CUBE_TASKS = ("har", "hgr", "agr")
# Which benchmarks each encoder is evaluated on in the paper.
PAPER_CELLS = {
    "pool_mcag": ("cubelearn_har", "cubelearn_hgr", "cubelearn_agr", "mmradpose",
                  "ci4r_asl", "ci4r_asl_hearing", "ci4r_gait"),
    "pool_mag": ("cubelearn_har", "cubelearn_hgr", "cubelearn_agr"),
    "pool_cag": ("mmradpose",),
    "pool_mcg": ("ci4r_asl",),
    "pool_mca": ("ci4r_gait", "ci4r_gait_loso"),
}


def _commands(benchmark: str, ckpt: str, cfg: str, seed: int) -> list[tuple[list[str], str | None]]:
    """Build the evaluation command(s) for one benchmark and seed.

    Args:
        benchmark: Benchmark name.
        ckpt: Encoder checkpoint path.
        cfg: Encoder config path.
        seed: Classifier seed.

    Returns:
        ``(command, fold)`` pairs; LOSO yields one per fold.
    """
    s = str(seed)
    if benchmark.startswith("cubelearn_"):
        task = benchmark.split("_", 1)[1]
        return [(["scripts/eval_cubelearn.sh", ckpt, cfg, task, s], None)]
    if benchmark == "ci4r_gait_loso":
        return [(["scripts/eval_ci4r_gait_loso.sh", ckpt, cfg, s, f], f) for f in LOSO_FOLDS]
    script = {
        "mmradpose": ["scripts/eval_mmradpose.sh", ckpt, cfg, s],
        "ci4r_asl": ["scripts/eval_ci4r_asl.sh", ckpt, cfg, s],
        "ci4r_asl_hearing": ["scripts/eval_ci4r_asl.sh", ckpt, cfg, s, "hearing"],
        "ci4r_gait": ["scripts/eval_ci4r_gait.sh", ckpt, cfg, s],
    }[benchmark]
    return [(script, None)]


def _run(command: list[str]) -> list[tuple[str, float, float, float, float]]:
    """Run one evaluation script in eval-only mode and parse its FINAL lines.

    Args:
        command: Script and arguments.

    Returns:
        ``(split, acc, f1, precision, recall)`` per FINAL line.

    Raises:
        subprocess.CalledProcessError: If the script fails.
    """
    env = {**os.environ, "FR_EVAL_ONLY": "1"}
    logger.info("$ %s", " ".join(command))
    proc = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).splitlines()[-40:]
        logger.error("command failed (exit %d), last output:\n%s", proc.returncode,
                     "\n".join(tail))
        raise subprocess.CalledProcessError(proc.returncode, command)
    out = proc.stdout
    return [(m.group(1).lower(), *(float(m.group(i)) for i in range(2, 6)))
            for m in FINAL_RE.finditer(out)]


def _key(record: dict) -> tuple:
    """Identify a per-seed record independently of where it came from.

    Args:
        record: Per-seed result record.

    Returns:
        ``(model, variant, benchmark, split, fold, seed)``.
    """
    return tuple(record[k] for k in ("model", "variant", "benchmark", "split", "fold", "seed"))


def main() -> None:
    """Evaluate the released heads and compare every reproduced cell with the paper."""
    parser = argparse.ArgumentParser(description="Reproduce our cells from released weights.")
    parser.add_argument("--weights", type=Path, default=REPO / "weights",
                        help="download_weights target directory")
    parser.add_argument("--pools", nargs="+", default=list(PAPER_CELLS))
    parser.add_argument("--benchmarks", nargs="+", default=None, help="subset of benchmarks")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--out", type=Path, default=REPO / "reproduced" / "ours.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    records = []
    for pool in args.pools:
        ckpt = str(args.weights / "ours" / pool / "best.pt")
        cfg = f"configs/pretrain/{pool}.yaml"
        for benchmark in PAPER_CELLS[pool]:
            if args.benchmarks and benchmark not in args.benchmarks:
                continue
            for seed in args.seeds:
                for command, fold in _commands(benchmark, ckpt, cfg, seed):
                    for split, acc, f1, prec, rec in _run(command):
                        records.append({
                            "model": "ours", "variant": pool, "benchmark": benchmark,
                            "split": split, "fold": fold, "seed": seed, "acc": acc,
                            "f1": f1, "precision": prec, "recall": rec,
                        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in records))
    logger.info("%d records -> %s", len(records), args.out)

    shipped = {_key(r): r for r in load_records(RESULTS_DIR)}
    per_seed_diff = 0
    for r in records:
        ref = shipped.get(_key(r))
        same = ref is not None and all(f"{r[m]:.3f}" == f"{ref[m]:.3f}" for m in ("acc", "f1"))
        per_seed_diff += not same
        if not same:
            logger.info("seed differs from the logged run: %s -> %s", _key(r),
                        None if ref is None else (ref["acc"], ref["f1"]))
    logger.info("%d of %d reproduced seeds match the logged runs at three decimals",
                len(records) - per_seed_diff, len(records))

    tables = json.loads((RESULTS_DIR / "paper_tables.json").read_text())
    ran = {(r["variant"], r["benchmark"]) for r in records}
    ours = {
        label: [
            {**row, "cells": [c for c in row["cells"]
                              if (row["variant"], c["benchmark"]) in ran]}
            for row in rows if row["model"] == "ours"
        ]
        for label, rows in tables.items()
    }
    lines, mismatches = compare(records, ours)
    for line in lines:
        logger.info(line)
    logger.info("\n%d reproduced cells differ from the paper at three decimals", mismatches)


if __name__ == "__main__":
    main()
