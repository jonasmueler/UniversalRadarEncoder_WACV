"""Checks that the repository states exactly what the paper states.

The table cells themselves are checked against the per-seed records by
``python -m frozenradar.aggregate --check`` (see ``test_results.py``). This
module checks everything around them: the recipe and split values given in the
text, the five pretraining pools, and every number the prose states about our
encoder.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from frozenradar.data.splits import (
    CUBELEARN_OUTOFSET_USERS,
    CUBELEARN_USERS,
    MMRADPOSE_PARTICIPANTS,
)

ROOT = Path(__file__).resolve().parents[1]
PRETRAIN = ROOT / "configs" / "pretrain"
DOWNSTREAM = ROOT / "configs" / "downstream"
SCRIPTS = ROOT / "scripts"
SPLITS = ROOT / "splits"
PAPER = json.loads((ROOT / "results" / "paper_tables.json").read_text())

POOLS = {
    "pool_mcag": {"mmradpose": 1 / 4, "cubelearn": 1 / 12, "cubelearn_hgr": 1 / 12,
                  "cubelearn_agr": 1 / 12, "ci4r": 1 / 4, "multi3": 1 / 4},
    "pool_cag": {"cubelearn": 1 / 9, "cubelearn_hgr": 1 / 9, "cubelearn_agr": 1 / 9,
                 "ci4r": 1 / 3, "multi3": 1 / 3},
    "pool_mag": {"mmradpose": 1 / 3, "ci4r": 1 / 3, "multi3": 1 / 3},
    "pool_mcg": {"mmradpose": 1 / 3, "cubelearn": 1 / 9, "cubelearn_hgr": 1 / 9,
                 "cubelearn_agr": 1 / 9, "multi3": 1 / 3},
    "pool_mca": {"mmradpose": 1 / 3, "cubelearn": 1 / 9, "cubelearn_hgr": 1 / 9,
                 "cubelearn_agr": 1 / 9, "ci4r": 1 / 3},
}
HELD_OUT_POOL = {
    "mmradpose": "pool_cag",
    "cubelearn_har": "pool_mag",
    "cubelearn_hgr": "pool_mag",
    "cubelearn_agr": "pool_mag",
    "ci4r_asl": "pool_mcg",
    "ci4r_gait": "pool_mca",
    "ci4r_gait_loso": "pool_mca",
}


def _yaml(path: Path) -> dict:
    """Load one YAML file.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.
    """
    return yaml.safe_load(path.read_text())


def _cell(model: str, benchmark: str, split: str = "test", metric: str = "acc",
          fold: str | None = None, variant: str | None = None) -> float:
    """Return one printed table value (mean over seeds) from ``paper_tables.json``.

    Args:
        model: Row model key (``ours``).
        benchmark: Benchmark key of the cell.
        split: ``test`` (in-set) or ``outofset``.
        metric: ``acc`` or ``f1``.
        fold: LOSO fold subject or ``mean``; ``None`` elsewhere.
        variant: Encoder pool of the row.

    Returns:
        The printed mean.

    Raises:
        KeyError: If the cell is not in the paper tables.
    """
    for rows in PAPER.values():
        for row in rows:
            if row["model"] != model or (variant and row["variant"] != variant):
                continue
            for c in row["cells"]:
                if (c["benchmark"], c["split"], c["metric"], c["fold"]) == (
                        benchmark, split, metric, fold):
                    return c["paper"][0]
    raise KeyError((model, variant, benchmark, split, metric, fold))


def _ours(benchmark: str, held_out: bool = False, **kw: str) -> float:
    """Printed value of our all-sensor or sensor-held-out encoder.

    Args:
        benchmark: Benchmark key of the cell.
        held_out: Use the leave-one-sensor-out encoder of this benchmark.
        **kw: Passed to :func:`_cell` (split, metric, fold).

    Returns:
        The printed mean.
    """
    pool = HELD_OUT_POOL[benchmark] if held_out else "pool_mcag"
    return _cell("ours", benchmark, variant=pool, **kw)


def _points(a: float, b: float) -> float:
    """Difference of two accuracies in percentage points, rounded as in the text.

    Args:
        a: Larger value.
        b: Smaller value.

    Returns:
        ``100 * (a - b)`` rounded to one decimal.
    """
    return round(100 * (a - b) + 1e-9, 1)


CUBE = ("cubelearn_har", "cubelearn_hgr", "cubelearn_agr")


class TestRecipe:
    """Pretraining and head settings stated in Sec. 3 and 4."""

    def test_pretraining_recipe(self) -> None:
        """Grid, preprocessing, model and optimisation as stated in the paper."""
        cfg = _yaml(PRETRAIN / "pool_base.yaml")
        d, m, t = cfg["dataset"], cfg["model"], cfg["training"]
        assert d["dims"] == [128, 4, 4, 256] and d["axes"] == ["D", "A", "E", "R"]
        assert d["clutter_removal"] and d["apply_fft"] and d["normalize_magnitude"]
        assert d["format_mode"] == "iq" and m["n_input_channels"] == 2
        assert m["encoder_type"] == "hrnet_multi_phase" and m["backbone_type"] == "hrnet"
        assert m["latent_dim"] == 256 and m["phase_transformer_layers"] == 2
        assert m["projection_set"] is None and m["multi_projection_weights"] is None
        assert m["reconstruction_target"] == "projections"
        assert (t["beta"], t["beta_warmup_epochs"]) == (0.1, 5)
        assert (t["epochs"], t["batch_size"]) == (50, 16)
        assert (t["lr_scheduler"], t["grad_clip_norm"], t["loss_type"]) == ("cosine", 1.0, "mse")

    def test_pools(self) -> None:
        """The five encoders and their mixture weights (Sec. 4.2)."""
        for pool, weights in POOLS.items():
            cfg = _yaml(PRETRAIN / f"{pool}.yaml")
            assert cfg["_base_"] == "pool_base.yaml"
            got = dict(zip(cfg["dataset"]["sources"], cfg["dataset"]["mixture_weights"],
                           strict=True))
            assert got.keys() == weights.keys(), pool
            for src, w in weights.items():
                assert got[src] == pytest.approx(w, abs=2e-6), (pool, src)

    def test_held_out_pools_exclude_their_sensor(self) -> None:
        """Every sensor-held-out row comes from a pool without that sensor."""
        sensor_sources = {"mmradpose": {"mmradpose"},
                          "cubelearn": {"cubelearn", "cubelearn_hgr", "cubelearn_agr"},
                          "ci4r_asl": {"ci4r"}, "ci4r_gait": {"multi3"}}
        for bench, pool in HELD_OUT_POOL.items():
            sensor = next(s for s in sensor_sources if bench.startswith(s))
            assert not sensor_sources[sensor] & set(POOLS[pool]), (bench, pool)

    def test_head(self) -> None:
        """Head: 1 layer on mmRadPose and CubeLearn, 4 on CI4R, dropout 0.3 throughout."""
        base = _yaml(DOWNSTREAM / "base.yaml")
        assert base["model"]["temporal_layers"] == 1 and base["model"]["dropout"] == 0.3
        assert base["training"]["label_smoothing"] == 0.1
        assert "dropout" not in _yaml(DOWNSTREAM / "cubelearn.yaml")["model"]
        layers = {"mmradpose": 1, "cubelearn": 1, "ci4r_asl": 4, "ci4r_gait": 4,
                  "ci4r_gait_loso": 4}
        for name, n in layers.items():
            text = (SCRIPTS / f"eval_{name}.sh").read_text()
            assert "--model.dropout" not in text
            assert "--training.epochs 200" in text
            assert "--model.alpha_schedule fixed --model.alpha_max 0.0" in text
            found = re.search(r"--model\.temporal_layers (\d+)", text)
            assert (int(found.group(1)) if found else 1) == n, name

    def test_three_seeds_per_cell(self) -> None:
        """Every record file holds seeds 0, 1 and 2 for each of our cells."""
        for path in (ROOT / "results" / "ours").glob("*.json"):
            seeds = Counter((r["variant"], r["split"], r["fold"], r["seed"])
                            for r in json.loads(path.read_text()))
            assert {k[3] for k in seeds} == {0, 1, 2}, path.name


class TestSplits:
    """Benchmark splits stated in Sec. 4.3."""

    def test_mmradpose(self) -> None:
        """Participant-disjoint split p4-p11 / p3 / p1, p2, p12."""
        assert MMRADPOSE_PARTICIPANTS == {
            "train": [f"p{i}" for i in range(4, 12)], "val": ["p3"],
            "test": ["p1", "p2", "p12"]}

    def test_cubelearn(self) -> None:
        """Users 0-5 in-set with 15/5/10 samples, users 6-7 out-of-set."""
        assert CUBELEARN_USERS == [0, 1, 2, 3, 4, 5]
        assert CUBELEARN_OUTOFSET_USERS == [6, 7]
        sizes = {f: len((SPLITS / "cubelearn" / f).read_text().split())
                 for f in ("train_samples.csv", "validation_samples.csv",
                           "test_samples.csv")}
        assert sizes == {"train_samples.csv": 15, "validation_samples.csv": 5,
                         "test_samples.csv": 10}

    def test_ci4r_asl(self) -> None:
        """1496 repetitions, 1047/150/299, pinned digest."""
        man = json.loads((SPLITS / "ci4r_asl_manifest.json").read_text())
        assert man["n_units"] == 1496 and man["order_digest"] == "d684d5a9125a4bd1"
        assert man["counts"] == {"train": 1047, "val": 150, "test": 299}
        assert Counter(e["split"] for e in man["entries"]) == Counter(man["counts"])

    def test_ci4r_gait(self) -> None:
        """652 recordings, 457/65/130, pinned digest, LOSO fold sizes 116/91/221/214."""
        man = json.loads((SPLITS / "ci4r_gait_manifest.json").read_text())
        assert man["n_units"] == 652 and man["order_digest"] == "a8d6e1becf5193ac"
        assert man["counts"] == {"train": 457, "val": 65, "test": 130}
        subjects = Counter(e["base"][4:8] for e in man["entries"])
        assert subjects == {"0010": 116, "0014": 91, "0029": 221, "0030": 214, "0000": 10}


class TestProse:
    """Numbers about our encoder in the abstract, Results and Conclusion."""

    def test_no_domain_shift(self) -> None:
        """Sec. 6.1: in-set accuracy and the subject-mixed CI4R protocols."""
        assert _ours("cubelearn_hgr") == 0.980
        for bench in ("cubelearn_har", "cubelearn_agr"):
            assert _ours(bench) >= 0.99
        assert (_ours("ci4r_asl"), _ours("ci4r_gait")) == (0.971, 0.951)
        assert _cell("ours", "ci4r_asl_hearing", variant="pool_mcag") == 0.982

    def test_unseen_subjects(self) -> None:
        """Sec. 6.2: unseen users and participants."""
        assert _ours("cubelearn_har", split="outofset") == 0.930
        assert _ours("cubelearn_hgr", split="outofset") == 0.825
        assert _ours("cubelearn_agr", split="outofset") == 0.905
        assert _ours("mmradpose") == 0.880

    def test_in_set_to_out_of_set_drops(self) -> None:
        """Sec. 6.2: our encoder loses 6.5 to 15.5 points from in-set to out-of-set."""
        drops = [_points(_ours(b), _ours(b, split="outofset")) for b in CUBE]
        assert (min(drops), max(drops)) == (6.5, 15.5)

    def test_unseen_sensors(self) -> None:
        """Sec. 6.3 and abstract: holding the sensor out costs at most 5.3 points."""
        cells = [(b, s) for b in CUBE for s in ("test", "outofset")] + [
            ("mmradpose", "test"), ("ci4r_asl", "test"), ("ci4r_gait", "test")]
        cost = {c: _points(_ours(c[0], split=c[1]), _ours(c[0], True, split=c[1]))
                for c in cells}
        assert max(cost.values()) == 5.3 and max(cost, key=cost.get) == (
            "cubelearn_har", "outofset")

    def test_held_out_encoder(self) -> None:
        """Sec. 6.3: the values quoted for the sensor-held-out encoder."""
        assert _ours("ci4r_asl", True) == 0.967 and _ours("cubelearn_agr", True) == 0.996
        assert _ours("mmradpose", True) == 0.877
        assert _ours("cubelearn_hgr", True, split="outofset") == 0.784
        assert _ours("cubelearn_har", True, split="outofset") == 0.877
        assert _ours("cubelearn_agr", True, split="outofset") == 0.956

    def test_loso(self) -> None:
        """Sec. 6.3: 0.693 on average and a drop of 20.7 points from the subject-mixed protocol."""
        mean = _ours("ci4r_gait_loso", True, fold="mean")
        assert mean == 0.693
        assert _points(_ours("ci4r_gait", True), mean) == 20.7
