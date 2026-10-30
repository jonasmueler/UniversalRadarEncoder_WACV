"""End-to-end test of the classification head: training, then eval-only re-evaluation."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
import torch

from frozenradar.classification import train as cls_train
from frozenradar.classification.config import (
    ClassificationConfig,
    load_config,
    parse_overrides,
)

EMBED = 8
N_CLASSES = 3
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "downstream" / "base.yaml"


def _write_split(split_dir: Path, n_per_class: int, seed: int) -> None:
    """Write class-separable synthetic embedding sequences in the builder's format.

    Args:
        split_dir: Split directory to create.
        n_per_class: Sequences per class.
        seed: Generator seed.
    """
    split_dir.mkdir(parents=True)
    gen = torch.Generator().manual_seed(seed)
    meta = []
    for i in range(n_per_class * N_CLASSES):
        label = i % N_CLASSES
        n_frames = 6 + i % 4
        mu = torch.randn(n_frames, EMBED, generator=gen) * 0.1
        mu[:, label] += 1.0
        torch.save({"mu": mu, "logvar": torch.zeros_like(mu)}, split_dir / f"rec_{i:04d}.pt")
        meta.append({
            "activity": f"ac{label}", "activity_idx": label, "participant": "p0",
            "angle": "an0", "recording": f"r{i}", "n_frames": n_frames,
        })
    torch.save(meta, split_dir / "meta.pt")


def _config(seq_dir: Path, ckpt_dir: Path, eval_only: bool) -> ClassificationConfig:
    """Build a small CPU config of the paper's head recipe.

    Args:
        seq_dir: Sequence-cache root.
        ckpt_dir: Head checkpoint directory.
        eval_only: Whether to skip training.

    Returns:
        The resolved config.
    """
    args = [
        "--dataset.sequences_dir", str(seq_dir), "--model.embed_dim", str(EMBED),
        "--model.num_classes", str(N_CLASSES), "--model.temporal_layers", "1",
        "--model.alpha_schedule", "fixed", "--model.alpha_max", "0.0",
        "--training.epochs", "5", "--training.warmup_epochs", "0", "--training.lr", "0.01",
        "--training.batch_size", "4", "--training.num_workers", "0",
        "--training.device", "cpu", "--training.use_bfloat16", "false",
        "--training.checkpoint_dir", str(ckpt_dir),
        "--training.eval_only", str(eval_only).lower(),
    ]
    return load_config(str(CONFIG), overrides=parse_overrides(args))


def _final_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the logged ``=== FINAL`` lines.

    Args:
        caplog: Pytest log capture.

    Returns:
        The FINAL lines in logging order.
    """
    return [r.getMessage() for r in caplog.records if re.search(r"=== FINAL", r.getMessage())]


def test_eval_only_reproduces_trained_head(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Re-evaluating the saved head reproduces the FINAL lines and keeps config.json.

    Args:
        tmp_path: Temporary directory.
        caplog: Pytest log capture.
    """
    seq_dir, ckpt_dir = tmp_path / "seq", tmp_path / "head"
    for split, n, seed in (("train", 8, 0), ("val", 3, 1), ("test", 4, 2)):
        _write_split(seq_dir / split, n, seed)

    with caplog.at_level(logging.INFO):
        cls_train.main(_config(seq_dir, ckpt_dir, eval_only=False))
    trained = _final_lines(caplog)
    config_json = (ckpt_dir / "config.json").read_text()
    caplog.clear()

    with caplog.at_level(logging.INFO):
        cls_train.main(_config(seq_dir, ckpt_dir, eval_only=True))
    assert _final_lines(caplog) == trained
    assert (ckpt_dir / "config.json").read_text() == config_json


def test_eval_only_requires_a_head(tmp_path: Path) -> None:
    """Eval-only without a ``best.pt`` fails.

    Args:
        tmp_path: Temporary directory.
    """
    seq_dir = tmp_path / "seq"
    for split, n, seed in (("train", 2, 0), ("val", 1, 1), ("test", 1, 2)):
        _write_split(seq_dir / split, n, seed)
    with pytest.raises(FileNotFoundError):
        cls_train.main(_config(seq_dir, tmp_path / "missing", eval_only=True))
