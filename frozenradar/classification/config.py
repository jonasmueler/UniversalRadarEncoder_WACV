"""Configuration of the temporal classification head (YAML plus CLI overrides)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Args:
        base: Base dictionary to merge into.
        override: Dictionary whose values take precedence.

    Returns:
        New dict with values from *override* merged on top of *base*.
    """
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _cast_value(v: str) -> Any:
    """Cast a string CLI value to the appropriate Python type.

    Args:
        v: Raw string value from the command line.

    Returns:
        Value cast to bool, None, int, float, or left as str.
    """
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() in ("null", "none"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


@dataclass(frozen=True)
class DatasetConfig:
    """Sequence-cache settings and temporal augmentation.

    Attributes:
        data_root: mmRadPose raw ADC root (not read by the head).
        sequences_dir: Sequence-cache root with ``train/``, ``val/``, ``test/``.
        crop_ratio_min: Minimum fraction of a sequence kept by random cropping.
        crop_ratio_max: Maximum fraction (1.0 = no cropping).
        time_mask_ratio: Fraction of frames zero-masked during training.
        outofset_test_dir: Optional extra test cache (CubeLearn out-of-set
            users 6-7) evaluated with the best checkpoint.
    """

    data_root: str = "~/data/mmRadPose_rawdata"
    sequences_dir: str = "cache/sequences"
    crop_ratio_min: float = 0.9
    crop_ratio_max: float = 1.0
    time_mask_ratio: float = 0.1
    outofset_test_dir: str = ""


@dataclass(frozen=True)
class ModelConfig:
    """Classifier architecture.

    Attributes:
        model_type: Head type; only ``"transformer"`` is supported.
        embed_dim: VAE latent dimension.
        temporal_layers: Depth of the temporal transformer.
        max_seq_len: Maximum sequence length (frames).
        num_heads: Number of attention heads.
        ff_ratio: Feed-forward expansion ratio.
        dropout: Dropout rate.
        num_classes: Number of classes.
        alpha_schedule: Posterior-noise schedule; only ``"fixed"`` is supported.
        alpha_max: Posterior-noise scale during training (0.0 = mean only).
    """

    model_type: str = "transformer"
    embed_dim: int = 256
    temporal_layers: int = 4
    max_seq_len: int = 1024
    num_heads: int = 8
    ff_ratio: float = 4.0
    dropout: float = 0.1
    num_classes: int = 12
    alpha_schedule: str = "fixed"
    alpha_max: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters.

    Attributes:
        epochs: Total training epochs.
        lr: Peak learning rate.
        final_lr: Final learning rate after cosine decay.
        warmup_epochs: Number of warmup epochs.
        weight_decay: AdamW weight decay.
        batch_size: Training batch size.
        num_workers: DataLoader workers.
        use_bfloat16: Whether to use bfloat16 mixed precision.
        grad_clip_norm: Max gradient norm (0 = disabled).
        label_smoothing: Label smoothing factor for cross-entropy.
        seed: Random seed.
        device: Torch device string.
        checkpoint_dir: Directory for saving checkpoints.
        log_every_n_steps: Logging frequency.
        val_every_n_epochs: Validation frequency.
        eval_only: Skip training and evaluate ``<checkpoint_dir>/best.pt``
            (a released or already trained head) on the test splits.
    """

    epochs: int = 100
    lr: float = 1e-4
    final_lr: float = 1e-6
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    batch_size: int = 16
    num_workers: int = 4
    use_bfloat16: bool = True
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.1
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "runs/classification"
    log_every_n_steps: int = 20
    val_every_n_epochs: int = 1
    eval_only: bool = False


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases logging configuration.

    Attributes:
        enabled: Whether to enable W&B logging.
        project: W&B project name.
        entity: W&B entity (team or user).
        run_name: Optional run name.
    """

    enabled: bool = False
    project: str = "vae-activity-classification"
    entity: str | None = None
    run_name: str | None = None


@dataclass(frozen=True)
class ClassificationConfig:
    """Top-level configuration of the classification head.

    Attributes:
        dataset: Dataset settings.
        model: Model architecture settings.
        training: Training hyperparameters.
        wandb: W&B logging settings.
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


_SECTION_MAP: dict[str, type] = {
    "dataset": DatasetConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "wandb": WandbConfig,
}


def load_config(
    path: str,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> ClassificationConfig:
    """Load the classification config from YAML with optional overrides.

    A ``_base_`` key names a YAML file (relative to this one) whose values the
    current file overrides; CLI overrides are applied last.

    Args:
        path: Path to the YAML config file.
        overrides: Nested dict of ``{section: {key: value}}`` overrides.

    Returns:
        Fully resolved ``ClassificationConfig``.
    """
    cfg_path = Path(path)
    with open(cfg_path) as f:
        raw = yaml.safe_load(f) or {}

    if "_base_" in raw:
        base_path = cfg_path.parent / raw.pop("_base_")
        with open(base_path) as f:
            base = yaml.safe_load(f) or {}
        raw = _deep_merge(base, raw)

    if overrides:
        raw = _deep_merge(raw, overrides)

    sections = {}
    for name, cls in _SECTION_MAP.items():
        section_raw = raw.get(name, {})
        cast = {
            k: _cast_value(v) if isinstance(v, str) else v
            for k, v in section_raw.items()
        }
        sections[name] = cls(**cast)

    return ClassificationConfig(**sections)


def parse_overrides(args: list[str]) -> dict[str, dict[str, str]]:
    """Parse CLI override arguments into a nested dict.

    Tokens without a dot or without a following value are skipped.

    Args:
        args: List of ``["--section.key", "value", ...]`` strings.

    Returns:
        Nested dict ``{section: {key: value}}``.
    """
    overrides: dict[str, dict[str, str]] = {}
    i = 0
    while i < len(args):
        key = args[i].lstrip("-")
        if "." not in key or i + 1 >= len(args):
            i += 1
            continue
        section, param = key.split(".", 1)
        value = args[i + 1]
        overrides.setdefault(section, {})[param] = value
        i += 2
    return overrides
