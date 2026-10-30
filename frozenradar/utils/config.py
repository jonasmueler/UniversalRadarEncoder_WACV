"""Pretraining configuration: dataclasses, YAML loading and CLI overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    """Pretraining data: sources, frame geometry, preprocessing and roots.

    Source names are listed in :mod:`frozenradar.data.builders` (``ci4r`` is
    CI4R-ASL, ``multi3`` is CI4R-Gait). The per-dataset splits are fixed in
    :mod:`frozenradar.data.splits` and :mod:`frozenradar.data.ci4r_splits`.

    Attributes:
        name: Dataset identifier; the default source when ``sources`` is empty.
        root: mmRadPose raw ADC root.
        axes: Ordered axis names after preprocessing, e.g. ``["D", "A", "E", "R"]``.
        dims: Size per axis, same order as ``axes``.
        apply_fft: Whether to apply the FFT chain during preprocessing.
        clutter_removal: Whether to subtract Doppler-mean clutter.
        format_mode: Encoder input format, ``"iq"`` or ``"phase_diff"``.
        target_range_bins: Range length all sources are aligned to (multi-source
            training requires 256).
        normalize_magnitude: Rescale every frame to unit RMS.
        sources: Pretraining sources; empty means ``[name]``.
        mixture_weights: Per-source sampling weights (default: equal).
        participants_train: mmRadPose training participants.
        participants_val: mmRadPose validation participants.
        participants_test: mmRadPose test participants.
        angles: mmRadPose angle IDs to include (``None`` = all).
        cubelearn_root: CubeLearn HAR (``activity_organized``) root.
        cubelearn_split_dir: Directory of the CubeLearn split CSVs (``None`` =
            the copy in ``splits/cubelearn``).
        cubelearn_task: CubeLearn task of the ``cubelearn`` source.
        cubelearn_frames_per_recording: Frames per CubeLearn HAR recording.
        cubelearn_hgr_root: CubeLearn hand-gesture (``hand_organized``) root.
        cubelearn_agr_root: CubeLearn arm-gesture (``arm_organized``) root.
        ci4r_root: CI4R-ASL train-only store. The benchmark splits at the
            repetition level (5 per ``.bin``), so raw recordings hold train and
            test frames and must never be used directly.
        ci4r_frames_per_recording: Optional per-repetition frame cap.
        multi3_root: CI4R-Gait train-only store (whole train recordings).
        multi3_frames_per_recording: Optional per-recording frame cap.
    """

    name: str = "mmradpose"
    root: str = "~/data/mmRadPose_rawdata"
    axes: list[str] = field(default_factory=lambda: ["D", "A", "E", "R"])
    dims: list[int] = field(default_factory=lambda: [128, 4, 4, 64])
    apply_fft: bool = True
    clutter_removal: bool = True
    format_mode: str = "iq"
    target_range_bins: int | None = None
    normalize_magnitude: bool = False
    sources: list[str] = field(default_factory=list)
    mixture_weights: list[float] | None = None
    participants_train: list[str] = field(
        default_factory=lambda: ["p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11"],
    )
    participants_val: list[str] = field(default_factory=lambda: ["p3"])
    participants_test: list[str] = field(
        default_factory=lambda: ["p1", "p2", "p12"],
    )
    angles: list[str] | None = None
    cubelearn_root: str | None = None
    cubelearn_split_dir: str | None = None
    cubelearn_task: str = "har"
    cubelearn_frames_per_recording: int = 20
    cubelearn_hgr_root: str | None = None
    cubelearn_agr_root: str | None = None
    ci4r_root: str | None = None
    ci4r_frames_per_recording: int | None = None
    multi3_root: str | None = None
    multi3_frames_per_recording: int | None = None

    def __post_init__(self) -> None:
        """Default ``sources`` to ``[name]`` when not explicitly configured."""
        if not self.sources:
            object.__setattr__(self, "sources", [self.name])

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Spatial shape of one frame as a tuple, e.g. ``(128, 4, 4, 256)``."""
        return tuple(self.dims)


@dataclass(frozen=True)
class ModelConfig:
    """Encoder and decoder architecture.

    Attributes:
        encoder_type: Encoder type; only ``"hrnet_multi_phase"`` is built.
        latent_dim: Latent dimension (``None`` = derived from the compression ratio).
        hrnet_config: Name of the HRNet backbone configuration.
        multi_decoder_hidden: Hidden width of the projection decoders.
        multi_projection_weights: Optional loss weight per projection name.
        projection_set: Subset of the six projection names to reconstruct
            (``None`` = all six); only used with ``reconstruction_target="projections"``.
        reconstruction_target: ``"projections"`` decodes 2-D log-power views;
            ``"full_cube"`` decodes the full ``(D, A, E, R)`` log-power cube.
        backbone_type: Backbone; only ``"hrnet"`` is supported.
        n_input_channels: Input channels per element (2 for IQ, 7 for phase_diff).
        phase_transformer_dim: Width of the transformer read-out on the HRNet features.
        phase_transformer_heads: Attention heads of the read-out.
        phase_transformer_layers: Layers of the read-out.
        doppler_gate: Must be ``False``; kept so released configs parse.
        doppler_se: Must be ``False``; kept so released configs parse.
    """

    encoder_type: str = "hrnet_multi_phase"
    latent_dim: int | None = None
    hrnet_config: str = "hr_tiny_feat64_zyx_l4_in64"
    multi_decoder_hidden: int = 256
    multi_projection_weights: dict[str, float] | None = None
    projection_set: list[str] | None = None
    reconstruction_target: str = "projections"
    backbone_type: str = "hrnet"
    n_input_channels: int = 7
    phase_transformer_dim: int = 1024
    phase_transformer_heads: int = 8
    phase_transformer_layers: int = 2
    doppler_gate: bool = False
    doppler_se: bool = False

    def __post_init__(self) -> None:
        """Validate the architecture choices.

        Raises:
            ValueError: On an unknown reconstruction target or backbone, or an
                empty ``projection_set``.
        """
        if self.reconstruction_target not in ("projections", "full_cube"):
            raise ValueError(
                f"reconstruction_target must be 'projections' or 'full_cube', "
                f"got {self.reconstruction_target!r}"
            )
        if self.backbone_type != "hrnet":
            raise ValueError(f"backbone_type must be 'hrnet', got {self.backbone_type!r}")
        if self.projection_set is not None and not self.projection_set:
            raise ValueError("projection_set must be null or a non-empty list")


@dataclass(frozen=True)
class CompressionConfig:
    """Compression rate used when ``model.latent_dim`` is not set.

    Attributes:
        target_ratio: Fraction of the input's real scalars kept in the latent.
    """

    target_ratio: float = 0.9


@dataclass(frozen=True)
class TrainingConfig:
    """Pretraining hyperparameters and loader settings.

    Attributes:
        epochs: Number of epochs.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        beta: KL weight of the beta-VAE.
        beta_warmup_epochs: Epochs of linear KL warm-up.
        loss_type: Reconstruction loss; only ``"mse"`` is supported.
        n_decoder_samples: Unused; kept so released configs parse.
        nll_gamma: Unused; kept so released configs parse.
        lr_scheduler: Learning-rate schedule.
        grad_clip_norm: Max gradient norm.
        seed: Random seed.
        device: Torch device string.
        checkpoint_dir: Run directory for checkpoints and ``config.json``.
        resume: ``""`` starts fresh, ``"auto"`` resumes from
            ``<checkpoint_dir>/last.pt`` if it exists, anything else is a
            checkpoint path to resume from.
        log_every_n_steps: Logging interval in steps.
        val_every_n_epochs: Validation interval in epochs.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        steps_per_epoch: Frames drawn per epoch by the source-balanced sampler
            (``None`` = ``len(dataset)``). Fixing it keeps the number of
            optimisation steps equal across pretraining pools of different size.
    """

    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    beta: float = 5.0
    beta_warmup_epochs: int = 2
    loss_type: str = "mse"
    n_decoder_samples: int = 128
    nll_gamma: float = 1.0
    lr_scheduler: str = "cosine"
    grad_clip_norm: float = 2.0
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "./checkpoints"
    resume: str = ""
    log_every_n_steps: int = 1
    val_every_n_epochs: int = 5
    batch_size: int = 16
    num_workers: int = 4
    steps_per_epoch: int | None = None


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases logging configuration.

    Attributes:
        enabled: Whether to log to W&B.
        project: W&B project name.
        entity: W&B entity (team or user).
        run_name: Optional run name.
    """

    enabled: bool = False
    project: str = "beta-vae-radar"
    entity: str | None = None
    run_name: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level pretraining configuration.

    Attributes:
        dataset: Data settings.
        model: Architecture settings.
        compression: Compression rate.
        training: Training hyperparameters.
        wandb: W&B logging settings.
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


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


def _load_yaml_recursive(path: Path) -> dict[str, Any]:
    """Load a YAML file, recursively resolving ``_base_`` inheritance.

    Args:
        path: Path to the YAML config file.

    Returns:
        Fully merged raw dict with all base configs resolved.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    base_name = raw.pop("_base_", None)
    if base_name is not None:
        base_path = path.parent / base_name
        base_raw = _load_yaml_recursive(base_path)
        raw = _deep_merge(base_raw, raw)

    return raw


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load an experiment config from YAML with optional base inheritance.

    If the YAML contains a ``_base_`` key, that file is loaded first and the
    current file's values are deep-merged on top.  CLI overrides (from
    :func:`parse_overrides`) are applied last.

    Args:
        path: Path to the YAML config file.
        overrides: Optional nested dict of CLI overrides to apply on top.

    Returns:
        Fully resolved ``ExperimentConfig``.
    """
    path = Path(path)
    raw = _load_yaml_recursive(path)

    if overrides:
        raw = _deep_merge(raw, overrides)

    return ExperimentConfig(
        dataset=DatasetConfig(**raw.get("dataset", {})),
        model=ModelConfig(**raw.get("model", {})),
        compression=CompressionConfig(**raw.get("compression", {})),
        training=TrainingConfig(**raw.get("training", {})),
        wandb=WandbConfig(**raw.get("wandb", {})),
    )


def parse_overrides(argv: list[str]) -> dict[str, Any]:
    """Parse ``--section.key value`` CLI overrides into a nested dict.

    Supports dot-separated keys of any depth. Values are cast to int, float,
    bool, None, or left as strings. Example::

        ["--training.lr", "0.001", "--wandb.enabled", "true"]
        → {"training": {"lr": 0.001}, "wandb": {"enabled": True}}

    Args:
        argv: Raw CLI tokens (typically from ``parser.parse_known_args``).

    Returns:
        Nested dict suitable for deep-merging into a YAML config dict.
    """
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:]
        if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
            value: Any = True  # a flag without a value means true
        else:
            value = _cast_value(argv[i + 1])
            i += 1
        parts = key.split(".")
        d = overrides
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
        i += 1
    return overrides


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


def compute_latent_dim(
    input_shape: tuple[int, ...],
    target_ratio: float,
) -> int:
    """Compute latent dimension from input shape and compression ratio.

    Args:
        input_shape: Spatial shape of one sample, e.g. ``(64, 4, 4, 128)``.
        target_ratio: Fraction of scalar elements to keep (0.0–1.0).

    Returns:
        Integer latent dimension (``input_dim * 2 * target_ratio`` since
        complex64 has real + imaginary).
    """
    input_dim = prod(input_shape) * 2  # complex → real + imag
    return max(1, int(input_dim * target_ratio))
