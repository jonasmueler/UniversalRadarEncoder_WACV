"""Build the pretraining datasets and loaders from an experiment config."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from ..utils.config import ExperimentConfig
from .ci4r_pretrain_dataset import CI4RPretrainDataset, Multi3PretrainDataset
from .cubelearn_dataset import CubeLearnVAEDataset
from .dataset import RadarVAEDataset
from .multi import MultiRadarDataset
from .samplers import build_balanced_sampler
from .splits import (
    CUBELEARN_USERS,
    MMRADPOSE_PARTICIPANTS,
    SplitName,
    cubelearn_sample_ids,
)

# Common range length so mmRadPose (native R=64) and CubeLearn (native R=256)
# produce the same frame shape and can be batched together.
_ALIGNED_RANGE_BINS = 256


def build_source_dataset(name: str, split: SplitName, cfg: ExperimentConfig) -> Dataset:
    """Build one source dataset for one split using the canonical splits.

    Args:
        name: Source name (see the module docstring).
        split: Which split to build.
        cfg: Experiment configuration.

    Returns:
        A frame dataset returning ``(D, A, E, R)`` complex64 tensors.

    Raises:
        ValueError: On an unknown source, a missing root, or a ``"test"``
            request for a CI4R source.
    """
    ds = cfg.dataset
    if name == "mmradpose":
        return RadarVAEDataset(
            root=ds.root,
            participants=MMRADPOSE_PARTICIPANTS[split],
            angles=ds.angles,
            apply_fft=ds.apply_fft,
            clutter_removal=ds.clutter_removal,
            target_range_bins=ds.target_range_bins,
        )
    if name == "cubelearn":
        if ds.cubelearn_root is None:
            raise ValueError("dataset.cubelearn_root must be set to use the 'cubelearn' source")
        sample_ids = cubelearn_sample_ids(ds.cubelearn_split_dir)
        return CubeLearnVAEDataset(
            root=ds.cubelearn_root,
            users=CUBELEARN_USERS,
            sample_ids=sample_ids[split],
            n_gestures=6,  # HAR has 6 activities; 12 classes is the HGR task
            frames_per_recording=ds.cubelearn_frames_per_recording,
            apply_fft=ds.apply_fft,
            clutter_removal=ds.clutter_removal,
        )
    if name in ("cubelearn_hgr", "cubelearn_agr"):
        root = ds.cubelearn_hgr_root if name == "cubelearn_hgr" else ds.cubelearn_agr_root
        if root is None:
            raise ValueError(f"dataset.{name}_root must be set to use the {name!r} source")
        sample_ids = cubelearn_sample_ids(ds.cubelearn_split_dir)
        return CubeLearnVAEDataset(
            root=root,
            users=CUBELEARN_USERS,
            sample_ids=sample_ids[split],
            n_gestures=12,  # HGR/AGR have 12 gesture classes
            frames_per_recording=10,  # 1 s recordings for HGR/AGR
            apply_fft=ds.apply_fft,
            clutter_removal=ds.clutter_removal,
        )
    # The CI4R stores hold benchmark-train units only (CI4R-ASL repetitions,
    # CI4R-Gait recordings); their "val" split is a VAE-side slice of those.
    if name == "ci4r":
        if ds.ci4r_root is None:
            raise ValueError("dataset.ci4r_root must be set to use the 'ci4r' source")
        if split == "test":
            raise ValueError("the 'ci4r' source has no test split — it is a held-out benchmark")
        return CI4RPretrainDataset(
            root=ds.ci4r_root,
            split=split,
            apply_fft=ds.apply_fft,
            clutter_removal=ds.clutter_removal,
            frames_per_recording=ds.ci4r_frames_per_recording,
        )
    if name == "multi3":
        if ds.multi3_root is None:
            raise ValueError("dataset.multi3_root must be set to use the 'multi3' source")
        if split == "test":
            raise ValueError("the 'multi3' source has no test split — it is a held-out benchmark")
        return Multi3PretrainDataset(
            root=ds.multi3_root,
            split=split,
            apply_fft=ds.apply_fft,
            clutter_removal=ds.clutter_removal,
            frames_per_recording=ds.multi3_frames_per_recording,
        )
    raise ValueError(f"Unknown dataset source: {name!r}")


def build_vae_dataset(cfg: ExperimentConfig, split: SplitName) -> MultiRadarDataset:
    """Build the (possibly multi-source) VAE dataset for one split.

    Args:
        cfg: Experiment configuration (``dataset.sources`` selects the sources).
        split: Which split to build.

    Returns:
        A :class:`MultiRadarDataset` wrapping the configured sources.

    Raises:
        ValueError: If multiple sources are combined without an aligned range,
            or a source is empty.
    """
    sources = cfg.dataset.sources
    if len(sources) > 1 and cfg.dataset.target_range_bins != _ALIGNED_RANGE_BINS:
        raise ValueError(
            "multi-source training requires "
            f"dataset.target_range_bins={_ALIGNED_RANGE_BINS} so sources share a range dim"
        )
    built = [(name, build_source_dataset(name, split, cfg)) for name in sources]
    # An empty source (bad root, incomplete store) would otherwise only
    # re-normalise the sampler mixture.
    for name, source_ds in built:
        if len(source_ds) == 0:
            raise ValueError(
                f"Source {name!r} produced 0 frames for split {split!r} — "
                "check its root path"
            )
    return MultiRadarDataset(built)


def build_vae_dataloaders(cfg: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders for VAE training.

    The train loader uses a source-balanced sampler; the val loader is
    sequential. Only train and val are built here (test is never seen by the
    unsupervised VAE).

    Args:
        cfg: Experiment configuration.

    Returns:
        ``(train_loader, val_loader)``.

    Raises:
        ValueError: If ``dataset.mixture_weights`` does not match the sources.
    """
    tc = cfg.training

    weights = None
    if cfg.dataset.mixture_weights is not None:
        if len(cfg.dataset.mixture_weights) != len(cfg.dataset.sources):
            raise ValueError(
                f"dataset.mixture_weights has {len(cfg.dataset.mixture_weights)} entries "
                f"but there are {len(cfg.dataset.sources)} sources; they must match"
            )
        weights = dict(zip(cfg.dataset.sources, cfg.dataset.mixture_weights))

    train_ds = build_vae_dataset(cfg, "train")
    val_ds = build_vae_dataset(cfg, "val")
    # steps_per_epoch=None draws len(train_ds) frames per epoch.
    sampler = build_balanced_sampler(
        train_ds, weights, num_samples=tc.steps_per_epoch,
    )

    persistent = tc.num_workers > 0
    # Extra keyword args only valid when workers are used.
    worker_kwargs = (
        {"persistent_workers": True, "prefetch_factor": 4} if persistent else {}
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=tc.batch_size,
        sampler=sampler,
        num_workers=tc.num_workers,
        pin_memory=True,
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tc.batch_size,
        shuffle=False,
        num_workers=tc.num_workers,
        pin_memory=True,
        **worker_kwargs,
    )
    return train_loader, val_loader
