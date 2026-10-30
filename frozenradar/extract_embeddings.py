"""Frozen-encoder embedding extraction: shared helpers and the mmRadPose CLI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from frozenradar.data.ci4r_dataset import CI4RVAEDataset
from frozenradar.data.dataset import RadarVAEDataset
from frozenradar.data.multi3_dataset import Multi3VAEDataset
from frozenradar.data.transforms import format_for_model
from frozenradar.train import build_model
from frozenradar.utils.config import ExperimentConfig, load_config, parse_overrides

logger = logging.getLogger(__name__)


class _IQDataset(torch.utils.data.Dataset):
    """Wrapper applying the encoder's input format to complex radar frames.

    Mirrors the pretraining input pipeline (including ``normalize_magnitude``)
    so the frozen encoder sees inputs formatted as in training.

    Args:
        inner: Radar dataset returning complex64 ``(D, A, E, R)`` frames.
        encoder_type: ``model.encoder_type`` of the config.
        format_mode: ``"iq"`` or ``"phase_diff"``.
        normalize_magnitude: Whether to rescale frames to unit RMS.
    """

    def __init__(
        self,
        inner: torch.utils.data.Dataset,
        encoder_type: str,
        format_mode: str = "iq",
        normalize_magnitude: bool = False,
    ) -> None:
        self._inner = inner
        self._encoder_type = encoder_type
        self._format_mode = format_mode
        self._normalize_magnitude = normalize_magnitude

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return the formatted frame ``idx``.

        Args:
            idx: Frame index.

        Returns:
            Float tensor ``(D, A, E, R, C)``.
        """
        x = self._inner[idx]
        return format_for_model(
            x.unsqueeze(0), self._encoder_type,
            format_mode=self._format_mode,
            normalize_magnitude=self._normalize_magnitude,
        ).squeeze(0)


def _load_model(
    cfg: ExperimentConfig,
    checkpoint_path: Path,
) -> torch.nn.Module:
    """Build the VAE and load checkpoint weights.

    Args:
        cfg: Experiment configuration.
        checkpoint_path: Path to the checkpoint file.

    Returns:
        Model in eval mode with frozen parameters.
    """
    model = build_model(cfg, cfg.dataset.input_shape)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded model from %s (epoch %s)", checkpoint_path, ckpt.get("epoch", "?"))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _extract_split(
    model: torch.nn.Module,
    dataset: _IQDataset,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_bfloat16: bool = True,
) -> None:
    """Extract mu and logvar for one split, saving per-batch files.

    Args:
        model: VAE model in eval mode.
        dataset: Dataset split to process.
        output_dir: Directory for per-batch .pt files.
        batch_size: Batch size for inference.
        num_workers: DataLoader workers.
        device: Torch device.
        use_bfloat16: Whether to use bfloat16 autocast.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    n_samples = n_batches = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc=output_dir.name)):
        batch = batch.to(device, non_blocking=True)

        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=use_bfloat16,
        ):
            features = model._encode(batch)
            mu = model.mu_head(features)
            logvar = model.logvar_head(features)

        torch.save(
            {"mu": mu.float().cpu(), "logvar": logvar.float().cpu()},
            output_dir / f"batch_{batch_idx:06d}.pt",
        )
        n_samples += mu.shape[0]
        n_batches += 1

    logger.info("Saved %d samples (%d batches) to %s", n_samples, n_batches, output_dir)


def extract_raw_tree(
    inner: CI4RVAEDataset | Multi3VAEDataset,
    cfg: ExperimentConfig,
    checkpoint_path: Path,
    root: Path,
    output_dir: Path,
) -> None:
    """Extract embeddings of a CI4R raw tree and save its frame index.

    Next to the ``batch_NNNNNN.pt`` files an ``index.pt`` holds the
    ``(recording_key, frame_idx)`` iteration order and the per-recording frame
    counts. Recording keys are stored relative to ``root`` so the sequence
    builders do not depend on absolute paths.

    Args:
        inner: Raw-tree frame dataset built over ``root``.
        cfg: Encoder configuration.
        checkpoint_path: Encoder checkpoint.
        root: Root the dataset was built over.
        output_dir: Output directory.

    Raises:
        FileNotFoundError: If ``root`` holds no recording.
    """
    device = torch.device(cfg.training.device)
    model = _load_model(cfg, checkpoint_path).to(device)
    logger.info("Latent dim: %d", model.latent_dim)

    dataset = _IQDataset(
        inner,
        encoder_type=cfg.model.encoder_type,
        format_mode=cfg.dataset.format_mode,
        normalize_magnitude=cfg.dataset.normalize_magnitude,
    )
    logger.info("Processing %s: %d frames", root, len(dataset))
    if len(dataset) == 0:
        raise FileNotFoundError(f"No .bin recordings found under {root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    root_str = str(root)
    index = [(key.removeprefix(root_str).lstrip("/"), frame) for key, frame in inner.index]
    frame_counts = {
        key.removeprefix(root_str).lstrip("/"): count
        for key, count in inner.frame_counts.items()
    }
    torch.save({"index": index, "frame_counts": frame_counts}, output_dir / "index.pt")

    _extract_split(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        device=device,
    )


def main(cfg: ExperimentConfig, checkpoint_path: Path, output_dir: Path) -> None:
    """Extract the mmRadPose embeddings of the train, val and test participants.

    Args:
        cfg: Experiment configuration.
        checkpoint_path: Path to the VAE checkpoint.
        output_dir: Root directory for saved embeddings.
    """
    ds = cfg.dataset
    tc = cfg.training
    device = torch.device(tc.device)

    model = _load_model(cfg, checkpoint_path).to(device)
    logger.info("Latent dim: %d", model.latent_dim)

    splits = {
        "train": ds.participants_train,
        "val": ds.participants_val,
        "test": ds.participants_test,
    }

    for split_name, participants in splits.items():
        if not participants:
            continue
        dataset = _IQDataset(
            RadarVAEDataset(
                root=ds.root,
                participants=participants,
                angles=ds.angles,
                apply_fft=ds.apply_fft,
                clutter_removal=ds.clutter_removal,
                target_range_bins=ds.target_range_bins,
            ),
            encoder_type=cfg.model.encoder_type,
            format_mode=ds.format_mode,
            normalize_magnitude=ds.normalize_magnitude,
        )
        logger.info("Processing %s split: %d samples", split_name, len(dataset))

        _extract_split(
            model=model,
            dataset=dataset,
            output_dir=output_dir / split_name,
            batch_size=tc.batch_size,
            num_workers=tc.num_workers,
            device=device,
            use_bfloat16=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract mmRadPose mu/logvar embeddings for the train, val and test splits.",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to VAE checkpoint (.pt).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="cache/mmradpose_emb",
        help="Root directory for saved embeddings.",
    )
    args, extra = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    overrides = parse_overrides(extra)
    cfg = load_config(args.config, overrides=overrides)
    main(cfg, Path(args.checkpoint), Path(args.output_dir))
