"""Extract frozen-encoder embeddings from CubeLearn recordings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from frozenradar.data.cubelearn_dataset import CubeLearnVAEDataset
from frozenradar.data.splits import CUBELEARN_USERS
from frozenradar.extract_embeddings import _extract_split, _IQDataset, _load_model
from frozenradar.utils.config import ExperimentConfig, load_config, parse_overrides

logger = logging.getLogger(__name__)


def main(
    cfg: ExperimentConfig,
    checkpoint_path: Path,
    cubelearn_root: Path,
    output_dir: Path,
    task: str = "har",
    users: list[int] | None = None,
) -> None:
    """Extract the embeddings of one CubeLearn task.

    Args:
        cfg: Experiment configuration (for model architecture).
        checkpoint_path: Path to the VAE checkpoint.
        cubelearn_root: Directory containing CubeLearn .npy files.
        output_dir: Root directory for saved embeddings.
        task: CubeLearn task (hgr/agr/har).
        users: User IDs to extract (``None`` = the in-set users 0-5).
    """
    tc = cfg.training
    device = torch.device(tc.device)

    model = _load_model(cfg, checkpoint_path).to(device)
    logger.info("Latent dim: %d", model.latent_dim)

    frames_per_recording = 20 if task == "har" else 10
    n_gestures = 6 if task == "har" else 12
    if users is None:
        users = CUBELEARN_USERS

    # All 30 samples per (user, gesture); the split is applied by the sequence builder.
    all_samples = list(range(30))

    dataset = _IQDataset(
        CubeLearnVAEDataset(
            root=cubelearn_root,
            users=users,
            sample_ids=all_samples,
            n_gestures=n_gestures,
            frames_per_recording=frames_per_recording,
            apply_fft=cfg.dataset.apply_fft,
            clutter_removal=cfg.dataset.clutter_removal,
        ),
        encoder_type=cfg.model.encoder_type,
        format_mode=cfg.dataset.format_mode,
        normalize_magnitude=cfg.dataset.normalize_magnitude,
    )
    logger.info("Processing CubeLearn %s (users %s): %d frames", task, users, len(dataset))

    _extract_split(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        batch_size=tc.batch_size,
        num_workers=tc.num_workers,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract VAE embeddings from CubeLearn data.",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (for model architecture).",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to VAE checkpoint (.pt).",
    )
    parser.add_argument(
        "--cubelearn-root", type=str, required=True,
        help="Directory containing CubeLearn .npy files.",
    )
    parser.add_argument(
        "--task", type=str, default="har", choices=["hgr", "agr", "har"],
        help="CubeLearn task (determines frames per recording).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="cache/cubelearn_emb",
        help="Output directory for embeddings.",
    )
    parser.add_argument(
        "--users", type=int, nargs="+", default=None,
        help="User IDs to extract (default: in-set users 0-5).",
    )
    args, extra = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    overrides = parse_overrides(extra)
    cfg = load_config(args.config, overrides=overrides)
    main(
        cfg,
        Path(args.checkpoint),
        Path(args.cubelearn_root),
        Path(args.output_dir),
        args.task,
        args.users,
    )
