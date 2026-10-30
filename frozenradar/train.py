"""Pretrain the beta-VAE radar encoder."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from frozenradar.data.builders import build_vae_dataloaders
from frozenradar.models.hrnet_multi_phase_vae import HRNetMultiPhaseBetaVAE
from frozenradar.training.multi_trainer import train_hierarchical
from frozenradar.utils.config import (
    ExperimentConfig,
    compute_latent_dim,
    load_config,
    parse_overrides,
)

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: ExperimentConfig, input_shape: tuple[int, ...]) -> torch.nn.Module:
    """Construct the beta-VAE model from config.

    Args:
        cfg: Experiment configuration.
        input_shape: Spatial shape ``(D, A, E, R)`` of one radar frame.

    Returns:
        Instantiated model (on CPU).

    Raises:
        ValueError: On an unknown ``model.encoder_type``.
    """
    latent_dim = cfg.model.latent_dim
    if latent_dim is None:
        latent_dim = compute_latent_dim(input_shape, cfg.compression.target_ratio)

    mc = cfg.model
    if mc.encoder_type != "hrnet_multi_phase":
        raise ValueError(f"Unknown encoder_type: {mc.encoder_type!r}")
    return HRNetMultiPhaseBetaVAE(
        latent_dim=latent_dim,
        input_shape=input_shape,
        n_input_channels=mc.n_input_channels,
        hrnet_config=mc.hrnet_config,
        transformer_dim=mc.phase_transformer_dim,
        transformer_heads=mc.phase_transformer_heads,
        transformer_layers=mc.phase_transformer_layers,
        decoder_hidden=mc.multi_decoder_hidden,
        doppler_gate=mc.doppler_gate,
        doppler_se=mc.doppler_se,
        projection_set=mc.projection_set,
        reconstruction_target=mc.reconstruction_target,
        backbone_type=mc.backbone_type,
    )


def _save_run_config(cfg: ExperimentConfig) -> None:
    """Write the fully resolved config as JSON into the run's checkpoint dir.

    Args:
        cfg: Experiment configuration to persist.
    """
    run_dir = Path(cfg.training.checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "config": dataclasses.asdict(cfg),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(record, f, indent=2, default=str)
    logger.info("Saved run config to %s", run_dir / "config.json")


def main(cfg: ExperimentConfig) -> None:
    """Set up data, model, and start training.

    Args:
        cfg: Fully resolved experiment configuration.
    """
    _save_run_config(cfg)
    _seed_everything(cfg.training.seed)

    # Fixed input shape: let cuDNN autotune kernels and allow TF32 matmuls.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_loader, val_loader = build_vae_dataloaders(cfg)

    input_shape = cfg.dataset.input_shape
    model = build_model(cfg, input_shape)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Encoder: %s", cfg.model.encoder_type)
    logger.info("Input shape: %s", input_shape)
    logger.info("Latent dim: %d", model.latent_dim)
    logger.info("Parameters: %s", f"{n_params:,}")

    device = torch.device(cfg.training.device)
    model = model.to(device)

    train_hierarchical(
        model, train_loader, val_loader, cfg,
        mask_collator=None,
        projection_weights=cfg.model.multi_projection_weights,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain the beta-VAE radar encoder.",
        epilog="Any config value can be overridden via --section.key value, "
        "e.g. --training.lr 0.001 --wandb.enabled true",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    args, extra = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    overrides = parse_overrides(extra)
    cfg = load_config(args.config, overrides=overrides)
    main(cfg)
