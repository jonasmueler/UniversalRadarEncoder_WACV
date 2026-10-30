"""Saving and loading of pretraining checkpoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    loss: float,
    path: str | Path,
    extras: dict[str, Any] | None = None,
) -> None:
    """Save model and optimizer state to disk atomically.

    The checkpoint is written to a temporary file and moved into place with
    ``os.replace``, so an interrupted save never leaves a corrupt checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        epoch: Current epoch number.
        loss: Current loss value (for bookkeeping).
        path: File path for the checkpoint.
        extras: Additional entries merged into the checkpoint dict (e.g.
            ``scheduler_state_dict``, ``global_step``, ``best_val_loss``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if extras:
        payload.update(extras)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    logger.info("Saved checkpoint to %s (epoch %d)", path, epoch)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
) -> dict[str, Any]:
    """Load model (and optionally optimizer) state from a checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: If provided, optimizer state is also restored.

    Returns:
        The full checkpoint dict (state dicts included), so callers can restore
        scheduler state, step counters and other extra entries.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt["epoch"])
    return ckpt
