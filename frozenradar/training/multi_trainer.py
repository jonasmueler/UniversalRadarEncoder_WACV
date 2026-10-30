"""Pretraining loop of the multi-projection beta-VAE."""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..data.transforms import format_for_model
from ..models.hrnet_multi_phase_vae import HRNetMultiPhaseBetaVAE
from ..training.multi_losses import multi_projection_loss
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.config import ExperimentConfig

try:
    import wandb
except ImportError:  # optional, only for --wandb.enabled
    wandb = None

try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
except ImportError:  # optional, only for --wandb.enabled
    FigureCanvasAgg = None
    Figure = None

_MaskCollator = Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]]

logger = logging.getLogger(__name__)


def _require_logging_deps() -> None:
    """Check that the optional Weights & Biases logging dependencies are installed.

    Raises:
        ImportError: If ``wandb`` or ``matplotlib`` is missing.
    """
    if wandb is None or Figure is None:
        raise ImportError(
            "wandb logging needs wandb and matplotlib: pip install -e '.[logging]'"
        )


def _init_wandb(cfg: ExperimentConfig) -> bool:
    """Initialise Weights & Biases if enabled.

    Args:
        cfg: Experiment configuration.

    Returns:
        Whether wandb was initialised.

    Raises:
        ImportError: If logging is enabled but its dependencies are missing.
    """
    if not cfg.wandb.enabled:
        return False
    _require_logging_deps()

    flat_cfg: dict[str, object] = {}
    for section_name, section in [
        ("dataset", cfg.dataset),
        ("model", cfg.model),
        ("compression", cfg.compression),
        ("training", cfg.training),
    ]:
        for k, v in dataclasses.asdict(section).items():
            flat_cfg[f"{section_name}/{k}"] = v

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        config=flat_cfg,
    )
    return True


def train_hierarchical(
    model: HRNetMultiPhaseBetaVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    mask_collator: _MaskCollator | None = None,
    projection_weights: dict[str, float] | None = None,
) -> None:
    """Run the multi-projection beta-VAE training loop.

    Optimises the MSE projection loss plus ``beta * KL / d_z`` with AdamW,
    a per-epoch cosine learning-rate schedule, gradient-norm clipping and a
    linear KL warm-up over ``training.beta_warmup_epochs``.  Writes
    ``last.pt`` every epoch and ``best.pt`` on improved validation loss, and
    resumes from a checkpoint when ``training.resume`` is set.

    Args:
        model: Multi-projection VAE already on the target device.
        train_loader: Training data loader yielding complex frames.
        val_loader: Validation data loader yielding complex frames.
        cfg: Experiment configuration.
        mask_collator: Optional input-mask generator; its ``"doppler"`` mask
            zeroes chirps before encoding.
        projection_weights: Optional per-projection loss weights.

    Raises:
        ValueError: If ``training.loss_type`` selects an unsupported loss.
    """
    tc = cfg.training
    if tc.loss_type != "mse":
        raise ValueError(
            f"training.loss_type {tc.loss_type!r} is not supported; "
            "the projections are trained with MSE"
        )
    device = next(model.parameters()).device
    use_wandb = _init_wandb(cfg)

    param_groups = [{"params": list(model.parameters())}]
    optimizer = AdamW(
        param_groups,
        lr=tc.lr,
        weight_decay=tc.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=tc.epochs)

    global_step = 0
    best_val_loss = float("inf")
    start_epoch = 1

    resume_path = None
    if tc.resume == "auto":
        candidate = Path(tc.checkpoint_dir) / "last.pt"
        if candidate.exists():
            resume_path = candidate
    elif tc.resume:
        resume_path = Path(tc.resume)
    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, model, optimizer)
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            # No scheduler state stored: replay one step per finished epoch.
            for _ in range(start_epoch - 1):
                scheduler.step()
        logger.info(
            "RESUMING from %s: starting at epoch %d/%d (global step %d, "
            "best val loss %.4f)",
            resume_path, start_epoch, tc.epochs, global_step, best_val_loss,
        )
        if start_epoch > tc.epochs:
            logger.info("Training already complete (epoch %d); nothing to do.",
                        start_epoch - 1)
            if use_wandb:
                wandb.finish()
            return

    def _extras() -> dict[str, object]:
        """Collect the training state stored next to the model weights.

        Returns:
            Scheduler state, global step and best validation loss.
        """
        return {
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "best_val_loss": best_val_loss,
        }

    for epoch in range(start_epoch, tc.epochs + 1):
        if tc.beta_warmup_epochs > 0:
            beta = min(1.0, epoch / tc.beta_warmup_epochs) * tc.beta
        else:
            beta = tc.beta

        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            x = format_for_model(
                batch.to(device), cfg.model.encoder_type,
                format_mode=cfg.dataset.format_mode,
                normalize_magnitude=cfg.dataset.normalize_magnitude,
            )

            # Targets come from the full, unmasked input.
            targets = model.compute_targets(x)

            masks = None
            if mask_collator is not None:
                _, masks = mask_collator(x)

            projections, mu, logvar = model(x, masks=masks)
            loss, metrics = multi_projection_loss(
                projections, targets, mu, logvar, beta,
                projection_weights,
            )

            optimizer.zero_grad()
            loss.backward()
            if tc.grad_clip_norm > 0:
                clip_grad_norm_(model.parameters(), tc.grad_clip_norm)
            optimizer.step()

            epoch_loss += metrics["loss"]
            epoch_recon += metrics["recon_total"]
            epoch_kl += metrics["kl"]
            n_batches += 1
            global_step += 1

            if global_step % tc.log_every_n_steps == 0:
                logger.info(
                    "step %d | loss %.4f | recon %.4f | kl %.4f | beta %.3f",
                    global_step,
                    metrics["loss"],
                    metrics["recon_total"],
                    metrics["kl"],
                    beta,
                )
                if use_wandb:
                    log_dict = {
                        "train/loss": metrics["loss"],
                        "train/recon_total": metrics["recon_total"],
                        "train/kl": metrics["kl"],
                        "train/beta": beta,
                        "train/lr": scheduler.get_last_lr()[0],
                    }
                    for k, v in metrics.items():
                        if k.startswith("recon/"):
                            log_dict[f"train/{k}"] = v
                    wandb.log(log_dict, step=global_step)

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_recon = epoch_recon / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)
        elapsed = time.time() - t0
        logger.info(
            "epoch %d/%d | loss %.4f (recon %.4f, kl %.4f) | "
            "beta %.3f | lr %.2e | %.1fs",
            epoch,
            tc.epochs,
            avg_loss,
            avg_recon,
            avg_kl,
            beta,
            scheduler.get_last_lr()[0],
            elapsed,
        )

        if epoch % tc.val_every_n_epochs == 0:
            val_metrics, val_proj, val_tgt = _validate(
                model, val_loader, cfg, beta, device,
                mask_collator, projection_weights,
            )
            logger.info(
                "epoch %d/%d | val loss %.4f (recon %.4f, kl %.4f)",
                epoch,
                tc.epochs,
                val_metrics["loss"],
                val_metrics["recon_total"],
                val_metrics["kl"],
            )
            if use_wandb:
                wandb.log(
                    {f"val/{k}": v for k, v in val_metrics.items()},
                    step=global_step,
                )
                _log_projection_images(val_proj, val_tgt, global_step)
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_val_loss,
                    path=f"{tc.checkpoint_dir}/best.pt",
                    extras=_extras(),
                )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            avg_loss,
            path=f"{tc.checkpoint_dir}/last.pt",
            extras=_extras(),
        )

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    if use_wandb:
        wandb.finish()


@torch.no_grad()
def _validate(
    model: HRNetMultiPhaseBetaVAE,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    beta: float,
    device: torch.device,
    mask_collator: _MaskCollator | None = None,
    projection_weights: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Run one validation pass.

    Args:
        model: The multi-projection VAE model.
        val_loader: Validation data loader.
        cfg: Experiment configuration.
        beta: Current beta weight.
        device: Torch device.
        mask_collator: Optional input-mask generator.
        projection_weights: Optional per-projection loss weights.

    Returns:
        Batch-averaged metrics, and the projections and targets of the first
        sample of the first batch (on CPU).
    """
    model.eval()
    totals: dict[str, float] = {}
    n_batches = 0
    first_projections: dict[str, torch.Tensor] = {}
    first_targets: dict[str, torch.Tensor] = {}

    for batch in val_loader:
        x = format_for_model(
            batch.to(device), cfg.model.encoder_type,
            format_mode=cfg.dataset.format_mode,
            normalize_magnitude=cfg.dataset.normalize_magnitude,
        )
        targets = model.compute_targets(x)

        masks = None
        if mask_collator is not None:
            _, masks = mask_collator(x)

        projections, mu, logvar = model(x, masks=masks)
        _, metrics = multi_projection_loss(
            projections, targets, mu, logvar, beta, projection_weights,
        )

        if n_batches == 0:
            first_projections = {
                k: v[0].cpu() for k, v in projections.items()
            }
            first_targets = {
                k: v[0].cpu() for k, v in targets.items()
            }

        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    denom = max(n_batches, 1)
    return (
        {k: v / denom for k, v in totals.items()},
        first_projections,
        first_targets,
    )


def _log_projection_images(
    projections: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    step: int,
) -> None:
    """Log ground-truth vs. predicted projection images to wandb.

    Args:
        projections: Predicted 2D projections of one sample (on CPU).
        targets: Ground-truth 2D projections of one sample (on CPU).
        step: Global training step for wandb logging.

    Raises:
        ImportError: If ``wandb`` or ``matplotlib`` is missing.
    """
    _require_logging_deps()

    images = {}
    for name in projections:
        pred = projections[name].numpy()
        gt = targets[name].numpy()
        vmin = min(gt.min(), pred.min())
        vmax = max(gt.max(), pred.max())

        fig = Figure(figsize=(8, 3))
        canvas = FigureCanvasAgg(fig)
        ax_gt, ax_pred = fig.subplots(1, 2)
        ax_gt.imshow(gt, aspect="auto", vmin=vmin, vmax=vmax)
        ax_gt.set_title("GT")
        im = ax_pred.imshow(pred, aspect="auto", vmin=vmin, vmax=vmax)
        ax_pred.set_title("Pred")
        fig.colorbar(im, ax=[ax_gt, ax_pred], shrink=0.8)
        fig.suptitle(name, fontsize=10)
        canvas.draw()
        images[f"val/images/{name}"] = wandb.Image(np.asarray(canvas.buffer_rgba())[..., :3])

    wandb.log(images, step=step)
