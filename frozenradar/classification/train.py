"""Train or evaluate the temporal classification head on cached embeddings."""

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
from torch import nn
from torch.utils.data import DataLoader

from frozenradar.classification.build_sequences import ACTIVITY_NAMES
from frozenradar.classification.config import (
    ClassificationConfig,
    load_config,
    parse_overrides,
)
from frozenradar.classification.dataset import (
    ActivitySequenceDataset,
    collate_sequences,
)
from frozenradar.classification.model import TemporalClassifier
from frozenradar.core.metrics import (
    brier_score,
    categorical_nll,
    confidence_ece,
    predictive_entropy,
)
from frozenradar.utils.schedulers import WarmupCosineSchedule

try:
    import wandb
except ImportError:  # optional extra, only for --wandb.enabled true
    wandb = None

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(cfg: ClassificationConfig) -> TemporalClassifier:
    """Construct the classifier from config.

    Args:
        cfg: Experiment configuration.

    Returns:
        Initialised classifier model.

    Raises:
        ValueError: On an unsupported ``model.model_type``.
    """
    mc = cfg.model
    if mc.model_type != "transformer":
        raise ValueError(f"model_type must be 'transformer', got {mc.model_type!r}")
    return TemporalClassifier(
        embed_dim=mc.embed_dim,
        temporal_layers=mc.temporal_layers,
        max_seq_len=mc.max_seq_len,
        num_heads=mc.num_heads,
        ff_ratio=mc.ff_ratio,
        dropout=mc.dropout,
        num_classes=mc.num_classes,
        alpha_schedule=mc.alpha_schedule,
        alpha_max=mc.alpha_max,
    )


def _eval_loader(dataset: ActivitySequenceDataset, batch_size: int, num_workers: int) -> DataLoader:
    """Build a sequential loader for evaluation.

    Args:
        dataset: Sequence dataset.
        batch_size: Batch size.
        num_workers: DataLoader workers.

    Returns:
        A non-shuffling loader that keeps the last partial batch.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_sequences,
    )


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_schedule: WarmupCosineSchedule,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    use_bfloat16: bool,
    grad_clip_norm: float,
    epoch: int,
    log_every: int,
) -> tuple[float, float]:
    """Run one training epoch.

    Args:
        model: The classifier model.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        lr_schedule: Learning rate scheduler (stepped per iteration).
        device: Torch device.
        scaler: Gradient scaler for mixed precision (None if disabled).
        use_bfloat16: Whether to use bfloat16 autocast.
        grad_clip_norm: Max gradient norm (0 = disabled).
        epoch: Current epoch number (for logging).
        log_every: Log every N steps.

    Returns:
        Tuple of (average loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for step, (mu, logvar, y, pad_mask) in enumerate(loader):
        mu = mu.to(device, non_blocking=True)
        logvar = logvar.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            logits = model(mu, logvar, padding_mask=pad_mask)
            loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        lr = lr_schedule.step()

        total_loss += loss.item() * mu.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += mu.size(0)

        if log_every > 0 and (step + 1) % log_every == 0:
            step_acc = correct / total
            step_loss = total_loss / total
            logger.info(
                "[epoch %d | step %d/%d] loss=%.4f acc=%.4f lr=%.2e",
                epoch, step + 1, len(loader), step_loss, step_acc, lr,
            )

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_bfloat16: bool,
    num_classes: int = 12,
) -> dict[str, float]:
    """Evaluate the model on one split.

    Args:
        model: The classifier model.
        loader: Validation data loader.
        criterion: Loss function.
        device: Torch device.
        use_bfloat16: Whether to use bfloat16 autocast.
        num_classes: Number of activity classes.

    Returns:
        Dict with ``loss``, ``acc``, macro ``f1``/``precision``/``recall`` over
        the classes present in the labels, the calibration metrics ``nll``,
        ``brier``, ``ece``, ``ece_adaptive`` and ``entropy``, plus ``per_class``
        counts and the ``preds`` and ``labels`` tensors.
    """
    model.eval()
    total_loss = 0.0
    total = 0
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for mu, logvar, y, pad_mask in loader:
        mu = mu.to(device, non_blocking=True)
        logvar = logvar.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            logits = model(mu, logvar, padding_mask=pad_mask)
            loss = criterion(logits, y)

        total_loss += loss.item() * mu.size(0)
        total += mu.size(0)
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu())
        all_labels.append(y.cpu())

    probs = torch.cat(all_probs)
    preds = probs.argmax(dim=1)
    labels = torch.cat(all_labels)

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    active_classes = 0
    per_class: list[dict[str, float]] = []

    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class.append({
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
        })

        if (tp + fn) > 0:  # class present in the labels
            precision_sum += prec
            recall_sum += rec
            f1_sum += f1
            active_classes += 1

    n = max(active_classes, 1)
    ece_w, _ = confidence_ece(probs, labels, mode="equal_width")
    ece_a, _ = confidence_ece(probs, labels, mode="adaptive")
    return {
        "loss": total_loss / total,
        "acc": (preds == labels).float().mean().item(),
        "f1": f1_sum / n,
        "precision": precision_sum / n,
        "recall": recall_sum / n,
        "nll": categorical_nll(probs, labels).item(),
        "brier": brier_score(probs, labels).item(),
        "ece": float(ece_w),
        "ece_adaptive": float(ece_a),
        "entropy": predictive_entropy(probs).mean().item(),
        "per_class": per_class,
        "preds": preds,
        "labels": labels,
    }


def main(cfg: ClassificationConfig) -> None:
    """Train (unless ``eval_only``) and evaluate the classification head.

    Args:
        cfg: Fully resolved experiment configuration.

    Raises:
        FileNotFoundError: If ``eval_only`` is set and there is no ``best.pt``.
        ImportError: If W&B logging is enabled but ``wandb`` is not installed.
    """
    tc = cfg.training
    dc = cfg.dataset
    _seed_everything(tc.seed)

    device = torch.device(tc.device)

    train_ds = ActivitySequenceDataset(
        sequences_dir=Path(dc.sequences_dir) / "train",
        augment=True,
        crop_ratio_min=dc.crop_ratio_min,
        crop_ratio_max=dc.crop_ratio_max,
        time_mask_ratio=dc.time_mask_ratio,
    )
    val_ds = ActivitySequenceDataset(
        sequences_dir=Path(dc.sequences_dir) / "val",
    )
    test_dir = Path(dc.sequences_dir) / "test"
    test_ds = ActivitySequenceDataset(sequences_dir=test_dir) if test_dir.exists() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=tc.batch_size,
        shuffle=True,
        num_workers=tc.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_sequences,
    )
    val_loader = _eval_loader(val_ds, tc.batch_size, tc.num_workers)
    test_loader = (
        _eval_loader(test_ds, tc.batch_size, tc.num_workers) if test_ds is not None else None
    )

    logger.info(
        "Train: %d, Val: %d, Test: %d recordings",
        len(train_ds), len(val_ds), len(test_ds) if test_ds is not None else 0,
    )

    model = _build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc.lr,
        weight_decay=tc.weight_decay,
    )

    total_steps = tc.epochs * len(train_loader)
    warmup_steps = tc.warmup_epochs * len(train_loader)
    lr_schedule = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        start_lr=tc.final_lr,
        ref_lr=tc.lr,
        final_lr=tc.final_lr,
        T_max=total_steps,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=tc.label_smoothing)

    use_amp = tc.use_bfloat16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ckpt_dir = Path(tc.checkpoint_dir)
    if tc.eval_only:
        if not (ckpt_dir / "best.pt").exists():
            raise FileNotFoundError(f"eval_only needs a trained head at {ckpt_dir / 'best.pt'}")
        logger.info("eval_only: evaluating %s without training", ckpt_dir / "best.pt")
    else:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        run_record = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv,
            "config": dataclasses.asdict(cfg),
        }
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(run_record, f, indent=2, default=str)
        logger.info("Saved run config to %s", ckpt_dir / "config.json")
    best_val_acc = 0.0

    wandb_run = None
    if cfg.wandb.enabled:
        if wandb is None:
            raise ImportError("wandb is not installed: pip install -e '.[logging]'")
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            config={
                "dataset": vars(dc) if hasattr(dc, "__dict__") else str(dc),
                "model": vars(cfg.model) if hasattr(cfg.model, "__dict__") else str(cfg.model),
                "training": vars(tc) if hasattr(tc, "__dict__") else str(tc),
            },
        )

    # In eval_only mode the range is empty and no training happens.
    for epoch in range(1, 0 if tc.eval_only else tc.epochs + 1):
        model.set_epoch(epoch, tc.epochs)
        train_loss, train_acc = _train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            device=device,
            scaler=scaler,
            use_bfloat16=use_amp,
            grad_clip_norm=tc.grad_clip_norm,
            epoch=epoch,
            log_every=tc.log_every_n_steps,
        )
        logger.info(
            "[epoch %d/%d] train_loss=%.4f train_acc=%.4f",
            epoch, tc.epochs, train_loss, train_acc,
        )

        if epoch % tc.val_every_n_epochs == 0:
            val_metrics = _validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                use_bfloat16=use_amp,
                num_classes=cfg.model.num_classes,
            )
            logger.info(
                "[epoch %d/%d] val_loss=%.4f val_acc=%.4f "
                "val_f1=%.4f val_prec=%.4f val_rec=%.4f",
                epoch, tc.epochs,
                val_metrics["loss"], val_metrics["acc"],
                val_metrics["f1"], val_metrics["precision"],
                val_metrics["recall"],
            )

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_metrics["loss"],
                    "val/acc": val_metrics["acc"],
                    "val/f1": val_metrics["f1"],
                    "val/precision": val_metrics["precision"],
                    "val/recall": val_metrics["recall"],
                    "lr": optimizer.param_groups[0]["lr"],
                })

            val_acc = val_metrics["acc"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_metrics["loss"],
                        "val_f1": val_metrics["f1"],
                    },
                    ckpt_dir / "best.pt",
                )
                logger.info("Saved best checkpoint (val_acc=%.4f)", val_acc)
        else:
            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                })

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc": best_val_acc,
            },
            ckpt_dir / "last.pt",
        )

    if not tc.eval_only:
        logger.info("Training complete. Best val_acc=%.4f", best_val_acc)

    # Final evaluation with the val-selected best checkpoint on the test split
    # (val if there is none) and the optional out-of-set split.
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        logger.info("Loaded best checkpoint for final evaluation")

    eval_targets = [
        ("test", test_loader) if test_loader is not None else ("val", val_loader),
    ]
    if dc.outofset_test_dir:
        outofset_ds = ActivitySequenceDataset(sequences_dir=Path(dc.outofset_test_dir))
        logger.info("Out-of-set test: %d recordings", len(outofset_ds))
        eval_targets.append(
            ("outofset", _eval_loader(outofset_ds, tc.batch_size, tc.num_workers))
        )

    for eval_name, eval_loader in eval_targets:
        final_metrics = _validate(
            model=model,
            loader=eval_loader,
            criterion=criterion,
            device=device,
            use_bfloat16=use_amp,
            num_classes=cfg.model.num_classes,
        )
        logger.info(
            "=== FINAL %s:  acc=%.3f  f1=%.3f  prec=%.3f  rec=%.3f ===",
            eval_name.upper(), final_metrics["acc"], final_metrics["f1"],
            final_metrics["precision"], final_metrics["recall"],
        )
        logger.info(
            "=== CALIBRATION %s:  nll=%.4f  brier=%.4f  ece=%.4f  ece_adaptive=%.4f  "
            "entropy=%.4f ===",
            eval_name.upper(), final_metrics["nll"], final_metrics["brier"],
            final_metrics["ece"], final_metrics["ece_adaptive"], final_metrics["entropy"],
        )
        if wandb_run is not None:
            wandb_run.log({
                f"{eval_name}/acc": final_metrics["acc"],
                f"{eval_name}/f1": final_metrics["f1"],
                f"{eval_name}/nll": final_metrics["nll"],
                f"{eval_name}/brier": final_metrics["brier"],
                f"{eval_name}/ece": final_metrics["ece"],
                f"{eval_name}/ece_adaptive": final_metrics["ece_adaptive"],
                f"{eval_name}/entropy": final_metrics["entropy"],
            })
        logger.info("=== Per-class %s results ===", eval_name)
        for c, pc in enumerate(final_metrics["per_class"]):
            name = ACTIVITY_NAMES[c] if c < len(ACTIVITY_NAMES) else f"ac{c}"
            logger.info(
                "  [%2d] %-28s  prec=%.3f  rec=%.3f  f1=%.3f  (tp=%d fp=%d fn=%d)",
                c, name, pc["precision"], pc["recall"], pc["f1"],
                int(pc["tp"]), int(pc["fp"]), int(pc["fn"]),
            )

        num_cls = cfg.model.num_classes
        cm = torch.zeros(num_cls, num_cls, dtype=torch.long)
        preds, labels = final_metrics["preds"], final_metrics["labels"]
        for i in range(len(preds)):
            cm[labels[i], preds[i]] += 1
        logger.info("=== Confusion matrix %s (rows=true, cols=pred) ===", eval_name)
        header = "          " + " ".join(f"{i:4d}" for i in range(num_cls))
        logger.info(header)
        for r in range(num_cls):
            row = " ".join(f"{cm[r, c]:4d}" for c in range(num_cls))
            logger.info("  [%2d] %s", r, row)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train activity classifier on VAE embeddings.",
        epilog="Override any config value via --section.key value, "
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
