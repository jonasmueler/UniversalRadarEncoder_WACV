"""Dataset over cached embedding sequences for the temporal classification head."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from frozenradar.classification.build_sequences import RecordingMeta

logger = logging.getLogger(__name__)


class ActivitySequenceDataset(Dataset):
    """Full-sequence dataset over cached per-frame embeddings.

    Each sample is one variable-length sequence labelled with its class index.
    Random cropping and time masking are applied when ``augment=True``.

    Args:
        sequences_dir: Directory containing ``rec_NNNN.pt`` files and ``meta.pt``.
        augment: Whether to apply temporal augmentations (training only).
        crop_ratio_min: Minimum fraction of sequence to keep when cropping.
        crop_ratio_max: Maximum fraction (1.0 = no cropping).
        time_mask_ratio: Fraction of frames to zero-mask.
    """

    def __init__(
        self,
        sequences_dir: str | Path,
        augment: bool = False,
        crop_ratio_min: float = 0.5,
        crop_ratio_max: float = 1.0,
        time_mask_ratio: float = 0.1,
    ) -> None:
        self.sequences_dir = Path(sequences_dir)
        self.augment = augment
        self.crop_ratio_min = crop_ratio_min
        self.crop_ratio_max = crop_ratio_max
        self.time_mask_ratio = time_mask_ratio

        raw_meta = torch.load(self.sequences_dir / "meta.pt", weights_only=False)
        self._meta = [RecordingMeta(**m) for m in raw_meta]
        logger.info(
            "Loaded %d recordings from %s", len(self._meta), self.sequences_dir,
        )

    def __len__(self) -> int:
        return len(self._meta)

    def __getitem__(
        self, idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return mu, logvar, and activity label for one recording.

        Args:
            idx: Sample index.

        Returns:
            Tuple of ``(mu, logvar, label)`` where mu and logvar have shape
            ``(T, latent_dim)`` and label is the class index.
        """
        path = self.sequences_dir / f"rec_{idx:04d}.pt"
        data = torch.load(path, weights_only=True)
        mu = data["mu"]
        logvar = data["logvar"]
        label = self._meta[idx].activity_idx

        if self.augment:
            mu, logvar = self._temporal_augment(mu, logvar)

        return mu, logvar, label

    def _temporal_augment(
        self, mu: torch.Tensor, logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random temporal crop and time masking.

        Args:
            mu: Latent means ``(T, D)``.
            logvar: Latent log-variances ``(T, D)``.

        Returns:
            Augmented ``(mu, logvar)`` pair.
        """
        T = mu.shape[0]

        if self.crop_ratio_min < 1.0:
            ratio = random.uniform(self.crop_ratio_min, self.crop_ratio_max)
            crop_len = max(1, int(T * ratio))
            start = random.randint(0, T - crop_len)
            mu = mu[start : start + crop_len]
            logvar = logvar[start : start + crop_len]
            T = crop_len

        if self.time_mask_ratio > 0:
            n_mask = max(1, int(T * self.time_mask_ratio))
            mask_idx = random.sample(range(T), n_mask)
            mu[mask_idx] = 0.0
            logvar[mask_idx] = 0.0

        return mu, logvar


def collate_sequences(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length recordings with padding.

    Args:
        batch: List of ``(mu, logvar, label)`` tuples from the dataset.

    Returns:
        Tuple ``(mu, logvar, labels, padding_mask)``: ``mu`` and ``logvar`` are
        ``(B, T_max, D)`` zero-padded, ``labels`` is ``(B,)`` long and
        ``padding_mask`` is ``(B, T_max)`` bool, ``True`` at padded positions.
    """
    mus, logvars, labels = zip(*batch)

    mu_padded = pad_sequence(mus, batch_first=True)
    logvar_padded = pad_sequence(logvars, batch_first=True)

    lengths = torch.tensor([m.shape[0] for m in mus])
    T_max = mu_padded.shape[1]
    padding_mask = torch.arange(T_max).unsqueeze(0) >= lengths.unsqueeze(1)

    return mu_padded, logvar_padded, torch.tensor(labels), padding_mask
