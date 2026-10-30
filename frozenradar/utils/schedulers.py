"""Learning-rate schedule of the classification head."""

from __future__ import annotations

import math

from torch.optim import Optimizer


class WarmupCosineSchedule:
    """Learning rate schedule with linear warmup and cosine decay.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of linear warmup steps.
        start_lr: Learning rate at the start of warmup.
        ref_lr: Peak learning rate after warmup.
        final_lr: Learning rate at the end of cosine decay.
        T_max: Total number of schedule steps (including warmup).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        start_lr: float,
        ref_lr: float,
        T_max: int,
        final_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0

    def step(self) -> float:
        """Advance one step and update the optimizer learning rate.

        Returns:
            New learning rate.
        """
        self._step += 1
        if self._step < self.warmup_steps:
            progress = self._step / max(1, self.warmup_steps)
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = (self._step - self.warmup_steps) / max(1, self.T_max)
            new_lr = max(
                self.final_lr,
                self.final_lr
                + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        return new_lr
