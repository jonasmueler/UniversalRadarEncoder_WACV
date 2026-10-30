"""Proper scoring rules, calibration and uncertainty for categorical predictions."""

from __future__ import annotations

import torch


def _check_probs(probs: torch.Tensor, labels: torch.Tensor) -> None:
    """Validate a ``(B, C)`` probability tensor against ``(B,)`` labels.

    Args:
        probs: Predictive class probabilities.
        labels: Integer class labels.

    Raises:
        ValueError: If the shapes are not ``(B, C)`` and ``(B,)``.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be (B, C), got {tuple(probs.shape)}")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError(f"labels must be (B,) matching probs batch, got {tuple(labels.shape)}")


def categorical_nll(
    probs: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mean negative log-likelihood of the true class under a categorical prediction.

    Args:
        probs: Predictive class probabilities ``(B, C)`` (rows sum to 1).
        labels: Integer class labels ``(B,)``.
        eps: Clamp on the probability before the log for numerical safety.

    Returns:
        Scalar mean NLL in nats (lower is better).

    Raises:
        ValueError: On malformed ``probs`` / ``labels`` shapes.
    """
    _check_probs(probs, labels)
    true_p = probs.gather(1, labels.long().unsqueeze(1)).squeeze(1)
    return -torch.log(true_p.clamp_min(eps)).mean()


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Multiclass Brier score: mean squared error against the one-hot label.

    Args:
        probs: Predictive class probabilities ``(B, C)``.
        labels: Integer class labels ``(B,)``.

    Returns:
        Scalar Brier score in ``[0, 2]`` (lower is better).

    Raises:
        ValueError: On malformed ``probs`` / ``labels`` shapes.
    """
    _check_probs(probs, labels)
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, labels.long().unsqueeze(1), 1.0)
    return (probs - onehot).pow(2).sum(dim=1).mean()


def predictive_entropy(
    probs: torch.Tensor,
    base: str = "e",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Shannon entropy of each predictive distribution.

    Args:
        probs: Predictive class probabilities ``(B, C)``.
        base: ``"e"`` for nats or ``"2"`` for bits.
        eps: Clamp inside the log; the ``p * log p`` product still tends to 0
            as ``p -> 0`` so this only guards the log of exact zeros.

    Returns:
        Per-sample entropy ``(B,)`` (non-negative).

    Raises:
        ValueError: On a non-``(B, C)`` input or an unknown ``base``.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be (B, C), got {tuple(probs.shape)}")
    ent = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)
    if base == "2":
        ent = ent / torch.log(torch.tensor(2.0, device=probs.device, dtype=probs.dtype))
    elif base != "e":
        raise ValueError(f"base must be 'e' or '2', got {base!r}")
    return ent


def confidence_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
    mode: str = "equal_width",
) -> tuple[torch.Tensor, list[tuple[int, float, float]]]:
    """Expected Calibration Error via top-label confidence binning (Guo et al. 2017).

    Bins predictions by the confidence of the predicted (top-1) class and
    accumulates the gap between average confidence and accuracy per bin::

        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        probs: Predictive class probabilities ``(B, C)``.
        labels: Integer class labels ``(B,)``.
        n_bins: Number of confidence bins.
        mode: ``"equal_width"`` for fixed ``[0, 1]`` bins, or ``"adaptive"``
            for equal-mass (quantile) bins that reduce bias when confidences
            concentrate near 1.

    Returns:
        Tuple ``(ece, bin_stats)`` where ``bin_stats`` is a list of
        ``(count, mean_confidence, mean_accuracy)`` per non-empty bin, for
        reliability diagrams.

    Raises:
        ValueError: On malformed shapes or an unknown ``mode``.
    """
    _check_probs(probs, labels)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels.long()).float()
    n = probs.shape[0]

    if mode == "equal_width":
        edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device, dtype=probs.dtype)
    elif mode == "adaptive":
        quantiles = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device, dtype=probs.dtype)
        edges = torch.quantile(conf, quantiles)
        edges = torch.unique(edges)  # dedupe collapsed quantiles (tied confidences)
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"mode must be 'equal_width' or 'adaptive', got {mode!r}")

    ece = torch.zeros((), device=probs.device, dtype=probs.dtype)
    bin_stats: list[tuple[int, float, float]] = []
    n_edges = edges.shape[0]
    for i in range(n_edges - 1):
        lo, hi = edges[i], edges[i + 1]
        # Left-closed bins; the last bin includes the right edge (conf == 1).
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_conf = conf[in_bin].mean()
        bin_acc = correct[in_bin].mean()
        ece = ece + (count / n) * (bin_acc - bin_conf).abs()
        bin_stats.append((count, float(bin_conf), float(bin_acc)))
    return ece, bin_stats


def uncertainty_decomposition(
    samples: torch.Tensor,
    base: str = "e",
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Total / aleatoric / epistemic uncertainty from an MC ensemble (BALD).

    Given ``S`` sampled categorical predictions per input, decomposes the
    predictive entropy into aleatoric (expected entropy of the members) and
    epistemic (mutual information between the label and the model) parts::

        total     = H(mean_s p_s)
        aleatoric = mean_s H(p_s)
        epistemic = total - aleatoric   (>= 0, the mutual information / BALD)

    Args:
        samples: MC categorical predictions ``(B, S, C)`` (each ``(B, s, :)``
            sums to 1).
        base: ``"e"`` for nats or ``"2"`` for bits.
        eps: Clamp inside the logs.

    Returns:
        Dict with per-sample ``"total"``, ``"aleatoric"``, ``"epistemic"``
        tensors ``(B,)``.

    Raises:
        ValueError: On a malformed sample tensor or negative mutual information
            beyond floating-point tolerance (indicates a bug upstream).
    """
    if samples.ndim != 3:
        raise ValueError(f"samples must be (B, S, C), got {tuple(samples.shape)}")
    p_bar = samples.mean(dim=1)  # (B, C)
    total = predictive_entropy(p_bar, base=base, eps=eps)  # (B,)
    member_ent = -(samples * torch.log(samples.clamp_min(eps))).sum(dim=2)  # (B, S)
    aleatoric = member_ent.mean(dim=1)  # (B,)
    if base == "2":
        aleatoric = aleatoric / torch.log(
            torch.tensor(2.0, device=samples.device, dtype=samples.dtype)
        )
    epistemic = total - aleatoric
    if bool((epistemic < -1e-4).any()):
        raise ValueError("epistemic (mutual information) is negative beyond tolerance")
    return {"total": total, "aleatoric": aleatoric, "epistemic": epistemic.clamp_min(0.0)}
