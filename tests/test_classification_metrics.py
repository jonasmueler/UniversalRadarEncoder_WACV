"""Tests of the classification and calibration metrics in ``frozenradar.core.metrics``."""

import math

import pytest
import torch

import frozenradar.core.metrics as M


def _onehot(labels: torch.Tensor, c: int) -> torch.Tensor:
    """One-hot encode labels as float.

    Args:
        labels: Integer labels.
        c: Number of classes.

    Returns:
        ``(N, c)`` float tensor.
    """
    return torch.nn.functional.one_hot(labels, c).float()


class TestCategoricalNLL:
    """Negative log-likelihood of the true class."""

    def test_confident_correct_near_zero(self) -> None:
        """Confident correct predictions have NLL near zero."""
        labels = torch.tensor([0, 1, 2])
        probs = _onehot(labels, 3).clamp(1e-6, 1.0)
        assert M.categorical_nll(probs, labels).item() == pytest.approx(0.0, abs=1e-4)

    def test_uniform_equals_log_c(self) -> None:
        """A uniform prediction has NLL ``log C``."""
        c = 5
        probs = torch.full((4, c), 1.0 / c)
        labels = torch.tensor([0, 1, 2, 3])
        assert M.categorical_nll(probs, labels).item() == pytest.approx(math.log(c), rel=1e-5)


class TestBrierScore:
    """Multi-class Brier score."""

    def test_perfect_zero(self) -> None:
        """One-hot correct predictions have Brier score zero."""
        labels = torch.tensor([0, 1, 2])
        assert M.brier_score(_onehot(labels, 3), labels).item() == pytest.approx(0.0, abs=1e-6)

    def test_uniform_closed_form(self) -> None:
        """A uniform prediction over C classes scores ``1 - 1/C``."""
        # Uniform over C: (1 - 1/C)^2 + (C-1)*(1/C)^2 = 1 - 1/C.
        c = 4
        probs = torch.full((10, c), 1.0 / c)
        labels = torch.randint(0, c, (10,))
        assert M.brier_score(probs, labels).item() == pytest.approx(1 - 1 / c, rel=1e-5)


class TestPredictiveEntropy:
    """Entropy of the predictive distribution."""

    def test_uniform_is_log_c(self) -> None:
        """A uniform prediction has entropy ``log C``."""
        c = 6
        probs = torch.full((3, c), 1.0 / c)
        assert M.predictive_entropy(probs).mean().item() == pytest.approx(math.log(c), rel=1e-5)

    def test_onehot_is_zero(self) -> None:
        """A one-hot prediction has zero entropy."""
        labels = torch.tensor([0, 2])
        probs = _onehot(labels, 3).clamp(1e-12, 1.0)
        assert M.predictive_entropy(probs).max().item() == pytest.approx(0.0, abs=1e-4)

    def test_bits_base(self) -> None:
        """A fair coin has one bit of entropy in base 2."""
        probs = torch.tensor([[0.5, 0.5]])
        assert M.predictive_entropy(probs, base="2").item() == pytest.approx(1.0, rel=1e-5)


class TestConfidenceECE:
    """Expected calibration error of the top-1 confidence."""

    def test_perfectly_calibrated_zero(self) -> None:
        """Accuracy equal to confidence in every bin gives ECE zero."""
        # In each confidence group the accuracy equals the confidence.
        # Group A: conf 0.6, 10 samples, 6 correct. Group B: conf 0.9, 10 samples, 9 correct.
        rows, labels = [], []
        for conf, n_correct in [(0.6, 6), (0.9, 9)]:
            for i in range(10):
                rows.append([conf, 1 - conf])
                labels.append(0 if i < n_correct else 1)
        probs = torch.tensor(rows)
        lab = torch.tensor(labels)
        ece, _ = M.confidence_ece(probs, lab, n_bins=15)
        assert ece.item() == pytest.approx(0.0, abs=1e-6)

    def test_confident_wrong_is_one(self) -> None:
        """Always wrong at confidence 1.0 gives ECE one."""
        probs = torch.tensor([[1.0, 0.0]] * 8)
        labels = torch.ones(8, dtype=torch.long)  # always wrong, conf 1.0
        ece, _ = M.confidence_ece(probs, labels, n_bins=10)
        assert ece.item() == pytest.approx(1.0, rel=1e-5)

    def test_adaptive_matches_on_calibrated(self) -> None:
        """Equal-width and adaptive binning both give zero on calibrated data."""
        probs = torch.tensor([[0.6, 0.4]] * 10 + [[0.9, 0.1]] * 10)
        labels = torch.tensor([0] * 6 + [1] * 4 + [0] * 9 + [1])
        ece_w, _ = M.confidence_ece(probs, labels, mode="equal_width")
        ece_a, _ = M.confidence_ece(probs, labels, mode="adaptive")
        assert ece_w.item() == pytest.approx(0.0, abs=1e-6)
        assert ece_a.item() == pytest.approx(0.0, abs=1e-6)

    def test_empty_bins_no_nan(self) -> None:
        """Empty equal-width bins do not produce NaN."""
        # All confidence in one narrow region -> most equal-width bins empty.
        probs = torch.tensor([[0.99, 0.01]] * 20)
        labels = torch.zeros(20, dtype=torch.long)
        ece, stats = M.confidence_ece(probs, labels, n_bins=15)
        assert torch.isfinite(ece).item()
        assert len(stats) >= 1


class TestUncertaintyDecomposition:
    """Total = aleatoric + epistemic decomposition over sampled predictions."""

    def test_identical_members_zero_epistemic(self) -> None:
        """Identical samples have zero epistemic uncertainty."""
        base = torch.tensor([[0.7, 0.2, 0.1]])
        samples = base.unsqueeze(1).repeat(1, 8, 1)  # (1, 8, 3) identical
        out = M.uncertainty_decomposition(samples)
        assert out["epistemic"].item() == pytest.approx(0.0, abs=1e-5)
        assert out["total"].item() == pytest.approx(out["aleatoric"].item(), abs=1e-5)

    def test_spread_onehots_max_epistemic(self) -> None:
        """Distinct one-hot samples covering all classes are purely epistemic."""
        # Members are distinct one-hots covering all C classes uniformly:
        # p_bar uniform -> total = log C, each member entropy 0 -> aleatoric 0.
        c = 3
        samples = _onehot(torch.arange(c), c).unsqueeze(0)  # (1, C, C)
        out = M.uncertainty_decomposition(samples.clamp(1e-12, 1.0))
        assert out["aleatoric"].item() == pytest.approx(0.0, abs=1e-4)
        assert out["epistemic"].item() == pytest.approx(math.log(c), rel=1e-3)

    def test_identity_holds_random(self) -> None:
        """Total equals aleatoric plus epistemic on random samples."""
        torch.manual_seed(0)
        logits = torch.randn(5, 10, 4)
        samples = torch.softmax(logits, dim=-1)
        out = M.uncertainty_decomposition(samples)
        lhs = out["total"]
        rhs = out["aleatoric"] + out["epistemic"]
        assert torch.allclose(lhs, rhs, atol=1e-5)
        assert (out["epistemic"] >= -1e-6).all()


class TestShapeGuards:
    """Input shape validation."""

    def test_bad_probs_shape(self) -> None:
        """A 3-D probability tensor is rejected."""
        with pytest.raises(ValueError):
            M.categorical_nll(torch.rand(3, 4, 5), torch.zeros(3, dtype=torch.long))

    def test_bad_samples_shape(self) -> None:
        """A 2-D sample tensor is rejected."""
        with pytest.raises(ValueError):
            M.uncertainty_decomposition(torch.rand(3, 4))
