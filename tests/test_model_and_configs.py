"""Tests for the multi-projection beta-VAE and the pretraining configs."""

from __future__ import annotations

import pytest
import torch

from frozenradar.models.hrnet_multi_phase_vae import HRNetMultiPhaseBetaVAE
from frozenradar.models.projections import PROJECTIONS
from frozenradar.training import multi_trainer
from frozenradar.training.multi_losses import multi_projection_loss
from frozenradar.utils.config import ModelConfig, load_config

SMALL_SHAPE = (16, 4, 4, 32)  # (D, A, E, R) kept small for CPU tests
LATENT = 64


def _small_model(**kwargs: object) -> HRNetMultiPhaseBetaVAE:
    """Build a CPU-sized model.

    Args:
        **kwargs: Extra constructor arguments.

    Returns:
        The model, initialised with a fixed seed.
    """
    torch.manual_seed(0)
    return HRNetMultiPhaseBetaVAE(
        latent_dim=LATENT,
        input_shape=SMALL_SHAPE,
        n_input_channels=2,
        hrnet_config="hr_tiny_feat32_zyx_l4_in32",
        transformer_dim=64,
        transformer_heads=4,
        transformer_layers=1,
        decoder_hidden=32,
        **kwargs,  # type: ignore[arg-type]
    )


def _small_batch() -> torch.Tensor:
    """Return a fixed random I/Q batch ``(2, D, A, E, R, 2)``.

    Returns:
        The batch.
    """
    torch.manual_seed(1)
    return torch.randn(2, *SMALL_SHAPE, 2)


def _assert_forward_and_loss(model: HRNetMultiPhaseBetaVAE) -> None:
    """Check output shapes, a finite loss and a working backward pass.

    Args:
        model: Model under test.
    """
    x = _small_batch()
    targets = model.compute_targets(x)
    projections, mu, logvar = model(x)
    assert set(projections) == set(targets)
    for name, pred in projections.items():
        assert pred.shape == targets[name].shape
    loss, metrics = multi_projection_loss(projections, targets, mu, logvar, beta=0.1)
    assert torch.isfinite(loss)
    loss.backward()


class TestDefaultArchitecture:
    """Default kwargs build the paper architecture."""

    def test_six_decoders_and_hrnet(self) -> None:
        """All six projection decoders and the HRNet backbone are built."""
        model = _small_model()
        assert set(model.decoders) == set(PROJECTIONS)
        assert type(model.backbone).__name__ == "HighResolution3DNet"
        assert model.projection_set is None
        assert model.reconstruction_target == "projections"

    def test_default_targets_cover_all_projections(self) -> None:
        """Targets exist for every projection."""
        model = _small_model()
        targets = model.compute_targets(_small_batch())
        assert set(targets) == set(PROJECTIONS)


class TestProjectionSubset:
    """Tests for ``projection_set``."""

    def test_range_doppler_only(self) -> None:
        """A single-projection model trains end to end."""
        model = _small_model(projection_set=["range_doppler"])
        assert list(model.decoders) == ["range_doppler"]
        _assert_forward_and_loss(model)

    def test_unknown_projection_rejected(self) -> None:
        """Unknown projection names raise."""
        with pytest.raises(ValueError, match="Unknown projection"):
            _small_model(projection_set=["range_doppler", "bogus"])


class TestUnsupportedOptions:
    """Options without an implementation are rejected."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"reconstruction_target": "full_cube"},
            {"doppler_gate": True},
            {"doppler_se": True},
            {"backbone_type": "resnet"},
        ],
    )
    def test_rejected(self, kwargs: dict[str, object]) -> None:
        """The constructor raises for an unsupported option.

        Args:
            kwargs: Constructor arguments selecting the option.
        """
        with pytest.raises(ValueError):
            _small_model(**kwargs)


class _WandbStub:
    """Minimal stand-in for the ``wandb`` module that records logged images."""

    def __init__(self) -> None:
        self.logged: dict[str, object] = {}

    @staticmethod
    def Image(fig: object) -> str:
        """Return a placeholder for a logged figure.

        Args:
            fig: Matplotlib figure.

        Returns:
            Placeholder string.
        """
        return "img"

    def log(self, images: dict[str, object], step: int) -> None:
        """Record logged entries.

        Args:
            images: Logged entries.
            step: Global step (ignored).
        """
        self.logged.update(images)


class TestImageLogging:
    """Tests for the wandb projection-image logger."""

    def test_projection_images(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One image is logged per 2D projection.

        Args:
            monkeypatch: Pytest fixture used to replace ``wandb``.
        """
        pytest.importorskip("matplotlib")
        stub = _WandbStub()
        monkeypatch.setattr(multi_trainer, "wandb", stub)
        proj = torch.rand(16, 32)
        multi_trainer._log_projection_images(
            {"range_doppler": proj}, {"range_doppler": proj.clone()}, step=0,
        )
        assert "val/images/range_doppler" in stub.logged


class TestLatentDim:
    """Tests for the latent bottleneck."""

    def test_small_latent_bottleneck(self) -> None:
        """``encode`` returns the latent mean ``(B, latent_dim)``."""
        model = _small_model()
        x = _small_batch()
        mu = model.encode(x)
        assert mu.shape == (2, LATENT)


class TestConfigs:
    """Tests for config validation and the released pretraining configs."""

    def test_invalid_reconstruction_target_rejected(self) -> None:
        """An unknown reconstruction target is rejected by the config."""
        with pytest.raises(ValueError, match="reconstruction_target"):
            ModelConfig(reconstruction_target="cube")

    def test_invalid_backbone_rejected(self) -> None:
        """An unknown backbone is rejected by the config."""
        with pytest.raises(ValueError, match="backbone_type"):
            ModelConfig(backbone_type="resnet")

    def test_empty_projection_set_rejected(self) -> None:
        """An empty projection set is rejected by the config."""
        with pytest.raises(ValueError, match="projection_set"):
            ModelConfig(projection_set=[])

    @pytest.mark.parametrize(
        ("name", "sources"),
        [
            ("pool_mcag", ["mmradpose", "cubelearn", "cubelearn_hgr", "cubelearn_agr",
                           "ci4r", "multi3"]),
            ("pool_mca", ["mmradpose", "cubelearn", "cubelearn_hgr", "cubelearn_agr", "ci4r"]),
            ("pool_cag", ["cubelearn", "cubelearn_hgr", "cubelearn_agr", "ci4r", "multi3"]),
            ("pool_mag", ["mmradpose", "ci4r", "multi3"]),
            ("pool_mcg", ["mmradpose", "cubelearn", "cubelearn_hgr", "cubelearn_agr", "multi3"]),
        ],
    )
    def test_pool_configs_resolve(self, name: str, sources: list[str]) -> None:
        """Every pool config resolves onto the shared base recipe.

        Args:
            name: Pool config name.
            sources: Expected pretraining sources.
        """
        cfg = load_config(f"configs/pretrain/{name}.yaml")
        assert cfg.dataset.sources == sources
        assert len(cfg.dataset.mixture_weights) == len(sources)
        assert abs(sum(cfg.dataset.mixture_weights) - 1.0) < 1e-5
        assert cfg.training.checkpoint_dir.endswith(name)
        assert cfg.wandb.run_name == name

    def test_recipe_is_the_published_one(self) -> None:
        """The base recipe carries the published hyperparameters."""
        cfg = load_config("configs/pretrain/pool_mcag.yaml")
        assert cfg.model.encoder_type == "hrnet_multi_phase"
        assert cfg.model.latent_dim == 256
        assert cfg.model.projection_set is None
        assert cfg.model.reconstruction_target == "projections"
        assert cfg.model.backbone_type == "hrnet"
        assert cfg.model.n_input_channels == 2
        assert cfg.dataset.dims == [128, 4, 4, 256]
        assert cfg.dataset.format_mode == "iq"
        assert cfg.dataset.clutter_removal is True
        assert cfg.dataset.normalize_magnitude is True
        assert cfg.training.beta == 0.1
        assert cfg.training.beta_warmup_epochs == 5
        assert cfg.training.epochs == 50
        assert cfg.training.batch_size == 16
        assert cfg.training.steps_per_epoch == 169000
        assert cfg.training.lr == 3e-4
        assert cfg.training.weight_decay == 0.01
        assert cfg.training.grad_clip_norm == 1.0
