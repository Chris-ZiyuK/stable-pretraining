"""Unit tests for the spectral contrastive loss, SpectralCL, and its forward fn."""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer

from stable_pretraining.forward import spectral_cl
from stable_pretraining.losses import SpectralContrastiveLoss
from stable_pretraining.methods import SpectralCL

pytestmark = pytest.mark.unit

LOSS_MODULE = "stable_pretraining.losses.spectral_contrastive"


def project_to_ball(z, mu):
    """Keep rows inside the ball of radius sqrt(mu); pull the others onto it."""
    radius = mu**0.5
    rows = [r if r.norm() < radius else F.normalize(r, dim=0) * radius for r in z]
    return torch.stack(rows)


def reference_loss(z1, z2, mu):
    """Algorithm 1 of HaoChen et al. (NeurIPS 2021) as plain loops.

    Includes the authors' ball projection and their division by mu.
    """
    z1, z2 = z1.double(), z2.double()
    if mu is not None:
        z1, z2 = project_to_ball(z1, mu), project_to_ball(z2, mu)
    n = z1.size(0)
    positive = sum(z1[i] @ z2[i] for i in range(n)) / n
    negative = sum(
        (z1[i] @ z2[j]) ** 2 for i in range(n) for j in range(n) if i != j
    ) / (n * (n - 1))
    loss = -2 * positive + negative
    return loss if mu is None else loss / mu


def encoder():
    return VisionTransformer(
        img_size=32,
        patch_size=8,
        embed_dim=24,
        depth=2,
        num_heads=3,
        num_classes=0,
        dynamic_img_size=True,
    )


@pytest.fixture
def views():
    """Two views with row norms spanning roughly 0.4 to 16.

    For mu in {1, 10, 100} some rows then fall inside the ball and some outside.
    """
    g = torch.Generator().manual_seed(0)
    scale = torch.tensor([0.1, 0.3, 0.6, 1.0, 2.0, 4.0]).unsqueeze(1)
    return (
        scale * torch.randn(6, 16, generator=g),
        scale * torch.randn(6, 16, generator=g),
    )


class TestSpectralContrastiveLoss:
    """The loss matches the paper's estimator and handles edge cases."""

    @pytest.mark.parametrize("mu", [None, 0.5, 1.0, 10.0, 100.0])
    def test_matches_algorithm_1_reference(self, views, mu):
        z1, z2 = views
        loss = SpectralContrastiveLoss(mu=mu)(z1, z2)
        torch.testing.assert_close(
            loss.double(), reference_loss(z1, z2, mu), rtol=1e-4, atol=1e-4
        )

    def test_default_mu_is_one(self):
        assert SpectralContrastiveLoss().mu == 1.0

    def test_orthonormal_aligned_embeddings_reach_the_minimum(self):
        z = torch.eye(5, 8)
        assert SpectralContrastiveLoss(mu=None)(z, z).item() == pytest.approx(-2.0)

    def test_the_minimum_is_independent_of_mu_on_the_ball_boundary(self):
        mu = 3.0
        z = mu**0.5 * torch.eye(5, 8)
        assert SpectralContrastiveLoss(mu=mu)(z, z).item() == pytest.approx(-2.0)

    def test_rows_outside_the_ball_lose_their_scale(self, views):
        z1, z2 = views
        loss = SpectralContrastiveLoss(mu=0.01)
        torch.testing.assert_close(loss(z1, z2), loss(3.0 * z1, 2.0 * z2))

    def test_rows_inside_the_ball_are_left_untouched(self, views):
        z1, z2 = views
        mu = 1e4
        torch.testing.assert_close(
            SpectralContrastiveLoss(mu=mu)(z1, z2),
            SpectralContrastiveLoss(mu=None)(z1, z2) / mu,
        )

    def test_without_mu_the_scale_matters(self, views):
        z1, z2 = views
        loss = SpectralContrastiveLoss(mu=None)
        assert not torch.allclose(loss(z1, z2), loss(3.0 * z1, 3.0 * z2))

    def test_symmetric_in_the_two_views(self, views):
        z1, z2 = views
        loss = SpectralContrastiveLoss()
        torch.testing.assert_close(loss(z1, z2), loss(z2, z1))

    def test_backward_gives_finite_nonzero_gradients(self, views):
        z1, z2 = (v.clone().requires_grad_() for v in views)
        SpectralContrastiveLoss()(z1, z2).backward()
        for z in (z1, z2):
            assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0

    def test_loss_is_float32_under_autocast(self, views):
        z1, z2 = (v.bfloat16() for v in views)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = SpectralContrastiveLoss(mu=10.0)(z1, z2)
        assert loss.dtype == torch.float32
        torch.testing.assert_close(
            loss.double(), reference_loss(z1, z2, 10.0), rtol=1e-3, atol=1e-3
        )

    def test_gathered_negatives_match_the_single_process_loss(self):
        g = torch.Generator().manual_seed(1)
        scale = torch.tensor([0.2, 0.5, 1.0, 2.0, 0.3, 0.7, 1.5, 3.0]).unsqueeze(1)
        z1 = scale * torch.randn(8, 16, generator=g)
        z2 = scale.flip(0) * torch.randn(8, 16, generator=g)
        loss_fn = SpectralContrastiveLoss(mu=10.0)
        full = loss_fn(z1, z2)

        # The loss gathers z_j after projecting it, so the fake collective must
        # hand back projected shards too.
        gathered = [project_to_ball(s, 10.0).float() for s in z2.chunk(2)]
        per_rank = []
        for rank in range(2):
            with (
                patch(f"{LOSS_MODULE}.all_gather", return_value=gathered),
                patch(f"{LOSS_MODULE}.get_rank", return_value=rank),
            ):
                per_rank.append(loss_fn(z1.chunk(2)[rank], z2.chunk(2)[rank]))
        torch.testing.assert_close(torch.stack(per_rank).mean(), full)

    @pytest.mark.parametrize("mu", [0.0, -1.0])
    def test_rejects_non_positive_mu(self, mu):
        with pytest.raises(ValueError, match="mu"):
            SpectralContrastiveLoss(mu=mu)

    @pytest.mark.parametrize(
        "shape1,shape2",
        [((4, 8), (4, 9)), ((4, 8), (3, 8)), ((4, 8, 1), (4, 8, 1))],
    )
    def test_rejects_mismatched_or_non_2d_inputs(self, shape1, shape2):
        with pytest.raises(ValueError, match="2-D"):
            SpectralContrastiveLoss()(torch.randn(*shape1), torch.randn(*shape2))

    def test_rejects_a_single_sample(self):
        with pytest.raises(ValueError, match="at least 2"):
            SpectralContrastiveLoss()(torch.randn(1, 8), torch.randn(1, 8))


class TestSpectralCL:
    """The method class wires backbone, projector, and loss together."""

    def test_train_forward_shapes_and_backward(self):
        model = SpectralCL(
            encoder_name=encoder(), projector_dims=(16, 8), mu=7.0
        ).train()
        v1, v2 = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
        out = model(v1, v2)

        assert model.spectral_loss.mu == 7.0
        assert out.embedding.shape == (8, 24)
        assert out.projection.shape == (8, 8)
        assert out.loss.ndim == 0 and torch.isfinite(out.loss)
        out.loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)

    def test_eval_returns_embedding_without_a_loss(self):
        model = SpectralCL(encoder_name=encoder(), projector_dims=(16, 8)).eval()
        with torch.no_grad():
            out = model(torch.randn(3, 3, 32, 32))
        assert out.embedding.shape == (3, 24)
        assert out.projection is None and out.loss.item() == 0

    def test_projector_needs_an_output_dimension(self):
        with pytest.raises(ValueError, match="at least one"):
            SpectralCL(encoder_name=encoder(), projector_dims=())


class TestSpectralCLForward:
    """The functional form mirrors the ``simclr`` forward contract."""

    @staticmethod
    def make_module(training):
        module = Mock()
        module.training = training
        module.backbone.side_effect = lambda x: torch.randn(x.size(0), 12)
        module.projector.side_effect = lambda h: torch.randn(h.size(0), 6)
        module.spectral_loss = SpectralContrastiveLoss()
        return spectral_cl.__get__(module, type(module)), module

    @staticmethod
    def two_views(batch=4):
        return {
            "views": [
                {"image": torch.randn(batch, 3, 8, 8), "label": torch.arange(batch)}
                for _ in range(2)
            ]
        }

    def test_training_returns_loss_embedding_and_label(self):
        forward, module = self.make_module(training=True)
        out = forward(self.two_views(), "train")

        assert module.backbone.call_count == 2
        assert module.projector.call_count == 2
        assert out["embedding"].shape == (8, 12) and out["label"].shape == (8,)
        assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
        module.log.assert_called_once()

    def test_eval_skips_projector_and_loss(self):
        forward, module = self.make_module(training=False)
        out = forward(self.two_views(), "val")

        assert "loss" not in out
        module.projector.assert_not_called()

    def test_single_view_batch_returns_embedding_and_label(self):
        forward, _ = self.make_module(training=False)
        out = forward(
            {"image": torch.randn(4, 3, 8, 8), "label": torch.arange(4)}, "val"
        )
        assert out["embedding"].shape == (4, 12) and out["label"].shape == (4,)

    def test_requires_exactly_two_views(self):
        forward, _ = self.make_module(training=True)
        batch = self.two_views()
        batch["views"].append(batch["views"][0])
        with pytest.raises(ValueError, match="exactly 2 views"):
            forward(batch, "train")
