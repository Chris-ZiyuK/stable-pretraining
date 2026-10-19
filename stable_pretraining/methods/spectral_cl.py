"""Spectral Contrastive Learning.

Contrastive self-supervised learning with the spectral contrastive loss: positive
pairs are pulled together while squared inner products of independent pairs are
pushed to zero. It needs no temperature, momentum encoder, or stop-gradient.

References:
    HaoChen et al., "Provable Guarantees for Self-Supervised Deep Learning with
    Spectral Contrastive Loss", NeurIPS 2021. https://arxiv.org/abs/2106.04156

Example::

    from stable_pretraining.methods import SpectralCL
    import lightning as pl

    model = SpectralCL(encoder_name="resnet18", low_resolution=True)

    trainer = pl.Trainer(max_epochs=200)
    trainer.fit(model, dataloader)
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.utils import all_gather, get_rank


def _project_to_ball(z: torch.Tensor, radius: float) -> torch.Tensor:
    """Rescale rows whose norm exceeds ``radius`` onto the sphere of that radius.

    Rows already inside the ball are left untouched, which is what makes this a
    projection onto the ball rather than a normalisation onto its boundary.
    """
    norm = z.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(z.dtype).tiny)
    return z * (radius / norm).clamp(max=1.0)


class SpectralContrastiveLoss(nn.Module):
    """Spectral contrastive loss over two views of a batch.

    For a batch of ``N`` positive pairs ``(z_i, z'_i)`` the loss is::

        L = -2/N * sum_i z_i . z'_i  +  1/(N(N-1)) * sum_{i != j} (z_i . z'_j)^2

    Unlike InfoNCE it has no temperature and no log-sum-exp: the second term is
    the mean of squared inner products of independent pairs.

    Following the authors' implementation, embeddings are first projected onto
    the ball of radius ``sqrt(mu)`` (rows that already lie inside it are kept
    as they are) and the loss is divided by ``mu``. Both terms grow with the
    squared embedding norm, so the division keeps the loss, and therefore the
    gradient scale, comparable across values of ``mu``.

    Under distributed training the ``z'`` candidates are gathered from every
    rank, so each local anchor is contrasted against the global batch. With a
    single process this is exactly the estimator in Algorithm 1 of the paper.

    Args:
        mu: Squared radius of the ball the embeddings are projected into, and the
            divisor of the loss. ``None`` disables both and uses the embeddings
            and the loss exactly as given.

    Note:
        Reference: HaoChen et al., NeurIPS 2021, https://arxiv.org/abs/2106.04156.
        The ball projection and the division by ``mu`` follow the authors' code,
        https://github.com/jhaochenz96/spectral_contrastive_learning.
    """

    def __init__(self, mu: Optional[float] = 1.0) -> None:
        super().__init__()
        if mu is not None and mu <= 0:
            raise ValueError(f"mu must be positive or None, got {mu}")
        self.mu = mu

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute the spectral contrastive loss.

        Args:
            z_i: Embeddings of the first view, shape ``[N, D]``.
            z_j: Embeddings of the second view, shape ``[N, D]``. Row ``k`` of
                ``z_j`` is the positive of row ``k`` of ``z_i``.

        Returns:
            Scalar loss in float32.

        Raises:
            ValueError: If the views differ in shape, are not 2-D, or the global
                batch holds fewer than two samples (no negative pairs exist).
        """
        if z_i.ndim != 2 or z_i.shape != z_j.shape:
            raise ValueError(
                "z_i and z_j must be 2-D tensors of the same shape, got "
                f"{tuple(z_i.shape)} and {tuple(z_j.shape)}"
            )

        # The squared term is the loss's only scale-sensitive part and overflows
        # in fp16, so it must not run under an enclosing autocast region.
        with torch.autocast(device_type=z_i.device.type, enabled=False):
            z_i = z_i.float()
            z_j = z_j.float()
            if self.mu is not None:
                radius = self.mu**0.5
                z_i = _project_to_ball(z_i, radius)
                z_j = _project_to_ball(z_j, radius)

            candidates = torch.cat(all_gather(z_j), dim=0)
            n, m = z_i.size(0), candidates.size(0)
            if m < 2:
                raise ValueError("SpectralContrastiveLoss needs at least 2 samples")

            positive = (z_i * z_j).sum(dim=-1).mean()

            # Row a of the local block sits at column ``offset + a`` of the
            # gathered candidates; that entry is its positive, not a negative.
            offset = get_rank() * n
            own = torch.arange(n, device=z_i.device)
            is_positive = torch.zeros(n, m, dtype=torch.bool, device=z_i.device)
            is_positive[own, offset + own] = True

            squared = (z_i @ candidates.T).pow(2).masked_fill(is_positive, 0.0)
            negative = squared.sum() / (n * (m - 1))

            loss = -2.0 * positive + negative
            if self.mu is not None:
                loss = loss / self.mu
        return loss


@dataclass
class SpectralCLOutput(ModelOutput):
    """Output from SpectralCL forward pass.

    :ivar loss: Spectral contrastive loss (0 in eval mode)
    :ivar embedding: Backbone features [B, D] in eval mode, [2B, D] in train mode
    :ivar projection: Projector outputs [2B, P] (None in eval mode)
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None


def _views_from_batch(batch: dict) -> Optional[list[dict]]:
    """Return the view dicts of a multi-view batch, or None for a single view.

    Accepts both outputs of ``MultiViewTransform``: ``{"views": [...]}`` for a
    list of transforms and ``{name: view}`` for a dict of transforms. Entries
    that are not views, such as the ``batch_idx`` injected by ``Module``, are
    ignored.
    """
    if "views" in batch:
        return list(batch["views"])
    if "image" in batch:
        return None
    return [v for v in batch.values() if isinstance(v, dict) and "image" in v]


def _build_spectral_projector(in_dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    """MLP projector: Linear -> BN -> ReLU -> ... -> Linear -> BN.

    :param in_dim: Backbone output dimension
    :param hidden_dims: Sequence of hidden + output dimensions, e.g. (2048, 1000)
    """
    if len(hidden_dims) < 1:
        raise ValueError("hidden_dims must contain at least one entry (the output dim)")
    layers = []
    prev = in_dim
    for i, dim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev, dim, bias=False))
        layers.append(nn.BatchNorm1d(dim))
        if i < len(hidden_dims) - 1:
            layers.append(nn.ReLU(inplace=True))
        prev = dim
    return nn.Sequential(*layers)


class SpectralCL(Module):
    """Spectral Contrastive Learning.

    Architecture:
        - **Backbone**: any feature extractor producing a flat [B, D] embedding
          (timm ViT/ResNet with the head removed)
        - **Projector**: MLP with BatchNorm after every layer and ReLU after all
          but the last, as in the paper
        - **Loss**: spectral contrastive loss on projections confined to the
          ball of radius ``sqrt(mu)``

    :param encoder_name: timm model name (e.g. ``"resnet18"``,
        ``"vit_small_patch16_224"``) or a pre-instantiated ``nn.Module`` whose
        ``forward`` returns a ``[B, D]`` tensor.
    :param projector_dims: Hidden + output dimensions of the MLP projector. The
        default ``(2048, 1000)`` is what the authors' CIFAR-10 code builds; the
        paper text describes 1000 hidden units.
    :param mu: Squared radius of the ball the projections are confined to. The
        paper tunes it over ``{1, 3, 10}`` on CIFAR and uses 10 on ImageNet.
    :param low_resolution: Adapt first conv for 32x32 inputs (CIFAR-style).
    :param pretrained: Load pretrained timm weights for the encoder.
    :param optim: Optimizer configuration in the format of
        :class:`~stable_pretraining.Module`. Needed to train the model directly
        through ``Trainer`` / ``Manager``.

    The model supports two calling conventions:

    - **Tensors**: ``model(view1, view2)`` returns a :class:`SpectralCLOutput`.
    - **Batch dicts**: ``model(batch, stage=...)`` is the contract used by
      :class:`~stable_pretraining.Module`, so the model trains directly with
      ``Trainer`` / ``Manager`` on batches from ``MultiViewTransform``. It
      returns a dict with ``"embedding"``, ``"label"`` when the views carry
      labels, and ``"loss"`` in training mode.

    Example::

        model = SpectralCL(
            encoder_name="resnet18",
            projector_dims=(2048, 1000),
            mu=1.0,
            low_resolution=True,
        )

        v1 = torch.randn(64, 3, 32, 32)
        v2 = torch.randn(64, 3, 32, 32)
        out = model(v1, v2)
        out.loss.backward()

        # eval: single view, no loss
        model.eval()
        out = model(v1)
        features = out.embedding  # [64, embed_dim]

    Example with Manager::

        model = SpectralCL(
            encoder_name="resnet18",
            low_resolution=True,
            optim={"optimizer": {"type": "SGD", "lr": 0.06, "momentum": 0.9}},
        )
        trainer = pl.Trainer(max_epochs=200)
        spt.Manager(trainer=trainer, module=model, data=data)()
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 1000),
        mu: float = 1.0,
        low_resolution: bool = False,
        pretrained: bool = False,
        optim: Optional[dict] = None,
    ):
        super().__init__(**({} if optim is None else {"optim": optim}))

        if isinstance(encoder_name, str):
            self.backbone = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            self.backbone = encoder_name

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            embed_dim = self.backbone(dummy).shape[-1]
        self.embed_dim = embed_dim

        self.projector = _build_spectral_projector(embed_dim, list(projector_dims))
        self.spectral_loss = SpectralContrastiveLoss(mu=mu)

    def forward(
        self,
        view1: Union[torch.Tensor, dict],
        view2: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ) -> Union[SpectralCLOutput, dict[str, torch.Tensor]]:
        """Forward pass.

        :param view1: First augmented view [B, C, H, W] (or single view at eval),
            or a batch dict, in which case ``view2`` must be ``None``.
        :param view2: Second augmented view [B, C, H, W]. If ``None``, returns
            only the backbone embedding (eval mode).
        :param stage: Lightning stage, only used to name the logged loss when a
            batch dict is given.
        :return: :class:`SpectralCLOutput` for tensors, a dict for a batch dict.
        """
        if isinstance(view1, dict):
            return self._forward_batch(view1, stage)

        if view2 is None:
            embedding = self.backbone(view1)
            return SpectralCLOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
                projection=None,
            )

        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        loss = self.spectral_loss(z1, z2)
        return SpectralCLOutput(
            loss=loss,
            embedding=torch.cat([h1, h2], dim=0),
            projection=torch.cat([z1, z2], dim=0),
        )

    def _forward_batch(self, batch: dict, stage: Optional[str]) -> dict:
        """Forward a batch dict, as :class:`~stable_pretraining.Module` calls it.

        Training needs exactly two views. At eval a single ``{"image": ...}`` batch
        or any number of views is embedded, and no loss is returned.
        """
        views = _views_from_batch(batch)
        out = {}

        if not self.training:
            if views is None:
                out["embedding"] = self.backbone(batch["image"])
                if "label" in batch:
                    out["label"] = batch["label"]
            else:
                out["embedding"] = torch.cat(
                    [self.backbone(v["image"]) for v in views], dim=0
                )
                if views and "label" in views[0]:
                    out["label"] = torch.cat([v["label"] for v in views], dim=0)
            return out

        if views is None or len(views) != 2:
            found = 1 if views is None else len(views)
            raise ValueError(
                f"SpectralCL training needs exactly 2 views per sample, got {found}"
            )

        output = self(views[0]["image"], views[1]["image"])
        out["loss"] = output.loss
        out["embedding"] = output.embedding
        if "label" in views[0]:
            out["label"] = torch.cat([v["label"] for v in views], dim=0)
        if self._trainer is not None:
            self.log(
                f"{stage or 'fit'}/loss",
                output.loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        return out
