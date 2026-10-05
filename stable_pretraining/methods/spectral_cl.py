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
from stable_pretraining.losses import SpectralContrastiveLoss
from stable_pretraining.methods.simclr import _build_projector


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
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 1000),
        mu: float = 1.0,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()

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

        self.projector = _build_projector(
            embed_dim, list(projector_dims), final_bn_no_bias=False
        )
        self.spectral_loss = SpectralContrastiveLoss(mu=mu)

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> SpectralCLOutput:
        """Forward pass.

        :param view1: First augmented view [B, C, H, W] (or single view at eval).
        :param view2: Second augmented view [B, C, H, W]. If ``None``, returns
            only the backbone embedding (eval mode).
        :return: :class:`SpectralCLOutput`.
        """
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
