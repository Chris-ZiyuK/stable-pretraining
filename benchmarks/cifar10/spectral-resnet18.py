"""Spectral contrastive learning on CIFAR-10, with a SimCLR control.

Follows the CIFAR-10 recipe of the authors' code for HaoChen et al., "Provable
Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss"
(NeurIPS 2021): CIFAR-style ResNet-18, a BatchNorm MLP projector
(512 -> 2048 -> 1000), SGD with momentum 0.9 and weight decay 5e-4, batch size
512, a learning rate of ``0.03 * 512 / 256`` with 10 warmup epochs and a cosine
decay to zero, and the SimSiam augmentations. The paper trains for 800 epochs.

Settings are read from environment variables: ``MAX_EPOCHS`` (default 200),
``MU`` (spectral loss radius squared, default 1) and ``LOSS`` (``spectral`` or
``simclr``). ``LOSS=simclr`` swaps in the SimCLR method with everything else
unchanged.
"""

import os
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods import SimCLR, SpectralCL


def spectral_forward(self, batch, stage):
    """Forward for two-view methods: SpectralCL and the SimCLR control.

    Batch format:
        - Training: dict with two named views, ``"view_0"`` and ``"view_1"``
        - Eval: single dict with an ``"image"`` key

    Returns:
        Dictionary with ``"embedding"``, ``"label"`` and, in training, ``"loss"``.
    """
    out = {}

    if self.training:
        view_0, view_1 = batch["view_0"], batch["view_1"]
        output = self.model(view_0["image"], view_1["image"])
        out["label"] = torch.cat([view_0["label"], view_1["label"]])
        out["loss"] = output.loss
        self.log(
            f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True
        )
    else:
        output = self.model(batch["image"])
        out["label"] = batch["label"]

    out["embedding"] = output.embedding
    return out


def _view_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

    # No per-epoch requeue checkpoint: this benchmark saves no checkpoints.
    spt.set(requeue_checkpoint=False)

    batch_size = 512
    num_workers = int(os.environ.get("NUM_WORKERS", 8))
    max_epochs = int(os.environ.get("MAX_EPOCHS", 200))
    warmup_epochs = 10
    mu = float(os.environ.get("MU", 1.0))
    loss_name = os.environ.get("LOSS", "spectral")
    base_lr = 0.03

    train_transform = transforms.MultiViewTransform(
        {"view_0": _view_transform(), "view_1": _view_transform()}
    )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((32, 32)),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )

    data_dir = get_data_dir("cifar10")
    cifar_train = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True
    )
    cifar_val = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True
    )

    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.FromTorchDataset(
                cifar_train, names=["image", "label"], transform=train_transform
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=num_workers > 0,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=spt.data.FromTorchDataset(
                cifar_val, names=["image", "label"], transform=val_transform
            ),
            batch_size=256,
            num_workers=num_workers,
        ),
    )

    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    backbone.fc = nn.Identity()

    if loss_name == "spectral":
        model = SpectralCL(encoder_name=backbone, projector_dims=(2048, 1000), mu=mu)
    elif loss_name == "simclr":
        model = SimCLR(
            encoder_name=backbone, projector_dims=(2048, 1000), temperature=0.5
        )
    else:
        raise ValueError(f"LOSS must be 'spectral' or 'simclr', got {loss_name!r}")

    module = spt.Module(
        model=model,
        forward=spectral_forward,
        optim={
            "optimizer": {
                "type": "SGD",
                "lr": base_lr * batch_size / 256,
                "momentum": 0.9,
                "weight_decay": 5e-4,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "peak_step": warmup_epochs / max_epochs,
                "start_factor": 0.01,
                "end_lr": 0.0,
                "total_steps": len(data.train) * max_epochs,
            },
            "interval": "step",
        },
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        callbacks=[
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(model.embed_dim, 10),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(10),
                    "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
                },
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=20000,
                metrics={
                    "accuracy": torchmetrics.classification.MulticlassAccuracy(10)
                },
                input_dim=model.embed_dim,
                k=10,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=False,
        precision="16-mixed",
        enable_checkpointing=False,
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data, seed=0)
    manager()


if __name__ == "__main__":
    main()
