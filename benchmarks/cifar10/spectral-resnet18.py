"""Spectral contrastive learning on CIFAR-10, with a SimCLR control.

Follows the CIFAR-10 recipe of the authors' code for HaoChen et al., "Provable
Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss"
(NeurIPS 2021): CIFAR-style ResNet-18, a BatchNorm MLP projector
(512 -> 2048 -> 1000), SGD with momentum 0.9 and weight decay 5e-4, batch size
512, a learning rate of ``0.03 * batch_size / 256`` with 10 warmup epochs and a
cosine decay to zero, and the SimSiam augmentations. The paper trains for 800
epochs; ``--loss simclr`` swaps in NT-Xent with everything else unchanged.

Usage::

    python benchmarks/cifar10/spectral-resnet18.py --loss spectral --mu 1
    python benchmarks/cifar10/spectral-resnet18.py --loss simclr --temperature 0.5
"""

import argparse
import sys
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor
from torch import nn

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--loss", choices=("spectral", "simclr"), default="spectral")
parser.add_argument("--mu", type=float, default=1.0, help="spectral loss radius^2")
parser.add_argument("--temperature", type=float, default=0.5, help="NT-Xent only")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--warmup-epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--base-lr", type=float, default=0.03, help="lr at batch size 256")
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


def make_view_transform():
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


train_transform = transforms.MultiViewTransform(
    [make_view_transform(), make_view_transform()]
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
cifar_val = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],
    transform=train_transform,
)
val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=val_transform,
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    drop_last=True,
    shuffle=True,
    persistent_workers=args.num_workers > 0,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=args.num_workers,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

backbone = spt.backbone.from_torchvision(
    "resnet18",
    low_resolution=True,
)
backbone.fc = torch.nn.Identity()

projector = nn.Sequential(
    nn.Linear(512, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 1000),
    nn.BatchNorm1d(1000),
)

if args.loss == "spectral":
    forward_fn = forward.spectral_cl
    loss_modules = {"spectral_loss": spt.losses.SpectralContrastiveLoss(mu=args.mu)}
else:
    forward_fn = forward.simclr
    loss_modules = {"simclr_loss": spt.losses.NTXEntLoss(temperature=args.temperature)}

total_steps = args.epochs * len(train_dataloader)

module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward_fn,
    **loss_modules,
    optim={
        "optimizer": {
            "type": "SGD",
            "lr": args.base_lr * args.batch_size / 256,
            "momentum": 0.9,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "total_steps": total_steps,
            "peak_step": args.warmup_epochs * len(train_dataloader),
            "start_factor": 0.01,
            "end_lr": 0.0,
        },
        "interval": "step",
    },
)

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 10),
    loss=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
    input_dim=512,
    k=10,
)

trainer = pl.Trainer(
    max_epochs=args.epochs,
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe, LearningRateMonitor(logging_interval="step")],
    precision="16-mixed",
    logger=False,
    enable_checkpointing=False,
)

manager = spt.Manager(trainer=trainer, module=module, data=data, seed=args.seed)
manager()
