# CIFAR-10 ResNet-18: Spectral Contrastive Learning vs SimCLR — 200 Epochs

`spectral-resnet18.py` trains a CIFAR-style ResNet-18 with the spectral
contrastive loss of HaoChen et al. (NeurIPS 2021), with `--loss simclr` as a
control. The recipe follows the authors' CIFAR-10 code: batch 512, SGD with
momentum 0.9 and weight decay 5e-4, learning rate `0.03 * 512 / 256` with 10
warmup epochs and a cosine decay to zero, the SimSiam augmentations, and a
BatchNorm MLP projector (512 -> 2048 -> 1000). Everything but the loss is
identical across the three runs below. Each run is **200 epochs**, one seed
(0), mixed precision, on one GPU (about 53 minutes on an A40).

## Top-1 accuracy table

Accuracy on the CIFAR-10 test set after epoch 200, from the online linear
probe and the online 10-NN probe attached to the backbone embeddings.

| Method | Loss setting | KNN top-1 | Linear top-1 |
|---|---|---:|---:|
| SimCLR         | NT-Xent, temperature 0.5      | 83.0% | 83.5% |
| Spectral (μ=1)  | spectral, ball radius 1       | 81.6% | 81.0% |
| Spectral (μ=10) | spectral, ball radius √10     | **83.8%** | **84.2%** |

Last-5-epoch mean of the linear probe: 83.5% / 81.1% / 84.2%.

## Reproducing

```bash
python benchmarks/cifar10/spectral-resnet18.py --loss simclr --epochs 200
python benchmarks/cifar10/spectral-resnet18.py --loss spectral --mu 1 --epochs 200
python benchmarks/cifar10/spectral-resnet18.py --loss spectral --mu 10 --epochs 200
```

## Comparison with the paper

The paper (Table 1 and 3, ResNet-18, CIFAR-10, 200 epochs, linear evaluation)
reports:

| Method | Paper linear top-1 | This run (online probe) |
|---|---:|---:|
| SimCLR (paper's reproduction) | 83.7% | 83.5% |
| Spectral (μ=1)  | 86.5% | 81.0% |
| Spectral (μ=10) | 88.7% | 84.2% |

- The SimCLR control matches the paper's SimCLR number, so the training and
  evaluation setup behaves as expected.
- Spectral with μ=10 beats SimCLR here (+0.7 linear, +0.9 kNN) and μ=10 beats
  μ=1, the same ordering as in the paper. The margin over SimCLR is smaller than
  the paper's.
- Spectral with μ=1 ends below SimCLR here, whereas it is above SimCLR in the
  paper.

## Caveats

- Accuracy comes from the **online probes** trained during pretraining on
  detached embeddings of augmented views. The paper evaluates a frozen backbone
  with a separately trained linear classifier (SGD, learning rate 30, 100
  epochs), so absolute numbers are not directly comparable.
- Single seed per setting; differences of a point or less are within what a
  second seed could change.
- The absolute Spectral numbers are lower than the paper's. We did not run the
  paper's offline evaluation, and we did not identify the cause of the gap.
