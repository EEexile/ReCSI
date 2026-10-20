"""Compact sentence-level residual MLP; context enters in forest inference.

Use compat.compact_head_bundle to convert an existing historical head. Fitting
the archived experiments uses the separate compatibility parameterization.
"""
import torch
from torch import nn


class ResidualCorrector(nn.Module):
    def __init__(self, classes=7):
        super().__init__()
        if classes not in (4, 7):
            raise ValueError("The released protocols use 4 or 7 classes")
        self.classes = classes
        self.current = nn.Sequential(nn.Linear(3072 + classes, 64), nn.ReLU())
        self.output = nn.Linear(64, classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, current):
        if current.ndim != 2 or current.shape[1] != 3072 + self.classes:
            raise ValueError("Expected concatenated current audio/text/fused/logit vectors")
        return self.output(self.current(current))


def fit_normalizer(x):
    if x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError("Normalization requires finite FP32 TRAIN features")
    return {"mean": x.mean(0), "std": x.std(0, unbiased=False).clamp_min(1e-4)}


def normalize(x, normalizer):
    return (x - normalizer["mean"]) / normalizer["std"]


def unary_probability(logits, dataset):
    # Match the distinct historical source exporters; preserve this distinction.
    if dataset == "MELD":
        p = logits.float().softmax(1).cpu().double()
    elif dataset == "IEMOCAP":
        p = logits.cpu().double().softmax(1)
    else:
        raise ValueError(dataset)
    p = p.clamp_min(1e-12)
    return p / p.sum(1, keepdim=True)


def stage1_probability(logits):
    """The original Stage-I-only grammar control always softmaxes in CPU FP64."""
    p = logits.cpu().double().softmax(1).clamp_min(1e-12)
    return p / p.sum(1, keepdim=True)
