"""Validate portable Stage-I feature caches without data/model downloads."""
import math
import re

import torch

from .forest import same_speaker_parents
from .io import load_tensor_dict

LABELS = {"MELD": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
          "IEMOCAP": ["ang", "hap", "neu", "sad"]}


def load_cache(path, dataset, require_labels=True):
    d = load_tensor_dict(path)
    c = len(LABELS[dataset])
    n = len(d["ids"])
    if n == 0 or len(set(d["ids"])) != n or d["x"].shape != (n, 3072+c):
        raise ValueError("Duplicate IDs, empty cache or incorrect feature shape")
    if d["x"].dtype != torch.float32 or not torch.isfinite(d["x"]).all():
        raise ValueError("Cache features must be finite FP32")
    if require_labels:
        y = d["y"]
        if y.shape != (n,) or y.dtype != torch.long or y.min() < 0 or y.max() >= c:
            raise ValueError("Invalid labels")
    if d.get("label_order", LABELS[dataset]) != LABELS[dataset]:
        raise ValueError("The class vocabulary/order differs from the released protocol")
    if not re.fullmatch(r"[0-9a-f]{64}", d["stage1_sha256"]):
        raise ValueError("A Stage-I checkpoint SHA256 is required")
    metadata = d["metadata"]
    if [r["id"] for r in metadata] != d["ids"]:
        raise ValueError("Metadata and cache IDs must have identical order")
    for row in metadata:
        if not row["speaker"] or not row["dialogue"]:
            raise ValueError("Speaker and dialogue identifiers are required")
        if dataset == "MELD":
            if not isinstance(row["turn"], int) or row["turn"] < 0:
                raise ValueError("MELD requires a nonnegative integer turn")
        elif not (math.isfinite(row["start"]) and math.isfinite(row["end"]) and 0 <= row["start"] < row["end"]):
            raise ValueError("IEMOCAP requires true start/end timestamps")
    d["parent_same"] = same_speaker_parents(metadata, dataset)
    return d
