"""TRAIN count statistics and exact log-domain sum-product on directed forests.

Pseudo-emissions are p_i(c)/pi_c. Root factors are pi; edge factors are the
row-stochastic A[previous_class, current_class]. Future unlabeled observations
participate in the downward pass; this is offline smoothing, not causal decoding.
"""
from collections import defaultdict
import heapq

import numpy as np
import torch


def topology(parent):
    parents = [int(v) for v in parent]
    n = len(parents)
    if not n:
        raise ValueError("An empty forest is not supported")
    children, roots = [[] for _ in parents], []
    for i, p in enumerate(parents):
        if not -1 <= p < n or p == i:
            raise ValueError("Invalid parent index")
        (roots if p < 0 else children[p]).append(i)
    queue, order = roots.copy(), []
    heapq.heapify(queue)
    while queue:
        i = heapq.heappop(queue)
        order.append(i)
        for child in children[i]:
            heapq.heappush(queue, child)
    if len(order) != n:
        raise ValueError("Cycle in parent routes")
    return parents, roots, order


def same_speaker_parents(metadata, dataset):
    grouped = defaultdict(list)
    for i, row in enumerate(metadata):
        grouped[row["dialogue"]].append(i)
    parents = torch.full((len(metadata),), -1, dtype=torch.long)
    for i, row in enumerate(metadata):
        if dataset == "MELD":
            key = lambda j: (metadata[j]["turn"], metadata[j]["id"])
            eligible = lambda r: r["turn"] < row["turn"]
        elif dataset == "IEMOCAP":
            key = lambda j: (metadata[j]["end"], metadata[j]["start"], metadata[j]["id"])
            eligible = lambda r: r["end"] <= row["start"]
        else:
            raise ValueError(dataset)
        candidates = [j for j in grouped[row["dialogue"]]
                      if j != i and metadata[j]["speaker"] == row["speaker"] and eligible(metadata[j])]
        if candidates:
            parents[i] = max(candidates, key=key)
    topology(parents)
    return parents


def fit_train_statistics(labels, parent, classes):
    y = np.asarray(labels, dtype=np.int64)
    parents, _, _ = topology(parent)
    if len(parents) != len(y) or y.min() < 0 or y.max() >= classes:
        raise ValueError("Invalid TRAIN labels or route length")
    prior = (np.bincount(y, minlength=classes)+1.)/(len(y)+classes)
    counts = np.zeros((classes, classes), dtype=np.int64)
    for i, p in enumerate(parents):
        if p >= 0:
            counts[y[p], y[i]] += 1
    transition = (counts+1.)/(counts.sum(1, keepdims=True)+classes)
    return {"prior": torch.from_numpy(prior), "A_same": torch.from_numpy(transition),
            "counts_same": torch.from_numpy(counts), "laplace_alpha": 1.0}


@torch.no_grad()
def smooth_forest(probability, prior, transition, parent):
    p, pi, A = probability, prior, transition
    if p.ndim != 2 or p.dtype != torch.float64 or pi.dtype != p.dtype or A.dtype != p.dtype:
        raise ValueError("Sum-product requires FP64 probabilities and factors")
    c = p.shape[1]
    if pi.shape != (c,) or A.shape != (c, c) or len(parent) != len(p):
        raise ValueError("Shape mismatch")
    if p.device != pi.device or p.device != A.device:
        raise ValueError("Factors and probabilities must share a device")
    if any(not torch.isfinite(v).all() or not (v > 0).all() for v in (p, pi, A)):
        raise ValueError("Positive finite probabilities are required")
    if not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=1e-12, rtol=0):
        raise ValueError("Unary rows must sum to one")
    if not torch.allclose(A.sum(1), torch.ones_like(pi), atol=1e-12, rtol=0) or abs(float(pi.sum())-1)>1e-12:
        raise ValueError("Prior and transition rows must sum to one")
    parents, roots, order = topology(parent)
    emission, logpi, logA = p.log()-pi.log()[None, :], pi.log(), A.log()
    phi, message = emission.clone(), torch.zeros_like(p)
    for i in reversed(order):
        if parents[i] >= 0:
            message[i] = torch.logsumexp(logA + phi[i][None, :], dim=1)
            phi[parents[i]] += message[i]
    logZ = torch.stack([torch.logsumexp(logpi+phi[i], dim=0) for i in roots]).sum()
    down = torch.empty_like(p)
    for i in order:
        if parents[i] < 0:
            down[i] = logpi
        else:
            pidx = parents[i]
            cavity = down[pidx]+phi[pidx]-message[i]
            down[i] = torch.logsumexp(cavity[:, None]+logA, dim=0)
    return (phi+down).softmax(1), logZ
