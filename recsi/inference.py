"""Temperature calibration and exact same-speaker forest inference."""
from collections import defaultdict
import heapq
import math

import numpy as np
import torch
import torch.nn.functional as F

def normalize_probabilities(p):
    p = p.double().clamp_min(1e-12)
    return p / p.sum(1, keepdim=True)


def apply_temperature(p, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    return (normalize_probabilities(p).log()/temperature).softmax(1)


def fit_temperature(probabilities, labels):
    p = normalize_probabilities(probabilities)
    labels = labels.to(device=p.device, dtype=torch.long)
    logp = p.log()
    theta = torch.zeros((), dtype=torch.float64, device=p.device, requires_grad=True)
    trace = []
    optimizer = torch.optim.LBFGS([theta], lr=1, max_iter=100, max_eval=150,
        history_size=20, tolerance_grad=1e-9, tolerance_change=1e-12,
        line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logp/theta.exp(), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite calibration objective")
        loss.backward()
        if not torch.isfinite(theta.grad):
            raise FloatingPointError("Nonfinite calibration gradient")
        trace.append({"logT": float(theta.detach()), "NLL": float(loss.detach()),
                      "gradient": float(theta.grad)})
        return loss

    before = float(closure().detach())
    optimizer.step(closure)
    after = float(closure().detach())
    temperature = float(theta.detach().exp())
    if not math.isfinite(temperature) or temperature <= 0 or after > before+1e-10:
        raise FloatingPointError("Temperature fit failed the original finite/NLL gate")
    return {"temperature": temperature, "NLL_before": before, "NLL_after": after,
            "trace": trace, "fit_count": len(labels), "fit_only": "MELD dev"}


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


def classification_metrics(labels, predictions, classes):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.int64)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError("Expected matching nonempty label/prediction vectors")
    if min(y.min(), p.min()) < 0 or max(y.max(), p.max()) >= classes:
        raise ValueError("Class index outside the fixed label vocabulary")
    cm = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    support, guessed, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    recall = np.divide(tp, support, out=np.zeros(classes), where=support > 0)
    precision = np.divide(tp, guessed, out=np.zeros(classes), where=guessed > 0)
    f1 = np.divide(2*tp, support+guessed, out=np.zeros(classes), where=support+guessed > 0)
    return {"W-F1": 100*float((f1*support).sum()/len(y)),
            "WA": 100*float(tp.sum()/len(y)), "UA": 100*float(recall.mean()),
            "Macro-F1": 100*float(f1.mean()), "confusion": cm.tolist(),
            "per_class_precision": (100*precision).tolist(),
            "per_class_recall": (100*recall).tolist(), "per_class_f1": (100*f1).tolist(),
            "n": len(y), "correct": int(tp.sum())}
