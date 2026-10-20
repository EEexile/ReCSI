"""Residual correction, original training, and compact-head conversion."""
import copy
import hashlib
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .inference import classification_metrics

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


class ArchivedTrainingCorrector(nn.Module):
    """Original creation/RNG order and trainable layout, for reproduction only."""

    def __init__(self, classes=7):
        super().__init__()
        if classes not in (4, 7):
            raise ValueError("The released protocols use 4 or 7 classes")
        self.classes = classes
        self.current = nn.Sequential(nn.Linear(3072 + classes, 64), nn.ReLU())
        self.history = nn.Sequential(nn.Linear(3072, 64), nn.ReLU())
        self.output = nn.Linear(129, classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, current, history=None, count=None):
        if history is None:
            history = current.new_zeros((len(current), 3072))
        if count is None:
            count = current.new_zeros((len(current), 1))
        if current.ndim != 2 or current.shape[1] != 3072 + self.classes:
            raise ValueError("Expected concatenated current audio/text/fused/logit vectors")
        return self.output(torch.cat([self.current(current), self.history(history), count], 1))


def _layout(state, classes):
    if classes not in (4, 7) or not isinstance(state, dict):
        raise ValueError("Expected a 4- or 7-class ReCSI state dictionary")
    shapes = {"current.0.weight": (64, 3072 + classes), "current.0.bias": (64,),
              "output.weight": (classes, 64), "output.bias": (classes,)}
    if "history.0.weight" in state:
        shapes.update({"history.0.weight": (64, 3072), "history.0.bias": (64,),
                       "output.weight": (classes, 129)})
        name = "archived-zero-context.v1"
    else:
        name = "compact.v1"
    if set(state) != set(shapes):
        raise ValueError("Unsupported head keys; no keys may be silently discarded")
    first = state["current.0.weight"]
    if not isinstance(first, torch.Tensor) or first.dtype not in (torch.float32, torch.float64):
        raise ValueError("Expected finite FP32 or FP64 head tensors")
    for key, shape in shapes.items():
        value = state[key]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != first.dtype or value.device != first.device
                or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid shape, dtype, device or values for {key}")
    return name


@torch.no_grad()
def compact_state_dict(state, classes):
    """Fold a fixed zero-context head, without modifying the source state.

    Input must belong to ReCSI's zero-history/count protocol. Nonzero-context
    use of a generic archived head is not supported by this conversion.
    Folding is after member averaging; folding each member first is different.
    """
    layout = _layout(state, classes)
    keys = ("current.0.weight", "current.0.bias", "output.weight", "output.bias")
    result = {key: state[key].detach().clone() for key in keys}
    if layout == "archived-zero-context.v1":
        result["output.weight"] = state["output.weight"][:, :64].detach().clone()
        constant = (state["output.weight"][:, 64:128].double()
                    @ state["history.0.bias"].double().relu())
        result["output.bias"] = (state["output.bias"].double() + constant).to(
            state["output.bias"].dtype)
    _layout(result, classes)
    return result


def readout_from_state(state, classes, *, compact=False):
    """Load strictly; archived states keep the historical arithmetic by default."""
    layout = _layout(state, classes)
    if compact:
        state = compact_state_dict(state, classes)
    cls = (ArchivedTrainingCorrector if layout == "archived-zero-context.v1" and not compact
           else ResidualCorrector)
    first = state["current.0.weight"]
    model = cls(classes).to(device=first.device, dtype=first.dtype)
    model.load_state_dict(state, strict=True)
    return model


def compact_head_bundle(bundle):
    """Export the already-averaged inference head; retain the original for fitting."""
    required = {"model", "classes", "dataset", "normalizer", "stage1_sha256"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Expected a ReCSI head bundle, not a raw checkpoint container")
    c = bundle["classes"]
    if bundle["dataset"] not in ("MELD", "IEMOCAP") or c != (7 if bundle["dataset"] == "MELD" else 4):
        raise ValueError("Dataset and head class count do not match")
    source_layout = _layout(bundle["model"], c)
    if bundle.get("model_layout", source_layout) != source_layout:
        raise ValueError("Declared model layout does not match its tensors")
    norm = bundle["normalizer"]
    if not isinstance(norm, dict) or set(norm) != {"mean", "std"}:
        raise ValueError("Expected the original TRAIN normalizer")
    for key in ("mean", "std"):
        value = norm[key]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (3072 + c,)
                or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError("Invalid TRAIN normalizer")
    if not (norm["std"] > 0).all():
        raise ValueError("Normalizer standard deviations must be positive")
    # Member histories/weights stay in the original training artifact. Folding
    # happens on the averaged state, not before nonlinear member averaging.
    result = {key: copy.deepcopy(value) for key, value in bundle.items()
              if key not in ("model", "members", "conversion")}
    result["model"] = compact_state_dict(bundle["model"], c)
    result["model_layout"] = "compact.v1"
    result["inference_only"] = True
    result["conversion"] = {
        "operation": "fixed_zero_context_bias_fold_after_parameter_averaging",
        "source_layout": source_layout,
        "bitwise_equivalence_guaranteed": False,
        "training_equivalence_claimed": False,
        "benchmark_test_repeated": False,
    }
    return result


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def average_states(states):
    if len(states) != 3 or any(list(s) != list(states[0]) for s in states):
        raise ValueError("Exactly three compatible selected member states are required")
    return {key: torch.stack([s[key] for s in states]).mean(0) for key in states[0]}


def score(y, z):
    return classification_metrics(y.cpu().numpy(), z.argmax(1).cpu().numpy(), z.shape[1])


def train_head(train, dev, dataset, init_seed, device="cpu", order_seeds=(42, 43, 44)):
    if tuple(order_seeds) != (42, 43, 44):
        raise ValueError("Released primary uses order seeds 42,43,44")
    if dataset == "MELD" and dev is None or dataset == "IEMOCAP" and dev is not None:
        raise ValueError("MELD uses dev selection; IEMOCAP Stage II uses a fixed epoch")
    c, epochs = (7, 30) if dataset == "MELD" else (4, 1)
    if dev is not None:
        if set(train["ids"]) & set(dev["ids"]) or train["stage1_sha256"] != dev["stage1_sha256"]:
            raise ValueError("Train/dev overlap or different Stage-I sources")
    norm = fit_normalizer(train["x"])
    tx, ty = normalize(train["x"], norm).to(device), train["y"].to(device)
    base = train["x"][:, -c:].to(device)
    if dev is not None:
        dx, dy = normalize(dev["x"], norm).to(device), dev["y"].to(device)
        db = dev["x"][:, -c:].to(device)
    seed_all(init_seed)
    initial = cpu_state(ArchivedTrainingCorrector(c))
    members, selected_states = [], []
    for order_seed in order_seeds:
        seed_all(init_seed)
        model = ArchivedTrainingCorrector(c).to(device)
        model.load_state_dict(initial)
        if sum(p.numel() for p in model.parameters()) != (394702 if c == 7 else 394120):
            raise RuntimeError("Archived training parameterization changed")
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=.0001)
        generator = torch.Generator().manual_seed(order_seed)
        selected = copy.deepcopy(initial)
        best = {"epoch": 0, "metrics": score(dy, db)} if dev is not None else None
        history, orders = [], []
        for epoch in range(1, epochs+1):
            model.train()
            order = torch.randperm(len(tx), generator=generator)
            orders.append(order)
            loss_sum, exposure, updates = 0., 0, 0
            lr = optimizer.param_groups[0]["lr"]
            for start in range(0, len(order), 128):
                idx = order[start:start+128].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(base[idx]+model(tx[idx]), ty[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite CE loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach())*len(idx)
                exposure += len(idx)
                updates += 1
            row = {"epoch": epoch, "loss": loss_sum/len(tx), "exposure": exposure,
                   "updates": updates, "learning_rate": lr,
                   "order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest()}
            if dev is not None:
                model.eval()
                with torch.no_grad():
                    metrics = score(dy, db+model(dx))
                row["dev_metrics"] = metrics
                if metrics["W-F1"] > best["metrics"]["W-F1"]:
                    best = {"epoch": epoch, "metrics": metrics}
                    selected = cpu_state(model)
            else:
                selected = cpu_state(model)
            scheduler.step()
            row["next_learning_rate"] = optimizer.param_groups[0]["lr"]
            history.append(row)
        if dev is None:
            best = {"epoch": 1, "policy": "fixed one epoch; no held-out selection"}
        selected_states.append(selected)
        members.append({"order_seed": order_seed, "selection": best, "history": history,
                        "orders": orders, "selected_model": selected})
    averaged = average_states(selected_states)
    return {"model": averaged, "normalizer": norm, "members": members,
            "classes": c, "dataset": dataset, "initialization_seed": init_seed,
            "order_seeds": list(order_seeds), "weight_per_member": 1/3,
            "stage1_sha256": train["stage1_sha256"], "train_ids": train["ids"]}


@torch.no_grad()
def head_probabilities(bundle, cache, device="cpu"):
    c = bundle["classes"]
    if cache["stage1_sha256"] != bundle["stage1_sha256"]:
        raise ValueError("Head and cache refer to different Stage-I checkpoints")
    # Never silently convert a sealed historical study or change its arithmetic.
    model = readout_from_state(bundle["model"], c).to(device)
    model.eval()
    x = normalize(cache["x"], bundle["normalizer"]).to(device)
    base = cache["x"][:, -c:].to(device)
    z = base+model(x)
    return unary_probability(z, bundle["dataset"])
