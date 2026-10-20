"""Original cached-feature CE recipe: three shuffled members per initialization.

Every MELD member completes 30 epochs; the earliest strictly best dev W-F1,
including epoch zero, is retained. Each IEMOCAP member completes one fixed
epoch, without any Stage-II held-out checkpoint selection. Selected member
parameters are averaged, not their predictions.
"""
import copy
import hashlib
import random

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import classification_metrics
from .compat import ArchivedTrainingCorrector, readout_from_state
from .model import fit_normalizer, normalize, unary_probability


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
