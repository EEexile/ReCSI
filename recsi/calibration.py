"""One dev-NLL scalar temperature; transfer unchanged across IEMOCAP folds."""
import math

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
