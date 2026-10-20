"""Historical training compatibility and explicit inference-only conversion.

The archived ReCSI protocol always supplied zero history and zero count. At
fixed weights, their constant contribution can be folded into the output bias.
This does not reproduce training a compact model from scratch, and floating
point reassociation need not preserve every prediction near a tie.
"""
import copy

import torch
from torch import nn

from .model import ResidualCorrector


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
