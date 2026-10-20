"""A deliberately synthetic four-utterance demonstration of the inference math."""
import json

import torch

from .forest import smooth_forest
from .model import ResidualCorrector


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    # Interleaved speakers A,B,A,B. Edges link each speaker only to themselves.
    p = torch.tensor([[.65, .15, .10, .10], [.1, .1, .7, .1],
                      [.38, .39, .13, .10], [.1, .2, .6, .1]], dtype=torch.float64)
    prior = torch.full((4,), .25, dtype=torch.float64)
    A = torch.full((4, 4), .1, dtype=torch.float64)
    A.diagonal().fill_(.7)
    q, logZ = smooth_forest(p, prior, A, [-1, -1, 0, 1])
    net = ResidualCorrector(4)
    delta = net(torch.zeros(4, 3076))
    assert torch.equal(delta, torch.zeros_like(delta))
    print(json.dumps({"scope": "SYNTHETIC illustration; not a benchmark result",
                      "zero_output_initialization": True, "raw_p": p.tolist(),
                      "same_speaker_marginals": q.tolist(), "logZ": float(logZ)}, indent=2))
