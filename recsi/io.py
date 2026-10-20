"""Local-only artifact IO and small provenance helpers."""
import hashlib
import json
import os
from pathlib import Path

import torch


def offline():
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_tensor_dict(path):
    # Release caches contain only tensors, strings, numbers and containers.
    return torch.load(path, map_location="cpu", weights_only=True)


def save_tensor_dict(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(value, handle)


def code_hashes():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.py"))}
