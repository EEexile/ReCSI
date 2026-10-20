"""Export [audio768,text768,fused1536,logitsC] from a local Stage-I checkpoint.

Metadata is a JSON list in precisely the same order as the original CSV. It
contains id/dialogue/speaker and MELD turn or actual IEMOCAP start/end seconds.
CSV order, dynamic padding, batch size16, BF16 and TF32 follow the archived
exporter. The hook reads fusion inputs and does not change network operations.
"""
import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import sys

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["MELD", "IEMOCAP"], required=True)
    p.add_argument("--config", required=True, help="Public local-asset config; same scientific values as checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--split", choices=["train", "dev", "test", "heldout"], required=True)
    p.add_argument("--fold", type=int, choices=range(1, 6))
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from recsi.cache import load_cache
    from recsi.io import digest, read_json, save_tensor_dict
    sys.path.insert(0, str(ROOT/a.dataset.lower()))
    data = importlib.import_module("data")
    models = importlib.import_module("models")
    cfg = read_json(a.config)
    if a.dataset == "MELD":
        if a.split == "heldout":
            p.error("MELD uses train/dev/test")
        rows = data.read_rows(cfg["csv_paths"][a.split], a.split)
        ds = data.MELDDataset(rows, dict(cfg, audio_dir=cfg["audio_dirs"][a.split]))
    else:
        if a.fold is None or a.split not in ("train", "heldout"):
            p.error("IEMOCAP requires --fold and train/heldout")
        tr, heldout = data.split_rows(data.read_rows(cfg["csv_path"]), a.fold)
        rows = tr if a.split == "train" else heldout
        ds = data.IEMOCAPDataset(rows, cfg)
    # Original Stage-I checkpoints also contain Python/NumPy RNG state.
    # This command is only for a checkpoint that the user created and trusts.
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    old_config = checkpoint["config"]
    for key in ("variant", "num_classes", "hidden_size", "num_heads", "dropout", "head_dropout",
                "max_audio_seconds", "max_text_tokens", "sample_rate", "amp", "seed",
                "auxiliary_weights", "auxiliary_init_seed", "freeze_text_layers"):
        if cfg.get(key) != old_config.get(key):
            raise ValueError("Export config changes checkpoint scientific setting: " + key)
    model = models.Model(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False).to(a.device).eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    captured = {}

    def hook(module, args):
        ha, ht, am, tm = args[:4]
        captured["audio"] = models.masked_mean(ha, am).float()
        captured["text"] = models.masked_mean(ht, tm).float()

    handle = model.fusion.register_forward_pre_hook(hook)
    xs, ys, ids = [], [], []
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=data.collate, num_workers=0)
    try:
        for cpu in loader:
            batch = {k: v.to(a.device) if torch.is_tensor(v) else v for k, v in cpu.items()}
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if a.device.startswith("cuda") and cfg["amp"] == "bf16" else contextlib.nullcontext()
            with torch.no_grad(), amp:
                result = model(batch, compute_auxiliary=False)
            xs.append(torch.cat([captured["audio"], captured["text"], result["pooled"].float(), result["logits"].float()], 1).cpu())
            ys.append(cpu["labels"])
            ids.extend(cpu["file_ids"])
    finally:
        handle.remove()
    metadata = read_json(a.metadata)
    if [r["id"] for r in metadata] != ids:
        raise ValueError("Metadata must exactly match exported order; do not reorder/filter cache points")
    cache = {"x": torch.cat(xs), "y": torch.cat(ys), "ids": ids, "metadata": metadata,
             "label_order": data.LABELS, "stage1_sha256": digest(a.checkpoint), "split": a.split,
             "feature_order": ["audio_masked_mean768", "self_attended_text_masked_mean768", "fused1536", "raw_logits"]}
    save_tensor_dict(a.output, cache)
    load_cache(a.output, a.dataset)
    print(json.dumps({"path": a.output, "count": len(ids), "sha256": digest(a.output),
                      "note": "Contains private/local features and labels; excluded from GitHub source package"}))


if __name__ == "__main__":
    main()
