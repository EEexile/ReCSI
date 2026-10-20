"""Training and evaluation entry point: python -m recsi."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import time

import numpy as np
import torch

from .data import (LABELS, code_hashes, digest, load_cache, load_tensor_dict,
                   offline, read_json, save_tensor_dict, write_json)
from .inference import (apply_temperature, classification_metrics,
                        fit_temperature, fit_train_statistics, smooth_forest)
from .model import (compact_head_bundle, head_probabilities,
                    stage1_probability, train_head)

def configure(device, threads=2):
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select CPU explicitly or use an existing CUDA installation")


def fit_study(config_path, output):
    cfg = read_json(config_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    device = cfg.get("device", "cpu")
    configure(device, cfg.get("cpu_threads", 2))
    seeds = cfg.get("init_seeds", [42, 43, 44])
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in (42, 43, 44) for s in seeds):
        raise ValueError("Use the predeclared initialization seeds, without test-driven selection")
    names = [r["name"] for r in cfg["cohorts"]]
    if len(names) != len(set(names)) or cfg.get("calibration_cohort", "MELD") not in names:
        raise ValueError("Unique cohort names and a MELD calibration cohort are required")
    synthetic = bool(cfg.get("synthetic_fixture", False))
    iem_folds = [r.get("fold") for r in cfg["cohorts"] if r["dataset"] == "IEMOCAP"]
    if iem_folds and sorted(iem_folds) != [1, 2, 3, 4, 5] and not synthetic:
        raise ValueError("The complete IEMOCAP primary study includes all five folds")
    records = []
    calibrators = None
    for row in cfg["cohorts"]:
        name, dataset = row["name"], row["dataset"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Cohort name must be a plain directory name")
        if not isinstance(row.get("eval_sha256"), str) or not re.fullmatch("[0-9a-f]{64}", row["eval_sha256"]):
            raise ValueError("Provide a previously recorded eval-cache SHA256 before fitting")
        tr = load_cache(row["train_cache"], dataset)
        dv = load_cache(row["dev_cache"], dataset) if dataset == "MELD" else None
        if not synthetic:
            train_count = 9988 if dataset == "MELD" else [4446,4508,4380,4500,4290][row["fold"]-1]
            eval_count = 2610 if dataset == "MELD" else [1085,1023,1151,1031,1241][row["fold"]-1]
            if len(tr["ids"]) != train_count or row.get("expected_eval_count") != eval_count:
                raise ValueError("The released primary requires complete original train/evaluation cohorts")
            if dv is not None and len(dv["ids"]) != 1108:
                raise ValueError("MELD dev must contain all 1108 examples")
        if dataset == "IEMOCAP":
            fold = row["fold"]
            if fold not in range(1, 6) or any(i.startswith(f"Ses{fold:02d}") for i in tr["ids"]):
                raise ValueError("IEMOCAP train cache contains the held-out Session")
        folder = out/name
        folder.mkdir()
        grammar = fit_train_statistics(tr["y"].numpy(), tr["parent_same"], len(LABELS[dataset]))
        save_tensor_dict(folder/"grammar.pt", grammar)
        head_rows = []
        local_calibrators = {}
        for seed in seeds:
            bundle = train_head(tr, dv, dataset, seed, device)
            path = folder/f"head_init{seed}.pt"
            save_tensor_dict(path, bundle)
            head_rows.append({"seed": seed, "file": str(path.relative_to(out)), "sha256": digest(path)})
            if name == cfg.get("calibration_cohort", "MELD"):
                if dataset != "MELD":
                    raise ValueError("Temperature fitting is MELD-dev only")
                p = head_probabilities(bundle, dv, device)
                local_calibrators[str(seed)] = fit_temperature(p, dv["y"])
        if local_calibrators:
            local_calibrators["StageI"] = fit_temperature(stage1_probability(dv["x"][:, -7:]), dv["y"])
            calibrators = local_calibrators
        records.append({"name": name, "dataset": dataset, "fold": row.get("fold"),
            "stage1_sha256": tr["stage1_sha256"], "train_cache_sha256": digest(row["train_cache"]),
            "dev_cache_sha256": digest(row["dev_cache"]) if dv is not None else None,
            "grammar": {"file": str((folder/"grammar.pt").relative_to(out)), "sha256": digest(folder/"grammar.pt")},
            "heads": head_rows, "eval_cache": str(Path(row["eval_cache"]).resolve()),
            "eval_sha256": row["eval_sha256"], "expected_eval_count": row.get("expected_eval_count")})
        write_json(folder/"fit_progress.json", {"all_heads_complete": True, "evaluation_loaded": False})
    if calibrators is None:
        raise ValueError("No MELD calibrators were fitted")
    lock = {"schema": "recsi.study-lock.v1", "created_unix": time.time(), "cohorts": records,
            "calibrators": calibrators, "code_sha256": code_hashes(), "config_sha256": digest(config_path),
            "init_seeds": seeds, "eval_cache_loaded_during_fit": False,
            "synthetic_fixture": synthetic,
            "IEM_temperatures_transferred_without_refitting": True}
    write_json(out/"selection_lock.json", lock, exclusive=True)
    return out/"selection_lock.json"


def evaluate_study(study, output, device="cpu"):
    configure(device)
    study, out = Path(study), Path(output)
    lockpath = study/"selection_lock.json"
    lock = read_json(lockpath)
    locksha = digest(lockpath)
    if lock["code_sha256"] != code_hashes():
        raise RuntimeError("Code changed after global selection lock")
    # Validate model/grammar bindings before claiming the single evaluation.
    for row in lock["cohorts"]:
        for artifact in [row["grammar"], *row["heads"]]:
            if digest(study/artifact["file"]) != artifact["sha256"]:
                raise RuntimeError("A sealed model/grammar changed")
    write_json(study/"evaluation_started.json", {"lock_sha256": locksha, "time_unix": time.time()}, exclusive=True)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for cohort in lock["cohorts"]:
        if digest(cohort["eval_cache"]) != cohort["eval_sha256"]:
            raise RuntimeError("Evaluation cache changed after the predeclared binding")
        dataset, name = cohort["dataset"], cohort["name"]
        cache = load_cache(cohort["eval_cache"], dataset)
        if cache["stage1_sha256"] != cohort["stage1_sha256"]:
            raise ValueError("Evaluation features use a different Stage I")
        if cohort["expected_eval_count"] is not None and len(cache["ids"]) != cohort["expected_eval_count"]:
            raise ValueError("Incomplete evaluation split")
        if dataset == "IEMOCAP" and not all(i.startswith(f"Ses{cohort['fold']:02d}") for i in cache["ids"]):
            raise ValueError("Wrong held-out IEMOCAP Session")
        grammar = load_tensor_dict(study/cohort["grammar"]["file"])
        c = len(LABELS[dataset])
        baseline = cache["x"][:, -c:].argmax(1)
        for head in [{"seed": "StageI", "file": None}, *cohort["heads"]]:
            if head["file"] is None:
                raw = stage1_probability(cache["x"][:, -c:])
            else:
                bundle = load_tensor_dict(study/head["file"])
                if set(bundle["train_ids"]) & set(cache["ids"]):
                    raise ValueError("Train/evaluation overlap")
                raw = head_probabilities(bundle, cache, device)
            T = lock["calibrators"][str(head["seed"])]["temperature"]
            p = apply_temperature(raw, T)
            q, logZ = smooth_forest(p, grammar["prior"], grammar["A_same"], cache["parent_same"])
            variants = [("StageI", raw)] if head["file"] is None else []
            variants += [("StageI_calibrated_chain" if head["file"] is None else "ReCSI", q)]
            for method, prob in variants:
                pred = prob.argmax(1)
                metrics = classification_metrics(cache["y"].numpy(), pred.numpy(), c)
                rows.append({"cohort": name, "dataset": dataset, "method": method, "head_seed": head["seed"],
                    **metrics, "corrected": int(((pred == cache["y"]) & (baseline != cache["y"])).sum()),
                    "regressed": int(((pred != cache["y"]) & (baseline == cache["y"])).sum())})
            np.savez_compressed(out/f"{name}_init{head['seed']}.npz", ids=np.asarray(cache["ids"]),
                labels=cache["y"].numpy(), raw_p=raw.numpy(), calibrated_p=p.numpy(), q=q.numpy(),
                parent_same=cache["parent_same"].numpy())
    summary = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for method in ("StageI", "StageI_calibrated_chain", "ReCSI"):
            group = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            by_seed = []
            for seed in sorted({str(r["head_seed"]) for r in group}):
                values = [r for r in group if str(r["head_seed"]) == seed]
                by_seed.append({"head_seed": seed, "fold_count": len(values),
                    **{key: statistics.mean(v[key] for v in values) for key in ("W-F1", "WA", "UA", "Macro-F1")}})
            summary.append({"dataset": dataset, "method": method, "per_head_fold_means": by_seed,
                "mean": {key: statistics.mean(v[key] for v in by_seed) for key in ("W-F1", "WA", "UA", "Macro-F1")}})
    result = {"rows": rows, "summary": summary, "selection_lock_sha256": locksha,
              "synthetic_fixture": lock["synthetic_fixture"],
              "historical_IEM_test_selected_LOSO": True,
              "head_initializations_are_not_independent_StageI_seeds": True}
    write_json(out/"result.json", result)
    with (out/"results.csv").open("w", newline="", encoding="utf8") as handle:
        keys = ["cohort", "dataset", "method", "head_seed", "W-F1", "WA", "UA", "Macro-F1", "corrected", "regressed"]
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out/"result.json"


def main():
    offline()
    parser = argparse.ArgumentParser(prog="recsi")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fit", help="Train correction heads and fit MELD-dev temperatures")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate", help="Evaluate the selected heads and speaker model")
    p.add_argument("--study", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("compact-head", help="Convert an existing head to compact inference form")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fit":
        print(fit_study(args.config, args.output))
    elif args.command == "evaluate":
        print(evaluate_study(args.study, args.output, args.device))
    else:
        bundle = compact_head_bundle(load_tensor_dict(args.input))
        bundle["conversion"]["source_bundle_sha256"] = digest(args.input)
        save_tensor_dict(args.output, bundle)
        print(json.dumps({"status": "PASS", "model_layout": bundle["model_layout"],
                          "head_parameters": sum(v.numel() for v in bundle["model"].values()),
                          "training_run": False, "benchmark_test_repeated": False}))


if __name__ == "__main__":
    main()
