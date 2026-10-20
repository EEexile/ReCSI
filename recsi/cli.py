"""Small local-only commands; production training is always explicitly invoked."""
import argparse
import json

from .io import offline


def main():
    offline()
    parser = argparse.ArgumentParser(prog="recsi")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fit", help="Fit all configured heads and temperatures, then seal a global lock")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate", help="One complete evaluation of all sealed cohorts")
    p.add_argument("--study", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("check-cache", help="Validate a local portable cache")
    p.add_argument("cache")
    p.add_argument("--dataset", choices=["MELD", "IEMOCAP"], required=True)
    p = sub.add_parser("compact-head", help="Export an existing zero-context ReCSI head for inference; no fitting")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    sub.add_parser("demo", help="Synthetic math demonstration; no benchmark/model/data download")
    args = parser.parse_args()
    if args.command == "fit":
        from .pipeline import fit_study
        print(fit_study(args.config, args.output))
    elif args.command == "evaluate":
        from .pipeline import evaluate_study
        print(evaluate_study(args.study, args.output, args.device))
    elif args.command == "check-cache":
        from .cache import load_cache
        d = load_cache(args.cache, args.dataset)
        print(json.dumps({"count": len(d["ids"]), "shape": list(d["x"].shape), "status": "PASS"}))
    elif args.command == "compact-head":
        from .compat import compact_head_bundle
        from .io import digest, load_tensor_dict, save_tensor_dict
        bundle = compact_head_bundle(load_tensor_dict(args.input))
        bundle["conversion"]["source_bundle_sha256"] = digest(args.input)
        save_tensor_dict(args.output, bundle)  # Exclusive: never overwrite the source or a sealed head.
        print(json.dumps({"status": "PASS", "model_layout": bundle["model_layout"],
                          "head_parameters": sum(v.numel() for v in bundle["model"].values()),
                          "training_run": False, "benchmark_test_repeated": False}))
    else:
        from .demo import main as demo
        demo()
