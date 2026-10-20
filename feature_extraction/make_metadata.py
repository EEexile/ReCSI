"""Create dialogue metadata from CSV files and IEMOCAP timestamps."""
import argparse
import csv
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["MELD", "IEMOCAP"], required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--split", required=True, choices=["train", "dev", "test", "heldout"])
    p.add_argument("--fold", type=int, choices=range(1, 6))
    p.add_argument("--timestamps", help="IEM JSON mapping ID to start/end/dialogue/speaker")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    with open(a.csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    result = []
    if a.dataset == "MELD":
        if a.split == "heldout":
            p.error("MELD has official train/dev/test splits")
        for row in rows:
            match = re.fullmatch(r"meld_(train|dev|test)_dia(\d+)_utt(\d+)", row["file"])
            if not match or match[1] != a.split:
                raise ValueError("MELD IDs must preserve official dialogue/turn information")
            result.append({"id": a.split+"/"+row["file"], "dialogue": a.split+"/"+match[2],
                           "speaker": row["speaker"], "turn": int(match[3])})
    else:
        if not a.timestamps or a.fold is None or a.split not in ("train", "heldout"):
            p.error("IEMOCAP requires --timestamps, --fold and train/heldout")
        timestamps = json.loads(Path(a.timestamps).read_text(encoding="utf8"))
        rows = [r for r in rows if (int(r["file"][3:5]) == a.fold) == (a.split == "heldout")]
        for row in rows:
            value = timestamps[row["file"]]
            if value["speaker"] != row["speaker"]:
                raise ValueError("Timestamp and CSV speaker differ")
            result.append({"id": row["file"], **value})
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
