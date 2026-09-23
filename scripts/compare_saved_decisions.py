"""Compare saved two-stage verdicts with detector-preserving decisions.

This is a counterfactual analysis of existing scores, not a new inference run.
Use only records produced with a real detector, not VLM-only records.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def compare(records: list[dict], threshold: float) -> dict:
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not records:
        raise ValueError("No saved records to compare")
    if any(r["verdict"] not in ("pass", "defect", "false_alarm") for r in records):
        raise ValueError("Unknown saved verdict")
    metrics = {}
    for policy in ("saved", "detector_preserved"):
        tp = fp = tn = fn = 0
        for record in records:
            score = float(record["anomaly_score"])
            if not math.isfinite(score):
                raise ValueError("Saved anomaly scores must be finite")
            actual = record["true_type"] != "good"
            predicted = record["verdict"] == "defect" if policy == "saved" else score >= threshold
            tp += actual and predicted
            fp += not actual and predicted
            tn += not actual and not predicted
            fn += actual and not predicted
        metrics[policy] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "accuracy": (tp + tn) / len(records) if records else 0.0,
        }
    return {
        "analysis": "counterfactual from saved detector scores; no new inference",
        "n_images": len(records), "threshold": threshold, **metrics,
        "true_defects_removed_by_stage2": sum(
            r["true_type"] != "good" and float(r["anomaly_score"]) >= threshold
            and r["verdict"] != "defect" for r in records
        ),
        "false_alarms_removed_by_stage2": sum(
            r["true_type"] == "good" and float(r["anomaly_score"]) >= threshold
            and r["verdict"] != "defect" for r in records
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    with args.records.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    print(json.dumps(compare(records, args.threshold), indent=2))


if __name__ == "__main__":
    main()
