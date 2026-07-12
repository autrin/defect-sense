"""Run PatchCore across MVTec AD categories and write results to CSV.

Usage:
    python train_patchcore.py                 # all 15 categories
    python train_patchcore.py bottle          # single category
    python train_patchcore.py bottle hazelnut # subset
"""
import csv
import sys
import time
from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

RESULTS_CSV = Path("results/patchcore_baseline.csv")


def run_one(category: str) -> dict:
    torch.set_float32_matmul_precision("high")

    datamodule = MVTecAD(
        root="./datasets/MVTecAD",
        category=category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )
    engine = Engine(max_epochs=1, accelerator="gpu", devices=1)

    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = engine.test(model=model, datamodule=datamodule)
    test_seconds = time.perf_counter() - t0

    metrics = results[0]
    return {
        "category": category,
        "model": "patchcore",
        "image_auroc": round(metrics.get("image_AUROC", 0.0), 4),
        "image_f1": round(metrics.get("image_F1Score", 0.0), 4),
        "pixel_auroc": round(metrics.get("pixel_AUROC", 0.0), 4),
        "pixel_f1": round(metrics.get("pixel_F1Score", 0.0), 4),
        "fit_seconds": round(fit_seconds, 1),
        "test_seconds": round(test_seconds, 1),
    }


def write_csv(rows: list[dict]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {RESULTS_CSV.resolve()}")


def main() -> None:
    categories = sys.argv[1:] or ALL_CATEGORIES
    unknown = [c for c in categories if c not in ALL_CATEGORIES]
    if unknown:
        raise SystemExit(f"Unknown categor{'ies' if len(unknown) > 1 else 'y'}: {unknown}")

    rows = []
    for i, category in enumerate(categories, start=1):
        print(f"\n{'=' * 60}\n[{i}/{len(categories)}] {category}\n{'=' * 60}")
        try:
            row = run_one(category)
            rows.append(row)
            print(f"  -> {row}")
        except Exception as e:
            print(f"  FAILED: {e}")
            rows.append({
                "category": category, "model": "patchcore",
                "image_auroc": None, "image_f1": None,
                "pixel_auroc": None, "pixel_f1": None,
                "fit_seconds": None, "test_seconds": None,
            })

    write_csv(rows)


if __name__ == "__main__":
    main()