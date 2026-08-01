"""Run a Stage 1 anomaly-detection benchmark across MVTec AD categories.

Usage:
    python benchmark.py                                   # patchcore, all 15 categories
    python benchmark.py --model efficient_ad              # EfficientAD, all categories
    python benchmark.py bottle hazelnut                   # subset
    python benchmark.py --model patchcore --output results/patchcore.csv
"""
import argparse
import csv
import time
from pathlib import Path

from defect_sense.taxonomy import ALL_CATEGORIES


def run_one(model_name: str, category: str) -> dict:
    import torch
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine

    from defect_sense.detectors.anomalib_detector import _build_model

    torch.set_float32_matmul_precision("high")

    datamodule = MVTecAD(
        root="./datasets/MVTecAD",
        category=category,
        # EfficientAD's student-teacher training requires batch size 1.
        train_batch_size=1 if model_name == "efficient_ad" else 32,
        eval_batch_size=32,
        num_workers=4,
    )
    model = _build_model(model_name)
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
        "model": model_name,
        "image_auroc": round(metrics.get("image_AUROC", 0.0), 4),
        "image_f1": round(metrics.get("image_F1Score", 0.0), 4),
        "pixel_auroc": round(metrics.get("pixel_AUROC", 0.0), 4),
        "pixel_f1": round(metrics.get("pixel_F1Score", 0.0), 4),
        "fit_seconds": round(fit_seconds, 1),
        "test_seconds": round(test_seconds, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("categories", nargs="*", default=None,
                        help="Categories to run (default: all 15)")
    parser.add_argument("--model", choices=["patchcore", "efficient_ad"], default="patchcore")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV (default: results/<model>_baseline.csv)")
    args = parser.parse_args()

    categories = args.categories or ALL_CATEGORIES
    unknown = [c for c in categories if c not in ALL_CATEGORIES]
    if unknown:
        raise SystemExit(f"Unknown categor{'ies' if len(unknown) > 1 else 'y'}: {unknown}")

    output = args.output or Path(f"results/{args.model}_baseline.csv")

    rows = []
    for i, category in enumerate(categories, start=1):
        print(f"\n{'=' * 60}\n[{i}/{len(categories)}] {args.model} / {category}\n{'=' * 60}")
        try:
            row = run_one(args.model, category)
            rows.append(row)
            print(f"  -> {row}")
        except Exception as e:
            print(f"  FAILED: {e}")
            rows.append({
                "category": category, "model": args.model,
                "image_auroc": None, "image_f1": None,
                "pixel_auroc": None, "pixel_f1": None,
                "fit_seconds": None, "test_seconds": None,
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {output.resolve()}")


if __name__ == "__main__":
    main()
