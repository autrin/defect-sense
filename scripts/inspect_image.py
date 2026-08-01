"""Inspect a single image through the two-stage pipeline.

Prints the JSON inspection result and saves an annotated overlay next to
the input (suffix `_inspected.png`).

Usage:
    python scripts/inspect_image.py path/to/image.png --category bottle \
        --ckpt results/Patchcore/MVTecAD/bottle/v5/weights/lightning/model.ckpt
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from defect_sense.detectors import AnomalibDetector
from defect_sense.pipeline.two_stage import TwoStagePipeline
from defect_sense.taxonomy import defect_types_for
from defect_sense.viz import draw_regions, overlay_heatmap
from defect_sense.vlm.adjudicator import VLMAdjudicator
from defect_sense.vlm.client import DEFAULT_MODEL, OllamaClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--category", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--detector", choices=["patchcore", "efficient_ad"], default="patchcore")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--vlm-model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root", default="./datasets/MVTecAD")
    args = parser.parse_args()

    pipeline = TwoStagePipeline(
        detector=AnomalibDetector(model_name=args.detector, ckpt_path=args.ckpt),
        adjudicator=VLMAdjudicator(
            client=OllamaClient(model=args.vlm_model),
            category=args.category,
            defect_types=defect_types_for(args.category, args.dataset_root),
        ),
        threshold=args.threshold,
    )

    image = Image.open(args.image)
    result = pipeline.inspect(image)
    print(json.dumps(result.to_dict(), indent=2))

    annotated = image.convert("RGB")
    if result.detection is not None and result.detection.anomaly_map is not None:
        annotated = overlay_heatmap(annotated, result.detection.anomaly_map)
    if result.regions:
        annotated = draw_regions(annotated, result.regions)
    out_path = args.image.with_stem(args.image.stem + "_inspected")
    annotated.save(out_path)
    print(f"Annotated image: {out_path}")


if __name__ == "__main__":
    main()
