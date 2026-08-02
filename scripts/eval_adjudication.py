"""Evaluate the two-stage pipeline (or a VLM-only ablation) on an MVTec category.

Requires a trained Stage 1 checkpoint (two-stage mode) and a running Ollama
server with the VLM pulled.

Usage:
    python scripts/eval_adjudication.py bottle --ckpt results/Patchcore/MVTecAD/bottle/v5/weights/lightning/model.ckpt
    python scripts/eval_adjudication.py bottle --mode vlm-only --limit 5
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from defect_sense.eval.harness import (
    collect_run_metadata,
    evaluate_category,
    summarize,
    write_outputs,
)
from defect_sense.pipeline.two_stage import Detection, TwoStagePipeline
from defect_sense.taxonomy import defect_types_for
from defect_sense.vlm.adjudicator import PROMPT_VERSION, VLMAdjudicator
from defect_sense.vlm.client import DEFAULT_MODEL, OllamaClient


class AlwaysFlagDetector:
    """VLM-only ablation: every image goes straight to the VLM, no heatmap."""

    def predict(self, image: Image.Image) -> Detection:
        if image.width < 1 or image.height < 1:
            raise ValueError("Inspection image must not be empty")
        return Detection(score=1.0, anomaly_map=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category")
    parser.add_argument("--dataset-root", default="./datasets/MVTecAD")
    parser.add_argument(
        "--mode", choices=["two-stage", "vlm-only"], default="two-stage"
    )
    parser.add_argument(
        "--ckpt", default=None, help="Stage 1 checkpoint (two-stage mode)"
    )
    parser.add_argument(
        "--detector", choices=["patchcore", "efficient_ad"], default="patchcore"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Stage 1 triage threshold on the normalized anomaly score",
    )
    parser.add_argument("--vlm-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max images per defect type (quick runs)",
    )
    parser.add_argument("--out-dir", default="results/adjudication")
    args = parser.parse_args()

    if args.mode == "two-stage":
        from defect_sense.detectors import AnomalibDetector

        detector = AnomalibDetector(model_name=args.detector, ckpt_path=args.ckpt)
        threshold = args.threshold
    else:
        detector = AlwaysFlagDetector()
        threshold = 0.0

    adjudicator = VLMAdjudicator(
        client=OllamaClient(model=args.vlm_model),
        category=args.category,
        defect_types=defect_types_for(args.category, args.dataset_root),
    )
    pipeline = TwoStagePipeline(
        detector=detector, adjudicator=adjudicator, threshold=threshold
    )

    print(f"Evaluating {args.category} ({args.mode}, VLM={args.vlm_model})")
    records = evaluate_category(
        pipeline, args.dataset_root, args.category, limit=args.limit
    )
    summary = summarize(args.category, records)

    out_dir = Path(args.out_dir) / args.mode
    run_metadata = collect_run_metadata(
        category=args.category,
        mode=args.mode,
        detector=args.detector if args.mode == "two-stage" else None,
        checkpoint=Path(args.ckpt).as_posix() if args.ckpt else None,
        threshold=threshold,
        vlm_model=args.vlm_model,
        prompt_version=PROMPT_VERSION,
        limit_per_type=args.limit,
        dataset_root=Path(args.dataset_root).as_posix(),
    )
    csv_path, json_path = write_outputs(
        records,
        summary,
        out_dir,
        run_metadata=run_metadata,
    )
    print(f"\n{json.dumps(asdict(summary), indent=2)}")
    print(f"\nPer-image records: {csv_path}\nSummary: {json_path}")


if __name__ == "__main__":
    main()
