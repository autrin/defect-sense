"""Evaluation harness for the two-stage pipeline on MVTec AD test sets.

MVTec AD ships ground truth for free: each test image lives in a folder
named after its defect type ("good" for clean images). That lets us score
three things the AUROC benchmark can't:

1. Triage quality — does stage 1 send the right images to the VLM?
2. Defect-type accuracy — does the VLM name the actual defect?
3. Cost — what fraction of images pay the VLM latency, and how much?
"""
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageRecord:
    path: str
    true_type: str            # folder name: "good" or a defect type
    verdict: str              # pipeline verdict: pass | defect | false_alarm
    anomaly_score: float
    triaged_to_vlm: bool
    pred_type: str | None
    type_correct: bool | None  # None when not applicable (good image / not adjudicated)
    stage1_seconds: float
    stage2_seconds: float
    report: str = ""


@dataclass
class EvalSummary:
    category: str
    n_images: int
    # Triage / end-to-end detection (defect vs good), after both stages:
    tp: int
    fp: int
    tn: int
    fn: int
    detection_precision: float
    detection_recall: float
    detection_accuracy: float
    # Stage 2 defect typing, over true-defect images the VLM saw:
    n_typed: int
    type_accuracy: float | None
    per_type_accuracy: dict[str, float] = field(default_factory=dict)
    # Cost profile:
    vlm_call_rate: float = 0.0
    mean_stage1_seconds: float = 0.0
    mean_stage2_seconds: float = 0.0
    mean_seconds_per_image: float = 0.0


def iter_test_images(dataset_root: str | Path, category: str):
    """Yield (path, true_type) over a category's test split, sorted."""
    test_dir = Path(dataset_root) / category / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"No test split at {test_dir}. Run scripts/download_mvtec.py first."
        )
    for type_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        for img in sorted(type_dir.iterdir()):
            if img.suffix.lower() in IMAGE_EXTS:
                yield img, type_dir.name


def evaluate_category(
    pipeline,
    dataset_root: str | Path,
    category: str,
    limit: int | None = None,
    progress: bool = True,
) -> list[ImageRecord]:
    """Run the pipeline over a category's test set and collect per-image records.

    `limit` caps images *per defect type* so quick runs still see every class.
    """
    records: list[ImageRecord] = []
    per_type_seen: Counter[str] = Counter()

    for path, true_type in iter_test_images(dataset_root, category):
        if limit is not None and per_type_seen[true_type] >= limit:
            continue
        per_type_seen[true_type] += 1

        result = pipeline.inspect(Image.open(path))
        is_true_defect = true_type != "good"
        type_correct = None
        if is_true_defect and result.triaged_to_vlm and result.defect_type:
            type_correct = result.defect_type == true_type

        records.append(ImageRecord(
            path=str(path),
            true_type=true_type,
            verdict=result.verdict,
            anomaly_score=result.anomaly_score,
            triaged_to_vlm=result.triaged_to_vlm,
            pred_type=result.defect_type,
            type_correct=type_correct,
            stage1_seconds=result.stage1_seconds,
            stage2_seconds=result.stage2_seconds,
            report=result.report,
        ))
        if progress:
            mark = "?" if type_correct is None else ("Y" if type_correct else "N")
            print(f"  [{len(records):>3}] {true_type:<20} -> {result.verdict:<12} "
                  f"type={result.defect_type} correct={mark}")
    return records


def summarize(category: str, records: list[ImageRecord]) -> EvalSummary:
    def is_pred_defect(r: ImageRecord) -> bool:
        return r.verdict == "defect"

    tp = sum(1 for r in records if r.true_type != "good" and is_pred_defect(r))
    fp = sum(1 for r in records if r.true_type == "good" and is_pred_defect(r))
    tn = sum(1 for r in records if r.true_type == "good" and not is_pred_defect(r))
    fn = sum(1 for r in records if r.true_type != "good" and not is_pred_defect(r))

    typed = [r for r in records if r.type_correct is not None]
    per_type: dict[str, float] = {}
    for t in sorted({r.true_type for r in typed}):
        subset = [r for r in typed if r.true_type == t]
        per_type[t] = sum(r.type_correct for r in subset) / len(subset)

    n = len(records)
    return EvalSummary(
        category=category,
        n_images=n,
        tp=tp, fp=fp, tn=tn, fn=fn,
        detection_precision=tp / (tp + fp) if tp + fp else 0.0,
        detection_recall=tp / (tp + fn) if tp + fn else 0.0,
        detection_accuracy=(tp + tn) / n if n else 0.0,
        n_typed=len(typed),
        type_accuracy=(sum(r.type_correct for r in typed) / len(typed)) if typed else None,
        per_type_accuracy=per_type,
        vlm_call_rate=sum(r.triaged_to_vlm for r in records) / n if n else 0.0,
        mean_stage1_seconds=sum(r.stage1_seconds for r in records) / n if n else 0.0,
        mean_stage2_seconds=sum(r.stage2_seconds for r in records) / n if n else 0.0,
        mean_seconds_per_image=sum(r.stage1_seconds + r.stage2_seconds for r in records) / n if n else 0.0,
    )


def write_outputs(
    records: list[ImageRecord],
    summary: EvalSummary,
    out_dir: str | Path = "results/adjudication",
) -> tuple[Path, Path]:
    """Write per-image CSV and a summary JSON; returns their paths."""
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{summary.category}_records.csv"
    json_path = out / f"{summary.category}_summary.json"

    fields = [f for f in ImageRecord.__dataclass_fields__]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)

    json_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return csv_path, json_path
