"""Train with pilot-only calibration, then compare policies on the remaining images.

Run `detector` first, then `vlm` with the same --out-dir. Separate processes keep
PatchCore and Ollama from competing for GPU memory. VLM calls are journaled so
an interrupted evaluation can resume without re-running completed images.
Run `report` afterwards to verify the saved evidence and regenerate comparisons
without Ollama. Add --localization to score cached maps at the fixed threshold.
"""

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image

from defect_sense.eval.harness import (
    ImageRecord, collect_run_metadata, iter_test_images, summarize, write_outputs,
)
from defect_sense.pipeline.two_stage import Detection, TwoStagePipeline
from defect_sense.taxonomy import defect_types_for
from defect_sense.vlm.adjudicator import PROMPT_VERSION, VLMAdjudicator
from defect_sense.vlm.client import DEFAULT_MODEL, OllamaClient


def image_key(path):
    path = Path(path)
    return f"{path.parent.name}/{path.name}"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(path):
    """Hash UTF-8 evidence consistently across Git's LF/CRLF conversion."""
    text = Path(path).read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_split(dataset_root, category, pilot):
    with Path(pilot).open(newline="", encoding="utf-8") as stream:
        # Pilot CSVs may contain Windows paths even when reproduced on Linux.
        calibration = {
            image_key(row["path"].replace("\\", "/")) for row in csv.DictReader(stream)
        }
    images = list(iter_test_images(dataset_root, category))
    available = {image_key(path) for path, _ in images}
    if not calibration or not calibration < available:
        raise ValueError("Pilot must identify a nonempty proper subset of this category")
    manifest = [
        {"path": path.as_posix(), "true_type": label, "sha256": file_hash(path),
         "split": "calibration" if image_key(path) in calibration else "evaluation"}
        for path, label in images
    ]
    calibration_hashes = {item["sha256"] for item in manifest if item["split"] == "calibration"}
    evaluation_hashes = {item["sha256"] for item in manifest if item["split"] == "evaluation"}
    if calibration_hashes & evaluation_hashes:
        raise ValueError("Image content overlaps between calibration and evaluation")
    return calibration, manifest


def validate_run(output):
    """Reject incomplete detector data or mixed/duplicated resume records."""
    metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    groups = {
        split: [item for item in manifest if item["split"] == split]
        for split in ("calibration", "evaluation")
    }
    groups["training"] = metadata["training_images"]
    if len(groups["calibration"]) != config["n_calibration"] or len(groups["evaluation"]) != config["n_evaluation"]:
        raise ValueError("Manifest counts do not match run configuration")
    if len(groups["calibration"]) + len(groups["evaluation"]) != len(manifest):
        raise ValueError("Unknown manifest split")
    seen_hashes = set()
    for split, items in groups.items():
        if not items or len({item["path"] for item in items}) != len(items):
            raise ValueError(f"Empty or duplicate paths in {split} split")
        hashes = {item["sha256"] for item in items}
        if hashes & seen_hashes:
            raise ValueError("Image content overlaps between experiment splits")
        seen_hashes.update(hashes)
        for item in items:
            if file_hash(item["path"]) != item["sha256"]:
                raise ValueError(f"Image changed: {item['path']}")
    if file_hash(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since detector inference")
    if not math.isfinite(config["threshold"]):
        raise ValueError("Threshold must be finite")

    rows = [json.loads(line) for line in (output / "detector.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = {item["path"]: item["true_type"] for item in groups["evaluation"]}
    bases = {row["record"]["path"]: row["record"] for row in rows}
    if len(bases) != len(rows) or bases.keys() != expected.keys():
        raise ValueError("Detector records must cover the evaluation split exactly once")
    for path, base in bases.items():
        score = base["anomaly_score"]
        verdict = "defect" if score >= config["threshold"] else "pass"
        if not math.isfinite(score) or base["true_type"] != expected[path] or base["verdict"] != verdict:
            raise ValueError(f"Detector record disagrees with frozen experiment: {path}")
        if base["triaged_to_vlm"] or base["decision_policy"] != "detector-only":
            raise ValueError(f"Expected detector-only source record: {path}")
        if not math.isfinite(base["stage1_seconds"]) or base["stage1_seconds"] < 0:
            raise ValueError(f"Invalid detector latency: {path}")

    journal = output / "vlm.jsonl"
    completed = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] if journal.exists() else []
    done = set()
    for payload in completed:
        path = payload["path"]
        if path in done or path not in bases:
            raise ValueError(f"Duplicate or unexpected VLM record: {path}")
        done.add(path)
        base = bases[path]
        if any(payload[key] != base[key] for key in ("true_type", "anomaly_score", "verdict", "stage1_seconds")):
            raise ValueError(f"VLM journal disagrees with detector record: {path}")
        if payload["decision_policy"] != "advisory" or payload.get("vlm_error"):
            raise ValueError(f"Expected successful advisory attempt: {path}")
        if not math.isfinite(payload["stage2_seconds"]) or payload["stage2_seconds"] < 0:
            raise ValueError(f"Invalid VLM latency: {path}")
        raw = payload["raw_adjudication"]
        called = base["verdict"] == "defect"
        valid = raw is not None and raw["parse_ok"]
        decision = raw["is_defect"] if valid else None
        predicted_type = raw["defect_type"] if valid and decision else None
        type_correct = predicted_type == base["true_type"] if called and base["true_type"] != "good" else None
        if (
            payload["triaged_to_vlm"] != called
            or (raw is not None) != called
            or payload["parse_ok"] != (raw["parse_ok"] if raw else None)
            or payload["vlm_is_defect"] != decision
            or payload["vlm_confidence"] != (raw["confidence"] if valid else None)
            or payload["vlm_disagrees"] != (decision is False)
            or payload["pred_type"] != predicted_type
            or payload["type_correct"] != type_correct
        ):
            raise ValueError(f"VLM journal disagrees with raw adjudication: {path}")
    return metadata, rows, completed


def policy_records(rows, completed):
    by_path = {payload["path"]: payload for payload in completed}
    if len(by_path) != len(rows):
        raise ValueError("VLM stage is incomplete; resume it before reporting")
    detector = [ImageRecord(**row["record"]) for row in rows]
    advisory, override = [], []
    for base in detector:
        payload = by_path[base.path]
        record = ImageRecord(**{key: value for key, value in payload.items() if key != "raw_adjudication"})
        advisory.append(record)
        raw = payload["raw_adjudication"]
        override.append(replace(
            record, verdict="false_alarm" if record.vlm_disagrees else record.verdict,
            decision_policy="override", bbox=raw["bbox"] if raw else None,
            confidence=raw["confidence"] if raw and raw["parse_ok"] else None,
            report=raw["report"] if raw else record.report,
        ))
    return {"detector-only": detector, "advisory": advisory, "override": override}


def wilson_interval(successes, total):
    """Two-sided 95% binomial interval; descriptive, not a deployment guarantee."""
    if not total:
        return None
    z = 1.959963984540054
    rate = successes / total
    scale = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / scale
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def localization_metrics(output, rows, dataset_root, category):
    """Score cached maps at 0.5 in native ground-truth coordinates, without tuning."""
    records = []
    for row in rows:
        base = row["record"]
        path = Path(base["path"])
        anomaly_map = np.load(output / "maps" / row["map"], allow_pickle=False)
        if anomaly_map.ndim != 2 or not np.isfinite(anomaly_map).all():
            raise ValueError(f"Invalid anomaly map: {row['map']}")
        with Image.open(path) as image:
            size = image.size
        resized = Image.fromarray(anomaly_map.astype(np.float32)).resize(size, Image.Resampling.BILINEAR)
        predicted = np.asarray(resized) >= 0.5
        mask_hash = None
        if base["true_type"] == "good":
            truth = np.zeros(predicted.shape, dtype=bool)
        else:
            mask_path = Path(dataset_root) / category / "ground_truth" / base["true_type"] / f"{path.stem}_mask.png"
            with Image.open(mask_path) as mask:
                truth = np.asarray(mask.convert("L")) > 0
            if truth.shape != predicted.shape:
                raise ValueError(f"Ground-truth mask dimensions differ: {mask_path}")
            mask_hash = file_hash(mask_path)
        tp = int(np.count_nonzero(predicted & truth))
        fp = int(np.count_nonzero(predicted & ~truth))
        fn = int(np.count_nonzero(~predicted & truth))
        records.append({
            "path": base["path"], "tp": tp, "fp": fp, "fn": fn,
            "map_sha256": file_hash(output / "maps" / row["map"]),
            "mask_sha256": mask_hash,
        })
    tp, fp, fn = (sum(record[key] for record in records) for key in ("tp", "fp", "fn"))
    return {
        "threshold": 0.5, "resize": "PIL bilinear to native image dimensions",
        "aggregation": "micro over all evaluation pixels, including good images",
        "tp": tp, "fp": fp, "fn": fn,
        "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
        "pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
        "pixel_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "pixel_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "records": records,
    }


def run_report(args):
    metadata, rows, completed = validate_run(args.out_dir)
    config = metadata["config"]
    policies = policy_records(rows, completed)
    report = {
        "report_version": 1,
        "category": config["category"],
        "verification": {
            "image_hashes_match": True,
            "split_content_disjoint": True,
            "checkpoint_hash_matches": True,
            "evaluation_records_complete_and_unique": True,
            "report_script_sha256": text_hash(__file__),
        },
        "policies": {}, "per_type": {},
    }
    for name, records in policies.items():
        summary = summarize(config["category"], records)
        write_outputs(records, summary, args.out_dir / name, metadata)
        report["policies"][name] = {
            **asdict(summary),
            "recall_wilson_95": wilson_interval(summary.tp, summary.tp + summary.fn),
            "specificity_wilson_95": wilson_interval(summary.tn, summary.tn + summary.fp),
        }
        report["per_type"][name] = {
            label: asdict(summarize(config["category"], [r for r in records if r.true_type == label]))
            for label in sorted({r.true_type for r in records})
        }
    report["harmful_vetoes"] = sum(r.vlm_disagrees and r.true_type != "good" for r in policies["advisory"])
    report["corrected_false_alarms"] = sum(r.vlm_disagrees and r.true_type == "good" for r in policies["advisory"])
    report["text_hash_format"] = "sha256 of UTF-8 text with LF newlines"
    report["source_sha256"] = {
        name: text_hash(args.out_dir / name)
        for name in ("run.json", "manifest.json", "detector.jsonl", "vlm.jsonl")
    }
    if args.localization:
        report["localization"] = localization_metrics(args.out_dir, rows, config["dataset_root"], config["category"])
    (args.out_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: report[name] for name in ("policies", "harmful_vetoes", "corrected_false_alarms")}, indent=2))


def run_detector(args):
    if not math.isfinite(args.threshold):
        raise ValueError("Threshold must be finite")

    import torch
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from lightning import seed_everything

    from defect_sense.detectors.anomalib_detector import AnomalibDetector, _build_model

    output = args.out_dir
    if (output / "manifest.json").exists():
        raise ValueError("Use a new output directory for each detector experiment")
    calibration, manifest = make_split(args.dataset_root, args.category, args.pilot)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    class PilotCalibrationData(MVTecAD):
        def _create_val_split(self):
            # Isolate the experiment split here; no changes to dataset files.
            all_test = self.test_data
            is_calibration = [image_key(p) in calibration for p in all_test.samples.image_path]
            self.val_data = all_test.subsample([i for i, used in enumerate(is_calibration) if used])
            self.test_data = all_test.subsample([i for i, used in enumerate(is_calibration) if not used])
            if {image_key(p) for p in self.val_data.samples.image_path} != calibration:
                raise ValueError("Anomalib calibration split differs from the manifest")
            expected = {image_key(item["path"]) for item in manifest if item["split"] == "evaluation"}
            if {image_key(p) for p in self.test_data.samples.image_path} != expected:
                raise ValueError("Anomalib evaluation split differs from the manifest")

    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    data = PilotCalibrationData(
        root=str(args.dataset_root), category=args.category,
        train_batch_size=8, eval_batch_size=8, num_workers=0, seed=args.seed,
    )
    model = _build_model("patchcore")
    engine = Engine(
        max_epochs=1, accelerator="gpu", devices=1, logger=False,
        enable_progress_bar=False, default_root_dir=str(output / "training"),
    )
    start = time.perf_counter()
    engine.fit(model=model, datamodule=data)
    fit_seconds = time.perf_counter() - start
    checkpoint = engine.best_model_path
    metadata = collect_run_metadata(
        category=args.category, detector="patchcore", seed=args.seed,
        checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
        threshold=args.threshold, prompt_version=PROMPT_VERSION, vlm_model=DEFAULT_MODEL,
        calibration="existing pilot only", n_calibration=len(data.val_data),
        n_evaluation=len(data.test_data), fit_seconds=fit_seconds,
        dataset_root=args.dataset_root.as_posix(),
    )
    metadata["training_images"] = [
        {"path": str(p), "sha256": file_hash(p)} for p in data.train_data.samples.image_path
    ]
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    detector = AnomalibDetector(_model=model, _engine=engine)
    records = []
    cache = output / "maps"
    cache.mkdir(exist_ok=True)
    with (output / "detector.jsonl").open("w", encoding="utf-8") as stream:
        for item in manifest:
            if item["split"] != "evaluation":
                continue
            path = Path(item["path"])
            start = time.perf_counter()
            with Image.open(path) as image:
                detection = detector.predict(image)
            elapsed = time.perf_counter() - start
            name = f"{path.parent.name}_{path.stem}.npy"
            np.save(cache / name, detection.anomaly_map)
            record = ImageRecord(
                path=path.as_posix(), true_type=item["true_type"],
                verdict="defect" if detection.score >= args.threshold else "pass",
                anomaly_score=detection.score, triaged_to_vlm=False,
                pred_type=None, type_correct=None, stage1_seconds=elapsed, stage2_seconds=0,
                decision_policy="detector-only",
            )
            stream.write(json.dumps({"record": asdict(record), "map": name}) + "\n")
            stream.flush()
            records.append(record)
            print(f"[{len(records)}] {image_key(path)} score={detection.score:.6f}", flush=True)
    write_outputs(records, summarize(args.category, records), output / "detector-only", metadata)
    print(json.dumps(asdict(summarize(args.category, records)), indent=2), flush=True)


def run_vlm(args):
    import requests

    output = args.out_dir
    metadata, rows, completed = validate_run(output)
    config = metadata["config"]
    if config["prompt_version"] != PROMPT_VERSION or config["category"] != args.category:
        raise ValueError("Run configuration does not match current prompt/category")
    response = requests.get("http://localhost:11434/api/tags", timeout=10)
    response.raise_for_status()
    model_info = next(m for m in response.json()["models"] if m["name"] == config["vlm_model"])
    if "ollama_model" in metadata and metadata["ollama_model"]["digest"] != model_info["digest"]:
        raise ValueError("Ollama model changed since this run started")
    metadata["ollama_model"] = model_info
    metadata.setdefault("vlm_started_at_utc", datetime.now(timezone.utc).isoformat())
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    journal = output / "vlm.jsonl"
    done = {r["path"] for r in completed}
    client = OllamaClient(model=config["vlm_model"], timeout=180, retries=0)
    adjudicator = VLMAdjudicator(
        client=client, category=args.category,
        defect_types=defect_types_for(args.category, config["dataset_root"]),
    )

    class CachedDetector:
        def predict(self, image):
            return self.detection

    detector = CachedDetector()
    pipeline = TwoStagePipeline(
        detector, adjudicator, threshold=config["threshold"], decision_policy="advisory"
    )
    with journal.open("a", encoding="utf-8") as stream:
        for row in rows:
            base = ImageRecord(**row["record"])
            if base.path in done:
                continue
            detector.detection = Detection(
                base.anomaly_score, np.load(output / "maps" / row["map"], allow_pickle=False)
            )
            with Image.open(base.path) as image:
                result = pipeline.inspect(image)
            # An outage is not model-quality evidence; stop rather than report a win.
            if result.vlm_error:
                raise RuntimeError("VLM unavailable; resume after restoring Ollama")
            record = replace(
                base, verdict=result.verdict, triaged_to_vlm=result.triaged_to_vlm,
                pred_type=result.defect_type,
                type_correct=(result.defect_type == base.true_type)
                if base.true_type != "good" and result.triaged_to_vlm else None,
                stage2_seconds=result.stage2_seconds, report=result.report,
                bbox=result.bbox, region_count=len(result.regions), decision_policy="advisory",
                parse_ok=result.adjudication.parse_ok if result.adjudication else None,
                vlm_is_defect=result.to_dict()["vlm_is_defect"],
                vlm_confidence=result.to_dict()["vlm_confidence"], vlm_disagrees=result.vlm_disagrees,
            )
            payload = asdict(record)
            payload["raw_adjudication"] = asdict(result.adjudication) if result.adjudication else None
            stream.write(json.dumps(payload) + "\n")
            stream.flush()
            completed.append(payload)
            print(f"[{len(completed)}/{len(rows)}] {image_key(base.path)} "
                  f"detector={base.verdict} vlm={record.vlm_is_defect} "
                  f"seconds={record.stage2_seconds:.1f}", flush=True)

    metadata.setdefault("completed_at_utc", datetime.now(timezone.utc).isoformat())
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for name, records in policy_records(rows, completed).items():
        write_outputs(records, summarize(args.category, records), output / name, metadata)
        print(name, json.dumps(asdict(summarize(args.category, records)), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["detector", "vlm", "report"])
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/MVTecAD"))
    parser.add_argument("--pilot", type=Path, default=Path("results/adjudication/two-stage/bottle_records.csv"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--localization", action="store_true", help="Include fixed-threshold pixel metrics in the offline report")
    args = parser.parse_args()
    {"detector": run_detector, "vlm": run_vlm, "report": run_report}[args.stage](args)


if __name__ == "__main__":
    main()
