import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from defect_sense.eval.harness import ImageRecord, summarize
from scripts.eval_controlled import (
    file_hash, localization_metrics, make_split, policy_records, run_report,
    text_hash, validate_run, wilson_interval,
)


def test_pilot_calibration_is_disjoint_from_evaluation(tmp_path):
    root = tmp_path / "data"
    folder = root / "bottle" / "test" / "good"
    folder.mkdir(parents=True)
    (folder / "001.png").write_bytes(b"calibration")
    (folder / "002.png").write_bytes(b"evaluation")
    pilot = tmp_path / "pilot.csv"
    with pilot.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path"])
        writer.writeheader()
        writer.writerow({"path": r"datasets\MVTecAD\bottle\test\good\001.png"})
    calibration, manifest = make_split(root, "bottle", pilot)
    assert calibration == {"good/001.png"}
    assert [r["split"] for r in manifest] == ["calibration", "evaluation"]
    assert all(len(r["sha256"]) == 64 for r in manifest)
    (folder / "001.png").unlink()
    with pytest.raises(ValueError, match="proper subset"):
        make_split(root, "bottle", pilot)


@pytest.fixture
def saved_run(tmp_path):
    manifest, rows, journal = [], [], []
    for i, (label, score, parse_ok, decision) in enumerate([
        ("crack", 0.9, True, False),
        ("crack", 0.8, False, None),
        ("good", 0.2, None, None),
        ("good", 0.7, True, False),
    ]):
        image = tmp_path / f"{i}.png"
        image.write_bytes(f"evaluation {i}".encode())
        manifest.append({"path": str(image), "true_type": label, "split": "evaluation", "sha256": file_hash(image)})
        base = ImageRecord(
            path=str(image), true_type=label, verdict="defect" if score >= 0.5 else "pass",
            anomaly_score=score, triaged_to_vlm=False, pred_type=None, type_correct=None,
            stage1_seconds=0.1, stage2_seconds=0.0,
        )
        rows.append({"record": asdict(base), "map": f"{i}.npy"})
        called = score >= 0.5
        record = replace(
            base, decision_policy="advisory", triaged_to_vlm=called,
            stage2_seconds=1.0 if called else 0.0, parse_ok=parse_ok,
            vlm_is_defect=decision, vlm_disagrees=decision is False,
            vlm_confidence=0.95 if parse_ok else None,
            type_correct=False if label != "good" and called else None,
        )
        raw = {
            "parse_ok": parse_ok, "is_defect": decision if parse_ok else True,
            "defect_type": "none" if parse_ok else "unknown",
            "confidence": 0.95, "bbox": None, "report": "saved reply",
        } if called else None
        journal.append({**asdict(record), "raw_adjudication": raw})
    calibration = tmp_path / "calibration.png"
    calibration.write_bytes(b"calibration")
    manifest.append({"path": str(calibration), "split": "calibration", "sha256": file_hash(calibration)})
    train = tmp_path / "train.png"
    train.write_bytes(b"training")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    metadata = {
        "config": {
            "category": "bottle", "n_calibration": 1, "n_evaluation": 4,
            "threshold": 0.5, "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint),
        },
        "training_images": [{"path": str(train), "sha256": file_hash(train)}],
    }
    for name, data in (("run.json", metadata), ("manifest.json", manifest)):
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    for name, data in (("detector.jsonl", rows), ("vlm.jsonl", journal)):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
    return tmp_path


def test_offline_report_recomputes_paired_policies(saved_run, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Offline reports must not request model inference")

    monkeypatch.setattr("requests.get", no_network)
    run_report(SimpleNamespace(out_dir=saved_run, localization=False))
    report = json.loads((saved_run / "comparison.json").read_text())
    assert report["harmful_vetoes"] == 1
    assert report["corrected_false_alarms"] == 1
    for name, counts in (
        ("detector-only", (2, 1, 1, 0)),
        ("advisory", (2, 1, 1, 0)),
        ("override", (1, 0, 2, 1)),
    ):
        summary = report["policies"][name]
        assert tuple(summary[key] for key in ("tp", "fp", "tn", "fn")) == counts
        assert (saved_run / name / "bottle_records.csv").exists()
    assert report["policies"]["advisory"]["n_invalid_responses"] == 1
    assert report["policies"]["advisory"]["type_accuracy"] == 0
    assert report["policies"]["detector-only"]["vlm_call_rate"] == 0


@pytest.mark.parametrize("filename", ["detector.jsonl", "vlm.jsonl"])
def test_reject_duplicate_journal_rows(saved_run, filename):
    path = saved_run / filename
    text = path.read_text()
    path.write_text(text + text.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="exactly once|Duplicate"):
        validate_run(saved_run)


@pytest.mark.parametrize("field,value", [
    ("anomaly_score", 0.123), ("true_type", "good"), ("verdict", "false_alarm"),
    ("vlm_is_defect", True), ("vlm_disagrees", False), ("pred_type", "crack"),
    ("vlm_confidence", 0.1), ("type_correct", True),
    ("stage2_seconds", float("nan")), ("vlm_error", "unavailable"),
])
def test_reject_mismatched_vlm_records(saved_run, field, value):
    path = saved_run / "vlm.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        validate_run(saved_run)


def test_partial_journal_can_resume_but_not_report(saved_run):
    path = saved_run / "vlm.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    _, rows, completed = validate_run(saved_run)
    assert len(completed) == 1
    with pytest.raises(ValueError, match="incomplete"):
        policy_records(rows, completed)


@pytest.mark.parametrize("filename", ["calibration.png", "train.png", "0.png", "model.ckpt"])
def test_reject_changed_inputs(saved_run, filename):
    (saved_run / filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        validate_run(saved_run)


def test_reject_content_overlap_even_with_distinct_paths(saved_run):
    path = saved_run / "manifest.json"
    manifest = json.loads(path.read_text())
    calibration = saved_run / "calibration.png"
    calibration.write_bytes((saved_run / "0.png").read_bytes())
    manifest[-1]["sha256"] = file_hash(calibration)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlaps"):
        validate_run(saved_run)


def test_wilson_interval_does_not_claim_certainty():
    low, high = wilson_interval(48, 48)
    assert low == pytest.approx(0.925899870333, abs=1e-10)
    assert high == pytest.approx(1.0)
    assert wilson_interval(0, 0) is None


def test_text_evidence_hash_ignores_platform_line_endings(tmp_path):
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    lf.write_bytes(b'{"result": 1}\n{"result": 2}\n')
    crlf.write_bytes(b'{"result": 1}\r\n{"result": 2}\r\n')
    assert text_hash(lf) == text_hash(crlf)
    assert file_hash(lf) != file_hash(crlf)


def test_fixed_threshold_localization_counts_good_image_false_positives(tmp_path):
    (tmp_path / "maps").mkdir()
    mask_dir = tmp_path / "data" / "bottle" / "ground_truth" / "crack"
    mask_dir.mkdir(parents=True)
    rows = []
    for label in ("crack", "good"):
        image = tmp_path / f"{label}.png"
        Image.new("RGB", (2, 2)).save(image)
        np.save(tmp_path / "maps" / f"{label}.npy", np.array([[0.8, 0.8], [0.1, 0.1]]))
        rows.append({"record": {"path": str(image), "true_type": label}, "map": f"{label}.npy"})
    Image.fromarray(np.array([[255, 0], [255, 0]], dtype=np.uint8)).save(mask_dir / "crack_mask.png")
    result = localization_metrics(tmp_path, rows, tmp_path / "data", "bottle")
    assert (result["tp"], result["fp"], result["fn"]) == (1, 3, 1)
    assert result["pixel_f1"] == pytest.approx(1 / 3)
    assert result["pixel_iou"] == pytest.approx(1 / 5)


def test_published_controlled_results_match_saved_evidence():
    output = Path(__file__).resolve().parents[1] / "results" / "controlled_bottle_20260923"
    rows = [json.loads(line) for line in (output / "detector.jsonl").read_text(encoding="utf-8").splitlines()]
    completed = [json.loads(line) for line in (output / "vlm.jsonl").read_text(encoding="utf-8").splitlines()]
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert len(rows) == len(completed) == 63
    for name, digest in comparison["source_sha256"].items():
        assert text_hash(output / name) == digest
    for name, records in policy_records(rows, completed).items():
        summary = asdict(summarize("bottle", records))
        published = json.loads((output / name / "bottle_summary.json").read_text(encoding="utf-8"))
        assert {key: published[key] for key in summary} == summary
        assert {key: comparison["policies"][name][key] for key in summary} == summary
    assert comparison["harmful_vetoes"] == 27
    assert comparison["corrected_false_alarms"] == 0
    assert comparison["policies"]["override"]["fn"] == 27
    assert comparison["policies"]["detector-only"]["fn"] == 0
    assert comparison["policies"]["advisory"]["n_invalid_responses"] == 4
    assert comparison["policies"]["advisory"]["type_accuracy"] == 7 / 48
