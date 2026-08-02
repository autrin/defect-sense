import json

from defect_sense.eval.harness import (
    ImageRecord,
    evaluate_category,
    summarize,
    write_outputs,
)


def make_record(
    true_type, verdict, pred_type=None, type_correct=None, triaged=True, s1=0.1, s2=2.0
):
    return ImageRecord(
        path="x.png",
        true_type=true_type,
        verdict=verdict,
        anomaly_score=0.5,
        triaged_to_vlm=triaged,
        pred_type=pred_type,
        type_correct=type_correct,
        stage1_seconds=s1,
        stage2_seconds=s2 if triaged else 0.0,
    )


def test_summary_confusion_and_type_accuracy():
    records = [
        make_record("good", "pass", triaged=False),  # TN
        make_record("good", "defect", pred_type="contamination"),  # FP
        make_record("good", "false_alarm"),  # TN (VLM caught it)
        make_record("crack", "defect", "crack", type_correct=True),  # TP, typed right
        make_record("crack", "defect", "hole", type_correct=False),  # TP, typed wrong
        make_record("hole", "pass", triaged=False),  # FN
    ]
    s = summarize("hazelnut", records)
    assert (s.tp, s.fp, s.tn, s.fn) == (2, 1, 2, 1)
    assert s.detection_precision == 2 / 3
    assert s.detection_recall == 2 / 3
    assert s.n_typed == 2
    assert s.type_accuracy == 0.5
    assert s.per_type_accuracy == {"crack": 0.5}
    assert s.vlm_call_rate == 4 / 6


def test_cost_metrics():
    records = [
        make_record("good", "pass", triaged=False, s1=0.1),
        make_record("crack", "defect", "crack", type_correct=True, s1=0.1, s2=4.0),
    ]
    s = summarize("hazelnut", records)
    assert s.vlm_call_rate == 0.5
    assert abs(s.mean_stage2_seconds - 2.0) < 1e-9
    assert abs(s.mean_seconds_per_image - 2.1) < 1e-9


def test_evaluate_category_walks_folders(tmp_path):
    from PIL import Image

    for folder, count in (("good", 3), ("crack", 2)):
        d = tmp_path / "hazelnut" / "test" / folder
        d.mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (16, 16)).save(d / f"{i:03d}.png")

    class FakePipeline:
        def inspect(self, image):
            from defect_sense.pipeline.two_stage import InspectionResult

            return InspectionResult(
                verdict="pass", anomaly_score=0.1, triaged_to_vlm=False
            )

    records = evaluate_category(FakePipeline(), tmp_path, "hazelnut", progress=False)
    assert len(records) == 5
    assert sum(r.true_type == "good" for r in records) == 3

    limited = evaluate_category(
        FakePipeline(), tmp_path, "hazelnut", limit=1, progress=False
    )
    assert len(limited) == 2  # one per defect type, including good


def test_write_outputs(tmp_path):
    records = [make_record("crack", "defect", "crack", type_correct=True)]
    records[0].confidence = 0.92
    records[0].bbox = (1, 2, 10, 12)
    records[0].parse_ok = True
    records[0].region_count = 2
    summary = summarize("hazelnut", records)
    metadata = {"git_commit": "abc123", "config": {"mode": "two-stage"}}
    csv_path, json_path = write_outputs(
        records,
        summary,
        tmp_path,
        run_metadata=metadata,
    )
    assert csv_path.exists()
    data = json.loads(json_path.read_text())
    assert data["category"] == "hazelnut"
    assert data["tp"] == 1
    assert data["run"] == metadata
    csv_text = csv_path.read_text()
    assert "confidence,bbox,parse_ok,region_count" in csv_text
    assert "0.92" in csv_text
