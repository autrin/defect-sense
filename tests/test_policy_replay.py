"""Replay saved controlled-run opinions through real runtime policy decisions."""

import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from defect_sense.pipeline.two_stage import Detection, TwoStagePipeline
from defect_sense.vlm.adjudicator import Adjudication
from scripts.eval_controlled import policy_records


@pytest.fixture(scope="module")
def saved_policy_snapshot():
    root = Path(__file__).resolve().parents[1] / "results/controlled_bottle_20260923"
    paths = [root / name for name in ("run.json", "detector.jsonl", "vlm.jsonl")]
    if not all(path.exists() for path in paths):
        pytest.skip("Optional controlled-run evidence is not available")
    config = json.loads(paths[0].read_text(encoding="utf-8"))["config"]
    rows = [json.loads(line) for line in paths[1].read_text(encoding="utf-8").splitlines()]
    # Snapshot once. Ignore only an unfinished trailing line if the producer is
    # currently appending; malformed complete lines must still fail this test.
    completed = [
        json.loads(line)
        for line in paths[2].read_text(encoding="utf-8").splitlines(keepends=True)
        if line.endswith("\n")
    ]
    if not completed:
        pytest.skip("No complete controlled-run journal entries yet")
    by_path = {payload["path"]: payload for payload in completed}
    assert len(by_path) == len(completed), "Duplicate journal paths"
    paired_rows = [row for row in rows if row["record"]["path"] in by_path]
    assert len(paired_rows) == len(completed), "Journal paths must have detector records"
    return config, paired_rows, by_path, policy_records(paired_rows, completed)


@pytest.mark.parametrize("policy", ["detector-only", "advisory", "override"])
def test_saved_policy_reconstruction_matches_runtime(saved_policy_snapshot, policy):
    config, rows, completed, reconstructed = saved_policy_snapshot
    expected_by_path = {record.path: record for record in reconstructed[policy]}
    outcomes = Counter()

    class CachedDetector:
        def __init__(self, score):
            self.score = score

        def predict(self, image):
            return Detection(self.score)

    class CachedAdjudicator:
        def __init__(self, raw):
            self.raw = raw
            self.calls = 0

        def adjudicate(self, image, **kwargs):
            self.calls += 1
            assert self.raw is not None, "Runtime called VLM for an unadjudicated row"
            return Adjudication(**self.raw)

    for row in rows:
        base = row["record"]
        expected = expected_by_path[base["path"]]
        adjudicator = CachedAdjudicator(completed[base["path"]]["raw_adjudication"])
        pipeline = TwoStagePipeline(
            detector=CachedDetector(base["anomaly_score"]),
            adjudicator=adjudicator,
            threshold=config["threshold"],
            decision_policy=policy,
        )
        # Decisions need only scores/opinions; no images, maps, torch, or HTTP.
        result = pipeline.inspect(Image.new("RGB", (16, 16)))
        payload = result.to_dict()
        actual = {
            "verdict": result.verdict,
            "anomaly_score": result.anomaly_score,
            "pred_type": result.defect_type,
            "type_correct": (result.defect_type == base["true_type"])
            if base["true_type"] != "good" and result.triaged_to_vlm else None,
            "triaged_to_vlm": result.triaged_to_vlm,
            "decision_policy": result.decision_policy,
            "parse_ok": payload["vlm_parse_ok"],
            "confidence": result.confidence,
            "vlm_is_defect": payload["vlm_is_defect"],
            "vlm_confidence": payload["vlm_confidence"],
            "vlm_disagrees": result.vlm_disagrees,
            "vlm_error": result.vlm_error,
        }
        assert actual == {
            key: getattr(expected, key) for key in actual
        }, f"{policy}: {base['path']}"
        assert adjudicator.calls == int(expected.triaged_to_vlm)
        outcomes[result.verdict] += 1
    print(f"{policy}: replayed {len(rows)} saved records; verdicts={dict(outcomes)}")
