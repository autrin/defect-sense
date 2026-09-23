import json

import numpy as np
import pytest
from PIL import Image

from defect_sense.pipeline.two_stage import Detection, TwoStagePipeline
from defect_sense.vlm.adjudicator import VLMAdjudicator
from defect_sense.vlm.client import OllamaClient


class FakeDetector:
    def __init__(self, score, anomaly_map=None):
        self.score = score
        self.anomaly_map = anomaly_map
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return Detection(score=self.score, anomaly_map=self.anomaly_map)


class CountingClient(OllamaClient):
    def __init__(self, reply):
        self.calls = 0

        def transport(payload):
            self.calls += 1
            return {"message": {"content": reply}}

        super().__init__(transport=transport)


def make_pipeline(score, reply, amap=None, threshold=0.5, decision_policy="advisory"):
    client = CountingClient(reply)
    pipeline = TwoStagePipeline(
        detector=FakeDetector(score, amap),
        adjudicator=VLMAdjudicator(client=client, category="bottle",
                                   defect_types=["broken_large", "broken_small", "contamination"]),
        threshold=threshold,
        decision_policy=decision_policy,
    )
    return pipeline, client


def test_clean_image_skips_vlm():
    pipeline, client = make_pipeline(score=0.1, reply="{}")
    result = pipeline.inspect(Image.new("RGB", (64, 64)))
    assert result.verdict == "pass"
    assert not result.triaged_to_vlm
    assert client.calls == 0
    assert result.stage2_seconds == 0.0


def test_flagged_image_reaches_vlm_with_regions():
    amap = np.zeros((64, 64))
    amap[10:20, 10:20] = 1.0
    reply = json.dumps({"is_defect": True, "defect_type": "contamination",
                        "confidence": 0.8, "bbox": [10, 10, 20, 20], "report": "dirt"})
    pipeline, client = make_pipeline(score=0.9, reply=reply, amap=amap)
    result = pipeline.inspect(Image.new("RGB", (64, 64)))
    assert result.verdict == "defect"
    assert result.defect_type == "contamination"
    assert client.calls == 1
    assert len(result.regions) == 1
    assert result.regions[0].bbox == (10, 10, 20, 20)


def test_vlm_overrules_false_positive():
    reply = json.dumps({"is_defect": False, "defect_type": "none",
                        "confidence": 0.9, "bbox": None, "report": "reflection, not a defect"})
    pipeline, _ = make_pipeline(score=0.99, reply=reply, decision_policy="override")
    result = pipeline.inspect(Image.new("RGB", (64, 64)))
    assert result.verdict == "false_alarm"
    assert result.defect_type is None


def test_region_boxes_scaled_from_heatmap_to_image_coords():
    amap = np.zeros((32, 32))
    amap[8:16, 8:16] = 1.0  # heatmap is quarter resolution of the 128px image
    reply = json.dumps({"is_defect": True, "defect_type": "broken_large",
                        "confidence": 0.8, "report": "crack"})
    pipeline, _ = make_pipeline(score=0.9, reply=reply, amap=amap)
    result = pipeline.inspect(Image.new("RGB", (128, 128)))
    assert result.regions[0].bbox == (32, 32, 64, 64)


def test_to_dict_is_json_serializable():
    reply = json.dumps({"is_defect": True, "defect_type": "broken_small",
                        "confidence": 0.8, "bbox": [1, 2, 3, 4], "report": "chip"})
    pipeline, _ = make_pipeline(score=0.9, reply=reply)
    result = pipeline.inspect(Image.new("RGB", (64, 64)))
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["verdict"] == "defect"
    assert payload["bbox"] is None  # no detector localization was provided
    assert payload["confidence"] is None
    assert payload["vlm_confidence"] == 0.8


@pytest.mark.parametrize("score", [0.5, 0.500049889087677, 0.99, 1.0])
def test_advisory_preserves_detector_flag_despite_confident_vlm_veto(score):
    reply = json.dumps({"is_defect": False, "defect_type": "none",
                        "confidence": 0.95, "report": "looks fine"})
    amap = np.zeros((64, 64))
    amap[10:20, 10:20] = 1.0
    pipeline, _ = make_pipeline(score, reply, amap=amap)
    result = pipeline.inspect(Image.new("RGB", (64, 64)))
    assert result.verdict == "defect"
    assert result.vlm_disagrees
    assert result.defect_type is None
    assert result.bbox == (10, 10, 20, 20)
    assert result.confidence is None
    assert "detector verdict retained" in result.report
    assert result.to_dict()["vlm_is_defect"] is False


def test_advisory_invalid_response_preserves_detector_flag():
    pipeline, _ = make_pipeline(0.9, "not json")
    result = pipeline.inspect(Image.new("RGB", (64, 64)))
    assert result.verdict == "defect"
    assert not result.vlm_disagrees
    assert not result.adjudication.parse_ok
    assert "invalid" in result.report
    payload = result.to_dict()
    assert payload["vlm_is_defect"] is None
    assert payload["vlm_confidence"] is None
    assert payload["vlm_parse_ok"] is False
    assert result.defect_type is None


def test_unknown_policy_rejected():
    with pytest.raises(ValueError, match="decision_policy"):
        make_pipeline(0.9, "{}", decision_policy="typo")


def test_detector_only_needs_no_vlm_and_keeps_localization():
    amap = np.zeros((64, 64))
    amap[10:20, 10:20] = 1.0
    pipeline = TwoStagePipeline(
        detector=FakeDetector(0.9, amap), decision_policy="detector-only"
    )
    result = pipeline.inspect(Image.new("RGB", (64, 64)))
    assert result.verdict == "defect"
    assert result.bbox == (10, 10, 20, 20)
    assert not result.triaged_to_vlm
    assert result.stage2_seconds == 0
    assert result.adjudication is None


def test_vlm_policies_require_adjudicator():
    with pytest.raises(ValueError, match="adjudicator"):
        TwoStagePipeline(detector=FakeDetector(0.9), decision_policy="advisory")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_threshold_and_score_rejected(value):
    with pytest.raises(ValueError, match="threshold"):
        make_pipeline(0.9, "{}", threshold=value)
    pipeline, _ = make_pipeline(value, "{}")
    with pytest.raises(ValueError, match="score"):
        pipeline.inspect(Image.new("RGB", (64, 64)))


@pytest.mark.parametrize("policy", ["advisory", "override"])
@pytest.mark.parametrize("failure", ["connection", "http", "timeout", "json", "truncated"])
def test_vlm_service_failure_preserves_only_advisory_detector_verdict(policy, failure):
    import requests

    pipeline, _ = make_pipeline(0.9, "{}", decision_policy=policy)

    def transport(payload):
        if failure == "http":
            raise requests.HTTPError("server error")
        if failure == "timeout":
            raise requests.Timeout("timed out")
        if failure == "json":
            raise requests.exceptions.JSONDecodeError("invalid JSON", "", 0)
        if failure == "truncated":
            raise requests.exceptions.ChunkedEncodingError("response interrupted")
        raise requests.ConnectionError("offline")

    pipeline.adjudicator.client = OllamaClient(transport=transport, retries=0)
    if policy == "override":
        with pytest.raises(ConnectionError):
            pipeline.inspect(Image.new("RGB", (64, 64)))
    else:
        result = pipeline.inspect(Image.new("RGB", (64, 64)))
        assert result.verdict == "defect"
        assert result.triaged_to_vlm
        assert result.vlm_error == "unavailable"
        assert result.adjudication is None
        assert result.confidence is None
        assert "unavailable" in result.report


@pytest.mark.parametrize("payload", [
    None, [], {}, {"message": None}, {"message": []},
    {"message": {}}, {"message": {"content": None}}, {"message": {"content": {}}},
])
@pytest.mark.parametrize("policy", ["advisory", "override"])
def test_malformed_vlm_envelope_uses_service_failure_policy(payload, policy):
    pipeline, _ = make_pipeline(0.9, "{}", decision_policy=policy)
    pipeline.adjudicator.client = OllamaClient(transport=lambda request: payload)
    if policy == "override":
        with pytest.raises(ConnectionError, match="invalid response"):
            pipeline.inspect(Image.new("RGB", (64, 64)))
    else:
        result = pipeline.inspect(Image.new("RGB", (64, 64)))
        assert result.verdict == "defect"
        assert result.vlm_error == "unavailable"
        assert result.to_dict()["vlm_is_defect"] is None


@pytest.mark.parametrize("score,verdict", [(0.49, "pass"), (0.5, "defect"), (0.9, "defect")])
def test_default_policy_never_calls_supplied_vlm(score, verdict):
    client = CountingClient("{}")
    pipeline = TwoStagePipeline(
        detector=FakeDetector(score),
        adjudicator=VLMAdjudicator(client, "bottle", ["contamination"]),
    )
    result = pipeline.inspect(Image.new("RGB", (64, 64)))
    assert result.verdict == verdict
    assert result.decision_policy == "detector-only"
    assert result.adjudication is None
    assert not result.triaged_to_vlm
    assert client.calls == 0


def test_saved_pilot_replay_preserves_all_detector_decisions():
    """Replay saved VLM decisions through real orchestration, without inference."""
    import csv
    from pathlib import Path

    records_path = Path(__file__).resolve().parents[1] / "results/adjudication/two-stage/bottle_records.csv"
    with records_path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    recovered = 0
    for record in records:
        reply = json.dumps({
            "is_defect": record["verdict"] == "defect",
            "defect_type": record["pred_type"] or "none",
            "confidence": float(record["confidence"] or 0),
            "report": record["report"],
        }) if record["parse_ok"] != "False" else "not json"
        pipeline, _ = make_pipeline(float(record["anomaly_score"]), reply)
        result = pipeline.inspect(Image.new("RGB", (64, 64)))
        assert (result.verdict == "defect") == (record["true_type"] != "good")
        recovered += record["verdict"] == "false_alarm" and result.verdict == "defect"
    assert len(records) == 20
    assert recovered == 9
