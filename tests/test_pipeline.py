import json

import numpy as np
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


def make_pipeline(score, reply, amap=None, threshold=0.5):
    client = CountingClient(reply)
    pipeline = TwoStagePipeline(
        detector=FakeDetector(score, amap),
        adjudicator=VLMAdjudicator(client=client, category="bottle",
                                   defect_types=["broken_large", "broken_small", "contamination"]),
        threshold=threshold,
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
    pipeline, _ = make_pipeline(score=0.99, reply=reply)
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
    assert payload["bbox"] == [1, 2, 3, 4]
