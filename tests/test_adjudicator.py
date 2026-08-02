import json

import numpy as np
from PIL import Image

from defect_sense.regions import Region
from defect_sense.vlm.adjudicator import VLMAdjudicator, _extract_json
from defect_sense.vlm.client import OllamaClient

BOTTLE_TYPES = ["broken_large", "broken_small", "contamination"]


def make_client(reply: str, captured: dict | None = None) -> OllamaClient:
    def transport(payload):
        if captured is not None:
            captured.update(payload)
        return {"message": {"content": reply}}

    return OllamaClient(transport=transport)


def make_adjudicator(reply: str, captured: dict | None = None) -> VLMAdjudicator:
    return VLMAdjudicator(
        client=make_client(reply, captured),
        category="bottle",
        defect_types=BOTTLE_TYPES,
    )


def test_happy_path_defect():
    reply = json.dumps(
        {
            "is_defect": True,
            "defect_type": "contamination",
            "confidence": 0.9,
            "bbox": [10, 20, 50, 60],
            "report": "Dark particulate contamination near the rim.",
        }
    )
    adj = make_adjudicator(reply)
    result = adj.adjudicate(Image.new("RGB", (100, 100)), anomaly_score=0.8)
    assert result.is_defect
    assert result.defect_type == "contamination"
    assert result.bbox == (10, 20, 50, 60)
    assert result.confidence == 0.9
    assert result.parse_ok


def test_false_alarm():
    reply = json.dumps(
        {
            "is_defect": False,
            "defect_type": "none",
            "confidence": 0.7,
            "bbox": None,
            "report": "The flagged region is a normal reflection.",
        }
    )
    result = make_adjudicator(reply).adjudicate(Image.new("RGB", (100, 100)))
    assert not result.is_defect
    assert result.defect_type == "none"
    assert result.bbox is None


def test_code_fenced_json_is_parsed():
    reply = '```json\n{"is_defect": true, "defect_type": "broken_small", "confidence": 0.5, "report": "chip"}\n```'
    result = make_adjudicator(reply).adjudicate(Image.new("RGB", (64, 64)))
    assert result.parse_ok
    assert result.defect_type == "broken_small"


def test_garbage_output_flags_unknown_not_crash():
    result = make_adjudicator("I think it looks broken???").adjudicate(
        Image.new("RGB", (64, 64))
    )
    assert not result.parse_ok
    assert result.defect_type == "unknown"
    assert result.is_defect  # fail-safe: unparseable => keep it flagged for a human


def test_out_of_taxonomy_type_coerced_to_unknown():
    reply = json.dumps(
        {
            "is_defect": True,
            "defect_type": "shattered",
            "confidence": 1.0,
            "report": "x",
        }
    )
    result = make_adjudicator(reply).adjudicate(Image.new("RGB", (64, 64)))
    assert result.defect_type == "unknown"


def test_bbox_clamped_and_invalid_dropped():
    reply = json.dumps(
        {
            "is_defect": True,
            "defect_type": "contamination",
            "confidence": 1.2,
            "bbox": [-10, 5, 9999, 50],
            "report": "x",
        }
    )
    result = make_adjudicator(reply).adjudicate(Image.new("RGB", (100, 100)))
    assert result.bbox == (0, 5, 100, 50)
    assert result.confidence == 1.0  # clamped

    reply = json.dumps(
        {
            "is_defect": True,
            "defect_type": "contamination",
            "confidence": 0.5,
            "bbox": [5, 5],
            "report": "x",
        }
    )
    result = make_adjudicator(reply).adjudicate(Image.new("RGB", (100, 100)))
    assert result.bbox is None


def test_bbox_scaled_back_to_original_resolution():
    reply = json.dumps(
        {
            "is_defect": True,
            "defect_type": "contamination",
            "confidence": 0.9,
            "bbox": [0, 0, 100, 100],
            "report": "x",
        }
    )
    adj = make_adjudicator(reply)
    adj.max_image_side = 100  # force 4x downscale of a 400px image
    result = adj.adjudicate(Image.new("RGB", (400, 400)))
    assert result.bbox == (0, 0, 400, 400)


def test_payload_contains_images_schema_and_regions():
    captured: dict = {}
    reply = json.dumps(
        {"is_defect": False, "defect_type": "none", "confidence": 1.0, "report": "ok"}
    )
    adj = make_adjudicator(reply, captured)
    amap = np.zeros((32, 32))
    amap[8:16, 8:16] = 1.0
    regions = [Region(bbox=(8, 8, 16, 16), area=64, peak_score=1.0, mean_score=1.0)]
    adj.adjudicate(
        Image.new("RGB", (32, 32)), anomaly_score=0.9, regions=regions, anomaly_map=amap
    )

    assert captured["model"] == "qwen3-vl:8b"
    assert captured["format"]["properties"]["defect_type"]["enum"] == [
        *BOTTLE_TYPES,
        "none",
    ]
    user_msg = captured["messages"][-1]
    assert len(user_msg["images"]) == 2  # original + heatmap overlay
    assert "[8, 8, 16, 16]" in user_msg["content"]
    assert "bottle" in user_msg["content"]
    assert "unaltered original photo" in user_msg["content"]
    assert "heatmap colors are not product features" in user_msg["content"]


def test_extract_json_prose_wrapped():
    assert _extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert _extract_json("no json here") is None
