from fastapi.testclient import TestClient
import pytest

from app.main import app

client = TestClient(app)


def test_inspection_console_and_health_endpoint():
    page = client.get("/")
    assert page.status_code == 200
    assert "Visual quality control, with evidence." in page.text
    assert "Run inspection" in page.text

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["vlm_model"] == "qwen3-vl:8b"


def test_inspect_rejects_unknown_category_before_inference():
    response = client.post(
        "/inspect",
        data={"category": "not-a-category"},
        files={"file": ("sample.png", b"content", "image/png")},
    )
    assert response.status_code == 400
    assert "Unknown category" in response.json()["detail"]


def test_inspect_rejects_non_image_media_type():
    response = client.post(
        "/inspect",
        data={"category": "bottle"},
        files={"file": ("sample.txt", b"content", "text/plain")},
    )
    assert response.status_code == 415
    assert response.json()["detail"] == "Upload must be an image"


def test_inspect_rejects_unreadable_image():
    response = client.post(
        "/inspect",
        data={"category": "bottle"},
        files={"file": ("sample.png", b"not an image", "image/png")},
    )
    assert response.status_code == 400
    assert "Not a readable image" in response.json()["detail"]


def test_vlm_only_fallback_retains_vlm_decisions(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "CKPT", None)
    main.get_pipeline.cache_clear()
    try:
        assert main.get_pipeline("bottle").decision_policy == "override"
    finally:
        main.get_pipeline.cache_clear()


def test_inspection_api_retains_detector_verdict_when_advisory_vlm_disagrees(monkeypatch):
    import io
    import json

    import app.main as main
    from PIL import Image
    from defect_sense.pipeline.two_stage import Detection, TwoStagePipeline
    from defect_sense.vlm.adjudicator import VLMAdjudicator
    from defect_sense.vlm.client import OllamaClient

    class Detector:
        def predict(self, image):
            return Detection(score=0.9)

    reply = json.dumps({"is_defect": False, "defect_type": "none",
                        "confidence": 0.95, "report": "looks fine"})
    pipeline = TwoStagePipeline(
        detector=Detector(),
        decision_policy="advisory",
        adjudicator=VLMAdjudicator(
            client=OllamaClient(transport=lambda payload: {"message": {"content": reply}}),
            category="bottle", defect_types=["contamination"],
        ),
    )
    monkeypatch.setattr(main, "CKPT", "test.ckpt")
    monkeypatch.setattr(main, "CKPT_CATEGORY", "bottle")
    monkeypatch.setattr(main, "get_pipeline", lambda category: pipeline)
    contents = io.BytesIO()
    Image.new("RGB", (16, 16)).save(contents, format="PNG")
    response = client.post("/inspect", data={"category": "bottle"},
                           files={"file": ("sample.png", contents.getvalue(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["verdict"] == "defect"
    assert result["decision_policy"] == "advisory"
    assert result["confidence"] is None
    assert result["vlm_confidence"] == 0.95
    assert result["vlm_disagrees"]
    assert result["mode"] == "two-stage"


@pytest.mark.parametrize("checkpoint,category,policy,expected_policy,mode", [
    ("test.ckpt", "bottle", "detector-only", "detector-only", "detector-only"),
    ("test.ckpt", "bottle", "advisory", "advisory", "two-stage"),
    ("test.ckpt", "bottle", "override", "override", "two-stage"),
    ("test.ckpt", "cable", "detector-only", "override", "vlm-only"),
    (None, "bottle", "detector-only", "override", "vlm-only"),
])
def test_real_pipeline_factory_and_api_policy(
    monkeypatch, checkpoint, category, policy, expected_policy, mode
):
    import io
    import json

    import app.main as main
    from PIL import Image
    from defect_sense.detectors import AnomalibDetector
    from defect_sense.pipeline.two_stage import Detection
    from defect_sense.vlm.client import OllamaClient

    monkeypatch.setattr(main, "CKPT", checkpoint)
    monkeypatch.setattr(main, "CKPT_CATEGORY", "bottle")
    monkeypatch.setattr(main, "DECISION_POLICY", policy)
    monkeypatch.setattr(main, "THRESHOLD", 0.5)
    monkeypatch.setattr(AnomalibDetector, "predict", lambda self, image: Detection(0.9))
    calls = []
    reply = json.dumps({
        "is_defect": False, "defect_type": "none", "confidence": 0.9, "report": "fine",
    })

    def make_client(**kwargs):
        assert expected_policy != "detector-only", "Detector-only must not construct a VLM"
        calls.append("constructed")
        return OllamaClient(transport=lambda payload: {"message": {"content": reply}})

    monkeypatch.setattr(main, "OllamaClient", make_client)
    main.get_pipeline.cache_clear()
    try:
        pipeline = main.get_pipeline(category)
        assert pipeline.decision_policy == expected_policy
        assert (pipeline.adjudicator is None) == (expected_policy == "detector-only")
        contents = io.BytesIO()
        Image.new("RGB", (16, 16)).save(contents, format="PNG")
        response = client.post(
            "/inspect", data={"category": category},
            files={"file": ("sample.png", contents.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["mode"] == mode
        assert result["decision_policy"] == expected_policy
        assert result["verdict"] == ("false_alarm" if expected_policy == "override" else "defect")
        assert len(calls) == (0 if expected_policy == "detector-only" else 1)
    finally:
        main.get_pipeline.cache_clear()
