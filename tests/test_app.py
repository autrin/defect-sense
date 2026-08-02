from fastapi.testclient import TestClient

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
