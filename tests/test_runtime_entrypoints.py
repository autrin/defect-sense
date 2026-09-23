"""Policy wiring tests: real pipelines, fake detector/VLM, no saved evaluation output."""

import json
import sys

import pytest
from PIL import Image

from defect_sense.pipeline.two_stage import Detection
from defect_sense.vlm.client import OllamaClient


@pytest.mark.parametrize("policy", [None, "advisory", "override"])
def test_inspect_image_policy(monkeypatch, tmp_path, capsys, policy):
    from scripts import inspect_image

    image = tmp_path / "sample.png"
    Image.new("RGB", (16, 16)).save(image)
    monkeypatch.setattr(
        inspect_image.AnomalibDetector, "predict", lambda self, image: Detection(0.9)
    )
    calls = []

    def make_client(**kwargs):
        assert policy is not None, "Default detector-only must not construct a VLM"
        calls.append("constructed")
        return OllamaClient(transport=lambda payload: {"message": {"content": json.dumps({
            "is_defect": False, "defect_type": "none", "confidence": 0.9, "report": "fine",
        })}})

    monkeypatch.setattr(inspect_image, "OllamaClient", make_client)
    # A detector-only checkpoint need not have a VLM taxonomy.
    category = "custom_product" if policy is None else "bottle"
    argv = ["inspect_image.py", str(image), "--category", category, "--ckpt", "test.ckpt"]
    if policy is not None:
        argv += ["--decision-policy", policy]
    monkeypatch.setattr(sys, "argv", argv)
    inspect_image.main()
    result = json.JSONDecoder().raw_decode(capsys.readouterr().out)[0]
    assert result["decision_policy"] == (policy or "detector-only")
    assert result["verdict"] == ("false_alarm" if policy == "override" else "defect")
    assert result["triaged_to_vlm"] == (policy is not None)
    assert len(calls) == (0 if policy is None else 1)
    assert image.with_stem("sample_inspected").exists()


@pytest.mark.parametrize("mode,policy,expected", [
    ("two-stage", None, "detector-only"),
    ("two-stage", "advisory", "advisory"),
    ("two-stage", "override", "override"),
    ("vlm-only", None, "override"),
    ("vlm-only", "override", "override"),
])
def test_eval_adjudication_policy(monkeypatch, tmp_path, mode, policy, expected):
    from scripts import eval_adjudication
    from defect_sense.detectors import AnomalibDetector

    monkeypatch.setattr(AnomalibDetector, "predict", lambda self, image: Detection(0.9))
    captured = {}

    def make_client(**kwargs):
        assert expected != "detector-only", "Detector-only must not construct a VLM"
        return OllamaClient(transport=lambda payload: {"message": {"content": json.dumps({
            "is_defect": False, "defect_type": "none", "confidence": 0.9, "report": "fine",
        })}})

    def evaluate(pipeline, *args, **kwargs):
        captured["result"] = pipeline.inspect(Image.new("RGB", (16, 16)))
        assert (pipeline.adjudicator is None) == (expected == "detector-only")
        return []

    def write_outputs(records, summary, out_dir, *, run_metadata):
        captured["out_dir"] = out_dir
        captured["metadata"] = run_metadata
        return out_dir / "records.csv", out_dir / "summary.json"

    monkeypatch.setattr(eval_adjudication, "OllamaClient", make_client)
    monkeypatch.setattr(eval_adjudication, "evaluate_category", evaluate)
    monkeypatch.setattr(eval_adjudication, "collect_run_metadata", lambda **kwargs: kwargs)
    monkeypatch.setattr(eval_adjudication, "write_outputs", write_outputs)
    argv = ["eval_adjudication.py", "bottle", "--mode", mode, "--out-dir", str(tmp_path)]
    if mode == "two-stage":
        argv += ["--ckpt", "test.ckpt"]
    if policy is not None:
        argv += ["--decision-policy", policy]
    monkeypatch.setattr(sys, "argv", argv)
    eval_adjudication.main()
    assert captured["result"].decision_policy == expected
    assert captured["result"].verdict == ("false_alarm" if expected == "override" else "defect")
    assert captured["metadata"]["decision_policy"] == expected
    assert captured["out_dir"] == tmp_path / mode / expected
