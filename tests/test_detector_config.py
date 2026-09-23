import pytest

from defect_sense.detectors.anomalib_detector import (
    EFFICIENT_AD_TRAINING_STEPS,
    trainer_kwargs,
)


def test_default_training_schedules_are_model_appropriate():
    assert trainer_kwargs("patchcore") == {"max_epochs": 1}
    assert trainer_kwargs("efficient_ad") == {
        "max_steps": EFFICIENT_AD_TRAINING_STEPS,
    }


def test_training_schedule_can_be_overridden():
    assert trainer_kwargs("efficient_ad", max_epochs=10) == {"max_epochs": 10}
    assert trainer_kwargs("patchcore", max_steps=500) == {"max_steps": 500}


def test_training_schedule_rejects_conflicting_limits():
    with pytest.raises(ValueError, match="only one"):
        trainer_kwargs("efficient_ad", max_epochs=10, max_steps=500)


@pytest.mark.parametrize("score", [None, [], [0.1, 0.9], float("nan"), float("inf"), 0.75])
def test_inference_requires_one_finite_score(monkeypatch, score):
    import sys
    from types import ModuleType, SimpleNamespace

    from PIL import Image

    from defect_sense.detectors.anomalib_detector import AnomalibDetector

    data = ModuleType("anomalib.data")
    data.PredictDataset = lambda path: path
    monkeypatch.setitem(sys.modules, "anomalib.data", data)
    detector = AnomalibDetector(
        _model=object(),
        _engine=SimpleNamespace(predict=lambda **kwargs: [SimpleNamespace(pred_score=score)]),
    )
    if score == 0.75:
        assert detector.predict(Image.new("RGB", (16, 16))).score == 0.75
    else:
        with pytest.raises(ValueError, match="one finite"):
            detector.predict(Image.new("RGB", (16, 16)))


@pytest.mark.parametrize("model,limits,expected", [
    ("patchcore", {}, {"max_epochs": 1}),
    ("efficient_ad", {}, {"max_steps": EFFICIENT_AD_TRAINING_STEPS}),
    ("efficient_ad", {"max_epochs": 10}, {"max_epochs": 10}),
    ("patchcore", {"max_steps": 500}, {"max_steps": 500}),
])
def test_benchmark_passes_only_configured_training_limit(monkeypatch, model, limits, expected):
    import sys
    from types import ModuleType, SimpleNamespace

    import benchmark
    from defect_sense.detectors import anomalib_detector

    captured = {}
    data = ModuleType("anomalib.data")
    engine_module = ModuleType("anomalib.engine")
    torch = ModuleType("torch")
    torch.set_float32_matmul_precision = lambda value: None

    def datamodule(**kwargs):
        captured["data"] = kwargs
        return object()

    def engine(**kwargs):
        captured["engine"] = kwargs
        return SimpleNamespace(
            fit=lambda **kwargs: None,
            test=lambda **kwargs: [{"image_AUROC": 0.9}],
        )

    data.MVTecAD = datamodule
    engine_module.Engine = engine
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "anomalib.data", data)
    monkeypatch.setitem(sys.modules, "anomalib.engine", engine_module)
    monkeypatch.setattr(anomalib_detector, "_build_model", lambda name: object())
    row = benchmark.run_one(model, "bottle", **limits)
    assert captured["engine"] == {**expected, "accelerator": "gpu", "devices": 1}
    assert captured["data"]["val_split_mode"] == row["val_split_mode"] == "from_test"
    assert captured["data"]["val_split_ratio"] == row["val_split_ratio"] == 0.5
    assert captured["data"]["seed"] == row["seed"] == 42
