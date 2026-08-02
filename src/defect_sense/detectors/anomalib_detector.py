"""Stage 1 detector backed by anomalib (PatchCore or EfficientAD).

anomalib and torch are imported lazily so the rest of the package (VLM
client, pipeline logic, eval math) stays usable and testable without them.

Targets the anomalib 2.x API.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..pipeline.two_stage import Detection

MODEL_NAMES = ("patchcore", "efficient_ad")
EFFICIENT_AD_TRAINING_STEPS = 70_000


def trainer_kwargs(
    name: str,
    *,
    max_epochs: int | None = None,
    max_steps: int | None = None,
) -> dict[str, int]:
    """Return a finite, model-appropriate Anomalib training schedule."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")
    if max_epochs is not None and max_steps is not None:
        raise ValueError("Set only one of max_epochs or max_steps")
    if max_epochs is not None:
        return {"max_epochs": max_epochs}
    if max_steps is not None:
        return {"max_steps": max_steps}
    if name == "efficient_ad":
        return {"max_steps": EFFICIENT_AD_TRAINING_STEPS}
    return {"max_epochs": 1}


def _model_class(name: str):
    from anomalib.models import EfficientAd, Patchcore

    classes = {"patchcore": Patchcore, "efficient_ad": EfficientAd}
    if name not in classes:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")
    return classes[name]


def _build_model(name: str):
    cls = _model_class(name)
    if name == "patchcore":
        return cls(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
        )
    return cls()


def _to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


@dataclass
class AnomalibDetector:
    """Wraps a trained anomalib model behind the pipeline's Detector protocol."""

    model_name: str = "patchcore"
    ckpt_path: str | Path | None = None
    _model: object | None = None
    _engine: object | None = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from anomalib.engine import Engine

        if self.ckpt_path is None:
            raise FileNotFoundError(
                "No checkpoint given. Train one first, e.g. "
                "`python benchmark.py --model patchcore bottle` — checkpoints land "
                "under results/<Model>/MVTecAD/<category>/<version>/weights/lightning/model.ckpt"
            )
        self._model = _model_class(self.model_name).load_from_checkpoint(
            str(self.ckpt_path)
        )
        self._engine = Engine(accelerator="auto", devices=1)

    def fit(self, category: str, data_root: str | Path = "./datasets/MVTecAD") -> Path:
        """Train on a category's good images; returns the checkpoint path."""
        from anomalib.data import MVTecAD
        from anomalib.engine import Engine

        datamodule = MVTecAD(
            root=str(data_root),
            category=category,
            train_batch_size=1 if self.model_name == "efficient_ad" else 32,
            eval_batch_size=32,
            num_workers=4,
        )
        model = _build_model(self.model_name)
        engine = Engine(
            **trainer_kwargs(self.model_name),
            accelerator="auto",
            devices=1,
        )
        engine.fit(model=model, datamodule=datamodule)

        self._model, self._engine = model, engine
        ckpt = getattr(engine, "best_model_path", None) or getattr(
            engine.trainer.checkpoint_callback, "best_model_path", ""
        )
        if ckpt:
            self.ckpt_path = Path(ckpt)
        return self.ckpt_path

    def predict(self, image: Image.Image) -> Detection:
        """Score a single image; returns image-level score + anomaly map."""
        import tempfile

        from anomalib.data import PredictDataset

        self._ensure_loaded()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.png"
            image.convert("RGB").save(path)
            dataset = PredictDataset(path=path)
            predictions = self._engine.predict(model=self._model, dataset=dataset)

        if not predictions:
            raise RuntimeError("anomalib returned no predictions")
        batch = predictions[0]

        score = _to_numpy(getattr(batch, "pred_score", None))
        amap = _to_numpy(getattr(batch, "anomaly_map", None))
        if amap is not None:
            amap = np.squeeze(amap)
            if amap.ndim != 2:
                amap = amap[0] if amap.ndim == 3 else None
        return Detection(
            score=float(np.squeeze(score)) if score is not None else 0.0,
            anomaly_map=amap,
        )
