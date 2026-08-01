"""The two-stage inspection pipeline.

Stage 1 (always runs): the anomaly detector scores the image. Images below
the triage threshold pass through untouched — no VLM call, no cost.
Stage 2 (flagged images only): region proposals are extracted from the
heatmap and the VLM adjudicates: real defect or false alarm, defect type,
box, and report.
"""
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image

from ..regions import Region, extract_regions
from ..vlm.adjudicator import Adjudication, VLMAdjudicator


@dataclass(frozen=True)
class Detection:
    """Stage 1 output: image-level score plus per-pixel anomaly map."""
    score: float
    anomaly_map: np.ndarray | None = None


class Detector(Protocol):
    def predict(self, image: Image.Image) -> Detection: ...


@dataclass
class InspectionResult:
    verdict: str                    # "pass" | "defect" | "false_alarm"
    anomaly_score: float
    triaged_to_vlm: bool
    defect_type: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    report: str = ""
    confidence: float | None = None
    regions: list[Region] = field(default_factory=list)
    detection: Detection | None = None
    adjudication: Adjudication | None = None
    stage1_seconds: float = 0.0
    stage2_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "anomaly_score": round(self.anomaly_score, 4),
            "triaged_to_vlm": self.triaged_to_vlm,
            "defect_type": self.defect_type,
            "bbox": list(self.bbox) if self.bbox else None,
            "confidence": self.confidence,
            "report": self.report,
            "regions": [list(r.bbox) for r in self.regions],
            "stage1_seconds": round(self.stage1_seconds, 3),
            "stage2_seconds": round(self.stage2_seconds, 3),
        }


@dataclass
class TwoStagePipeline:
    detector: Detector
    adjudicator: VLMAdjudicator
    threshold: float = 0.5
    rel_region_threshold: float = 0.5
    max_regions: int = 5

    def inspect(self, image: Image.Image) -> InspectionResult:
        t0 = time.perf_counter()
        detection = self.detector.predict(image)
        stage1 = time.perf_counter() - t0

        if detection.score < self.threshold:
            return InspectionResult(
                verdict="pass",
                anomaly_score=detection.score,
                triaged_to_vlm=False,
                report="No anomaly detected; passed stage-1 triage.",
                detection=detection,
                stage1_seconds=stage1,
            )

        regions: list[Region] = []
        if detection.anomaly_map is not None:
            regions = extract_regions(
                detection.anomaly_map,
                rel_threshold=self.rel_region_threshold,
                max_regions=self.max_regions,
            )
            # Region boxes are in heatmap coordinates; scale to image coordinates.
            h, w = detection.anomaly_map.shape
            if (w, h) != image.size:
                regions = [r.scaled(image.width / w, image.height / h) for r in regions]

        t0 = time.perf_counter()
        adjudication = self.adjudicator.adjudicate(
            image,
            anomaly_score=detection.score,
            regions=regions,
            anomaly_map=detection.anomaly_map,
        )
        stage2 = time.perf_counter() - t0

        return InspectionResult(
            verdict="defect" if adjudication.is_defect else "false_alarm",
            anomaly_score=detection.score,
            triaged_to_vlm=True,
            defect_type=adjudication.defect_type if adjudication.is_defect else None,
            bbox=adjudication.bbox,
            report=adjudication.report,
            confidence=adjudication.confidence,
            regions=regions,
            detection=detection,
            adjudication=adjudication,
            stage1_seconds=stage1,
            stage2_seconds=stage2,
        )
