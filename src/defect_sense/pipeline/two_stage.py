"""The two-stage inspection pipeline.

Stage 1 (always runs): the anomaly detector scores the image. Images below
the triage threshold pass through untouched — no VLM call, no cost.
Region proposals come from the detector heatmap. Optional Stage 2 supplies
advisory defect typing and a report. Detector-only is the default; VLM calls
and overrides require explicit opt-in.
"""

import time
import math
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
    def predict(self, image: Image.Image, /) -> Detection: ...


@dataclass
class InspectionResult:
    verdict: str  # "pass" | "defect" | "false_alarm"
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
    decision_policy: str = "detector-only"
    vlm_disagrees: bool = False
    vlm_error: str | None = None

    def to_dict(self) -> dict:
        valid = self.adjudication is not None and self.adjudication.parse_ok
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
            "decision_policy": self.decision_policy,
            "vlm_disagrees": self.vlm_disagrees,
            "vlm_is_defect": self.adjudication.is_defect if valid else None,
            "vlm_confidence": self.adjudication.confidence if valid else None,
            "vlm_report": self.adjudication.report if self.adjudication else None,
            "vlm_parse_ok": self.adjudication.parse_ok if self.adjudication else None,
            "vlm_error": self.vlm_error,
        }


@dataclass
class TwoStagePipeline:
    detector: Detector
    adjudicator: VLMAdjudicator | None = None
    threshold: float = 0.5
    rel_region_threshold: float = 0.5
    max_regions: int = 5
    decision_policy: str = "detector-only"  # detector-only | advisory | override

    def __post_init__(self):
        if self.decision_policy not in ("detector-only", "advisory", "override"):
            raise ValueError("decision_policy must be 'detector-only', 'advisory' or 'override'")
        if self.decision_policy != "detector-only" and self.adjudicator is None:
            raise ValueError("An adjudicator is required for VLM policies")
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")

    def inspect(self, image: Image.Image) -> InspectionResult:
        t0 = time.perf_counter()
        detection = self.detector.predict(image)
        stage1 = time.perf_counter() - t0
        if not math.isfinite(detection.score):
            raise ValueError("Detector score must be finite")

        if detection.score < self.threshold:
            return InspectionResult(
                verdict="pass",
                anomaly_score=detection.score,
                triaged_to_vlm=False,
                report="No anomaly detected; passed stage-1 triage.",
                detection=detection,
                stage1_seconds=stage1,
                decision_policy=self.decision_policy,
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

        if self.decision_policy == "detector-only":
            return InspectionResult(
                verdict="defect",
                anomaly_score=detection.score,
                triaged_to_vlm=False,
                bbox=regions[0].bbox if regions else None,
                report="Anomaly detector flagged a defect; inspect the highlighted regions.",
                regions=regions,
                detection=detection,
                stage1_seconds=stage1,
                decision_policy=self.decision_policy,
            )

        t0 = time.perf_counter()
        try:
            adjudication = self.adjudicator.adjudicate(
                image,
                anomaly_score=detection.score,
                regions=regions,
                anomaly_map=detection.anomaly_map,
            )
        except ConnectionError:
            if self.decision_policy != "advisory":
                raise
            return InspectionResult(
                verdict="defect",
                anomaly_score=detection.score,
                triaged_to_vlm=True,
                bbox=regions[0].bbox if regions else None,
                report="Detector flagged a defect. VLM unavailable; detector verdict retained.",
                regions=regions,
                detection=detection,
                stage1_seconds=stage1,
                stage2_seconds=time.perf_counter() - t0,
                decision_policy=self.decision_policy,
                vlm_error="unavailable",
            )
        stage2 = time.perf_counter() - t0

        advisory = self.decision_policy == "advisory"
        disagrees = adjudication.parse_ok and not adjudication.is_defect
        report = adjudication.report
        if advisory:
            if not adjudication.parse_ok:
                report = "Detector flagged a defect. VLM response was invalid; detector verdict retained."
            elif disagrees:
                report = (
                    "Detector flagged a defect. VLM disagrees; detector verdict retained. "
                    f"VLM opinion: {adjudication.report}"
                )
            else:
                report = f"Detector flagged a defect. VLM opinion: {adjudication.report}"

        return InspectionResult(
            verdict="defect" if advisory or adjudication.is_defect else "false_alarm",
            anomaly_score=detection.score,
            triaged_to_vlm=True,
            defect_type=(
                adjudication.defect_type
                if adjudication.parse_ok and adjudication.is_defect else None
            ),
            bbox=(regions[0].bbox if regions else None) if advisory else adjudication.bbox,
            report=report,
            # Self-reported VLM confidence is not confidence in the detector verdict.
            confidence=None if advisory or not adjudication.parse_ok else adjudication.confidence,
            regions=regions,
            detection=detection,
            adjudication=adjudication,
            stage1_seconds=stage1,
            stage2_seconds=stage2,
            decision_policy=self.decision_policy,
            vlm_disagrees=disagrees,
        )
