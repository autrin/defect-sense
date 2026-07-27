"""Stage 2: VLM adjudication of images flagged by the anomaly detector.

Given the original image plus the detector's evidence (anomaly score,
heatmap overlay, region proposals), the VLM decides whether a defect is
really present, names its type from a closed per-category taxonomy, refines
the bounding box, and writes a short human-readable inspection report.
"""
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from ..regions import Region
from ..viz import draw_regions, overlay_heatmap
from .client import OllamaClient

SYSTEM_PROMPT = (
    "You are a meticulous industrial quality-control inspector. You examine "
    "product photos flagged by an automated anomaly detector and give a final "
    "verdict. The detector has high recall but imperfect precision, so some "
    "flagged images are actually fine. Be precise and never invent defects "
    "you cannot see."
)


@dataclass(frozen=True)
class Adjudication:
    is_defect: bool
    defect_type: str  # one of the taxonomy, "none", or "unknown"
    confidence: float
    bbox: tuple[int, int, int, int] | None
    report: str
    raw_text: str
    latency_s: float
    parse_ok: bool = True


def _response_schema(defect_types: list[str]) -> dict[str, Any]:
    """JSON schema for Ollama structured output."""
    return {
        "type": "object",
        "properties": {
            "is_defect": {"type": "boolean"},
            "defect_type": {"type": "string", "enum": [*defect_types, "none"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "bbox": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
            },
            "report": {"type": "string"},
        },
        "required": ["is_defect", "defect_type", "confidence", "report"],
    }


def _extract_json(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model output, tolerating code fences and prose."""
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    for attempt in (candidate, _outermost_braces(candidate)):
        if not attempt:
            continue
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _outermost_braces(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else None


def _clamp_bbox(bbox: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (int(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((max(0, min(x0, width)), max(0, min(x1, width))))
    y0, y1 = sorted((max(0, min(y0, height)), max(0, min(y1, height))))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


@dataclass
class VLMAdjudicator:
    client: OllamaClient
    category: str
    defect_types: list[str]
    max_image_side: int = 768  # downscale large images before sending
    send_overlay: bool = True
    _schema: dict[str, Any] = field(init=False)

    def __post_init__(self):
        self._schema = _response_schema(self.defect_types)

    def _prepare(self, image: Image.Image) -> Image.Image:
        if max(image.size) <= self.max_image_side:
            return image
        scale = self.max_image_side / max(image.size)
        return image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.BILINEAR
        )

    def _build_prompt(self, anomaly_score: float | None, regions: list[Region], size: tuple[int, int]) -> str:
        lines = [
            f"Product category: {self.category}.",
            f"Possible defect types: {', '.join(self.defect_types)}.",
            f"Image size: {size[0]}x{size[1]} pixels.",
        ]
        if anomaly_score is not None:
            lines.append(f"Anomaly detector score: {anomaly_score:.3f} (higher = more anomalous).")
        if regions:
            lines.append("Detector-proposed suspect regions, as [x0, y0, x1, y1] pixel boxes:")
            for i, r in enumerate(regions, start=1):
                lines.append(f"  {i}. {list(r.bbox)} (peak score {r.peak_score:.3f})")
        if self.send_overlay and regions:
            lines.append(
                "Two images are attached: the original photo, then the same photo "
                "with the detector's anomaly heatmap and numbered region boxes overlaid."
            )
        else:
            lines.append("The product photo is attached.")
        lines.append(
            "Inspect the image. Decide whether it truly contains a manufacturing "
            "defect. If yes, pick the single best-matching defect_type from the "
            "list, give a tight [x0, y0, x1, y1] pixel bounding box around the "
            "defect, and write a 1-3 sentence inspection report describing what "
            "you see and where. If the image is actually fine, set is_defect to "
            "false, defect_type to \"none\", bbox to null, and briefly say why "
            "the flagged region is benign. Respond with JSON only."
        )
        return "\n".join(lines)

    def adjudicate(
        self,
        image: Image.Image,
        anomaly_score: float | None = None,
        regions: list[Region] | None = None,
        anomaly_map: np.ndarray | None = None,
    ) -> Adjudication:
        regions = regions or []
        prepared = self._prepare(image)
        sx = prepared.width / image.width
        sy = prepared.height / image.height
        scaled_regions = [r.scaled(sx, sy) for r in regions]

        images = [_png_bytes(prepared)]
        if self.send_overlay and anomaly_map is not None and scaled_regions:
            evidence = overlay_heatmap(prepared, anomaly_map)
            images.append(_png_bytes(draw_regions(evidence, scaled_regions)))

        prompt = self._build_prompt(anomaly_score, scaled_regions, prepared.size)
        response = self.client.chat(
            prompt, images=images, format_schema=self._schema, system=SYSTEM_PROMPT
        )
        return self._parse(response.text, response.latency_s, prepared.size, (1 / sx, 1 / sy))

    def _parse(
        self,
        text: str,
        latency_s: float,
        size: tuple[int, int],
        upscale: tuple[float, float],
    ) -> Adjudication:
        obj = _extract_json(text)
        if obj is None:
            return Adjudication(
                is_defect=True, defect_type="unknown", confidence=0.0, bbox=None,
                report="", raw_text=text, latency_s=latency_s, parse_ok=False,
            )

        defect_type = str(obj.get("defect_type", "")).strip().lower().replace(" ", "_")
        if defect_type not in (*self.defect_types, "none"):
            defect_type = "unknown"

        is_defect = bool(obj.get("is_defect", defect_type not in ("none", "")))
        if defect_type == "none":
            is_defect = False

        try:
            confidence = min(1.0, max(0.0, float(obj.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        bbox = _clamp_bbox(obj.get("bbox"), *size)
        if bbox is not None:  # map back to original-image coordinates
            ux, uy = upscale
            bbox = (round(bbox[0] * ux), round(bbox[1] * uy), round(bbox[2] * ux), round(bbox[3] * uy))

        return Adjudication(
            is_defect=is_defect,
            defect_type=defect_type,
            confidence=confidence,
            bbox=bbox,
            report=str(obj.get("report", "")).strip(),
            raw_text=text,
            latency_s=latency_s,
        )
