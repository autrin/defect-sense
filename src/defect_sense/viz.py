"""Visualization helpers: heatmap overlays and region boxes, PIL-only."""
import numpy as np
from PIL import Image, ImageDraw

from .regions import Region

# Simple blue -> cyan -> yellow -> red colormap anchors.
_ANCHORS = np.array([
    [0.0, 0, 0, 255],
    [0.35, 0, 255, 255],
    [0.65, 255, 255, 0],
    [1.0, 255, 0, 0],
])


def _colormap(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB uint8 via piecewise-linear interpolation."""
    rgb = np.stack(
        [np.interp(values, _ANCHORS[:, 0], _ANCHORS[:, c + 1]) for c in range(3)],
        axis=-1,
    )
    return rgb.astype(np.uint8)


def overlay_heatmap(image: Image.Image, anomaly_map: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a normalized anomaly heatmap over the image.

    Blend weight scales with anomaly intensity, so clean areas stay visible.
    """
    amap = np.asarray(anomaly_map, dtype=np.float64)
    lo, hi = amap.min(), amap.max()
    norm = (amap - lo) / (hi - lo) if hi > lo else np.zeros_like(amap)

    heat = Image.fromarray(_colormap(norm)).resize(image.size, Image.BILINEAR)
    weight = Image.fromarray((norm * alpha * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)

    base = image.convert("RGB")
    return Image.composite(heat, base, weight)


def draw_regions(image: Image.Image, regions: list[Region], color: str = "#ff2d2d") -> Image.Image:
    """Draw region boxes (with rank labels) on a copy of the image."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width = max(2, round(min(out.size) / 200))
    for i, region in enumerate(regions, start=1):
        x0, y0, x1, y1 = region.bbox
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=width)
        draw.text((x0 + width + 1, y0 + width), str(i), fill=color)
    return out
