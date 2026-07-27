"""Turn an anomaly heatmap into bounding-box region proposals.

The detector outputs a per-pixel anomaly map. We threshold it relative to
its own dynamic range, find connected components, and return the largest
ones as boxes. These proposals anchor the VLM's attention (and its report)
to concrete image locations.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    """A suspect region in image coordinates (x0, y0, x1, y1 inclusive-exclusive)."""
    bbox: tuple[int, int, int, int]
    area: int
    peak_score: float
    mean_score: float

    def scaled(self, sx: float, sy: float) -> "Region":
        x0, y0, x1, y1 = self.bbox
        return Region(
            bbox=(round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)),
            area=self.area,
            peak_score=self.peak_score,
            mean_score=self.mean_score,
        )


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a boolean mask, as lists of (row, col) arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for r0, c0 in zip(*np.nonzero(mask)):
        if seen[r0, c0]:
            continue
        queue = deque([(r0, c0)])
        seen[r0, c0] = True
        pixels = []
        while queue:
            r, c = queue.popleft()
            pixels.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(np.array(pixels))
    return components

def extract_regions(
    anomaly_map: np.ndarray,
    rel_threshold: float = 0.5,
    min_area_frac: float = 0.0005,
    max_regions: int = 5,
) -> list[Region]:
    """Extract up to `max_regions` boxes from an anomaly map.

    The threshold is relative to the map's own range: pixels above
    min + rel_threshold * (max - min) are candidates. Components smaller
    than `min_area_frac` of the image are dropped as noise. Regions are
    returned sorted by peak score, highest first.
    """
    amap = np.asarray(anomaly_map, dtype=np.float64)
    if amap.ndim != 2:
        raise ValueError(f"Expected a 2D anomaly map, got shape {amap.shape}")
    lo, hi = float(amap.min()), float(amap.max())
    if hi <= lo:  # flat map — nothing stands out
        return []

    mask = amap >= lo + rel_threshold * (hi - lo)
    min_area = max(1, int(min_area_frac * amap.size))

    regions = []
    for pixels in _connected_components(mask):
        if len(pixels) < min_area:
            continue
        rows, cols = pixels[:, 0], pixels[:, 1]
        values = amap[rows, cols]
        regions.append(Region(
            bbox=(int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1),
            area=int(len(pixels)),
            peak_score=float(values.max()),
            mean_score=float(values.mean()),
        ))

    regions.sort(key=lambda r: r.peak_score, reverse=True)
    return regions[:max_regions]
