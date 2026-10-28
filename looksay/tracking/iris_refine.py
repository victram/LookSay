"""Optional image-based refinement of the iris centre.

MediaPipe's iris landmarks come from a 64x64 eye crop and jitter by a pixel
or two. Inside the predicted iris circle (masked by the eye contour, so lids
and lashes are excluded) the iris/pupil is the darkest region; a
darkness-weighted centroid gives a sub-pixel estimate that we blend with
the landmark position. It can be thrown off by strong glasses reflections,
so it is a user setting (compare both in the K debug view).
"""

from __future__ import annotations

import numpy as np

from looksay.tracking.features import EYES

BLEND = 0.5  # weight of the image-based estimate
MAX_SHIFT = 0.6  # never move the centre by more than this fraction of the iris radius


def refine_iris(gray: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """Return a copy of `points` (normalised) with iris landmarks shifted, and per-eye shifts in pixels."""
    import cv2

    h, w = gray.shape[:2]
    out = points.copy()
    shifts: dict[str, tuple[float, float]] = {}
    px = points[:, :2] * np.array([w, h], dtype=np.float32)
    for name, spec in EYES.items():
        iris_idx = list(spec["iris"])
        ring = px[iris_idx]
        center = ring.mean(axis=0)
        radius = float(np.linalg.norm(ring[1:] - center, axis=1).mean())
        if radius < 2.0:
            continue  # too few pixels to refine meaningfully
        r = int(np.ceil(radius * 1.3))
        x0, y0 = int(center[0]) - r, int(center[1]) - r
        x1, y1 = x0 + 2 * r + 1, y0 + 2 * r + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue
        roi = gray[y0:y1, x0:x1].astype(np.float32)
        mask = np.zeros(roi.shape, np.uint8)
        contour = (px[list(spec["contour"])] - [x0, y0]).astype(np.int32)
        cv2.fillPoly(mask, [contour], 255)
        yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
        inside = (mask > 0) & ((xx - (center[0] - x0)) ** 2 + (yy - (center[1] - y0)) ** 2 <= (radius * 1.15) ** 2)
        if inside.sum() < 12:
            continue
        values = roi[inside]
        # Halfway between the darkest pixel and the bright surround: the pupil/iris core.
        threshold = values.min() + 0.5 * (np.percentile(values, 80) - values.min())
        weights = np.clip(threshold - roi, 0, None) * inside
        total = weights.sum()
        if total <= 0:
            continue
        cx = float((weights * xx).sum() / total) + x0
        cy = float((weights * yy).sum() / total) + y0
        shift = (np.array([cx, cy]) - center) * BLEND
        norm = float(np.linalg.norm(shift))
        if norm > MAX_SHIFT * radius:
            shift *= MAX_SHIFT * radius / norm
        out[iris_idx, 0] += shift[0] / w
        out[iris_idx, 1] += shift[1] / h
        shifts[name] = (float(shift[0]), float(shift[1]))
    return out, shifts
