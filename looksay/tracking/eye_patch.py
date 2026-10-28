"""Normalised eye-image patches for appearance-based gaze features.

Landmark-only gaze is limited by how precisely the iris landmarks are
placed. The pixels around the eye carry more: where the dark iris sits and
how the eyelids are shaped (the upper lid follows vertical gaze). As in
WebGazer and appearance-based gaze research, each eye is cut out along its
corner axis (so head roll and distance do not change the patch), resized to
a small fixed grid, and contrast-normalised. The calibration model then
learns from these pixels together with the geometric features.
"""

from __future__ import annotations

import numpy as np

PATCH_W, PATCH_H = 24, 12
PATCH_SIZE = PATCH_W * PATCH_H
SPAN = 1.3  # patch width in eye widths (corner to corner = 1.0)


def eye_patch(gray: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Cut the eye between corners `a` (image-left) and `b` (image-right), pixel coordinates."""
    import cv2

    axis = np.asarray(b, np.float64) - np.asarray(a, np.float64)
    width = float(np.linalg.norm(axis))
    if width < 6:
        return None
    unit = axis / width
    center = (np.asarray(a) + np.asarray(b)) / 2.0
    scale = PATCH_W / (SPAN * width)
    # Rotation + scale mapping the eye axis to the patch's horizontal axis.
    cos, sin = unit
    m = np.array([[cos * scale, sin * scale, 0.0], [-sin * scale, cos * scale, 0.0]])
    m[:, 2] = np.array([PATCH_W / 2, PATCH_H / 2 + PATCH_H * 0.05]) - m[:, :2] @ center
    patch = cv2.warpAffine(gray, m, (PATCH_W, PATCH_H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    patch = patch.astype(np.float32)
    std = float(patch.std())
    if std < 1.0:
        return None  # flat / dark image
    return ((patch - patch.mean()) / std).astype(np.float32)


def eye_patches(gray: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    """(2, PATCH_H, PATCH_W) patches for the right and left eye, or None."""
    from looksay.tracking.features import EYES

    h, w = gray.shape[:2]
    px = points[:, :2] * np.array([w, h], dtype=np.float32)
    out = []
    for name in ("right", "left"):
        spec = EYES[name]
        p = eye_patch(gray, px[spec["a"]], px[spec["b"]])
        if p is None:
            return None
        out.append(p)
    return np.stack(out)
