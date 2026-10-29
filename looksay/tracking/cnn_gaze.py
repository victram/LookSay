"""Deep appearance features from the GazeFollower gaze CNN (MGazeNet, 7M).

The network was trained on a large webcam gaze dataset, so it already knows
what an eye looking at a given spot looks like, across people, lighting and
head poses. It sees a 224x224 face crop, two 112x112 eye crops and their
positions, and returns 258 values: a rough on-screen gaze estimate (first 2)
and 256 learned features. The calibration model maps these to the screen
for this user, exactly as GazeFollower does with its ridge calibration.

Crops are made the same way as GazeFollower's MediaPipe alignment (frame
resized to 640x480, eye boxes from the eye corners), so the network gets
input like it saw in training. Model: github.com/GanchengZhu/GazeFollower,
CC BY-NC-SA 4.0 (see NOTICE.md).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from looksay.paths import resources_dir

log = logging.getLogger(__name__)

CNN_FEATURES = 258
INPUT_SIZE = (640, 480)
FACE_SIZE = 224
EYE_SIZE = 112


def model_path():
    return resources_dir() / "models" / "gaze_cnn.mnn"


def crop_rects(px: np.ndarray, width: int, height: int):
    """Face and eye boxes (x, y, w, h) from 478 landmarks in pixels, or None if out of frame."""
    x0, y0 = max(0.0, px[:, 0].min()), max(0.0, px[:, 1].min())
    x1, y1 = min(float(width), px[:, 0].max()), min(float(height), px[:, 1].max())
    delta = ((x1 - x0) - (y1 - y0)) / 4
    fx0, fy0 = max(0, int(x0 + delta)), max(0, int(y0 - delta))
    fx1, fy1 = min(width - 1, int(x1 - delta)), min(height - 1, int(y1 + delta))
    scale = abs(px[362, 0] - px[133, 0]) / 100
    eyes = []
    for outer, inner in ((33, 133), (362, 263)):  # image-left eye first, as in GazeFollower
        xs = (px[outer, 0] - 20 * scale, px[inner, 0] + 20 * scale)
        eh = abs(xs[1] - xs[0]) * 0.75
        cy = (px[outer, 1] + px[inner, 1]) / 2
        ex0, ey0, ex1, ey1 = int(xs[0]), int(cy - eh * 0.6), int(xs[1]), int(cy + eh * 0.4)
        if ex0 <= 0 or ey0 <= 0 or ex1 >= width or ey1 >= height or ex1 - ex0 < 5 or ey1 - ey0 < 5:
            return None
        eyes.append((ex0, ey0, ex1 - ex0, ey1 - ey0))
    if fx1 - fx0 < 5 or fy1 - fy0 < 5:
        return None
    return (fx0, fy0, fx1 - fx0, fy1 - fy0), eyes[0], eyes[1]


class CnnGaze:
    """Runs the gaze CNN on the CPU (about 25 ms per frame on a cloud test machine). Raises if MNN or the model is missing."""

    def __init__(self, path=None, threads: int = 2):
        import MNN  # optional dependency: without it LookSay uses its geometric + eye-image features

        self._mnn = MNN
        rt = MNN.nn.create_runtime_manager(({"precision": "low", "backend": 0, "numThread": threads},))
        self._module = MNN.nn.load_module_from_file(str(path or model_path()), ["face", "left", "right", "rect"],
                                                    ["output_0"], runtime_manager=rt)
        nhwc = MNN.expr.NHWC
        self._face = MNN.expr.placeholder((1, FACE_SIZE, FACE_SIZE, 3), nhwc)
        self._left = MNN.expr.placeholder((1, EYE_SIZE, EYE_SIZE, 3), nhwc)
        self._right = MNN.expr.placeholder((1, EYE_SIZE, EYE_SIZE, 3), nhwc)
        self._rect = MNN.expr.placeholder((1, 12))
        self.last_ms = 0.0

    def features(self, rgb: np.ndarray, points: np.ndarray) -> np.ndarray | None:
        """258 CNN outputs for this frame. rgb: HxWx3 uint8; points: (478, 3) normalised landmarks."""
        import cv2

        start = time.perf_counter()
        w, h = INPUT_SIZE
        frame = cv2.resize(rgb, INPUT_SIZE, interpolation=cv2.INTER_AREA) if rgb.shape[1::-1] != INPUT_SIZE else rgb
        px = np.asarray(points[:, :2], dtype=np.float64) * (w, h)
        rects = crop_rects(px, w, h)
        if rects is None:
            return None
        face, left, right = rects

        def crop(r, size):
            x, y, rw, rh = r
            return cv2.resize(frame[y:y + rh, x:x + rw], (size, size)).astype(np.float32) / 255.0

        rect = np.array([*face[2:], *face[:2], *left[2:], *left[:2], *right[2:], *right[:2]], dtype=np.float32)
        rect = (rect / np.array([w, h] * 6, dtype=np.float32)).reshape(1, 12)
        try:
            self._face.write(crop(face, FACE_SIZE)[None])
            self._left.write(crop(left, EYE_SIZE)[None])
            self._right.write(np.ascontiguousarray(cv2.flip(crop(right, EYE_SIZE), 1))[None])
            self._rect.write(rect)
            out = self._module.onForward([self._face, self._left, self._right, self._rect])
            values = np.array(out[0].read(), dtype=np.float64).reshape(-1)
        except Exception:
            log.exception("Gaze CNN inference failed")
            return None
        self.last_ms = (time.perf_counter() - start) * 1000
        return values if values.shape[0] == CNN_FEATURES and np.all(np.isfinite(values)) else None


def load_cnn() -> CnnGaze | None:
    try:
        return CnnGaze()
    except Exception as exc:  # missing MNN wheel or model file: fall back silently
        log.warning("Gaze CNN unavailable: %s", exc)
        return None
