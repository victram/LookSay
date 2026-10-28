"""Face/iris landmark detection with MediaPipe FaceLandmarker (Tasks API).

The model file (face_landmarker.task, ~3.7 MB, Apache-2.0) is bundled, so
detection runs fully offline on the CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from looksay.paths import resources_dir


class TrackerUnavailable(RuntimeError):
    pass


@dataclass
class Detection:
    points: np.ndarray | None  # (478, 3) normalised landmarks, None if no face
    inference_ms: float
    blendshapes: dict[str, float] = field(default_factory=dict)
    matrix: np.ndarray | None = None  # 4x4 facial transformation (cm, camera space)


def model_path() -> Path:
    return resources_dir() / "models" / "face_landmarker.task"


class LandmarkDetector:
    def __init__(self, path: Path | None = None):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except Exception as exc:  # ImportError or a broken native library
            raise TrackerUnavailable(f"MediaPipe not available: {exc}") from exc
        path = path or model_path()
        if not path.exists():
            raise TrackerUnavailable(f"Landmark model missing: {path}")
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            # Both are nearly free (~0 ms measured) and carry gaze/head-pose signal.
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        try:
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:  # missing native libraries, corrupt model file
            raise TrackerUnavailable(f"Could not start landmark model: {exc}") from exc
        self._last_ts = 0

    def detect(self, frame_rgb: np.ndarray, timestamp_s: float) -> Detection:
        ts = max(self._last_ts + 1, int(timestamp_s * 1000))  # must be strictly increasing
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        start = time.perf_counter()
        result = self._landmarker.detect_for_video(image, ts)
        ms = (time.perf_counter() - start) * 1000.0
        if not result.face_landmarks:
            return Detection(None, ms)
        points = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32)
        blendshapes = {}
        if getattr(result, "face_blendshapes", None):
            blendshapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        matrix = None
        if getattr(result, "facial_transformation_matrixes", None):
            matrix = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)
        return Detection(points, ms, blendshapes, matrix)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
