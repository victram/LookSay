"""Face/iris landmark detection with MediaPipe FaceLandmarker (Tasks API).

The model file (face_landmarker.task, ~3.7 MB, Apache-2.0) is bundled, so
detection runs fully offline on the CPU.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from looksay.paths import resources_dir


class TrackerUnavailable(RuntimeError):
    pass


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
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        try:
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:  # missing native libraries, corrupt model file
            raise TrackerUnavailable(f"Could not start landmark model: {exc}") from exc
        self._last_ts = 0

    def detect(self, frame_rgb: np.ndarray, timestamp_s: float) -> tuple[np.ndarray | None, float]:
        """Returns ((478, 3) normalised landmarks or None, inference milliseconds)."""
        ts = max(self._last_ts + 1, int(timestamp_s * 1000))  # must be strictly increasing
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        start = time.perf_counter()
        result = self._landmarker.detect_for_video(image, ts)
        ms = (time.perf_counter() - start) * 1000.0
        if not result.face_landmarks:
            return None, ms
        lms = result.face_landmarks[0]
        return np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float32), ms

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
