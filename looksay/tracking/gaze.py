"""Combine features, calibration, smoothing and drift correction into a gaze point."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from looksay.calibration.model import GazeModel
from looksay.tracking.features import FaceFeatures
from looksay.tracking.smoothing import OneEuroFilter

MAX_DRIFT = 0.08  # normalised units; drift correction never moves gaze further than this
AUTO_DRIFT_GAIN = 0.15


@dataclass
class GazeSample:
    x: float  # normalised 0..1 in the calibrated window
    y: float
    confidence: float
    state: str  # "ok" | "low_confidence" | "no_face" | "blink" | "uncalibrated"
    timestamp: float

    @property
    def usable(self) -> bool:
        return self.state == "ok"


class GazeEstimator:
    def __init__(self):
        self.model: GazeModel | None = None
        self.eye = "both"
        self.sensitivity = 1.0
        self.confidence_threshold = 0.5
        self.offset = np.zeros(2)
        self.filter = OneEuroFilter.from_strength(5)
        self.last: GazeSample | None = None
        self._last_raw: np.ndarray | None = None

    def set_smoothing(self, strength: int) -> None:
        self.filter = OneEuroFilter.from_strength(strength)

    def set_model(self, model: GazeModel | None) -> None:
        self.model = model
        self.offset = np.zeros(2)
        self.filter.reset()

    def raw_point(self, features: FaceFeatures) -> np.ndarray | None:
        """Calibrated point before sensitivity, drift and smoothing."""
        if self.model is None:
            return None
        vec = features.vector(self.eye)
        if vec is None:
            return None
        return self.model.predict(vec)[0]

    def update(self, features: FaceFeatures) -> GazeSample:
        t = features.timestamp
        held = self.last
        if not features.face_found:
            self.filter.reset()
            sample = GazeSample(held.x if held else 0.5, held.y if held else 0.5, 0.0, "no_face", t)
        elif features.eyes_closed():
            # Keep the last position so a blink does not move the cursor.
            sample = GazeSample(held.x if held else 0.5, held.y if held else 0.5, features.confidence, "blink", t)
        elif self.model is None:
            sample = GazeSample(0.5, 0.5, features.confidence, "uncalibrated", t)
        else:
            raw = self.raw_point(features)
            if raw is None:
                sample = GazeSample(0.5, 0.5, 0.0, "no_face", t)
            else:
                self._last_raw = raw
                p = 0.5 + (raw - 0.5) * self.sensitivity + self.offset
                p = self.filter(np.clip(p, -0.05, 1.05), t)
                state = "ok" if features.confidence >= self.confidence_threshold else "low_confidence"
                sample = GazeSample(float(p[0]), float(p[1]), features.confidence, state, t)
        self.last = sample
        return sample

    # -------------------------------------------------------------- drift
    def recenter(self, target: tuple[float, float], observed: np.ndarray) -> None:
        """Explicit drift correction: the user looked at `target`, we measured `observed` (raw)."""
        p = 0.5 + (np.asarray(observed) - 0.5) * self.sensitivity
        self.offset = np.clip(np.asarray(target) - p, -MAX_DRIFT, MAX_DRIFT)
        self.filter.reset()

    def nudge_towards(self, target: tuple[float, float]) -> None:
        """Implicit drift correction after a dwell selection on a known target centre."""
        if self.last is None or not self.last.usable:
            return
        err = np.asarray(target) - np.array([self.last.x, self.last.y])
        self.offset = np.clip(self.offset + AUTO_DRIFT_GAIN * err, -MAX_DRIFT, MAX_DRIFT)
