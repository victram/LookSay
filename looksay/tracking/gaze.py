"""Combine features, calibration, smoothing and drift correction into a pointer position.

Three pointer modes:

* ``eyes``: calibrated gaze, One Euro filter, fixation stabiliser.
* ``head``: nose/head pointer (needs the head calibration). Precise and
  steady, for users who can move their head a little.
* ``eyes_head`` (default): eyes for large jumps, nose for fine-tuning.
  When the gaze lands somewhere new, the cursor jumps there; while the gaze
  stays in that area, eye jitter is ignored and small head movements move
  the cursor slowly and precisely onto the exact key. This "gaze + head
  pinpointing" combines the speed of the eyes with the precision of the head.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from looksay.calibration.head_model import HeadModel, default_head_model
from looksay.calibration.model import GazeModel
from looksay.tracking.features import FaceFeatures
from looksay.tracking.smoothing import OneEuroFilter

MAX_DRIFT = 0.08  # normalised units; drift correction never moves gaze further than this
AUTO_DRIFT_GAIN = 0.15
# Fixation stabiliser (dispersion-based, I-DT style): while the recent points
# stay within FIXATION_RADIUS, output their mean so the cursor holds still.
FIXATION_WINDOW_S = 0.3
FIXATION_RADIUS = 0.03
# Eyes+nose: the cursor jumps to the gaze when the gaze is this far from it
# for JUMP_HOLD_S (larger than webcam gaze noise, smaller than a real move).
JUMP_DISTANCE = 0.10
JUMP_HOLD_S = 0.12
FINE_GAIN = 0.5  # head fine-tuning speed relative to the full head-pointer mapping


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
        self.stabilize = True
        self.fixation_locked = False
        self.last: GazeSample | None = None
        self.last_raw: np.ndarray | None = None  # calibrated point before sensitivity/drift/filters
        self._window: deque[tuple[float, np.ndarray]] = deque()
        self.mode = "eyes_head"
        self.head_model: HeadModel | None = None  # calibrated nose pointer, if any
        self.head_gain = 1.0
        self.head_filter = OneEuroFilter(min_cutoff=1.5, beta=0.3)
        self._anchor: np.ndarray | None = None  # eyes+nose: gaze point of the current fixation
        self._head_ref: np.ndarray | None = None
        self._far_since: float | None = None
        self.last_head: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        """True if the current mode has the calibration it needs."""
        return self.head_model is not None if self.mode == "head" else self.model is not None

    def set_head_model(self, model: HeadModel | None) -> None:
        self.head_model = model
        self._reset_filters()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._reset_filters()

    def set_smoothing(self, strength: int) -> None:
        self.filter = OneEuroFilter.from_strength(strength)

    def set_model(self, model: GazeModel | None) -> None:
        self.model = model
        self.offset = np.zeros(2)
        self._reset_filters()

    def _reset_filters(self) -> None:
        self.filter.reset()
        self.head_filter.reset()
        self._window.clear()
        self.fixation_locked = False
        self._anchor = self._head_ref = self._far_since = None

    def _stabilise(self, p: np.ndarray, t: float) -> np.ndarray:
        self._window.append((t, p))
        while self._window and t - self._window[0][0] > FIXATION_WINDOW_S:
            self._window.popleft()
        pts = np.array([q for _, q in self._window])
        centre = pts.mean(axis=0)
        span = t - self._window[0][0]
        self.fixation_locked = (span >= FIXATION_WINDOW_S * 0.6
                                and float(np.max(np.linalg.norm(pts - centre, axis=1))) <= FIXATION_RADIUS)
        if self.fixation_locked:
            return centre
        if np.linalg.norm(p - centre) > FIXATION_RADIUS * 2:
            # A saccade: restart the window so the cursor follows immediately.
            self._window.clear()
            self._window.append((t, p))
        return p

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
        hx, hy = (held.x, held.y) if held else (0.5, 0.5)
        head_mode = self.mode == "head"
        if not features.face_found:
            self._reset_filters()
            sample = GazeSample(hx, hy, 0.0, "no_face", t)
        elif features.eyes_closed() and not head_mode:
            # Keep the last position so a blink does not move the cursor.
            sample = GazeSample(hx, hy, features.confidence, "blink", t)
        elif not self.ready:
            sample = GazeSample(0.5, 0.5, features.confidence, "uncalibrated", t)
        else:
            head = features.head_vector()
            self.last_head = self.head_filter(head, t) if head is not None else None
            if head_mode:
                p, confidence = self._head_point(), self._head_confidence(features)
            else:
                raw = self.raw_point(features)
                self.last_raw = raw
                p = self.filter(np.clip(0.5 + (raw - 0.5) * self.sensitivity + self.offset, -0.05, 1.05), t)
                if self.mode == "eyes_head" and self.last_head is not None:
                    p = self._pinpoint(p, t)
                confidence = features.confidence
            if self.stabilize and self.mode != "eyes_head":
                p = self._stabilise(p, t)
            p = np.clip(p, -0.02, 1.02)
            state = "ok" if confidence >= self.confidence_threshold else "low_confidence"
            sample = GazeSample(float(p[0]), float(p[1]), confidence, state, t)
        self.last = sample
        return sample

    def _head_point(self) -> np.ndarray:
        p = self.head_model.predict(self.last_head)[0]
        return 0.5 + (p - 0.5) * self.head_gain + self.offset

    @staticmethod
    def _head_confidence(f: FaceFeatures) -> float:
        """The nose pointer only needs a visible, not too rotated face."""
        size = float(np.clip((f.face_size - 0.06) / 0.08, 0, 1))
        return size * float(np.clip(1 - (abs(f.yaw) - 35) / 15, 0, 1))

    def _pinpoint(self, gaze: np.ndarray, t: float) -> np.ndarray:
        """Eyes+nose fusion: jump with the eyes, fine-tune with the head."""
        hm = self.head_model or default_head_model()
        if self._anchor is None:
            self._anchor, self._head_ref = gaze.copy(), self.last_head.copy()
        delta = (hm.predict(self.last_head)[0] - hm.predict(self._head_ref)[0]) * FINE_GAIN * self.head_gain
        cursor = self._anchor + delta
        if np.linalg.norm(gaze - cursor) > JUMP_DISTANCE:
            self._far_since = self._far_since if self._far_since is not None else t
            if t - self._far_since >= JUMP_HOLD_S:
                self._anchor, self._head_ref, self._far_since = gaze.copy(), self.last_head.copy(), None
                cursor = self._anchor
        else:
            self._far_since = None
        self.fixation_locked = self._far_since is None
        return cursor

    # -------------------------------------------------------------- drift
    def recenter(self, target: tuple[float, float], observed: np.ndarray) -> None:
        """Explicit drift correction: the user looked at `target`, we measured `observed` (raw)."""
        p = 0.5 + (np.asarray(observed) - 0.5) * self.sensitivity
        self.offset = np.clip(np.asarray(target) - p, -MAX_DRIFT, MAX_DRIFT)
        self._reset_filters()

    def nudge_towards(self, target: tuple[float, float]) -> None:
        """Implicit drift correction after a dwell selection on a known target centre."""
        if self.last is None or not self.last.usable:
            return
        err = np.asarray(target) - np.array([self.last.x, self.last.y])
        self.offset = np.clip(self.offset + AUTO_DRIFT_GAIN * err, -MAX_DRIFT, MAX_DRIFT)
