"""Turn MediaPipe face-mesh landmarks into compact gaze features.

Per eye we measure where the iris sits between the eye corners (horizontal)
and between the eyelids (vertical), normalised by eye width so the values
do not depend on distance to the camera. Head yaw/pitch are estimated from
the nose position relative to the face outline, so the calibration model
can partially compensate small head movements.

Landmark indices refer to the 478-point MediaPipe face mesh with irises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Subject's right eye (appears on the left of a non-mirrored image).
RIGHT_EYE = {"outer": 33, "inner": 133, "upper": 159, "lower": 145, "iris": 468}
LEFT_EYE = {"outer": 263, "inner": 362, "upper": 386, "lower": 374, "iris": 473}
IRIS_RING = {"right": (469, 470, 471, 472), "left": (474, 475, 476, 477)}
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
CHEEK_RIGHT = 234
CHEEK_LEFT = 454
NUM_LANDMARKS_WITH_IRIS = 478

# Eye-openness ratio below which an eye is treated as closed.
EYE_CLOSED_RATIO = 0.13


@dataclass
class EyeFeatures:
    h: float  # iris position across the eye, ~0 (outer) .. ~1 (inner)
    v: float  # iris offset from the eye axis, normalised by eye width
    openness: float  # lid distance / eye width


@dataclass
class FaceFeatures:
    """Everything the UI thread needs from one processed frame."""

    timestamp: float
    face_found: bool
    right: EyeFeatures | None = None
    left: EyeFeatures | None = None
    yaw: float = 0.0
    pitch: float = 0.0
    face_size: float = 0.0  # face width / frame width
    confidence: float = 0.0
    inference_ms: float = 0.0
    landmarks_2d: np.ndarray | None = field(default=None, repr=False)  # (N, 2) normalised, for overlays

    def vector(self, eye: str = "both") -> np.ndarray | None:
        """Feature vector fed to the calibration model."""
        if not self.face_found or self.right is None or self.left is None:
            return None
        if eye == "left":
            h, v = self.left.h, self.left.v
        elif eye == "right":
            h, v = self.right.h, self.right.v
        else:
            # Mirror the left eye so both 'h' axes point the same screen direction.
            h = (self.right.h + (1.0 - self.left.h)) / 2.0
            v = (self.right.v + self.left.v) / 2.0
        return np.array([h, v, self.yaw, self.pitch], dtype=np.float64)

    def eyes_closed(self) -> bool:
        if not self.face_found or self.right is None or self.left is None:
            return False
        return self.right.openness < EYE_CLOSED_RATIO and self.left.openness < EYE_CLOSED_RATIO


def _eye(points: np.ndarray, idx: dict[str, int], outer_to_inner_sign: float) -> EyeFeatures:
    outer, inner = points[idx["outer"]], points[idx["inner"]]
    upper, lower = points[idx["upper"]], points[idx["lower"]]
    iris = points[idx["iris"]]
    axis = inner - outer
    width = float(np.linalg.norm(axis)) or 1e-6
    unit = axis / width
    normal = np.array([-unit[1], unit[0]]) * outer_to_inner_sign
    rel = iris - outer
    h = float(np.dot(rel, unit) / width)
    v = float(np.dot(rel, normal) / width)
    openness = float(np.linalg.norm(lower - upper) / width)
    return EyeFeatures(h=h, v=v, openness=openness)


def compute_features(points: np.ndarray, frame_w: int, frame_h: int, timestamp: float,
                     inference_ms: float = 0.0) -> FaceFeatures:
    """points: (N, 2 or 3) landmarks in normalised image coordinates."""
    if points is None or len(points) < NUM_LANDMARKS_WITH_IRIS:
        return FaceFeatures(timestamp=timestamp, face_found=False, inference_ms=inference_ms)
    pts = np.asarray(points, dtype=np.float64)[:, :2] * np.array([frame_w, frame_h], dtype=np.float64)
    # The normal must point the same way (down the face) for both eyes; the
    # eyes' outer->inner axes point in opposite directions, hence the signs.
    right = _eye(pts, RIGHT_EYE, 1.0)
    left = _eye(pts, LEFT_EYE, -1.0)

    cheek_r, cheek_l = pts[CHEEK_RIGHT], pts[CHEEK_LEFT]
    face_w = float(np.linalg.norm(cheek_l - cheek_r)) or 1e-6
    top, chin = pts[FOREHEAD], pts[CHIN]
    face_h = float(np.linalg.norm(chin - top)) or 1e-6
    nose = pts[NOSE_TIP]
    yaw = float((nose[0] - (cheek_r[0] + cheek_l[0]) / 2.0) / face_w)
    pitch = float((nose[1] - (top[1] + chin[1]) / 2.0) / face_h)

    features = FaceFeatures(
        timestamp=timestamp, face_found=True, right=right, left=left, yaw=yaw, pitch=pitch,
        face_size=face_w / frame_w, inference_ms=inference_ms,
        landmarks_2d=np.asarray(points, dtype=np.float32)[:, :2],
    )
    features.confidence = estimate_confidence(features)
    return features


def estimate_confidence(f: FaceFeatures) -> float:
    """Heuristic 0..1 confidence from face size, head pose and eye openness.

    MediaPipe does not expose a per-landmark score here, so we penalise the
    conditions known to degrade webcam gaze estimates.
    """
    if not f.face_found or f.right is None or f.left is None:
        return 0.0
    # Face too small (far away / low resolution) degrades iris precision.
    size = np.clip((f.face_size - 0.08) / 0.12, 0.0, 1.0)
    # Strong head rotation hides one eye and breaks the geometry.
    pose = np.clip(1.0 - (abs(f.yaw) - 0.12) / 0.2, 0.0, 1.0) * np.clip(1.0 - (abs(f.pitch) - 0.15) / 0.2, 0.0, 1.0)
    openness = min(f.right.openness, f.left.openness)
    eyes = np.clip((openness - EYE_CLOSED_RATIO) / 0.12, 0.0, 1.0)
    # Iris should sit inside the eye; outside means a bad detection.
    inside = 1.0 if all(-0.1 < e.h < 1.1 for e in (f.right, f.left)) else 0.3
    return float(size * pose * eyes * inside)
