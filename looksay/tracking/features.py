"""Turn MediaPipe face-mesh output into gaze features.

Signals used, per frame:

* **Iris position inside each eye.** The iris centre (mean of the 5 iris
  landmarks) is measured relative to the centroid of the 16-point eye
  contour, in an eye-aligned frame (rotated by head roll) and normalised by
  eye width, so it does not depend on distance or head tilt.
* **Eyelid aperture.** The upper lid follows vertical gaze, so openness is
  an important vertical cue (webcam iris landmarks are weakest vertically).
* **Gaze blendshapes.** MediaPipe's learned eyeLook{In,Out,Up,Down}
  coefficients per eye: a second, independent estimate of eye rotation.
* **3D head pose.** Yaw/pitch/roll and translation (cm) from MediaPipe's
  facial transformation matrix, so the calibration can compensate head
  movement instead of mistaking it for eye movement.

Landmark indices refer to the 478-point MediaPipe face mesh with irises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Each eye: image-left corner "a", image-right corner "b" (so +h always means
# the iris moved towards image-right for both eyes), contour, lid pairs, iris.
RIGHT_EYE = {  # subject's right eye (image-left in a non-mirrored frame)
    "a": 33, "b": 133,
    "contour": (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    "lids": ((159, 145), (158, 153), (160, 144)),
    "iris": (468, 469, 470, 471, 472),
}
LEFT_EYE = {
    "a": 362, "b": 263,
    "contour": (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
    "lids": ((386, 374), (385, 380), (387, 373)),
    "iris": (473, 474, 475, 476, 477),
}
EYES = {"right": RIGHT_EYE, "left": LEFT_EYE}
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
CHEEK_RIGHT = 234
CHEEK_LEFT = 454
NUM_LANDMARKS_WITH_IRIS = 478

# Eye-openness ratio below which an eye is treated as closed.
EYE_CLOSED_RATIO = 0.12

FEATURE_NAMES = ("h_r", "v_r", "h_l", "v_l", "open", "look_h", "look_v",
                 "yaw", "pitch", "roll", "tx", "ty", "tz")


@dataclass
class EyeFeatures:
    h: float  # iris offset from eye centre along the eye axis / eye width (+ = image-right)
    v: float  # iris offset perpendicular to the axis / eye width (+ = down)
    openness: float  # mean lid distance / eye width
    iris_center: tuple[float, float] = (0.0, 0.0)  # pixels
    eye_center: tuple[float, float] = (0.0, 0.0)  # pixels
    iris_radius: float = 0.0  # pixels
    width: float = 0.0  # pixels (corner to corner)
    roll: float = 0.0  # radians, eye-axis angle
    look_h: float = 0.0  # blendshape horizontal (+ = image-right), 0 if unavailable
    look_v: float = 0.0  # blendshape vertical (+ = down)


@dataclass
class FaceFeatures:
    """Everything the UI thread needs from one processed frame."""

    timestamp: float
    face_found: bool
    right: EyeFeatures | None = None
    left: EyeFeatures | None = None
    yaw: float = 0.0  # degrees
    pitch: float = 0.0
    roll: float = 0.0
    tx: float = 0.0  # head translation, cm (camera space) when the 3D pose is available
    ty: float = 0.0
    tz: float = 0.0
    face_size: float = 0.0  # face width / frame width
    confidence: float = 0.0
    inference_ms: float = 0.0
    pose_3d: bool = False
    # Head-pointer cue: nose tip relative to the face, in face widths. Turning
    # or tilting the head moves it; shifting the whole head does not.
    nose_x: float = 0.0
    nose_y: float = 0.0
    blendshapes: dict[str, float] = field(default_factory=dict)
    landmarks_2d: np.ndarray | None = field(default=None, repr=False)  # (N, 2) normalised, for overlays
    frame_size: tuple[int, int] = (0, 0)

    def vector(self, eye: str = "both") -> np.ndarray | None:
        """Feature vector fed to the calibration model (order = FEATURE_NAMES)."""
        if not self.face_found or self.right is None or self.left is None:
            return None
        r, l = self.right, self.left
        if eye == "right":
            l = r
        elif eye == "left":
            r = l
        look_h = (r.look_h + l.look_h) / 2.0
        look_v = (r.look_v + l.look_v) / 2.0
        return np.array([r.h, r.v, l.h, l.v, (r.openness + l.openness) / 2.0, look_h, look_v,
                         self.yaw, self.pitch, self.roll, self.tx, self.ty, self.tz], dtype=np.float64)

    def head_vector(self) -> np.ndarray | None:
        """Features for the nose/head pointer: nose offsets plus 3D yaw/pitch when available."""
        if not self.face_found:
            return None
        return np.array([self.nose_x, self.nose_y, self.yaw if self.pose_3d else 0.0,
                         self.pitch if self.pose_3d else 0.0], dtype=np.float64)

    def eyes_closed(self) -> bool:
        if not self.face_found or self.right is None or self.left is None:
            return False
        blink = min(self.blendshapes.get("eyeBlinkLeft", 0.0), self.blendshapes.get("eyeBlinkRight", 0.0))
        geometric = self.right.openness < EYE_CLOSED_RATIO and self.left.openness < EYE_CLOSED_RATIO
        return geometric or blink > 0.6


def _eye(pts: np.ndarray, spec: dict) -> EyeFeatures:
    a, b = pts[spec["a"]], pts[spec["b"]]
    axis = b - a
    width = float(np.linalg.norm(axis)) or 1e-6
    unit = axis / width
    normal = np.array([-unit[1], unit[0]])  # +90°: points down the face for an upright head
    center = pts[list(spec["contour"])].mean(axis=0)
    iris_pts = pts[list(spec["iris"])]
    iris = iris_pts.mean(axis=0)
    radius = float(np.linalg.norm(iris_pts[1:] - iris, axis=1).mean())
    rel = iris - center
    lids = [float(np.linalg.norm(pts[u] - pts[d])) for u, d in spec["lids"]]
    return EyeFeatures(
        h=float(np.dot(rel, unit) / width), v=float(np.dot(rel, normal) / width),
        openness=float(np.mean(lids) / width), iris_center=(float(iris[0]), float(iris[1])),
        eye_center=(float(center[0]), float(center[1])), iris_radius=radius, width=width,
        roll=math.atan2(unit[1], unit[0]),
    )


def pose_from_matrix(matrix: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """(yaw, pitch, roll) in degrees and (tx, ty, tz) in cm from a 4x4 face transform."""
    m = np.asarray(matrix, dtype=np.float64)
    r = m[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll, float(m[0, 3]), float(m[1, 3]), float(m[2, 3])


def compute_features(points: np.ndarray | None, frame_w: int, frame_h: int, timestamp: float,
                     inference_ms: float = 0.0, blendshapes: dict[str, float] | None = None,
                     matrix: np.ndarray | None = None) -> FaceFeatures:
    """points: (N, 2 or 3) landmarks in normalised image coordinates."""
    if points is None or len(points) < NUM_LANDMARKS_WITH_IRIS:
        return FaceFeatures(timestamp=timestamp, face_found=False, inference_ms=inference_ms,
                            frame_size=(frame_w, frame_h))
    pts = np.asarray(points, dtype=np.float64)[:, :2] * np.array([frame_w, frame_h], dtype=np.float64)
    right, left = _eye(pts, RIGHT_EYE), _eye(pts, LEFT_EYE)
    bs = dict(blendshapes or {})
    if bs:
        # Subject's right eye: "In" = towards the nose = image-right.
        right.look_h = bs.get("eyeLookInRight", 0.0) - bs.get("eyeLookOutRight", 0.0)
        left.look_h = bs.get("eyeLookOutLeft", 0.0) - bs.get("eyeLookInLeft", 0.0)
        right.look_v = bs.get("eyeLookDownRight", 0.0) - bs.get("eyeLookUpRight", 0.0)
        left.look_v = bs.get("eyeLookDownLeft", 0.0) - bs.get("eyeLookUpLeft", 0.0)

    cheek_r, cheek_l = pts[CHEEK_RIGHT], pts[CHEEK_LEFT]
    face_w = float(np.linalg.norm(cheek_l - cheek_r)) or 1e-6
    if matrix is not None:
        yaw, pitch, roll, tx, ty, tz = pose_from_matrix(matrix)
        pose_3d = True
    else:  # geometric fallback (same sign conventions, rough degrees)
        top, chin = pts[FOREHEAD], pts[CHIN]
        face_h = float(np.linalg.norm(chin - top)) or 1e-6
        nose = pts[NOSE_TIP]
        yaw = math.degrees(math.asin(np.clip(2 * (nose[0] - (cheek_r[0] + cheek_l[0]) / 2) / face_w, -1, 1)))
        pitch = math.degrees(math.asin(np.clip(2 * (nose[1] - (top[1] + chin[1]) / 2) / face_h, -1, 1)))
        roll = math.degrees((right.roll + left.roll) / 2)
        center = (cheek_r + cheek_l) / 2
        tx, ty, tz = center[0] / frame_w - 0.5, center[1] / frame_h - 0.5, 1.0 / (face_w / frame_w)
        pose_3d = False

    nose = pts[NOSE_TIP]
    eyes_mid = (np.array(right.eye_center) + np.array(left.eye_center)) / 2
    nose_x = float((nose[0] - (cheek_r[0] + cheek_l[0]) / 2) / face_w)
    nose_y = float((nose[1] - eyes_mid[1]) / face_w)
    features = FaceFeatures(
        timestamp=timestamp, face_found=True, right=right, left=left, yaw=yaw, pitch=pitch, roll=roll,
        nose_x=nose_x, nose_y=nose_y,
        tx=tx, ty=ty, tz=tz, face_size=face_w / frame_w, inference_ms=inference_ms, pose_3d=pose_3d,
        blendshapes=bs, landmarks_2d=np.asarray(points, dtype=np.float32)[:, :2], frame_size=(frame_w, frame_h),
    )
    features.confidence = estimate_confidence(features)
    return features


def estimate_confidence(f: FaceFeatures) -> float:
    """Heuristic 0..1 confidence from face size, head pose, eye openness and iris plausibility.

    MediaPipe exposes no per-landmark score, so we penalise the conditions
    known to degrade webcam gaze estimates.
    """
    if not f.face_found or f.right is None or f.left is None:
        return 0.0
    # Too small = far away / low resolution: few pixels per iris.
    size = float(np.clip((f.face_size - 0.08) / 0.12, 0.0, 1.0))
    # Strong rotation hides one eye and breaks the eye geometry.
    pose = float(np.clip(1.0 - (abs(f.yaw) - 20) / 15, 0, 1) * np.clip(1.0 - (abs(f.pitch) - 20) / 15, 0, 1))
    openness = min(f.right.openness, f.left.openness)
    eyes = float(np.clip((openness - EYE_CLOSED_RATIO) / 0.1, 0.0, 1.0))
    # Iris must sit inside the eye, and both eyes should roughly agree.
    inside = all(abs(e.h) < 0.35 and abs(e.v) < 0.35 for e in (f.right, f.left))
    agree = abs(f.right.h - f.left.h) < 0.15 and abs(f.right.v - f.left.v) < 0.15
    plausible = 1.0 if inside and agree else (0.6 if inside else 0.2)
    return size * pose * eyes * plausible
