"""Qt-facing tracking pipeline: capture thread -> analysis thread -> UI signals."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from looksay.diagnostics.stats import TrackingStats
from looksay.tracking.camera import CaptureThread, LatestFrameSlot
from looksay.tracking.features import EYES, NOSE_TIP, FaceFeatures, compute_features
from looksay.tracking.landmarks import LandmarkDetector, TrackerUnavailable

log = logging.getLogger(__name__)

PREVIEW_WIDTH = 320
PREVIEW_INTERVAL_S = 1 / 15
DEBUG_INTERVAL_S = 1 / 20
# Iris landmarks are computed from an eye crop of the input frame, so more
# input pixels means steadier irises; measured cost is flat up to 1280 px.
INFERENCE_MAX_WIDTH = 1280
FACE_VIEW_H = 540
EYE_VIEW = (480, 320)

# BGR-free palette (frames are RGB)
C_MESH = (150, 150, 150)
C_CONTOUR = (255, 200, 0)
C_IRIS = (0, 230, 120)
C_REFINED = (255, 60, 200)
C_CENTER = (255, 255, 255)
C_AXIS = ((230, 60, 60), (60, 200, 60), (60, 120, 255))


@dataclass
class DebugFrame:
    face: QImage
    right_eye: QImage
    left_eye: QImage
    shifts: dict[str, tuple[float, float]] = field(default_factory=dict)


def _to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class TrackingWorker(QThread):
    features_ready = Signal(object)  # FaceFeatures
    preview_ready = Signal(QImage)
    debug_ready = Signal(object)  # DebugFrame
    unavailable = Signal(str)

    def __init__(self, camera_index: int = 0, resolution: tuple[int, int] = (1280, 720), fps: int = 30,
                 inference_every: int = 1, parent=None):
        super().__init__(parent)
        self.stats = TrackingStats()
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.inference_every = max(1, inference_every)
        self.preview_enabled = False
        self.landmarks_overlay = False
        self.debug_enabled = False
        self.iris_refinement = False
        self._running = True
        self._capture: CaptureThread | None = None

    def stop(self) -> None:
        self._running = False
        if self._capture:
            self._capture.stop()
        self.wait(3000)

    def run(self) -> None:
        try:
            import cv2
            detector = LandmarkDetector()
        except (TrackerUnavailable, ImportError) as exc:
            log.warning("Tracking unavailable: %s", exc)
            self.unavailable.emit(str(exc))
            return
        log.info("Landmark model loaded; opening camera %s", self.camera_index)
        slot = LatestFrameSlot(self.stats)
        self._capture = CaptureThread(slot, self.stats, self.camera_index, self.resolution, self.fps)
        self._capture.start()
        last_preview = last_debug = 0.0
        try:
            while self._running:
                frame = slot.take(timeout=0.5)
                if frame is None:
                    continue
                if frame.index % self.inference_every:
                    self.stats.frames_skipped += 1
                    continue
                image = frame.image
                h, w = image.shape[:2]
                if w > INFERENCE_MAX_WIDTH:
                    scale = INFERENCE_MAX_WIDTH / w
                    image = cv2.resize(image, (INFERENCE_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
                    h, w = image.shape[:2]
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                det = detector.detect(rgb, frame.timestamp)
                self.stats.record_inference(det.inference_ms)
                self.stats.process_fps.tick()
                points, shifts = det.points, {}
                if points is not None and self.iris_refinement:
                    from looksay.tracking.iris_refine import refine_iris
                    points, shifts = refine_iris(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), points)
                features = compute_features(points, w, h, frame.timestamp, det.inference_ms,
                                            det.blendshapes, det.matrix)
                self.features_ready.emit(features)
                if self.preview_enabled and frame.timestamp - last_preview >= PREVIEW_INTERVAL_S:
                    last_preview = frame.timestamp
                    self.preview_ready.emit(self._preview(rgb, points))
                if self.debug_enabled and frame.timestamp - last_debug >= DEBUG_INTERVAL_S:
                    last_debug = frame.timestamp
                    try:
                        self.debug_ready.emit(render_debug(rgb, points, features, det.matrix, det.points, shifts))
                    except Exception:  # a drawing problem must never stop tracking
                        log.exception("Debug view rendering failed")
        finally:
            self._capture.stop()
            detector.close()

    def _preview(self, rgb: np.ndarray, points: np.ndarray | None) -> QImage:
        import cv2

        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (PREVIEW_WIDTH, int(h * PREVIEW_WIDTH / w)), interpolation=cv2.INTER_AREA)
        if self.landmarks_overlay and points is not None:
            sh, sw = small.shape[:2]
            for i in range(0, len(points), 4):  # sparse mesh keeps the preview readable
                cv2.circle(small, (int(points[i, 0] * sw), int(points[i, 1] * sh)), 1, C_MESH, -1)
            for spec in EYES.values():
                for i in spec["contour"] + spec["iris"]:
                    cv2.circle(small, (int(points[i, 0] * sw), int(points[i, 1] * sh)), 1, C_IRIS, -1)
        return _to_qimage(small[:, ::-1])  # mirrored, like a mirror


# --------------------------------------------------------------------- debug
def _blank(size: tuple[int, int]) -> np.ndarray:
    return np.full((size[1], size[0], 3), 24, np.uint8)


def render_debug(rgb: np.ndarray, points: np.ndarray | None, f: FaceFeatures, matrix: np.ndarray | None,
                 raw_points: np.ndarray | None = None, shifts: dict | None = None) -> DebugFrame:
    """Face view and magnified eye views with landmark overlays (camera orientation, not mirrored)."""
    import cv2

    h, w = rgb.shape[:2]
    face_w = FACE_VIEW_H  # square: eyes in the upper half, HUD space over the chin
    if points is None or not f.face_found:
        face = cv2.resize(rgb, (face_w, FACE_VIEW_H), interpolation=cv2.INTER_AREA)
        return DebugFrame(_to_qimage(face), _to_qimage(_blank(EYE_VIEW)), _to_qimage(_blank(EYE_VIEW)))
    px = points[:, :2] * np.array([w, h], dtype=np.float32)

    # Face crop with margin, keeping 4:3.
    x0, y0 = px.min(axis=0)
    x1, y1 = px.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_h = max((y1 - y0), (x1 - x0)) * 0.62
    half_w = half_h
    cy += half_h * 0.12  # extra room below the chin for the HUD lines
    fx0, fy0 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
    fx1, fy1 = int(min(w, cx + half_w)), int(min(h, cy + half_h))
    crop = rgb[fy0:fy1, fx0:fx1]
    scale = FACE_VIEW_H / max(1, crop.shape[0])
    face = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), FACE_VIEW_H), interpolation=cv2.INTER_LINEAR)

    def fp(p):
        return int((p[0] - fx0) * scale), int((p[1] - fy0) * scale)

    for i in range(0, len(px), 3):
        cv2.circle(face, fp(px[i]), 1, C_MESH, -1, cv2.LINE_AA)
    for spec, eye in ((EYES["right"], f.right), (EYES["left"], f.left)):
        cv2.polylines(face, [np.array([fp(px[i]) for i in spec["contour"]], np.int32)], True, C_CONTOUR, 1,
                      cv2.LINE_AA)
        ic = fp(eye.iris_center)
        cv2.circle(face, ic, max(2, int(eye.iris_radius * scale)), C_IRIS, 1, cv2.LINE_AA)
        # Eye-in-head direction: iris offset from the eye centre, magnified.
        ec = fp(eye.eye_center)
        tip = (int(ec[0] + (ic[0] - ec[0]) * 6), int(ec[1] + (ic[1] - ec[1]) * 6))
        cv2.arrowedLine(face, ec, tip, C_REFINED, 2, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(face, fp(px[NOSE_TIP]), 7, (255, 255, 0), 2, cv2.LINE_AA)  # nose pointer
    if matrix is not None:  # head pose axes from the nose
        r = np.asarray(matrix)[:3, :3]
        origin = fp(px[NOSE_TIP])
        length = 0.25 * FACE_VIEW_H
        for k, color in enumerate(C_AXIS):
            d = r[:, k]
            end = (int(origin[0] + d[0] * length), int(origin[1] - d[1] * length))
            cv2.line(face, origin, end, color, 2, cv2.LINE_AA)
    if face.shape[1] != face_w:  # pad to a fixed size
        canvas = np.full((FACE_VIEW_H, face_w, 3), 24, np.uint8)
        off = max(0, (face_w - face.shape[1]) // 2)
        canvas[:, off:off + min(face_w, face.shape[1])] = face[:, :face_w]
        face = canvas

    eyes = {}
    for name, eye in (("right", f.right), ("left", f.left)):
        eyes[name] = _eye_view(rgb, px, EYES[name], eye,
                               None if raw_points is None else raw_points[:, :2] * np.array([w, h], np.float32))
    return DebugFrame(_to_qimage(face), _to_qimage(eyes["right"]), _to_qimage(eyes["left"]), dict(shifts or {}))


def _eye_view(rgb: np.ndarray, px: np.ndarray, spec: dict, eye, raw_px: np.ndarray | None) -> np.ndarray:
    import cv2

    vw, vh = EYE_VIEW
    cx, cy = eye.eye_center
    half_w = max(8.0, eye.width * 0.85)
    half_h = half_w * vh / vw
    # Rotate so the eye axis is horizontal (head roll removed).
    angle = math.degrees(eye.roll)
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    m[0, 2] += half_w - cx
    m[1, 2] += half_h - cy
    s = vw / (2 * half_w)
    m[:2] *= s
    view = cv2.warpAffine(rgb, m, (vw, vh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(24, 24, 24))

    def tp(p):
        q = m @ np.array([p[0], p[1], 1.0])
        return int(round(q[0])), int(round(q[1]))

    cv2.polylines(view, [np.array([tp(px[i]) for i in spec["contour"]], np.int32)], True, C_CONTOUR, 1,
                  cv2.LINE_AA)
    for i in (spec["a"], spec["b"]):
        cv2.drawMarker(view, tp(px[i]), C_CONTOUR, cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)
    ec = tp(eye.eye_center)
    cv2.drawMarker(view, ec, C_CENTER, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
    ic = tp(eye.iris_center)
    cv2.circle(view, ic, max(2, int(eye.iris_radius * s)), C_IRIS, 1, cv2.LINE_AA)
    cv2.circle(view, ic, 2, C_IRIS, -1, cv2.LINE_AA)
    cv2.line(view, ec, ic, C_REFINED, 1, cv2.LINE_AA)
    if raw_px is not None:  # landmark-only iris centre, to compare with the refined one
        raw = raw_px[list(spec["iris"])].mean(axis=0)
        cv2.drawMarker(view, tp(raw), (255, 255, 0), cv2.MARKER_DIAMOND, 6, 1, cv2.LINE_AA)
    # Scale bar: 0.1 eye widths, the typical range of h for the whole screen.
    bar = int(0.1 * eye.width * s)
    cv2.line(view, (8, vh - 10), (8 + bar, vh - 10), C_CENTER, 2)
    return view
