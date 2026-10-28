"""Qt-facing tracking pipeline: capture thread -> analysis thread -> UI signals."""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from looksay.diagnostics.stats import TrackingStats
from looksay.tracking.camera import CaptureThread, LatestFrameSlot
from looksay.tracking.features import IRIS_RING, LEFT_EYE, RIGHT_EYE, FaceFeatures, compute_features
from looksay.tracking.landmarks import LandmarkDetector, TrackerUnavailable

log = logging.getLogger(__name__)

PREVIEW_WIDTH = 320
PREVIEW_INTERVAL_S = 1 / 15
INFERENCE_MAX_WIDTH = 640  # larger frames are downscaled before landmark detection


class TrackingWorker(QThread):
    features_ready = Signal(object)  # FaceFeatures
    preview_ready = Signal(QImage)
    unavailable = Signal(str)

    def __init__(self, camera_index: int = 0, resolution: tuple[int, int] = (640, 480), fps: int = 30,
                 inference_every: int = 1, parent=None):
        super().__init__(parent)
        self.stats = TrackingStats()
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.inference_every = max(1, inference_every)
        self.preview_enabled = False
        self.landmarks_overlay = False
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
        last_preview = 0.0
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
                points, ms = detector.detect(rgb, frame.timestamp)
                self.stats.record_inference(ms)
                self.stats.process_fps.tick()
                features = compute_features(points, w, h, frame.timestamp, ms) if points is not None \
                    else FaceFeatures(timestamp=frame.timestamp, face_found=False, inference_ms=ms)
                self.features_ready.emit(features)
                if self.preview_enabled and frame.timestamp - last_preview >= PREVIEW_INTERVAL_S:
                    last_preview = frame.timestamp
                    self.preview_ready.emit(self._preview(rgb, points))
        finally:
            self._capture.stop()
            detector.close()

    def _preview(self, rgb: np.ndarray, points: np.ndarray | None) -> QImage:
        import cv2

        h, w = rgb.shape[:2]
        scale = PREVIEW_WIDTH / w
        small = cv2.resize(rgb, (PREVIEW_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
        if self.landmarks_overlay and points is not None:
            sh, sw = small.shape[:2]
            eye_ids = list(RIGHT_EYE.values()) + list(LEFT_EYE.values())
            for i in range(0, len(points), 4):  # sparse face mesh keeps the preview readable
                cv2.circle(small, (int(points[i, 0] * sw), int(points[i, 1] * sh)), 1, (160, 160, 160), -1)
            for i in eye_ids:
                cv2.circle(small, (int(points[i, 0] * sw), int(points[i, 1] * sh)), 2, (0, 200, 255), -1)
            for ring in IRIS_RING.values():
                for i in ring:
                    cv2.circle(small, (int(points[i, 0] * sw), int(points[i, 1] * sh)), 1, (0, 255, 0), -1)
        small = np.ascontiguousarray(small[:, ::-1])  # mirror for a natural "mirror" view
        sh, sw = small.shape[:2]
        return QImage(small.data, sw, sh, 3 * sw, QImage.Format.Format_RGB888).copy()
