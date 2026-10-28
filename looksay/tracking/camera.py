"""Webcam capture on a background thread with a bounded (size-1) frame slot.

The capture thread always overwrites the slot with the newest frame, so the
analysis thread never works on stale images and memory stays constant:
if analysis is slower than the camera, older frames are dropped (counted in
stats) - this is the adaptive frame skipping. Frames are never written to disk.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from looksay.diagnostics.stats import TrackingStats

log = logging.getLogger(__name__)

RECONNECT_DELAY_S = 2.0
MAX_CONSECUTIVE_FAILURES = 15


@dataclass
class Frame:
    image: np.ndarray  # BGR
    timestamp: float
    index: int


class LatestFrameSlot:
    def __init__(self, stats: TrackingStats):
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._stats = stats

    def put(self, frame: Frame) -> None:
        with self._cond:
            if self._frame is not None:
                self._stats.frames_dropped += 1
            self._frame = frame
            self._cond.notify()

    def take(self, timeout: float) -> Frame | None:
        with self._cond:
            if self._frame is None:
                self._cond.wait(timeout)
            frame, self._frame = self._frame, None
            return frame


class CaptureThread(threading.Thread):
    def __init__(self, slot: LatestFrameSlot, stats: TrackingStats, index: int = 0,
                 resolution: tuple[int, int] = (640, 480), fps: int = 30):
        super().__init__(name="looksay-capture", daemon=True)
        self.slot = slot
        self.stats = stats
        self.index = index
        self.resolution = resolution
        self.fps = fps
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _open(self):
        import cv2

        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self) -> None:
        index = 0
        while not self._stop.is_set():
            try:
                cap = self._open()
            except Exception as exc:  # OpenCV missing or backend crash
                cap = None
                self.stats.camera_error = str(exc)
            if cap is None:
                self.stats.camera_ok = False
                self.stats.camera_error = self.stats.camera_error or "camera not available or permission denied"
                self._stop.wait(RECONNECT_DELAY_S)
                continue
            self.stats.camera_ok, self.stats.camera_error = True, ""
            failures = 0
            min_interval = 1.0 / max(1, self.fps)
            last = 0.0
            while not self._stop.is_set():
                ok, image = cap.read()
                if not ok or image is None:
                    failures += 1
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        log.warning("Camera stopped delivering frames; reconnecting")
                        self.stats.camera_ok = False
                        self.stats.camera_error = "camera disconnected"
                        break
                    time.sleep(0.02)
                    continue
                failures = 0
                now = time.monotonic()
                if now - last < min_interval * 0.5:
                    continue  # camera ignored the FPS request; throttle
                last = now
                index += 1
                self.stats.frames_captured += 1
                self.stats.capture_fps.tick()
                self.slot.put(Frame(image, now, index))
            cap.release()
            if not self._stop.is_set():
                self._stop.wait(RECONNECT_DELAY_S)
