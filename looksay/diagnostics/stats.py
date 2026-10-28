"""Lightweight runtime statistics (no logging of camera content)."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


class RateCounter:
    """Events per second over a sliding window."""

    def __init__(self, window_s: float = 2.0, clock=time.monotonic):
        self.window_s = window_s
        self._clock = clock
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def tick(self) -> None:
        now = self._clock()
        with self._lock:
            self._events.append(now)
            self._trim(now)

    def _trim(self, now: float) -> None:
        while self._events and now - self._events[0] > self.window_s:
            self._events.popleft()

    @property
    def rate(self) -> float:
        now = self._clock()
        with self._lock:
            self._trim(now)
            if len(self._events) < 2:
                return 0.0
            span = self._events[-1] - self._events[0]
            return (len(self._events) - 1) / span if span > 0 else 0.0


@dataclass
class TrackingStats:
    capture_fps: RateCounter = field(default_factory=RateCounter)
    process_fps: RateCounter = field(default_factory=RateCounter)
    frames_captured: int = 0
    frames_dropped: int = 0
    frames_skipped: int = 0  # intentionally not analysed (inference_every)
    last_inference_ms: float = 0.0
    avg_inference_ms: float = 0.0
    camera_ok: bool = False
    camera_error: str = ""
    camera_note: str = ""  # e.g. automatic resolution fallback
    capture_size: tuple[int, int] = (0, 0)  # actual frame size delivered by the camera

    def record_inference(self, ms: float) -> None:
        self.last_inference_ms = ms
        self.avg_inference_ms = ms if self.avg_inference_ms == 0 else 0.9 * self.avg_inference_ms + 0.1 * ms

    @property
    def drop_ratio(self) -> float:
        return self.frames_dropped / self.frames_captured if self.frames_captured else 0.0
