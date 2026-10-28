"""Deliberate-blink detection for optional blink-to-select.

Natural blinks last ~100-300 ms. A *deliberate* blink is both eyes closed
for between `min_s` and `max_s`; anything longer (resting the eyes) is
ignored. Only used when the user enables it.
"""

from __future__ import annotations


class BlinkDetector:
    def __init__(self, min_s: float = 0.45, max_s: float = 1.6):
        self.min_s = min_s
        self.max_s = max_s
        self._closed_since: float | None = None

    @property
    def closed(self) -> bool:
        return self._closed_since is not None

    def update(self, eyes_closed: bool, now: float) -> bool:
        """Returns True exactly once when a deliberate blink ends."""
        if eyes_closed:
            if self._closed_since is None:
                self._closed_since = now
            return False
        if self._closed_since is None:
            return False
        duration = now - self._closed_since
        self._closed_since = None
        return self.min_s <= duration <= self.max_s

    def reset(self) -> None:
        self._closed_since = None
