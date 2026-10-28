"""Dwell-time selection state machine (UI-independent and unit tested).

Feed it the target under the gaze point at every tick. A target is selected
once gaze has rested on it for `dwell_s`. Safeguards against accidental
selection:

* short excursions off the target (< `grace_s`, e.g. a saccade or jitter)
  do not reset progress, but the time away does not count either;
* after a selection nothing can be selected for `cooldown_s`, and the same
  target needs a full new dwell (so double letters are deliberate);
* when tracking is paused (face lost / low confidence) progress resets;
* time steps are capped so a stalled frame cannot complete a dwell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

MAX_STEP_S = 0.1


@dataclass
class DwellState:
    target: Hashable | None
    progress: float  # 0..1
    selected: Hashable | None = None
    cooling_down: bool = False


class DwellSelector:
    def __init__(self, dwell_s: float = 1.0, cooldown_s: float = 0.4, grace_s: float = 0.15,
                 blink_min_hold_s: float = 0.3):
        self.dwell_s = dwell_s
        self.cooldown_s = cooldown_s
        self.grace_s = grace_s
        self.blink_min_hold_s = blink_min_hold_s
        self.reset()

    def reset(self) -> None:
        self._target: Hashable | None = None
        self._elapsed = 0.0
        self._last_t: float | None = None
        self._away_since: float | None = None
        self._cooldown_until = 0.0

    def cancel(self) -> None:
        """Drop the current target and progress but keep the cooldown running."""
        self._target, self._elapsed, self._away_since = None, 0.0, None

    @property
    def target(self) -> Hashable | None:
        return self._target

    @property
    def progress(self) -> float:
        return min(1.0, self._elapsed / self.dwell_s) if self.dwell_s > 0 else 0.0

    def _select(self, now: float) -> Hashable:
        selected = self._target
        self._elapsed = 0.0
        self._cooldown_until = now + self.cooldown_s
        return selected

    def update(self, target: Hashable | None, now: float, active: bool = True, hold: bool = False) -> DwellState:
        """Advance the state machine.

        active=False pauses and resets (face lost, low confidence).
        hold=True freezes progress without resetting (eyes closed during a blink).
        """
        dt = 0.0 if self._last_t is None else min(MAX_STEP_S, max(0.0, now - self._last_t))
        self._last_t = now
        cooling = now < self._cooldown_until

        if hold:
            return DwellState(self._target, self.progress, cooling_down=cooling)

        if not active:
            self._target, self._elapsed, self._away_since = None, 0.0, None
            return DwellState(None, 0.0, cooling_down=cooling)

        if target != self._target:
            if self._target is not None and self._away_since is None:
                self._away_since = now
            if self._target is None or (self._away_since is not None and now - self._away_since >= self.grace_s):
                self._target, self._elapsed, self._away_since = target, 0.0, None
            return DwellState(self._target, self.progress, cooling_down=cooling)

        self._away_since = None
        if self._target is None or cooling:
            return DwellState(self._target, self.progress, cooling_down=cooling)

        self._elapsed += dt
        if self._elapsed >= self.dwell_s:
            selected = self._select(now)
            return DwellState(self._target, 0.0, selected=selected, cooling_down=True)
        return DwellState(self._target, self.progress)

    def held_for(self) -> float:
        return self._elapsed

    def trigger(self, now: float) -> Hashable | None:
        """Explicit selection (deliberate blink): select the current target if it was held long enough."""
        if self._target is None or now < self._cooldown_until or self._elapsed < self.blink_min_hold_s:
            return None
        return self._select(now)
