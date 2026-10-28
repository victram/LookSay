"""One Euro filter: smooth when the gaze is still, responsive when it moves.

Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy
Input in Interactive Systems", CHI 2012.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.005, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    @classmethod
    def from_strength(cls, strength: int) -> "OneEuroFilter":
        """Map the 0 (raw) .. 10 (very smooth) user setting to filter parameters."""
        strength = max(0, min(10, strength))
        min_cutoff = 4.0 * (0.05 / 4.0) ** (strength / 10.0)  # 4 Hz .. 0.05 Hz
        return cls(min_cutoff=min_cutoff, beta=0.004)

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return x
        dt = t - self._t
        if dt <= 0:
            return self._x
        self._t = t
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = _alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x
