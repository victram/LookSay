"""Gaze mapping: eye features -> normalised screen coordinates.

A second-order polynomial ridge regression, fitted per axis on samples
collected while the user looks at known targets. Outputs are in the 0..1
coordinate space of the calibrated window, so resizing the window does
not invalidate the calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MODEL_VERSION = 1
MIN_SAMPLES_PER_POINT = 5
STD_FLOOR = 0.02


def calibration_targets(n_points: int, margin: float = 0.08) -> list[tuple[float, float]]:
    """Target positions (normalised). Centre first so the user starts at a natural spot."""
    lo, mid, hi = margin, 0.5, 1.0 - margin
    if n_points == 5:
        return [(mid, mid), (lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    grid = [(x, y) for y in (lo, mid, hi) for x in (lo, mid, hi)]
    grid.remove((mid, mid))
    return [(mid, mid)] + grid


def _design(x: np.ndarray, quadratic: bool) -> np.ndarray:
    h, v, yaw, pitch = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    cols = [np.ones_like(h), h, v, yaw, pitch]
    if quadratic:
        cols += [h * v, h * h, v * v]
    return np.stack(cols, axis=1)


@dataclass
class GazeModel:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray  # (n_terms, 2)
    quadratic: bool
    eye: str = "both"
    fit_error: float = 0.0  # mean error of per-target medians, normalised units
    version: int = MODEL_VERSION

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        z = (x - self.mean) / self.std
        return _design(z, self.quadratic) @ self.coef

    def to_dict(self) -> dict:
        return {
            "version": self.version, "mean": self.mean.tolist(), "std": self.std.tolist(),
            "coef": self.coef.tolist(), "quadratic": self.quadratic, "eye": self.eye,
            "fit_error": self.fit_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GazeModel":
        if data.get("version") != MODEL_VERSION:
            raise ValueError("Unsupported calibration model version")
        return cls(
            mean=np.array(data["mean"]), std=np.array(data["std"]), coef=np.array(data["coef"]),
            quadratic=bool(data["quadratic"]), eye=data.get("eye", "both"),
            fit_error=float(data.get("fit_error", 0.0)),
        )


@dataclass
class CalibrationSession:
    """Collects feature samples per target and fits a GazeModel."""

    targets: list[tuple[float, float]]
    eye: str = "both"
    samples: dict[int, list[np.ndarray]] = field(default_factory=dict)

    def add_sample(self, target_index: int, vector: np.ndarray) -> None:
        self.samples.setdefault(target_index, []).append(np.asarray(vector, dtype=np.float64))

    def usable_points(self) -> int:
        return sum(1 for s in self.samples.values() if len(s) >= MIN_SAMPLES_PER_POINT)

    @staticmethod
    def _inliers(samples: list[np.ndarray]) -> np.ndarray:
        arr = np.array(samples)
        median = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - median), axis=0) + 1e-9
        keep = np.all(np.abs(arr - median) <= 3.0 * 1.4826 * mad + 1e-6, axis=1)
        return arr[keep] if keep.sum() >= MIN_SAMPLES_PER_POINT // 2 + 1 else arr

    def fit(self, ridge: float = 1e-2) -> GazeModel | None:
        xs, ys, medians, med_targets = [], [], [], []
        for idx, samples in sorted(self.samples.items()):
            if len(samples) < MIN_SAMPLES_PER_POINT:
                continue
            inl = self._inliers(samples)
            xs.append(inl)
            ys.append(np.tile(self.targets[idx], (len(inl), 1)))
            medians.append(np.median(inl, axis=0))
            med_targets.append(self.targets[idx])
        if len(xs) < 4:
            return None
        x, y = np.vstack(xs), np.vstack(ys)
        mean = x.mean(axis=0)
        std = np.maximum(x.std(axis=0), STD_FLOOR)
        quadratic = len(xs) >= 8
        a = _design((x - mean) / std, quadratic)
        reg = ridge * np.eye(a.shape[1])
        reg[0, 0] = 0.0  # do not shrink the intercept
        coef = np.linalg.solve(a.T @ a + len(x) * reg, a.T @ y)
        model = GazeModel(mean=mean, std=std, coef=coef, quadratic=quadratic, eye=self.eye)
        pred = model.predict(np.array(medians))
        model.fit_error = float(np.mean(np.linalg.norm(pred - np.array(med_targets), axis=1)))
        return model
