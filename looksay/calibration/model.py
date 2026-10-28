"""Gaze mapping: eye/head features -> normalised screen coordinates.

Model: ridge regression on standardised features (iris offsets of both
eyes, lid aperture, gaze blendshapes, 3D head pose), plus quadratic terms
of the combined horizontal/vertical eye signals when there are enough
calibration points. The ridge strength is chosen by leave-one-point-out
cross-validation, and that cross-validated error (not the optimistic
training error) is what we report as the calibration accuracy.

Outputs are in the 0..1 space of the calibrated window, so resizing the
window does not invalidate the calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from looksay.tracking.features import FEATURE_NAMES

MODEL_VERSION = 2
MIN_SAMPLES_PER_POINT = 5
RIDGE_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
# Extra samples (smooth pursuit, confirmed selections) within this distance of
# a left-out calibration point are excluded from that fold, so the reported
# cross-validated error stays honest.
CV_EXCLUSION_RADIUS = 0.12
PURSUIT_WEIGHT = 0.5
MAX_EXTRA_SAMPLES = 1500
# Smallest standard deviation per feature (its natural noise level), so a
# feature that barely moved during calibration cannot dominate the model.
STD_FLOOR = np.array([0.004, 0.004, 0.004, 0.004, 0.004, 0.02, 0.02, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0])
N_FEATURES = len(FEATURE_NAMES)


def calibration_targets(n_points: int, margin: float = 0.08) -> list[tuple[float, float]]:
    """Target positions (normalised). Centre first so the user starts at a natural spot."""
    lo, mid, hi = margin, 0.5, 1.0 - margin
    if n_points == 5:
        return [(mid, mid), (lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    grid = [(x, y) for y in (lo, mid, hi) for x in (lo, mid, hi)]
    grid.remove((mid, mid))
    targets = [(mid, mid)] + grid
    if n_points >= 13:
        q1, q3 = (lo + mid) / 2, (mid + hi) / 2
        targets += [(q1, q1), (q3, q1), (q3, q3), (q1, q3)]
    return targets


def _design(z: np.ndarray, quadratic: bool) -> np.ndarray:
    cols = [np.ones(len(z)), *z.T]
    if quadratic:
        gh = (z[:, 0] + z[:, 2]) / 2.0  # horizontal iris offset, both eyes
        gv = (z[:, 1] + z[:, 3] + z[:, 4]) / 3.0  # vertical: iris + lid aperture
        cols += [gh * gh, gv * gv, gh * gv]
    return np.stack(cols, axis=1)


def _solve(a: np.ndarray, y: np.ndarray, ridge: float, w: np.ndarray | None = None) -> np.ndarray:
    if w is not None:
        sw = np.sqrt(w)[:, None]
        a, y = a * sw, y * sw
    reg = ridge * len(a) * np.eye(a.shape[1])
    reg[0, 0] = 0.0  # do not shrink the intercept
    return np.linalg.solve(a.T @ a + reg, a.T @ y)


def pursuit_path(t: float, duration: float) -> tuple[float, float]:
    """Smooth Lissajous path covering the screen; the moving calibration dot follows it."""
    u = min(max(t / duration, 0.0), 1.0)
    x = 0.5 + 0.42 * np.sin(2 * np.pi * 1.6 * u)
    y = 0.5 + 0.40 * np.sin(2 * np.pi * 2.3 * u + np.pi / 2)
    return float(x), float(y)


def align_pursuit(samples: list[tuple[float, np.ndarray]], duration: float,
                  lags=tuple(np.arange(0.0, 0.36, 0.033))) -> tuple[list[tuple[np.ndarray, tuple]], float]:
    """Pair pursuit samples with where the dot was `lag` seconds earlier.

    The eyes (and the camera pipeline) trail the moving dot by ~100-250 ms; the
    lag that best explains the data with a simple linear map is used.
    """
    if len(samples) < 30:
        return [], 0.0
    ts = np.array([t for t, _ in samples])
    x = np.array([v for _, v in samples])
    eye = x[:, :7]
    z = (eye - eye.mean(axis=0)) / np.maximum(eye.std(axis=0), 1e-4)
    a = np.hstack([np.ones((len(z), 1)), z])
    best = None
    for lag in lags:
        y = np.array([pursuit_path(t - lag, duration) for t in ts])
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        rms = float(np.sqrt(np.mean(np.sum((a @ coef - y) ** 2, axis=1))))
        if best is None or rms < best[0]:
            best = (rms, lag, y)
    _, lag, y = best
    keep = ts - lag > 1.0  # skip the pursuit onset (catch-up saccades)
    return [(x[i], tuple(y[i])) for i in range(len(x)) if keep[i]], float(lag)


@dataclass
class GazeModel:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray  # (n_terms, 2)
    quadratic: bool
    eye: str = "both"
    fit_error: float = 0.0  # cross-validated mean error, normalised units
    ridge: float = 0.01
    # (target_x, target_y, predicted_x, predicted_y) per calibration point (cross-validated)
    point_results: list[tuple[float, float, float, float]] = field(default_factory=list)
    version: int = MODEL_VERSION

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        z = (x - self.mean) / self.std
        return _design(z, self.quadratic) @ self.coef

    def error_px(self, width: float, height: float) -> float:
        """Mean cross-validated error in pixels for a window of this size."""
        if not self.point_results:
            return self.fit_error * float(np.hypot(width, height)) / np.sqrt(2)
        return float(np.mean([np.hypot((tx - px) * width, (ty - py) * height)
                              for tx, ty, px, py in self.point_results]))

    def to_dict(self) -> dict:
        return {
            "version": self.version, "mean": self.mean.tolist(), "std": self.std.tolist(),
            "coef": self.coef.tolist(), "quadratic": self.quadratic, "eye": self.eye,
            "fit_error": self.fit_error, "ridge": self.ridge, "point_results": self.point_results,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GazeModel":
        if data.get("version") != MODEL_VERSION:
            raise ValueError("Unsupported calibration model version (please recalibrate)")
        return cls(
            mean=np.array(data["mean"]), std=np.array(data["std"]), coef=np.array(data["coef"]),
            quadratic=bool(data["quadratic"]), eye=data.get("eye", "both"),
            fit_error=float(data.get("fit_error", 0.0)), ridge=float(data.get("ridge", 0.01)),
            point_results=[tuple(p) for p in data.get("point_results", [])],
        )


@dataclass
class CalibrationSession:
    """Collects feature samples per target and fits a GazeModel."""

    targets: list[tuple[float, float]]
    eye: str = "both"
    samples: dict[int, list[np.ndarray]] = field(default_factory=dict)
    # (vector, target, weight): smooth-pursuit samples and confirmed selections
    extra: list[tuple[np.ndarray, tuple[float, float], float]] = field(default_factory=list)

    def add_extra(self, vector: np.ndarray, target: tuple[float, float], weight: float = 1.0) -> None:
        self.extra.append((np.asarray(vector, dtype=np.float64), (float(target[0]), float(target[1])), weight))
        if len(self.extra) > MAX_EXTRA_SAMPLES:  # forget the oldest
            del self.extra[: len(self.extra) - MAX_EXTRA_SAMPLES]

    def to_dict(self) -> dict:
        return {
            "targets": self.targets, "eye": self.eye,
            "samples": {str(k): np.round(np.array(v), 5).tolist() for k, v in self.samples.items()},
            "extra": [[np.round(v, 5).tolist(), list(t), w] for v, t, w in self.extra],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationSession":
        session = cls([tuple(t) for t in data["targets"]], eye=data.get("eye", "both"))
        session.samples = {int(k): [np.array(r) for r in v] for k, v in data.get("samples", {}).items()}
        session.extra = [(np.array(v), tuple(t), float(w)) for v, t, w in data.get("extra", [])]
        return session

    def add_sample(self, target_index: int, vector: np.ndarray) -> None:
        self.samples.setdefault(target_index, []).append(np.asarray(vector, dtype=np.float64))

    def usable_points(self) -> int:
        return sum(1 for s in self.samples.values() if len(s) >= MIN_SAMPLES_PER_POINT)

    @staticmethod
    def _inliers(samples: list[np.ndarray]) -> np.ndarray:
        """Drop frames where the eye signal is far from this point's median (saccades, blinks)."""
        arr = np.array(samples)
        eye_cols = arr[:, :5]
        median = np.median(eye_cols, axis=0)
        mad = np.median(np.abs(eye_cols - median), axis=0) * 1.4826 + 1e-4
        keep = np.all(np.abs(eye_cols - median) <= 3.0 * mad, axis=1)
        return arr[keep] if keep.sum() >= MIN_SAMPLES_PER_POINT else arr

    def fit(self) -> GazeModel | None:
        points = []
        for idx, samples in sorted(self.samples.items()):
            if len(samples) >= MIN_SAMPLES_PER_POINT:
                points.append((idx, self._inliers(samples)))
        if len(points) < 4:
            return None
        xs = [p for _, p in points]
        ys = [np.tile(self.targets[i], (len(p), 1)) for i, p in points]
        groups = [np.full(len(p), i) for i, p in points]
        weights = [np.ones(len(p)) for _, p in points]
        if self.extra:
            xs.append(np.array([v for v, _, _ in self.extra]))
            ys.append(np.array([t for _, t, _ in self.extra]))
            groups.append(np.full(len(self.extra), -1))
            weights.append(np.array([w for _, _, w in self.extra]))
        x, y = np.vstack(xs), np.vstack(ys)
        groups, w = np.concatenate(groups), np.concatenate(weights)
        mean = x.mean(axis=0)
        std = np.maximum(x.std(axis=0), STD_FLOOR[: x.shape[1]])
        z = (x - mean) / std
        quadratic = len(points) >= 8 or len(self.extra) >= 100

        # Leave-one-point-out cross-validation to pick the ridge strength.
        best = None
        for ridge in RIDGE_GRID:
            results = []
            for idx, _ in points:
                target = np.array(self.targets[idx])
                near = (groups == -1) & (np.linalg.norm(y - target, axis=1) < CV_EXCLUSION_RADIUS)
                train = (groups != idx) & ~near
                coef = _solve(_design(z[train], quadratic), y[train], ridge, w[train])
                pred = np.median(_design(z[groups == idx], quadratic) @ coef, axis=0)
                results.append((*self.targets[idx], float(pred[0]), float(pred[1])))
            err = float(np.mean([np.hypot(tx - px, ty - py) for tx, ty, px, py in results]))
            if best is None or err < best[0]:
                best = (err, ridge, results)
        err, ridge, results = best
        coef = _solve(_design(z, quadratic), y, ridge, w)
        return GazeModel(mean=mean, std=std, coef=coef, quadratic=quadratic, eye=self.eye,
                         fit_error=err, ridge=ridge, point_results=results)
