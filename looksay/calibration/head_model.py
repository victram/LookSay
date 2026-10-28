"""Nose/head pointer mapping: head features -> normalised screen coordinates.

Head features are the nose-tip offsets relative to the face (and 3D yaw and
pitch when available). A 5-point "point your nose at the dot" calibration
fits a linear ridge model; the leave-one-point-out error is reported, as for
the eye model.

Without a head calibration, `default_head_model()` still gives a sensible
*relative* mapping, which is all the eyes+nose mode needs for fine-tuning:
turning the head to the user's right moves the cursor right (the nose moves
to image-left in a non-mirrored camera), tilting it up moves the cursor up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HEAD_MODEL_VERSION = 1
HEAD_STD_FLOOR = np.array([0.003, 0.003, 0.5, 0.5])
MIN_SAMPLES = 5
DEFAULT_GAIN = 2.2  # screen widths per face width of nose offset


@dataclass
class HeadModel:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray  # (5, 2): intercept + 4 features
    fit_error: float = 0.0
    point_results: list[tuple[float, float, float, float]] = field(default_factory=list)
    calibrated: bool = True
    version: int = HEAD_MODEL_VERSION

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        z = (x - self.mean) / self.std
        return np.hstack([np.ones((len(z), 1)), z]) @ self.coef

    def error_px(self, width: float, height: float) -> float:
        if not self.point_results:
            return 0.0
        return float(np.mean([np.hypot((tx - px) * width, (ty - py) * height)
                              for tx, ty, px, py in self.point_results]))

    def to_dict(self) -> dict:
        return {"version": self.version, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "coef": self.coef.tolist(), "fit_error": self.fit_error, "point_results": self.point_results}

    @classmethod
    def from_dict(cls, data: dict) -> "HeadModel":
        if data.get("version") != HEAD_MODEL_VERSION:
            raise ValueError("Unsupported head model version")
        return cls(mean=np.array(data["mean"]), std=np.array(data["std"]), coef=np.array(data["coef"]),
                   fit_error=float(data.get("fit_error", 0.0)),
                   point_results=[tuple(p) for p in data.get("point_results", [])])


def default_head_model() -> HeadModel:
    coef = np.zeros((5, 2))
    coef[0] = (0.5, 0.5)
    coef[1, 0] = -DEFAULT_GAIN  # nose to image-left = head turned to the user's right = cursor right
    coef[2, 1] = DEFAULT_GAIN  # nose lower relative to the eyes = head tilted down = cursor down
    return HeadModel(mean=np.zeros(4), std=np.ones(4), coef=coef, calibrated=False)


def _solve(a: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    reg = ridge * len(a) * np.eye(a.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(a.T @ a + reg, a.T @ y)


def fit_head_model(targets: list[tuple[float, float]], samples: dict[int, list[np.ndarray]],
                   ridge: float = 0.01) -> HeadModel | None:
    points = []
    for idx, rows in sorted(samples.items()):
        if len(rows) >= MIN_SAMPLES:
            arr = np.array(rows)
            med = np.median(arr, axis=0)
            mad = np.median(np.abs(arr - med), axis=0) * 1.4826 + 1e-4
            keep = np.all(np.abs(arr - med) <= 3 * mad, axis=1)
            points.append((idx, arr[keep] if keep.sum() >= MIN_SAMPLES else arr))
    if len(points) < 4:
        return None
    x = np.vstack([p for _, p in points])
    y = np.vstack([np.tile(targets[i], (len(p), 1)) for i, p in points])
    groups = np.concatenate([np.full(len(p), i) for i, p in points])
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), HEAD_STD_FLOOR[: x.shape[1]])
    a = np.hstack([np.ones((len(x), 1)), (x - mean) / std])
    results = []
    for idx, _ in points:
        train = groups != idx
        coef = _solve(a[train], y[train], ridge)
        pred = np.median(a[~train] @ coef, axis=0)
        results.append((*targets[idx], float(pred[0]), float(pred[1])))
    coef = _solve(a, y, ridge)
    err = float(np.mean([np.hypot(tx - px, ty - py) for tx, ty, px, py in results]))
    return HeadModel(mean=mean, std=std, coef=coef, fit_error=err, point_results=results)
