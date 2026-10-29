"""Gaze mapping: eye/head features -> normalised screen coordinates.

Model: ridge regression on standardised features (iris offsets of both
eyes, lid aperture, gaze blendshapes, 3D head pose), plus quadratic terms
of the combined horizontal/vertical eye signals when there are enough
calibration points. The ridge strength is chosen by leave-one-point-out
cross-validation, and that cross-validated error (not the optimistic
training error) is what we report as the calibration accuracy.

Appearance features follow the 13 geometric ones: the 258 outputs of the
gaze CNN when it is available (cnn_gaze.py), else raw eye-image patches.
They are standardised per column, reduced with PCA fitted on the
calibration data, and cross-validation decides how many components join
the regression.

Outputs are in the 0..1 space of the calibrated window, so resizing the
window does not invalidate the calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import base64

import numpy as np

from looksay.tracking.features import FEATURE_NAMES

MODEL_VERSION = 4  # v4: gaze-CNN appearance features, per-column scaling before PCA
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
N_PCA = 32  # most appearance components cross-validation may keep
PCA_STD_FLOOR = 0.05


def _project(x: np.ndarray, pca_mean: np.ndarray | None, pca_comp: np.ndarray | None,
             pca_scale: np.ndarray | None = None) -> np.ndarray:
    """Geometric features + PCA-reduced appearance (the mean is used when a frame has none or another kind)."""
    geo = x[:, :N_FEATURES]
    if pca_mean is None or pca_comp is None:
        return geo
    app = x[:, N_FEATURES:] if x.shape[1] - N_FEATURES == len(pca_mean) else np.tile(pca_mean, (len(x), 1))
    centred = app - pca_mean
    if pca_scale is not None:
        centred = centred / pca_scale
    return np.hstack([geo, centred @ pca_comp.T])


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
    pca_mean: np.ndarray | None = None
    pca_comp: np.ndarray | None = None  # (k, appearance size)
    pca_scale: np.ndarray | None = None  # per-column std of the appearance features
    version: int = MODEL_VERSION

    @property
    def uses_appearance(self) -> bool:
        return self.pca_comp is not None

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        z = (_project(x, self.pca_mean, self.pca_comp, self.pca_scale) - self.mean) / self.std
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
            "pca_mean": None if self.pca_mean is None else self.pca_mean.tolist(),
            "pca_comp": None if self.pca_comp is None else self.pca_comp.tolist(),
            "pca_scale": None if self.pca_scale is None else self.pca_scale.tolist(),
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
            pca_mean=None if data.get("pca_mean") is None else np.array(data["pca_mean"]),
            pca_comp=None if data.get("pca_comp") is None else np.array(data["pca_comp"]),
            pca_scale=None if data.get("pca_scale") is None else np.array(data["pca_scale"]),
        )


def _pack(rows: list[np.ndarray]) -> dict:
    """Compact storage for feature rows (float16, base64): eye patches make rows long."""
    if not rows:
        return {"n": 0, "d": 0, "b": ""}
    arr = np.array(rows, dtype=np.float16)
    return {"n": arr.shape[0], "d": arr.shape[1], "b": base64.b64encode(arr.tobytes()).decode("ascii")}


def _unpack(data: dict) -> list[np.ndarray]:
    if not data.get("n"):
        return []
    arr = np.frombuffer(base64.b64decode(data["b"]), dtype=np.float16).reshape(data["n"], data["d"])
    return [row.astype(np.float64) for row in arr]


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
            "version": MODEL_VERSION, "targets": self.targets, "eye": self.eye,
            "samples": {str(k): _pack(v) for k, v in self.samples.items()},
            "extra": _pack([v for v, _, _ in self.extra]),
            "extra_meta": [[t[0], t[1], w] for _, t, w in self.extra],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationSession":
        if data.get("version") != MODEL_VERSION:
            raise ValueError("Calibration data from an older version")
        session = cls([tuple(t) for t in data["targets"]], eye=data.get("eye", "both"))
        session.samples = {int(k): _unpack(v) for k, v in data.get("samples", {}).items()}
        vectors = _unpack(data.get("extra", {}))
        session.extra = [(v, (m[0], m[1]), float(m[2])) for v, m in zip(vectors, data.get("extra_meta", []))]
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
        # Frames whose appearance part is missing or of another kind (CNN vs patches) keep only
        # their geometric features; their appearance columns become NaN and are filled below.
        widths = [len(v) for s in self.samples.values() for v in s] + [len(v) for v, _, _ in self.extra]
        if not widths:
            return None
        width = max(set(widths), key=widths.count)

        def conform(v: np.ndarray) -> np.ndarray:
            if len(v) == width:
                return v
            return np.concatenate([v[:N_FEATURES], np.full(width - N_FEATURES, np.nan)])

        points = []
        for idx, samples in sorted(self.samples.items()):
            if len(samples) >= MIN_SAMPLES_PER_POINT:
                points.append((idx, self._inliers([conform(v) for v in samples])))
        if len(points) < 4:
            return None
        xs = [p for _, p in points]
        ys = [np.tile(self.targets[i], (len(p), 1)) for i, p in points]
        groups = [np.full(len(p), i) for i, p in points]
        weights = [np.ones(len(p)) for _, p in points]
        if self.extra:
            xs.append(np.array([conform(v) for v, _, _ in self.extra]))
            ys.append(np.array([t for _, t, _ in self.extra]))
            groups.append(np.full(len(self.extra), -1))
            weights.append(np.array([w for _, _, w in self.extra]))
        x, y = np.vstack(xs), np.vstack(ys)
        groups, w = np.concatenate(groups), np.concatenate(weights)
        pca_mean = pca_full = pca_scale = None
        if x.shape[1] > N_FEATURES:
            app = x[:, N_FEATURES:]
            col_mean = np.nanmean(app, axis=0)
            col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
            app = np.where(np.isnan(app), col_mean, app)
            pca_mean = app.mean(axis=0)
            # Standardise columns first, so a few large-valued outputs cannot dominate the PCA.
            pca_scale = np.maximum(app.std(axis=0), 1e-3 * (app.std(axis=0).max() + 1e-12))
            _, _, vt = np.linalg.svd((app - pca_mean) / pca_scale, full_matrices=False)
            pca_full = vt[: min(N_PCA, len(app) - 1)]
            x = np.hstack([x[:, :N_FEATURES], app])
        n_distinct = len(points) + (len(self.extra) // 10)
        # Candidate amounts of appearance information; cross-validation decides.
        ks = [0] if pca_full is None else sorted({0, *(k for k in (4, 8, 16, N_PCA) if k <= len(pca_full)
                                                        and k <= max(4, n_distinct // 2))})

        best = None
        for k in ks:
            comp = pca_full[:k] if k else None
            feats = _project(x, pca_mean if k else None, comp, pca_scale)
            mean = feats.mean(axis=0)
            floor = np.concatenate([STD_FLOOR, np.full(feats.shape[1] - N_FEATURES, PCA_STD_FLOOR)])
            std = np.maximum(feats.std(axis=0), floor[: feats.shape[1]])
            z = (feats - mean) / std
            quadratic = len(points) >= 8 or len(self.extra) >= 100
            # Leave-one-point-out cross-validation picks k and the ridge strength.
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
                if best is None or err < best[0] - 1e-9:
                    best = (err, ridge, results, k, mean, std, z, quadratic)
        err, ridge, results, k, mean, std, z, quadratic = best
        coef = _solve(_design(z, quadratic), y, ridge, w)
        pca_comp = pca_full[:k] if k else None
        if not k:
            pca_mean = pca_scale = None
        return GazeModel(mean=mean, std=std, coef=coef, quadratic=quadratic, eye=self.eye,
                         fit_error=err, ridge=ridge, point_results=results, pca_mean=pca_mean, pca_comp=pca_comp,
                         pca_scale=pca_scale)
