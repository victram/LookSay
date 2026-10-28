import numpy as np
import pytest

from looksay.calibration.model import CalibrationSession, GazeModel, calibration_targets
from looksay.tracking.blink import BlinkDetector
from looksay.tracking.features import (LEFT_EYE, RIGHT_EYE, FaceFeatures, compute_features,
                                       estimate_confidence)
from looksay.tracking.gaze import MAX_DRIFT, GazeEstimator
from looksay.tracking.smoothing import OneEuroFilter


def synthetic_landmarks(iris_dx=0.0, iris_dy=0.0, eye_open=0.012):
    """478 normalised points with a plausible face/eye geometry."""
    pts = np.full((478, 3), 0.5)
    pts[234, :2], pts[454, :2] = (0.35, 0.5), (0.65, 0.5)  # cheeks
    pts[10, :2], pts[152, :2] = (0.5, 0.3), (0.5, 0.7)  # forehead, chin
    pts[1, :2] = (0.5, 0.52)  # nose
    for eye, (ox, ix) in ((RIGHT_EYE, (0.40, 0.46)), (LEFT_EYE, (0.60, 0.54))):
        pts[eye["outer"], :2] = (ox, 0.45)
        pts[eye["inner"], :2] = (ix, 0.45)
        pts[eye["upper"], :2] = ((ox + ix) / 2, 0.45 - eye_open)
        pts[eye["lower"], :2] = ((ox + ix) / 2, 0.45 + eye_open)
        pts[eye["iris"], :2] = ((ox + ix) / 2 + iris_dx, 0.45 + iris_dy)
    return pts


def test_features_follow_iris_movement():
    centre = compute_features(synthetic_landmarks(), 640, 480, 0.0)
    right = compute_features(synthetic_landmarks(iris_dx=0.01), 640, 480, 0.0)
    down = compute_features(synthetic_landmarks(iris_dy=0.005), 640, 480, 0.0)
    assert centre.face_found and centre.confidence > 0.5
    assert right.vector()[0] != pytest.approx(centre.vector()[0])
    # Both eyes move the same way on screen -> mirrored 'h' stays consistent per eye.
    assert right.right.h > centre.right.h and right.left.h < centre.left.h
    assert down.vector()[1] > centre.vector()[1]


def test_eye_selection_vectors_differ():
    f = compute_features(synthetic_landmarks(iris_dx=0.01), 640, 480, 0.0)
    assert f.vector("left")[0] != f.vector("right")[0]


def test_closed_eyes_detected_and_lower_confidence():
    f = compute_features(synthetic_landmarks(eye_open=0.001), 640, 480, 0.0)
    assert f.eyes_closed()
    assert estimate_confidence(f) == 0.0


def test_missing_face():
    f = compute_features(None, 640, 480, 1.0)
    assert not f.face_found and f.vector() is None and f.confidence == 0.0


def _synthetic_session(n_points, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    targets = calibration_targets(n_points)
    session = CalibrationSession(targets)
    for i, (tx, ty) in enumerate(targets):
        for _ in range(20):
            h = 0.4 + 0.2 * tx + rng.normal(0, noise)
            v = -0.1 + 0.1 * ty + rng.normal(0, noise / 2)
            session.add_sample(i, np.array([h, v, rng.normal(0, 0.003), rng.normal(0, 0.003)]))
    return session


@pytest.mark.parametrize("n", [5, 9])
def test_calibration_fit_recovers_mapping(n):
    model = _synthetic_session(n).fit()
    assert model is not None
    assert model.fit_error < 0.03
    p = model.predict(np.array([0.5, -0.05, 0.0, 0.0]))[0]
    assert p == pytest.approx([0.5, 0.5], abs=0.05)


def test_calibration_rejects_outliers():
    session = _synthetic_session(9)
    for _ in range(3):
        session.add_sample(0, np.array([5.0, 5.0, 0.0, 0.0]))
    assert session.fit().fit_error < 0.03


def test_calibration_needs_enough_points():
    session = CalibrationSession(calibration_targets(9))
    session.add_sample(0, np.zeros(4))
    assert session.fit() is None


def test_model_serialisation_roundtrip():
    model = _synthetic_session(9).fit()
    clone = GazeModel.from_dict(model.to_dict())
    x = np.array([0.45, -0.07, 0.0, 0.0])
    assert np.allclose(model.predict(x), clone.predict(x))
    with pytest.raises(ValueError):
        GazeModel.from_dict({**model.to_dict(), "version": 99})


def test_gaze_estimator_states_and_drift():
    est = GazeEstimator()
    f = compute_features(synthetic_landmarks(), 640, 480, 0.0)
    assert est.update(f).state == "uncalibrated"
    est.set_model(_synthetic_session(9).fit())
    est.set_smoothing(0)
    s = est.update(compute_features(synthetic_landmarks(), 640, 480, 0.1))
    assert s.state == "ok"
    closed = est.update(compute_features(synthetic_landmarks(eye_open=0.001), 640, 480, 0.2))
    assert closed.state == "blink" and (closed.x, closed.y) == (s.x, s.y)
    assert est.update(compute_features(None, 640, 480, 0.3)).state == "no_face"
    est.recenter((0.5, 0.5), np.array([0.9, 0.9]))
    assert np.all(np.abs(est.offset) <= MAX_DRIFT)


def test_one_euro_filter_smooths_noise():
    rng = np.random.default_rng(1)
    f = OneEuroFilter.from_strength(8)
    outs = [f(np.array([0.5, 0.5]) + rng.normal(0, 0.02, 2), i / 30) for i in range(120)]
    assert np.std(np.array(outs[30:]), axis=0).max() < 0.01


def test_blink_detector_only_counts_deliberate_blinks():
    b = BlinkDetector(min_s=0.4, max_s=1.5)
    b.update(True, 0.0)
    assert not b.update(False, 0.15)  # natural blink
    b.update(True, 1.0)
    assert b.update(False, 1.6)  # deliberate
    b.update(True, 2.0)
    assert not b.update(False, 5.0)  # resting eyes


def test_real_face_landmarks_if_available():
    """Runs the bundled MediaPipe model on a real portrait shipped with matplotlib."""
    cv2 = pytest.importorskip("cv2")
    mpl = pytest.importorskip("matplotlib")
    from pathlib import Path

    from looksay.tracking.landmarks import LandmarkDetector, TrackerUnavailable

    path = Path(mpl.get_data_path()) / "sample_data" / "grace_hopper.jpg"
    if not path.exists():
        pytest.skip("sample image not available")
    try:
        det = LandmarkDetector()
    except TrackerUnavailable as exc:
        pytest.skip(str(exc))
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    pts, ms = det.detect(rgb, 0.0)
    det.close()
    assert pts is not None and pts.shape == (478, 3)
    f = compute_features(pts, rgb.shape[1], rgb.shape[0], 0.0, ms)
    assert isinstance(f, FaceFeatures) and f.face_found and f.confidence > 0.5
    assert 0.3 < f.right.h < 0.7 and 0.3 < f.left.h < 0.7
