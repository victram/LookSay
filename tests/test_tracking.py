import numpy as np
import pytest

from looksay.calibration.model import CalibrationSession, GazeModel, calibration_targets
from looksay.tracking.blink import BlinkDetector
from looksay.tracking.features import (FEATURE_NAMES, LEFT_EYE, RIGHT_EYE, FaceFeatures, compute_features,
                                       estimate_confidence, pose_from_matrix)
from looksay.tracking.gaze import MAX_DRIFT, GazeEstimator
from looksay.tracking.smoothing import OneEuroFilter


def synthetic_landmarks(iris_dx=0.0, iris_dy=0.0, eye_open=0.012):
    """478 normalised points with a plausible face/eye geometry."""
    pts = np.full((478, 3), 0.5)
    pts[234, :2], pts[454, :2] = (0.35, 0.5), (0.65, 0.5)  # cheeks
    pts[10, :2], pts[152, :2] = (0.5, 0.3), (0.5, 0.7)  # forehead, chin
    pts[1, :2] = (0.5, 0.52)  # nose
    for eye, (xa, xb) in ((RIGHT_EYE, (0.40, 0.46)), (LEFT_EYE, (0.54, 0.60))):
        cx, cy, half = (xa + xb) / 2, 0.45, (xb - xa) / 2
        contour = eye["contour"]
        for k, idx in enumerate(contour):  # ellipse around the eye
            ang = 2 * np.pi * k / len(contour)
            pts[idx, :2] = (cx - half * np.cos(ang), cy + eye_open * np.sin(ang))
        pts[eye["a"], :2], pts[eye["b"], :2] = (xa, cy), (xb, cy)
        for up, down in eye["lids"]:
            pts[up, 1], pts[down, 1] = cy - eye_open, cy + eye_open
        ic = (cx + iris_dx, cy + iris_dy)
        iris = eye["iris"]
        pts[iris[0], :2] = ic
        for k, idx in enumerate(iris[1:]):
            ang = np.pi / 2 * k
            pts[idx, :2] = (ic[0] + 0.008 * np.cos(ang), ic[1] + 0.008 * np.sin(ang))
    return pts


def test_features_follow_iris_movement():
    centre = compute_features(synthetic_landmarks(), 640, 480, 0.0)
    right = compute_features(synthetic_landmarks(iris_dx=0.01), 640, 480, 0.0)
    down = compute_features(synthetic_landmarks(iris_dy=0.005), 640, 480, 0.0)
    assert centre.face_found and centre.confidence > 0.5
    assert right.vector()[0] != pytest.approx(centre.vector()[0])
    # Both eyes use the same image-left -> image-right axis, so both increase.
    assert right.right.h > centre.right.h and right.left.h > centre.left.h
    assert down.vector()[1] > centre.vector()[1]


def test_eye_selection_uses_one_eye():
    f = compute_features(synthetic_landmarks(iris_dx=0.01), 640, 480, 0.0)
    right, left = f.vector("right"), f.vector("left")
    assert right[0] == right[2] == pytest.approx(f.right.h)
    assert left[0] == left[2] == pytest.approx(f.left.h)
    assert len(f.vector()) == len(FEATURE_NAMES)


def test_head_pose_from_matrix_and_blendshapes():
    yaw = np.radians(10)
    m = np.eye(4)
    m[:3, :3] = [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    m[:3, 3] = [1.0, 2.0, -45.0]
    bs = {"eyeLookInRight": 0.4, "eyeLookOutLeft": 0.4, "eyeLookUpRight": 0.3, "eyeLookUpLeft": 0.3}
    f = compute_features(synthetic_landmarks(), 640, 480, 0.0, blendshapes=bs, matrix=m)
    assert f.pose_3d and f.yaw == pytest.approx(10, abs=0.01) and f.tz == -45.0
    assert f.right.look_h == pytest.approx(0.4) and f.left.look_h == pytest.approx(0.4)
    assert f.right.look_v == pytest.approx(-0.3)
    assert pose_from_matrix(np.eye(4))[:3] == (0.0, 0.0, 0.0)


def test_blink_blendshape_closes_eyes():
    f = compute_features(synthetic_landmarks(), 640, 480, 0.0,
                         blendshapes={"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.9})
    assert f.eyes_closed()


def test_iris_refinement_moves_towards_dark_pupil():
    cv2 = pytest.importorskip("cv2")
    from looksay.tracking.iris_refine import refine_iris

    w, h = 1280, 960
    pts = synthetic_landmarks(eye_open=0.02).astype(np.float32)
    gray = np.full((h, w), 200, np.uint8)
    for eye in (RIGHT_EYE, LEFT_EYE):
        cx, cy = pts[eye["iris"][0], :2] * [w, h]
        cv2.circle(gray, (int(cx + 4), int(cy)), 7, 20, -1)  # true pupil 4 px to the right
    refined, shifts = refine_iris(gray, pts)
    assert set(shifts) == {"right", "left"}
    for dx, dy in shifts.values():
        assert dx > 0.5 and abs(dy) < 1.0


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
            session.add_sample(i, _vector(tx, ty, rng, noise))
    return session


def _vector(tx, ty, rng, noise=0.002, yaw=0.0):
    h = -0.1 + 0.2 * tx + rng.normal(0, noise)
    v = -0.05 + 0.1 * ty + rng.normal(0, noise / 2)
    return np.array([h, v, h + rng.normal(0, noise), v + rng.normal(0, noise / 2), 0.3 - 0.05 * ty,
                     0.5 * h, 0.5 * v, yaw + rng.normal(0, 0.3), rng.normal(0, 0.3), rng.normal(0, 0.3),
                     rng.normal(0, 0.1), rng.normal(0, 0.1), -45 + rng.normal(0, 0.2)])


@pytest.mark.parametrize("n", [5, 9, 13])
def test_calibration_fit_recovers_mapping(n):
    model = _synthetic_session(n).fit()
    assert model is not None
    assert model.fit_error < 0.03
    assert len(model.point_results) == n
    p = model.predict(_vector(0.5, 0.5, np.random.default_rng(5), noise=0))[0]
    assert p == pytest.approx([0.5, 0.5], abs=0.05)
    assert model.error_px(1920, 1080) < 0.03 * 2203


def test_calibration_rejects_outliers():
    session = _synthetic_session(9)
    for _ in range(3):
        session.add_sample(0, np.full(13, 5.0))
    assert session.fit().fit_error < 0.03


def test_calibration_needs_enough_points():
    session = CalibrationSession(calibration_targets(9))
    session.add_sample(0, np.zeros(13))
    assert session.fit() is None


def test_model_serialisation_roundtrip():
    model = _synthetic_session(9).fit()
    clone = GazeModel.from_dict(model.to_dict())
    x = _vector(0.3, 0.7, np.random.default_rng(1))
    assert np.allclose(model.predict(x), clone.predict(x))
    assert clone.point_results == model.point_results
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


def test_fixation_stabiliser_holds_still_then_follows_saccade():
    est = GazeEstimator()
    est.set_smoothing(0)

    class Stub:
        version, fit_error = 2, 0.0

        def __init__(self):
            self.p = np.array([0.5, 0.5])

        def predict(self, x):
            return self.p[None, :]

    stub = Stub()
    est.model = stub
    rng = np.random.default_rng(0)
    f = compute_features(synthetic_landmarks(), 640, 480, 0.0)
    outs = []
    for i in range(30):
        stub.p = np.array([0.5, 0.5]) + rng.normal(0, 0.006, 2)
        f.timestamp = i / 30
        outs.append(est.update(f))
    assert est.fixation_locked
    jitter = np.std([[s.x, s.y] for s in outs[15:]], axis=0).max()
    assert jitter < 0.004  # raw jitter is 0.006
    stub.p = np.array([0.9, 0.2])  # saccade
    for i in range(30, 40):
        f.timestamp = i / 30
        s = est.update(f)
    assert abs(s.x - 0.9) < 0.05 and abs(s.y - 0.2) < 0.05


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
    d = det.detect(rgb, 0.0)
    det.close()
    assert d.points is not None and d.points.shape == (478, 3)
    assert d.matrix is not None and "eyeLookInLeft" in d.blendshapes
    f = compute_features(d.points, rgb.shape[1], rgb.shape[0], 0.0, d.inference_ms, d.blendshapes, d.matrix)
    assert isinstance(f, FaceFeatures) and f.face_found and f.pose_3d and f.confidence > 0.5
    assert abs(f.right.h) < 0.15 and abs(f.left.h) < 0.15
    assert 20 < abs(f.tz) < 120  # a plausible camera distance in cm


class _ConstModel:
    """Eye model stub returning a settable point."""
    version, fit_error = 2, 0.0

    def __init__(self):
        self.p = np.array([0.5, 0.5])

    def predict(self, x):
        return self.p[None, :]


def _face(t, nose_x=0.0, nose_y=0.3):
    f = compute_features(synthetic_landmarks(), 640, 480, t)
    f.nose_x, f.nose_y = nose_x, nose_y
    return f


def test_eyes_plus_nose_ignores_eye_jitter_and_fine_tunes_with_head():
    est = GazeEstimator()
    est.set_smoothing(0)
    stub = _ConstModel()
    est.model = stub
    assert est.mode == "eyes_head" and est.ready
    rng = np.random.default_rng(0)
    xs = []
    for i in range(30):  # fixating with noisy eyes, head still
        stub.p = np.array([0.5, 0.5]) + rng.normal(0, 0.02, 2)
        s = est.update(_face(i / 30))
        xs.append((s.x, s.y))
    assert np.std(np.array(xs[5:]), axis=0).max() < 0.005  # eye jitter does not move the cursor
    start = np.array(xs[-1])
    for i in range(30, 60):  # small head turn to the user's right -> cursor moves right
        stub.p = np.array([0.5, 0.5]) + rng.normal(0, 0.02, 2)
        s = est.update(_face(i / 30, nose_x=-0.02))
    assert 0.01 < s.x - start[0] < 0.06 and abs(s.y - start[1]) < 0.01
    for i in range(60, 75):  # gaze jumps far away -> cursor follows the eyes
        stub.p = np.array([0.15, 0.85])
        s = est.update(_face(i / 30, nose_x=-0.02))
    assert abs(s.x - 0.15) < 0.03 and abs(s.y - 0.85) < 0.03


def test_nose_mode_needs_head_calibration_and_follows_head():
    from looksay.calibration.head_model import fit_head_model

    est = GazeEstimator()
    est.set_mode("head")
    assert not est.ready
    assert est.update(_face(0.0)).state == "uncalibrated"
    targets = calibration_targets(5, margin=0.12)
    rng = np.random.default_rng(1)
    samples = {i: [np.array([-(tx - 0.5) * 0.4, 0.3 + (ty - 0.5) * 0.3, 0, 0]) + rng.normal(0, 0.002, 4)
                   for _ in range(15)] for i, (tx, ty) in enumerate(targets)}
    model = fit_head_model(targets, samples)
    assert model is not None and model.fit_error < 0.03
    est.set_head_model(model)
    est.set_smoothing(0)
    for i in range(20):
        s = est.update(_face(i / 30, nose_x=-0.12, nose_y=0.3 + 0.1))  # looking right and down
    assert s.state == "ok" and s.x > 0.7 and s.y > 0.7
    # eyes closed does not freeze the nose pointer
    closed = compute_features(synthetic_landmarks(eye_open=0.001), 640, 480, 2.0)
    assert est.update(closed).state != "blink"


def test_head_vector_from_landmarks_moves_with_nose():
    pts = synthetic_landmarks()
    f0 = compute_features(pts, 640, 480, 0.0)
    pts2 = pts.copy()
    pts2[1, 0] -= 0.02  # nose moves to image-left
    f1 = compute_features(pts2, 640, 480, 0.0)
    assert f1.nose_x < f0.nose_x
    assert f0.head_vector().shape == (4,)


def test_cursor_animator_moves_smoothly_between_camera_frames():
    from looksay.tracking.smoothing import CursorAnimator

    anim = CursorAnimator(tau_s=0.05)
    anim.step(np.array([0.0, 0.0]), 0.0)
    xs = [anim.step(np.array([1.0, 0.0]), i / 60)[0] for i in range(1, 31)]  # target jumps once
    steps = np.diff([0.0] + xs)
    assert all(d > 0 for d in steps)  # moves every UI frame, never backwards
    assert max(steps) < 0.3  # no single big jump
    assert xs[-1] > 0.99  # arrives within half a second


def test_pursuit_alignment_finds_the_lag_and_improves_the_model():
    from looksay.calibration.model import PURSUIT_WEIGHT, align_pursuit, pursuit_path

    rng = np.random.default_rng(3)

    def curved(tx, ty):  # real eye -> screen mappings are curved towards the edges
        v = _vector(tx, ty, rng)
        bend = 0.04 * (tx - 0.5) ** 2 + 0.03 * (ty - 0.5) ** 2
        v[[0, 2]] += bend
        v[[1, 3]] += 0.03 * (tx - 0.5) * (ty - 0.5)
        return v

    duration, true_lag = 14.0, 0.165
    samples = [(i / 30, curved(*pursuit_path(i / 30 - true_lag, duration))) for i in range(int(duration * 30))]
    pairs, lag = align_pursuit(samples, duration)
    assert abs(lag - true_lag) <= 0.034
    assert len(pairs) > 300
    session = CalibrationSession(calibration_targets(5))
    for i, (tx, ty) in enumerate(session.targets):
        for _ in range(20):
            session.add_sample(i, curved(tx, ty))
    before = session.fit()
    for vec, target in pairs:
        session.add_extra(vec, target, PURSUIT_WEIGHT)
    after = session.fit()
    assert not before.quadratic and after.quadratic
    assert after.fit_error < before.fit_error


def test_session_serialisation_roundtrip_with_extras():
    session = _synthetic_session(9)
    session.add_extra(_vector(0.3, 0.3, np.random.default_rng(0)), (0.3, 0.3), 1.0)
    clone = CalibrationSession.from_dict(session.to_dict())
    assert len(clone.extra) == 1 and set(clone.samples) == set(session.samples)
    assert clone.fit().fit_error == pytest.approx(session.fit().fit_error, rel=1e-3)


def _rendered_patch(tx, ty, rng):
    """A tiny eye image whose dark iris position depends on where the user looks."""
    from looksay.tracking.eye_patch import PATCH_H, PATCH_W

    yy, xx = np.mgrid[0:PATCH_H, 0:PATCH_W]
    cx, cy = PATCH_W / 2 + (tx - 0.5) * 8, PATCH_H / 2 + (ty - 0.5) * 4
    img = 200 - 150 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 6.0)
    img = img - 60 * (yy < 2 + 3 * (1 - ty))  # upper lid lower when looking down
    img = img + rng.normal(0, 6, img.shape)
    return ((img - img.mean()) / img.std()).ravel()


def test_eye_image_features_help_when_there_is_enough_data_and_never_hurt():
    rng = np.random.default_rng(7)
    targets = calibration_targets(13)
    geo_only, with_patches = CalibrationSession(targets), CalibrationSession(targets)
    for i, (tx, ty) in enumerate(targets):
        for _ in range(25):
            geo = _vector(tx, ty, rng, noise=0.03)  # very noisy landmarks
            geo_only.add_sample(i, geo)
            p = _rendered_patch(tx, ty, rng)
            with_patches.add_sample(i, np.concatenate([geo, p, p]))
    # 13 points only: CV chooses how much appearance to use, so it is never worse.
    m_geo, m_13 = geo_only.fit(), with_patches.fit()
    assert m_13.fit_error <= m_geo.fit_error + 1e-9
    # With pursuit-like data (hundreds of distinct targets) the eye images win clearly.
    for _ in range(500):
        tx, ty = rng.uniform(0.05, 0.95, 2)
        geo = _vector(tx, ty, rng, noise=0.03)
        p = _rendered_patch(tx, ty, rng)
        geo_only.add_extra(geo, (tx, ty), 0.5)
        with_patches.add_extra(np.concatenate([geo, p, p]), (tx, ty), 0.5)
    m_geo, m_app = geo_only.fit(), with_patches.fit()
    assert m_app.uses_appearance and not m_geo.uses_appearance
    assert m_app.fit_error < m_geo.fit_error * 0.7
    # serialisation keeps the PCA basis; a frame without patches still predicts
    clone = GazeModel.from_dict(m_app.to_dict())
    x = np.concatenate([_vector(0.3, 0.7, rng, noise=0), _rendered_patch(0.3, 0.7, rng),
                        _rendered_patch(0.3, 0.7, rng)])
    assert np.allclose(clone.predict(x), m_app.predict(x))
    assert clone.predict(x[:13]).shape == (1, 2)
    # compact session storage round-trips
    restored = CalibrationSession.from_dict(with_patches.to_dict())
    assert restored.fit().fit_error == pytest.approx(m_app.fit_error, rel=0.1)


def test_eye_patch_follows_the_iris():
    cv2 = pytest.importorskip("cv2")
    from looksay.tracking.eye_patch import PATCH_H, PATCH_W, eye_patch

    def darkest_x(shift):
        gray = np.full((200, 300), 190, np.uint8)
        cv2.circle(gray, (150 + shift, 100), 9, 30, -1)
        p = eye_patch(gray, np.array([120.0, 100.0]), np.array([180.0, 100.0]))
        assert p.shape == (PATCH_H, PATCH_W)
        return np.unravel_index(np.argmin(p), p.shape)[1]

    assert darkest_x(-8) < darkest_x(0) < darkest_x(8)


def test_real_face_eye_patches():
    cv2 = pytest.importorskip("cv2")
    mpl = pytest.importorskip("matplotlib")
    from pathlib import Path

    from looksay.tracking.eye_patch import eye_patches
    from looksay.tracking.landmarks import LandmarkDetector, TrackerUnavailable

    path = Path(mpl.get_data_path()) / "sample_data" / "grace_hopper.jpg"
    if not path.exists():
        pytest.skip("sample image not available")
    try:
        det = LandmarkDetector()
    except TrackerUnavailable as exc:
        pytest.skip(str(exc))
    img = cv2.imread(str(path))
    d = det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.0)
    det.close()
    patches = eye_patches(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), d.points)
    assert patches is not None and patches.shape == (2, 12, 24)
    f = compute_features(d.points, img.shape[1], img.shape[0], 0.0, 0.0, d.blendshapes, d.matrix, patches)
    assert f.vector().shape == (13 + 2 * 12 * 24,)


def test_gaze_cnn_on_real_face():
    """The bundled gaze CNN gives 258 finite outputs on a real portrait, from sane crops."""
    cv2 = pytest.importorskip("cv2")
    mpl = pytest.importorskip("matplotlib")
    pytest.importorskip("MNN")
    from pathlib import Path

    from looksay.tracking.cnn_gaze import CNN_FEATURES, CnnGaze, crop_rects
    from looksay.tracking.landmarks import LandmarkDetector, TrackerUnavailable

    path = Path(mpl.get_data_path()) / "sample_data" / "grace_hopper.jpg"
    if not path.exists():
        pytest.skip("sample image not available")
    try:
        det = LandmarkDetector()
    except TrackerUnavailable as exc:
        pytest.skip(str(exc))
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    points = det.detect(rgb, 0.0).points
    det.close()
    face, left, right = crop_rects(points[:, :2].astype(float) * (640, 480), 640, 480)
    assert face[2] > 100 and face[3] > 100
    assert left[0] + left[2] < right[0] + right[2] and left[2] > 10 and right[2] > 10
    assert left[1] > face[1] and right[1] > face[1]
    cnn = CnnGaze()
    out = cnn.features(rgb, points)
    assert out is not None and out.shape == (CNN_FEATURES,) and np.all(np.isfinite(out))
    assert np.allclose(cnn.features(rgb, points), out, atol=1e-3)  # deterministic
    f = compute_features(points, rgb.shape[1], rgb.shape[0], 0.0, cnn=out)
    assert f.vector().shape == (13 + CNN_FEATURES,)


def test_cnn_features_mixed_with_other_frames():
    """CNN-width frames dominate; a few frames with patches or nothing extra must not break the fit."""
    rng = np.random.default_rng(3)
    targets = calibration_targets(9)
    session = CalibrationSession(targets)
    basis = rng.normal(size=(2, 258))
    for i, (tx, ty) in enumerate(targets):
        for j in range(20):
            geo = _vector(tx, ty, rng, noise=0.03)
            deep = np.array([tx - 0.5, ty - 0.5]) @ basis + rng.normal(scale=0.05, size=258)
            if j == 0:
                session.add_sample(i, geo)  # CNN missing on this frame
            elif j == 1:
                session.add_sample(i, np.concatenate([geo, rng.random(576)]))  # patches instead
            else:
                session.add_sample(i, np.concatenate([geo, deep]))
    model = session.fit()
    assert model is not None and model.uses_appearance and len(model.pca_mean) == 258
    assert model.predict(np.concatenate([_vector(0.5, 0.5, rng, noise=0), rng.random(576)])).shape == (1, 2)
    clone = GazeModel.from_dict(model.to_dict())
    x = np.concatenate([_vector(0.2, 0.8, rng, noise=0), np.array([-0.3, 0.3]) @ basis])
    assert np.allclose(clone.predict(x), model.predict(x))
