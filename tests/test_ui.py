"""End-to-end UI tests driven only by simulated gaze (no mouse clicks).

A stub calibration model maps the right-eye feature (h, v) straight to a
normalised screen point, so "looking at" a button means emitting features
whose (h, v) is that button's centre.
"""

import time

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")

from looksay.prediction.engine import Predictor  # noqa: E402
from looksay.speech.tts import NullSpeech  # noqa: E402
from looksay.storage.database import Database  # noqa: E402
from looksay.storage.settings import Settings  # noqa: E402
from looksay.tracking.features import EyeFeatures, FaceFeatures  # noqa: E402
from tests.conftest import DICTS  # noqa: E402


class IdentityModel:
    version = 2
    fit_error = 0.0
    ridge = 0.01
    quadratic = False
    point_results: list = []

    def error_px(self, width, height):
        return 0.0

    def predict(self, x):
        x = np.atleast_2d(x)
        return x[:, :2].copy()


def features(x, y, face=True, eyes_open=True):
    eye = EyeFeatures(h=x, v=y, openness=0.3 if eyes_open else 0.02)
    return FaceFeatures(timestamp=time.monotonic(), face_found=face, right=eye if face else None,
                        left=eye if face else None, face_size=0.3, confidence=1.0 if face else 0.0)


@pytest.fixture
def window(qapp, tmp_path):
    from looksay.ui.main_window import MainWindow

    db = Database(tmp_path / "ui.sqlite3")
    settings = Settings(db)
    for key, value in {"input_source": "gaze", "dwell_ms": 400, "cooldown_ms": 700, "tracking_eye": "right",
                       "smoothing": 0, "auto_drift": False, "pointer_mode": "eyes"}.items():
        settings.set(key, value)
    w = MainWindow(settings, db, Predictor(DICTS, store=db), NullSpeech(), enable_tracking=False)
    w.resize(1200, 800)
    w.show()
    qapp.processEvents()
    w.timer.stop()  # the test drives ticks itself
    w.estimator.set_model(IdentityModel())
    w.enable_tracking = True  # features are injected by the test instead of a camera thread
    yield w
    w.close()
    db.close()


def gaze(w, qapp, x, y, seconds, **kw):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        w._on_features(features(x, y, **kw))
        w._tick()
        qapp.processEvents()
        time.sleep(0.015)


def look_at(w, qapp, button, seconds=1.0):
    c = button.mapTo(w, button.rect().center())
    gaze(w, qapp, c.x() / w.width(), c.y() / w.height(), seconds)
    gaze(w, qapp, 0.5, 0.02, 0.2)  # look away at the canvas (no button)


def key(w, value):
    return next(b for b in w.keyboard._letter_buttons if b.property("base") == value)


def test_type_accept_suggestion_and_speak_by_gaze(window, qapp):
    w = window
    look_at(w, qapp, key(w, "h"))
    look_at(w, qapp, key(w, "e"))
    assert w.buffer.text == "He"
    hello = next(b for b in w.keyboard.suggestion_buttons if b.text().lower() == "hello")
    look_at(w, qapp, hello)
    assert w.buffer.text == "Hello "
    look_at(w, qapp, w.keyboard.backspace_btn)
    assert w.buffer.text == "Hello"
    look_at(w, qapp, w.keyboard.speak_btn)
    assert w.speech.spoken == ["Hello"]


def test_clear_requires_gaze_confirmation(window, qapp):
    w = window
    w.buffer.text = "Some text"
    look_at(w, qapp, w.keyboard.clear_btn)
    assert w.confirm.isVisible() and w.buffer.text == "Some text"
    # Keys under the overlay must not be selectable.
    look_at(w, qapp, w.keyboard.space_btn)
    assert w.buffer.text == "Some text"
    look_at(w, qapp, w.confirm.no_btn)
    assert not w.confirm.isVisible() and w.buffer.text == "Some text"
    look_at(w, qapp, w.keyboard.clear_btn)
    look_at(w, qapp, w.confirm.yes_btn)
    assert w.buffer.text == ""


def test_face_loss_pauses_selection(window, qapp):
    w = window
    c = key(w, "a").mapTo(w, key(w, "a").rect().center())
    gaze(w, qapp, c.x() / w.width(), c.y() / w.height(), 0.2)
    gaze(w, qapp, 0, 0, 0.5, face=False)
    assert w.buffer.text == ""
    assert w.status_label.text().startswith("Face not found")


def test_blink_holds_cursor_and_does_not_select(window, qapp):
    w = window
    b = key(w, "a")
    c = b.mapTo(w, b.rect().center())
    gaze(w, qapp, c.x() / w.width(), c.y() / w.height(), 0.15)
    gaze(w, qapp, 0.9, 0.9, 0.5, eyes_open=False)  # eyes closed: features ignored, progress frozen
    assert w.buffer.text == ""
    assert w.gaze_sample.state == "blink"


def test_pause_only_allows_resume(window, qapp):
    w = window
    look_at(w, qapp, w.pause_btn)
    assert w.paused
    look_at(w, qapp, key(w, "a"))
    assert w.buffer.text == ""
    look_at(w, qapp, w.pause_btn)
    assert not w.paused


def test_language_switch_keeps_text_and_changes_layout(window, qapp):
    w = window
    w.buffer.text = "Hello "
    look_at(w, qapp, w.keyboard.language_btn)
    assert w.lang == "es" and w.buffer.text == "Hello "
    assert any(b.property("base") == "ñ" for b in w.keyboard._letter_buttons)
    assert w.keyboard.speak_btn.text().endswith("Hablar")
    look_at(w, qapp, w.keyboard.language_btn)
    assert w.lang == "ca"
    assert any(b.property("base") == "ç" for b in w.keyboard._letter_buttons)


def test_word_mode_and_phrases(window, qapp):
    w = window
    look_at(w, qapp, w.keyboard.mode_btn)
    assert w.keyboard.mode == "words"
    word = w.keyboard.word_buttons[0]
    text = word.text()
    look_at(w, qapp, word)
    assert w.buffer.text.strip().lower() == text.lower()
    look_at(w, qapp, w.keyboard.phrases_btn)
    assert w.pages.currentWidget() is w.phrases_page
    first = w.phrases_page.phrase_buttons[0]
    phrase = first.text()
    look_at(w, qapp, first)
    assert phrase in w.buffer.text and w.speech.spoken[-1] == phrase
    assert w.pages.currentWidget() is w.keyboard


def test_help_key_speaks_emergency_phrase(window, qapp):
    w = window
    look_at(w, qapp, w.keyboard.help_btn)
    assert w.speech.spoken[-1] == "I need help"


def test_settings_change_by_gaze_persists(window, qapp, tmp_path):
    w = window
    look_at(w, qapp, w.keyboard.settings_btn)
    page = w.settings_page
    look_at(w, qapp, page.group_buttons["selection"])
    row = next(r for r in page.rows if r[0].text().startswith("Dwell"))
    look_at(w, qapp, row[3])  # "+"
    assert w.settings["dwell_ms"] == 500 and w.dwell.dwell_s == pytest.approx(0.5)
    look_at(w, qapp, page.group_buttons["display"])
    theme_row = next(r for r in page.rows if r[0].text() == "Theme")
    look_at(w, qapp, theme_row[3])
    assert w.settings["theme"] == "night"
    reloaded = Settings(w.db)
    assert reloaded["dwell_ms"] == 500 and reloaded["theme"] == "night"


def test_calibration_overlay_fits_model(qapp, monkeypatch):
    from PySide6.QtWidgets import QWidget

    import looksay.ui.calibration_view as cv

    class Clock:
        t = 0.0

        def monotonic(self):
            return self.t

    clock = Clock()
    monkeypatch.setattr(cv, "time", clock)
    parent = QWidget()
    parent.resize(1000, 600)
    results = []
    overlay = cv.CalibrationOverlay(parent, raw_point=lambda f: None)
    overlay.finished.connect(results.append)
    overlay.start(9, "right")
    rng = np.random.default_rng(0)
    for _ in range(2000):
        clock.t += 0.033
        if overlay.state == "collect":
            tx, ty = overlay.targets[overlay.index]
            f = features(0.4 + 0.2 * tx + rng.normal(0, 0.002), -0.1 + 0.1 * ty + rng.normal(0, 0.001))
            overlay.feed(f)
        overlay._tick()
        if overlay.state in ("done", "failed"):
            break
    assert overlay.state == "done"
    overlay.accept_btn.click()
    assert results and results[0] is not None and results[0].fit_error < 0.02


def test_camera_preview_is_optional_in_main_view(window, qapp):
    from PySide6.QtGui import QImage

    w = window
    image = QImage(320, 240, QImage.Format.Format_RGB888)
    image.fill(0)
    w._on_preview(image)
    assert not w.preview_label.isVisible()  # off by default
    w.settings.set("camera_preview", True)
    w._on_preview(image)
    assert w.preview_label.isVisible()
    w.settings.set("camera_preview", False)
    assert not w.preview_label.isVisible()


def test_first_use_starts_calibration_when_face_appears(window, qapp):
    from types import SimpleNamespace

    from looksay.diagnostics.stats import TrackingStats

    w = window
    w.estimator.set_model(None)
    w.worker = SimpleNamespace(stats=TrackingStats(), stop=lambda: None)
    w._on_features(features(0.5, 0.5, face=False))
    assert not w.calibration.isVisible()
    w._on_features(features(0.5, 0.5))
    assert w.calibration.isVisible() and w.calibration.state == "intro"
    w.calibration._cancel()
    w._on_features(features(0.5, 0.5))
    assert not w.calibration.isVisible()  # only once per session


def test_k_key_toggles_debug_overlay_with_live_info(window, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    w = window
    w._on_features(features(0.5, 0.5))
    QTest.keyClick(w, Qt.Key.Key_K)
    assert w.debug_overlay.isVisible()
    assert w.debug_overlay.geometry() == w.rect()
    w._update_debug_info()
    text = "\n".join(t for t, _ in w.debug_overlay.lines)
    assert "FACE" in text and "IRIS" in text and "GAZE" in text and "DWELL" in text
    # The overlay ignores input: keys below still work by gaze.
    look_at(w, qapp, key(w, "a"))
    assert w.buffer.text == "A"
    QTest.keyClick(w, Qt.Key.Key_K)
    assert not w.debug_overlay.isVisible()


def test_nose_calibration_overlay_fits_head_model(qapp, monkeypatch):
    from PySide6.QtWidgets import QWidget

    import looksay.ui.calibration_view as cv
    from looksay.calibration.head_model import HeadModel

    class Clock:
        t = 0.0

        def monotonic(self):
            return self.t

    clock = Clock()
    monkeypatch.setattr(cv, "time", clock)
    parent = QWidget()
    parent.resize(1000, 600)
    results = []
    overlay = cv.CalibrationOverlay(parent, raw_point=lambda f: None)
    overlay.finished.connect(results.append)
    overlay.start(9, "both", mode="head")
    rng = np.random.default_rng(0)
    for _ in range(3000):
        clock.t += 0.033
        if overlay.state == "collect":
            tx, ty = overlay.targets[overlay.index]
            f = features(0.5, 0.5)
            f.nose_x = -(tx - 0.5) * 0.4 + rng.normal(0, 0.002)  # head turned right -> nose to image-left
            f.nose_y = 0.3 + (ty - 0.5) * 0.3 + rng.normal(0, 0.002)
            overlay.feed(f)
        overlay._tick()
        if overlay.state in ("done", "failed"):
            break
    assert overlay.state == "done"
    overlay.accept_btn.click()
    assert isinstance(results[0], HeadModel) and results[0].fit_error < 0.02


def test_selections_refine_the_model_unless_corrected(window, qapp):
    from tests.test_tracking import _synthetic_session, _vector

    w = window
    w.settings.set("pointer_mode", "eyes_head")
    session = _synthetic_session(9)
    w.calib_session = session
    w.estimator.model = session.fit()
    rng = np.random.default_rng(0)
    now = time.monotonic()
    btn = key(w, "a")
    for _ in range(4):
        w._recent_vectors = [(now, _vector(0.2, 0.6, rng)) for _ in range(5)]
        w._queue_learning_sample(btn)
    w._on_key("backspace", "")  # a correction discards pending samples
    assert w._pending_samples == []
    model_before = w.estimator.model
    model_before.fit_error = 1.0  # pretend the current model is poor, so the refit is adopted
    for _ in range(5):
        w._recent_vectors = [(now, _vector(0.2, 0.6, rng)) for _ in range(5)]
        w._queue_learning_sample(btn)
    w._confirm_learning_sample(now + 10)
    assert len(session.extra) == 5
    assert w.estimator.model is not model_before  # refitted after 5 confirmed samples


def test_sticky_keys_keep_the_target_near_its_border(window, qapp):
    w = window
    b = key(w, "g")
    r = b.mapTo(w, b.rect().center())
    gaze(w, qapp, r.x() / w.width(), r.y() / w.height(), 0.2)  # start dwelling on G
    assert w._dwell_button is b
    edge = b.mapTo(w, b.rect().topRight())
    x = (edge.x() + b.width() * 0.15) / w.width()  # 15% past the right edge (over H)
    gaze(w, qapp, x, r.y() / w.height(), 0.6)
    assert w.buffer.text == "G"
