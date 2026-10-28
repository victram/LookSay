"""Main window: wires text editing, prediction, speech, tracking and dwell selection."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget

from looksay.calibration.head_model import HeadModel
from looksay.calibration.model import CalibrationSession, GazeModel
from looksay.diagnostics.stats import RateCounter
from looksay.i18n.strings import LANGUAGES, tr
from looksay.input.dwell import DwellSelector
from looksay.input.text_buffer import TextBuffer
from looksay.paths import exports_dir
from looksay.prediction.engine import Predictor
from looksay.speech.tts import NullSpeech, SpeechEngine
from looksay.storage.database import Database
from looksay.storage.settings import Settings
from looksay.tracking.blink import BlinkDetector
from looksay.tracking.features import FaceFeatures
from looksay.tracking.gaze import GazeEstimator, GazeSample
from looksay.tracking.smoothing import CursorAnimator
from looksay.ui import theme
from looksay.ui.calibration_view import CalibrationOverlay
from looksay.ui.debug_view import BAD, DIM, FG, OK, WARN, DebugOverlay
from looksay.ui.feedback import ClickSound
from looksay.ui.keyboard import KeyboardPanel
from looksay.ui.pages import ConfirmOverlay, DiagnosticsPage, PhrasesPage, SettingsPage
from looksay.ui.widgets import GazeButton, GazeCursor, MessageCanvas

log = logging.getLogger(__name__)

TICK_MS = 16  # 60 Hz UI loop: the cursor is animated between camera frames
MOUSE_PRIORITY_S = 1.5  # in "both" mode, mouse movement overrides gaze for this long
STALE_FEATURES_S = 0.6
MESSAGE_S = 5.0
WORD_GRID_SIZE = 15
# Online refinement: a gaze selection becomes a calibration sample if it is
# not corrected (delete/undo) within CONFIRM_S; the model is refitted every
# REFIT_EVERY confirmed samples.
CONFIRM_S = 3.0
REFIT_EVERY = 5
CORRECTION_KEYS = {"backspace", "delete_word", "undo", "clear"}
STICKY_MARGIN = 0.25  # keep the current key while the cursor is within this fraction of its size around it
PREVIEW_W = 200

CAMERA_KEYS = {"camera_index", "camera_resolution", "camera_fps"}


class MainWindow(QWidget):
    def __init__(self, settings: Settings, db: Database, predictor: Predictor,
                 speech: SpeechEngine | None = None, enable_tracking: bool = True):
        super().__init__()
        self.setObjectName("Root")
        self.setWindowTitle("LookSay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setMouseTracking(True)
        self.settings = settings
        self.db = db
        self.predictor = predictor
        self.speech = speech or NullSpeech()
        self.enable_tracking = enable_tracking
        self.buffer = TextBuffer(on_word_committed=self._word_committed)
        self.dwell = DwellSelector()
        self.estimator = GazeEstimator()
        self.blink = BlinkDetector()
        self.cursor_animator = CursorAnimator()
        self.calib_session: CalibrationSession | None = None
        self._recent_vectors: list[tuple[float, np.ndarray]] = []
        self._pending_samples: list[tuple[float, np.ndarray, tuple[float, float]]] = []
        self._confirmed_samples = 0
        self.click_sound: ClickSound | None = None
        self.ui_fps = RateCounter()
        self.worker = None
        self.tracker_error = ""
        self.paused = False
        self.gaze_sample: GazeSample | None = None
        self.last_features: FaceFeatures | None = None
        self._mouse_pos: QPoint | None = None
        self._mouse_moved_at = -1e9
        self._dwell_button: GazeButton | None = None
        self._message = ""
        self._message_until = 0.0
        self._model_before_calibration: GazeModel | None = None
        self._head_before_calibration: HeadModel | None = None
        self._last_source = "mouse"
        self._auto_calibrated = False

        self._build()
        self._load_calibration()
        self._apply_all_settings()
        settings.subscribe(self._on_setting)

        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.diag_timer = QTimer(self)
        self.diag_timer.setInterval(500)
        self.diag_timer.timeout.connect(self._refresh_diagnostics)
        self.diag_timer.start()

        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key.Key_K), self, activated=self.toggle_debug,
                  context=Qt.ShortcutContext.ApplicationShortcut)
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)

        if db.recovered_from_corruption:
            self.show_message(tr("db_recovered", self.lang))
        self._restart_tracking()
        self.refresh()

    # ================================================================ build
    @property
    def lang(self) -> str:
        return self.settings["language"]

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.canvas = MessageCanvas()
        root.addWidget(self.canvas, 40)
        self.preview_label = QLabel(self.canvas)  # optional camera preview, off by default
        self.preview_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.preview_label.hide()

        strip = QWidget()
        strip.setObjectName("Page")
        strip_layout = QHBoxLayout(strip)
        strip_layout.setContentsMargins(16, 4, 16, 0)
        self.status_label = QLabel("")
        self.status_label.setObjectName("Status")
        self.message_label = QLabel("")
        self.message_label.setObjectName("Title")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pause_btn = GazeButton("", "special")
        self.pause_btn.setMinimumSize(180, 52)
        self.pause_btn.setMaximumHeight(60)
        self.pause_btn.clicked.connect(self._toggle_pause)
        strip_layout.addWidget(self.status_label, 3)
        strip_layout.addWidget(self.message_label, 4)
        strip_layout.addWidget(self.pause_btn, 1)
        root.addWidget(strip, 0)

        self.pages = QStackedWidget()
        self.keyboard = KeyboardPanel()
        self.keyboard.action.connect(self._on_key)
        self.phrases_page = PhrasesPage()
        self.phrases_page.back.connect(self.show_keyboard)
        self.phrases_page.phrase_chosen.connect(self._phrase_chosen)
        self.phrases_page.save_requested.connect(self._save_phrase)
        self.phrases_page.delete_requested.connect(self._delete_phrase)
        self.settings_page = SettingsPage(self.settings.get, self._step_setting, self.speech.voices)
        self.settings_page.back.connect(self.show_keyboard)
        self.settings_page.command.connect(self._on_command)
        self.diagnostics_page = DiagnosticsPage()
        self.diagnostics_page.back.connect(self._leave_diagnostics)
        for page in (self.keyboard, self.phrases_page, self.settings_page, self.diagnostics_page):
            self.pages.addWidget(page)
        root.addWidget(self.pages, 60)

        self.confirm = ConfirmOverlay(self)
        self.calibration = CalibrationOverlay(self, self.estimator.raw_point)
        self.calibration.model_ready.connect(self._provisional_model)
        self.calibration.finished.connect(self._calibration_finished)
        self.calibration.recentered.connect(self._recentered)
        self.debug_overlay = DebugOverlay(self)
        self.gaze_cursor = GazeCursor(self)
        self.gaze_cursor.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        for overlay in (self.confirm, self.calibration, self.gaze_cursor):
            overlay.setGeometry(self.rect())
        self._place_debug_overlay()

    # ================================================================ debug
    def _place_debug_overlay(self) -> None:
        # Full window, so the camera views are big enough to judge the tracking.
        # It ignores input; the gaze cursor stays visible on top of it.
        self.debug_overlay.setGeometry(self.rect())

    def _update_debug_info(self) -> None:
        f, g, est = self.last_features, self.gaze_sample, self.estimator
        stats = self.worker.stats if self.worker else None
        lines: list[tuple[str, QColor]] = []
        eye_lines: dict[str, list[str]] = {"right": [], "left": []}
        message = ""
        if self.worker is None:
            message = self.tracker_error or "Camera not running (input = mouse, or no camera)"
        elif f is None:
            message = "Waiting for camera frames…"
        elif not f.face_found:
            message = "No face detected"
        if f is not None and f.face_found and f.right and f.left:
            conf_color = OK if f.confidence >= est.confidence_threshold else (WARN if f.confidence > 0.2 else BAD)
            lines += [
                (f"FACE    conf {f.confidence * 100:3.0f}% (min {est.confidence_threshold * 100:.0f}%)"
                 f"  size {f.face_size:.2f}", conf_color),
                (f"POSE    yaw {f.yaw:+5.1f}°  pitch {f.pitch:+5.1f}°  roll {f.roll:+5.1f}°"
                 f"   {'3D' if f.pose_3d else 'approx'}", FG),
                (f"HEAD    x {f.tx:+5.1f} y {f.ty:+5.1f} cm  distance {abs(f.tz):4.1f} cm" if f.pose_3d
                 else f"HEAD    x {f.tx:+.2f}  y {f.ty:+.2f}", FG),
                (f"IRIS    h R {f.right.h:+.3f} L {f.left.h:+.3f}   v R {f.right.v:+.3f} L {f.left.v:+.3f}", FG),
                (f"LIDS    open R {f.right.openness:.2f} L {f.left.openness:.2f}"
                 f"   blink R {f.blendshapes.get('eyeBlinkRight', 0):.2f} L {f.blendshapes.get('eyeBlinkLeft', 0):.2f}"
                 f"  {'CLOSED' if f.eyes_closed() else ''}", WARN if f.eyes_closed() else FG),
                (f"LOOK    h {(f.right.look_h + f.left.look_h) / 2:+.2f}  v {(f.right.look_v + f.left.look_v) / 2:+.2f}"
                 "   (blendshapes)" if f.blendshapes else "LOOK    n/a", FG),
            ]
            for name, e in (("right", f.right), ("left", f.left)):
                side = "Right" if name == "right" else "Left"
                bs = f.blendshapes
                shift = self.debug_overlay.frame.shifts.get(name) if self.debug_overlay.frame else None
                eye_lines[name] = [
                    f"h {e.h:+.3f}   v {e.v:+.3f}   open {e.openness:.2f}",
                    f"iris r {e.iris_radius:.1f}px  eye {e.width:.0f}px",
                    f"in {bs.get('eyeLookIn' + side, 0):.2f} out {bs.get('eyeLookOut' + side, 0):.2f} "
                    f"up {bs.get('eyeLookUp' + side, 0):.2f} dn {bs.get('eyeLookDown' + side, 0):.2f}",
                ] + ([f"refine {np.hypot(*shift):.1f}px"] if shift else [])
        hm = est.head_model
        head = est.last_head
        lines.append((f"POINTER {est.mode}   nose x {head[0]:+.3f} y {head[1]:+.3f}" if head is not None
                       else f"POINTER {est.mode}", FG))
        if est.mode != "eyes":
            lines.append((f"NOSE    {'calibrated ' + format(hm.error_px(self.width(), self.height()), '.0f') + ' px' if hm else 'default mapping'}"
                          f"   speed {est.head_gain:.1f}"
                          + (f"   anchor {est._anchor[0]:.2f},{est._anchor[1]:.2f}" if est._anchor is not None else ""),
                          FG if hm or est.mode == "eyes_head" else WARN))
        if est.ready and g is not None:
            raw = est.last_raw
            state_color = OK if g.state == "ok" else WARN
            lines += [
                (f"GAZE    {g.x * self.width():5.0f}, {g.y * self.height():4.0f} px   state {g.state}"
                 + (f"   raw {raw[0]:.3f}, {raw[1]:.3f}" if raw is not None else ""), state_color),
                (f"FILTER  1€ {self.settings['smoothing']}  sens {est.sensitivity:.1f}"
                 f"  drift {est.offset[0]:+.3f},{est.offset[1]:+.3f}"
                 f"  fixation {'LOCKED' if est.fixation_locked else 'moving'}", FG),
            ]
            m = est.model
            if m is None:
                m = hm
            err = m.error_px(self.width(), self.height())
            pct = 100 * err / float(np.hypot(self.width(), self.height()))
            lines.append((f"CALIB   {len(m.point_results)} pts  error {err:.0f} px ({pct:.1f}%)"
                          f"  {'eyes' if m is est.model else 'nose'}",
                          OK if pct < 4 else (WARN if pct < 8 else BAD)))
        else:
            lines.append(("CALIB   not calibrated (Settings > Calibrate)", WARN))
        target = self._dwell_button.text() if self._dwell_button is not None else "—"
        lines.append((f"DWELL   «{target}» {self.dwell.progress * 100:3.0f}%  of {self.dwell.dwell_s * 1000:.0f} ms"
                      f"  input {self._last_source}", FG))
        if stats is not None:
            lines.append((f"CAMERA  {'OK' if stats.camera_ok else stats.camera_error or '…'}"
                          f"  {'%dx%d' % stats.capture_size if stats.capture_size[0] else self.settings['camera_resolution']}"
                          f" {stats.capture_fps.rate:4.1f} fps {stats.camera_note}"
                          f"  track {stats.process_fps.rate:4.1f} fps  {stats.avg_inference_ms:4.1f} ms",
                          FG if stats.camera_ok else BAD))
            lines.append((f"CPU     dropped {stats.frames_dropped}/{stats.frames_captured}"
                          f" ({stats.drop_ratio * 100:.1f}%)  UI {self.ui_fps.rate:4.1f} fps"
                          f"  iris refine {'on' if self.settings['iris_refinement'] else 'off'}", DIM))
        self.debug_overlay.set_info(lines, eye_lines, message)

    def toggle_debug(self) -> None:
        """Physical K key: show/hide the tracking debug overlay."""
        visible = not self.debug_overlay.isVisible()
        self._place_debug_overlay()
        self.debug_overlay.setVisible(visible)
        if visible:
            self.debug_overlay.raise_()
            self.gaze_cursor.raise_()
            self._update_debug_info()
        self._configure_worker()

    def _toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # ============================================================= settings
    def _step_setting(self, key: str, direction: int) -> None:
        if key == "speech_voice":
            count = len(self.speech.voices())
            value = (self.settings[key] + direction) % (count + 1) if count else 0
            self.settings.set(key, value)
        else:
            self.settings.step(key, direction)

    def _apply_all_settings(self) -> None:
        for key in ("language", "theme", "text_size", "dwell_ms", "cooldown_ms", "keyboard_mode", "keyboard_layout",
                    "sensitivity", "smoothing", "confidence_threshold", "tracking_eye", "gaze_cursor_size",
                    "fixation_stabilizer", "pointer_mode", "head_gain",
                    "speech_rate", "speech_volume", "speech_voice", "learn_words", "auto_capitalize",
                    "sound_feedback"):
            self._on_setting(key, self.settings[key], initial=True)

    def _on_setting(self, key: str, value: Any, initial: bool = False) -> None:
        s = self.settings
        if key == "language":
            self.predictor.set_language(value)
            self.db.seed_phrases(value, self.predictor.default_phrases(value))
            self.speech.set_language(value)
            for page in (self.keyboard, self.phrases_page, self.settings_page, self.diagnostics_page):
                page.lang = value
            self.keyboard.set_language(value)
            self.phrases_page.retranslate()
            self.settings_page.retranslate()
            self.diagnostics_page.retranslate()
            self.calibration.lang = value
            self._reload_phrases()
        elif key in ("theme", "contrast", "key_scale"):
            self._apply_theme()
        elif key == "text_size":
            self.canvas.set_point_size(value)
        elif key in ("dwell_ms", "cooldown_ms"):
            self.dwell.dwell_s = s["dwell_ms"] / 1000.0
            self.dwell.cooldown_s = s["cooldown_ms"] / 1000.0
        elif key == "keyboard_mode":
            self.keyboard.set_mode(value)
        elif key == "keyboard_layout":
            self.keyboard.set_layout(value)
        elif key == "sensitivity":
            self.estimator.sensitivity = value
        elif key == "smoothing":
            self.estimator.set_smoothing(value)
        elif key == "pointer_mode":
            self.estimator.set_mode(value)
        elif key == "head_gain":
            self.estimator.head_gain = value
        elif key == "fixation_stabilizer":
            self.estimator.stabilize = value
        elif key == "confidence_threshold":
            self.estimator.confidence_threshold = value
        elif key == "tracking_eye":
            self.estimator.eye = value
        elif key == "gaze_cursor_size":
            self.gaze_cursor.size_px = value
        elif key == "speech_rate":
            self.speech.set_rate(value)
        elif key == "speech_volume":
            self.speech.set_volume(value)
        elif key == "speech_voice":
            self.speech.set_voice(value)
        elif key == "learn_words":
            self.predictor.learning_enabled = value
        elif key == "auto_capitalize":
            self.buffer.auto_capitalize = value
        elif key == "sound_feedback":
            if value and self.click_sound is None:
                self.click_sound = ClickSound()
        elif key in CAMERA_KEYS or key == "input_source":
            if not initial:
                self._restart_tracking()
        elif key in ("camera_preview", "show_landmarks", "inference_every", "iris_refinement"):
            self._configure_worker()
        if not initial:
            self.settings_page.refresh()
            self.refresh()

    def _apply_theme(self) -> None:
        s = self.settings
        p = theme.palette(s["theme"], s["contrast"])
        self.setStyleSheet(theme.stylesheet(p, s["key_scale"]))
        self.canvas.set_colors(p.canvas, p.canvas_text, p.accent)
        GazeButton.accent = QColor(p.accent)
        self.gaze_cursor.color = QColor(p.accent)
        self.calibration.dot_color = QColor(p.accent)
        self.calibration.text_color = QColor(p.text)

    # ============================================================= tracking
    def _restart_tracking(self) -> None:
        self._stop_tracking()
        if not self.enable_tracking or self.settings["input_source"] == "mouse":
            return
        try:
            from looksay.tracking.worker import TrackingWorker
        except Exception as exc:  # OpenCV / MediaPipe import problems
            self.tracker_error = str(exc)
            return
        width, height = (int(v) for v in self.settings["camera_resolution"].split("x"))
        self.worker = TrackingWorker(self.settings["camera_index"], (width, height), self.settings["camera_fps"],
                                     self.settings["inference_every"])
        self.worker.features_ready.connect(self._on_features)
        self.worker.preview_ready.connect(self._on_preview)
        self.worker.debug_ready.connect(self.debug_overlay.set_frame)
        self.worker.unavailable.connect(self._tracker_unavailable)
        self._configure_worker()
        self.tracker_error = ""
        self.worker.start()

    def _configure_worker(self) -> None:
        if not self.settings["camera_preview"]:
            self.preview_label.hide()
        if self.worker is None:
            return
        self.worker.inference_every = self.settings["inference_every"]
        self.worker.landmarks_overlay = self.settings["show_landmarks"]
        self.worker.iris_refinement = self.settings["iris_refinement"]
        self.worker.debug_enabled = self.debug_overlay.isVisible()
        self.worker.preview_enabled = self.settings["camera_preview"] or self.pages.currentWidget() is self.diagnostics_page
        if not self.worker.preview_enabled:
            self.diagnostics_page.set_preview(None)

    def _on_preview(self, image: QImage) -> None:
        if self.pages.currentWidget() is self.diagnostics_page:
            self.diagnostics_page.set_preview(image)
        if self.settings["camera_preview"]:
            self.preview_label.setPixmap(QPixmap.fromImage(image).scaledToWidth(PREVIEW_W))
            self.preview_label.adjustSize()
            self.preview_label.move(self.canvas.width() - self.preview_label.width() - 12, 12)
            self.preview_label.show()

    def _stop_tracking(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.gaze_sample = None
        self.last_features = None

    def _tracker_unavailable(self, reason: str) -> None:
        self.tracker_error = reason
        self.worker = None

    def _load_calibration(self) -> None:
        profile = self.db.active_calibration()
        if profile:
            try:
                self.estimator.set_model(GazeModel.from_dict(profile["model"]))
                if "session" in profile["model"]:
                    self.calib_session = CalibrationSession.from_dict(profile["model"]["session"])
            except (ValueError, KeyError, TypeError) as exc:
                log.warning("Ignoring unusable calibration profile: %s", exc)
        head = self.settings.get("_head_model")
        if head:
            try:
                self.estimator.set_head_model(HeadModel.from_dict(head))
            except (ValueError, KeyError, TypeError) as exc:
                log.warning("Ignoring unusable head calibration: %s", exc)

    def _on_features(self, features: FaceFeatures) -> None:
        self.last_features = features
        self._remember_vector(features)
        if (not self.estimator.ready and not self._auto_calibrated and features.face_found
                and not self.calibration.isVisible() and not self.confirm.isVisible()):
            # First use: calibrate as soon as a face is visible, so no caregiver is needed.
            self._auto_calibrated = True
            self.start_calibration("head" if self.estimator.mode == "head" else "full")
        if self.calibration.collecting:
            self.calibration.feed(features)
        self.gaze_sample = self.estimator.update(features)
        if self.settings["blink_select"] and self._last_source == "gaze" and not self.paused:
            if self.blink.update(features.eyes_closed(), features.timestamp):
                selected = self.dwell.trigger(time.monotonic())
                if selected is not None:
                    self._activate(selected, from_gaze=True)

    def _provisional_model(self, model) -> None:
        """Use a just-fitted model at once, so the result buttons can be selected with it."""
        if isinstance(model, HeadModel):
            self.estimator.set_head_model(model)
        else:
            self.estimator.set_model(model)

    def _calibration_finished(self, model) -> None:
        if isinstance(model, HeadModel):
            self.settings.set("_head_model", model.to_dict())
            self.estimator.set_head_model(model)
        elif self.calibration.mode == "head":
            self.estimator.set_head_model(self._head_before_calibration)
        elif model is not None:
            screen = (self.width(), self.height())
            error_px = model.error_px(*screen)
            self.calib_session = self.calibration.session
            self.db.save_calibration(datetime.now().strftime("%Y-%m-%d %H:%M"), model.version,
                                     len(self.calibration.targets), screen, error_px, self._model_record(model))
            self.estimator.set_model(model)
            self._confirmed_samples = 0
        elif self.calibration.mode == "full":
            self.estimator.set_model(self._model_before_calibration)
        self.dwell.reset()
        self.refresh()

    def _model_record(self, model: GazeModel) -> dict:
        record = model.to_dict()
        if self.calib_session is not None:
            record["session"] = self.calib_session.to_dict()  # lets the model keep learning after restarts
        return record

    # ---------------------------------------------------- online refinement
    def _remember_vector(self, features: FaceFeatures) -> None:
        vec = features.vector(self.estimator.eye) if features.face_found and not features.eyes_closed() else None
        if vec is not None and features.confidence >= self.estimator.confidence_threshold:
            self._recent_vectors.append((features.timestamp, vec))
        cutoff = features.timestamp - 0.3
        self._recent_vectors = [(t, v) for t, v in self._recent_vectors if t >= cutoff]

    def _queue_learning_sample(self, button: GazeButton) -> None:
        if (not self.settings["learn_from_selections"] or self.calib_session is None
                or self.estimator.mode == "head" or len(self._recent_vectors) < 3):
            return
        center = button.mapTo(self, button.rect().center())
        target = (center.x() / self.width(), center.y() / self.height())
        vec = np.median(np.array([v for _, v in self._recent_vectors]), axis=0)
        self._pending_samples.append((time.monotonic(), vec, target))

    def _confirm_learning_sample(self, now: float) -> None:
        refit = False
        while self._pending_samples and now - self._pending_samples[0][0] >= CONFIRM_S:
            _, vec, target = self._pending_samples.pop(0)
            self.calib_session.add_extra(vec, target, 1.0)
            self._confirmed_samples += 1
            refit = refit or self._confirmed_samples % REFIT_EVERY == 0
        if not refit:
            return
        model = self.calib_session.fit()
        current = self.estimator.model
        # Only adopt a refit that does not validate worse than what we have.
        if model is not None and (current is None or model.fit_error <= current.fit_error * 1.1):
            offset = self.estimator.offset.copy()
            self.estimator.model = model
            self.estimator.offset = offset * 0.5  # the refit absorbs part of the drift
            self.db.update_active_calibration(model.error_px(self.width(), self.height()), self._model_record(model))
            log.info("Gaze model refined from %d selections", self._confirmed_samples)

    def _recentered(self, target: tuple, observed: np.ndarray) -> None:
        self.estimator.recenter(target, observed)

    def start_calibration(self, mode: str = "full") -> None:
        if self.worker is None or (mode == "recenter" and self.estimator.model is None):
            self.show_message(tr("calib_no_camera" if self.worker is None else "status_not_calibrated", self.lang))
            return
        self._model_before_calibration = self.estimator.model
        self._head_before_calibration = self.estimator.head_model
        self.dwell.reset()
        self.calibration.pursuit_enabled = self.settings["calibration_pursuit"]
        self.calibration.start(self.settings["calibration_points"], self.settings["tracking_eye"], mode)
        self.gaze_cursor.raise_()

    # ============================================================ dwell loop
    def _current_point(self, now: float) -> tuple[QPointF | None, str, bool, bool]:
        """Returns (local point, source, active, hold)."""
        source = self.settings["input_source"]
        global_pos = QCursor.pos()
        if self._mouse_pos is None or global_pos != self._mouse_pos:
            if self._mouse_pos is not None:
                self._mouse_moved_at = now
            self._mouse_pos = global_pos
        sample = self.gaze_sample
        gaze_fresh = (sample is not None and self.last_features is not None
                      and now - self.last_features.timestamp < STALE_FEATURES_S
                      and self.estimator.ready)
        use_mouse = source == "mouse" or (source == "both" and (
            not gaze_fresh or now - self._mouse_moved_at < MOUSE_PRIORITY_S))
        if use_mouse:
            return QPointF(self.mapFromGlobal(global_pos)), "mouse", True, False
        if not gaze_fresh:
            return None, "gaze", False, False
        pos = self.cursor_animator.step(np.array([sample.x, sample.y]), now)
        point = QPointF(pos[0] * self.width(), pos[1] * self.height())
        return point, "gaze", sample.usable, sample.state == "blink"

    def _in_active_layer(self, button: GazeButton) -> bool:
        """False for buttons hidden behind an open confirmation or calibration overlay."""
        for overlay in (self.confirm, self.calibration):
            if overlay.isVisible():
                return overlay.isAncestorOf(button)
        return True

    def _target_at(self, point: QPointF | None) -> GazeButton | None:
        if point is None or not self.rect().contains(point.toPoint()):
            return None
        if self.paused:
            local = self.pause_btn.mapFrom(self, point.toPoint())
            return self.pause_btn if self.pause_btn.rect().contains(local) else None
        if self.confirm.isVisible():
            root = self.confirm
        elif self.calibration.isVisible():
            if self.calibration.state not in ("done", "failed"):
                return None
            root = self.calibration
        else:
            root = self
        widget = root.childAt(root.mapFrom(self, point.toPoint()))
        while widget is not None and widget is not root:
            if isinstance(widget, GazeButton):
                return widget if widget.isEnabled() and widget.isVisible() else None
            widget = widget.parentWidget()
        return None

    def _tick(self) -> None:
        self.ui_fps.tick()
        now = time.monotonic()
        point, source, active, hold = self._current_point(now)
        self._last_source = source
        target = self._target_at(point) if active or hold else None
        current_btn = self._dwell_button
        if (target is not current_btn and current_btn is not None and point is not None and current_btn.isVisible()
                and current_btn.isEnabled() and not self.paused and self._in_active_layer(current_btn)):
            # Sticky keys: small wobbles across the border do not switch to the neighbour.
            r = current_btn.rect()
            mx, my = int(r.width() * STICKY_MARGIN), int(r.height() * STICKY_MARGIN)
            if r.adjusted(-mx, -my, mx, my).contains(current_btn.mapFrom(self, point.toPoint())):
                target = current_btn
        state = self.dwell.update(target, now, active=active, hold=hold)
        self._confirm_learning_sample(now)
        current = state.target if isinstance(state.target, GazeButton) else None
        if current is not self._dwell_button and self._dwell_button is not None:
            self._dwell_button.set_dwell(False, 0.0)
        self._dwell_button = current
        if current is not None:
            current.set_dwell(True, state.progress)
        if state.selected is not None:
            self._activate(state.selected, from_gaze=source == "gaze")

        show_cursor = source == "gaze" and point is not None and self.settings["gaze_cursor"]
        self.gaze_cursor.set_point(point if show_cursor else None, active)
        if self._message and now > self._message_until:
            self._message = ""
            self.message_label.setText("")
        self._update_status(source)
        if self.debug_overlay.isVisible():
            self._debug_tick = (getattr(self, "_debug_tick", 0) + 1) % 3
            if self._debug_tick == 0:  # ~10 Hz is plenty for text
                self._update_debug_info()

    def _activate(self, button: GazeButton, from_gaze: bool) -> None:
        if from_gaze:
            self._queue_learning_sample(button)
        if from_gaze and self.settings["auto_drift"]:
            center = button.mapTo(self, button.rect().center())
            self.estimator.nudge_towards((center.x() / self.width(), center.y() / self.height()))
        button.set_dwell(False, 0.0)
        if self.click_sound and self.settings["sound_feedback"]:
            self.click_sound.play()
        button.click()

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.dwell.cancel()  # keep the cooldown, so one long look cannot toggle twice
        self._update_status(self._last_source, force=True)

    def _update_status(self, source: str, force: bool = False) -> None:
        lang = self.lang
        stats = self.worker.stats if self.worker else None
        if self.paused:
            text = tr("status_paused", lang)
        elif self.settings["input_source"] == "mouse" or not self.enable_tracking:
            text = tr("status_mouse", lang)
        elif self.tracker_error:
            text = tr("status_no_tracker", lang)
        elif stats is not None and not stats.camera_ok and stats.camera_error:
            text = tr("status_no_camera", lang)
        elif self.last_features is None:
            text = tr("status_starting", lang)
        elif not self.estimator.ready:
            text = tr("status_head_not_calibrated" if self.estimator.mode == "head" else "status_not_calibrated", lang)
        elif self.gaze_sample is not None and self.gaze_sample.state == "no_face":
            text = tr("status_lost", lang)
        elif self.gaze_sample is not None and self.gaze_sample.state == "low_confidence":
            text = tr("status_low_conf", lang)
        else:
            conf = self.gaze_sample.confidence if self.gaze_sample else 0.0
            text = f"{tr('status_tracking', lang)} · {conf * 100:.0f}%"
        if source == "mouse" and self.settings["input_source"] == "both" and self.estimator.ready:
            text += f" ({tr('status_mouse', lang)})"
        if force or self.status_label.text() != text:
            self.status_label.setText(text)
        pause_text = tr("resume" if self.paused else "pause", lang)
        if self.pause_btn.text() != pause_text:
            self.pause_btn.setText(pause_text)
            self.pause_btn.set_role("active" if self.paused else "special")

    def show_message(self, text: str) -> None:
        self._message = text
        self._message_until = time.monotonic() + MESSAGE_S
        self.message_label.setText(text)

    # ============================================================== editing
    def _word_committed(self, word: str, prev: str | None) -> None:
        self.predictor.learn(word, prev)

    def refresh(self) -> None:
        self.canvas.set_text(self.buffer.text)
        prefix = self.buffer.current_prefix()
        prev = self.buffer.previous_word()
        count = self.settings["suggestion_count"]
        capitalize = self.buffer.auto_capitalize and self.buffer.at_sentence_start()

        def fmt(words):
            return [w[:1].upper() + w[1:] if capitalize else w for w in words]

        if self.keyboard.mode == "words":
            words = self.predictor.suggest(prefix, prev, WORD_GRID_SIZE + count)
            if prefix and len(words) < WORD_GRID_SIZE + count:
                # Few completions: fill the grid with whole-word predictions for this context.
                words += [w for w in self.predictor.suggest("", prev, WORD_GRID_SIZE + count) if w not in words]
            words = fmt(words[:WORD_GRID_SIZE + count])
            self.keyboard.set_suggestions(words[:count], count)
            self.keyboard.set_word_grid(words[count:])
        else:
            self.keyboard.set_suggestions(fmt(self.predictor.suggest(prefix, prev, count)), count)
        self.keyboard.update_case(self.buffer.shift_active or (capitalize and not prefix))

    def _on_key(self, name: str, value: str) -> None:
        if name in CORRECTION_KEYS:
            self._pending_samples.clear()  # recent selections were probably wrong: do not learn from them
        b = self.buffer
        if name == "char":
            b.insert_char(value)
        elif name == "suggest":
            if value:
                b.accept_word(value)
        elif name == "space":
            b.space()
        elif name == "backspace":
            b.backspace()
        elif name == "delete_word":
            b.delete_word()
        elif name == "undo":
            b.undo()
        elif name == "shift":
            b.toggle_shift()
        elif name == "symbols":
            self.keyboard.toggle_symbols()
        elif name == "mode":
            self.settings.set("keyboard_mode", "words" if self.keyboard.mode == "letters" else "letters")
        elif name == "phrases":
            self._reload_phrases()
            self.pages.setCurrentWidget(self.phrases_page)
        elif name == "speak":
            self.speak()
        elif name == "stop":
            self.speech.stop()
        elif name == "clear":
            if b.text:
                self.confirm.ask(tr("confirm_clear", self.lang), self.lang, self._confirm_clear)
                self.gaze_cursor.raise_()
        elif name == "language":
            self.settings.set("language", LANGUAGES[(LANGUAGES.index(self.lang) + 1) % len(LANGUAGES)])
        elif name == "settings":
            self.settings_page.refresh()
            self.pages.setCurrentWidget(self.settings_page)
        elif name == "help":
            self.emergency()
        self.refresh()

    def _confirm_clear(self, yes: bool) -> None:
        if yes:
            self.buffer.clear()
            self.refresh()

    def show_keyboard(self) -> None:
        self.pages.setCurrentWidget(self.keyboard)
        self._configure_worker()
        self.refresh()

    # =============================================================== speech
    def speak(self) -> None:
        text = self.buffer.last_sentence() if self.settings["speak_mode"] == "sentence" else self.buffer.text
        if not text.strip():
            return
        if isinstance(self.speech, NullSpeech) or not self.speech.available:
            self.show_message(tr("speech_unavailable", self.lang))
        elif not self.speech.voices():
            self.show_message(tr("no_voices", self.lang))
        self._say(text)

    def _say(self, text: str, volume: float | None = None) -> None:
        self.speech.set_volume(self.settings["speech_volume"] if volume is None else volume)
        self.speech.speak(text)

    def emergency(self) -> None:
        """Speak the first emergency phrase immediately. Never contacts anyone automatically."""
        phrases = [p for p in self.db.phrases(self.lang) if p["is_emergency"]]
        text = phrases[0]["text"] if phrases else tr("help", self.lang)
        self.show_message(text)
        self._say(text, volume=1.0)  # full volume; normal volume is restored on the next speak

    # ============================================================== phrases
    def _reload_phrases(self) -> None:
        self.phrases_page.set_phrases(self.db.phrases(self.lang))

    def _phrase_chosen(self, text: str) -> None:
        self.buffer.insert_phrase(text)
        self._say(text)
        self.show_keyboard()

    def _save_phrase(self) -> None:
        if self.buffer.text.strip():
            self.db.add_phrase(self.lang, self.buffer.text)
            self._reload_phrases()
            self.show_message(f"✓ {self.buffer.text.strip()}")

    def _delete_phrase(self, phrase_id: int, text: str) -> None:
        def done(yes: bool) -> None:
            if yes:
                self.db.delete_phrase(phrase_id)
                self._reload_phrases()
        self.confirm.ask(tr("confirm_delete_phrase", self.lang, text=text), self.lang, done)
        self.gaze_cursor.raise_()

    # ============================================================= commands
    def _on_command(self, name: str) -> None:
        lang = self.lang
        if name == "calibrate":
            self.start_calibration("full")
        elif name == "calibrate_head":
            self.start_calibration("head")
        elif name == "recenter":
            self.start_calibration("recenter")
        elif name == "diagnostics":
            self.pages.setCurrentWidget(self.diagnostics_page)
            self._configure_worker()
            self._refresh_diagnostics()
        elif name == "export":
            path = exports_dir() / f"looksay-export-{datetime.now():%Y%m%d-%H%M%S}.json"
            self.db.export_to_file(path)
            self.show_message(tr("exported", lang, path=str(path)))
        elif name == "import":
            candidates = sorted(exports_dir().glob("*.json"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                self.show_message(tr("no_import", lang, path=str(exports_dir())))
                return
            try:
                self.db.import_from_file(candidates[-1])
            except (ValueError, OSError) as exc:
                self.show_message(tr("import_failed", lang, error=str(exc)))
                return
            self.settings.reload()
            self._reload_phrases()
            self.show_message(tr("imported", self.lang, path=candidates[-1].name))
        elif name == "reset_settings":
            self.confirm.ask(tr("confirm_reset_settings", lang), lang, self._reset_settings)
        elif name == "reset_data":
            self.confirm.ask(tr("confirm_reset_data", lang), lang, self._reset_data)
        self.gaze_cursor.raise_()

    def _reset_settings(self, yes: bool) -> None:
        if yes:
            self.db.reset_settings()
            self.settings.reload()

    def _reset_data(self, yes: bool) -> None:
        if yes:
            self.db.reset_all()
            self.estimator.set_model(None)
            self.predictor.session_recent.clear()
            self.settings.reload()  # re-applies language: reloads vocabulary and default phrases
            self.show_message(tr("data_reset_done", self.lang))

    # ========================================================== diagnostics
    def _leave_diagnostics(self) -> None:
        self.pages.setCurrentWidget(self.settings_page)
        self._configure_worker()

    def _refresh_diagnostics(self) -> None:
        if self.pages.currentWidget() is not self.diagnostics_page:
            return
        lang = self.lang
        f = self.last_features
        stats = self.worker.stats if self.worker else None
        yes, no = tr("diag_detected", lang), tr("diag_not_detected", lang)
        sample = self.gaze_sample
        profile = self.db.active_calibration()
        values = {
            "diag_camera": ("OK" if stats and stats.camera_ok else (stats.camera_error if stats else
                            (self.tracker_error or tr("status_mouse", lang)))) or "…",
            "diag_face": yes if f and f.face_found else no,
            "diag_iris": yes if f and f.right is not None else no,
            "diag_gaze": f"{sample.x * self.width():.0f}, {sample.y * self.height():.0f}"
            if sample and self.estimator.ready else "—",
            "diag_confidence": f"{f.confidence * 100:.0f}%" if f else "—",
            "diag_eye_open": f"{f.left.openness:.2f} / {f.right.openness:.2f}" if f and f.right else "—",
            "diag_capture_fps": f"{stats.capture_fps.rate:.1f}" if stats else "—",
            "diag_process_fps": f"{stats.process_fps.rate:.1f}" if stats else "—",
            "diag_ui_fps": f"{self.ui_fps.rate:.1f}",
            "diag_inference": f"{stats.avg_inference_ms:.1f} ms" if stats else "—",
            "diag_dropped": f"{stats.frames_dropped} / {stats.frames_captured} ({stats.drop_ratio * 100:.1f}%)"
            if stats else "—",
            "diag_calibration": f"{profile['error_px']:.0f} px ({profile['n_points']} pts)" if profile
            else tr("diag_none", lang),
        }
        self.diagnostics_page.set_values(values)

    # ============================================================== closing
    def closeEvent(self, event) -> None:
        self.timer.stop()
        self._stop_tracking()
        self.speech.stop()
        self.db.prune_learning()
        super().closeEvent(event)
