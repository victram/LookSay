"""Guided 5/9-point calibration and single-point drift recentering."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from looksay.calibration.head_model import fit_head_model
from looksay.calibration.model import MIN_SAMPLES_PER_POINT, CalibrationSession, GazeModel, calibration_targets
from looksay.i18n.strings import tr
from looksay.tracking.features import FaceFeatures
from looksay.ui.widgets import GazeButton

INTRO_S = 2.5
SETTLE_S = 0.8
COLLECT_S = 1.5
MAX_COLLECT_S = 3.5


class CalibrationOverlay(QWidget):
    finished = Signal(object)  # GazeModel accepted, or None if cancelled
    model_ready = Signal(object)  # provisional model, so the result buttons work by gaze
    recentered = Signal(tuple, object)  # target, observed raw point

    def __init__(self, parent: QWidget, raw_point: Callable[[FaceFeatures], np.ndarray | None]):
        super().__init__(parent)
        self.setObjectName("Overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._raw_point = raw_point
        self.lang = "en"
        self.dot_color = QColor("#1458b8")
        self.text_color = QColor("#111111")
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self.state = "idle"
        self.mode = "full"
        self.session: CalibrationSession | None = None
        self.targets: list[tuple[float, float]] = []
        self.index = 0
        self._state_t = 0.0
        self._recenter_samples: list[np.ndarray] = []
        self.model: GazeModel | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(80, 80, 80, 80)
        self.message = QLabel("")
        self.message.setObjectName("Message")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setWordWrap(True)
        root.addWidget(self.message, 1)
        self.buttons = QWidget()
        row = QHBoxLayout(self.buttons)
        row.setSpacing(40)
        self.cancel_btn = GazeButton("", "special")
        self.retry_btn = GazeButton("", "special")
        self.accept_btn = GazeButton("", "special")
        for b in (self.cancel_btn, self.retry_btn, self.accept_btn):
            b.setMinimumHeight(140)
            row.addWidget(b)
        self.cancel_btn.clicked.connect(self._cancel)
        self.retry_btn.clicked.connect(self._retry)
        self.accept_btn.clicked.connect(self._accept)
        root.addWidget(self.buttons, 1)
        self.hide()

    # ------------------------------------------------------------ control
    def start(self, n_points: int, eye: str, mode: str = "full") -> None:
        self.mode = mode
        if mode == "recenter":
            self.targets = [(0.5, 0.5)]
        elif mode == "head":  # nose pointing: keep targets reachable with a comfortable head turn
            self.targets = calibration_targets(9 if n_points >= 9 else 5, margin=0.12)
        else:
            self.targets = calibration_targets(n_points)
        self.session = CalibrationSession(self.targets, eye=eye)
        self._recenter_samples = []
        self.index = 0
        self.model = None
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        self.setFocus()
        self._set_state("intro")
        self._timer.start()

    @property
    def collecting(self) -> bool:
        return self.isVisible() and self.state in ("intro", "settle", "collect")

    def _set_state(self, state: str) -> None:
        self.state = state
        self._state_t = time.monotonic()
        result = state in ("done", "failed")
        self.buttons.setVisible(result)
        self.accept_btn.setVisible(state == "done")
        self.cancel_btn.setText(tr("cancel", self.lang))
        self.retry_btn.setText(tr("retry", self.lang))
        self.accept_btn.setText(tr("accept", self.lang))
        if state == "intro":
            key = {"recenter": "calib_recenter", "head": "calib_head_intro"}.get(self.mode, "calib_intro")
            self.message.setText(tr(key, self.lang))
            self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        elif state in ("settle", "collect"):
            self.message.setText("")
        self.update()

    def feed(self, features: FaceFeatures) -> None:
        if self.state != "collect" or self.session is None:
            return
        if not features.face_found:
            return
        if self.mode == "head":
            vec = features.head_vector()
            if vec is not None:
                self.session.add_sample(self.index, vec)
            return
        if features.eyes_closed():
            return
        if self.mode == "recenter":
            raw = self._raw_point(features)
            if raw is not None:
                self._recenter_samples.append(raw)
            return
        vec = features.vector(self.session.eye)
        if vec is not None:
            self.session.add_sample(self.index, vec)

    def _tick(self) -> None:
        elapsed = time.monotonic() - self._state_t
        if self.state == "intro" and elapsed >= INTRO_S:
            self._set_state("settle")
        elif self.state == "settle" and elapsed >= SETTLE_S:
            self._set_state("collect")
        elif self.state == "collect":
            have = len(self._recenter_samples) if self.mode == "recenter" else len(
                self.session.samples.get(self.index, []))
            if (elapsed >= COLLECT_S and have >= MIN_SAMPLES_PER_POINT * 2) or elapsed >= MAX_COLLECT_S:
                self._next_point()
        self.update()

    def _next_point(self) -> None:
        if self.mode == "recenter":
            self._timer.stop()
            self.hide()
            if len(self._recenter_samples) >= MIN_SAMPLES_PER_POINT:
                observed = np.median(np.array(self._recenter_samples), axis=0)
                self.recentered.emit(self.targets[0], observed)
            self.finished.emit(None)
            return
        self.index += 1
        if self.index < len(self.targets):
            self._set_state("settle")
            return
        if self.mode == "head":
            self.model = fit_head_model(self.targets, self.session.samples)
        else:
            self.model = self.session.fit()
        if self.model is None:
            self.message.setText(tr("calib_failed", self.lang))
            self._set_state("failed")
            self.message.setText(tr("calib_failed", self.lang))
        else:
            error_px = self.model.error_px(self.width(), self.height())
            pct = 100 * error_px / float(np.hypot(self.width(), self.height()))
            quality = "calib_quality_good" if pct < 4 else ("calib_quality_fair" if pct < 8 else "calib_quality_poor")
            self._set_state("done")
            self.message.setText(tr("calib_done", self.lang, error=f"{error_px:.0f}", pct=f"{pct:.1f}")
                                 + "\n" + tr(quality, self.lang))
            self.model_ready.emit(self.model)

    def _accept(self) -> None:
        self._timer.stop()
        self.hide()
        self.finished.emit(self.model)

    def _cancel(self) -> None:
        self._timer.stop()
        self.hide()
        self.model = None
        self.finished.emit(None)

    def _retry(self) -> None:
        self.start(len(self.targets), self.session.eye if self.session else "both", self.mode)

    def keyPressEvent(self, event) -> None:  # caregiver escape hatch
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------ drawing
    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.state == "done" and self.model is not None:
            self._paint_results()
            return
        if self.state not in ("settle", "collect") or not self.targets:
            return
        tx, ty = self.targets[self.index]
        center = QPointF(tx * self.width(), ty * self.height())
        elapsed = time.monotonic() - self._state_t
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.state == "settle":
            radius = 34 - 22 * min(1.0, elapsed / SETTLE_S)
        else:
            radius = 12
        painter.setPen(QPen(self.text_color, 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, radius + 8, radius + 8)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.dot_color)
        painter.drawEllipse(center, radius, radius)
        painter.setBrush(self.text_color)
        painter.drawEllipse(center, 3, 3)
        painter.end()

    def _paint_results(self) -> None:
        """Each target (ring) and where the cross-validated model placed the gaze (dot)."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        for tx, ty, px, py in self.model.point_results:
            target = QPointF(tx * w, ty * h)
            pred = QPointF(px * w, py * h)
            painter.setPen(QPen(self.text_color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(target, 14, 14)
            painter.setPen(QPen(self.dot_color, 2, Qt.PenStyle.DashLine))
            painter.drawLine(target, pred)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.dot_color)
            painter.drawEllipse(pred, 6, 6)
        painter.end()
