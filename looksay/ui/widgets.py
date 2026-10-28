"""Reusable widgets: dwell-selectable button, message canvas, gaze cursor."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QTextLayout, QTextOption
from PySide6.QtWidgets import QPushButton, QSizePolicy, QWidget


class GazeButton(QPushButton):
    """A button that can be selected by dwelling on it (or by mouse click).

    Dwell progress is drawn as a calm bar filling along the bottom edge plus
    an outline while the button is targeted; nothing flashes.
    """

    accent = QColor("#1458b8")

    def __init__(self, text: str = "", role: str = "key", parent: QWidget | None = None):
        super().__init__(text, parent)
        self._progress = 0.0
        self._targeted = False
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(40, 40)
        self.set_role(role)

    def set_role(self, role: str) -> None:
        if self.property("role") != role:
            self.setProperty("role", role)
            self.style().unpolish(self)
            self.style().polish(self)

    def set_dwell(self, targeted: bool, progress: float) -> None:
        if targeted != self._targeted or abs(progress - self._progress) > 0.004:
            self._targeted, self._progress = targeted, progress
            self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self._targeted:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        painter.setPen(QPen(self.accent, 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 10, 10)
        if self._progress > 0:
            bar_h = max(6.0, rect.height() * 0.09)
            bar = QRectF(rect.left() + 6, rect.bottom() - bar_h - 4, (rect.width() - 12) * self._progress, bar_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.accent)
            painter.drawRoundedRect(bar, bar_h / 2, bar_h / 2)
        painter.end()


class MessageCanvas(QWidget):
    """Large writing area showing the message with a caret at the end."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._text = ""
        self._point_size = 36
        self.background = QColor("#ffffff")
        self.foreground = QColor("#111111")
        self.caret_color = QColor("#1458b8")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_text(self, text: str) -> None:
        self._text = text
        self.update()

    def set_point_size(self, size: int) -> None:
        self._point_size = size
        self.update()

    def set_colors(self, background: str, foreground: str, caret: str) -> None:
        self.background, self.foreground, self.caret_color = QColor(background), QColor(foreground), QColor(caret)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.background)
        margin = 24
        width = self.width() - 2 * margin
        font = QFont(self.font())
        font.setPointSize(self._point_size)
        layout = QTextLayout(self._text, font)
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(option)
        layout.beginLayout()
        y = 0.0
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            line.setPosition(QPointF(0, y))
            y += line.height() * 1.15
        layout.endLayout()
        line_h = layout.lineAt(0).height() * 1.15 if layout.lineCount() else font.pointSizeF() * 1.6
        # Keep the end of the message (and the caret) visible: scroll up if needed.
        overflow = max(0.0, y - (self.height() - 2 * margin))
        overflow = line_h * int((overflow + line_h - 1) // line_h) if overflow else 0.0
        origin = QPointF(margin, margin - overflow)
        painter.setPen(self.foreground)
        layout.draw(painter, origin)
        painter.setPen(QPen(self.caret_color))
        layout.drawCursor(painter, origin, len(self._text), max(3, self._point_size // 10))
        painter.end()


class GazeCursor(QWidget):
    """Small high-contrast ring showing where the user is looking. Ignores mouse events."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self._pos: QPointF | None = None
        self._ok = True
        self.size_px = 24
        self.color = QColor("#1458b8")

    def set_point(self, pos: QPointF | None, ok: bool) -> None:
        if pos != self._pos or ok != self._ok:
            old = self._pos
            self._pos, self._ok = pos, ok
            r = self.size_px + 6
            for p in (old, pos):
                if p is not None:
                    self.update(int(p.x() - r), int(p.y() - r), 2 * r, 2 * r)

    def paintEvent(self, event) -> None:
        if self._pos is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.size_px / 2
        # Dark outer + light inner ring stays visible on any background.
        painter.setPen(QPen(QColor(0, 0, 0, 200), 5))
        painter.drawEllipse(self._pos, r, r)
        pen = QPen(self.color if self._ok else QColor("#9e9e9e"), 3)
        if not self._ok:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawEllipse(self._pos, r, r)
        painter.end()
