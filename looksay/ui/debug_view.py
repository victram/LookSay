"""Tracking debug overlay (toggle with the physical K key).

Three camera views - the face with mesh, eye contours, iris circles,
eye-direction arrows and head-pose axes, plus a magnified, roll-corrected
view of each eye - and HUD lines with every number the gaze pipeline uses.
It never receives mouse/gaze input, so it cannot block selection.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from looksay.tracking.worker import DebugFrame

BG = QColor(14, 16, 19, 235)
FG = QColor(235, 235, 235)
DIM = QColor(150, 156, 165)
OK = QColor(80, 220, 120)
WARN = QColor(255, 190, 60)
BAD = QColor(255, 90, 80)


class DebugOverlay(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.frame: DebugFrame | None = None
        self.lines: list[tuple[str, QColor]] = []
        self.eye_lines: dict[str, list[str]] = {"right": [], "left": []}
        self.message = ""
        self._font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.hide()

    def set_frame(self, frame: DebugFrame) -> None:
        self.frame = frame
        if self.isVisible():
            self.update()

    def set_info(self, lines: list[tuple[str, QColor]], eye_lines: dict[str, list[str]], message: str = "") -> None:
        self.lines, self.eye_lines, self.message = lines, eye_lines, message
        if self.isVisible():
            self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), BG)
        pad = 8
        h = self.height() - 2 * pad
        x = pad
        font = QFont(self._font)
        font.setPixelSize(max(9, min(14, h // 24)))
        p.setFont(font)
        lh = p.fontMetrics().height()

        face_w = int(h * 4 / 3)
        eye_h = (h - pad) // 2
        eye_w = int(eye_h * 3 / 2)
        face_rect = QRectF(x, pad, face_w, h)
        self._image(p, self.frame.face if self.frame else None, face_rect, "FACE · camera view")
        x += face_w + pad
        for i, (name, label) in enumerate((("right", "RIGHT EYE (R)"), ("left", "LEFT EYE (L)"))):
            rect = QRectF(x, pad + i * (eye_h + pad), eye_w, eye_h)
            img = None if self.frame is None else (self.frame.right_eye if name == "right" else self.frame.left_eye)
            self._image(p, img, rect, label)
        x += eye_w + pad

        # HUD column
        y = pad + lh
        p.setPen(FG)
        bold = QFont(font)
        bold.setBold(True)
        p.setFont(bold)
        p.drawText(int(x), int(y), "LOOKSAY TRACKING DEBUG  ·  press K to close")
        p.setFont(font)
        y += lh * 1.4
        hud = list(self.lines)
        for name, tag in (("right", "R EYE"), ("left", "L EYE")):
            if self.eye_lines.get(name):
                hud.append((f"{tag}   " + "   ".join(self.eye_lines[name]), FG))
        for text, color in hud:
            if y > self.height() - pad:
                break
            p.setPen(color)
            p.drawText(QRectF(x, y - p.fontMetrics().ascent(), self.width() - x - pad, lh),
                       Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextSingleLine, text)
            y += lh
        if self.message:
            p.setPen(WARN)
            p.drawText(face_rect.adjusted(10, 10, -10, -10), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       self.message)
        p.end()

    def _image(self, p: QPainter, img: QImage | None, rect: QRectF, label: str) -> None:
        p.fillRect(rect, QColor(24, 24, 24))
        if img is not None and not img.isNull():
            scaled = img.size().scaled(rect.size().toSize(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRectF(rect.center().x() - scaled.width() / 2, rect.center().y() - scaled.height() / 2,
                            scaled.width(), scaled.height())
            p.drawImage(target, img)
        p.setPen(QPen(QColor(70, 76, 84), 1))
        p.drawRect(rect)
        p.setPen(DIM)
        self._shadow_text(p, rect.left() + 6, rect.top() + p.fontMetrics().ascent() + 4, label)

    @staticmethod
    def _shadow_text(p: QPainter, x: float, y: float, text: str) -> None:
        pen = p.pen()
        p.setPen(QColor(0, 0, 0))
        p.drawText(int(x) + 1, int(y) + 1, text)
        p.setPen(pen)
        p.drawText(int(x), int(y), text)
