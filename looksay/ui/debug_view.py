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

BG = QColor(14, 16, 19)
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
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)
        pad = 10
        W, H = self.width() - 2 * pad, self.height() - 2 * pad
        font = QFont(self._font)
        font.setPixelSize(max(10, min(16, H // 48)))
        p.setFont(font)

        # Layout: large face view on the left, both eye zooms stacked on the right.
        eye_h = (H - pad) / 2
        eye_w = min(eye_h * 3 / 2, W * 0.42)
        eye_h = eye_w * 2 / 3
        face_rect = QRectF(pad, pad, W - eye_w - pad, H)
        face_img = self._image(p, self.frame.face if self.frame else None, face_rect)
        # Title on top, numbers at the bottom: the eyes stay visible in between.
        self._hud(p, face_img.adjusted(8, 8, -8, -8), [("FACE · LOOKSAY TRACKING DEBUG · K to close", DIM)], top=True)
        self._hud(p, face_img.adjusted(8, 8, -8, -8), list(self.lines), top=False)
        if self.message:
            p.setPen(WARN)
            big = QFont(font)
            big.setPixelSize(font.pixelSize() * 2)
            p.setFont(big)
            p.drawText(face_img, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.message)
            p.setFont(font)

        x = face_rect.right() + pad
        y0 = pad + (H - 2 * eye_h - pad) / 2
        for i, (name, label) in enumerate((("right", "RIGHT EYE (R)"), ("left", "LEFT EYE (L)"))):
            rect = QRectF(x, y0 + i * (eye_h + pad), eye_w, eye_h)
            img = None if self.frame is None else (self.frame.right_eye if name == "right" else self.frame.left_eye)
            shown = self._image(p, img, rect)
            self._hud(p, shown.adjusted(6, 6, -6, -6), [(label, DIM)], top=True)
            self._hud(p, shown.adjusted(6, 6, -6, -6), [(t, FG) for t in self.eye_lines.get(name, [])], top=False)
        p.end()

    def _image(self, p: QPainter, img: QImage | None, rect: QRectF) -> QRectF:
        """Draw `img` fitted into `rect`; returns the rectangle actually covered by the image."""
        p.fillRect(rect, QColor(24, 24, 24))
        target = QRectF(rect)
        if img is not None and not img.isNull():
            scaled = img.size().scaled(rect.size().toSize(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRectF(rect.center().x() - scaled.width() / 2, rect.center().y() - scaled.height() / 2,
                            scaled.width(), scaled.height())
            p.drawImage(target, img)
        p.setPen(QPen(QColor(70, 76, 84), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(target)
        return target

    @staticmethod
    def _hud(p: QPainter, area: QRectF, lines: list[tuple[str, QColor]], top: bool) -> None:
        """Lines of text on a translucent box drawn over the camera image."""
        if not lines:
            return
        fm = p.fontMetrics()
        lh = fm.height()
        width = min(area.width(), max(fm.horizontalAdvance(t) for t, _ in lines) + 16)
        height = min(area.height(), lh * len(lines) + 10)
        y = area.top() if top else area.bottom() - height
        box = QRectF(area.left(), y, width, height)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 165))
        p.drawRoundedRect(box, 6, 6)
        ty = box.top() + 5
        for text, color in lines:
            if ty + lh > box.bottom() + 1:
                break
            p.setPen(color)
            p.drawText(QRectF(box.left() + 8, ty, box.width() - 12, lh),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextSingleLine, text)
            ty += lh

    @staticmethod
    def _shadow_text(p: QPainter, x: float, y: float, text: str) -> None:
        pen = p.pen()
        p.setPen(QColor(0, 0, 0))
        p.drawText(int(x) + 1, int(y) + 1, text)
        p.setPen(pen)
        p.drawText(int(x), int(y), text)
