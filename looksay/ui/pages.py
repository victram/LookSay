"""Secondary screens: quick phrases, settings, diagnostics and confirmations.

Every control is a GazeButton (no sliders, combo boxes or scroll bars), so
the whole app stays operable by dwell alone. Long lists are paged.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from looksay.config import SETTING_GROUPS, SETTING_SPECS
from looksay.i18n.strings import LANGUAGE_NAMES, tr
from looksay.ui.widgets import GazeButton

SETTINGS_ROWS_PER_PAGE = 5
PHRASES_PER_PAGE = 8


def _hbox(*widgets, stretches=None, spacing=8) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setSpacing(spacing)
    layout.setContentsMargins(0, 0, 0, 0)
    for i, w in enumerate(widgets):
        layout.addWidget(w, stretches[i] if stretches else 1)
    return layout


# --------------------------------------------------------------------- phrases
class PhrasesPage(QWidget):
    back = Signal()
    phrase_chosen = Signal(str)
    save_requested = Signal()
    delete_requested = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Page")
        self.lang = "en"
        self.phrases: list[dict[str, Any]] = []
        self.page = 0
        self.delete_mode = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 12)
        root.setSpacing(8)
        self.back_btn = GazeButton("", "special")
        self.save_btn = GazeButton("", "special")
        self.delete_btn = GazeButton("", "special")
        self.prev_btn = GazeButton("", "special")
        self.next_btn = GazeButton("", "special")
        self.back_btn.clicked.connect(self.back.emit)
        self.save_btn.clicked.connect(self.save_requested.emit)
        self.delete_btn.clicked.connect(self._toggle_delete)
        self.prev_btn.clicked.connect(lambda: self._turn(-1))
        self.next_btn.clicked.connect(lambda: self._turn(1))
        top = QWidget()
        top.setLayout(_hbox(self.back_btn, self.save_btn, self.delete_btn, self.prev_btn, self.next_btn,
                            stretches=[1, 2, 2, 1, 1]))
        root.addWidget(top, 1)
        self.hint = QLabel("")
        self.hint.setObjectName("Status")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.hint)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.phrase_buttons: list[GazeButton] = []
        for i in range(PHRASES_PER_PAGE):
            b = GazeButton("", "suggestion")
            b.clicked.connect(lambda _=False, i=i: self._chosen(i))
            grid.addWidget(b, i // 2, i % 2)
            self.phrase_buttons.append(b)
        root.addLayout(grid, 4)
        self.retranslate()

    def set_phrases(self, phrases: list[dict[str, Any]]) -> None:
        self.phrases = phrases
        self.page = min(self.page, max(0, (len(phrases) - 1) // PHRASES_PER_PAGE))
        self._refresh()

    def _turn(self, direction: int) -> None:
        pages = max(1, (len(self.phrases) + PHRASES_PER_PAGE - 1) // PHRASES_PER_PAGE)
        self.page = (self.page + direction) % pages
        self._refresh()

    def _toggle_delete(self) -> None:
        self.delete_mode = not self.delete_mode
        self.retranslate()

    def _chosen(self, slot: int) -> None:
        idx = self.page * PHRASES_PER_PAGE + slot
        if idx >= len(self.phrases):
            return
        phrase = self.phrases[idx]
        if self.delete_mode:
            self.delete_requested.emit(phrase["id"], phrase["text"])
        else:
            self.phrase_chosen.emit(phrase["text"])

    def _refresh(self) -> None:
        start = self.page * PHRASES_PER_PAGE
        for i, b in enumerate(self.phrase_buttons):
            idx = start + i
            if idx < len(self.phrases):
                p = self.phrases[idx]
                b.setText(p["text"])
                b.setEnabled(True)
                b.set_role("emergency" if p["is_emergency"] else "suggestion")
            else:
                b.setText("")
                b.setEnabled(False)
                b.set_role("suggestion")
        many = len(self.phrases) > PHRASES_PER_PAGE
        self.prev_btn.setEnabled(many)
        self.next_btn.setEnabled(many)

    def retranslate(self) -> None:
        lang = self.lang
        self.back_btn.setText(tr("back", lang))
        self.save_btn.setText(tr("save_phrase", lang))
        self.delete_btn.setText(tr("delete_mode", lang))
        self.delete_btn.set_role("active" if self.delete_mode else "special")
        self.prev_btn.setText(tr("prev_page", lang))
        self.next_btn.setText(tr("next_page", lang))
        self.hint.setText(tr("delete_mode_on", lang) if self.delete_mode else tr("phrases_title", lang))
        self._refresh()


# -------------------------------------------------------------------- settings
class SettingsPage(QWidget):
    back = Signal()
    command = Signal(str)  # calibrate, recenter, diagnostics, export, import, reset_settings, reset_data

    def __init__(self, get_value: Callable[[str], Any], step_value: Callable[[str, int], None],
                 voice_names: Callable[[], list[str]], parent=None):
        super().__init__(parent)
        self.setObjectName("Page")
        self._get = get_value
        self._step = step_value
        self._voice_names = voice_names
        self.lang = "en"
        self.group = SETTING_GROUPS[0]
        self.page = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 12)
        root.setSpacing(6)

        self.back_btn = GazeButton("", "special")
        self.back_btn.clicked.connect(self.back.emit)
        self.group_buttons: dict[str, GazeButton] = {}
        top_widgets = [self.back_btn]
        for g in SETTING_GROUPS:
            b = GazeButton("", "special")
            b.clicked.connect(lambda _=False, g=g: self._select_group(g))
            self.group_buttons[g] = b
            top_widgets.append(b)
        top = QWidget()
        top.setLayout(_hbox(*top_widgets))
        root.addWidget(top, 1)

        self.rows: list[tuple[QLabel, GazeButton, QLabel, GazeButton]] = []
        rows_box = QVBoxLayout()
        rows_box.setSpacing(6)
        for i in range(SETTINGS_ROWS_PER_PAGE):
            label = QLabel("")
            label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            minus = GazeButton("◀", "special")
            value = QLabel("")
            value.setObjectName("Value")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            plus = GazeButton("▶", "special")
            minus.clicked.connect(lambda _=False, i=i: self._step_row(i, -1))
            plus.clicked.connect(lambda _=False, i=i: self._step_row(i, 1))
            rows_box.addLayout(_hbox(label, minus, value, plus, stretches=[3, 1, 2, 1]), 1)
            self.rows.append((label, minus, value, plus))
        root.addLayout(rows_box, 5)

        self.prev_btn = GazeButton("", "special")
        self.next_btn = GazeButton("", "special")
        self.prev_btn.clicked.connect(lambda: self._turn(-1))
        self.next_btn.clicked.connect(lambda: self._turn(1))
        self.command_buttons: dict[str, GazeButton] = {}
        bottom = [self.prev_btn, self.next_btn]
        for name in ("calibrate", "calibrate_head", "recenter", "diagnostics", "export", "import", "reset_settings",
                     "reset_data"):
            b = GazeButton("", "emergency" if name == "reset_data" else "special")
            b.clicked.connect(lambda _=False, n=name: self.command.emit(n))
            self.command_buttons[name] = b
            bottom.append(b)
        bw = QWidget()
        bw.setLayout(_hbox(*bottom))
        root.addWidget(bw, 1)
        self.retranslate()

    def _specs(self):
        return [s for s in SETTING_SPECS if s.group == self.group]

    def _select_group(self, group: str) -> None:
        self.group, self.page = group, 0
        self.refresh()

    def _turn(self, direction: int) -> None:
        pages = max(1, (len(self._specs()) + SETTINGS_ROWS_PER_PAGE - 1) // SETTINGS_ROWS_PER_PAGE)
        self.page = (self.page + direction) % pages
        self.refresh()

    def _step_row(self, row: int, direction: int) -> None:
        specs = self._specs()
        idx = self.page * SETTINGS_ROWS_PER_PAGE + row
        if idx < len(specs):
            self._step(specs[idx].key, direction)
            self.refresh()

    def format_value(self, key: str, value: Any) -> str:
        lang = self.lang
        if isinstance(value, bool):
            return tr("val_on" if value else "val_off", lang)
        if key == "language":
            return LANGUAGE_NAMES.get(value, value)
        if key == "input_source" and value in ("both", "gaze", "mouse"):
            return tr(f"val_{value}", lang)
        if key == "speech_voice":
            if value == 0:
                return tr("val_auto", lang)
            names = self._voice_names()
            return names[(value - 1) % len(names)] if names else tr("val_auto", lang)
        if key in ("speech_rate", "speech_volume", "sensitivity", "key_scale", "confidence_threshold", "head_gain"):
            return f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
        if isinstance(value, str) and f"val_{value}" in _VALUE_KEYS:
            return tr(f"val_{value}", lang)
        return str(value)

    def refresh(self) -> None:
        specs = self._specs()
        start = self.page * SETTINGS_ROWS_PER_PAGE
        for i, (label, minus, value, plus) in enumerate(self.rows):
            idx = start + i
            visible = idx < len(specs)
            for w in (label, minus, value, plus):
                w.setVisible(visible)
            if visible:
                spec = specs[idx]
                label.setText(tr(f"set_{spec.key}", self.lang))
                value.setText(self.format_value(spec.key, self._get(spec.key)))
                minus.setText("◀" if spec.choices or spec.is_bool else "−")
                plus.setText("▶" if spec.choices or spec.is_bool else "+")
        many = len(specs) > SETTINGS_ROWS_PER_PAGE
        self.prev_btn.setEnabled(many)
        self.next_btn.setEnabled(many)
        for g, b in self.group_buttons.items():
            b.set_role("active" if g == self.group else "special")

    def retranslate(self) -> None:
        lang = self.lang
        self.back_btn.setText(tr("back", lang))
        for g, b in self.group_buttons.items():
            b.setText(tr(f"group_{g}", lang))
        self.prev_btn.setText(tr("prev_page", lang))
        self.next_btn.setText(tr("next_page", lang))
        for name, b in self.command_buttons.items():
            b.setText(tr(name, lang))
        self.refresh()


_VALUE_KEYS = {
    "val_day", "val_night", "val_normal", "val_high", "val_letters", "val_words", "val_qwerty", "val_frequency",
    "val_both", "val_left", "val_right", "val_all", "val_sentence", "val_eyes_head", "val_eyes", "val_head",
}


# ----------------------------------------------------------------- diagnostics
class DiagnosticsPage(QWidget):
    back = Signal()

    FIELDS = ("diag_camera", "diag_face", "diag_iris", "diag_gaze", "diag_confidence", "diag_eye_open",
              "diag_capture_fps", "diag_process_fps", "diag_ui_fps", "diag_inference", "diag_dropped",
              "diag_calibration")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Page")
        self.lang = "en"
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 12)
        root.setSpacing(12)
        left = QVBoxLayout()
        self.back_btn = GazeButton("", "special")
        self.back_btn.clicked.connect(self.back.emit)
        self.back_btn.setMaximumHeight(110)
        left.addWidget(self.back_btn, 1)
        self.preview = QLabel("")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        left.addWidget(self.preview, 4)
        root.addLayout(left, 2)
        grid = QGridLayout()
        self.labels: dict[str, tuple[QLabel, QLabel]] = {}
        for row, key in enumerate(self.FIELDS):
            name, value = QLabel(""), QLabel("—")
            value.setObjectName("Value")
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            self.labels[key] = (name, value)
        root.addLayout(grid, 3)
        self.retranslate()

    def set_preview(self, image: QImage | None) -> None:
        if image is None:
            self.preview.clear()
        else:
            self.preview.setPixmap(QPixmap.fromImage(image).scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def set_values(self, values: dict[str, str]) -> None:
        for key, text in values.items():
            if key in self.labels:
                self.labels[key][1].setText(text)

    def retranslate(self) -> None:
        self.back_btn.setText(tr("back", self.lang))
        for key, (name, _) in self.labels.items():
            name.setText(tr(key, self.lang))


# ---------------------------------------------------------------- confirmation
class ConfirmOverlay(QWidget):
    """Full-window yes/no question. Blocks everything underneath while shown."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("Overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self._callback: Callable[[bool], None] | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(60, 60, 60, 60)
        root.setSpacing(30)
        self.message = QLabel("")
        self.message.setObjectName("Message")
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.message, 2)
        self.yes_btn = GazeButton("", "key")
        self.no_btn = GazeButton("", "key")
        self.yes_btn.clicked.connect(lambda: self._answer(True))
        self.no_btn.clicked.connect(lambda: self._answer(False))
        # Buttons far apart to avoid an accidental "yes".
        root.addLayout(_hbox(self.no_btn, QWidget(), self.yes_btn, stretches=[2, 1, 2], spacing=40), 2)
        self.hide()

    def ask(self, message: str, lang: str, callback: Callable[[bool], None], yes_key: str = "yes",
            no_key: str = "no") -> None:
        self.message.setText(message)
        self.yes_btn.setText(tr(yes_key, lang))
        self.no_btn.setText(tr(no_key, lang))
        self.no_btn.setVisible(bool(no_key))
        self._callback = callback
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()

    def _answer(self, value: bool) -> None:
        self.hide()
        callback, self._callback = self._callback, None
        if callback:
            callback(value)
