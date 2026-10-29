"""The on-screen keyboard: suggestions, letter/word modes and control keys."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget

from looksay.i18n.layouts import letter_rows, symbol_rows
from looksay.i18n.strings import tr
from looksay.ui.widgets import GazeButton

WORD_GRID_ROWS = 3
WORD_GRID_COLS = 5


def _row(buttons: list[GazeButton], stretches: list[int] | None = None) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setSpacing(10)
    layout.setContentsMargins(0, 0, 0, 0)
    for i, b in enumerate(buttons):
        layout.addWidget(b, stretches[i] if stretches else 1)
    return layout


class KeyboardPanel(QWidget):
    """Emits action(name, value) for every key; the main window applies it."""

    action = Signal(str, str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Page")
        self.lang = "en"
        self.layout_name = "qwerty"
        self.mode = "letters"
        self._symbols = False

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 6, 40, 32)
        root.setSpacing(12)

        self.suggestion_buttons = [self._button("", "suggestion", "suggest") for _ in range(6)]
        self.suggestion_row = QWidget()
        self.suggestion_row.setLayout(_row(self.suggestion_buttons))
        root.addWidget(self.suggestion_row, 1)

        self.stack = QStackedWidget()
        self.letters_page = QWidget()
        self.letters_layout = QVBoxLayout(self.letters_page)
        self.letters_layout.setContentsMargins(0, 0, 0, 0)
        self.letters_layout.setSpacing(10)
        self.words_page = QWidget()
        words_layout = QVBoxLayout(self.words_page)
        words_layout.setContentsMargins(0, 0, 0, 0)
        words_layout.setSpacing(10)
        self.word_buttons: list[GazeButton] = []
        for _ in range(WORD_GRID_ROWS):
            row = [self._button("", "suggestion", "suggest") for _ in range(WORD_GRID_COLS)]
            self.word_buttons += row
            words_layout.addLayout(_row(row), 1)
        self.stack.addWidget(self.letters_page)
        self.stack.addWidget(self.words_page)
        root.addWidget(self.stack, 3)

        # One control row, as in the "Papel" design: space, delete, phrases, settings, Speak, HELP.
        # Speak turns into Stop while the voice is talking. The rarer editing tools (shift, symbols,
        # delete word, undo, word mode, clear) live in Settings, so the main screen stays calm.
        self.space_btn = self._button("", "special", "space")
        self.backspace_btn = self._button("", "special", "backspace")
        self.phrases_btn = self._button("", "special", "phrases")
        self.settings_btn = self._button("", "special", "settings")
        self.speak_btn = self._button("", "primary", "speak")
        self.help_btn = self._button("", "emergency", "help")
        control_row = QWidget()
        control_row.setLayout(_row(
            [self.space_btn, self.backspace_btn, self.phrases_btn, self.settings_btn, self.speak_btn, self.help_btn],
            [3, 1, 1, 1, 2, 1],
        ))
        root.addWidget(control_row, 1)
        self.speaking = False

        self._letter_buttons: list[GazeButton] = []
        self.rebuild_letters()
        self.retranslate()

    # ---------------------------------------------------------------- build
    def _button(self, text: str, role: str, name: str) -> GazeButton:
        button = GazeButton(text, role)
        button.clicked.connect(lambda: self.action.emit(name, button.property("value") or button.text()))
        return button

    def rebuild_letters(self) -> None:
        while self.letters_layout.count():
            item = self.letters_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    w = item.layout().takeAt(0).widget()
                    if w:
                        w.hide()
                        w.setParent(None)
                        w.deleteLater()
        self._letter_buttons = []
        rows = symbol_rows(self.lang) if self._symbols else letter_rows(self.lang, self.layout_name)
        for keys in rows:
            buttons = []
            for key in keys:
                b = self._button(key, "key", "char")
                b.setProperty("base", key)
                b.setProperty("value", key)
                buttons.append(b)
            self._letter_buttons += buttons
            self.letters_layout.addLayout(_row(buttons), 1)
        self.update_case(False)

    def update_case(self, upper: bool) -> None:
        for b in self._letter_buttons:
            base = b.property("base")
            b.setText(base.upper() if upper and base.isalpha() else base)
            b.setProperty("value", b.text())

    def set_speaking(self, speaking: bool) -> None:
        if speaking != self.speaking:
            self.speaking = speaking
            self.speak_btn.setText(tr("stop" if speaking else "speak", self.lang))

    # ------------------------------------------------------------ state
    def set_language(self, lang: str) -> None:
        self.lang = lang
        self.rebuild_letters()
        self.retranslate()

    def set_layout(self, layout_name: str) -> None:
        self.layout_name = layout_name
        self.rebuild_letters()

    def toggle_symbols(self) -> None:
        self._symbols = not self._symbols
        self.rebuild_letters()
        self.retranslate()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.stack.setCurrentWidget(self.words_page if mode == "words" else self.letters_page)
        self.retranslate()

    def set_suggestions(self, words: list[str], count: int) -> None:
        for i, b in enumerate(self.suggestion_buttons):
            b.setVisible(i < count)
            text = words[i] if i < len(words) else ""
            b.setText(text)
            b.setProperty("value", text)
            b.setEnabled(bool(text))

    def set_word_grid(self, words: list[str]) -> None:
        for i, b in enumerate(self.word_buttons):
            text = words[i] if i < len(words) else ""
            b.setText(text)
            b.setProperty("value", text)
            b.setEnabled(bool(text))

    def retranslate(self) -> None:
        lang = self.lang
        self.space_btn.setText(tr("space", lang))
        self.backspace_btn.setText(tr("delete", lang))
        self.phrases_btn.setText(tr("phrases", lang))
        self.settings_btn.setText(tr("settings", lang))
        self.speak_btn.setText(tr("stop" if self.speaking else "speak", lang))
        self.help_btn.setText(tr("help", lang))
