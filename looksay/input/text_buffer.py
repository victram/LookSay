"""The message being composed, with undo and word-aware editing.

The caret always sits at the end of the text: with gaze input, arbitrary
caret placement is slow and error-prone, and appending/deleting at the end
covers composing and correcting a message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from looksay.prediction.engine import TOKEN_RE, tokenize

SENTENCE_END = ".!?"
CLOSING_PUNCT = ".,!?;:)"
MAX_UNDO = 200

CommitListener = Callable[[str, "str | None"], None]


@dataclass
class TextBuffer:
    text: str = ""
    auto_capitalize: bool = True
    _undo: list[str] = field(default_factory=list)
    _shift_once: bool = False
    on_word_committed: CommitListener | None = None

    # ------------------------------------------------------------- queries
    def current_prefix(self) -> str:
        """The partial word right before the caret ('' after a space or punctuation)."""
        match = re.search(r"(?:" + TOKEN_RE.pattern + r"|[^\W\d_]+['’·]?)$", self.text)
        return match.group(0) if match else ""

    def previous_word(self) -> str | None:
        """The last complete word before the current prefix, within the current sentence."""
        body = self.text[: len(self.text) - len(self.current_prefix())]
        tail = re.split(r"[.!?]\s*", body)[-1]
        words = tokenize(tail)
        return words[-1] if words else None

    def at_sentence_start(self) -> bool:
        stripped = self.text[: len(self.text) - len(self.current_prefix())].rstrip()
        return not stripped or stripped[-1] in SENTENCE_END

    def last_sentence(self) -> str:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", self.text.strip()) if p.strip()]
        return parts[-1] if parts else ""

    @property
    def shift_active(self) -> bool:
        return self._shift_once

    # ------------------------------------------------------------- editing
    def _snapshot(self) -> None:
        self._undo.append(self.text)
        if len(self._undo) > MAX_UNDO:
            del self._undo[0]

    def _commit_current_word(self) -> None:
        word = self.current_prefix()
        if word and self.on_word_committed:
            self.on_word_committed(word, self.previous_word())

    def toggle_shift(self) -> None:
        self._shift_once = not self._shift_once

    def insert_char(self, ch: str) -> None:
        self._snapshot()
        if ch.isalpha():
            if self._shift_once or (self.auto_capitalize and not self.current_prefix() and self.at_sentence_start()):
                ch = ch.upper()
            self._shift_once = False
            self.text += ch
        elif ch in CLOSING_PUNCT:
            self.insert_punctuation(ch, snapshot=False)
        else:
            self.text += ch

    def insert_punctuation(self, mark: str, snapshot: bool = True) -> None:
        if snapshot:
            self._snapshot()
        self._commit_current_word()
        self.text = self.text.rstrip(" ") + mark + " "

    def space(self) -> None:
        self._snapshot()
        self._commit_current_word()
        if not self.text.endswith(" "):
            self.text += " "

    def accept_word(self, word: str) -> None:
        """Replace the partial word with a suggestion and add a space."""
        self._snapshot()
        prefix = self.current_prefix()
        base = self.text[: len(self.text) - len(prefix)]
        if self.auto_capitalize and self._sentence_start_before(base):
            word = word[:1].upper() + word[1:]
        elif prefix[:1].isupper():
            word = word[:1].upper() + word[1:]
        self._shift_once = False
        if base and not base.endswith((" ", "'", "’", "(", "\n")):
            base += " "
        self.text = base + word
        self._commit_current_word()
        self.text += "" if word.endswith(("'", "’")) else " "

    def insert_phrase(self, phrase: str) -> None:
        self._snapshot()
        base = self.text
        if base and not base.endswith(" "):
            base += " "
        self.text = base + phrase.strip() + " "

    @staticmethod
    def _sentence_start_before(base: str) -> bool:
        stripped = base.rstrip()
        return not stripped or stripped[-1] in SENTENCE_END

    def backspace(self) -> None:
        if not self.text:
            return
        self._snapshot()
        self.text = self.text[:-1]

    def delete_word(self) -> None:
        if not self.text:
            return
        self._snapshot()
        self.text = re.sub(r"\S*\s*$", "", self.text, count=1)

    def clear(self) -> None:
        if not self.text:
            return
        self._snapshot()
        self.text = ""

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.text = self._undo.pop()
        return True
