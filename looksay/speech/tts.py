"""Offline text-to-speech adapters.

Preferred backend is Qt's QTextToSpeech, which wraps the operating system's
own offline voices (SAPI / WinRT on Windows, AVSpeechSynthesizer on macOS,
speech-dispatcher on Linux) and never blocks the UI. pyttsx3 is a fallback,
run on its own thread. If neither works, NullSpeech keeps the app usable.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Protocol

from looksay.i18n.strings import LOCALES

log = logging.getLogger(__name__)


class SpeechEngine(Protocol):
    name: str

    @property
    def available(self) -> bool: ...
    def set_language(self, lang: str) -> None: ...
    def voices(self) -> list[str]: ...
    def set_voice(self, index: int) -> None: ...
    def set_rate(self, rate: float) -> None: ...  # -1 .. 1
    def set_volume(self, volume: float) -> None: ...  # 0 .. 1
    def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...
    def is_speaking(self) -> bool: ...


class NullSpeech:
    name = "none"
    available = False

    def __init__(self):
        self.spoken: list[str] = []

    def set_language(self, lang: str) -> None: ...
    def voices(self) -> list[str]: return []
    def set_voice(self, index: int) -> None: ...
    def set_rate(self, rate: float) -> None: ...
    def set_volume(self, volume: float) -> None: ...
    def speak(self, text: str) -> None: self.spoken.append(text)
    def stop(self) -> None: ...
    def is_speaking(self) -> bool: return False


class QtSpeech:
    """QTextToSpeech wrapper. `voices()` lists voices for the current language."""

    def __init__(self, engine: str | None = None):
        from PySide6.QtTextToSpeech import QTextToSpeech

        self._cls = QTextToSpeech
        self._tts = QTextToSpeech(engine) if engine else QTextToSpeech()
        self.name = f"qt:{self._tts.engine() or engine or 'default'}"
        self._lang = "en"
        self._voices: list = []
        self._voice_index = 0

    @property
    def available(self) -> bool:
        return self._tts.state() != self._cls.State.Error

    def set_language(self, lang: str) -> None:
        from PySide6.QtCore import QLocale

        self._lang = lang
        locale = QLocale(LOCALES.get(lang, "en-US"))
        self._tts.setLocale(locale)
        self._voices = [v for v in self._tts.availableVoices() if v.locale().language() == locale.language()]
        self.set_voice(self._voice_index)

    def voices(self) -> list[str]:
        return [v.name() for v in self._voices]

    def set_voice(self, index: int) -> None:
        """index 0 = automatic (first voice for the language), n = n-th voice."""
        self._voice_index = index
        if self._voices:
            self._tts.setVoice(self._voices[max(0, index - 1) % len(self._voices)])

    def set_rate(self, rate: float) -> None:
        self._tts.setRate(max(-1.0, min(1.0, rate)))

    def set_volume(self, volume: float) -> None:
        self._tts.setVolume(max(0.0, min(1.0, volume)))

    def speak(self, text: str) -> None:
        if text.strip():
            self._tts.stop()
            self._tts.say(text)

    def stop(self) -> None:
        self._tts.stop()

    def is_speaking(self) -> bool:
        return self._tts.state() == self._cls.State.Speaking


class Pyttsx3Speech:
    """pyttsx3 on a dedicated thread (its runAndWait() blocks)."""

    name = "pyttsx3"

    def __init__(self):
        import pyttsx3

        self._engine = pyttsx3.init()
        self._all_voices = list(self._engine.getProperty("voices") or [])
        self._voices = self._all_voices
        self._voice_index = 0
        self._queue: queue.Queue = queue.Queue()
        self._speaking = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="looksay-tts", daemon=True)
        self._thread.start()

    available = True

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            kind, value = item
            try:
                if kind == "say":
                    self._speaking.set()
                    self._engine.say(value)
                    self._engine.runAndWait()
                else:
                    self._engine.setProperty(kind, value)
            except Exception as exc:  # driver errors must not kill the app
                log.warning("pyttsx3 error: %s", exc)
            finally:
                self._speaking.clear()

    def set_language(self, lang: str) -> None:
        def matches(voice) -> bool:
            langs = [l.decode() if isinstance(l, bytes) else str(l) for l in (getattr(voice, "languages", []) or [])]
            text = " ".join(langs + [voice.id or "", voice.name or ""]).lower()
            return lang in text.replace("_", "-").split("-") or f"{lang}-" in text or f"/{lang}" in text
        self._voices = [v for v in self._all_voices if matches(v)]
        self.set_voice(self._voice_index)

    def voices(self) -> list[str]:
        return [v.name for v in self._voices]

    def set_voice(self, index: int) -> None:
        self._voice_index = index
        if self._voices:
            self._queue.put(("voice", self._voices[max(0, index - 1) % len(self._voices)].id))

    def set_rate(self, rate: float) -> None:
        self._queue.put(("rate", int(175 * (1.0 + 0.6 * rate))))

    def set_volume(self, volume: float) -> None:
        self._queue.put(("volume", max(0.0, min(1.0, volume))))

    def speak(self, text: str) -> None:
        if text.strip():
            self._queue.put(("say", text))

    def stop(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._engine.stop()
        except Exception:
            pass

    def is_speaking(self) -> bool:
        return self._speaking.is_set()


def create_speech(engine: str | None = None) -> SpeechEngine:
    """Pick the best working backend. `engine` forces a Qt engine (e.g. 'mock' in tests)."""
    try:
        from PySide6.QtTextToSpeech import QTextToSpeech

        engines = [e for e in QTextToSpeech.availableEngines() if e != "mock"]
        if engine or engines:
            speech = QtSpeech(engine or engines[0])
            if speech.available:
                return speech
    except Exception as exc:
        log.info("Qt speech unavailable: %s", exc)
    try:
        return Pyttsx3Speech()
    except Exception as exc:
        log.info("pyttsx3 unavailable: %s", exc)
    return NullSpeech()
