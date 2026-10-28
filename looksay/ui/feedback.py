"""Optional, soft audio click on selection (generated locally, no assets needed)."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from looksay.paths import data_dir


def _write_click(path: Path) -> None:
    rate, duration, freq = 22050, 0.06, 660.0
    frames = bytearray()
    n = int(rate * duration)
    for i in range(n):
        envelope = math.sin(math.pi * i / n)  # smooth in/out, no harsh edges
        frames += struct.pack("<h", int(9000 * envelope * math.sin(2 * math.pi * freq * i / rate)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


class ClickSound:
    def __init__(self):
        self._effect = None
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtMultimedia import QSoundEffect

            path = data_dir() / "click.wav"
            if not path.exists():
                _write_click(path)
            self._effect = QSoundEffect()
            self._effect.setSource(QUrl.fromLocalFile(str(path)))
            self._effect.setVolume(0.5)
        except Exception:
            self._effect = None

    def play(self) -> None:
        if self._effect is not None:
            self._effect.play()
