"""Render the README screenshot of the real UI (headless, no camera needed).

    python tools/screenshots.py            # -> docs/screenshots/main.png
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LOOKSAY_DATA_DIR", tempfile.mkdtemp(prefix="looksay-shots-"))

OUT = ROOT / "docs" / "screenshots"
SIZE = (1280, 800)


def main() -> None:
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    from looksay.paths import resources_dir

    for font_file in sorted((resources_dir() / "fonts").glob("*.otf")):
        QFontDatabase.addApplicationFont(str(font_file))
    app.setFont(QFont("Inter", 12))

    from looksay.prediction.engine import Predictor
    from looksay.speech.tts import NullSpeech
    from looksay.storage.database import Database
    from looksay.storage.settings import Settings
    from looksay.ui.main_window import MainWindow

    OUT.mkdir(parents=True, exist_ok=True)
    db = Database(Path(os.environ["LOOKSAY_DATA_DIR"]) / "shots.sqlite3")
    settings = Settings(db)
    settings.set("language", "es")
    w = MainWindow(settings, db, Predictor(resources_dir() / "dictionaries", store=db), NullSpeech(),
                   enable_tracking=False)
    w.resize(*SIZE)
    w.show()
    w.timer.stop()

    def shot(name: str) -> None:
        w._update_status("mouse", force=True)
        for _ in range(3):
            app.processEvents()
        w.grab().save(str(OUT / f"{name}.png"))
        print("saved", OUT / f"{name}.png")

    def key(value):
        return next(b for b in w.keyboard._letter_buttons if b.property("base") == value)

    for ch in "necesito un vaso de ag":
        w._on_key("space" if ch == " " else "char", ch)
    key("u").set_dwell(True, 0.6)
    shot("main")

    w.close()
    db.close()


if __name__ == "__main__":
    main()
