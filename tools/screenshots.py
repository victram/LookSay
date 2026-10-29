"""Render README screenshots of the real UI (headless, no camera needed).

    python tools/screenshots.py            # -> docs/screenshots/*.png
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
    import numpy as np
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    from looksay.paths import resources_dir

    for font_file in sorted((resources_dir() / "fonts").glob("*.otf")):
        QFontDatabase.addApplicationFont(str(font_file))
    app.setFont(QFont("Inter", 12))

    from looksay.calibration.model import CalibrationSession, calibration_targets
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

    settings.set("theme", "night")
    shot("main-night")
    settings.set("theme", "day")
    key("u").set_dwell(False, 0.0)

    w.buffer.clear()
    w.buffer.insert_phrase("Quiero")
    w.refresh()
    settings.set("keyboard_mode", "words")
    w.keyboard.word_buttons[2].set_dwell(True, 0.4)
    shot("words")
    w.keyboard.word_buttons[2].set_dwell(False, 0.0)
    settings.set("keyboard_mode", "letters")

    w._on_key("phrases", "")
    shot("phrases")
    w.show_keyboard()
    w._on_key("settings", "")
    w.settings_page._select_group("tracking")
    shot("settings")
    w.show_keyboard()

    # Calibration result screen with a cross-validated point map (synthetic, realistic error).
    rng = np.random.default_rng(4)
    targets = calibration_targets(13)
    session = CalibrationSession(targets)
    for i, (tx, ty) in enumerate(targets):
        for _ in range(25):
            # Noise and a slight curvature typical of a webcam calibration.
            h = -0.1 + 0.2 * tx + 0.03 * (ty - 0.5) ** 2 + rng.normal(0, 0.012)
            v = -0.05 + 0.1 * ty + 0.02 * (tx - 0.5) ** 2 + rng.normal(0, 0.008)
            session.add_sample(i, np.array([h, v, h, v, 0.3 - 0.05 * ty, h, v, 0, 0, 0, 0, 0, -45])
                               + np.r_[np.zeros(7), rng.normal(0, 0.3, 6)])
    calib = w.calibration
    calib.lang = "es"
    calib.start(13, "both")
    calib._timer.stop()
    calib.session = session
    calib.index = len(targets) - 1
    calib.pursuit_enabled = False
    calib._next_point()
    shot("calibration")
    calib._cancel()

    w.close()
    db.close()


if __name__ == "__main__":
    main()
