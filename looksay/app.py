"""Application entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from looksay import APP_NAME, __version__


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="looksay", description="Offline gaze-based communication.")
    parser.add_argument("--mouse", action="store_true", help="do not start the camera (mouse/dwell simulation)")
    parser.add_argument("--fullscreen", action="store_true", help="start in full screen (F11 toggles)")
    parser.add_argument("--data-dir", help="store settings and learning data in this folder")
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.data_dir:
        os.environ["LOOKSAY_DATA_DIR"] = args.data_dir

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from looksay.paths import database_path, resources_dir
    from looksay.prediction.engine import Predictor
    from looksay.speech.tts import create_speech
    from looksay.storage.database import Database
    from looksay.storage.settings import Settings
    from looksay.ui.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    from PySide6.QtGui import QFont, QFontDatabase

    for font_file in sorted((resources_dir() / "fonts").glob("*.otf")):
        QFontDatabase.addApplicationFont(str(font_file))  # bundled Inter (OFL), same look on every OS
    app.setFont(QFont("Inter", 12))
    icon = resources_dir() / "icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))

    logging.getLogger(__name__).info("%s %s starting; data in %s", APP_NAME, __version__, database_path().parent)
    db = Database(database_path())
    settings = Settings(db)
    predictor = Predictor(resources_dir() / "dictionaries", store=db)
    window = MainWindow(settings, db, predictor, create_speech(), enable_tracking=not args.mouse)
    if args.fullscreen:
        window.showFullScreen()
    else:
        window.showMaximized()
    code = app.exec()
    db.close()
    return code
