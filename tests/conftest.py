import os
import sys
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LOOKSAY_DATA_DIR", tempfile.mkdtemp(prefix="looksay-test-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looksay.paths import resources_dir  # noqa: E402

DICTS = resources_dir() / "dictionaries"


@pytest.fixture(scope="session")
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def db(tmp_path):
    from looksay.storage.database import Database

    database = Database(tmp_path / "test.sqlite3")
    yield database
    database.close()
