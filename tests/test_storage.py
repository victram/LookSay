"""Integration tests for the SQLite layer."""

import json
import sqlite3

from looksay.config import DEFAULTS
from looksay.storage.database import MIGRATIONS, Database
from looksay.storage.settings import Settings


def test_fresh_database_is_fully_migrated(db):
    assert db.schema_version == len(MIGRATIONS)
    tables = {r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"settings", "saved_phrases", "personal_vocabulary", "word_bigrams", "calibration_profiles"} <= tables


def test_migration_from_v1(tmp_path):
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO schema_version VALUES (1)")
    conn.executescript(MIGRATIONS[0])
    conn.execute("INSERT INTO settings VALUES ('dwell_ms', '1500')")
    conn.close()
    db = Database(path)
    assert db.schema_version == len(MIGRATIONS)
    assert db.load_settings()["dwell_ms"] == 1500
    db.close()


def test_settings_persist_across_restart(tmp_path):
    path = tmp_path / "s.sqlite3"
    db = Database(path)
    s = Settings(db)
    s.set("language", "ca")
    s.set("dwell_ms", 1600)
    s.set("theme", "night")
    db.close()
    db2 = Database(path)
    s2 = Settings(db2)
    assert (s2["language"], s2["dwell_ms"], s2["theme"]) == ("ca", 1600, "night")
    db2.close()


def test_invalid_stored_values_fall_back_to_valid(db):
    db.save_setting("dwell_ms", 99999)
    db.save_setting("language", "xx")
    db.conn.execute("INSERT OR REPLACE INTO settings VALUES ('theme', 'not json')")
    values = db.load_settings()
    assert values["dwell_ms"] == 3000
    assert values["language"] == DEFAULTS["language"]
    assert values["theme"] == DEFAULTS["theme"]


def test_settings_listeners_and_step(db):
    s = Settings(db)
    seen = []
    s.subscribe(lambda k, v: seen.append((k, v)))
    s.step("dwell_ms", 1)
    s.step("language", 1)
    assert seen == [("dwell_ms", 1100), ("language", "es")]


def test_phrases_crud_and_seed_once(db):
    db.seed_phrases("en", [("I need help", True), ("Yes", False)])
    db.seed_phrases("en", [("Other", False)])
    texts = [p["text"] for p in db.phrases("en")]
    assert texts == ["I need help", "Yes"]
    db.add_phrase("en", "Good morning")
    db.add_phrase("en", "Good morning")  # duplicate ignored
    phrases = db.phrases("en")
    assert len(phrases) == 3 and phrases[0]["is_emergency"]
    db.delete_phrase(phrases[-1]["id"])
    assert len(db.phrases("en")) == 2
    # Deleting all seeded phrases must not bring them back on next start.
    for p in db.phrases("en"):
        db.delete_phrase(p["id"])
    db.seed_phrases("en", [("I need help", True)])
    assert db.phrases("en") == []


def test_learning_is_capped(db):
    for _ in range(600):
        db.record_word("en", "hello", "say")
    assert db.personal_vocabulary("en")["hello"][0] == 500
    assert db.personal_bigrams("en")[("say", "hello")][0] == 500


def test_calibration_profiles(db):
    assert db.active_calibration() is None
    db.save_calibration("a", 1, 9, (1920, 1080), 55.0, {"version": 1, "x": 1})
    db.save_calibration("b", 1, 5, (1920, 1080), 40.0, {"version": 1, "x": 2})
    active = db.active_calibration()
    assert active["name"] == "b" and active["model"]["x"] == 2 and active["n_points"] == 5


def test_export_import_roundtrip(tmp_path, db):
    s = Settings(db)
    s.set("language", "es")
    db.add_phrase("es", "Hola amigo")
    path = db.export_to_file(tmp_path / "export.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["settings"]["language"] == "es"
    other = Database(tmp_path / "other.sqlite3")
    other.import_from_file(path)
    assert other.load_settings()["language"] == "es"
    assert [p["text"] for p in other.phrases("es")] == ["Hola amigo"]
    other.close()


def test_import_rejects_foreign_files(db):
    import pytest

    with pytest.raises(ValueError):
        db.import_data({"format": "something-else"})


def test_reset_all_wipes_user_data(db):
    db.save_setting("dwell_ms", 2000)
    db.add_phrase("en", "x")
    db.record_word("en", "abc", None)
    db.save_calibration("a", 1, 9, (1, 1), 1.0, {})
    db.reset_all()
    assert db.load_settings()["dwell_ms"] == DEFAULTS["dwell_ms"]
    assert db.phrases("en") == [] and db.personal_vocabulary("en") == {} and db.active_calibration() is None


def test_corrupted_database_is_replaced(tmp_path):
    path = tmp_path / "bad.sqlite3"
    path.write_bytes(b"this is definitely not a sqlite database" * 100)
    db = Database(path)
    assert db.recovered_from_corruption
    assert db.schema_version == len(MIGRATIONS)
    assert list(tmp_path.glob("bad.sqlite3.corrupt-*"))
    db.close()
