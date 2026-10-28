"""Local SQLite persistence with versioned migrations.

Everything LookSay remembers lives in one file: settings, saved phrases,
personal vocabulary, word bigrams and calibration profiles. No camera data
is ever stored. A corrupted file is moved aside and replaced with a fresh
database so the app can always start.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from looksay.config import DEFAULTS, SPECS_BY_KEY

log = logging.getLogger(__name__)

EXPORT_FORMAT = "looksay-export"
EXPORT_VERSION = 1

# Caps that keep personal learning bounded (prediction stays predictable and
# the database stays small).
MAX_WORD_COUNT = 500
MAX_PERSONAL_WORDS = 20_000
MAX_BIGRAMS = 50_000

MIGRATIONS: list[str] = [
    # v1: initial schema
    """
    CREATE TABLE settings (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    CREATE TABLE saved_phrases (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        lang         TEXT NOT NULL,
        text         TEXT NOT NULL,
        is_emergency INTEGER NOT NULL DEFAULT 0,
        position     INTEGER NOT NULL DEFAULT 0,
        created_at   REAL NOT NULL,
        UNIQUE(lang, text)
    );
    CREATE TABLE personal_vocabulary (
        lang      TEXT NOT NULL,
        word      TEXT NOT NULL,
        count     INTEGER NOT NULL DEFAULT 0,
        last_used REAL NOT NULL,
        PRIMARY KEY (lang, word)
    );
    CREATE TABLE word_bigrams (
        lang      TEXT NOT NULL,
        prev      TEXT NOT NULL,
        word      TEXT NOT NULL,
        count     INTEGER NOT NULL DEFAULT 0,
        last_used REAL NOT NULL,
        PRIMARY KEY (lang, prev, word)
    );
    CREATE TABLE calibration_profiles (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       TEXT NOT NULL,
        created_at REAL NOT NULL,
        version    INTEGER NOT NULL,
        n_points   INTEGER NOT NULL,
        screen_w   INTEGER NOT NULL,
        screen_h   INTEGER NOT NULL,
        error_px   REAL NOT NULL,
        model_json TEXT NOT NULL,
        active     INTEGER NOT NULL DEFAULT 0
    );
    """,
    # v2: index used when pruning old bigrams
    """
    CREATE INDEX idx_bigrams_last_used ON word_bigrams(last_used);
    """,
]


class Database:
    """Thread-safe wrapper around a single SQLite connection."""

    def __init__(self, path: Path | str):
        self.path = Path(path) if path != ":memory:" else path
        if self.path != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.recovered_from_corruption = False
        self.conn = self._open()

    # ------------------------------------------------------------------ setup
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL" if self.path != ":memory:" else "PRAGMA journal_mode=MEMORY")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.DatabaseError:
            conn.close()  # Windows cannot move a file that is still open
            raise
        return conn

    def _open(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            ok = conn.execute("PRAGMA quick_check").fetchone()[0]
            if ok != "ok":
                raise sqlite3.DatabaseError(f"quick_check: {ok}")
            self._migrate(conn)
            return conn
        except sqlite3.DatabaseError as exc:
            if self.path == ":memory:" or not self.path.exists():
                raise  # not corruption: e.g. no permission to create the file
            log.error("Database unusable (%s); moving it aside and starting fresh", exc)
            if conn is not None:
                conn.close()
            backup = self.path.with_name(f"{self.path.name}.corrupt-{int(time.time())}")
            for suffix in ("", "-wal", "-shm"):
                src = Path(str(self.path) + suffix)
                if src.exists():
                    shutil.move(str(src), str(backup) + suffix)
            self.recovered_from_corruption = True
            conn = self._connect()
            self._migrate(conn)
            return conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        current = row[0] if row else 0
        if row is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (0)")
        for version, script in enumerate(MIGRATIONS[current:], start=current + 1):
            conn.execute("BEGIN")
            try:
                for statement in filter(str.strip, script.split(";")):
                    conn.execute(statement)
                conn.execute("UPDATE schema_version SET version = ?", (version,))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    @property
    def schema_version(self) -> int:
        return self.conn.execute("SELECT version FROM schema_version").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, tuple(params))

    # --------------------------------------------------------------- settings
    def load_settings(self) -> dict[str, Any]:
        values = dict(DEFAULTS)
        for key, raw in self._execute("SELECT key, value FROM settings"):
            spec = SPECS_BY_KEY.get(key)
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if spec is not None:
                values[key] = spec.clamp(value)
            else:
                values[key] = value  # internal flags (e.g. first_run)
        return values

    def save_setting(self, key: str, value: Any) -> None:
        self._execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )

    def reset_settings(self) -> None:
        self._execute("DELETE FROM settings WHERE key IN (%s)" % ",".join("?" * len(DEFAULTS)), DEFAULTS.keys())

    # ---------------------------------------------------------------- phrases
    def phrases(self, lang: str) -> list[dict[str, Any]]:
        rows = self._execute(
            "SELECT id, text, is_emergency FROM saved_phrases WHERE lang = ? "
            "ORDER BY is_emergency DESC, position, id",
            (lang,),
        ).fetchall()
        return [{"id": r[0], "text": r[1], "is_emergency": bool(r[2])} for r in rows]

    def add_phrase(self, lang: str, text: str, is_emergency: bool = False) -> None:
        text = text.strip()
        if not text:
            return
        position = self._execute(
            "SELECT COALESCE(MAX(position), 0) + 1 FROM saved_phrases WHERE lang = ?", (lang,)
        ).fetchone()[0]
        self._execute(
            "INSERT OR IGNORE INTO saved_phrases (lang, text, is_emergency, position, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (lang, text, int(is_emergency), position, time.time()),
        )

    def delete_phrase(self, phrase_id: int) -> None:
        self._execute("DELETE FROM saved_phrases WHERE id = ?", (phrase_id,))

    def seed_phrases(self, lang: str, phrases: list[tuple[str, bool]]) -> None:
        """Insert default phrases the first time a language is used."""
        count = self._execute("SELECT COUNT(*) FROM saved_phrases WHERE lang = ?", (lang,)).fetchone()[0]
        flag = f"_seeded_phrases_{lang}"
        seeded = self._execute("SELECT 1 FROM settings WHERE key = ?", (flag,)).fetchone()
        if count == 0 and not seeded:
            for text, emergency in phrases:
                self.add_phrase(lang, text, emergency)
        self.save_setting(flag, True)

    # ------------------------------------------------------- personal learning
    def record_word(self, lang: str, word: str, prev: str | None) -> None:
        now = time.time()
        with self._lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.execute(
                    "INSERT INTO personal_vocabulary (lang, word, count, last_used) VALUES (?, ?, 1, ?) "
                    "ON CONFLICT(lang, word) DO UPDATE SET count = MIN(count + 1, ?), last_used = excluded.last_used",
                    (lang, word, now, MAX_WORD_COUNT),
                )
                if prev:
                    self.conn.execute(
                        "INSERT INTO word_bigrams (lang, prev, word, count, last_used) VALUES (?, ?, ?, 1, ?) "
                        "ON CONFLICT(lang, prev, word) DO UPDATE SET count = MIN(count + 1, ?), "
                        "last_used = excluded.last_used",
                        (lang, prev, word, now, MAX_WORD_COUNT),
                    )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def prune_learning(self) -> None:
        """Keep personal tables bounded by dropping the least recently used rows."""
        self._execute(
            "DELETE FROM personal_vocabulary WHERE rowid IN (SELECT rowid FROM personal_vocabulary "
            "ORDER BY last_used DESC LIMIT -1 OFFSET ?)",
            (MAX_PERSONAL_WORDS,),
        )
        self._execute(
            "DELETE FROM word_bigrams WHERE rowid IN (SELECT rowid FROM word_bigrams "
            "ORDER BY last_used DESC LIMIT -1 OFFSET ?)",
            (MAX_BIGRAMS,),
        )

    def personal_vocabulary(self, lang: str) -> dict[str, tuple[int, float]]:
        rows = self._execute("SELECT word, count, last_used FROM personal_vocabulary WHERE lang = ?", (lang,))
        return {w: (c, t) for w, c, t in rows}

    def personal_bigrams(self, lang: str) -> dict[tuple[str, str], tuple[int, float]]:
        rows = self._execute("SELECT prev, word, count, last_used FROM word_bigrams WHERE lang = ?", (lang,))
        return {(p, w): (c, t) for p, w, c, t in rows}

    # ------------------------------------------------------------ calibration
    def save_calibration(self, name: str, version: int, n_points: int, screen: tuple[int, int],
                         error_px: float, model: dict) -> int:
        with self._lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.execute("UPDATE calibration_profiles SET active = 0")
                cur = self.conn.execute(
                    "INSERT INTO calibration_profiles (name, created_at, version, n_points, screen_w, screen_h, "
                    "error_px, model_json, active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
                    (name, time.time(), version, n_points, screen[0], screen[1], error_px, json.dumps(model)),
                )
                # Keep the 10 most recent profiles.
                self.conn.execute(
                    "DELETE FROM calibration_profiles WHERE id NOT IN "
                    "(SELECT id FROM calibration_profiles ORDER BY created_at DESC LIMIT 10)"
                )
                self.conn.execute("COMMIT")
                return cur.lastrowid
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def active_calibration(self) -> dict[str, Any] | None:
        row = self._execute(
            "SELECT id, name, created_at, version, n_points, screen_w, screen_h, error_px, model_json "
            "FROM calibration_profiles WHERE active = 1 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        try:
            model = json.loads(row[8])
        except json.JSONDecodeError:
            return None
        return {
            "id": row[0], "name": row[1], "created_at": row[2], "version": row[3], "n_points": row[4],
            "screen": (row[5], row[6]), "error_px": row[7], "model": model,
        }

    # -------------------------------------------------------- export / import
    def export_data(self) -> dict[str, Any]:
        settings = {k: v for k, v in self.load_settings().items() if k in DEFAULTS}
        phrases = [
            {"lang": lang, "text": text, "is_emergency": bool(em)}
            for lang, text, em in self._execute(
                "SELECT lang, text, is_emergency FROM saved_phrases ORDER BY lang, position, id"
            )
        ]
        return {"format": EXPORT_FORMAT, "version": EXPORT_VERSION, "settings": settings, "phrases": phrases}

    def export_to_file(self, path: Path) -> Path:
        path.write_text(json.dumps(self.export_data(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def import_data(self, data: dict[str, Any]) -> None:
        if data.get("format") != EXPORT_FORMAT:
            raise ValueError("Not a LookSay export file")
        for key, value in dict(data.get("settings", {})).items():
            spec = SPECS_BY_KEY.get(key)
            if spec is not None:
                self.save_setting(key, spec.clamp(value))
        for phrase in data.get("phrases", []):
            lang = phrase.get("lang")
            if lang in ("en", "es", "ca") and isinstance(phrase.get("text"), str):
                self.add_phrase(lang, phrase["text"], bool(phrase.get("is_emergency")))

    def import_from_file(self, path: Path) -> None:
        self.import_data(json.loads(path.read_text(encoding="utf-8")))

    def reset_all(self) -> None:
        """One-click local data reset: wipes every user table."""
        with self._lock:
            self.conn.execute("BEGIN")
            for table in ("settings", "saved_phrases", "personal_vocabulary", "word_bigrams", "calibration_profiles"):
                self.conn.execute(f"DELETE FROM {table}")
            self.conn.execute("COMMIT")
