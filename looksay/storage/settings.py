"""In-memory settings backed by the database, with change listeners."""

from __future__ import annotations

from typing import Any, Callable

from looksay.config import DEFAULTS, SPECS_BY_KEY
from looksay.storage.database import Database

Listener = Callable[[str, Any], None]


class Settings:
    def __init__(self, db: Database):
        self._db = db
        self._values = db.load_settings()
        self._listeners: list[Listener] = []

    def __getitem__(self, key: str) -> Any:
        return self._values.get(key, DEFAULTS.get(key))

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, DEFAULTS.get(key, default))

    def set(self, key: str, value: Any) -> None:
        spec = SPECS_BY_KEY.get(key)
        if spec is not None:
            value = spec.clamp(value)
        if self._values.get(key) == value:
            return
        self._values[key] = value
        self._db.save_setting(key, value)
        for listener in list(self._listeners):
            listener(key, value)

    def step(self, key: str, direction: int) -> Any:
        self.set(key, SPECS_BY_KEY[key].step_value(self[key], direction))
        return self[key]

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def reload(self) -> None:
        """Re-read from the database (after import/reset) and notify every key."""
        self._values = self._db.load_settings()
        for key, value in self._values.items():
            for listener in list(self._listeners):
                listener(key, value)
