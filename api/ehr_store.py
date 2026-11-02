"""
ReadmitAI - EHR JSON File Store
Keeps synthetic patients and encounters in a single JSON file (no database server).
Writes are serialized with a lock and saved atomically so a crash never leaves a half-written file.
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_STORE_PATH = os.environ.get(
    "READMITAI_EHR_STORE", os.path.join(BASE_DIR, "storage", "ehr_store.json")
)


def _empty_store() -> Dict[str, Any]:
    return {"patients": {}, "encounters": {}, "tasks": {}, "counters": {"patient": 0, "encounter": 0, "task": 0}}


class JsonStore:
    def __init__(self, path: str = DEFAULT_STORE_PATH):
        self.path = path
        self._lock = threading.RLock()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return _empty_store()
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("tasks", {})  # stores created before follow-up tasks existed
        return data

    def _save(self, data: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # On Windows the replace is refused while another process (editor, antivirus, indexer)
        # briefly holds the file open, so retry for up to ~2 seconds before giving up.
        for attempt in range(20):
            try:
                os.replace(tmp_path, self.path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1)

    def read(self) -> Dict[str, Any]:
        """Returns a snapshot of the whole store (read-only use)."""
        with self._lock:
            return self._load()

    @contextmanager
    def transaction(self) -> Iterator[Dict[str, Any]]:
        """Loads the store, yields it for modification, and saves it if no error was raised."""
        with self._lock:
            data = self._load()
            yield data
            self._save(data)

    def next_id(self, data: Dict[str, Any], kind: str) -> str:
        data["counters"][kind] = data["counters"].get(kind, 0) + 1
        prefix = {"patient": "P", "encounter": "E", "task": "T"}[kind]
        return f"{prefix}{data['counters'][kind]:05d}"
