"""Cache of model answers on disk, keyed by everything that decides the answer.

The key is a hash of the model, the prompt version, the messages and the settings. A run that
finds its answers in the cache makes no calls, so every stage can be repeated offline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def cache_key(model: str, prompt_version: str, messages: list[dict], settings: dict) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt_version": prompt_version,
            "messages": messages,
            "settings": settings,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, record: dict) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
