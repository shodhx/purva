from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Record:
    id: str
    raw_text: str
    cleaned_text: str
    source_url: str
    source_name: str
    scrape_timestamp: str = field(default_factory=_utc_now)
    lid_label: Optional[str] = None
    lid_confidence: Optional[float] = None
    category: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = set()
        self.written = 0
        self.skipped_dupes = 0
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        self._seen.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue

    def __enter__(self):
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self._fh.close()

    def add(self, rec: Record) -> bool:
        if rec.id in self._seen:
            self.skipped_dupes += 1
            return False
        self._seen.add(rec.id)
        self._fh.write(rec.to_json() + "\n")
        self.written += 1
        return True