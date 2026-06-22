from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"missing {name}; add it to your .env file")
    return val