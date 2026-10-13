"""
Session recording.

Every run is appended to a JSONL transcript under
``<workspace>/.qwen35/sessions/``.  Two reasons:

* a failing trajectory can be replayed against :class:`~qwen35.backends.mock.MockBackend`
  without the model, which is what makes debugging a 0.8 B agent tractable;
* the operator can audit exactly which commands the model ran and which files it
  wrote -- useful when the permission posture is the thing being evaluated.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["SessionRecorder", "list_sessions", "load_session"]

_SLUG = re.compile(r"[^a-zA-Z0-9]+")


def _slug(text: str, limit: int = 40) -> str:
    s = _SLUG.sub("-", text.strip().lower()).strip("-")
    return (s[:limit] or "session").strip("-")


@dataclass
class SessionRecorder:
    path: Path
    _handle: Any = None
    _closed: bool = False

    @classmethod
    def open(cls, workspace: Path, task: str = "",
             directory: str = "sessions") -> "SessionRecorder":
        out_dir = workspace / ".qwen35" / directory
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"{stamp}-{_slug(task)}.jsonl"
        rec = cls(path=path)
        rec._handle = open(path, "a", encoding="utf-8")
        rec.record("session", {
            "started": datetime.now(timezone.utc).isoformat(),
            "workspace": str(workspace),
        })
        return rec

    def record(self, event: str, data: dict[str, Any]) -> None:
        if self._closed or self._handle is None:
            return
        payload = {"t": round(time.time(), 3), "event": event, **data}
        try:
            self._handle.write(json.dumps(payload, ensure_ascii=False,
                                          default=str) + "\n")
            self._handle.flush()
        except (OSError, TypeError):
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._handle:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def list_sessions(workspace: Path, limit: int = 20,
                  directory: str = "sessions") -> list[Path]:
    d = workspace / ".qwen35" / directory
    if not d.is_dir():
        return []
    files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    return files[:limit]


def load_session(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.is_file():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
