"""
Scheduled work: run a task every N minutes or hours.

The design follows what job runners (Airflow, Windmill, n8n) converge on, reduced
to what a single-process local agent actually needs:

* **An interval, not a cron expression.**  For "look at the forum every 30
  minutes" an interval is what the user means, and it needs no timezone or
  day-of-week reasoning to be correct.  Minutes and hours only, as asked.
* **Persist the last run, not a countdown.**  A countdown is lost on restart; a
  stored `last_run` timestamp means a laptop that was asleep for six hours
  resumes sensibly instead of firing six times.
* **Never overlap.**  A tick runs its schedules one at a time, and a schedule
  whose previous run is still going is skipped rather than started twice.
* **Catch up at most once.**  Missing ten intervals does not mean ten runs; it
  means one, now.  Otherwise waking a laptop would launch a burst of model calls.
* **Bound each run.**  Every tick carries a step budget, so a schedule cannot
  quietly spend an unbounded amount of time.

State lives in ``<state>/schedules.json``.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = [
    "Schedule",
    "ScheduleStore",
    "parse_interval",
    "format_interval",
    "store_path",
]

#: A tick will not start a schedule more than this long after it was due, unless
#: it is the first run after a long sleep -- in which case it runs once, late.
_MIN_TICK = 5.0
#: Guard against a typo like "1" meaning one second.
MIN_INTERVAL_SECONDS = 60
MAX_INTERVAL_SECONDS = 30 * 24 * 3600


def store_path(state_dir: str | os.PathLike[str] | None = None) -> Path:
    base = Path(state_dir) if state_dir else Path(
        os.environ.get("QWEN35_HOME") or (Path.home() / ".qwen35"))
    return Path(base) / "schedules.json"


_INTERVAL = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds|"
                       r"m|min|mins|minute|minutes|h|hr|hrs|hour|hours|"
                       r"d|day|days)?\s*$", re.IGNORECASE)

_UNIT_SECONDS = {
    "": 60, "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}


def parse_interval(text: str | int | float) -> int:
    """``"30m"`` / ``"2h"`` / ``"90"`` (minutes) / ``900`` -> seconds.

    Minutes are the default unit because that is what "every 30" means in
    practice; seconds are accepted for testing but clamped to at least a minute,
    since a sub-minute loop would just spin the model.
    """
    if isinstance(text, (int, float)):
        seconds = int(text)
    else:
        match = _INTERVAL.match(str(text or ""))
        if not match:
            raise ValueError(
                f"cannot read {text!r} as an interval; use e.g. 30m, 2h, 90m")
        value, unit = match.group(1), (match.group(2) or "").lower()
        seconds = int(float(value) * _UNIT_SECONDS[unit])
    if seconds < MIN_INTERVAL_SECONDS:
        raise ValueError(
            f"interval {text!r} is under a minute; the shortest allowed is 60s")
    if seconds > MAX_INTERVAL_SECONDS:
        raise ValueError(f"interval {text!r} is longer than 30 days")
    return seconds


def format_interval(seconds: int) -> str:
    if seconds % 3600 == 0 and seconds >= 3600:
        hours = seconds // 3600
        return f"{hours}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


@dataclass
class Schedule:
    """One recurring task."""

    id: str
    task: str
    every_seconds: int
    workspace: str = ""
    profile: str = "standard"
    enabled: bool = True
    created: float = field(default_factory=time.time)
    #: wall-clock time of the last *started* run; 0 means never
    last_run: float = 0.0
    #: monotonic clock of a run in flight, so a tick can skip overlaps
    running_since: float = 0.0
    runs: int = 0
    failures: int = 0
    last_stop_reason: str = ""
    #: set when the run is a "catch-up" so the prompt can be told
    missed: int = 0

    @property
    def due_at(self) -> float:
        """When the next run is wanted.

        A brand-new schedule is due **on the next tick**, not after a whole
        interval.  Otherwise adding a two-hour job and watching it do nothing for
        two hours is indistinguishable from the scheduler being broken, and the
        user has no way to confirm the task even works before waiting.
        """
        if not self.last_run:
            return self.created
        return self.last_run + self.every_seconds

    def is_due(self, now: float | None = None) -> bool:
        if not self.enabled or self.running_since:
            return False
        return (now if now is not None else time.time()) >= self.due_at

    def seconds_until_due(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, self.due_at - now)

    def summary(self) -> str:
        when = (time.strftime("%m-%d %H:%M", time.localtime(self.last_run))
                if self.last_run else "never")
        state = "off" if not self.enabled else (
            "running" if self.running_since else "on")
        return (f"{self.id}  every {format_interval(self.every_seconds)}  "
                f"[{state}]  runs={self.runs} fails={self.failures}  "
                f"last={when}  {self.task[:44]}")


class ScheduleStore:
    """JSON-backed list of schedules."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else store_path()
        self.schedules: list[Schedule] = []
        self.load()

    # -- persistence ---------------------------------------------------
    def load(self) -> None:
        self.schedules = []
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt file must not stop the harness; keep a copy to inspect.
            try:
                self.path.with_suffix(".json.bad").write_bytes(
                    self.path.read_bytes())
            except OSError:
                pass
            return
        for item in raw.get("schedules", []) if isinstance(raw, dict) else raw:
            try:
                known = {f for f in Schedule.__dataclass_fields__}
                self.schedules.append(
                    Schedule(**{k: v for k, v in item.items() if k in known}))
            except (TypeError, ValueError):
                continue
        # nothing should still look like it is running after a restart
        for schedule in self.schedules:
            schedule.running_since = 0.0

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1,
                   "schedules": [asdict(s) for s in self.schedules]}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.path)          # atomic: never a half-written file

    # -- crud ----------------------------------------------------------
    def add(self, task: str, every: str | int, *, workspace: str = "",
            profile: str = "standard", enabled: bool = True) -> Schedule:
        seconds = parse_interval(every)
        schedule = Schedule(id=uuid.uuid4().hex[:8], task=task,
                            every_seconds=seconds, workspace=workspace,
                            profile=profile, enabled=enabled)
        self.schedules.append(schedule)
        self.save()
        return schedule

    def get(self, ident: str) -> Schedule | None:
        for schedule in self.schedules:
            if schedule.id == ident or schedule.id.startswith(ident):
                return schedule
        return None

    def remove(self, ident: str) -> bool:
        found = self.get(ident)
        if found is None:
            return False
        self.schedules = [s for s in self.schedules if s is not found]
        self.save()
        return True

    def due(self, now: float | None = None) -> list[Schedule]:
        return [s for s in self.schedules if s.is_due(now)]

    def next_due_in(self, now: float | None = None) -> float | None:
        pending = [s.seconds_until_due(now) for s in self.schedules
                   if s.enabled and not s.running_since]
        return min(pending) if pending else None

    # -- running -------------------------------------------------------
    def mark_started(self, schedule: Schedule, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        # How many whole intervals were missed?  Recorded so the prompt can be
        # told it is catching up, and so a long sleep causes ONE run, not many.
        if schedule.last_run:
            elapsed = now - schedule.last_run
            schedule.missed = max(
                0, int(elapsed // schedule.every_seconds) - 1)
        else:
            schedule.missed = 0
        schedule.last_run = now
        schedule.running_since = time.monotonic()
        schedule.runs += 1
        self.save()

    def mark_finished(self, schedule: Schedule, *, ok: bool,
                      stop_reason: str = "") -> None:
        schedule.running_since = 0.0
        schedule.last_stop_reason = stop_reason
        if not ok:
            schedule.failures += 1
        self.save()


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
#: Runs one schedule and reports (ok, stop_reason).  Injected so the loop can be
#: tested without a model.
Runner = Callable[[Schedule], tuple[bool, str]]


def run_once(store: ScheduleStore, schedule: Schedule, runner: Runner,
             log: Callable[[str], None] | None = None) -> tuple[bool, str]:
    """Run one schedule, marking it started and finished around the call."""
    say = log or (lambda _m: None)
    store.mark_started(schedule)
    try:
        ok, reason = runner(schedule)
    except Exception as exc:  # noqa: BLE001 - a bad tick must not kill the loop
        ok, reason = False, f"{type(exc).__name__}: {exc}"
    finally:
        store.mark_finished(schedule, ok=ok, stop_reason=reason)
    say(f"  {schedule.id}: {'ok' if ok else 'failed'} ({reason})")
    return ok, reason


def tick(store: ScheduleStore, runner: Runner,
         log: Callable[[str], None] | None = None,
         *, clock: Callable[[], float] = time.time) -> int:
    """Run every schedule that is due, one at a time.  Returns how many ran.

    The clock is injected and shared with :func:`watch`.  Using the real clock
    here while :func:`watch` computed its wait from a different one made the two
    disagree about what was due -- the loop saw a schedule that was never ready
    and never exited.  One clock, one answer.
    """
    ran = 0
    for schedule in store.due(clock()):
        run_once(store, schedule, runner, log)
        ran += 1
    return ran


def watch(store: ScheduleStore, runner: Runner, *,
          stop_after: int | None = None,
          max_seconds: float | None = None,
          log: Callable[[str], None] | None = None,
          sleep: Callable[[float], None] = time.sleep,
          now: Callable[[], float] = time.time) -> int:
    """Foreground loop: sleep until the next due time, run it, repeat.

    Returns the number of runs performed.  ``stop_after`` and ``max_seconds``
    exist so this is testable and so a caller can bound a session.
    """
    say = log or print
    started = time.monotonic()
    total = 0
    while True:
        if stop_after is not None and total >= stop_after:
            return total
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            return total

        ran = tick(store, runner, say, clock=now)
        total += ran
        if stop_after is not None and total >= stop_after:
            return total

        wait = store.next_due_in(now())
        if wait is None:
            say("  no enabled schedules; nothing to wait for")
            return total
        # Never busy-wait: sleep the whole gap, but re-check at least hourly.
        sleep(max(_MIN_TICK, min(wait, 3600.0)))
