"""
Tests for scheduling and the permission model that network access and package
installation depend on.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen35.config import DEFAULT_PROFILE, PROFILES, get_permissions
from qwen35.schedule import (ScheduleStore, format_interval, parse_interval,
                             tick, watch)


# ==========================================================================
# Scheduling
# ==========================================================================
class TestIntervals(unittest.TestCase):
    def test_minutes_and_hours(self):
        self.assertEqual(parse_interval("30m"), 1800)
        self.assertEqual(parse_interval("2h"), 7200)
        self.assertEqual(parse_interval("90"), 5400)        # bare = minutes
        self.assertEqual(parse_interval("45min"), 2700)
        self.assertEqual(parse_interval("1.5h"), 5400)
        self.assertEqual(parse_interval(600), 600)

    def test_sub_minute_intervals_are_refused(self):
        """A sub-minute loop would just spin the model."""
        for bad in ("10s", "0", "-5"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_interval(bad)

    def test_nonsense_is_refused(self):
        for bad in ("abc", "", ">", "5 lightyears"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_interval(bad)

    def test_round_trip_formatting(self):
        self.assertEqual(format_interval(1800), "30m")
        self.assertEqual(format_interval(7200), "2h")
        self.assertEqual(format_interval(5400), "90m")


class TestScheduleStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "schedules.json"
        self.store = ScheduleStore(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_add_list_remove(self):
        schedule = self.store.add("check the forum", "30m")
        self.assertEqual(len(self.store.schedules), 1)
        self.assertTrue(self.path.is_file())
        self.assertTrue(self.store.remove(schedule.id))
        self.assertEqual(self.store.schedules, [])

    def test_a_new_schedule_runs_on_the_next_tick(self):
        """Otherwise adding a 2-hour job looks identical to a broken one."""
        self.store.add("task", "2h")
        self.assertEqual(len(self.store.due()), 1)

    def test_it_is_not_due_again_straight_after_running(self):
        self.store.add("task", "30m")
        tick(self.store, lambda _s: (True, "finished"))
        self.assertEqual(self.store.due(), [])
        self.assertAlmostEqual(self.store.next_due_in() or 0, 1800, delta=5)

    def test_a_long_sleep_causes_one_catch_up_run_not_many(self):
        """Waking a laptop must not launch a burst of model calls."""
        schedule = self.store.add("task", "30m")
        schedule.last_run = time.time() - 6 * 3600      # six hours later
        runs = []
        tick(self.store, lambda s: (runs.append(s.id), (True, "ok"))[1])
        self.assertEqual(len(runs), 1)
        self.assertEqual(schedule.missed, 11)           # 12 intervals - 1

    def test_a_run_in_flight_is_never_started_twice(self):
        schedule = self.store.add("task", "30m")
        schedule.running_since = time.monotonic()
        self.assertEqual(self.store.due(), [])

    def test_failures_are_counted_not_raised(self):
        self.store.add("task", "30m")

        def boom(_schedule):
            raise RuntimeError("model exploded")

        tick(self.store, boom)
        self.assertEqual(self.store.schedules[0].failures, 1)
        self.assertIn("RuntimeError", self.store.schedules[0].last_stop_reason)

    def test_state_survives_a_restart(self):
        schedule = self.store.add("task", "45m")
        tick(self.store, lambda _s: (True, "finished"))
        reloaded = ScheduleStore(self.path)
        self.assertEqual(len(reloaded.schedules), 1)
        self.assertEqual(reloaded.schedules[0].id, schedule.id)
        self.assertEqual(reloaded.schedules[0].runs, 1)
        # and nothing is left looking like it is still running
        self.assertEqual(reloaded.schedules[0].running_since, 0.0)

    def test_a_corrupt_file_does_not_break_startup(self):
        self.path.write_text("{not json", encoding="utf-8")
        store = ScheduleStore(self.path)
        self.assertEqual(store.schedules, [])
        self.assertTrue(self.path.with_suffix(".json.bad").is_file())

    def test_disabled_schedules_are_skipped(self):
        schedule = self.store.add("task", "30m")
        schedule.enabled = False
        self.assertEqual(self.store.due(), [])
        self.assertIsNone(self.store.next_due_in())

    def test_watch_stops_after_the_requested_runs(self):
        self.store.add("task", "30m")
        slept: list[float] = []
        total = watch(self.store, lambda _s: (True, "ok"), stop_after=3,
                      log=lambda _m: None, sleep=slept.append,
                      now=lambda: time.time() + 10_000)
        self.assertEqual(total, 3)


class TestPermissionProfiles(unittest.TestCase):
    def test_safe_was_merged_into_standard(self):
        self.assertEqual(DEFAULT_PROFILE, "standard")
        self.assertEqual(get_permissions("safe").to_dict(),
                         get_permissions("standard").to_dict())

    def test_standard_is_useful(self):
        """'safe' was too small to work in; the merged profile must not be."""
        perms = get_permissions("standard")
        self.assertTrue(perms.allow_network_commands)
        self.assertTrue(perms.allow_package_install)
        self.assertGreaterEqual(perms.max_read_bytes, 1_000_000)
        self.assertGreaterEqual(perms.max_command_seconds, 60)
        self.assertGreaterEqual(perms.max_commands_per_run, 100)
        # ...and still not a shell
        self.assertFalse(perms.allow_shell_metachars)

    def test_readonly_withholds_everything_active(self):
        perms = get_permissions("readonly")
        self.assertFalse(perms.write)
        self.assertFalse(perms.run_command)
        self.assertFalse(perms.allow_network_commands)

    def test_an_unknown_profile_is_rejected(self):
        with self.assertRaises(KeyError):
            get_permissions("nonsense")


if __name__ == "__main__":
    unittest.main(verbosity=2)
