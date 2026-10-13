"""
GUI tests.

These open a real Tk window (withdrawn, so nothing flashes on screen) and are
skipped automatically when no display is available.  What they cover is the part
that actually breaks in GUI code: widget construction, the worker->UI event
queue, the workspace picker's refusal path, and the image preview.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import tkinter as tk

    _TK_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - platform dependent
    tk = None  # type: ignore[assignment]
    _TK_ERROR = exc


def _display_available() -> bool:
    if tk is None:
        return False
    try:
        probe = tk.Tk()
        probe.withdraw()
        probe.destroy()
        return True
    except Exception:
        return False


TK_OK = _display_available()
SKIP_REASON = f"no usable display for Tkinter ({_TK_ERROR})"


@unittest.skipUnless(TK_OK, SKIP_REASON)
class SharedTkRoot:
    """Give every test in a class ONE Tk interpreter.

    Tk does not survive being created and destroyed dozens of times inside a
    single process.  Past roughly thirty interpreters this suite aborted with
    `Tcl_AsyncDelete: async handler deleted by the wrong thread` -- an access
    violation, not a test failure, so the run died with no summary.  Every class
    passed on its own; it was the accumulation that killed it.  A shared root per
    class cuts the count from ~35 to 6, and each test still gets a clean window
    because the root's children are destroyed first.
    """

    root: tk.Tk

    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def fresh_root(self) -> tk.Tk:
        for child in self.root.winfo_children():
            child.destroy()
        return self.root


class TestGuiConstruction(SharedTkRoot, unittest.TestCase):
    def setUp(self):
        from qwen35.gui import HarnessGUI

        self._tmp = tempfile.TemporaryDirectory()
        self.root = self.fresh_root()
        self.root.withdraw()
        self.gui = HarnessGUI(
            self.root,
            workspace=Path(self._tmp.name),
            model_dir=Path(self._tmp.name),
            auto_start_server=False,
            manage_server=False,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_builds_all_key_widgets(self):
        self.assertTrue(self.gui.transcript.winfo_exists())
        self.assertTrue(self.gui.input.winfo_exists())
        self.assertTrue(self.gui.image_label.winfo_exists())
        self.assertTrue(self.gui.send_button.winfo_exists())
        self.assertTrue(self.gui.stop_button.winfo_exists())

    def test_transcript_is_read_only(self):
        self.assertEqual(str(self.gui.transcript.cget("state")), "disabled")

    def test_initial_workspace_is_the_canonical_path(self):
        self.assertEqual(self.gui.workspace,
                         Path(os.path.realpath(self._tmp.name)))

    def test_serving_is_not_attempted_when_disabled(self):
        # auto_start_server=False must leave the queue empty
        self.assertEqual(self.gui._drain(), 0)

    def test_profile_label_maps_back_to_a_profile_name(self):
        self.gui.profile_var.set("readonly - 只读，不能执行命令")
        self.assertEqual(self.gui._selected_profile(), "readonly")
        self.gui.profile_var.set("safe")
        self.assertEqual(self.gui._selected_profile(), "safe")


@unittest.skipUnless(TK_OK, SKIP_REASON)
class TestGuiWorkspaceRefusal(SharedTkRoot, unittest.TestCase):
    def setUp(self):
        self.root = self.fresh_root()
        self.root.withdraw()

    def tearDown(self):
        pass

    def _build(self, workspace: str):
        from qwen35.gui import HarnessGUI

        top = tk.Toplevel(self.root)
        top.withdraw()
        gui = HarnessGUI(top, workspace=workspace, auto_start_server=False,
                         manage_server=False)
        return top, gui

    def test_drive_root_is_not_accepted_as_a_workspace(self):
        """The GUI must enforce the same confinement as the CLI."""
        from qwen35.workspace import validate_workspace

        top, gui = self._build("C:\\")
        try:
            self.assertNotEqual(str(gui.workspace), "C:\\")
            # whatever it fell back to must itself be a legal workspace
            self.assertTrue(validate_workspace(str(gui.workspace)))
        finally:
            top.destroy()

    def test_home_directory_is_not_accepted(self):
        top, gui = self._build(str(Path.home()))
        try:
            self.assertNotEqual(gui.workspace, Path(os.path.realpath(Path.home())))
        finally:
            top.destroy()


@unittest.skipUnless(TK_OK, SKIP_REASON)
class TestGuiEventRendering(SharedTkRoot, unittest.TestCase):
    """The worker thread posts events; the UI thread renders them."""

    def setUp(self):
        from qwen35.gui import HarnessGUI

        self._tmp = tempfile.TemporaryDirectory()
        self.root = self.fresh_root()
        self.root.withdraw()
        self.gui = HarnessGUI(
            self.root,
            workspace=Path(self._tmp.name),
            model_dir=Path(self._tmp.name),
            auto_start_server=False,
            manage_server=False,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def text(self) -> str:
        return self.gui.transcript.get("1.0", "end")

    def test_start_event_renders_backend_and_tools(self):
        self.gui.events.put(("event", "start", {
            "backend": "mock", "tools": ["read_file", "edit_file"],
            "context_window": 4096,
        }))
        self.assertEqual(self.gui._drain(), 1)
        body = self.text()
        self.assertIn("mock", body)
        self.assertIn("read_file", body)

    def test_tool_call_and_result_render(self):
        self.gui.events.put(("event", "thinking_start",
                             {"step": 1, "grammar": True, "thinking": True}))
        self.gui.events.put(("event", "tool_call",
                             {"step": 1, "call": "read_file(path=a.py)"}))
        self.gui.events.put(("event", "tool_ok", {"step": 1, "chars": 42}))
        self.gui._drain()
        body = self.text()
        self.assertIn("步骤 1", body)
        self.assertIn("read_file(path=a.py)", body)
        self.assertIn("42", body)

    def test_failing_command_is_shown_as_information_not_an_error(self):
        """A red test is information; a refusal is an error."""
        self.gui.events.put(("event", "tool_error", {
            "text": "$ python t.py\n[exit code 1, 0.2s]", "exit": 1,
            "refused": False,
        }))
        self.gui._drain()
        self.assertIn("exit 1", self.text())

    def test_refused_command_is_shown_as_an_error(self):
        self.gui.events.put(("event", "tool_error", {
            "text": "error: command refused -- shell metacharacter '&'",
            "exit": None, "refused": True,
        }))
        self.gui._drain()
        self.assertIn("refused", self.text())

    def test_answer_and_summary_render_on_result(self):
        class FakeResult:
            answer = "修好了。"
            stop_reason = "finished"
            steps = [1, 2, 3]
            tokens_in = 1234
            tokens_out = 56
            elapsed = 12.5

        self.gui._on_result(FakeResult(), "")
        body = self.text()
        self.assertIn("修好了。", body)
        self.assertIn("finished", body)
        self.assertIn("3 步", body)

    def test_fatal_error_does_not_raise(self):
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            self.gui._on_fatal(exc, "traceback text")
        self.assertIn("boom", self.text())
        self.assertFalse(self.gui._busy)

    def test_a_rendering_bug_does_not_kill_the_pump(self):
        """The drain loop must survive a bad event payload."""
        self.gui.events.put(("event", "tool_call", None))   # not a dict
        self.gui.events.put(("event", "start",
                             {"backend": "b", "tools": [], "context_window": 1}))
        handled = self.gui._drain()
        self.assertEqual(handled, 2)
        self.assertIn("b", self.text())

    def test_busy_state_toggles_the_buttons(self):
        self.gui._set_busy(True)
        self.assertEqual(str(self.gui.send_button.cget("state")), "disabled")
        self.assertEqual(str(self.gui.stop_button.cget("state")), "normal")
        self.gui._set_busy(False)
        self.assertEqual(str(self.gui.send_button.cget("state")), "normal")
        self.assertEqual(str(self.gui.stop_button.cget("state")), "disabled")


@unittest.skipUnless(TK_OK, SKIP_REASON)
class TestGuiImagePreview(SharedTkRoot, unittest.TestCase):
    def setUp(self):
        from qwen35.gui import HarnessGUI

        self._tmp = tempfile.TemporaryDirectory()
        self.root = self.fresh_root()
        self.root.withdraw()
        self.gui = HarnessGUI(
            self.root, workspace=Path(self._tmp.name), model_dir=Path(self._tmp.name),
            auto_start_server=False, manage_server=False,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_view_image_call_loads_a_preview(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        target = Path(self._tmp.name) / "pic.png"
        Image.new("RGB", (320, 240), "red").save(target)

        self.gui.events.put(("event", "tool_call", {
            "step": 1, "call": "view_image(path=pic.png, question=what?)",
        }))
        self.gui._drain()
        # a PhotoImage reference must be retained or Tk garbage-collects it
        self.assertIsNotNone(self.gui._image_ref)
        self.assertEqual(str(self.gui.image_label.cget("text")), "")

    def test_missing_image_does_not_raise(self):
        self.gui.events.put(("event", "tool_call", {
            "step": 1, "call": "view_image(path=nope.png)",
        }))
        self.gui._drain()
        self.assertIsNone(self.gui._image_ref)

    def test_clear_resets_the_preview(self):
        self.gui._set_image(None)
        self.assertIn("尚未", str(self.gui.image_label.cget("text")))


@unittest.skipUnless(TK_OK, SKIP_REASON)
class TestGuiEventContract(unittest.TestCase):
    """Every event the agent actually emits must be renderable."""

    def test_all_agent_events_render_without_error(self):
        from qwen35.gui import HarnessGUI

        with tempfile.TemporaryDirectory() as tmp:
            root = tk.Tk()
            root.withdraw()
            try:
                gui = HarnessGUI(root, workspace=Path(tmp), model_dir=Path(tmp),
                                 auto_start_server=False, manage_server=False)
                payloads = {
                    "start": {"backend": "b", "tools": ["t"], "context_window": 8},
                    "thinking_start": {"step": 1, "grammar": True, "thinking": False},
                    "raw": {"step": 1, "text": "hello", "seconds": 0.1},
                    "tool_call": {"step": 1, "call": "list_dir(path=.)"},
                    "tool_ok": {"step": 1, "chars": 3},
                    "tool_error": {"text": "x", "exit": 1, "refused": False},
                    "format_error": {"problem": "bad", "retry": 1},
                    "degenerate": {"step": 1},
                    "forced_finish": {"repeats": 5},
                    "loop_stop": {"call": "c", "repeats": 5},
                    "context_full": {"tokens": 10, "budget": 10},
                    "error": {"message": "m"},
                    "answer": {"text": "a"},
                    "done": {"stop_reason": "finished", "steps": 1,
                             "tokens_in": 1, "tokens_out": 1, "elapsed": 0.1},
                }
                for name, data in payloads.items():
                    gui.events.put(("event", name, data))
                handled = gui._drain()
                self.assertEqual(handled, len(payloads))
            finally:
                root.destroy()


@unittest.skipUnless(TK_OK, SKIP_REASON)
class TestGuiThreadSafety(SharedTkRoot, unittest.TestCase):
    """Tk variables must never be read from the worker thread.

    Regression test.  The first version of the GUI read ``self.context_var``
    inside the agent thread, which raises

        RuntimeError: main thread is not in main loop

    Tk variables are Tcl objects owned by the thread that made them.  The window
    looked like it started the model server and then failed for no visible
    reason -- an end-to-end run found it; synthetic-event tests could not.
    """

    class Exploding:
        """Stands in for a Tk variable: any access is a bug."""

        def get(self, *args, **kwargs):
            raise AssertionError(
                "a Tk variable was read off the main thread"
            )

    def setUp(self):
        from qwen35.gui import HarnessGUI

        self._tmp = tempfile.TemporaryDirectory()
        self.root = self.fresh_root()
        self.root.withdraw()
        self.gui = HarnessGUI(
            self.root, workspace=Path(self._tmp.name), model_dir=Path(self._tmp.name),
            auto_start_server=False, manage_server=False,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_snapshot_returns_plain_values(self):
        options = self.gui._snapshot_options()
        self.assertIsInstance(options["context"], int)
        self.assertIsInstance(options["steps"], int)
        self.assertIsInstance(options["thinking"], str)
        self.assertIsInstance(options["profile"], str)
        self.assertIn(options["profile"], ("safe", "readonly", "standard", "trusted"))

    def test_make_config_touches_no_tk_variable(self):
        """_make_config runs on the worker thread, so it must be Tk-free."""
        options = self.gui._snapshot_options()
        self.gui.context_var = self.Exploding()
        self.gui.steps_var = self.Exploding()
        self.gui.thinking_var = self.Exploding()
        self.gui.profile_var = self.Exploding()
        cfg = self.gui._make_config(options)      # must not raise
        self.assertEqual(cfg.runtime.context_window, options["context"])
        self.assertEqual(cfg.runtime.max_steps, options["steps"])
        self.assertEqual(cfg.runtime.thinking, options["thinking"])
        self.assertEqual(cfg.permission_profile, options["profile"])

    def test_snapshot_makes_the_worker_independent_of_later_ui_edits(self):
        """Changing the dropdown after sending must not change the run."""
        options = self.gui._snapshot_options()
        self.gui.context_var.set("262144")
        cfg = self.gui._make_config(options)
        self.assertEqual(cfg.runtime.context_window, options["context"])
        self.assertNotEqual(cfg.runtime.context_window, 262144)


@unittest.skipUnless(TK_OK, SKIP_REASON)
class TestGuiAttachments(SharedTkRoot, unittest.TestCase):
    """Drag-and-drop, the file picker, and the conversion hand-off."""

    def setUp(self):
        from qwen35.gui import HarnessGUI

        self._tmp = tempfile.TemporaryDirectory()
        self.root = self.fresh_root()
        self.root.withdraw()
        self.gui = HarnessGUI(
            self.root, workspace=Path(self._tmp.name), model_dir=Path(self._tmp.name),
            auto_start_server=False, manage_server=False,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _drain_until(self, predicate, timeout: float = 15.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.gui._drain()
            self.root.update()
            if predicate():
                return True
            time.sleep(0.03)
        return False

    def _make_file(self, name: str, text: str) -> Path:
        p = Path(self._tmp.name).parent / f"qwen-attach-{name}"
        p.write_text(text, encoding="utf-8")
        self.addCleanup(lambda: p.unlink(missing_ok=True))
        return p

    # -- the panel ----------------------------------------------------
    def test_panel_exists_and_starts_empty(self):
        self.assertTrue(self.gui.attach_list.winfo_exists())
        self.assertEqual(self.gui.attach_list.size(), 1)
        self.assertIn("还没有附件", self.gui.attach_list.get(0))

    def test_register_dnd_on_a_plain_tk_root_degrades_gracefully(self):
        """A plain Tk root has no tkdnd.  That must disable drops, not raise."""
        self.assertFalse(self.gui.register_dnd())
        self.assertIn("选择文件", str(self.gui.attach_hint.cget("text")))

    def test_register_dnd_succeeds_on_a_drop_capable_root(self):
        from qwen35.gui import HarnessGUI, _make_root

        root = _make_root()
        root.withdraw()
        try:
            gui = HarnessGUI(root, workspace=Path(self._tmp.name),
                             model_dir=Path(self._tmp.name),
                             auto_start_server=False, manage_server=False)
            self.assertTrue(gui.register_dnd(),
                            "drag-and-drop should be available here")
            self.assertTrue(gui._dnd_ready)
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    # -- adding files -------------------------------------------------
    def test_add_files_converts_and_lists(self):
        src = self._make_file("note.txt", "alpha beta gamma")
        self.gui._add_files([str(src)])
        self.assertTrue(self._drain_until(lambda: bool(self.gui.attachments)),
                        "the attachment never arrived")
        self.assertEqual(len(self.gui.attachments), 1)
        self.assertIn("✓", self.gui.attach_list.get(0))
        self.assertIn("note.txt", self.gui.attach_list.get(0))

    def test_a_drop_event_is_parsed(self):
        """tkinterdnd delivers brace-quoted paths; they must be unwrapped."""
        src = self._make_file("dropped.txt", "content")

        class FakeEvent:
            data = "{" + str(src) + "}"

        self.gui._on_drop(FakeEvent())
        self.assertTrue(self._drain_until(lambda: bool(self.gui.attachments)))
        self.assertEqual(self.gui.attachments[0].source.name, src.name)

    def test_a_drop_of_several_files_is_parsed(self):
        a = self._make_file("multi-a.txt", "a")
        b = self._make_file("multi-b.txt", "b")

        class FakeEvent:
            data = "{" + str(a) + "} {" + str(b) + "}"

        self.gui._on_drop(FakeEvent())
        self.assertTrue(self._drain_until(lambda: len(self.gui.attachments) == 2))
        names = sorted(x.source.name for x in self.gui.attachments)
        self.assertEqual(names, sorted([a.name, b.name]))

    def test_a_drop_with_a_missing_path_is_reported_not_crashed(self):
        class FakeEvent:
            data = "{C:\\definitely\\not\\here.txt}"

        self.gui._on_drop(FakeEvent())
        self.root.update()
        self.assertEqual(self.gui.attachments, [])
        self.assertIn("跳过", self.gui.transcript.get("1.0", "end"))

    def test_duplicate_drops_are_ignored(self):
        src = self._make_file("twice.txt", "x")
        self.gui._add_files([str(src)])
        self.assertTrue(self._drain_until(lambda: bool(self.gui.attachments)))
        self.gui._add_files([str(src)])
        self.root.update()
        self.assertEqual(len(self.gui.attachments), 1)
        self.assertEqual(len(self.gui._pending_paths), 1)

    def test_unsupported_file_is_shown_as_failed(self):
        bad = Path(self._tmp.name).parent / "qwen-attach-bad.zzz"
        bad.write_bytes(b"\x00\x01\x02")
        self.addCleanup(lambda: bad.unlink(missing_ok=True))
        self.gui._add_files([str(bad)])
        self.assertTrue(self._drain_until(lambda: bool(self.gui.attachments)))
        self.assertEqual(self.gui.attachments[0].kind, "unsupported")
        self.assertIn("✗", self.gui.attach_list.get(0))

    # -- managing the list --------------------------------------------
    def test_remove_selected(self):
        src = self._make_file("rm.txt", "x")
        self.gui._add_files([str(src)])
        self.assertTrue(self._drain_until(lambda: bool(self.gui.attachments)))
        self.gui.attach_list.selection_set(0)
        self.gui._remove_selected_attachments()
        self.assertEqual(self.gui.attachments, [])
        self.assertEqual(self.gui._pending_paths, [])

    def test_clear(self):
        src = self._make_file("clr.txt", "x")
        self.gui._add_files([str(src)])
        self.assertTrue(self._drain_until(lambda: bool(self.gui.attachments)))
        self.gui._clear_attachments()
        self.assertEqual(self.gui.attachments, [])
        self.assertIn("还没有附件", self.gui.attach_list.get(0))

    def test_register_dnd_does_not_raise(self):
        # returns True with tkinterdnd2, False without; either way it must work
        self.assertIn(self.gui.register_dnd(), (True, False))

    def test_clear_starts_a_new_conversation(self):
        """清空 must reset everything, not just the transcript text.

        It used to wipe the text and leave the attachments attached, so the next
        message still carried the files from the conversation the user believed
        they had just ended.
        """
        import tkinter.messagebox as messagebox

        from PIL import Image

        original = messagebox.askyesno
        messagebox.askyesno = lambda *a, **k: True       # auto-confirm
        try:
            note = self._make_file("note.txt", "hello from a file\n")
            picture = Path(self._tmp.name).parent / "qwen-clear-pic.png"
            Image.new("RGB", (24, 24), "red").save(picture)
            self.addCleanup(lambda: picture.unlink(missing_ok=True))
            self.gui._add_files([str(note), str(picture)])
            self.assertTrue(self._drain_until(
                lambda: len(self.gui.attachments) == 2), "attachments missing")

            store = Path(self._tmp.name) / ".qwen35" / "attachments"
            self.assertTrue([p for p in store.rglob("*") if p.is_file()])

            self.gui._clear()

            self.assertEqual(self.gui.attachments, [])
            self.assertEqual(self.gui._pending_paths, [])
            self.assertIsNone(self.gui._last_stats)
            self.assertFalse([p for p in store.rglob("*") if p.is_file()],
                             "converted copies should not be left behind")
            body = self.gui.transcript.get("1.0", "end")
            self.assertEqual(body.count("qwen35 harness"), 1,
                             "the banner should be reprinted exactly once")
            # the user's own files are never touched
            self.assertTrue(note.is_file())
            self.assertTrue(picture.is_file())
        finally:
            messagebox.askyesno = original

    def test_clear_is_refused_while_a_task_is_running(self):
        import tkinter.messagebox as messagebox

        seen: list[str] = []
        original_warn = messagebox.showwarning
        original_ask = messagebox.askyesno
        messagebox.showwarning = lambda title, msg, **k: seen.append(title)
        messagebox.askyesno = lambda *a, **k: True
        try:
            self.gui._busy = True
            self.gui._clear()
            self.assertTrue(seen, "a running task must block the reset")
        finally:
            messagebox.showwarning = original_warn
            messagebox.askyesno = original_ask
            self.gui._busy = False

    # -- composing the task -------------------------------------------
    def test_compose_task_without_attachments_is_unchanged(self):
        self.assertEqual(self.gui._compose_task("do the thing"), "do the thing")

    def test_small_attachment_is_inlined_in_the_task(self):
        src = self._make_file("small.txt", "the secret word is platypus")
        self.gui._add_files([str(src)])
        self.assertTrue(self._drain_until(lambda: bool(self.gui.attachments)))
        composed = self.gui._compose_task("这个词是什么？")
        self.assertIn("# Attached files", composed)
        self.assertIn("platypus", composed)
        self.assertIn("# Request", composed)
        self.assertIn("这个词是什么？", composed)

    def test_large_attachment_points_at_the_path_instead(self):
        src = self._make_file("big.txt", "\n".join(f"line {i}" for i in range(4000)))
        self.gui._add_files([str(src)])
        self.assertTrue(self._drain_until(lambda: bool(self.gui.attachments)))
        composed = self.gui._compose_task("summarise it")
        self.assertIn("read_file", composed)
        self.assertIn(self.gui.attachments[0].out_rel, composed)
        self.assertNotIn("<attached-text>", composed)

    def test_empty_input_with_no_attachments_does_nothing(self):
        self.gui._on_send()
        self.assertFalse(self.gui._busy)
        self.assertEqual(self.gui.transcript.get("1.0", "end").count("任务"), 0)


@unittest.skipUnless(TK_OK, SKIP_REASON)
class TestAdaptiveLayout(SharedTkRoot, unittest.TestCase):
    """Every row must reflow, and the window must be allowed to get narrow.

    The window used to be pinned at a minimum of 880 px because the fixed rows
    clipped below that -- which also meant the rows never had a chance to wrap.
    A window nobody can make small is not an adaptive one.
    """

    @classmethod
    def setUpClass(cls):
        # ONE Tk interpreter for the whole class.  Tk does not survive being
        # created and destroyed dozens of times in a single process: the run
        # aborts with `Tcl_AsyncDelete: async handler deleted by the wrong
        # thread`, an access violation, after enough cycles.  Each class passes
        # alone; it is the accumulation that kills it, so the fix is fewer
        # interpreters rather than fewer assertions.
        cls.root = tk.Tk()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def setUp(self):
        from qwen35.gui import HarnessGUI

        self.root = self.fresh_root()
        self._tmp = tempfile.TemporaryDirectory()
        self.gui = HarnessGUI(
            self.root, workspace=Path(self._tmp.name), model_dir=Path(self._tmp.name),
            auto_start_server=False, manage_server=False,
        )
        self.root.update()
        self.root.update_idletasks()

    def tearDown(self):
        self._tmp.cleanup()

    def _shape(self, frame) -> list[int]:
        rows: dict[int, int] = {}
        for child in frame._items:
            info = child.place_info()
            if info:
                rows[int(info["y"])] = rows.get(int(info["y"]), 0) + 1
        return [rows[key] for key in sorted(rows)]

    def test_every_row_is_a_flow_frame(self):
        from qwen35.gui import FlowFrame

        # header, workspace, options, actions, attachments, input+buttons
        self.assertGreaterEqual(len(self.gui._flows), 6)
        for frame in self.gui._flows:
            self.assertIsInstance(frame, FlowFrame)

    def test_the_window_minimum_leaves_room_to_wrap(self):
        """The old floor was 880 px; that is wider than several whole rows."""
        minimum = self.root.minsize()
        self.assertLessEqual(minimum[0], 600,
                             "the window is pinned too wide for the rows to wrap")
        self.assertLessEqual(minimum[1], 500)

    def test_rows_wrap_as_the_window_narrows(self):
        seen_wrapping = {"header": False, "options": False, "actions": False,
                         "attachments": False, "bottom": False}
        names = ["header", "workspace", "options", "actions", "attachments",
                 "bottom"]
        for width in (1400, 900, 760, 640, 560):
            self.root.geometry(f"{width}x780")
            for _ in range(4):
                self.root.update()
                self.root.update_idletasks()
            self.gui._reflow_all()
            self.root.update_idletasks()
            for index, frame in enumerate(self.gui._flows):
                if index >= len(names):
                    break
                if len(self._shape(frame)) > 1:
                    seen_wrapping[names[index]] = True
        unwrapped = [name for name, wrapped in seen_wrapping.items() if not wrapped]
        self.assertEqual(unwrapped, [],
                         f"these rows never wrapped: {unwrapped}")

    def test_nothing_overflows_its_row(self):
        """No child may be placed past the row's right edge.

        A long label or a wide combobox can be naturally wider than a narrow
        window; the row clamps it so the layout stays inside its bounds instead
        of quietly relying on the parent to clip.
        """
        for width in (1400, 900, 700, 560):
            self.root.geometry(f"{width}x780")
            for _ in range(4):
                self.root.update()
                self.root.update_idletasks()
            self.gui._reflow_all()
            self.root.update_idletasks()
            for frame in self.gui._flows:
                limit = frame.winfo_width() + 2
                for child in frame._items:
                    info = child.place_info()
                    if not info:
                        continue
                    right = int(info["x"]) + int(info["width"])
                    self.assertLessEqual(
                        right, limit,
                        f"{child} overflows {frame} at width {width} "
                        f"({right} > {limit})")
                    self.assertGreaterEqual(int(info["x"]), 0)

    def test_the_bottom_row_stacks_beside_then_below(self):
        """The row the user flagged: buttons beside the box, then underneath."""
        bottom = self.gui._flows[-1]
        self.root.geometry("1400x780")
        for _ in range(4):
            self.root.update()
            self.root.update_idletasks()
        self.gui._reflow_all()
        self.root.update_idletasks()
        wide = self._shape(bottom)
        self.root.geometry("560x780")
        for _ in range(4):
            self.root.update()
            self.root.update_idletasks()
        self.gui._reflow_all()
        self.root.update_idletasks()
        narrow = self._shape(bottom)
        self.assertEqual(len(wide), 1, f"expected one row when wide, got {wide}")
        self.assertGreater(len(narrow), 1,
                           f"expected the bottom row to wrap when narrow, got {narrow}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
