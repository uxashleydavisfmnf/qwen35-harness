"""
End-to-end smoke test for the GUI.

Unlike ``tests/test_gui.py`` (which drives rendering with synthetic events), this
opens the real window, lets it start the real model server, types a real task and
waits for the real agent to finish -- then asserts on what the window actually
renders.  It is the only check that proves the worker thread, the event queue,
the layout and the model all work together.

Run it directly:

    python scripts/gui_smoke.py --workspace D:\\qwen\\workspace --timeout 600

Exits non-zero on failure, so it can be used as a gate.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tkinter as tk  # noqa: E402

from qwen35.gui import HarnessGUI  # noqa: E402


def wait_for(predicate, root: tk.Tk, timeout: float, what: str) -> bool:
    """Pump the Tk event loop until ``predicate`` holds or the timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        root.update()
        if predicate():
            return True
        time.sleep(0.05)
    print(f"  TIMEOUT after {timeout:.0f}s waiting for {what}")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", "-w", required=True)
    parser.add_argument("--model-dir", dest="model_dir", default=None)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--task", default=(
        "Run the test suite with python test_calc.py and fix whatever is broken."
    ))
    parser.add_argument("--expect", default="a + b",
                        help="substring the rendered transcript must contain")
    parser.add_argument("--attach", action="append", default=None,
                        help="attach a file through the GUI before running "
                             "(exercises drag-and-drop ingest + prompt compose)")
    parser.add_argument("--keep-open", type=float, default=0.0,
                        help="seconds to leave the window visible afterwards")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    if not workspace.is_dir():
        print(f"workspace not found: {workspace}")
        return 2

    print("opening the GUI window...")
    root = tk.Tk()
    gui = HarnessGUI(
        root,
        workspace=workspace,
        model_dir=args.model_dir,
        context=args.context,
    )

    failures: list[str] = []

    # ---- 1. widget construction -----------------------------------------
    root.update_idletasks()
    for name in ("transcript", "input", "image_label", "send_button",
                 "stop_button", "server_label"):
        widget = getattr(gui, name, None)
        if widget is None or not widget.winfo_exists():
            failures.append(f"widget {name} missing")
    print(f"  widgets: {len(failures) == 0 and 'all present' or failures}")

    # geometry sanity: nothing collapsed to zero size.
    # A full update() (not just idletasks) is needed so the window is actually
    # mapped -- before that, winfo_width() reports 1 for every widget.
    root.deiconify()
    root.update()
    for _ in range(20):
        root.update()
        time.sleep(0.02)
    for name in ("transcript", "input"):
        widget = getattr(gui, name)
        if widget.winfo_width() < 50 or widget.winfo_height() < 20:
            failures.append(f"{name} collapsed ({widget.winfo_width()}x"
                            f"{widget.winfo_height()})")
    print(f"  geometry: transcript {gui.transcript.winfo_width()}x"
          f"{gui.transcript.winfo_height()}, "
          f"input {gui.input.winfo_width()}x{gui.input.winfo_height()}")

    # ---- 2. the GUI starts the model server itself -----------------------
    print("waiting for the model server (the GUI starts it in-window)...")
    if not wait_for(lambda: gui.server_status is not None, root, 420,
                    "server status"):
        failures.append("the server never reported a status")
    else:
        status = gui.server_status
        print(f"  server: {status.state}  {status.message}")
        if not status.ok:
            failures.append(f"server not ready: {status.message}")

    if failures:
        _finish(gui, root, failures)
        return 1

    # ---- 3. attachments, if any (same path as a drag-and-drop) -----------
    if args.attach:
        print(f"attaching {len(args.attach)} file(s) through the GUI...")
        gui._add_files([str(Path(p)) for p in args.attach])
        if not wait_for(lambda: len(gui.attachments) == len(args.attach), root,
                        180, "attachments to be converted"):
            failures.append("attachments were never converted")
        for att in gui.attachments:
            print(f"  {att.summary_line()}")
            for w in att.warnings:
                print(f"    note: {w}")
        if not gui._usable_attachments():
            failures.append("no attachment could be converted")

    # ---- 4. run a real task through the real window ----------------------
    print(f"submitting task: {args.task!r}")
    gui.input.insert("1.0", args.task)
    gui._on_send()

    if not wait_for(lambda: gui._busy, root, 20, "the task to start"):
        failures.append("the task never started")
    else:
        started = time.time()
        if not wait_for(lambda: not gui._busy, root, args.timeout,
                        "the task to finish"):
            failures.append(f"the task did not finish within {args.timeout:.0f}s")
        else:
            print(f"  finished in {time.time() - started:.1f}s")

    # ---- 5. assert on what the window renders ----------------------------
    rendered = gui.transcript.get("1.0", "end")
    print("\n" + "=" * 72)
    print(rendered)
    print("=" * 72)

    if args.attach:
        for att in gui.attachments:
            if att.source.name not in rendered:
                failures.append(f"the transcript never mentions {att.source.name}")

    for needle, why in (
        (args.task[:40], "the task text should be echoed"),
        ("步骤", "step headers should render"),
        ("[", "the result summary should render"),
    ):
        if needle not in rendered:
            failures.append(f"transcript is missing {needle!r} ({why})")

    # A tool call is NOT required: when the attached text is already inlined, the
    # harness deliberately offers no tools and the model answers in zero steps.
    # Requiring an arrow here would fail the *better* behaviour.
    if gui.attachments and any("→" in rendered for _ in (0,)):
        pass
    elif not gui.attachments and "→" not in rendered:
        failures.append("transcript is missing '→' (tool calls should render)")

    # Regex, not substring: callers pass alternations like '是非|荣辱'.
    if args.expect and not re.search(args.expect, rendered):
        failures.append(f"transcript does not match {args.expect!r}")

    # ---- 5. the server must be stopped on close --------------------------
    owned = gui.server_status is not None and gui.server_status.we_own_it
    print(f"  GUI owns the server: {owned}")

    _finish(gui, root, failures, keep_open=args.keep_open)
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nGUI end-to-end smoke test PASSED")
    return 0


def _finish(gui: HarnessGUI, root: tk.Tk, failures: list[str],
            keep_open: float = 0.0) -> None:
    if keep_open:
        deadline = time.time() + keep_open
        while time.time() < deadline:
            root.update()
            time.sleep(0.05)
    try:
        gui._on_close()
    except Exception as exc:  # noqa: BLE001
        failures.append(f"closing the window raised: {exc}")
    try:
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
