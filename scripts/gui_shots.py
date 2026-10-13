"""
Screenshot the real GUI at several window widths.

Written because coordinate assertions only prove that *place* was called with
plausible numbers -- they cannot show whether the result looks right.  With a
vision-capable reader on the other end this produces actual evidence: crop the
window at a wide size and at a narrow one, and look.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen35.gui import HarnessGUI, _make_root  # noqa: E402

OUT = Path(r"D:\qwen\state\shots")
OUT.mkdir(parents=True, exist_ok=True)
SIZES = [(1400, 860), (880, 860), (640, 860), (520, 860)]


def main() -> int:
    from PIL import ImageGrab

    root = _make_root()
    tmp = tempfile.TemporaryDirectory()
    gui = HarnessGUI(root, workspace=Path(tmp.name), model_dir=Path(tmp.name),
                     auto_start_server=False, manage_server=False)
    root.update()
    root.update_idletasks()
    gui._act_doctor()          # put some content in the transcript
    gui._act_tools()

    for width, height in SIZES:
        root.geometry(f"{width}x{height}+80+60")
        for _ in range(6):
            root.update()
            root.update_idletasks()
        gui._reflow_all()
        root.update()
        root.update_idletasks()
        time.sleep(0.35)

        root.lift()
        root.attributes("-topmost", True)
        root.update()
        time.sleep(0.25)

        left = root.winfo_rootx()
        top = root.winfo_rooty()
        w = root.winfo_width()
        h = root.winfo_height()
        # Measured: columns 0-5 of an un-inset grab are the light DWM window
        # shadow, not the application -- the app's dark background (30,31,38)
        # starts about 22 px in.  Inset past the shadow and below the title bar.
        inset_x, inset_top, inset_bottom, inset_right = 28, 40, 10, 8
        shot = ImageGrab.grab(
            bbox=(left + inset_x, top + inset_top,
                  left + w - inset_right, top + h - inset_bottom))
        target = OUT / f"gui-{width}.png"
        shot.save(target)
        print(f"  {target}  ({shot.width}x{shot.height})")

    root.attributes("-topmost", False)
    root.destroy()
    tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
