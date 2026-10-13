"""
One window at a time, and a lock that holds across processes.

Two problems, one mechanism.

**The window.**  Launching the GUI twice gave two windows sharing one port and
one ``server.json``, which is how the harness ended up in the state that produced

    a server is running but does not accept the stored API key

A second launch should bring the existing window forward instead of building a
rival one.

**The server start.**  ``ensure_server`` holds an in-process lock, but the CLI and
the GUI are different processes.  A named mutex fixes that, and -- unlike a lock
file -- Windows releases it automatically when the owner dies, so a crash cannot
leave a stale lock behind.  A crashed owner surfaces as ``WAIT_ABANDONED``, which
means "the previous holder died; you may take it".

Windows is the target; everything degrades to a no-op or a file lock elsewhere
rather than failing.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

IS_WINDOWS = sys.platform == "win32"

#: Used by FindWindow/EnumWindows.  Keep it distinctive: the sweep matches a
#: window whose title *starts with* this.
WINDOW_TITLE = "qwen35 - Qwen3.5-0.8B harness"
#: The " - " matters.  A console launched by gui.bat is titled
#: "管理员: Administrator:  qwen35 GUI", and a browser tab can contain the word
#: too; requiring the separator keeps the match to the real window.
WINDOW_TITLE_PREFIX = "qwen35 - "

#: ``Local\`` scopes the name to the logon session, so two different users on one
#: machine do not fight over a window that is not theirs.
INSTANCE_MUTEX = r"Local\qwen35-harness-gui"
SERVER_MUTEX = r"Local\qwen35-harness-server-start"

_ERROR_ALREADY_EXISTS = 183
_ERROR_INVALID_NAME = 123
_WAIT_OBJECT_0 = 0x0
_WAIT_ABANDONED = 0x80
_WAIT_TIMEOUT = 0x102

_SW_RESTORE = 9
_SW_SHOW = 5

_user32: Any = None
_kernel32: Any = None


def _win() -> bool:
    """Load kernel32/user32 with correct prototypes.

    ``use_last_error=True`` is not optional: without it ``ctypes.get_last_error``
    reports whatever the thread happened to leave behind, so ERROR_ALREADY_EXISTS
    is never seen.  The ``restype``/``argtypes`` matter just as much -- a HANDLE
    is pointer-sized, and letting ctypes assume ``c_int`` truncates it to 32 bits
    on a 64-bit interpreter, after which ReleaseMutex and CloseHandle are being
    handed a different value than CreateMutexW returned.
    """
    global _user32, _kernel32
    if not IS_WINDOWS:
        return False
    if _user32 is None:
        import ctypes
        from ctypes import wintypes

        _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _kernel32.CreateMutexW.restype = wintypes.HANDLE
        _kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL,
                                           wintypes.LPCWSTR]
        _kernel32.WaitForSingleObject.restype = wintypes.DWORD
        _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        _kernel32.ReleaseMutex.restype = wintypes.BOOL
        _kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
        _kernel32.CloseHandle.restype = wintypes.BOOL
        _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        _user32 = ctypes.WinDLL("user32", use_last_error=True)
        _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR,
                                           ctypes.c_int]
        _user32.IsWindowVisible.argtypes = [wintypes.HWND]
        _user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        _user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        _user32.GetForegroundWindow.restype = wintypes.HWND
        _user32.BringWindowToTop.argtypes = [wintypes.HWND]
    return True


# --------------------------------------------------------------------------
# Cross-process locks
# --------------------------------------------------------------------------
class NamedLock:
    """A Windows named mutex, or a lock file elsewhere.

    ``acquire()`` is non-blocking by default and returns whether the lock is
    ours.  An abandoned mutex (the previous holder crashed) counts as acquired:
    that is the whole point of using the OS instead of a file.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._handle: Any = None
        self._file: Any = None

    def acquire(self, timeout: float = 0.0) -> bool:
        if _win():
            import ctypes

            self._handle = _kernel32.CreateMutexW(None, True, self.name)
            if not self._handle:
                # Do not guess at the naming rules -- "Local\name" has one
                # backslash and is valid, "Local\\name" has two and is not.  Ask
                # the OS, and fail loudly: a silent False here would look
                # exactly like "another instance already has it".
                if ctypes.get_last_error() == _ERROR_INVALID_NAME:
                    raise ValueError(
                        f"{self.name!r} is not a valid Windows mutex name "
                        "(use at most one backslash, as in 'Local\\qwen35')")
                return False
            if ctypes.get_last_error() != _ERROR_ALREADY_EXISTS:
                return True                    # we created it; it is ours
            wait = _kernel32.WaitForSingleObject(
                self._handle, int(timeout * 1000) if timeout else 0)
            if wait in (_WAIT_OBJECT_0, _WAIT_ABANDONED):
                # WAIT_ABANDONED means the previous holder died without
                # releasing.  We now own it, which is exactly why an OS mutex
                # beats a lock file here.
                return True
            _kernel32.CloseHandle(self._handle)
            self._handle = None
            return False

        # Portable fallback: an exclusively-opened file.  A crash leaves the file
        # behind, so a stale one is aged out rather than trusted.
        path = Path(os.environ.get("QWEN35_HOME", Path.home() / ".qwen35")) / (
            self.name.replace("\\", "-") + ".lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + max(timeout, 0)
        while True:
            try:
                import fcntl

                self._file = open(path, "w")
                fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except Exception:  # noqa: BLE001
                if self._file:
                    self._file.close()
                    self._file = None
                if time.time() >= deadline:
                    return False
                time.sleep(0.05)

    def release(self) -> None:
        if self._handle is not None:
            _kernel32.ReleaseMutex(self._handle)
            _kernel32.CloseHandle(self._handle)
            self._handle = None
        if self._file is not None:
            try:
                self._file.close()
            except Exception:  # noqa: BLE001
                pass
            self._file = None

    def __enter__(self) -> "NamedLock":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


@contextmanager
def server_start_lock(timeout: float = 240.0) -> Iterator[bool]:
    """Serialise server starts across processes.

    Yields True if the lock was taken.  If it could not be, the caller should
    still proceed -- refusing to start a server because a lock is busy would be
    worse than a race that ``ensure_server``'s ownership check already catches.
    """
    lock = NamedLock(SERVER_MUTEX)
    held = lock.acquire(timeout)
    try:
        yield held
    finally:
        if held:
            lock.release()


# --------------------------------------------------------------------------
# Finding and raising the window that is already open
# --------------------------------------------------------------------------
def find_window(prefix: str = WINDOW_TITLE_PREFIX) -> Any:
    """Return the HWND of a visible top-level window whose title starts with
    ``prefix``, or 0."""
    if not _win():
        return 0
    import ctypes
    from ctypes import wintypes

    found: list[int] = []
    lowered = prefix.lower()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):  # type: ignore[no-untyped-def]
        if not _user32.IsWindowVisible(hwnd):
            return True
        length = _user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buffer, length + 1)
        if buffer.value.lower().startswith(lowered):
            found.append(hwnd)
            return False
        return True

    _user32.EnumWindows(callback, 0)
    return found[0] if found else 0


def focus_window(prefix: str = WINDOW_TITLE_PREFIX) -> bool:
    """Bring an existing qwen35 window to the front.  True if one was found.

    ``SetForegroundWindow`` is refused unless the caller is entitled to steal
    focus.  A freshly launched process normally is, but if the call is refused we
    flash the taskbar button instead -- a blinking icon is a worse answer than a
    raised window but a much better one than silence.
    """
    hwnd = find_window(prefix)
    if not hwnd:
        return False
    if not _win():
        return True

    _user32.ShowWindow(hwnd, _SW_RESTORE)
    _user32.SetForegroundWindow(hwnd)
    if _user32.GetForegroundWindow() == hwnd:
        return True
    _user32.BringWindowToTop(hwnd)
    _user32.ShowWindow(hwnd, _SW_SHOW)
    if _user32.GetForegroundWindow() == hwnd:
        return True
    _flash(hwnd)
    return True


def _flash(hwnd: Any) -> None:
    """Blink the taskbar button so the user can find the window themselves."""
    try:
        import ctypes
        from ctypes import wintypes

        class FLASHWINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("hwnd", wintypes.HWND),
                        ("dwFlags", wintypes.DWORD), ("uCount", wintypes.UINT),
                        ("dwTimeout", wintypes.DWORD)]

        info = FLASHWINFO(ctypes.sizeof(FLASHWINFO), hwnd, 0x00000003, 5, 0)
        _user32.FlashWindowEx(ctypes.byref(info))
    except Exception:  # noqa: BLE001 - purely cosmetic
        pass


def claim_single_instance() -> NamedLock | None:
    """Take the single-instance lock, or return None if a window already has it."""
    lock = NamedLock(INSTANCE_MUTEX)
    return lock if lock.acquire() else None
