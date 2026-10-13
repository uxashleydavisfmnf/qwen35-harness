"""
Workspace selection and confinement.

Two guarantees this module provides:

1. **Selection** -- the model (and the operator) may only ever operate inside a
   workspace that passes :func:`validate_workspace`.  Filesystem roots, the home
   directory itself, and every operating-system directory are refused, both as a
   workspace *and* as an ancestor of one.  Picking ``C:\\`` is impossible.

2. **Confinement** -- once chosen, every path the model asks for is re-resolved
   and proven to live under that workspace.  ``..`` traversal, absolute paths,
   symlink/junction escapes, NTFS alternate data streams and reserved device
   names are all rejected.  Confinement is enforced on the *real* path, so a
   symlink pointing outside the workspace does not become a hole.
"""

from __future__ import annotations

import json
import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

__all__ = [
    "WorkspaceError",
    "WorkspaceGuard",
    "validate_workspace",
    "candidate_workspaces",
    "registry_path",
    "remember_workspace",
    "recent_workspaces",
]


class WorkspaceError(Exception):
    """Raised when a workspace is refused, or a path escapes confinement."""


# --------------------------------------------------------------------------
# Reserved names / patterns that must never reach the filesystem
# --------------------------------------------------------------------------
_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_UNC_PREFIX = re.compile(r"^(\\\\|//)")


def _realpath(p: Path) -> Path:
    try:
        return Path(os.path.realpath(p))
    except OSError:
        return p


def _hard_system_paths() -> list[Path]:
    """OS directories: never a workspace, never an ancestor, never inside."""
    out: list[Path] = []
    env = os.environ
    for key in ("SystemRoot", "windir", "ProgramFiles", "ProgramFiles(x86)",
                "ProgramW6432", "ProgramData"):
        v = env.get(key)
        if v:
            out.append(Path(v))
    if env.get("SystemRoot"):
        out.append(Path(env["SystemRoot"]) / "System32")
    return out


def _user_data_paths() -> list[Path]:
    """Credential-bearing user directories.

    Not selectable as a workspace, and not an ancestor of one.  Being *inside*
    one is refused too -- with one deliberate exception, because the operating
    system's own temp directory lives under ``%LOCALAPPDATA%`` and exists
    precisely for scratch work, which is exactly what a sandbox is.
    """
    out: list[Path] = []
    for key in ("APPDATA", "LOCALAPPDATA"):
        v = os.environ.get(key)
        if v:
            out.append(Path(v))
    return out


def _extra_protected_paths() -> list[Path]:
    """Operator-supplied paths that must never be a workspace.

    Read from ``QWEN35_PROTECTED_PATHS`` (``os.pathsep``-separated).  The bundled
    launcher uses this to protect its own installation directory, so the model
    can never be pointed at the code that is running it.
    """
    raw = os.environ.get("QWEN35_PROTECTED_PATHS", "")
    out: list[Path] = []
    for part in raw.split(os.pathsep):
        part = part.strip().strip('"')
        if part:
            out.append(Path(part))
    return out


def _scratch_paths() -> list[Path]:
    out: list[Path] = []
    for key in ("TEMP", "TMP"):
        v = os.environ.get(key)
        if v:
            out.append(Path(v))
    try:
        import tempfile

        out.append(Path(tempfile.gettempdir()))
    except Exception:  # pragma: no cover - platform dependent
        pass
    return out


def _drive_roots() -> list[Path]:
    roots: list[Path] = []
    for letter in string.ascii_uppercase:
        p = Path(f"{letter}:\\")
        try:
            if p.exists():
                roots.append(p)
        except OSError:
            continue
    return roots


def _same_or_ancestor(candidate: Path, other: Path) -> bool:
    """True when ``candidate`` is ``other`` or a parent of it."""
    try:
        return other == candidate or other.is_relative_to(candidate)
    except (ValueError, OSError):
        return False


def _inside(candidate: Path, other: Path) -> bool:
    try:
        return candidate.is_relative_to(other) and candidate != other
    except (ValueError, OSError):
        return False


def validate_workspace(raw: str | os.PathLike[str], *, create: bool = False) -> Path:
    """Validate and canonicalise a candidate workspace directory.

    Raises :class:`WorkspaceError` with an operator-readable reason on refusal.
    Returns the resolved (realpath) workspace root on success.
    """
    text = str(raw).strip().strip('"').strip("'")
    if not text:
        raise WorkspaceError("empty path")

    if _UNC_PREFIX.match(text):
        raise WorkspaceError(
            "network (UNC) paths are refused: the harness cannot guarantee "
            "confinement on a share"
        )

    p = Path(text).expanduser()
    try:
        p = p.absolute()
    except OSError as exc:
        raise WorkspaceError(f"unusable path: {exc}") from exc

    if not p.exists():
        if create:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise WorkspaceError(f"could not create workspace: {exc}") from exc
        else:
            raise WorkspaceError(
                f"{p} does not exist (pass --create to create it)"
            )

    if not p.is_dir():
        raise WorkspaceError(f"{p} is not a directory")

    try:
        real = Path(os.path.realpath(p))
    except OSError as exc:
        raise WorkspaceError(f"could not resolve {p}: {exc}") from exc

    # --- 1. never a filesystem root --------------------------------------
    for root in _drive_roots():
        try:
            rreal = Path(os.path.realpath(root))
        except OSError:
            rreal = root
        if real == rreal:
            raise WorkspaceError(
                f"refusing {real}: a filesystem root is never a workspace. "
                "Choose a project directory below it."
            )

    if real.parent == real:  # POSIX '/'
        raise WorkspaceError(
            f"refusing {real}: a filesystem root is never a workspace."
        )

    # --- 2. never a protected directory, nor inside one, nor above one ----
    home: Path | None = None
    try:
        home = Path(os.path.realpath(Path.home()))
    except (RuntimeError, OSError):
        pass

    # The home directory gets its own message *before* the generic protected-path
    # sweep, because home contains AppData and the generic check would otherwise
    # explain the refusal in terms of "AppData" rather than the real problem.
    if home is not None:
        if real == home:
            raise WorkspaceError(
                f"refusing {real}: the home directory itself would expose every "
                "credential and profile on this account. Choose a subdirectory "
                "such as ~/work."
            )
        if _same_or_ancestor(real, home):
            raise WorkspaceError(
                f"refusing {real}: it is above the home directory and would "
                "grant write access to every user profile."
            )

    for prot in _hard_system_paths():
        preal = _realpath(prot)
        if _same_or_ancestor(real, preal):
            raise WorkspaceError(
                f"refusing {real}: it is (or contains) the protected system "
                f"location {preal}."
            )
        if _inside(real, preal):
            raise WorkspaceError(
                f"refusing {real}: it is inside the protected system location "
                f"{preal}."
            )

    for prot in _extra_protected_paths():
        preal = _realpath(prot)
        if not preal.exists():
            continue
        if _same_or_ancestor(real, preal):
            raise WorkspaceError(
                f"refusing {real}: it is (or contains) the operator-protected "
                f"path {preal}."
            )
        if _inside(real, preal):
            raise WorkspaceError(
                f"refusing {real}: it is inside the operator-protected path "
                f"{preal}."
            )

    scratch = [_realpath(p) for p in _scratch_paths()]
    in_scratch = any(real == s or _inside(real, s) for s in scratch)

    for prot in _user_data_paths():
        preal = _realpath(prot)
        if _same_or_ancestor(real, preal):
            raise WorkspaceError(
                f"refusing {real}: it is (or contains) the protected user-data "
                f"location {preal}."
            )
        if _inside(real, preal) and not in_scratch:
            raise WorkspaceError(
                f"refusing {real}: it is inside the protected user-data "
                f"location {preal}. (Only the OS temp directory under it is "
                "allowed.)"
            )

    # --- 3. sanity: must be writable-ish and addressable ------------------
    if _CONTROL_CHARS.search(str(real)):
        raise WorkspaceError("path contains control characters")

    return real


# --------------------------------------------------------------------------
# Guard
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedPath:
    """A confinement-checked path plus how to show it to the model."""

    abs: Path
    rel: str
    existed: bool

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.rel


class WorkspaceGuard:
    """Proves that every path the model touches lives inside the workspace."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(os.path.realpath(Path(root)))

    # -- containment -------------------------------------------------------
    def contains(self, path: Path) -> bool:
        try:
            real = Path(os.path.realpath(path))
        except OSError:
            return False
        try:
            return real == self.root or real.is_relative_to(self.root)
        except (ValueError, OSError):
            return False

    def resolve(self, raw: str, *, must_exist: bool = False) -> ResolvedPath:
        """Turn a model-supplied path into a confined absolute path.

        Rejects: absolute paths outside the workspace, ``..`` traversal,
        symlink/junction escapes, alternate data streams and reserved names.
        """
        if raw is None:
            raise WorkspaceError("missing path")
        if not isinstance(raw, str):
            raise WorkspaceError(f"path must be a string, got {type(raw).__name__}")

        text = raw.strip().replace("\u00a0", " ")
        if not text:
            raise WorkspaceError("empty path")
        if _CONTROL_CHARS.search(text):
            raise WorkspaceError("path contains control characters")
        if len(text) > 4096:
            raise WorkspaceError("path is absurdly long")

        # normalise separators so "a/b" works on Windows and C:\ works on posix
        norm = text.replace("\\", os.sep).replace("/", os.sep)
        norm = norm.lstrip(os.sep) if not Path(norm).is_absolute() else norm

        candidate = Path(norm)
        if not candidate.is_absolute():
            candidate = self.root / candidate

        # alternate data streams: "file.txt:secret" (but keep the drive colon)
        parts = candidate.parts
        for part in parts[1:] if candidate.drive else parts:
            if ":" in part:
                raise WorkspaceError(
                    f"alternate data streams are not allowed in {raw!r}"
                )

        # reserved device names
        for part in candidate.parts:
            stem = part.split(".")[0].lower()
            if stem in _RESERVED:
                raise WorkspaceError(f"reserved device name in {raw!r}")

        try:
            normalised = Path(os.path.normpath(candidate))
        except OSError as exc:
            raise WorkspaceError(f"bad path {raw!r}: {exc}") from exc

        # lexical containment first (cheap, catches ../ escapes even for
        # paths that do not exist yet)
        try:
            lex_ok = normalised == self.root or normalised.is_relative_to(self.root)
        except (ValueError, OSError):
            lex_ok = False
        if not lex_ok:
            raise WorkspaceError(
                f"path escapes the workspace: {raw!r} resolves to {normalised}, "
                f"which is outside {self.root}"
            )

        # real containment (catches symlinks/junctions pointing outside).
        # Resolve against the nearest existing ancestor so that creating new
        # files in new subdirectories still gets a real check.
        probe = normalised
        suffix_parts: list[str] = []
        while not probe.exists():
            if probe.parent == probe:
                break
            suffix_parts.append(probe.name)
            probe = probe.parent
        try:
            real_probe = Path(os.path.realpath(probe))
        except OSError as exc:
            raise WorkspaceError(f"could not resolve {raw!r}: {exc}") from exc
        for name in reversed(suffix_parts):
            real_probe = real_probe / name

        try:
            real_ok = (
                real_probe == self.root or real_probe.is_relative_to(self.root)
            )
        except (ValueError, OSError):
            real_ok = False
        if not real_ok:
            raise WorkspaceError(
                f"path escapes the workspace through a link: {raw!r} -> "
                f"{real_probe}"
            )

        existed = normalised.exists()
        if must_exist and not existed:
            raise WorkspaceError(f"no such file or directory: {self.rel(normalised)}")

        return ResolvedPath(abs=normalised, rel=self.rel(normalised), existed=existed)

    # -- presentation ------------------------------------------------------
    def rel(self, path: Path) -> str:
        try:
            r = Path(os.path.realpath(path)).relative_to(self.root)
        except (ValueError, OSError):
            try:
                r = path.relative_to(self.root)
            except ValueError:
                return str(path)
        s = r.as_posix()
        return s if s else "."

    def display(self, path: Path) -> str:
        return self.rel(path)


# --------------------------------------------------------------------------
# Workspace registry (so the picker can offer what you used before)
# --------------------------------------------------------------------------
def registry_path() -> Path:
    base = Path(os.environ.get("QWEN35_HOME", Path.home() / ".qwen35"))
    return base / "workspaces.json"


def recent_workspaces(limit: int = 12) -> list[str]:
    p = registry_path()
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = data.get("workspaces", []) if isinstance(data, dict) else []
    out: list[str] = []
    for item in items:
        path = item.get("path") if isinstance(item, dict) else item
        if isinstance(path, str) and path and path not in out:
            out.append(path)
    return out[:limit]


def remember_workspace(path: str | os.PathLike[str]) -> None:
    p = registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = recent_workspaces(limit=50)
    text = str(path)
    ordered = [text] + [e for e in existing if e != text]
    payload = {"workspaces": [{"path": e} for e in ordered[:50]]}
    try:
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass  # a non-writable registry must never break a run


def candidate_workspaces(max_depth: int = 2, limit: int = 40) -> list[str]:
    """Cheap discovery of plausible project directories for the picker."""
    found: list[str] = []
    seen: set[str] = set()

    def offer(p: Path) -> None:
        s = str(p)
        if s not in seen:
            seen.add(s)
            found.append(s)

    for entry in recent_workspaces():
        offer(Path(entry))

    markers = (".git", "package.json", "pyproject.toml", "Cargo.toml", "go.mod",
               "requirements.txt", "README.md", ".qwen35")

    for root in _drive_roots():
        if len(found) >= limit:
            break
        try:
            level1 = [d for d in root.iterdir() if d.is_dir()]
        except (OSError, PermissionError):
            continue
        for d1 in level1:
            if d1.name in {"$RECYCLE.BIN", "System Volume Information",
                           "Windows", "Program Files", "Program Files (x86)",
                           "ProgramData", "Users", "Recovery", "PerfLogs"}:
                continue
            try:
                if validate_workspace(d1) and any((d1 / m).exists() for m in markers):
                    offer(d1)
            except WorkspaceError:
                pass
            if max_depth >= 2:
                try:
                    level2 = [d for d in d1.iterdir() if d.is_dir()]
                except (OSError, PermissionError):
                    continue
                for d2 in level2:
                    if len(found) >= limit:
                        break
                    try:
                        if any((d2 / m).exists() for m in markers):
                            validate_workspace(d2)
                            offer(d2)
                    except WorkspaceError:
                        continue
        # home directory projects
    try:
        home = Path.home()
        for sub in ("Documents", "Projects", "work", "src", "code", "dev"):
            d = home / sub
            if d.is_dir():
                try:
                    validate_workspace(d)
                    for child in d.iterdir():
                        if child.is_dir() and any((child / m).exists() for m in markers):
                            try:
                                validate_workspace(child)
                                offer(child)
                            except WorkspaceError:
                                continue
                except (WorkspaceError, OSError):
                    continue
    except (RuntimeError, OSError):
        pass

    return found[:limit]
