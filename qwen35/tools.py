"""
The tool set.

Six tools, deliberately few and deliberately dissimilar in name -- SWE-agent's
ACI ablations showed interface design alone moved solve rate 11% -> 18%, and
Hammer showed that small models are misled by *similar* tool names and must be
steered by descriptions instead.

The important mechanism in this file is :func:`validate_syntax`, used by both
``write_file`` and ``edit_file``.  SWE-agent measured edit success at 90.5% on a
first attempt but only 57.2% after one prior failure, with cascading failed edits
accounting for 23.4% of unsolved runs.  For a 0.8 B model that cliff is the
dominant failure mode, so a write that breaks the file is **reverted** and the
model is told exactly what broke -- the invariant is enforced by the harness
rather than trusted to the model.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import Permissions, ModelProfile
from .permissions import CommandScreener
from .prompt import ToolSpec
from .workspace import WorkspaceError, WorkspaceGuard

__all__ = ["ToolResult", "ToolContext", "TOOLS", "build_specs", "call_tool",
           "validate_syntax"]

# Tuned, per SWE-agent's viewer ablation: a 100-line window scored 18.0% while
# full-file context scored 12.7%.  More information is not better information.
DEFAULT_READ_LINES = 100
MAX_DIR_ENTRIES = 50          # SWE-agent capped listings at 50
MAX_OBSERVATION_CHARS = 6000

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
                             ".tif", ".tiff"})

# When a program is missing, name the tool that replaces it.  Without this a
# small model retries `cat` until the step budget runs out.
_ALIAS_HINTS: dict[str, str] = {
    "cat": "use read_file instead",
    "type": "use read_file instead",
    "bat": "use read_file instead",
    "head": "use read_file with start_line and max_lines",
    "tail": "use read_file with start_line and max_lines",
    "less": "use read_file instead",
    "more": "use read_file instead",
    "ls": "use list_dir instead",
    "dir": "use list_dir instead",
    "tree": "use list_dir on subdirectories",
    "grep": "use read_file, or run findstr or rg if installed",
    "egrep": "use read_file instead",
    "findstr": "use read_file instead",
    "sed": "use edit_file instead",
    "awk": "use edit_file instead",
    "python3": "use `python` on Windows",
    "py3": "use `python` instead",
    "touch": "use write_file, which creates the file",
    "mkdir": "use write_file, which creates parent directories",
    "rm": "deleting files is not permitted",
    "del": "deleting files is not permitted",
    "mv": "moving files is not permitted",
    "cp": "use read_file then write_file",
    "copy": "use read_file then write_file",
}


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class ToolResult:
    ok: bool
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    # an image to attach in a following user turn (multimodal path)
    image_b64: str = ""
    image_mime: str = ""

    def for_model(self, limit: int = MAX_OBSERVATION_CHARS) -> str:
        s = self.text or ""
        if len(s) > limit:
            head = s[: limit - 200]
            s = (
                head
                + f"\n\n... [truncated: {len(self.text)} chars total, "
                  f"{len(self.text) - limit + 200} not shown. Narrow your next "
                  "read with start_line/max_lines.]"
            )
        return s


# --------------------------------------------------------------------------
# Syntax gate (validate and revert)
# --------------------------------------------------------------------------
def validate_syntax(path: Path, content: str) -> str | None:
    """Return an error message when ``content`` is not valid for ``path``.

    Only formats we can check *cheaply and correctly in-process* are validated.
    A false positive here would revert a good edit, which is worse than skipping
    the check, so anything uncertain returns ``None``.
    """
    # CPython's tokenizer accepts a leading UTF-8 BOM in a source file, but
    # ``ast.parse`` on a *string* rejects it with "invalid non-printable
    # character U+FEFF".  Validating the raw string made every edit to a BOM'd
    # file impossible -- the gate would refuse a change that merely preserved a
    # pre-existing, perfectly importable condition.  Stripping the BOM makes the
    # check agree with what ``import`` actually does.
    probe = content[1:] if content.startswith("\ufeff") else content

    suffix = path.suffix.lower()

    if suffix == ".py":
        try:
            ast.parse(probe)
        except SyntaxError as exc:
            line = exc.lineno or 0
            return (f"{path.name}: Python syntax error on line {line}: "
                    f"{exc.msg}")
        return None

    if suffix == ".json":
        try:
            json.loads(probe)
        except json.JSONDecodeError as exc:
            return (f"{path.name}: invalid JSON at line {exc.lineno} col "
                    f"{exc.colno}: {exc.msg}")
        return None

    if suffix == ".toml":
        try:
            import tomllib

            tomllib.loads(probe)
        except ImportError:
            return None
        except Exception as exc:  # tomllib raises TOMLDecodeError
            return f"{path.name}: invalid TOML: {exc}"
        return None

    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            yaml.safe_load(probe)
        except ImportError:
            return None
        except Exception as exc:
            return f"{path.name}: invalid YAML: {exc}"
        return None

    if suffix in (".csv",):
        # a stray unclosed quote is the common corruption
        if probe.count('"') % 2:
            return f"{path.name}: unbalanced quote characters"
        return None

    return None


# --------------------------------------------------------------------------
# Backups: make a destructive overwrite recoverable
# --------------------------------------------------------------------------
BACKUP_KEEP = 5


def _backup(ctx: ToolContext, rp, previous: bytes | None) -> str:
    """Stash the previous contents before an overwrite.

    A 0.8 B model does sometimes emit truncated file content, which would
    silently destroy work.  Refusing the write would risk a deadlock (the model
    may have no other way to proceed); keeping a copy makes the damage
    recoverable instead, and costs nothing on the happy path.
    """
    if not previous:
        return ""
    try:
        directory = ctx.guard.root / ".qwen35" / "backups"
        directory.mkdir(parents=True, exist_ok=True)
        safe = rp.rel.replace("/", "__").replace("\\", "__")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = directory / f"{safe}.{stamp}.bak"
        n = 1
        while target.exists():
            target = directory / f"{safe}.{stamp}.{n}.bak"
            n += 1
        target.write_bytes(previous)
        existing = sorted(directory.glob(f"{safe}.*.bak"),
                          key=lambda p: p.stat().st_mtime)
        for old in existing[:-BACKUP_KEEP]:
            try:
                old.unlink()
            except OSError:
                pass
        return ctx.guard.rel(target)
    except OSError:
        return ""


# --------------------------------------------------------------------------
# Context handed to every tool
# --------------------------------------------------------------------------
@dataclass
class ToolContext:
    guard: WorkspaceGuard
    perms: Permissions
    profile: ModelProfile
    screener: CommandScreener
    allow_vision: bool = True
    image_max_edge: int = 1024
    image_max_tokens: int = 1280
    last_read: dict[str, tuple[int, int]] = field(default_factory=dict)
    edits_applied: int = 0
    reverts: int = 0
    commands_run: int = 0
    #: original file -> converted Markdown, keyed by workspace-relative path and
    #: by bare filename.  Used to *redirect* a model that reaches for a binary
    #: original instead of the text the harness already extracted for it.  This
    #: was the single most common failure in the evaluation suite: eight wasted
    #: steps trying to open ``scores.xlsx`` while its contents sat in the prompt.
    attachment_map: dict[str, str] = field(default_factory=dict)

    @property
    def sandbox_home(self) -> Path:
        return self.screener.sandbox_home

    def converted_for(self, rel: str) -> str:
        """The converted-text path for an original file, if there is one."""
        return (self.attachment_map.get(rel)
                or self.attachment_map.get(Path(rel).name)
                or "")

    def converted_paths(self) -> list[str]:
        """Every converted-text path, de-duplicated and ordered."""
        seen: dict[str, None] = {}
        for value in self.attachment_map.values():
            seen.setdefault(value, None)
        return list(seen)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


def tool_list_dir(ctx: ToolContext, path: str = ".", **_: Any) -> ToolResult:
    try:
        rp = ctx.guard.resolve(path or ".", must_exist=True)
    except WorkspaceError as exc:
        return ToolResult(False, f"error: {exc}")

    if not rp.abs.is_dir():
        return ToolResult(False, f"error: {rp.rel} is a file, not a directory")

    try:
        entries = sorted(rp.abs.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except (OSError, PermissionError) as exc:
        return ToolResult(False, f"error: cannot list {rp.rel}: {exc}")

    dirs, files = [], []
    for e in entries:
        name = e.name
        if name in (".git", "node_modules", "__pycache__", ".venv", "venv",
                    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".qwen35"):
            dirs.append(f"{name}/  (skipped: bulky)")
            continue
        try:
            if e.is_dir():
                dirs.append(f"{name}/")
            else:
                # Point at the tool that handles it.  Without this hint the model
                # tries to run `view_image` as a shell command instead of calling
                # the tool it already has.
                hint = ""
                if ctx.allow_vision and e.suffix.lower() in _IMAGE_SUFFIXES:
                    hint = "  <- call the view_image tool to look at this"
                else:
                    converted = ctx.converted_for(f"{rp.rel}/{name}".lstrip("./"))
                    if converted:
                        # Steer the model to the text rather than the binary.
                        hint = f"  <- already converted to text: {converted}"
                files.append(f"{name}  ({_fmt_size(e.stat().st_size)}){hint}")
        except OSError:
            continue

    shown_dirs = dirs[:MAX_DIR_ENTRIES]
    remaining = MAX_DIR_ENTRIES - len(shown_dirs)
    shown_files = files[:max(0, remaining)]
    total = len(dirs) + len(files)

    lines = [f"{rp.rel}/ contains {total} entries:"]
    lines += [f"  {d}" for d in shown_dirs]
    lines += [f"  {f}" for f in shown_files]
    if total > len(shown_dirs) + len(shown_files):
        lines.append(
            f"  ({total - len(shown_dirs) - len(shown_files)} more not shown -- "
            "list a subdirectory instead of listing again)"
        )
    if total == 0:
        lines.append("  (empty directory)")
    return ToolResult(True, "\n".join(lines), meta={"count": total})


def tool_read_file(ctx: ToolContext, path: str = "", start_line: Any = 1,
                   max_lines: Any = DEFAULT_READ_LINES, **_: Any) -> ToolResult:
    if not path:
        return ToolResult(False, "error: `path` is required")
    try:
        rp = ctx.guard.resolve(path, must_exist=True)
    except WorkspaceError as exc:
        return ToolResult(False, f"error: {exc}")

    if rp.abs.is_dir():
        # A model that has been told "the attachment is at .qwen35/attachments/
        # NN-x.md" and then asks to read "." has lost the path.  Saying only
        # "use list_dir" sends it to a listing that *skips* .qwen35, so it is a
        # dead end -- exactly how one real run burned five steps and returned
        # nothing.  Point at the attachments instead.
        suggestions = sorted(set(ctx.converted_paths()))
        if suggestions:
            hint = ("\nThe attached files were converted to text; read one of "
                    "these instead:\n"
                    + "\n".join(f'  read_file(path="{p}", start_line=1, '
                                f'max_lines=100)' for p in suggestions[:4]))
            return ToolResult(
                False,
                f"error: {rp.rel} is a directory, not a file.{hint}",
            )
        return ToolResult(False, f"error: {rp.rel} is a directory; use list_dir")

    try:
        size = rp.abs.stat().st_size
    except OSError as exc:
        return ToolResult(False, f"error: cannot stat {rp.rel}: {exc}")

    if size > ctx.perms.max_read_bytes:
        return ToolResult(
            False,
            f"error: {rp.rel} is {_fmt_size(size)}, over the "
            f"{_fmt_size(ctx.perms.max_read_bytes)} read limit",
        )

    try:
        raw = rp.abs.read_bytes()
    except (OSError, PermissionError) as exc:
        return ToolResult(False, f"error: cannot read {rp.rel}: {exc}")

    if b"\x00" in raw[:4096]:
        converted = ctx.converted_for(rp.rel)
        if converted:
            return ToolResult(
                False,
                f"error: {rp.rel} is a binary file ({_fmt_size(size)}) and cannot "
                f"be read directly. You do not need to: its contents have already "
                f"been extracted for you as text at {converted}. Use "
                f'read_file(path="{converted}").',
            )
        return ToolResult(
            False,
            f"error: {rp.rel} looks binary ({_fmt_size(size)}). "
            "Use run_command for binary files, or view_image for images.",
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        note = " (note: file is not valid UTF-8; undecodable bytes replaced)"
    else:
        note = ""

    try:
        start = max(1, int(start_line))
    except (TypeError, ValueError):
        start = 1
    try:
        want = int(max_lines)
    except (TypeError, ValueError):
        want = DEFAULT_READ_LINES
    want = max(1, min(want, 1000))

    all_lines = text.splitlines()
    total = len(all_lines)
    if start > total and total > 0:
        return ToolResult(
            False,
            f"error: start_line {start} is past the end of {rp.rel} "
            f"({total} lines)",
        )

    chunk = all_lines[start - 1: start - 1 + want]
    width = len(str(start + len(chunk)))
    body = "\n".join(
        f"{start + i:>{width}} | {ln}" for i, ln in enumerate(chunk)
    )
    end = start + len(chunk) - 1
    header = f"{rp.rel} lines {start}-{end} of {total}{note}"
    if end < total:
        header += f"  (call read_file with start_line={end + 1} for more)"

    ctx.last_read[rp.rel] = (start, end)
    return ToolResult(True, f"{header}\n{body}",
                      meta={"lines": total, "start": start, "end": end})


def _write_bytes(ctx: ToolContext, rp, content: str) -> ToolResult:
    data = content.encode("utf-8")
    if len(data) > ctx.perms.max_write_bytes:
        return ToolResult(
            False,
            f"error: content is {_fmt_size(len(data))}, over the "
            f"{_fmt_size(ctx.perms.max_write_bytes)} write limit",
        )
    try:
        rp.abs.parent.mkdir(parents=True, exist_ok=True)
        rp.abs.write_bytes(data)
    except (OSError, PermissionError) as exc:
        return ToolResult(False, f"error: cannot write {rp.rel}: {exc}")
    return ToolResult(True, f"wrote {rp.rel} ({_fmt_size(len(data))}, "
                            f"{len(content.splitlines())} lines)")


def tool_write_file(ctx: ToolContext, path: str = "", content: str = "",
                    **_: Any) -> ToolResult:
    if not path:
        return ToolResult(False, "error: `path` is required")
    if not ctx.perms.write:
        return ToolResult(False, "error: writing is disabled by the active "
                                 "permission profile")
    try:
        rp = ctx.guard.resolve(path)
    except WorkspaceError as exc:
        return ToolResult(False, f"error: {exc}")

    if rp.abs.is_dir():
        return ToolResult(False, f"error: {rp.rel} is a directory")

    if content is None:
        content = ""
    if not isinstance(content, str):
        content = json.dumps(content, indent=2, ensure_ascii=False)

    problem = validate_syntax(rp.abs, content)
    if problem:
        ctx.reverts += 1
        return ToolResult(
            False,
            f"error: refused to write {rp.rel} -- {problem}\n"
            "The file was NOT changed. Fix the content and try again.",
            meta={"reverted": True},
        )

    existed = rp.abs.exists()
    previous: bytes | None = None
    if existed:
        try:
            previous = rp.abs.read_bytes()
        except OSError:
            previous = None

    backup = _backup(ctx, rp, previous)

    result = _write_bytes(ctx, rp, content)
    if not result.ok:
        return result

    ctx.edits_applied += 1
    result.text = ("created " if not existed else "overwrote ") + result.text

    if existed and previous:
        new_size = len(content.encode("utf-8"))
        # A dramatic shrink is the signature of truncated content, which is a
        # real failure mode for this model.  Not blocked (that risks a deadlock),
        # but said plainly and backed by the copy above.
        if len(previous) >= 300 and new_size * 4 < len(previous):
            result.text += (
                f"\nWARNING: {rp.rel} was {_fmt_size(len(previous))} and is now "
                f"{_fmt_size(new_size)}. If you meant a small change, you "
                "truncated the content -- use edit_file for targeted changes."
            )
        if backup:
            result.text += f"\n(previous version saved to {backup})"
    return result


def tool_edit_file(ctx: ToolContext, path: str = "", find: str = "",
                   replace: str = "", **_: Any) -> ToolResult:
    if not path:
        return ToolResult(False, "error: `path` is required")
    if not find:
        return ToolResult(
            False,
            "error: `find` is required -- it is the exact text to replace, "
            "copied from read_file output (without the line-number prefix)",
        )
    if not ctx.perms.edit:
        return ToolResult(False, "error: editing is disabled by the active "
                                 "permission profile")
    try:
        rp = ctx.guard.resolve(path, must_exist=True)
    except WorkspaceError as exc:
        return ToolResult(False, f"error: {exc}")

    try:
        original = rp.abs.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return ToolResult(False, f"error: cannot read {rp.rel}: {exc}")

    occurrences = original.count(find)
    if occurrences == 0:
        hint = ""
        stripped = find.strip()
        if stripped and stripped != find and original.count(stripped):
            hint = ("\n(Found it once the surrounding whitespace is trimmed -- "
                    "`find` must match the file exactly, including indentation.)")
        return ToolResult(
            False,
            f"error: `find` does not appear in {rp.rel}.{hint}\n"
            "Read the file again and copy the exact text you want to change.",
        )
    if occurrences > 1:
        return ToolResult(
            False,
            f"error: `find` appears {occurrences} times in {rp.rel}; the edit "
            "is ambiguous. Include more surrounding lines so it matches exactly "
            "once.",
        )

    updated = original.replace(find, replace, 1)
    problem = validate_syntax(rp.abs, updated)
    if problem:
        ctx.reverts += 1
        return ToolResult(
            False,
            f"error: that edit would break {rp.rel} -- {problem}\n"
            "The edit was REVERTED and the file is unchanged. "
            "Adjust `replace` and try again.",
            meta={"reverted": True},
        )

    backup = _backup(ctx, rp, original.encode("utf-8"))
    try:
        rp.abs.write_text(updated, encoding="utf-8")
    except (OSError, PermissionError) as exc:
        return ToolResult(False, f"error: cannot write {rp.rel}: {exc}")

    ctx.edits_applied += 1
    text = (f"edited {rp.rel}: replaced 1 occurrence "
            f"(1 line -> {len(replace.splitlines())} lines)")
    if backup:
        text += f"\n(previous version saved to {backup})"
    return ToolResult(True, text)


def tool_run_command(ctx: ToolContext, command: str = "", **_: Any) -> ToolResult:
    if not command:
        return ToolResult(False, "error: `command` is required")
    if ctx.commands_run >= ctx.perms.max_commands_per_run:
        return ToolResult(
            False,
            f"error: command budget exhausted "
            f"({ctx.perms.max_commands_per_run} per run)",
        )

    screening = ctx.screener.screen(command)
    if not screening.allowed:
        return ToolResult(
            False,
            f"error: command refused -- {screening.reason}",
            meta={"refused": True},
        )

    program = screening.argv[0]
    resolved = shutil.which(program)
    if resolved is None:
        # Small models reach for utilities their training data was full of
        # (`cat`, `ls`, `grep`) that simply do not exist on Windows.  Pointing at
        # the tool that does the job stops the retry loop.
        hint = _ALIAS_HINTS.get(program.lower(), "")
        extra = f" {hint[0].upper()}{hint[1:]}." if hint else ""
        return ToolResult(
            False,
            f"error: {program!r} is not installed on this machine.{extra}",
        )

    env = ctx.screener.scrub_env_for_child()
    ctx.sandbox_home.mkdir(parents=True, exist_ok=True)
    (ctx.sandbox_home / "tmp").mkdir(parents=True, exist_ok=True)

    argv = [resolved, *screening.argv[1:]]
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            # `cd subdir && cmd` is honoured by screening the remainder and
            # running it here, so the confinement check has already happened
            cwd=str(screening.cwd or ctx.guard.root),
            env=env,
            shell=False,                     # never a shell
            capture_output=True,
            timeout=ctx.perms.max_command_seconds,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        ctx.commands_run += 1
        return ToolResult(
            False,
            f"error: `{command}` exceeded the "
            f"{ctx.perms.max_command_seconds}s limit and was killed",
            meta={"timeout": True},
        )
    except (OSError, ValueError) as exc:
        return ToolResult(False, f"error: could not run `{command}`: {exc}")

    ctx.commands_run += 1
    elapsed = time.monotonic() - started

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    status = "ok" if proc.returncode == 0 else f"exit code {proc.returncode}"

    # silence is ambiguous and makes weak models re-run the same command
    if not out and not err:
        body = (f"`{command}` ran successfully and produced no output "
                f"({status}, {elapsed:.1f}s).")
        if proc.returncode != 0:
            body = (f"`{command}` produced no output but failed with "
                    f"{status} ({elapsed:.1f}s).")
        return ToolResult(proc.returncode == 0, body,
                          meta={"exit": proc.returncode})

    parts = [f"$ {command}", f"[{status}, {elapsed:.1f}s]"]
    if out:
        parts.append(out)
    if err:
        parts.append(f"[stderr]\n{err}")
    return ToolResult(
        proc.returncode == 0,
        "\n".join(parts),
        meta={"exit": proc.returncode, "truncated": len(out) + len(err) > 20000},
    )


def prepare_image_b64(ctx: ToolContext, rel: str) -> tuple[str, str, int, str]:
    """Load, downscale and base64 an image from the workspace.

    Returns ``(base64, mime, visual_tokens, description)``.  Shared by the
    ``view_image`` tool and by the loop, which inlines a single attached picture
    straight into the user message so the model does not have to reproduce a
    hash-bearing path to look at it.
    """
    from PIL import Image

    rp = ctx.guard.resolve(rel, must_exist=True)
    with Image.open(rp.abs) as im:
        im.load()
        width, height = im.size
        fmt = (im.format or "PNG").upper()
        rgb = im.convert("RGB")
        # visual tokens cost about W*H/1024, so an unbounded image would silently
        # eat the context
        edge = max(ctx.image_max_edge, 64)
        if max(rgb.size) > edge:
            ratio = edge / float(max(rgb.size))
            rgb = rgb.resize((max(1, int(rgb.width * ratio)),
                              max(1, int(rgb.height * ratio))), Image.LANCZOS)
        tokens = ctx.profile.approx_visual_tokens(*rgb.size)
        guard = 0
        while tokens > ctx.image_max_tokens and guard < 8:
            rgb = rgb.resize((max(1, rgb.width // 2), max(1, rgb.height // 2)),
                             Image.LANCZOS)
            tokens = ctx.profile.approx_visual_tokens(*rgb.size)
            guard += 1

        buf = io.BytesIO()
        rgb.save(buf, format="PNG", optimize=True)
        payload = buf.getvalue()
        out_w, out_h = rgb.size

    if len(payload) > 12_000_000:
        raise ValueError(f"image {rel} is too large to send")
    description = (f"{width}x{height} {fmt}, sent at {out_w}x{out_h}, "
                   f"~{tokens} visual tokens")
    return base64.b64encode(payload).decode("ascii"), "image/png", tokens, description


def tool_view_image(ctx: ToolContext, path: str = "", question: str = "",
                    **_: Any) -> ToolResult:
    if not ctx.perms.allow_vision or not ctx.allow_vision:
        return ToolResult(False, "error: image viewing is disabled")
    if not path:
        return ToolResult(False, "error: `path` is required")
    if not ctx.profile.supports_vision:
        return ToolResult(
            False,
            f"error: {ctx.profile.name} has no vision encoder configured",
        )

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return ToolResult(
            False,
            "error: Pillow is not installed, so images cannot be prepared "
            "(pip install pillow)",
        )
    try:
        encoded, mime, tokens, description = prepare_image_b64(ctx, path)
    except WorkspaceError as exc:
        return ToolResult(False, f"error: {exc}")
    except Exception as exc:  # PIL raises a wide variety
        return ToolResult(False, f"error: cannot read image {path}: {exc}")

    rp = ctx.guard.resolve(path)
    note = f"attached image: {rp.rel} ({description})"
    if question:
        note += f"\nquestion: {question}"

    return ToolResult(
        True,
        note + "\nThe image itself follows in the next message.",
        image_b64=encoded,
        image_mime=mime,
        meta={"visual_tokens": tokens, "path": rp.rel},
    )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolDef:
    spec: ToolSpec
    fn: Callable[..., ToolResult]


TOOLS: dict[str, ToolDef] = {
    "list_dir": ToolDef(
        ToolSpec(
            name="list_dir",
            description=(
                "List the files and directories in a folder of the workspace. "
                "Use this to find out what exists before reading anything. "
                "Pass a relative path; omit it to list the workspace root."
            ),
            params=("path",),
            required=(),
            defaults={"path": "."},
        ),
        tool_list_dir,
    ),
    "read_file": ToolDef(
        ToolSpec(
            name="read_file",
            description=(
                "Read a text file from the workspace, with line numbers. "
                "Returns a window of at most `max_lines` lines starting at "
                "`start_line`, so you can page through a large file. Always "
                "read a file before editing it."
            ),
            params=("path", "start_line", "max_lines"),
            required=("path",),
            types={"path": "string", "start_line": "integer", "max_lines": "integer"},
            defaults={"start_line": 1, "max_lines": DEFAULT_READ_LINES},
        ),
        tool_read_file,
    ),
    "write_file": ToolDef(
        ToolSpec(
            name="write_file",
            description=(
                "Create a new file, or replace an existing one completely. "
                "Provide the whole file content. For a small change to an "
                "existing file prefer edit_file -- it is safer and cheaper."
            ),
            params=("path", "content"),
            required=("path", "content"),
        ),
        tool_write_file,
    ),
    "edit_file": ToolDef(
        ToolSpec(
            name="edit_file",
            description=(
                "Replace one exact piece of text in a file. `find` must match "
                "the file exactly -- including indentation -- and must appear "
                "only once, so include a few surrounding lines when the text is "
                "ambiguous. Read the file first and copy the text verbatim."
            ),
            params=("path", "find", "replace"),
            required=("path", "find", "replace"),
        ),
        tool_edit_file,
    ),
    "run_command": ToolDef(
        ToolSpec(
            name="run_command",
            description=(
                "Run one command-line program and see its output. Use it to run "
                "tests and build, e.g. `python test_calc.py`. Commands already "
                "run in the workspace root, so pass paths like "
                "`subdir/script.py` rather than changing directory. There is no "
                "shell: no pipes, no `&&`, no `;`, no redirection and no "
                "variables. `cd subdir && <command>` is the only allowed "
                "combination. Paths must stay inside the workspace."
            ),
            params=("command",),
            required=("command",),
        ),
        tool_run_command,
    ),
    "view_image": ToolDef(
        ToolSpec(
            name="view_image",
            description=(
                "Look at an image file in the workspace (png, jpg, gif, bmp, "
                "webp). You are multimodal, so you actually see the picture. "
                "Use it for screenshots, diagrams, charts and photos."
            ),
            params=("path", "question"),
            required=("path",),
        ),
        tool_view_image,
    ),
}


def build_specs(names: list[str] | None = None,
                perms: Permissions | None = None) -> list[ToolSpec]:
    """Return the tool specs the model should be offered.

    ``names=None`` means "all of them"; an explicit empty list means **none**,
    which is how a conversational run is forced to answer without calling a tool.
    (The earlier ``names or list(TOOLS)`` treated ``[]`` as "unspecified", so the
    empty case silently offered every tool.)

    Tools the active profile cannot execute are *withheld from the schema* rather
    than advertised and then refused -- a small model that sees a tool it cannot
    use will keep calling it.
    """
    chosen = list(TOOLS) if names is None else list(names)
    specs: list[ToolSpec] = []
    for name in chosen:
        td = TOOLS.get(name)
        if td is None:
            continue
        if perms is not None:
            if name in ("write_file", "edit_file") and not (perms.write or perms.edit):
                continue
            if name == "run_command" and not perms.run_command:
                continue
            if name == "view_image" and not perms.allow_vision:
                continue
        specs.append(td.spec)
    return specs


def call_tool(name: str, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    td = TOOLS.get(name)
    if td is None:
        legal = ", ".join(sorted(TOOLS))
        return ToolResult(
            False,
            f"error: there is no tool called {name!r}. "
            f"The only tools are: {legal}.",
        )
    # drop parameters the tool does not declare -- the grammar allows repeats
    # and a weak model sometimes invents extra keys
    allowed = set(td.spec.params)
    clean = {k: v for k, v in (args or {}).items() if k in allowed}
    try:
        return td.fn(ctx, **clean)
    except TypeError as exc:
        return ToolResult(False, f"error: wrong arguments for {name}: {exc}")
    except Exception as exc:  # a tool must never kill the run
        return ToolResult(False, f"error: {name} failed: {type(exc).__name__}: {exc}")
