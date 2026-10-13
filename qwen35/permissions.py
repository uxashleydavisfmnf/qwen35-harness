"""
Permission enforcement: what the model is allowed to do, and how it is contained.

Design rules
------------
* **No shell, ever.**  Commands are parsed into an argv vector and executed with
  ``shell=False``.  Metacharacters are therefore inert by construction; we still
  reject them as defence in depth.
* **Program allowlist, not denylist-first.**  Only programs that are useless for
  privilege escalation and useful for real work are permitted.  ``cmd``,
  ``powershell``, ``reg``, ``schtasks``, ``diskpart`` and friends are refused
  outright, so the model cannot bootstrap its way out.
* **Credentials are stripped.**  The child environment is rebuilt from a small
  allowlist; anything that looks like a key, token or secret never reaches it.
* **``HOME`` is redirected into the workspace.**  ``~`` inside a child process
  resolves to a workspace-local scratch directory, so the most common accidental
  writes outside the sandbox (caches, temp files, ``~/.config``) land inside it.
* **Honest boundary.**  Because the allowlist includes interpreters
  (``python``, ``node``), ``run_command`` still grants the model code execution
  with your user's privileges.  That is inherent to a coding harness; the
  ``readonly`` profile removes it entirely.  This is documented, not hidden.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import Permissions
from .workspace import WorkspaceError, WorkspaceGuard

__all__ = ["Screening", "CommandScreener", "scrub_env"]


class CommandRefused(Exception):
    """Raised when a command must not run."""


# --------------------------------------------------------------------------
# Program policy
# --------------------------------------------------------------------------
# Always refused, regardless of profile.  These exist to change the security
# posture of the machine, so a small model must never reach them.
HARD_DENY: frozenset[str] = frozenset({
    # shells / interpreters that re-open the metacharacter surface
    "cmd", "command", "powershell", "pwsh", "bash", "sh", "zsh", "csh", "tcsh",
    "ksh", "dash", "fish", "cscript", "wscript", "mshta", "rundll32",
    "regsvr32", "wmic", "bitsadmin", "msiexec",
    # privilege / policy.  Elevation is refused outright: this harness runs as the
    # user and must never ask for more, so there is no legitimate reason for a
    # model to invoke an escalation helper.
    "runas", "sudo", "gsudo", "doas", "elevate", "runassu", "su",
    "reg", "regedit", "takeown", "icacls", "cacls", "attrib", "secedit",
    "gpupdate", "gpresult", "auditpol", "whoami", "net", "netsh", "sc", "sc.exe",
    "schtasks", "at", "taskkill", "tasklist", "tskill", "pskill",
    # disk / system integrity
    "format", "diskpart", "chkdsk", "fsutil", "vssadmin", "mountvol", "subst",
    "sfc", "bcdedit", "bootrec", "shutdown", "restart-computer", "powercfg",
    "mbr2gpt", "defrag", "sfc.exe", "convert",
    # credential / crypto tooling
    "certutil", "cipher", "vault", "cmdkey", "runexe", "klist",
    # remote / lateral
    "ssh", "scp", "sftp", "telnet", "ftp", "rdesktop", "mstsc", "psexec",
    "winrs", "curl.exe", "certreq",
    # destructive copy/move that can leave the workspace
    "robocopy", "xcopy", "move", "mv", "del", "erase", "rmdir", "rd", "rm",
    "shred", "sdelete", "rm.exe",
})

# Safe, work-relevant programs.  Anything not listed here is refused unless a
# profile explicitly widens the policy.
ALLOW: frozenset[str] = frozenset({
    # language runtimes / build tools
    "python", "python3", "py", "pip", "pip3", "node", "nodejs", "npm", "npx",
    "pnpm", "yarn", "deno", "bun", "cargo", "rustc", "go", "gofmt", "java",
    "javac", "gcc", "g++", "clang", "clang++", "make", "cmake", "ninja",
    "dotnet", "ruby", "perl", "php", "lua", "tsc", "esbuild", "vite",
    # test runners
    "pytest", "unittest", "tox", "nox", "jest", "vitest", "mocha", "nose2",
    # vcs (read-mostly; `git push`/`remote` are screened separately)
    "git",
    # inspection / text processing
    "ls", "dir", "cat", "type", "head", "tail", "less", "more", "wc", "sort",
    "uniq", "cut", "tr", "sed", "awk", "grep", "findstr", "rg", "find", "fd",
    "diff", "cmp", "comm", "patch", "tree", "stat", "file", "du", "df",
    "basename", "dirname", "realpath", "readlink", "echo", "printf", "pwd",
    "cd", "true", "false", "test", "sleep", "date", "time", "tee", "xargs",
    "jq", "yq", "column", "nl", "seq", "expr", "bc", "md5sum", "sha1sum",
    "sha256sum", "certutil-hash", "open", "start",
    # file management that stays inside the workspace
    "mkdir", "md", "touch", "cp", "copy", "ln", "mklink", "chmod", "tar",
    "zip", "unzip", "gzip", "gunzip", "7z",
    # environment discovery
    "which", "where", "env", "printenv", "set", "ver", "uname", "hostname",
    "nproc", "node-version", "py-version",
})

# Programs that are fine read-only but whose mutating subcommands are gated.
NETWORK_PROGRAMS = frozenset({"curl", "wget", "invoke-webrequest", "iwr",
                              "ping", "tracert", "nslookup", "dig"})
INSTALL_SUBCOMMANDS = frozenset({"install", "i", "add", "upgrade", "update",
                                 "publish", "deploy", "ci", "link"})

#: Programs that put code on disk outside the workspace unless they are pointed
#: somewhere else.  ``scrub_env`` points all of them at ``.qwen35/``.
INSTALLERS = frozenset({"pip", "pip3", "npm", "pnpm", "yarn", "cargo", "go",
                        "dotnet", "gem", "composer", "nuget", "poetry", "uv"})

#: Install flags that put files outside the workspace no matter what the
#: environment says.  These are the explicit opt-outs, so they are refused.
_INSTALL_ESCAPES = frozenset({
    "--user", "-g", "--global", "--system", "--break-system-packages",
    "--root", "--installroot", "--no-site-packages-warning",
})

#: Flags whose VALUE is a path that must stay inside the workspace.
_INSTALL_PATH_FLAGS = ("--target", "--prefix", "--location", "--cache-dir",
                       "--globalconfig", "--userconfig", "--root",
                       "--installroot", "--directory", "--path")

# git subcommands that reach the network or rewrite history
GIT_GATED = frozenset({"push", "remote", "fetch", "pull", "clone", "submodule",
                       "credential", "config", "filter-branch", "reset"})
GIT_DENIED = frozenset({"clean"})  # recursive delete

_METACHARS = (";", "|", "&", "`", "$", ">", "<", "\n", "\r", "\x00")
_VAR_EXPANSION = re.compile(r"%[A-Za-z_][A-Za-z0-9_]*%|\$\{?[A-Za-z_]")
_URL_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://")
_QUOTES = ("'", '"')

# `cd <dir> && <command>` -- the idiom every small model reaches for first.
# Rather than just refusing it, the harness honours it safely by using the
# directory as the child's working directory (still confined to the workspace)
# and screening only the real command.
_CD_PREFIX = re.compile(
    r"^\s*cd\s+(?P<dir>\"[^\"]+\"|'[^']+'|[^\s;&|]+)\s*(?:&&|;)\s*(?P<rest>.+)$",
    re.DOTALL,
)

# Programs that get a bespoke, teachable refusal instead of the generic
# "not on the allowlist" message.  (`cd` is handled inline, because its message
# needs to name the workspace root.)
_SPECIAL_REFUSALS: dict[str, str] = {
    "export": "`export` is a shell builtin; each command runs in a fresh process.",
    "source": "`.`/`source` is a shell builtin; each command runs in a fresh process.",
    "sudo": "privilege escalation is never permitted.",
    "su": "privilege escalation is never permitted.",
}


def _metachar_outside_quotes(raw: str) -> str | None:
    """Find a shell metacharacter that is *not* inside a quoted string.

    Scanning the raw string would reject legitimate commands like
    ``python -c "import x\\nprint(x)"``, where the newline is inside quotes and
    therefore inert -- the command is executed with ``shell=False``, so quoted
    metacharacters never reach a shell.
    """
    quote: str | None = None
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if quote:
            if ch == "\\" and quote == '"':
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in _QUOTES:
            quote = ch
            i += 1
            continue
        if ch in _METACHARS:
            return ch
        i += 1
    return None

# environment variables that must never reach a child process
_SECRET_NAME = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|SESSION|COOKIE|"
    r"PRIVATE|SIGNATURE|BEARER|APIKEY|ACCESS)",
    re.IGNORECASE,
)
_SECRET_PREFIX = ("AWS_", "OPENAI_", "ANTHROPIC_", "GEMINI_", "GOOGLE_", "AZURE_",
                  "HF_", "HUGGING", "GITHUB_", "GH_", "DSH_", "NPM_TOKEN",
                  "DOCKER_", "KUBE", "SLACK_", "STRIPE_", "TWILIO_", "SENDGRID_")

# the minimal environment a child genuinely needs
_ENV_ALLOW = (
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
    "TEMP", "TMP", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER", "OS", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    "PYTHONIOENCODING", "PYTHONUTF8", "PYTHONDONTWRITEBYTECODE",
    "TERM", "COLORTERM", "NO_COLOR", "FORCE_COLOR",
)


def _split_command(raw: str) -> list[str]:
    """Split a command line, **preserving Windows backslashes**.

    ``shlex.split(raw, posix=True)`` treats ``\\`` as an escape character.  On
    Windows the backslash is a path separator, so the split silently rewrote
    paths:

        cat C:\\Windows\\win.ini   ->  ['cat', 'C:Windowswin.ini']
        cat ..\\..\\secret.txt     ->  ['cat', '....secret.txt']
        pip install --target C:\\Windows\\Temp x
                                   ->  ['--target', 'C:WindowsTemp']

    The first is the dangerous one: an absolute path became a *drive-relative*
    one, so it no longer looked absolute and slipped straight through the path
    screen.  It also broke ordinary commands, because the mangled argv is what
    actually gets executed.

    POSIX splitting is kept everywhere else, but on Windows the non-POSIX split
    is used and the surrounding quotes are removed by hand -- non-POSIX mode
    leaves them attached to the token.
    """
    if os.name != "nt":
        return shlex.split(raw, posix=True)
    out: list[str] = []
    for part in shlex.split(raw, posix=False):
        if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'":
            out.append(part[1:-1])
        else:
            out.append(part)
    return out


def _basename(program: str) -> str:
    name = os.path.basename(program.strip().strip('"'))
    low = name.lower()
    for suffix in (".exe", ".cmd", ".bat", ".com", ".ps1", ".py", ".js", ".mjs"):
        if low.endswith(suffix):
            low = low[: -len(suffix)]
            break
    return low


def scrub_env(base: dict[str, str] | None = None,
              sandbox_home: Path | None = None) -> dict[str, str]:
    """Build a minimal child environment with secrets removed."""
    src = dict(os.environ if base is None else base)
    out: dict[str, str] = {}
    for k, v in src.items():
        if k.upper() in _ENV_ALLOW:
            out[k] = v
    if sandbox_home is not None:
        home = str(sandbox_home)
        # `.qwen35` in the workspace; sandbox_home is `<ws>/.qwen35/sandbox-home`
        state = sandbox_home.parent
        out["HOME"] = home
        out["USERPROFILE"] = home
        out["XDG_CONFIG_HOME"] = str(sandbox_home / ".config")
        out["XDG_CACHE_HOME"] = str(sandbox_home / ".cache")
        out["XDG_DATA_HOME"] = str(sandbox_home / ".local" / "share")
        out["MPLCONFIGDIR"] = str(sandbox_home / ".config" / "matplotlib")
        out["TEMP"] = str(sandbox_home / "tmp")
        out["TMP"] = str(sandbox_home / "tmp")

        # ---- package managers are pointed INSIDE the workspace ------------
        #
        # Redirecting the *cache* is not enough: a bare `pip install requests`
        # still installs into the interpreter's site-packages, outside the
        # workspace, and on a locked-down machine fails outright.  These variables
        # move the install itself, for every ecosystem at once, so the confinement
        # does not depend on the model remembering a flag.
        #
        # Everything lands under `<workspace>/.qwen35/`, written by the current
        # user with no elevation.  PYTHONPATH makes the result importable.
        packages = state / "packages"
        userbase = state / "userbase"
        out["PIP_CACHE_DIR"] = str(state / "cache" / "pip")
        out["PIP_TARGET"] = str(packages)
        out["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        out["PIP_NO_INPUT"] = "1"
        out["PIP_NO_WARN_SCRIPT_LOCATION"] = "1"
        out["PYTHONUSERBASE"] = str(userbase)
        # The operator's global user site-packages must not leak into the child,
        # or "installed in the workspace" would be a lie.
        out["PYTHONNOUSERSITE"] = "1"
        out["PIP_REQUIRE_VIRTUALENV"] = "0"   # we use --target, not a venv
        out["PYTHONPATH"] = os.pathsep.join([
            str(packages),
            str(userbase / "Lib" / "site-packages"),
            str(userbase / "lib" / "python" / "site-packages"),
        ])
        out["npm_config_cache"] = str(state / "cache" / "npm")
        out["npm_config_prefix"] = str(state / "node")
        out["npm_config_userconfig"] = str(state / "node" / "npmrc")
        out["npm_config_globalconfig"] = str(state / "node" / "etc" / "npmrc")
        out["NODE_PATH"] = str(state / "node" / "lib" / "node_modules")
        out["CARGO_HOME"] = str(state / "cargo")
        out["RUSTUP_HOME"] = str(state / "rustup")
        out["GOCACHE"] = str(state / "cache" / "go-build")
        out["GOPATH"] = str(state / "go")
        out["GOMODCACHE"] = str(state / "go" / "pkg" / "mod")
        out["DOTNET_CLI_HOME"] = str(state / "dotnet")
        out["DOTNET_TOOLS"] = str(state / "dotnet" / "tools")
        out["NUGET_PACKAGES"] = str(state / "nuget")
        out["GRADLE_USER_HOME"] = str(state / "gradle")
        out["MAVEN_OPTS"] = f"-Dmaven.repo.local={state / 'maven'}"

        for folder in (packages, userbase, state / "cache", state / "node",
                       state / "go", state / "cargo", state / "tmp"):
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass                          # a read-only workspace is not fatal

        # ``PYTHONPATH`` does **not** process ``.pth`` files; ``site.addsitedir``
        # does.  Several common packages rely on them (setuptools'
        # distutils-precedence, pywin32's shims, some namespace packages), so
        # they would silently half-work.  A ``sitecustomize`` module inside the
        # target directory is imported at interpreter startup and can call
        # ``addsitedir`` for us, which fixes the whole class at once.
        bootstrap = packages / "sitecustomize.py"
        if not bootstrap.exists():
            try:
                bootstrap.write_text(
                    "import pathlib, site, sys\n"
                    "# Written by qwen35: make workspace-installed packages\n"
                    "# importable, INCLUDING their .pth files.\n"
                    "_here = str(pathlib.Path(__file__).resolve().parent)\n"
                    "site.addsitedir(_here)\n"
                    "if _here not in sys.path:\n"
                    "    sys.path.insert(0, _here)\n",
                    encoding="utf-8")
            except OSError:
                pass
    out.setdefault("PYTHONIOENCODING", "utf-8")
    out.setdefault("PYTHONUTF8", "1")
    return out


@dataclass
class Screening:
    """Outcome of screening a command string."""

    allowed: bool
    argv: list[str] = field(default_factory=list)
    program: str = ""
    reason: str = ""
    gated_reason: str = ""
    cwd: Path | None = None


class CommandScreener:
    """Parses and authorises a single command line."""

    def __init__(self, guard: WorkspaceGuard, perms: Permissions):
        self.guard = guard
        self.perms = perms
        self.sandbox_home = guard.root / ".qwen35" / "sandbox-home"
        self._count = 0

    # -- child environment -------------------------------------------------
    def scrub_env_for_child(self) -> dict[str, str]:
        """Minimal environment for a child process, secrets removed.

        ``HOME``/``USERPROFILE``/``TEMP`` are redirected into the workspace so
        that ``~`` and the temp directory inside the child point at the sandbox
        rather than the operator's real profile.
        """
        return scrub_env(sandbox_home=self.sandbox_home)

    # -- public ------------------------------------------------------------
    def screen(self, command: str) -> Screening:
        if not isinstance(command, str) or not command.strip():
            return Screening(False, reason="empty command")

        raw = command.strip()

        if len(raw) > 4000:
            return Screening(False, reason="command is too long")

        if not self.perms.run_command:
            return Screening(
                False,
                reason="command execution is disabled by the active permission "
                       "profile",
            )

        if self._count >= self.perms.max_commands_per_run:
            return Screening(
                False,
                reason=f"command budget exhausted "
                       f"({self.perms.max_commands_per_run} per run)",
            )

        for expanded in ("~",):
            if raw.startswith(expanded):
                return Screening(
                    False,
                    reason="`~` is not expanded: commands run in the workspace, "
                           "so use a workspace-relative path.",
                )

        # --- `cd <dir> && <command>` ------------------------------------
        # Honoured rather than refused: the directory becomes the child's cwd,
        # which is validated to be inside the workspace, and only the real
        # command is screened.
        cwd: Path | None = None
        work = raw
        cd_match = _CD_PREFIX.match(raw)
        if cd_match:
            dir_arg = cd_match.group("dir").strip().strip('"').strip("'")
            try:
                resolved = self.guard.resolve(dir_arg, must_exist=True)
            except WorkspaceError as exc:
                return Screening(
                    False,
                    reason=(
                        f"`cd {dir_arg}` was refused ({exc}). Commands already "
                        f"run in the workspace root ({self.guard.root}), so you "
                        "do not need to change directory -- run the command "
                        "directly instead."
                    ),
                )
            if not resolved.abs.is_dir():
                return Screening(
                    False,
                    reason=f"`cd {dir_arg}` refused: it is a file, not a directory",
                )
            cwd = resolved.abs
            work = cd_match.group("rest").strip()
            if not work:
                return Screening(False, reason="`cd` with no following command")
            # chaining beyond one command is still refused
            extra = _metachar_outside_quotes(work)
            if extra in (";", "&", "|"):
                return Screening(
                    False,
                    reason=f"only one command may follow `cd`; found {extra!r}. "
                           "Run the commands one at a time.",
                )

        # --- metacharacters (only outside quotes; shell=False anyway) ----
        if not self.perms.allow_shell_metachars:
            found = _metachar_outside_quotes(work)
            if found is not None:
                pretty = {"\n": "\\n", "\r": "\\r", "\x00": "\\0"}.get(found, found)
                hint = ""
                if found in ("&", "|", ";"):
                    hint = (
                        " Commands run one at a time in the workspace root, so "
                        "chaining is unnecessary. To work in a subdirectory use "
                        "`cd subdir && <command>`; to run a script pass its path "
                        "(`python subdir/script.py`)."
                    )
                elif found in (">", "<"):
                    hint = (
                        " Rewrite the file with write_file/edit_file instead of "
                        "shell redirection."
                    )
                return Screening(
                    False,
                    reason=f"shell metacharacter {pretty!r} is not allowed.{hint}",
                )
            if _VAR_EXPANSION.search(work):
                return Screening(
                    False,
                    reason="variable expansion is not allowed in commands; write "
                           "the literal value instead.",
                )

        try:
            argv = _split_command(work)
        except ValueError as exc:
            return Screening(False, reason=f"could not parse command: {exc}")

        if not argv:
            return Screening(False, reason="empty command")

        program = _basename(argv[0])
        if not program:
            return Screening(False, reason="no program named in command")

        if program in HARD_DENY:
            return Screening(
                False,
                program=program,
                reason=(
                    f"{program!r} is permanently blocked: it can change machine "
                    "security state, and the harness only grants basic "
                    "permissions."
                ),
            )

        if program == "cd":
            return Screening(
                False,
                program=program,
                reason=(
                    "`cd` on its own does nothing here, and you do not need it: "
                    f"commands already run in the workspace root "
                    f"({self.guard.root}). To work in a subdirectory write "
                    "`cd subdir && <command>`, or simply pass the path: "
                    "`python subdir/script.py`."
                ),
            )

        if program in _SPECIAL_REFUSALS:
            return Screening(False, program=program,
                             reason=_SPECIAL_REFUSALS[program])

        gate = ""
        if program not in ALLOW:
            if program in NETWORK_PROGRAMS and self.perms.allow_network_commands:
                gate = "network"
            elif self.perms.allow_package_install and shutil.which(argv[0]):
                gate = "unlisted-but-permitted-by-profile"
            else:
                return Screening(
                    False,
                    program=program,
                    reason=(
                        f"{program!r} is not on the allowlist. Permitted programs "
                        "are language runtimes, build and test tools, git, and "
                        "read-only file/text utilities."
                    ),
                )

        if gate == "network" and not self.perms.allow_network_commands:
            return Screening(
                False, program=program,
                reason=f"{program!r} needs network permission, which this profile "
                       "withholds",
            )

        # ---- program-specific gates -------------------------------------
        sub = argv[1].lower() if len(argv) > 1 else ""

        if program == "git":
            if sub in GIT_DENIED:
                return Screening(False, program=program,
                                 reason="'git clean' deletes untracked files; refused")
            if sub in GIT_GATED and not self.perms.allow_network_commands:
                if sub == "config":
                    return Screening(
                        False, program=program,
                        reason="'git config' can alter credentials and hooks; refused")
                return Screening(
                    False, program=program,
                    reason=f"'git {sub}' reaches the network or rewrites history; "
                           "the active profile withholds network access",
                )
            if any(a in ("--upload-pack", "--receive-pack", "--exec") for a in argv):
                return Screening(False, program=program,
                                 reason="git option can execute an arbitrary program; refused")

        if program in ("pip", "pip3", "npm", "pnpm", "yarn", "cargo", "go", "dotnet"):
            if sub in INSTALL_SUBCOMMANDS and not self.perms.allow_package_install:
                return Screening(
                    False, program=program,
                    reason=(
                        f"'{program} {sub}' installs packages (network + arbitrary "
                        "build scripts); the active profile withholds that"
                    ),
                )

        # ---- keep installs inside the workspace --------------------------
        if program in INSTALLERS and sub in INSTALL_SUBCOMMANDS:
            problem = self._screen_install(argv)
            if problem:
                return Screening(False, program=program, reason=problem)

        if program in ("find", "fd") and any(a in ("-exec", "-execdir", "-ok", "-delete") for a in argv):
            return Screening(False, program=program,
                             reason=f"'{program}' with -exec/-delete can run or remove anything; refused")

        if program == "sed" and any(a.startswith("-i") for a in argv):
            # in-place edits are fine, but they must stay in the workspace,
            # which the path screen below enforces
            pass

        if program in ("xargs",):
            return Screening(False, program=program,
                             reason="'xargs' can invoke arbitrary programs; refused")

        # ---- path screening ---------------------------------------------
        check = self._screen_paths(argv)
        if check is not None:
            return Screening(False, program=program, reason=check)

        self._count += 1
        return Screening(True, argv=argv, program=program, cwd=cwd)

    def _outside_workspace(self, value: str) -> str:
        """Is this path outside the workspace?  Empty string when it is fine."""
        try:
            candidate = Path(value)
            if not candidate.is_absolute():
                return ""                 # relative to the workspace cwd
            resolved = candidate.resolve()
        except (OSError, ValueError):
            return f"cannot use the path {value!r}"
        try:
            resolved.relative_to(self.guard.root)
            return ""
        except ValueError:
            return (f"the path {value!r} is outside the workspace; packages must "
                    "be installed inside it (they go to .qwen35/)")

    def _screen_install(self, argv: list[str]) -> str:
        """Refuse an install that would land outside the workspace.

        The environment already points every package manager at ``.qwen35/`` (see
        :func:`scrub_env`), so the ordinary case is confined by construction.
        These are the explicit opt-outs, and the reason they matter is that a
        model which has just been told "you may install packages" will happily
        reach for ``pip install --user`` or ``npm install -g``.
        """
        for index, arg in enumerate(argv):
            low = arg.lower()
            if low in _INSTALL_ESCAPES:
                return (
                    f"{arg!r} installs outside the workspace. Packages here are "
                    "installed into .qwen35/ as the current user, with no admin "
                    "rights, so drop that flag."
                )
            flag = low.split("=", 1)[0]
            if flag in _INSTALL_PATH_FLAGS:
                if "=" in arg:
                    value = arg.split("=", 1)[1]
                else:
                    value = argv[index + 1] if index + 1 < len(argv) else ""
                if not value:
                    return f"{arg!r} needs a path"
                problem = self._outside_workspace(value)
                if problem:
                    return problem
        return ""

    def _screen_paths(self, argv: list[str]) -> str | None:
        """Reject arguments that would write or read outside the workspace."""
        for arg in argv[1:]:
            if not arg or arg.startswith("-"):
                continue
            if arg in ("/dev/null", "/dev/stdout", "nul", "NUL"):
                continue

            # A URL is not a filesystem path.  This is only reachable for the
            # network programs a profile explicitly permits, and http(s)/ftp
            # cannot read local files -- but file:// can, so it is refused.
            url = _URL_SCHEME.match(arg)
            if url:
                scheme = url.group(1).lower()
                if scheme in ("http", "https", "ftp", "ftps"):
                    continue
                return (
                    f"argument {arg!r} uses the {scheme!r} scheme, which can "
                    "read local files; refused"
                )

            # A POSIX-absolute path is an escape attempt on Windows (the child's
            # `cat` may be a real coreutils binary that honours /etc/passwd), and
            # on POSIX it is outside the workspace by definition.
            if arg[:1] in ("/", "\\"):
                return (
                    f"argument {arg!r} is an absolute path. The harness requires "
                    "paths relative to the workspace."
                )

            looks_like_path = (
                os.sep in arg
                or "/" in arg
                or "\\" in arg
                or arg.startswith("~")
                or (len(arg) > 1 and arg[1] == ":")
            )
            if not looks_like_path:
                continue

            if arg.startswith("~"):
                return (f"argument {arg!r} expands to the real home directory; "
                        "use a workspace-relative path instead")

            normalised = arg.replace("\\", "/")
            if normalised.startswith("//"):
                return f"argument {arg!r} is a network path; refused"

            try:
                self.guard.resolve(arg)
            except WorkspaceError as exc:
                return (
                    f"argument {arg!r} is refused: {exc}. Commands must operate "
                    "only inside the workspace; use relative paths."
                )
        return None
