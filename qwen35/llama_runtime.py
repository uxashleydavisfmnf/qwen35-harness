"""
llama.cpp runtime manager.

Downloads the official Windows CPU build of ``llama-server`` and launches it with
a flag set that this harness *probes for* rather than assumes.  llama.cpp ships a
nightly tag per build and its command line does change between versions (the
flash-attention flag in particular has moved from ``--flash-attn`` to
``-fa on|off|auto``), so :func:`build_command` checks ``--help`` output and only
passes flags the binary actually advertises.  A flag this harness guessed wrong
would otherwise be an opaque startup failure.

It also models the KV-cache economics for this specific architecture, because
the hybrid linear-attention layout is what makes a long window affordable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import ModelProfile

#: Serialises server starts *within this process*.  Concurrent starts are the
#: root cause of the "server is running but does not accept the stored API key"
#: state: the loser of the port race cannot bind, its wait_ready is satisfied by
#: the winner's /health, and it then saves a key the live server has never seen.
_START_LOCK = threading.Lock()

__all__ = [
    "runtime_dir",
    "installed_server",
    "detect_latest_tag",
    "download_runtime",
    "build_command",
    "LlamaServer",
    "cpu_threads",
    "kv_cache_report",
    "server_key_path",
    "load_server_key",
    "save_server_key",
    "new_api_key",
    "probe_server",
    "ensure_server",
    "ServerStatus",
]

GITHUB_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases"


def runtime_dir() -> Path:
    env = os.environ.get("QWEN35_LLAMA_DIR")
    if env:
        return Path(env)
    base = Path(os.environ.get("QWEN35_HOME", Path.home() / ".qwen35"))
    return base / "llama.cpp"


def cpu_threads() -> int:
    """Physical-core estimate: llama.cpp is fastest near physical core count."""
    logical = os.cpu_count() or 4
    env = os.environ.get("NUMBER_OF_PROCESSORS")
    if env and env.isdigit():
        logical = int(env)
    return max(1, logical // 2) if logical > 2 else max(1, logical)


# --------------------------------------------------------------------------
# Discovery / installation
# --------------------------------------------------------------------------
def installed_server() -> Path | None:
    """Find a llama-server binary in the managed runtime dir."""
    root = runtime_dir()
    if not root.is_dir():
        return None
    names = ("llama-server.exe", "llama-server")
    # exact match first (fast path), then a bounded recursive search
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    for name in names:
        for candidate in root.rglob(name):
            if candidate.is_file():
                return candidate
    return None


def _request_json(url: str) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": "qwen35-harness", "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def detect_latest_tag() -> str:
    """Newest ``b<number>`` build tag that actually carries Windows binaries."""
    data = _request_json(f"{GITHUB_RELEASES}?per_page=15")
    for release in data:
        tag = release.get("tag_name", "")
        if not tag.startswith("b"):
            continue
        for asset in release.get("assets", []):
            if asset.get("name", "").startswith("llama-") and "win-cpu-x64" in asset["name"]:
                return tag
    raise RuntimeError(
        "could not find a Windows CPU llama.cpp build in the last 15 releases"
    )


def asset_name(tag: str) -> str:
    return f"llama-{tag}-bin-win-cpu-x64.zip"


def download_runtime(tag: str | None = None, *, progress=None) -> Path:
    """Download and extract the Windows CPU build.  Returns the runtime dir."""
    tag = tag or detect_latest_tag()
    name = asset_name(tag)
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}"

    dest = runtime_dir()
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / name

    if not archive.is_file() or archive.stat().st_size < 1_000_000:
        if progress:
            progress(f"downloading {name}")
        req = urllib.request.Request(url, headers={"User-Agent": "qwen35-harness"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp, \
                    open(archive, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"could not download {url}: HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not download {url}: {exc.reason}") from exc

    if progress:
        progress(f"extracting {name}")
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            # flatten: the zip nests everything under build/bin/
            target = dest / Path(member).name
            if member.endswith("/"):
                continue
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)

    for exe in dest.glob("*.exe"):
        try:
            os.chmod(exe, 0o755)
        except OSError:
            pass

    server = installed_server()
    if server is None:
        raise RuntimeError(
            f"extracted {name} but no llama-server binary was found in {dest}"
        )
    return dest


# --------------------------------------------------------------------------
# Command construction with flag probing
# --------------------------------------------------------------------------
def help_text(server: Path) -> str:
    for flag in ("--help", "-h"):
        try:
            proc = subprocess.run(
                [str(server), flag], capture_output=True, text=True,
                timeout=25, encoding="utf-8", errors="replace",
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            if out.strip():
                return out
        except (OSError, subprocess.TimeoutExpired):
            continue
    return ""


@dataclass
class LaunchPlan:
    argv: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


def build_command(
    server: Path,
    weights: Path,
    *,
    mmproj: Path | None = None,
    context: int = 32768,
    threads: int | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    kv_cache_type: str = "q8_0",
    flash_attention: bool = True,
    parallel: int = 1,
    api_key: str = "",
    image_min_tokens: int = 1024,
    no_mmproj_offload: bool = True,
    mtp: bool = False,
    mtp_draft_max: int = 6,
    no_context_shift: bool = True,
    ctx_checkpoints: int = 0,
    checkpoint_min_step: int = 512,
    probe: bool = True,
    extra: list[str] | None = None,
) -> LaunchPlan:
    """Assemble a ``llama-server`` command line, probing for supported flags.

    Everything optional is looked up in ``--help`` first.  llama.cpp ships a
    nightly build per tag and its command line does change between versions (the
    flash-attention flag moved from ``--flash-attn`` to ``-fa on|off|auto``), so a
    flag this harness merely assumed would otherwise surface as an opaque
    startup failure with no explanation.
    """
    help_out = help_text(server) if probe else ""
    plan = LaunchPlan()

    def supported(*candidates: str) -> str | None:
        if not help_out:
            # no probe available: assume the modern spelling
            return candidates[0]
        for cand in candidates:
            if cand in help_out:
                return cand
        return None

    def add_flag(*candidates: str, value: str | None = None,
                 label: str = "") -> bool:
        flag = supported(*candidates)
        if flag is None:
            plan.unsupported.append(candidates[0])
            return False
        argv.append(flag)
        if value is not None:
            argv.append(value)
        if label:
            plan.notes.append(label)
        return True

    argv: list[str] = [str(server), "-m", str(weights)]

    argv += ["--host", host, "--port", str(port)]
    argv += ["-c", str(int(context))]
    argv += ["-t", str(int(threads or cpu_threads()))]

    # Always pin the slot count.  llama-server now defaults to several slots and
    # splits the requested context between them, which silently halves (or
    # quarters) the usable window; MTP also requires exactly one.
    if not add_flag("-np", "--parallel", value=str(int(parallel))):
        plan.notes.append(
            "WARNING: this build has no -np; the context may be split across "
            "several slots"
        )

    # --- chat template: mandatory for tools + thinking ---------------------
    if add_flag("--jinja", "-j",
                label="native chat template enabled (tools + enable_thinking)"):
        pass
    else:
        plan.notes.append(
            "WARNING: this build has no --jinja, so the model's own tool "
            "prompt and thinking switch cannot be applied; reliability will drop"
        )

    # --- security: a key stops other local processes (and any web page that
    #     reaches 127.0.0.1 via CORS) from driving the model ----------------
    if api_key:
        if not add_flag("--api-key", value=api_key):
            plan.notes.append(
                "WARNING: this build has no --api-key; the server is open to "
                "any local process"
            )

    # --- flash attention ---------------------------------------------------
    if flash_attention:
        if supported("-fa"):
            argv += ["-fa", "on"]
            plan.notes.append("flash attention on")
        elif supported("--flash-attn", "-fa"):
            argv += ["--flash-attn", "on"]
            plan.notes.append("flash attention on (legacy flag)")
        else:
            plan.unsupported.append("--flash-attn")

    # --- KV cache quantisation --------------------------------------------
    if kv_cache_type and kv_cache_type != "f16":
        ck = supported("--cache-type-k", "-ctk")
        cv = supported("--cache-type-v", "-ctv")
        if ck and cv:
            argv += [ck, kv_cache_type, cv, kv_cache_type]
            plan.notes.append(
                f"KV cache quantised to {kv_cache_type} "
                "(applies to the 6 full-attention layers)"
            )
        else:
            plan.unsupported.append("--cache-type-k")

    # --- vision ------------------------------------------------------------
    if mmproj and mmproj.is_file():
        if supported("--mmproj", "-mm"):
            add_flag("--mmproj", "-mm", value=str(mmproj),
                     label=f"vision projector attached ({mmproj.name})")
            # On a CPU-only host the projector must not be GPU-offloaded; that is
            # the default, so it has to be turned off explicitly.
            if no_mmproj_offload:
                if supported("--no-mmproj-offload"):
                    argv.append("--no-mmproj-offload")
                    plan.notes.append("vision projector kept on CPU")
                elif supported("-mmdev", "--mmproj-device"):
                    dev = supported("-mmdev", "--mmproj-device")
                    argv += [dev, "none"]
                    plan.notes.append("vision projector kept on CPU (device=none)")
                else:
                    plan.unsupported.append("--no-mmproj-offload")
            if image_min_tokens:
                add_flag("--image-min-tokens", value=str(int(image_min_tokens)),
                         label=f"min {image_min_tokens} visual tokens "
                               "(Qwen-VL grounding requirement)")
        else:
            plan.unsupported.append("--mmproj")
            plan.notes.append("WARNING: build has no --mmproj; vision disabled")

    # --- speculative decoding via the model's own MTP head -----------------
    if mtp:
        if supported("--spec-type"):
            argv += ["--spec-type", "draft-mtp",
                     "--spec-draft-n-max", str(int(mtp_draft_max))]
            plan.notes.append(
                f"self-speculative decoding via the MTP head "
                f"(draft max {mtp_draft_max})"
            )
        else:
            plan.unsupported.append("--spec-type")
            plan.notes.append("WARNING: build has no MTP support; running without it")

    # --- stability ---------------------------------------------------------
    if no_context_shift:
        # silently sliding the window corrupts an agent transcript; the harness
        # would rather be told the context is full
        add_flag("--no-context-shift",
                 label="context shifting disabled (the harness manages the window)")

    # --- context checkpoints ------------------------------------------------
    # Off by default: measured on this machine they made the post-compaction
    # re-prefill *slower* and cost 350-650 MB of RSS, with no benefit to ordinary
    # prefix reuse.  See RuntimeConfig for the three-way measurement.
    if ctx_checkpoints:
        if add_flag("-ctxcp", "--ctx-checkpoints",
                    value=str(int(ctx_checkpoints)),
                    label=f"context checkpoints: up to {ctx_checkpoints} per slot"):
            if checkpoint_min_step:
                add_flag("-cms", "--checkpoint-min-step",
                         value=str(int(checkpoint_min_step)),
                         label=f"checkpoint spacing {checkpoint_min_step} tokens")
    else:
        plan.notes.append("context checkpoints off "
                          "(measured: slower re-prefill, +350-650 MB)")

    if extra:
        argv += [str(x) for x in extra]

    plan.argv = argv
    return plan


# --------------------------------------------------------------------------
# KV cache economics
# --------------------------------------------------------------------------
def kv_cache_report(profile: ModelProfile, contexts: tuple[int, ...] = (
    8192, 32768, 65536, 131072, 262144,
)) -> list[dict[str, float | int | str]]:
    """Memory the KV cache needs at several window sizes.

    Only the ``full_attention`` layers grow a cache; the linear-attention layers
    hold a fixed-size recurrent state.  That ratio is the whole reason a 262 144
    token window is practical on a 16 GB machine.
    """
    rows: list[dict[str, float | int | str]] = []
    for ctx in contexts:
        f16 = profile.kv_bytes(ctx, 2.0) / (1024 ** 3)
        q8 = profile.kv_bytes(ctx, 1.0625) / (1024 ** 3)
        rows.append({
            "context": ctx,
            "full_attention_layers": profile.full_attention_layers,
            "layers_total": profile.num_hidden_layers,
            "kv_f16_gib": round(f16, 2),
            "kv_q8_gib": round(q8, 2),
            "hypothetical_all_attention_f16_gib": round(
                f16 * profile.num_hidden_layers / max(1, profile.full_attention_layers), 2
            ),
        })
    return rows


# --------------------------------------------------------------------------
# Process management
# --------------------------------------------------------------------------
class LlamaServer:
    """Supervises a ``llama-server`` child process."""

    def __init__(self, plan: LaunchPlan, *, log_path: Path | None = None) -> None:
        self.plan = plan
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self._log_handle = None

    def start(self) -> None:
        if self.proc is not None:
            return
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(self.log_path, "w", encoding="utf-8", errors="replace")
            stdout = self._log_handle
        else:
            stdout = subprocess.DEVNULL
        creation = 0
        if sys.platform == "win32":
            # let the child outlive a Ctrl-C in the parent shell
            creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self.proc = subprocess.Popen(
            self.plan.argv,
            stdout=stdout,
            stderr=subprocess.STDOUT if stdout is not subprocess.DEVNULL else subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creation,
        )

    def wait_ready(self, base_url: str, timeout: float = 180.0) -> tuple[bool, str]:
        from .backends.openai_compat import OpenAICompatBackend

        probe = OpenAICompatBackend(base_url)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                tail = ""
                if self.log_path and self.log_path.is_file():
                    try:
                        tail = self.log_path.read_text(
                            encoding="utf-8", errors="replace"
                        )[-1200:]
                    except OSError:
                        pass
                return False, (
                    f"llama-server exited with code {self.proc.returncode}.\n"
                    f"--- log tail ---\n{tail}"
                )
            ok, _ = probe.health()
            if ok:
                return True, "ready"
            time.sleep(1.0)
        return False, f"llama-server did not become ready within {timeout:.0f}s"

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        if self._log_handle:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


# --------------------------------------------------------------------------
# Server credentials
# --------------------------------------------------------------------------
def server_key_path() -> Path:
    base = Path(os.environ.get("QWEN35_HOME", Path.home() / ".qwen35"))
    return base / "server.json"


def load_server_key() -> str:
    p = server_key_path()
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("api_key", "")
        except (OSError, json.JSONDecodeError):
            return ""
    return ""


def save_server_key(key: str, base_url: str, pid: int, model: str) -> None:
    p = server_key_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "api_key": key, "base_url": base_url, "pid": pid, "model": model,
        "started": time.time(),
    }, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def new_api_key(length: int = 32) -> str:
    import secrets
    import string as _string

    alphabet = _string.ascii_letters + _string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


# --------------------------------------------------------------------------
# Server lifecycle, shared by the CLI and the GUI
# --------------------------------------------------------------------------
def port_in_use(host: str, port: int, timeout: float = 0.6) -> bool:
    """Is something already accepting connections on this port?

    llama-server **does not exit when its port is already taken** -- it logs the
    failure and keeps running.  That combination is dangerous here, because
    :meth:`LlamaServer.wait_ready` only polls ``/health``, which the *other*
    server answers: ``ensure_server`` then reports success and saves the key of a
    process that never bound anything.

    That is not hypothetical.  A real machine ended up with two llama-server
    processes, only one owning port 8080, and ``server.json`` holding the loser's
    key -- so every programmatic caller got an opaque ``HTTP 401 Invalid API Key``
    while the GUI (which reuses its in-memory key) worked fine.  Checking the port
    before starting removes the whole failure mode.
    """
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_server(base_url: str, api_key: str = "",
                 timeout: float = 6.0) -> tuple[str, str]:
    """Return ``(state, message)`` where state is up / unauthorized / down.

    ``/health`` is deliberately unauthenticated, so it only proves *something*
    is listening.  When a key is configured the key is verified too, because a
    stale key would otherwise surface much later as an opaque 401 mid-task.
    """
    base = base_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace").strip()
    except urllib.error.HTTPError as exc:
        return "down", f"HTTP {exc.code} from /health"
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return "down", f"{type(exc).__name__}: {exc}"

    status = "ok"
    if body.startswith("{"):
        try:
            status = json.loads(body).get("status", "ok")
        except json.JSONDecodeError:
            pass
    if status not in ("ok", "no slot available"):
        return "down", f"health reports {status!r}"

    if not api_key:
        return "up", "running without a key check"

    req = urllib.request.Request(f"{base}/v1/models",
                                 headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return "up", "authenticated"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return "unauthorized", (
                "a server is running but does not accept the stored API key"
            )
        return "up", f"/v1/models returned HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return "up", f"key probe inconclusive: {exc}"


@dataclass
class ServerStatus:
    ok: bool
    state: str = ""            # reused | started | failed
    message: str = ""
    base_url: str = ""
    api_key: str = ""
    server: "LlamaServer | None" = None
    notes: list[str] = field(default_factory=list)

    @property
    def we_own_it(self) -> bool:
        return self.server is not None


def stop_servers_on_port(host: str = "127.0.0.1", port: int = 8080) -> int:
    """Kill whatever llama-server owns the port.  Returns how many were stopped.

    Used only when the caller has explicitly asked to take the port back.  The
    pid is taken from the socket rather than from ``server.json``, because the
    recorded pid is exactly what has gone stale when this is needed.
    """
    pids: set[int] = set()
    try:
        import subprocess as _sp

        out = _sp.run(["netstat", "-ano", "-p", "TCP"], capture_output=True,
                      text=True, timeout=15, encoding="utf-8", errors="replace")
        for line in (out.stdout or "").splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].upper() == "TCP" and \
                    parts[3].upper() == "LISTENING" and \
                    parts[1].endswith(f":{port}"):
                try:
                    pids.add(int(parts[4]))
                except ValueError:
                    pass
    except Exception:  # noqa: BLE001 - fall through to the name-based sweep
        pass

    stopped = 0
    for pid in pids:
        try:
            check = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=15,
                encoding="utf-8", errors="replace")
            if "llama-server" not in (check.stdout or "").lower():
                continue          # not ours; never kill an unrelated process
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, timeout=15)
            stopped += 1
        except Exception:  # noqa: BLE001
            continue

    if not stopped:
        # Nothing identifiable owned the port by pid; fall back to the documented
        # sweep, which is also what stop.bat does.
        try:
            probe = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe",
                                    "/FO", "CSV", "/NH"], capture_output=True,
                                   text=True, timeout=15, encoding="utf-8",
                                   errors="replace")
            if "llama-server" in (probe.stdout or "").lower():
                subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                               capture_output=True, timeout=20)
                stopped = (probe.stdout or "").lower().count("llama-server")
        except Exception:  # noqa: BLE001
            pass
    return stopped


def ensure_server(
    *,
    model_dir: str | os.PathLike[str],
    context: int = 32768,
    port: int = 8080,
    host: str = "127.0.0.1",
    gguf: str | os.PathLike[str] | None = None,
    mmproj: str | os.PathLike[str] | None = None,
    kv_cache_type: str = "q8_0",
    mtp: bool = False,
    threads: int | None = None,
    ctx_checkpoints: int = 16,
    checkpoint_min_step: int = 512,
    force_restart: bool = False,
    timeout: float = 300.0,
    state_dir: str | os.PathLike[str] | None = None,
    progress: Any = None,
) -> ServerStatus:
    """Reuse a running server, or start one and wait until it is ready.

    This is the one place server lifecycle is decided, so the CLI, the GUI and
    any future front end behave identically.

    ``force_restart=True`` means "if something unusable already holds the port,
    stop it and take the port back".
    """
    # Serialise starts inside this process.  Two clicks on 重启服务 used to
    # launch two starts at once; the loser could not bind, but wait_ready was
    # satisfied by the *winner's* /health, so it saved its own key and the
    # running server ended up with a key nobody had.  The symptom was exactly
    # "a server is running but does not accept the stored API key".
    with _START_LOCK:
        return _ensure_server_locked(
            model_dir=model_dir, context=context, port=port, host=host,
            gguf=gguf, mmproj=mmproj, kv_cache_type=kv_cache_type, mtp=mtp,
            threads=threads, ctx_checkpoints=ctx_checkpoints,
            checkpoint_min_step=checkpoint_min_step,
            force_restart=force_restart, timeout=timeout, state_dir=state_dir,
            progress=progress)


def _ensure_server_locked(
    *,
    model_dir: str | os.PathLike[str],
    context: int = 32768,
    port: int = 8080,
    host: str = "127.0.0.1",
    gguf: str | os.PathLike[str] | None = None,
    mmproj: str | os.PathLike[str] | None = None,
    kv_cache_type: str = "q8_0",
    mtp: bool = False,
    threads: int | None = None,
    ctx_checkpoints: int = 16,
    checkpoint_min_step: int = 512,
    force_restart: bool = False,
    timeout: float = 300.0,
    state_dir: str | os.PathLike[str] | None = None,
    progress: Any = None,
) -> ServerStatus:
    base_url = f"http://{host}:{port}"
    api_key = load_server_key()

    state, message = probe_server(base_url, api_key)
    if state == "up":
        return ServerStatus(True, "reused", message, base_url, api_key)

    if state == "unauthorized" or port_in_use(host, port):
        # Something is on the port and we cannot use it: either it rejects the
        # stored key, or it does not answer the health check at all.
        detail = (message if state == "unauthorized"
                  else f"it did not answer the health check ({message})")
        if not force_restart:
            return ServerStatus(
                False, "failed",
                f"{detail}. It is most likely a llama-server left over from an "
                "earlier run. Stop it (stop.bat, or taskkill /IM "
                "llama-server.exe) and try again -- or let the harness do it "
                "for you (the GUI offers this, the CLI takes --force-restart).",
                base_url, "",
            )
        if progress:
            progress("stopping the server that holds the port")
        killed = stop_servers_on_port(host, port)
        if not killed:
            return ServerStatus(
                False, "failed",
                f"{detail}, and it could not be stopped automatically. "
                "Run stop.bat, then try again.",
                base_url, "",
            )
        # give the socket a moment to be released before rebinding
        for _ in range(20):
            if not port_in_use(host, port):
                break
            time.sleep(0.25)
        if port_in_use(host, port):
            return ServerStatus(
                False, "failed",
                f"stopped {killed} process(es) but {host}:{port} is still "
                "occupied. Something else is holding it.",
                base_url, "",
            )

    server_bin = installed_server()
    if server_bin is None:
        return ServerStatus(False, "failed",
                            "llama-server is not installed -- run install-runtime",
                            base_url, "")

    weights = Path(gguf) if gguf else None
    projector = Path(mmproj) if mmproj else None
    if weights is None or projector is None:
        from .config import discover_gguf

        found_w, found_m = discover_gguf(Path(model_dir))
        weights = weights or found_w
        projector = projector or found_m
    if weights is None:
        return ServerStatus(False, "failed",
                            f"no GGUF weights found in {model_dir}", base_url, "")

    api_key = new_api_key()
    plan = build_command(
        server_bin, weights,
        mmproj=projector, context=context, threads=threads,
        host=host, port=port, kv_cache_type=kv_cache_type,
        flash_attention=True, parallel=1, api_key=api_key, mtp=mtp,
        ctx_checkpoints=ctx_checkpoints,
        checkpoint_min_step=checkpoint_min_step,
    )

    base_state = Path(state_dir) if state_dir else server_key_path().parent
    log_path = base_state / "logs" / f"server-{time.strftime('%Y%m%d-%H%M%S')}.log"
    if progress:
        progress(f"starting llama-server ({weights.name})")

    server = LlamaServer(plan, log_path=log_path)
    try:
        server.start()
    except OSError as exc:
        return ServerStatus(False, "failed", f"could not start llama-server: {exc}",
                            base_url, "")

    ok, msg = server.wait_ready(base_url, timeout=timeout)
    if not ok:
        server.stop()
        return ServerStatus(False, "failed", msg, base_url, "")

    # --- prove the server answering is OURS --------------------------------
    # /health is answered by whoever owns the port, so "ready" only proves that
    # *some* server is up.  Ask it to authenticate with the key we just handed
    # our own child: a foreign server has never seen that key and will refuse.
    # Without this check the harness saved a key the live server did not have,
    # and every later run failed with "does not accept the stored API key".
    confirm, confirm_message = probe_server(base_url, api_key, timeout=15.0)
    if confirm != "up":
        server.stop()
        return ServerStatus(
            False, "failed",
            f"another server is holding {host}:{port} ({confirm_message}). It "
            "was not started by this window and does not accept this key. Use "
            "--force-restart / the GUI's restart offer to take the port back.",
            base_url, "",
        )

    # Belt and braces: confirm our own child is still alive before claiming it.
    if server.proc is not None and server.proc.poll() is not None:
        return ServerStatus(
            False, "failed",
            f"llama-server exited immediately with code {server.proc.returncode}; "
            f"something else is answering on {host}:{port}.",
            base_url, "",
        )

    save_server_key(api_key, base_url,
                    server.proc.pid if server.proc else 0, str(weights))
    return ServerStatus(True, "started", "server ready", base_url, api_key,
                        server, list(plan.notes))
