"""
Command line interface.

The workspace picker is where the permission story starts, so it is deliberately
noisy: it shows every candidate with a verdict, refuses filesystem roots and
system directories by name, and never silently falls back to something broad.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
import time
from pathlib import Path
from typing import Any

from .config import (DEFAULT_PROFILE, PROFILE_CHOICES, PROFILES, QWEN35_08B,
                     HarnessConfig, RuntimeConfig,
                     config_path, default_model_dir, discover_gguf,
                     get_permissions, load_config, save_config)
from .session import SessionRecorder, list_sessions
from .workspace import (WorkspaceError, candidate_workspaces,
                        recent_workspaces, remember_workspace,
                        validate_workspace)

# --------------------------------------------------------------------------
# Output helpers (colour only when it cannot hurt)
# --------------------------------------------------------------------------
_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(t: str) -> str:
    return _c(t, "2")


def bold(t: str) -> str:
    return _c(t, "1")


def green(t: str) -> str:
    return _c(t, "32")


def yellow(t: str) -> str:
    return _c(t, "33")


def red(t: str) -> str:
    return _c(t, "31")


def cyan(t: str) -> str:
    return _c(t, "36")


def hr(char: str = "-", width: int = 74) -> str:
    return dim(char * width)


# --------------------------------------------------------------------------
# Workspace selection
# --------------------------------------------------------------------------
def _print_workspace_refusal(raw: str, exc: Exception) -> None:
    print(f"  {red('REFUSED')} {raw}")
    print(f"          {exc}")


def choose_workspace(requested: str | None, *, create: bool = False,
                     interactive: bool = True) -> Path:
    """Resolve the workspace, validating anything the operator picks."""
    if requested:
        try:
            ws = validate_workspace(requested, create=create)
        except WorkspaceError as exc:
            print(red("The requested workspace was refused."))
            _print_workspace_refusal(requested, exc)
            print(
                "\nA workspace must be a real project directory: not a drive "
                "root, not the home directory itself, and not inside the "
                "operating system's own folders."
            )
            raise SystemExit(2)
        remember_workspace(ws)
        return ws

    # look for a workspace remembered in the current directory tree
    here = Path.cwd()
    for candidate in [here, *here.parents]:
        try:
            ws = validate_workspace(candidate)
        except WorkspaceError:
            continue
        if config_path(ws).is_file():
            print(f"Using workspace {bold(str(ws))} {dim('(remembered)')}")
            remember_workspace(ws)
            return ws
        break

    if not interactive or not sys.stdin.isatty():
        print(
            red("No workspace given.") + " Pass --workspace PATH.\n"
            "Refusing to guess: the model's file access is scoped to this "
            "choice."
        )
        raise SystemExit(2)

    options = candidate_workspaces(limit=14)
    valid: list[Path] = []
    print(bold("\nChoose a workspace"))
    print(dim("  The model can read and write here and nowhere else."))
    print(dim("  Drive roots, the home directory itself and system folders are refused.\n"))

    for raw in options:
        try:
            ws = validate_workspace(raw)
            valid.append(ws)
            mark = ""
            if config_path(ws).is_file():
                mark = dim("  (used before)")
            print(f"  {len(valid):>2}. {ws}{mark}")
        except WorkspaceError:
            continue

    if not valid:
        print(yellow("  No candidate directories found on this machine."))

    print(f"\n  Enter a number, or type a path.")
    print(dim("  (a path that looks like a drive root or system folder will be refused)\n"))

    for _attempt in range(6):
        try:
            answer = input("workspace> ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(130)

        if not answer:
            continue

        if answer.isdigit() and 1 <= int(answer) <= len(valid):
            ws = valid[int(answer) - 1]
            remember_workspace(ws)
            return ws

        try:
            ws = validate_workspace(answer, create=create)
        except WorkspaceError as exc:
            _print_workspace_refusal(answer, exc)
            continue

        remember_workspace(ws)
        return ws

    print(red("Too many invalid attempts."))
    raise SystemExit(2)


# --------------------------------------------------------------------------
# Config assembly
# --------------------------------------------------------------------------
def build_config(args: argparse.Namespace, workspace: Path,
                 *, interactive: bool = True) -> HarnessConfig:
    saved = load_config(workspace) or {}

    profile = getattr(args, "profile", None) or saved.get(
        "permission_profile", DEFAULT_PROFILE)
    perms_data = saved.get("permissions", {})
    perms = get_permissions(profile)
    if saved.get("permission_profile") == profile:
        for key, value in perms_data.items():
            if hasattr(perms, key):
                setattr(perms, key, value)

    rt_saved = saved.get("runtime", {})
    rt = RuntimeConfig()

    def pick(attr: str, arg_name: str) -> None:
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(rt, attr, val)
        elif attr in rt_saved:
            setattr(rt, attr, rt_saved[attr])

    for attr, arg_name in (
        ("backend", "backend"), ("base_url", "base_url"), ("api_key", "api_key"),
        ("model_dir", "model_dir"), ("gguf_path", "gguf"), ("mmproj_path", "mmproj"),
        ("context_window", "context"), ("max_steps", "max_steps"),
        ("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
        ("max_tokens", "max_tokens"), ("thinking", "thinking"),
        ("use_grammar", "grammar"), ("tool_prompt", "tool_prompt"),
        ("kv_cache_type", "kv_cache_type"), ("seed", "seed"),
        ("keep_recent_turns", "keep_recent_turns"),
    ):
        pick(attr, arg_name)

    if getattr(args, "mtp", False):
        rt.mtp = True
    if getattr(args, "no_mtp", False):
        rt.mtp = False
    if getattr(args, "no_flash_attn", False):
        rt.flash_attention = False
    if getattr(args, "threads", None):
        rt.threads = args.threads  # type: ignore[attr-defined]

    if not rt.model_dir:
        rt.model_dir = str(default_model_dir())
    if not rt.gguf_path or not rt.mmproj_path:
        weights, mmproj = discover_gguf(Path(rt.model_dir))
        rt.gguf_path = rt.gguf_path or (str(weights) if weights else "")
        rt.mmproj_path = rt.mmproj_path or (str(mmproj) if mmproj else "")

    return HarnessConfig(
        workspace=workspace,
        permissions=perms,
        runtime=rt,
        permission_profile=profile,
    )


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    print(bold(f"\nqwen35 harness - environment report"))
    print(hr("="))

    print(bold("\nModel (verified from the published config.json)"))
    print(f"  id                {QWEN35_08B.name}")
    print(f"  context window    {QWEN35_08B.context_window:,} tokens")
    print(f"  architecture      {QWEN35_08B.num_hidden_layers} layers = "
          f"{QWEN35_08B.num_hidden_layers - QWEN35_08B.full_attention_layers} "
          f"linear-attention (Gated DeltaNet) + "
          f"{QWEN35_08B.full_attention_layers} full-attention")
    print(f"  vision            {'yes' if QWEN35_08B.supports_vision else 'no'} "
          f"(image token {248056}, patch {16}, merge {2})")
    print(f"  MTP head          {QWEN35_08B.mtp_layers} layer (self-speculative decoding)")

    print(bold("\nKV cache cost (why a long window is affordable here)"))
    from .llama_runtime import kv_cache_report
    print(f"  {'context':>9}  {'KV f16':>9}  {'KV q8_0':>9}   "
          f"{'if all 24 layers cached':>24}")
    for row in kv_cache_report(QWEN35_08B):
        print(f"  {row['context']:>9,}  {row['kv_f16_gib']:>7.2f} GiB  "
              f"{row['kv_q8_gib']:>7.2f} GiB   "
              f"{row['hypothetical_all_attention_f16_gib']:>20.2f} GiB")
    print(dim("  Only the 6 full-attention layers grow a cache; the 18 linear "
              "layers\n  hold a constant-size recurrent state."))

    print(bold("\nRuntime"))
    try:
        from .llama_runtime import installed_server, runtime_dir
        srv = installed_server()
        print(f"  llama-server      {srv if srv else red('not installed')}")
        print(f"  runtime dir       {runtime_dir()}")
    except Exception as exc:  # pragma: no cover
        print(f"  llama-server      {red(str(exc))}")

    model_dir = Path(args.model_dir) if args.model_dir else default_model_dir()
    weights, mmproj = discover_gguf(model_dir)
    print(f"  model dir         {model_dir}")
    print(f"  weights           "
          f"{weights.name + ' (' + f'{weights.stat().st_size/1048576:.0f} MB' + ')' if weights else red('none found')}")
    print(f"  vision projector  {mmproj.name if mmproj else yellow('none found (vision disabled)')}")
    print(f"  CPU threads       {__import__('qwen35.llama_runtime', fromlist=['x']).cpu_threads()}")

    print(bold("\nPython"))
    print(f"  version           {sys.version.split()[0]}")
    for mod in ("torch", "transformers", "PIL", "numpy"):
        try:
            m = __import__(mod)
            print(f"  {mod:<17} {getattr(m, '__version__', 'present')}")
        except ImportError:
            print(f"  {mod:<17} {dim('not installed')}")

    print(bold("\nServer"))
    rt = RuntimeConfig(model_dir=str(model_dir))
    from .backends.openai_compat import OpenAICompatBackend
    probe = OpenAICompatBackend(rt.base_url)
    ok, msg = probe.health()
    print(f"  {rt.base_url:<17} {green('ready') if ok else yellow('not running')}  {dim(msg)}")

    print(bold("\nPermission profiles"))
    for name, p in PROFILES.items():
        bits = []
        bits.append("read" if p.read else "no-read")
        bits.append("write" if p.write else "no-write")
        bits.append("run" if p.run_command else "no-run")
        bits.append("net" if p.allow_network_commands else "no-net")
        print(f"  {name:<10} {', '.join(bits)}   "
              f"out<={p.max_output_bytes}B  cmds<={p.max_commands_per_run}")

    print(bold("\nWorkspace containment"))
    here = Path.cwd()
    try:
        ws = validate_workspace(here)
        print(f"  {here} -> {green('acceptable')}")
    except WorkspaceError as exc:
        print(f"  {here} -> {red('REFUSED')} ({exc})")
    for probe_path in ("C:\\", "D:\\", str(Path.home()), "C:\\Windows"):
        try:
            validate_workspace(probe_path)
            print(f"  {probe_path:<24} {yellow('accepted (!)')}")
        except WorkspaceError:
            print(f"  {probe_path:<24} {green('refused')}")

    print(hr("="))
    return 0


def cmd_workspaces(args: argparse.Namespace) -> int:
    print(bold("\nRecent workspaces"))
    recent = recent_workspaces()
    if not recent:
        print(dim("  (none yet)"))
    for path in recent:
        try:
            ws = validate_workspace(path)
            print(f"  {green('ok')}      {ws}")
        except WorkspaceError as exc:
            print(f"  {red('invalid')} {path}  {dim(str(exc))}")

    print(bold("\nDiscovered candidates"))
    for path in candidate_workspaces():
        print(f"  {path}")

    print(bold("\nPaths that are always refused"))
    for probe_path in ("C:\\", "D:\\", "/", str(Path.home()), "C:\\Windows",
                       os.environ.get("APPDATA", "")):
        if not probe_path:
            continue
        try:
            validate_workspace(probe_path)
            print(f"  {yellow('ACCEPTED')}  {probe_path}")
        except WorkspaceError:
            print(f"  {green('refused')}   {probe_path}")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    from .tools import TOOLS
    from .grammar import ToolSignature, build_grammar, validate_grammar
    print(bold("\nTools offered to the model"))
    for name, td in TOOLS.items():
        req = ", ".join(td.spec.required) or "-"
        print(f"  {cyan(name):<24} params: {', '.join(td.spec.params) or '-'}")
        print(f"  {'':<24} required: {req}")
        print(dim(f"  {'':<24} {td.spec.description.splitlines()[0][:96]}"))
    sigs = [ToolSignature(td.spec.name, td.spec.params) for td in TOOLS.values()]
    for thinking in (False, True):
        g = build_grammar(sigs, thinking=thinking, allow_plain_answer=True)
        print(f"\n  grammar (thinking={thinking}): {len(g)} bytes, "
              f"{g.count('::=')} rules, validator: "
              f"{validate_grammar(g) or green('clean')}")
    return 0


def cmd_grammar(args: argparse.Namespace) -> int:
    from .grammar import ToolSignature, build_grammar, nullable_rules, validate_grammar
    from .prompt import build_tool_schema
    from .tools import build_specs
    specs = build_specs(None, get_permissions(args.profile or "safe"))
    sigs = [ToolSignature(s.name, s.params) for s in specs]
    g = build_grammar(sigs, thinking=bool(args.thinking), allow_plain_answer=True)
    print(g)
    print("# validation:", validate_grammar(g) or "clean", file=sys.stderr)
    defined = {ln.split(" ::= ")[0]: ln.split(" ::= ")[1]
               for ln in g.splitlines() if " ::= " in ln}
    print("# root nullable:", nullable_rules(defined).get("root"), file=sys.stderr)
    print(f"# tools: {[s.name for s in specs]}", file=sys.stderr)
    return 0


def _server_key_path() -> Path:
    """Where the model server's API key lives (shared with the GUI)."""
    from .llama_runtime import server_key_path

    return server_key_path()


def _load_server_key() -> str:
    from .llama_runtime import load_server_key

    return load_server_key()


def cmd_install_runtime(args: argparse.Namespace) -> int:
    from .llama_runtime import download_runtime, installed_server, runtime_dir
    print(f"Runtime directory: {runtime_dir()}")
    existing = installed_server()
    if existing and not args.force:
        print(f"{green('already installed')}: {existing}")
        print(dim("(pass --force to re-download)"))
        return 0
    try:
        dest = download_runtime(args.tag, progress=lambda m: print(f"  {m}"))
    except Exception as exc:
        print(red(f"failed: {exc}"))
        return 1
    print(f"{green('installed')}: {installed_server()}  (in {dest})")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download GGUF weights (and the vision projector) from a HF mirror."""
    import urllib.error
    import urllib.request

    model_dir = Path(args.model_dir) if args.model_dir else default_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    repo = args.repo
    base = f"{args.mirror}/{repo}/resolve/main"
    files = [args.quant]
    if not args.no_vision:
        files.append(args.mmproj)

    print(f"Downloading into {model_dir}")
    for name in files:
        out = model_dir / name
        if out.is_file() and out.stat().st_size > 1_000_000 and not args.force:
            print(f"  {green('have')} {name}")
            continue
        url = f"{base}/{name}"
        print(f"  {dim('get')}  {name} ...", end="", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "qwen35-harness"})
            with urllib.request.urlopen(req, timeout=1800) as resp, open(out, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print(f"\r  {red('failed')} {name}: {exc}")
            if out.is_file():
                out.unlink(missing_ok=True)
            return 1
        print(f"\r  {green('done')} {name} ({out.stat().st_size/1048576:.0f} MB)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .backends.openai_compat import OpenAICompatBackend
    from .llama_runtime import (LlamaServer, build_command, cpu_threads,
                                installed_server, load_server_key, new_api_key,
                                port_in_use, probe_server, save_server_key,
                                stop_servers_on_port)

    server = installed_server()
    if not server:
        print(red("llama-server is not installed.") + " Run: qwen35 install-runtime")
        return 1

    model_dir = Path(args.model_dir) if args.model_dir else default_model_dir()
    weights = Path(args.gguf) if args.gguf else None
    mmproj = Path(args.mmproj) if args.mmproj else None
    if weights is None or mmproj is None:
        w, m = discover_gguf(model_dir)
        weights = weights or w
        mmproj = mmproj or m
    if weights is None:
        print(red(f"no GGUF weights in {model_dir}") + " Run: qwen35 fetch")
        return 1

    api_key = args.api_key or new_api_key()

    # If something unusable already holds the port, take it back rather than
    # starting a child that cannot bind and then reporting success because the
    # *other* server answered the health check.
    held = port_in_use(args.host, args.port)
    if held:
        if not getattr(args, "force_restart", False):
            state, message = probe_server(f"http://{args.host}:{args.port}",
                                          load_server_key())
            print(yellow(f"{args.host}:{args.port} is already in use") +
                  f" ({message if state != 'up' else 'a server is running'}).")
            print("  Start it with --force-restart to stop that server and take "
                  "the port back, or run stop.bat first.")
            return 1
        print(yellow(f"stopping whatever holds {args.host}:{args.port}…"))
        killed = stop_servers_on_port(args.host, args.port)
        print(dim(f"  stopped {killed} process(es)"))
        for _ in range(20):
            if not port_in_use(args.host, args.port):
                break
            time.sleep(0.25)
        if port_in_use(args.host, args.port):
            print(red(f"{args.host}:{args.port} is still occupied."))
            return 1

    plan = build_command(
        server, weights,
        mmproj=mmproj,
        context=args.context,
        threads=args.threads or cpu_threads(),
        host=args.host, port=args.port,
        kv_cache_type=args.kv_cache_type,
        flash_attention=not args.no_flash_attn,
        parallel=1,
        api_key=api_key,
        image_min_tokens=args.image_min_tokens,
        no_mmproj_offload=args.no_mmproj_offload,
        mtp=args.mtp,
        mtp_draft_max=args.mtp_draft_max,
        ctx_checkpoints=args.ctx_checkpoints,
        checkpoint_min_step=args.checkpoint_min_step,
    )

    print(bold("llama-server"))
    print(f"  weights   {weights}")
    print(f"  vision    {mmproj if mmproj else dim('none')}")
    print(f"  context   {args.context:,} tokens")
    print(f"  threads   {args.threads or cpu_threads()}")
    for note in plan.notes:
        print(f"  {dim('note')}      {note}")
    if plan.unsupported:
        print(f"  {yellow('unsupported flags')}: {', '.join(plan.unsupported)}")
    print(dim("\n  " + " ".join(_redact_key(plan.argv, api_key)) + "\n"))

    log = Path(os.environ.get("QWEN35_HOME", Path.home() / ".qwen35")) / "llama-server.log"
    srv = LlamaServer(plan, log_path=log)
    srv.start()
    base_url = f"http://{args.host}:{args.port}"
    ok, msg = srv.wait_ready(base_url, timeout=args.timeout)
    if not ok:
        print(red(f"server failed to start: {msg}"))
        srv.stop()
        return 1

    key_path = save_server_key(
        api_key, base_url,
        srv.proc.pid if srv.proc else 0, str(weights),
    ) or _server_key_path()

    print(green(f"ready at {base_url}"))
    print(dim(f"  log {log}"))
    print(dim(f"  api key stored in {key_path} (kept out of the child environment)"))
    print(dim("  pressing Ctrl-C stops the server"))
    try:
        while srv.proc and srv.proc.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        srv.stop()
    return 0


def _redact_key(argv: list[str], key: str) -> list[str]:
    """Never echo the API key into the terminal or a session log."""
    if not key:
        return list(argv)
    return [("<redacted>" if a == key else a) for a in argv]


def _make_backend(rt: RuntimeConfig, *, allow_mock: bool = False):
    from .backends import build_backend
    if not rt.api_key:
        rt.api_key = _load_server_key()
    return build_backend(rt, allow_mock=allow_mock)


def _human_gap(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"



def _schedule_runner(*, steps: int, context: int, thinking: str,
                     base_url: str = ""):
    """Build the callable the scheduler invokes for one tick.

    Each tick gets a *fresh* agent and a fresh context: a scheduled run is a new
    piece of work, and carrying the previous tick's transcript forward would grow
    without bound and confuse the model about what it has already done.
    """
    from .backends import build_backend
    from .loop import Agent
    from .prompt import wants_changes

    def run(schedule) -> tuple[bool, str]:
        workspace = Path(schedule.workspace) if schedule.workspace else \
            choose_workspace(None, create=True, interactive=False)
        rt = RuntimeConfig(context_window=context, max_steps=steps,
                           thinking=thinking)
        if base_url:
            rt.base_url = base_url
        cfg = HarnessConfig(workspace=workspace,
                            permissions=get_permissions(schedule.profile),
                            runtime=rt, permission_profile=schedule.profile)
        cfg.ensure_dirs()

        preamble = (
            f"[scheduled run #{schedule.runs}, every "
            f"{_human_gap(schedule.every_seconds)}]"
        )
        if schedule.missed:
            preamble += (f" NOTE: {schedule.missed} earlier run(s) were missed "
                         "while the machine was off; this is a catch-up, not a "
                         "backlog to work through.")
        task = f"{preamble}\n\n{schedule.task}"

        run_result = Agent(cfg, build_backend(rt)).run(task)
        ok = run_result.ok and bool((run_result.answer or "").strip())
        return ok, run_result.stop_reason

    return run


def cmd_schedule(args: argparse.Namespace) -> int:
    """Add, list, enable or remove recurring tasks."""
    import time as _time

    from .schedule import ScheduleStore, format_interval

    store = ScheduleStore()
    action = getattr(args, "schedule_command", "list")

    if action == "list":
        if not store.schedules:
            print("no schedules yet - add one with: qwen35 schedule add \"<task>\" --every 30m")
            return 0
        now = _time.time()
        for schedule in store.schedules:
            print("  " + schedule.summary())
            if schedule.enabled and not schedule.running_since:
                print(f"      next run in {_human_gap(schedule.seconds_until_due(now))}")
        return 0

    if action == "add":
        schedule = store.add(args.task, args.every,
                             workspace=args.workspace or "",
                             profile=args.profile or DEFAULT_PROFILE)
        print(f"added {schedule.id}: every "
              f"{format_interval(schedule.every_seconds)} -> {schedule.task}")
        print("run it in the foreground with: qwen35 watch")
        return 0

    if action == "rm":
        if store.remove(args.id):
            print(f"removed {args.id}")
            return 0
        print(f"no schedule matching {args.id!r}")
        return 1

    if action in ("on", "off"):
        schedule = store.get(args.id)
        if schedule is None:
            print(f"no schedule matching {args.id!r}")
            return 1
        schedule.enabled = action == "on"
        store.save()
        print(f"{schedule.id} {'enabled' if schedule.enabled else 'disabled'}")
        return 0

    print(f"unknown schedule action {action!r}")
    return 2


def cmd_watch(args: argparse.Namespace) -> int:
    """Work through the schedule list, forever (or once)."""
    from .schedule import ScheduleStore, format_interval, watch

    workspace = choose_workspace(args.workspace, create=True, interactive=False)
    store = ScheduleStore()

    if args.every and args.task:
        schedule = store.add(args.task, args.every, workspace=str(workspace),
                             profile=args.profile or DEFAULT_PROFILE)
        print(f"added {schedule.id}: every "
              f"{format_interval(schedule.every_seconds)}")

    if not store.schedules:
        print("nothing scheduled. Either add one first, or pass --every and "
              "--task to create one now.")
        return 2

    runner = _schedule_runner(steps=args.max_steps, context=args.context,
                              thinking=args.thinking,
                              base_url=args.base_url or "")
    print(bold(f"scheduler - {len(store.schedules)} schedule(s)"))
    for schedule in store.schedules:
        print("  " + schedule.summary())
    print(hr("-"))
    try:
        total = watch(store, runner,
                      stop_after=1 if args.once else None,
                      max_seconds=getattr(args, "max_seconds", None),
                      log=lambda m: print(m, flush=True))
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    print(f"\n{total} run(s) completed")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Measure prefill, prefix cache and generation on the running server."""
    from .bench import main as bench_main

    forwarded: list[str] = ["--base-url", args.base_url,
                            "--cold-tokens", str(args.cold_tokens),
                            "--grow-steps", str(args.grow_steps),
                            "--gen-tokens", str(args.gen_tokens)]
    if args.api_key:
        forwarded += ["--api-key", args.api_key]
    if getattr(args, "json_out", None):
        forwarded += ["--json", args.json_out]
    return bench_main(forwarded)


def cmd_ingest(args: argparse.Namespace) -> int:
    """Convert files and show what the model will actually receive."""
    from .ingest import build_preamble, ingest_many

    ws = choose_workspace(args.workspace, create=True, interactive=not args.yes)
    print(hr("="))
    print(f"{bold('workspace')} {ws}")
    print(f"{bold('files')}     {len(args.files)}")
    print(hr("="))
    atts = ingest_many(args.files, ws)
    for a in atts:
        mark = red("FAIL") if a.kind == "unsupported" else green("ok  ")
        print(f"  {mark} {a.source.name}")
        print(f"       {a.summary_line()}")
        if a.out_rel:
            print(f"       转换后 -> {a.out_rel}  ({a.stored_lines:,} 行)")
        for img in a.images[:3]:
            print(f"       图片   -> {img}")
        for w in a.warnings:
            print(f"       注意   {w}")
    if args.show:
        print(hr("-"))
        print(build_preamble([a for a in atts if a.kind != "unsupported"]))
    return 0


def _attach_to_task(args: argparse.Namespace, ws: Path,
                    task: str) -> tuple[str, list[Any]]:
    """Compose the task with its converted attachments.

    Returns the composed text *and* the attachments, because the agent needs the
    latter to redirect a model that reaches for an original binary file.
    """
    paths = getattr(args, "attach", None)
    if not paths:
        return task, []
    from .ingest import compose_task, ingest_many, inline_budget_for

    atts = ingest_many(paths, ws,
                       inline_tokens=inline_budget_for(getattr(args, "context", 0)))
    for a in atts:
        mark = "ok  " if a.kind != "unsupported" else "FAIL"
        print(f"  [{mark}] {a.summary_line()}")
        for w in a.warnings:
            print(f"         注意: {w}")
    return compose_task(task, atts), atts


def cmd_run(args: argparse.Namespace) -> int:
    from .loop import Agent
    from .backends import BackendError

    interactive = not args.yes
    ws = choose_workspace(args.workspace, create=args.create, interactive=interactive)
    cfg = build_config(args, ws, interactive=interactive)
    cfg.ensure_dirs()
    if args.save:
        save_config(cfg)
        print(dim(f"config saved to {config_path(ws)}"))

    task = " ".join(args.task).strip()
    if not task:
        if not sys.stdin.isatty():
            task = sys.stdin.read().strip()
    run_attachments: list[Any] = []
    if getattr(args, "attach", None):
        print(bold("附件"))
        task, run_attachments = _attach_to_task(args, ws, task)
    if not task:
        print(red("No task given."))
        return 2

    try:
        backend = _make_backend(cfg.runtime, allow_mock=args.backend == "mock")
    except BackendError as exc:
        print(red(str(exc)))
        return 1

    print(hr("="))
    print(f"{bold('task')}       {task[:96]}")
    print(f"{bold('workspace')}  {ws}")
    print(f"{bold('backend')}    {backend.describe()}")
    print(f"{bold('permissions')} {cfg.permission_profile}")
    print(f"{bold('context')}    {cfg.runtime.context_window:,} tokens "
          f"(model max {QWEN35_08B.context_window:,})")
    print(f"{bold('tools')}      {', '.join(s.name for s in _tool_names(cfg))}")
    print(hr("="))

    renderer = _Renderer(verbose=args.verbose, quiet=args.quiet)
    recorder = None if args.no_session else SessionRecorder.open(ws, task)
    agent = Agent(cfg, backend, tool_names=args.tools.split(",") if args.tools else None,
                  on_event=renderer.on_event, recorder=recorder,
                  attachments=run_attachments)
    try:
        result = agent.run(task)
    except KeyboardInterrupt:
        agent.stop()
        print(yellow("\ninterrupted"))
        return 130

    print(hr("="))
    print(bold("result"))
    print(f"  stop      {result.stop_reason}")
    print(f"  steps     {len(result.steps)}  ({result.tool_calls} tool calls)")
    print(f"  tokens    {result.tokens_in:,} in / {result.tokens_out:,} out")
    if result.tokens_in:
        print(f"  cache     {result.cached_in:,}/{result.tokens_in:,} prompt tokens "
              f"served from the prefix cache ({result.cache_hit_ratio * 100:.1f}%)")
    print(f"  elapsed   {result.elapsed:.1f}s")
    if result.answer:
        print(hr("-"))
        print(result.answer)
    print(hr("="))
    if recorder:
        print(dim(f"session: {recorder.path}"))
    return 0 if result.ok else 1


def _tool_names(cfg: HarnessConfig):
    from .tools import build_specs
    return build_specs(None, cfg.permissions)


def cmd_chat(args: argparse.Namespace) -> int:
    from .loop import Agent
    from .backends import BackendError

    ws = choose_workspace(args.workspace, create=args.create, interactive=True)
    cfg = build_config(args, ws, interactive=True)
    cfg.ensure_dirs()

    try:
        backend = _make_backend(cfg.runtime, allow_mock=args.backend == "mock")
    except BackendError as exc:
        print(red(str(exc)))
        return 1

    print(hr("="))
    print(bold("  qwen35 interactive"))
    print(f"  workspace    {ws}")
    print(f"  backend      {backend.describe()}")
    print(f"  permissions  {cfg.permission_profile}")
    print(f"  context      {cfg.runtime.context_window:,} tokens")
    print(dim("  /exit  /tools  /context  /sessions  /attach  /help"))
    print(hr("="))

    renderer = _Renderer(verbose=args.verbose, quiet=args.quiet)
    pending_attach: list[str] = list(getattr(args, "attach", None) or [])
    while True:
        try:
            line = input(bold("\ntask> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/exit", "/quit", "/q"):
            break
        if line == "/attach":
            raw = input("  文件路径（可多个，空格分隔）> ").strip()
            if raw:
                import shlex

                try:
                    paths = shlex.split(raw)
                except ValueError:
                    paths = raw.split()
                pending_attach.extend(paths)
                print(f"  已记录 {len(paths)} 个附件，将在下一条消息中转换")
            continue
        if line == "/help":
            print("  /exit      leave\n  /tools     list tools\n"
                  "  /context   show the last context budget\n"
                  "  /sessions  list recorded sessions\n"
                  "  /attach    attach files to the next message "
                  "(PDF/Word/Excel/PPT/image/text)")
            continue
        if line == "/tools":
            for s in _tool_names(cfg):
                print(f"  {s.name}({', '.join(s.params)})")
            continue
        if line == "/sessions":
            for p in list_sessions(ws):
                print(f"  {p.name}")
            continue
        if line == "/context":
            print(f"  window {cfg.runtime.context_window:,} tokens")
            continue

        task_text = line
        chat_attachments: list[Any] = []
        if pending_attach:
            print(bold("附件"))
            task_text, chat_attachments = _attach_to_task(
                argparse.Namespace(attach=list(pending_attach)), ws, line)
            pending_attach = []

        recorder = SessionRecorder.open(ws, task_text)
        agent = Agent(cfg, backend, on_event=renderer.on_event, recorder=recorder,
                      attachments=chat_attachments)
        try:
            result = agent.run(task_text)
        except KeyboardInterrupt:
            agent.stop()
            print(yellow("interrupted"))
            continue
        print(hr("-"))
        if result.answer:
            print(result.answer)
        print(dim(f"  [{result.stop_reason} | {len(result.steps)} steps | "
                  f"{result.tokens_in + result.tokens_out:,} tokens | "
                  f"{result.elapsed:.1f}s]"))
    return 0


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
def cmd_gui(args: argparse.Namespace) -> int:
    """Open the standalone window.  Thin wrapper over :mod:`qwen35.gui`."""
    from . import gui

    forwarded: list[str] = [
        "--context", str(args.context),
        "--profile", args.profile,
        "--port", str(args.port),
    ]
    if args.workspace:
        forwarded += ["--workspace", args.workspace]
    if args.model_dir:
        forwarded += ["--model-dir", args.model_dir]
    if getattr(args, "no_server", False):
        forwarded.append("--no-server")
    if getattr(args, "no_manage", False):
        forwarded.append("--no-manage")
    return gui.main(forwarded)


# --------------------------------------------------------------------------
# Live renderer
# --------------------------------------------------------------------------
class _Renderer:
    """Prints loop events as they happen.  Never raises."""

    def __init__(self, *, verbose: bool = False, quiet: bool = False) -> None:
        self.verbose = verbose
        self.quiet = quiet

    def on_event(self, event: str, data: dict[str, Any]) -> None:
        try:
            self._render(event, data)
        except Exception:
            pass

    def _render(self, event: str, data: dict[str, Any]) -> None:
        def first_line(text: str, limit: int = 150) -> str:
            line = str(text).strip().splitlines()[0] if str(text).strip() else ""
            return line[:limit]

        if event == "start":
            if not self.quiet:
                print(dim(f"  tools: {', '.join(data.get('tools', []))}"))
        elif event == "thinking_start":
            if self.verbose:
                flags = []
                if data.get("grammar"):
                    flags.append("grammar")
                if data.get("thinking"):
                    flags.append("thinking")
                print(dim(f"  step {data['step']} [{', '.join(flags) or 'plain'}]..."))
            else:
                print(dim(f"  step {data['step']} ..."), end="", flush=True)
        elif event == "tool_call":
            if not self.verbose:
                print()
            print(f"  {cyan('->')} {data['call']}")
        elif event == "tool_ok":
            if self.verbose:
                print(dim(f"     ok ({data.get('chars', 0)} chars)"))
        elif event == "tool_error":
            # A command that ran and exited non-zero (a failing test, say) is
            # information, not a harness error -- marking it "!!" made a normal
            # red test look like a tool malfunction.
            exit_code = data.get("exit")
            text = first_line(data.get("text", ""))
            if exit_code is not None and exit_code != 0 and not data.get("refused"):
                print(f"  {dim('--')} {text} {dim(f'(exit {exit_code})')}")
            else:
                print(f"  {yellow('!!')} {text}")
        elif event == "format_error":
            retry = data.get("retry")
            print(f"  {yellow('??')} {first_line(data.get('problem', ''))} "
                  f"{dim(f'(retry {retry})')}")
        elif event == "degenerate":
            print(f"  {yellow('~~')} model repeated itself; asking again")
        elif event == "forced_finish":
            print(f"  {dim('~~')} model would not stop; concluding with its answer")
        elif event == "loop_stop":
            print(f"  {red('##')} repeated the same action "
                  f"{data.get('repeats')} times; stopping")
        elif event == "context_full":
            print(f"  {yellow('!!')} context budget full "
                  f"({data.get('tokens')}/{data.get('budget')})")
        elif event == "error":
            print(f"  {red('!!')} {data.get('message')}")
        elif event == "answer":
            pass
        elif event == "done" and self.verbose:
            print(dim(f"  done: {data}"))


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qwen35",
        description="A lightweight, sandboxed agent harness built for "
                    "Qwen3.5-0.8B (vision, 262144-token context, hybrid "
                    "Gated-DeltaNet attention).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  qwen35 doctor\n"
            "  qwen35 install-runtime && qwen35 fetch\n"
            "  qwen35 serve\n"
            "  qwen35 run --workspace ./myproject \"add tests for utils.py\"\n"
            "  qwen35 chat --workspace ./myproject\n"
            "  qwen35 grammar --thinking off\n"
        ),
    )
    p.add_argument("--version", action="version", version="qwen35 1.0.0")
    sub = p.add_subparsers(dest="command")

    def add_common(sp: argparse.ArgumentParser, *, workspace: bool = True,
                   model: bool = True) -> None:
        if workspace:
            sp.add_argument("--workspace", "-w", help="workspace directory")
            sp.add_argument("--create", action="store_true",
                            help="create the workspace if missing")
        sp.add_argument("--profile", choices=PROFILE_CHOICES, default=None,
                        help="permission profile (default: safe)")
        if model:
            sp.add_argument("--backend", default=None,
                            choices=["auto", "llamacpp", "openai", "transformers", "mock"])
            sp.add_argument("--base-url", dest="base_url", default=None)
            sp.add_argument("--api-key", dest="api_key", default=None)
            sp.add_argument("--model-dir", dest="model_dir", default=None)
            sp.add_argument("--gguf", default=None)
            sp.add_argument("--mmproj", default=None)
            sp.add_argument("--context", type=int, default=None,
                            help="working context window (model max 262144)")
            sp.add_argument("--temperature", type=float, default=None)
            sp.add_argument("--top-p", dest="top_p", type=float, default=None)
            sp.add_argument("--top-k", dest="top_k", type=int, default=None)
            sp.add_argument("--max-tokens", dest="max_tokens", type=int, default=None)
            sp.add_argument("--max-steps", dest="max_steps", type=int, default=None)
            sp.add_argument("--thinking", choices=["on", "off", "auto"], default=None)
            sp.add_argument("--grammar", choices=["on", "off", "auto"], default=None)
            sp.add_argument("--tool-prompt", dest="tool_prompt",
                            choices=["auto", "grammar", "tools"], default=None)
            sp.add_argument("--kv-cache-type", dest="kv_cache_type",
                            choices=["f16", "q8_0", "q4_0"], default=None)
            sp.add_argument("--seed", type=int, default=None)
            sp.add_argument("--keep-recent-turns", dest="keep_recent_turns",
                            type=int, default=None)

    sp = sub.add_parser("doctor", help="report what is available")
    sp.add_argument("--model-dir", dest="model_dir", default=None)
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("workspaces", help="show and validate workspace choices")
    sp.set_defaults(func=cmd_workspaces)

    sp = sub.add_parser("tools", help="list tools and the generated grammar")
    sp.add_argument("--profile", choices=PROFILE_CHOICES, default=DEFAULT_PROFILE)
    sp.set_defaults(func=cmd_tools)

    sp = sub.add_parser("grammar", help="print the GBNF grammar")
    sp.add_argument("--profile", choices=PROFILE_CHOICES, default=DEFAULT_PROFILE)
    sp.add_argument("--thinking", choices=["on", "off"], default="on")
    sp.set_defaults(func=cmd_grammar)

    sp = sub.add_parser("install-runtime", help="download the llama.cpp CPU build")
    sp.add_argument("--tag", default=None)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_install_runtime)

    sp = sub.add_parser("fetch", help="download Qwen3.5-0.8B GGUF weights")
    sp.add_argument("--model-dir", dest="model_dir", default=None)
    sp.add_argument("--repo", default="unsloth/Qwen3.5-0.8B-GGUF")
    sp.add_argument("--mirror", default="https://hf-mirror.com")
    sp.add_argument("--quant", default="Qwen3.5-0.8B-Q4_K_M.gguf")
    sp.add_argument("--mmproj", default="mmproj-F16.gguf")
    sp.add_argument("--no-vision", action="store_true")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("serve", help="start llama-server for this model")
    sp.add_argument("--model-dir", dest="model_dir", default=None)
    sp.add_argument("--gguf", default=None)
    sp.add_argument("--mmproj", default=None)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--context", type=int, default=32768)
    sp.add_argument("--threads", type=int, default=None)
    sp.add_argument("--kv-cache-type", dest="kv_cache_type", default="q8_0")
    sp.add_argument("--no-flash-attn", dest="no_flash_attn", action="store_true")
    sp.add_argument("--api-key", dest="api_key", default=None)
    sp.add_argument("--timeout", type=float, default=300.0)
    sp.add_argument("--mtp", action="store_true",
                    help="self-speculative decoding via the model's MTP head")
    sp.add_argument("--mtp-draft-max", dest="mtp_draft_max", type=int, default=6)
    sp.add_argument("--image-min-tokens", dest="image_min_tokens", type=int,
                    default=1024, help="Qwen-VL grounding needs >=1024")
    sp.add_argument("--force-restart", dest="force_restart", action="store_true",
                    help="if something unusable already holds the port, stop it "
                         "and take the port back")
    sp.add_argument("--ctx-checkpoints", dest="ctx_checkpoints", type=int,
                    default=0,
                    help="context checkpoints per slot (default 0 = off). "
                         "Measured on this machine: enabling them made the "
                         "post-compaction re-prefill slower (17.2s -> 19.4s) and "
                         "cost 350-650 MB, so they are off by default.")
    sp.add_argument("--checkpoint-min-step", dest="checkpoint_min_step", type=int,
                    default=512,
                    help="minimum spacing between checkpoints, in tokens "
                         "(llama.cpp defaults to 8192, which a normal run never "
                         "reaches)")
    sp.add_argument("--no-mmproj-offload", dest="no_mmproj_offload",
                    action="store_true", default=True)
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("run", help="run one task")
    add_common(sp)
    sp.add_argument("task", nargs="*", help="the task (or pipe it on stdin)")
    sp.add_argument("--tools", default=None, help="comma-separated tool subset")
    sp.add_argument("--attach", action="append", default=None,
                    help="attach a file to the task (repeatable); PDF, Word, "
                         "Excel, PowerPoint, image, text, CSV, JSON")
    sp.add_argument("--yes", "-y", action="store_true",
                    help="non-interactive: never prompt for a workspace")
    sp.add_argument("--save", action="store_true", help="persist settings for this workspace")
    sp.add_argument("--no-session", dest="no_session", action="store_true")
    sp.add_argument("--verbose", "-v", action="store_true")
    sp.add_argument("--quiet", "-q", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("chat", help="interactive session")
    add_common(sp)
    sp.add_argument("--attach", action="append", default=None,
                    help="attach files to the first message (repeatable)")
    sp.add_argument("--verbose", "-v", action="store_true")
    sp.add_argument("--quiet", "-q", action="store_true")
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("ingest", help="convert files and show what the model sees")
    sp.add_argument("files", nargs="+", help="files to convert")
    sp.add_argument("--workspace", "-w", default=None)
    sp.add_argument("--create", action="store_true")
    sp.add_argument("--show", action="store_true",
                    help="print the prompt block the model would receive")
    sp.add_argument("--yes", "-y", action="store_true")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("gui", help="open the standalone-window GUI")
    sp.add_argument("--workspace", "-w", default=None,
                    help="workspace directory (root/system dirs are refused)")
    sp.add_argument("--model-dir", dest="model_dir", default=None)
    sp.add_argument("--context", type=int, default=32768)
    sp.add_argument("--profile", choices=PROFILE_CHOICES, default=DEFAULT_PROFILE)
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--no-server", dest="no_server", action="store_true",
                    help="do not start the model server automatically")
    sp.add_argument("--no-manage", dest="no_manage", action="store_true",
                    help="never stop the model server on exit")
    sp.set_defaults(func=cmd_gui)

    sp = sub.add_parser("bench", help="measure prefill, prefix cache and generation")
    sp.add_argument("--base-url", dest="base_url", default="http://127.0.0.1:8080")
    sp.add_argument("--api-key", dest="api_key", default="")
    sp.add_argument("--cold-tokens", dest="cold_tokens", type=int, default=1500)
    sp.add_argument("--grow-steps", dest="grow_steps", type=int, default=8)
    sp.add_argument("--gen-tokens", dest="gen_tokens", type=int, default=128)
    sp.add_argument("--json", dest="json_out", default=None)
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("schedule", help="add / list / remove recurring tasks")
    ssub = sp.add_subparsers(dest="schedule_command")
    sc = ssub.add_parser("list", help="show every schedule")
    sc.set_defaults(func=cmd_schedule)
    sc = ssub.add_parser("add", help="add a recurring task")
    sc.add_argument("task", help="what to do each time")
    sc.add_argument("--every", "-e", required=True,
                    help="interval: 30m, 2h, 90m (bare numbers are minutes)")
    sc.add_argument("--workspace", "-w", default=None)
    sc.add_argument("--profile", choices=PROFILE_CHOICES, default=None)
    sc.set_defaults(func=cmd_schedule)
    sc = ssub.add_parser("rm", help="remove a schedule")
    sc.add_argument("id")
    sc.set_defaults(func=cmd_schedule)
    for name in ("on", "off"):
        sc = ssub.add_parser(name, help=f"turn a schedule {name}")
        sc.add_argument("id")
        sc.set_defaults(func=cmd_schedule)
    sp.set_defaults(func=cmd_schedule, schedule_command="list")

    sp = sub.add_parser("watch", help="run scheduled tasks on a timer")
    sp.add_argument("--every", "-e", default=None,
                    help="create a schedule on the fly: 30m, 2h, ...")
    sp.add_argument("--task", default=None, help="the task for --every")
    sp.add_argument("--once", action="store_true",
                    help="run whatever is due now, then exit")
    sp.add_argument("--for", dest="max_seconds", type=float, default=None,
                    help="stop after this many seconds")
    sp.add_argument("--workspace", "-w", default=None)
    sp.add_argument("--profile", choices=PROFILE_CHOICES, default=None)
    sp.add_argument("--context", type=int, default=32768)
    sp.add_argument("--max-steps", dest="max_steps", type=int, default=20)
    sp.add_argument("--thinking", choices=["on", "off", "auto"], default="auto")
    sp.add_argument("--base-url", dest="base_url", default=None)
    sp.set_defaults(func=cmd_watch)



    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print()
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # a crash must still be a readable message
        if os.environ.get("QWEN35_DEBUG"):
            raise
        print(red(f"error: {type(exc).__name__}: {exc}"))
        print(dim("set QWEN35_DEBUG=1 for a traceback"))
        return 1
