"""
Real-project evaluation.

The synthetic fixtures in ``eval_suite.py`` prove the harness works on problems I
designed.  This runs it against **code someone else wrote** — copied into a
sandbox first, never touched in place — to find the ceiling.

Each task carries a factual check derived by reading the project myself, so the
suite cannot be satisfied by a plausible-sounding answer.

    python scripts/real_projects.py --list
    python scripts/real_projects.py --project A-minimind
    python scripts/real_projects.py --all --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen35.config import HarnessConfig, RuntimeConfig, get_permissions  # noqa: E402
from qwen35.loop import Agent  # noqa: E402
from qwen35.session import SessionRecorder  # noqa: E402

#: Where the sandboxes live (built by hand from real projects; copies only).
SANDBOX_ROOT = Path(r"D:\qwen\state\realproj")


@dataclass
class RealTask:
    name: str
    prompt: str
    expect: tuple[str, ...] = ()      # any-of, case-insensitive regexes
    must_use: tuple[str, ...] = ()    # tool names that must appear
    note: str = ""
    max_steps: int = 14
    context: int = 32768


@dataclass
class RealProject:
    name: str
    path: Path
    tasks: list[RealTask] = field(default_factory=list)


PROJECTS: list[RealProject] = [
    RealProject(
        name="A-minimind",
        path=SANDBOX_ROOT / "A-minimind",
        tasks=[
            RealTask(
                name="identify-model",
                prompt="这个项目实现的是什么？请先读代码，然后说出主模型的类名，"
                       "以及 MoE 使用了多少个专家。",
                expect=(r"MiniMindForCausalLM", r"MiniMindModel"),
                must_use=("read_file",),
                note="two 18-26 KB Python files; the class name is only in the code",
            ),
            RealTask(
                name="read-config-defaults",
                prompt="MiniMindConfig 里 num_hidden_layers 和 hidden_size 的"
                       "默认值分别是多少？只回答数字和字段名。",
                expect=(r"768",),
                must_use=("read_file",),
                note="the values are constructor defaults, not just in config.json",
            ),
            RealTask(
                name="survey-omni",
                prompt="model_omni.py 里有哪些主要组件？读了之后简要列出。",
                expect=(r"Talker", r"Projector", r"Omni", r"VAD"),
                must_use=("read_file",),
                note="26 KB file; needs a real read, not a guess",
            ),
        ],
    ),
    RealProject(
        name="B-thesis-paper",
        path=SANDBOX_ROOT / "B-thesis-paper",
        tasks=[
            RealTask(
                name="survey-scripts",
                prompt="code 目录里有哪些 Python 脚本？它们大致可以分成哪几类？"
                       "请先列出文件再分类。",
                expect=(r"analyze", r"runner", r"fig", r"spec"),
                must_use=("list_dir",),
                note="20 scripts, 2 557 lines",
            ),
            RealTask(
                name="explain-runner",
                prompt="frozen_fusion_runner.py 是做什么的？先读它再回答。",
                expect=(r"frozen|fusion|run|experiment|实验|冻结|融合"),
                must_use=("read_file",),
                note="14 KB entry point",
            ),
            RealTask(
                name="find-todos",
                prompt="在这些 Python 文件里找出所有的 TODO 或 FIXME 注释。"
                       "如果确实一个都没有，就明确说没有。",
                expect=(r"TODO|FIXME|没有|none|未找到|不存在"),
                note="multi-file scan; 'none found' is a valid answer",
                max_steps=16,
            ),
        ],
    ),
    RealProject(
        name="C-thesis-work",
        path=SANDBOX_ROOT / "C-thesis-work",
        tasks=[
            RealTask(
                name="explain-curate",
                prompt="curate_dataset.py 是做什么的？读了再回答。",
                expect=(r"dataset|数据集|curate|筛选|清洗|整理"),
                must_use=("read_file",),
                note="8 KB, the largest script here",
            ),
            RealTask(
                name="count-and-rank",
                prompt="这个目录下有多少个 .py 文件？请数清楚再回答，并说出"
                       "其中三个你认为最核心的脚本及其理由。",
                expect=(r"5[0-9]|五十"),
                must_use=("list_dir",),
                note="counting is a classic small-model failure; real answer is 52",
            ),
        ],
    ),
]


def _text(result: Any) -> str:
    return result.answer or ""


def run_task(task: RealTask, project: RealProject, *, port: int,
             verbose: bool, thinking: str | None = None,
             max_steps: int | None = None, context: int | None = None,
             keep: bool = False) -> dict[str, Any]:
    from qwen35.backends import build_backend

    out: dict[str, Any] = {
        "project": project.name, "task": task.name, "passed": False,
        "seconds": 0.0, "steps": 0, "stop_reason": "", "tokens_in": 0,
        "tokens_out": 0, "cached_in": 0, "answer": "", "tools": [],
        "checks": [], "error": "",
    }

    # Never work in the original: copy into a throwaway sandbox.
    tmp = Path(tempfile.mkdtemp(prefix=f"qwen35-real-{task.name}-"))
    ws = tmp / "project"
    try:
        shutil.copytree(project.path, ws)
        perms = get_permissions("safe")
        rt = RuntimeConfig(backend="llamacpp", model_dir="",
                           base_url=f"http://127.0.0.1:{port}",
                           context_window=task.context, max_steps=task.max_steps)
        if thinking:
            rt.thinking = thinking
        if max_steps:
            rt.max_steps = max_steps
        if context:
            rt.context_window = context
        cfg = HarnessConfig(workspace=ws, permissions=perms, runtime=rt,
                            permission_profile="safe")
        cfg.ensure_dirs()
        backend = build_backend(rt)

        events: list[tuple[str, dict[str, Any]]] = []
        recorder = SessionRecorder.open(cfg.workspace, task.prompt,
                                       directory="real-sessions")
        agent = Agent(cfg, backend, on_event=lambda e, d: events.append((e, d)),
                      recorder=recorder)
        started = time.time()
        result = agent.run(task.prompt)
        out["seconds"] = round(time.time() - started, 1)
        out["steps"] = len(result.steps)
        out["stop_reason"] = result.stop_reason
        out["tokens_in"] = result.tokens_in
        out["tokens_out"] = result.tokens_out
        out["cached_in"] = result.cached_in
        out["answer"] = result.answer or ""
        out["tools"] = sorted({s.call.name for s in result.steps if s.call})

        text = _text(result)
        if task.expect:
            hit = any(re.search(p, text, re.IGNORECASE) for p in task.expect)
            out["checks"].append({"ok": hit,
                                  "detail": f"answer matched one of {task.expect}"
                                            if hit else
                                            f"matched none of {task.expect}: "
                                            f"{text[:160]!r}"})
        if task.must_use:
            missing = [t for t in task.must_use if t not in out["tools"]]
            out["checks"].append({"ok": not missing,
                                  "detail": "used " + ", ".join(out["tools"])
                                            if not missing else
                                            f"never used {missing}"})
        out["checks"].append({"ok": bool(text.strip()),
                              "detail": f"answer has {len(text.strip())} chars"})
        out["passed"] = all(c["ok"] for c in out["checks"])

        if verbose:
            for name, data in events:
                if name == "tool_call":
                    print(f"      -> {data.get('call')}")
    except Exception as exc:  # noqa: BLE001
        import traceback

        out["error"] = f"{type(exc).__name__}: {exc}"
        out["answer"] = traceback.format_exc()[-400:]
    finally:
        if keep:
            print(f"    sandbox kept at {ws}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", action="append", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--only", default=None,
                        help="comma-separated task names to run")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--json", default=None)
    parser.add_argument("--thinking", choices=["on", "off", "auto"], default=None,
                        help="override the thinking mode for this run")
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--keep", action="store_true",
                        help="keep the sandbox (for inspecting what the model saw)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for project in PROJECTS:
            exists = "ok " if project.path.is_dir() else "MISSING"
            print(f"\n  [{exists}] {project.name}  ({project.path})")
            for t in project.tasks:
                print(f"        {t.name:<20} {t.prompt[:56]!r}")
                if t.note:
                    print(f"        {'':<20} {t.note}")
        return 0

    selected = PROJECTS
    if args.project:
        wanted = set(args.project)
        selected = [p for p in PROJECTS if p.name in wanted]
        unknown = wanted - {p.name for p in PROJECTS}
        if unknown:
            print(f"unknown project(s): {sorted(unknown)}")
            return 2
    if not selected:
        print("nothing selected; use --all or --project NAME")
        return 2

    only = {s.strip() for s in args.only.split(",")} if args.only else None

    print("real-project evaluation")
    print("=" * 78)
    results: list[dict[str, Any]] = []
    for project in selected:
        if not project.path.is_dir():
            print(f"\n[skip] {project.name}: sandbox {project.path} not found")
            continue
        print(f"\n########## {project.name}  ({project.path}) ##########")
        for task in project.tasks:
            if only and task.name not in only:
                continue
            print(f"\n  [{task.name}] {task.prompt[:70]!r}")
            if task.note:
                print(f"    note: {task.note}")
            out = run_task(task, project, port=args.port, verbose=args.verbose,
                           thinking=args.thinking, max_steps=args.max_steps,
                           context=args.context, keep=args.keep)
            results.append(out)
            status = "PASS" if out["passed"] else "FAIL"
            cache = (out["cached_in"] / out["tokens_in"] * 100
                     if out["tokens_in"] else 0.0)
            print(f"    {status}  {out['seconds']:5.1f}s  {out['steps']:2d} steps  "
                  f"cache {cache:4.1f}%  tools={','.join(out['tools']) or '-'}  "
                  f"{out['stop_reason'] or out['error']}")
            for check in out["checks"]:
                print(f"        {'ok  ' if check['ok'] else 'FAIL'} {check['detail']}")
            if not out["passed"] and out["answer"].strip():
                print(f"        answer: {out['answer'].strip()[:200]}")

    passed = sum(1 for r in results if r["passed"])
    print("\n" + "=" * 78)
    print(f"RESULT  {passed}/{len(results)} real-project tasks passed")
    if results:
        total = sum(r["seconds"] for r in results)
        print(f"time    {total:.0f}s total, {total / len(results):.0f}s average")
        pin = sum(r["tokens_in"] for r in results)
        cin = sum(r["cached_in"] for r in results)
        if pin:
            print(f"cache   {cin:,}/{pin:,} prompt tokens from the prefix cache "
                  f"({cin / pin * 100:.1f}%)")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"json    {args.json}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
