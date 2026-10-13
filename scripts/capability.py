"""
Capability probe: have the model actually write small projects, then run them.

"Can it code?" is not answerable by reading its output -- a 0.8 B model writes
plausible code that does not run.  So every task here is graded by **executing
the artefact** and checking observable behaviour, not by asking whether the code
looks right.

The tasks are deliberately graded from trivial to genuinely hard, so the result
is a ceiling rather than a single pass/fail:

    1. hello        one file, print a string               (near-trivial)
    2. calculator   argv parsing + arithmetic + errors     (easy)
    3. wordcount    read a file, aggregate, sort, format   (easy-moderate)
    4. todo         multi-command CLI + JSON persistence   (moderate)
    5. package      several files + imports + a test suite (hard for 0.8 B)
    6. refactor     change existing working code correctly (hard)

    python scripts/capability.py --list
    python scripts/capability.py --all --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen35.config import (DEFAULT_PROFILE, HarnessConfig, RuntimeConfig,  # noqa: E402
                           get_permissions)
from qwen35.loop import Agent  # noqa: E402

SAMPLE_TEXT = (
    "the quick brown fox jumps over the lazy dog\n"
    "the dog barks and the fox runs away\n"
    "a quick brown dog jumps over a lazy fox\n"
)


@dataclass
class Probe:
    name: str
    prompt: str
    #: runs after the agent finishes; returns (ok, detail)
    verify: Callable[[Path], tuple[bool, str]]
    level: str = "easy"
    max_steps: int = 24
    seconds: float = 900.0
    files: dict[str, str] = field(default_factory=dict)
    note: str = ""


def _run(ws: Path, argv: list[str], timeout: float = 60.0) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(argv, cwd=str(ws), capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8", errors="replace",
                              env=env)
    except subprocess.TimeoutExpired:
        return 124, "(timed out)"
    except OSError as exc:
        return 125, str(exc)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _py(*args: str) -> list[str]:
    return [sys.executable, *args]


# --------------------------------------------------------------------------
# Verifiers: each one EXECUTES what the model produced
# --------------------------------------------------------------------------
def verify_hello(ws: Path) -> tuple[bool, str]:
    script = next((p for p in ("hello.py", "main.py", "app.py")
                   if (ws / p).is_file()), None)
    if script is None:
        return False, "no hello.py / main.py was created"
    code, out = _run(ws, _py(script))
    if code != 0:
        return False, f"{script} exited {code}: {out.strip()[:160]}"
    ok = "hello" in out.lower()
    return ok, f"ran, output={out.strip()[:90]!r}"


def verify_calculator(ws: Path) -> tuple[bool, str]:
    script = next((p for p in ("calc.py", "calculator.py", "main.py")
                   if (ws / p).is_file()), None)
    if script is None:
        return False, "no calculator script was created"
    problems: list[str] = []
    # the model is not told the exact CLI shape, so try the common ones
    forms = [
        ([script, "3", "+", "4"], "7"),
        ([script, "3+4"], "7"),
        ([script, "add", "3", "4"], "7"),
    ]
    worked = ""
    for args, want in forms:
        code, out = _run(ws, _py(*args))
        if code == 0 and want in out:
            worked = " ".join(args)
            break
        problems.append(f"{' '.join(args)} -> {code}: {out.strip()[:50]}")
    if not worked:
        return False, "no supported invocation produced 7: " + " | ".join(problems[:3])
    # division by zero must be handled, not crash with a traceback
    for args in ([script, "1", "/", "0"], [script, "1/0"],
                 [script, "div", "1", "0"]):
        code, out = _run(ws, _py(*args))
        if code == 0 or "Error" in out or "error" in out:
            return True, f"'{worked}' worked and 1/0 was handled"
    return False, f"'{worked}' worked but 1/0 crashed with a traceback"


def verify_wordcount(ws: Path) -> tuple[bool, str]:
    script = next((p for p in ("wc.py", "wordcount.py", "count.py", "main.py")
                   if (ws / p).is_file()), None)
    if script is None:
        return False, "no word-count script was created"
    code, out = _run(ws, _py(script, "sample.txt"))
    if code != 0:
        return False, f"exited {code}: {out.strip()[:160]}"
    # sample has 3 lines, 27 words; the number must actually appear
    numbers = {int(n) for n in re.findall(r"\b\d+\b", out)}
    if 27 in numbers or 26 in numbers:          # some count "fox/dog" variants
        return True, f"counted correctly ({sorted(numbers)[:5]} in output)"
    if 3 in numbers:
        return True, f"at least the line count is right: {out.strip()[:90]!r}"
    return False, f"wrong counts, saw {sorted(numbers)[:6]}: {out.strip()[:110]!r}"


def verify_todo(ws: Path) -> tuple[bool, str]:
    script = next((p for p in ("todo.py", "tasks.py", "main.py")
                   if (ws / p).is_file()), None)
    if script is None:
        return False, "no todo script was created"
    added: list[str] = []
    for form in (["add", "buy milk"], ["add", "buy", "milk"]):
        code, out = _run(ws, _py(script, *form))
        if code == 0:
            added = form
            break
    if not added:
        return False, f"could not add an item (exit {code}): {out.strip()[:100]}"
    code, out = _run(ws, _py(script, "list"))
    if "milk" not in out.lower():
        return False, f"'list' did not show the item: {out.strip()[:110]!r}"
    # persistence: a new process must still see it
    code2, out2 = _run(ws, _py(script, "list"))
    stored = any(ws.glob("*.json")) or any(ws.glob("*.txt")) or any(ws.glob("*.db"))
    if "milk" not in out2.lower():
        return False, "the item did not persist across processes"
    return True, f"add+list works, persisted ({'json/txt' if stored else 'in-memory'})"


def verify_package(ws: Path) -> tuple[bool, str]:
    """A multi-file project whose own tests must pass."""
    code, out = _run(ws, _py("-m", "unittest", "discover", "-v"), timeout=90)
    if code == 0 and "OK" in out:
        return True, "its own unittest suite passes"
    code2, out2 = _run(ws, _py("-m", "pytest", "-q"), timeout=90)
    if code2 == 0:
        return True, "its own pytest suite passes"
    # maybe there is an importable package with a demo instead
    modules = [p.stem for p in ws.glob("*.py") if p.stem not in ("test", "setup")]
    if modules:
        code3, out3 = _run(ws, _py("-c", f"import {modules[0]}; print('ok')"))
        if code3 == 0:
            return False, (f"{modules[0]} imports, but no passing test suite "
                           f"(unittest: {out.strip()[-90:]!r})")
    return False, f"no passing tests: unittest exit {code} {out.strip()[-110:]!r}"


def verify_refactor(ws: Path) -> tuple[bool, str]:
    """Existing code with a bug: does the fix actually fix it?"""
    code, out = _run(ws, _py("test_shapes.py"))
    if code == 0:
        return True, "the existing test suite passes after the change"
    return False, f"tests still fail (exit {code}): {out.strip()[-160:]!r}"


BUGGY_SHAPES = '''\
"""Geometry helpers. There is a bug: area() of a circle is wrong."""

import math


def rectangle_area(w, h):
    return w * h


def circle_area(r):
    # BUG: forgot to square the radius
    return math.pi * r


def triangle_area(b, h):
    return 0.5 * b * h
'''

SHAPES_TEST = '''\
import unittest

import shapes


class TestAreas(unittest.TestCase):
    def test_rectangle(self):
        self.assertEqual(shapes.rectangle_area(3, 4), 12)

    def test_circle(self):
        # pi * 2**2
        self.assertAlmostEqual(shapes.circle_area(2), 12.566370614359172, places=5)

    def test_triangle(self):
        self.assertEqual(shapes.triangle_area(4, 5), 10)


if __name__ == "__main__":
    unittest.main()
'''


def verify_algorithm(ws: Path) -> tuple[bool, str]:
    script = next((p for p in ("prime.py", "primes.py", "main.py")
                   if (ws / p).is_file()), None)
    if script is None:
        return False, "no prime.py was created"
    code, out = _run(ws, _py(script))
    if code != 0:
        return False, f"exited {code}: {out.strip()[:150]}"
    found = {int(n) for n in re.findall(r"\b\d+\b", out)}
    want = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
            53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
    if want <= found:
        return True, f"all 25 primes below 100 present"
    missing = sorted(want - found)
    if len(want & found) >= 20:
        return True, f"{len(want & found)}/25 primes, missing {missing[:6]}"
    return False, f"only {len(want & found)}/25 primes: {out.strip()[:100]!r}"


def verify_csv_stats(ws: Path) -> tuple[bool, str]:
    script = next((p for p in ("stats.py", "csv_stats.py", "analyze.py", "main.py")
                   if (ws / p).is_file()), None)
    if script is None:
        return False, "no stats script was created"
    code, out = _run(ws, _py(script, "data.csv"))
    if code != 0:
        return False, f"exited {code}: {out.strip()[:150]}"
    # column b is 10,20,30,40,50 -> sum 150, mean 30
    numbers = {int(n) for n in re.findall(r"\b\d+\b", out)}
    if 150 in numbers and (30 in numbers or "30.0" in out):
        return True, "sum 150 and mean 30 both reported"
    if 150 in numbers:
        return True, "sum 150 reported"
    return False, f"wrong figures, saw {sorted(numbers)[:8]}: {out.strip()[:110]!r}"


def verify_fetch(ws: Path) -> tuple[bool, str]:
    """Uses the network.

    The target is ``pypi.org``, not ``example.com``: that is a machine-specific
    choice, not a preference.  ``example.com`` resolves but its TCP connection
    times out on this host, so a probe against it fails for a reason that has
    nothing to do with the harness.  ``pypi.org`` was measured reachable
    (HTTP 200, 0.11 s) and is stable.
    """
    script = next((p for p in ("fetch.py", "download.py", "get.py", "main.py")
                   if (ws / p).is_file()), None)
    if script is None:
        return False, "no fetch script was created"
    code, out = _run(ws, _py(script), timeout=90)
    if code != 0:
        return False, (f"exited {code}: {out.strip()[:170]}")
    if "200" in out:
        return True, f"fetched a real URL, status 200: {out.strip()[:90]!r}"
    return False, f"no status code seen: {out.strip()[:130]!r}"


def verify_install(ws: Path) -> tuple[bool, str]:
    """Did it install a package AND manage to use it?

    The install must land inside the workspace -- that is the whole point of the
    environment redirection -- so both halves are checked.
    """
    pkgs = ws / ".qwen35" / "packages"
    installed = pkgs.is_dir() and any(pkgs.iterdir())
    script = next((p for p in ("run_cowsay.py", "use_cowsay.py", "demo.py",
                               "main.py", "cowsay_demo.py")
                   if (ws / p).is_file()), None)
    if script is None:
        return False, ("no script using the package was created "
                       f"(installed={installed})")
    code, out = _run(ws, _py(script), timeout=90)
    if code != 0:
        return False, (f"the script failed (installed={installed}): "
                       f"{out.strip()[:150]}")
    if installed and out.strip():
        return True, f"installed into .qwen35/packages and ran: {out.strip()[:70]!r}"
    if out.strip() and "cow" in out.lower():
        return True, f"ran, output={out.strip()[:70]!r}"
    return False, f"ran but output looks wrong: {out.strip()[:110]!r}"


def verify_write_tests(ws: Path) -> tuple[bool, str]:
    """Given a module with NO tests, write tests that pass."""
    produced = list(ws.glob("test_*.py")) + list(ws.glob("*_test.py")) + \
        list((ws / "tests").glob("*.py") if (ws / "tests").is_dir() else [])
    if not produced:
        return False, "no test file was created"
    code, out = _run(ws, _py("-m", "unittest", "discover", "-v"), timeout=90)
    if code == 0 and "OK" in out:
        return True, f"wrote {len(produced)} test file(s) and they pass"
    code2, out2 = _run(ws, _py("-m", "pytest", "-q"), timeout=90)
    if code2 == 0:
        return True, f"wrote {len(produced)} test file(s) and pytest passes"
    return False, (f"tests were written but fail (exit {code}): "
                   f"{out.strip()[-130:]!r}")


QUEUE_MODULE = '''\
"""A tiny FIFO queue. There are no tests for it yet."""


class Queue:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("pop from an empty queue")
        return self._items.pop(0)

    def peek(self):
        if not self._items:
            raise IndexError("peek at an empty queue")
        return self._items[0]

    def __len__(self):
        return len(self._items)
'''

CSV_DATA = "name,a,b\nx,1,10\ny,2,20\nz,3,30\nw,4,40\nv,5,50\n"


PROBES: list[Probe] = [
    Probe(
        name="hello",
        level="trivial",
        prompt=("新建一个 hello.py，运行后会打印 Hello, World!。做完运行一次确认。"),
        verify=verify_hello,
        note="one file, one print — the floor",
    ),
    Probe(
        name="calculator",
        level="easy",
        prompt=("写一个命令行计算器 calc.py：支持加减乘除，"
                "从命令行参数读两个数和运算符，打印结果；"
                "除以零要给出友好的错误提示而不是崩溃。写完自己跑一遍验证。"),
        verify=verify_calculator,
        note="argv parsing + arithmetic + error handling",
    ),
    Probe(
        name="wordcount",
        level="easy",
        prompt=("写一个 wc.py：接受一个文件名参数，统计行数和单词数并打印出来。"
                "工作区里有 sample.txt，用它验证。"),
        verify=verify_wordcount,
        files={"sample.txt": SAMPLE_TEXT},
        note="file IO + aggregation + formatting",
    ),
    Probe(
        name="todo",
        level="moderate",
        prompt=("写一个命令行待办清单 todo.py：支持 add <内容> 添加、list 列出、"
                "done <序号> 标记完成。条目要保存到文件里，重开程序还在。"
                "写完自己验证 add 和 list。"),
        verify=verify_todo,
        max_steps=30,
        note="subcommands + persistence — where small models start to fray",
    ),
    Probe(
        name="refactor",
        level="hard",
        prompt=("shapes.py 里的 circle_area 有问题：它没有对半径平方。"
                "请修好它，然后运行 test_shapes.py 确认全部通过。"),
        verify=verify_refactor,
        files={"shapes.py": BUGGY_SHAPES, "test_shapes.py": SHAPES_TEST},
        max_steps=20,
        note="find a bug in existing code, fix it, verify",
    ),
    Probe(
        name="package",
        level="hard",
        prompt=("写一个小 Python 包：mytool/__init__.py 里放一个 add(a, b) 和一个 "
                "reverse(text) 函数，再写 tests/test_mytool.py 用 unittest 测试它们。"
                "最后运行 python -m unittest discover 确认全部通过。"),
        verify=verify_package,
        max_steps=36,
        note="multi-file layout + imports + a passing test suite — the likely ceiling",
    ),
    # ---- round two: different skill axes ---------------------------------
    Probe(
        name="primes",
        level="easy",
        prompt=("写一个 prime.py，打印 100 以内的所有质数，每个一行。写完运行确认。"),
        verify=verify_algorithm,
        note="a small algorithm with an exactly checkable answer",
    ),
    Probe(
        name="csv-stats",
        level="moderate",
        prompt=("工作区里有 data.csv。写一个 stats.py：接受文件名参数，"
                "计算每一列数字的总和与平均值，并把结果打印出来。"
                "用它处理 data.csv 并自己验证。"),
        verify=verify_csv_stats,
        files={"data.csv": CSV_DATA},
        max_steps=26,
        note="parse a file, aggregate per column, format — data work",
    ),
    Probe(
        name="write-tests",
        level="moderate",
        prompt=("工作区里的 queue.py 还没有测试。请写一份 unittest 测试覆盖 "
                "push/pop/peek/len 以及空队列的报错行为，然后运行确认全部通过。"),
        verify=verify_write_tests,
        files={"queue.py": QUEUE_MODULE},
        max_steps=28,
        note="read existing code, design tests, make them pass",
    ),
    Probe(
        name="fetch-url",
        level="moderate",
        prompt=("写一个 fetch.py，用 Python 标准库下载 https://pypi.org ，"
                "打印 HTTP 状态码和响应内容的开头。然后运行它确认成功。"),
        verify=verify_fetch,
        max_steps=24,
        note="uses the NETWORK — newly allowed, previously impossible",
    ),
    Probe(
        name="install-package",
        level="hard",
        prompt=("用 pip 安装 cowsay 这个包（会装在工作区里，不需要管理员权限），"
                "然后写一个 run_cowsay.py 导入它并打印一句话，运行确认成功。"),
        verify=verify_install,
        max_steps=30,
        note="uses PACKAGE INSTALL + network — the new confined install path",
    ),
]


def run_probe(probe: Probe, *, port: int, verbose: bool) -> dict[str, Any]:
    from qwen35.backends import build_backend

    out: dict[str, Any] = {"name": probe.name, "level": probe.level, "passed": False,
                           "seconds": 0.0, "steps": 0, "stop_reason": "",
                           "verify": "", "tools": [], "error": "",
                           "tokens_in": 0, "tokens_out": 0, "answer": ""}
    tmp = Path(tempfile.mkdtemp(prefix=f"cap-{probe.name}-"))
    started = time.time()
    try:
        for name, body in probe.files.items():
            (tmp / name).write_text(body, encoding="utf-8")
        rt = RuntimeConfig(backend="llamacpp", model_dir="",
                           base_url=f"http://127.0.0.1:{port}",
                           context_window=32768, max_steps=probe.max_steps,
                           thinking="on")
        cfg = HarnessConfig(workspace=tmp, permissions=get_permissions("standard"),
                            runtime=rt, permission_profile="standard")
        cfg.ensure_dirs()
        events: list[tuple[str, dict[str, Any]]] = []
        agent = Agent(cfg, build_backend(rt),
                      on_event=lambda e, d: events.append((e, d)))
        result = agent.run(probe.prompt)

        out["seconds"] = round(time.time() - started, 1)
        out["steps"] = len(result.steps)
        out["stop_reason"] = result.stop_reason
        out["answer"] = (result.answer or "")[:600]
        out["tools"] = sorted({s.call.name for s in result.steps if s.call})
        out["tokens_in"] = result.tokens_in
        out["tokens_out"] = result.tokens_out

        ok, detail = probe.verify(tmp)
        out["passed"] = bool(ok)
        out["verify"] = detail
        if verbose:
            for name, data in events:
                if name == "tool_call":
                    print(f"      -> {data.get('call')}")
    except Exception as exc:  # noqa: BLE001
        import traceback

        out["error"] = f"{type(exc).__name__}: {exc}"
        out["answer"] = traceback.format_exc()[-300:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--json", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for probe in PROBES:
            print(f"  {probe.level:9} {probe.name:12} {probe.prompt[:52]!r}")
            print(f"  {'':9} {'':12} {probe.note}")
        return 0

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    selected = [p for p in PROBES if not only or p.name in only]
    print(f"capability probe: {len(selected)} project(s)")
    print("=" * 78)

    results: list[dict[str, Any]] = []
    for probe in selected:
        print(f"\n[{probe.level}] {probe.name}  -- {probe.note}")
        print(f"  task: {probe.prompt[:76]!r}")
        out = run_probe(probe, port=args.port, verbose=args.verbose)
        results.append(out)
        print(f"  {'PASS' if out['passed'] else 'FAIL'}  {out['seconds']}s  "
              f"{out['steps']} steps  tools={','.join(out['tools']) or '-'}  "
              f"{out['stop_reason'] or out['error']}")
        print(f"      verify: {out['verify'][:150]}")
        if not out["passed"] and out["answer"]:
            print(f"      answer: {out['answer'][:170]}")

    passed = sum(1 for r in results if r["passed"])
    print("\n" + "=" * 78)
    for level in ("trivial", "easy", "moderate", "hard"):
        rows = [r for r in results if r["level"] == level]
        if rows:
            print(f"  {level:9} {sum(1 for r in rows if r['passed'])}/{len(rows)}")
    print(f"RESULT  {passed}/{len(results)} projects produced working code")
    if results:
        total = sum(r["seconds"] for r in results)
        print(f"time    {total:.0f}s total, {total / len(results):.0f}s average")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"json    {args.json}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
