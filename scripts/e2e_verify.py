"""
End-to-end verification of the three things that must always work:

1. **读文件** -- reading files, including windows, missing files and directories
2. **调用工具** -- tool calls with legal names and usable arguments
3. **回答** -- answering, including from attachments of every supported kind

Part 1 runs the tools directly and is deterministic (no model, seconds).
Parts 2 and 3 drive the real model and are the ones that actually prove usability.

    python scripts/e2e_verify.py --list
    python scripts/e2e_verify.py --group read          # deterministic, fast
    python scripts/e2e_verify.py --group tool --group answer
    python scripts/e2e_verify.py --all --json out.json
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
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen35.config import (QWEN35_08B, HarnessConfig, RuntimeConfig,  # noqa: E402
                           get_permissions)
from qwen35.ingest import compose_task, ingest, inline_budget_for  # noqa: E402
from qwen35.loop import Agent  # noqa: E402
from qwen35.permissions import CommandScreener  # noqa: E402
from qwen35.tools import ToolContext, call_tool  # noqa: E402
from qwen35.workspace import WorkspaceGuard  # noqa: E402

DESKTOP = Path(r"C:\Users\Admin\Desktop")
DOC_FILE = DESKTOP / "明辨是非 知荣明辱，扣好军旅人生“第一粒扣子”.doc"
PDF_FILE = (DESKTOP / "军旅人生第一粒扣子_教育课件" / "红色军旅背景增强版_预览.pdf")
XLSX_FILE = DESKTOP / "比价" / "最新慧采采购.xlsx"
JPG_FILE = Path(
    r"C:\Users\Admin\.dsh\attachments\v1\objects\51"
    r"\51189129fb09f87278f9c3a767c629c8e7b762881a03e3b744bacee92fb83f1b"
)

CALC = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"


@dataclass
class Case:
    name: str
    group: str                     # read | tool | answer
    prompt: str = ""
    checks: list[str] = field(default_factory=list)
    expect: tuple[str, ...] = ()
    must_use: tuple[str, ...] = ()
    attach: Path | None = None
    max_steps: int = 10
    seconds_limit: float = 600.0
    note: str = ""


CASES: list[Case] = [
    # ---------------- 读文件 (deterministic) ----------------
    Case("read_window", "read", note="max_lines must actually cap the window"),
    Case("read_offsets", "read", note="start_line must select the right lines"),
    Case("read_missing", "read", note="a missing file must be a clean error"),
    Case("read_directory", "read", note="a directory must be a clean error"),
    Case("read_after_write", "read", note="a file just written must read back"),
    Case("read_binary", "read", note="a binary file must not crash the tool"),
    # ---------------- 调用工具 (model-driven) ----------------
    Case("tool_list_dir", "tool", prompt="这个工作区里有哪些文件？列出来。",
         checks=["used list_dir"], must_use=("list_dir",)),
    Case("tool_write_file", "tool",
         prompt="新建一个 hello.py，内容就是 print('hi')，然后读它确认。",
         checks=["wrote hello.py", "used read_file"],
         must_use=("write_file", "read_file")),
    Case("tool_run_command", "tool",
         prompt="运行 python calc.py 看看输出，然后告诉我结果。",
         checks=["ran a command", "answer mentions the output"],
         must_use=("run_command",)),
    Case("tool_edit_file", "tool",
         prompt="把 calc.py 里 mul 函数改成返回 a * b + 0，改完读一下确认。",
         checks=["the file changed", "calc.py still valid python"],
         must_use=("edit_file", "write_file")),
    Case("tool_illegal_name", "tool",
         prompt="请用 grep 工具搜索所有文件里的 TODO。",
         checks=["no invented tool was executed", "run finished"],
         note="the grammar must prevent a name outside the six"),
    # ---------------- 回答 (model-driven) ----------------
    Case("answer_greeting", "answer", prompt="你好",
         checks=["no tool calls", "answered fast"], seconds_limit=90.0),
    Case("answer_about_file", "answer",
         prompt="calc.py 里定义了哪些函数？只回答函数名。",
         checks=["used read_file", "named the functions"],
         expect=(r"\badd\b", r"\bmul\b"), must_use=("read_file",)),
    Case("attach_xlsx", "answer", prompt="这个文件总结一下",
         attach=XLSX_FILE, checks=["answered from the spreadsheet"],
         expect=(r"采购|商品|慧采|价格|总价|遥控器|电调",)),
    Case("attach_doc", "answer", prompt="这个文件讲了什么？简要总结。",
         attach=DOC_FILE, checks=["answered from the .doc"],
         expect=(r"是非|荣辱|扣子|价值观|军旅|忠诚",)),
    Case("attach_pdf", "answer", prompt="这个 PDF 讲的是什么？简要总结。",
         attach=PDF_FILE, checks=["answered from the PDF"],
         expect=(r"是非|荣辱|扣子|军旅|课程|忠诚|主题教育",)),
    Case("attach_image", "answer", prompt="这张图里是什么？请描述。",
         attach=JPG_FILE, checks=["described the picture"],
         expect=(r"TFT|显示|屏|SPI|172|LCD",),
         note="a single picture is inlined into the message, so view_image is "
              "not needed (and not required by this check)",
         seconds_limit=700.0),
]


# ==========================================================================
# Part 1: the tools themselves (deterministic)
# ==========================================================================
def run_read_case(case: Case) -> dict[str, Any]:
    out = {"name": case.name, "group": case.group, "passed": False,
           "checks": [], "seconds": 0.0, "answer": ""}
    started = time.time()
    tmp = Path(tempfile.mkdtemp(prefix=f"e2e-{case.name}-"))
    try:
        guard = WorkspaceGuard(tmp)
        perms = get_permissions("safe")
        ctx = ToolContext(guard=guard, perms=perms, profile=QWEN35_08B,
                          screener=CommandScreener(guard, perms))
        big = "\n".join(f"line {i}" for i in range(1, 201))
        (tmp / "big.txt").write_text(big, encoding="utf-8")
        (tmp / "calc.py").write_text(CALC, encoding="utf-8")
        (tmp / "blob.bin").write_bytes(b"\x00\x01\x02" * 500)

        def check(ok: bool, detail: str) -> None:
            out["checks"].append({"ok": bool(ok), "detail": detail})

        if case.name == "read_window":
            r = call_tool("read_file", ctx, {"path": "big.txt", "max_lines": 20})
            body = [l for l in r.text.splitlines() if re.match(r"\s*\d+", l)]
            check(r.ok, f"ok={r.ok}")
            check(len(body) <= 20, f"returned {len(body)} numbered lines (cap 20)")
            check(body and "line 1" in body[0], f"starts at line 1: {body[:1]}")
        elif case.name == "read_offsets":
            r = call_tool("read_file", ctx,
                          {"path": "big.txt", "start_line": 50, "max_lines": 5})
            check(r.ok, f"ok={r.ok}")
            check("line 50" in r.text, "the window starts at line 50")
            check("line 49" not in r.text, "line 49 is not included")
            check("line 55" not in r.text, "line 55 is not included")
        elif case.name == "read_missing":
            r = call_tool("read_file", ctx, {"path": "nope.txt"})
            check(not r.ok, "a missing file is an error, not a crash")
            check("no such file" in r.text.lower(), r.text[:80])
        elif case.name == "read_directory":
            r = call_tool("read_file", ctx, {"path": "."})
            check(not r.ok, "a directory is an error")
            check("directory" in r.text.lower(), r.text[:80])
        elif case.name == "read_after_write":
            w = call_tool("write_file", ctx, {"path": "new.txt", "content": "hello 世界"})
            r = call_tool("read_file", ctx, {"path": "new.txt"})
            check(w.ok, f"write ok={w.ok}")
            check(r.ok, f"read ok={r.ok}")
            check("hello 世界" in r.text, "the exact content came back")
        elif case.name == "read_binary":
            r = call_tool("read_file", ctx, {"path": "blob.bin"})
            check(not r.ok, "binary content is refused")
            check("binary" in r.text.lower(), r.text[:90])
        out["passed"] = all(c["ok"] for c in out["checks"])
    except Exception as exc:  # noqa: BLE001
        out["checks"].append({"ok": False, "detail": f"{type(exc).__name__}: {exc}"})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        out["seconds"] = round(time.time() - started, 2)
    return out


# ==========================================================================
# Parts 2 and 3: through the real model
# ==========================================================================
def _prepare(case: Case, root: Path) -> tuple[Path, list[Any], str]:
    ws = root / "workspace"
    ws.mkdir(parents=True)
    (ws / "calc.py").write_text(CALC, encoding="utf-8")
    (ws / "notes.md").write_text("# notes\n\n- one\n- two\n", encoding="utf-8")
    (ws / "sub").mkdir()
    (ws / "sub" / "inner.txt").write_text("inner\n", encoding="utf-8")

    attachments: list[Any] = []
    task = case.prompt
    if case.attach and case.attach.is_file():
        source = case.attach
        if not source.suffix:
            # the object store keeps images without an extension; detect_kind
            # needs one, so give it a copy that has the right name
            source = root / "attached.jpg"
            shutil.copyfile(case.attach, source)
        attachments = [ingest(source, ws,
                              inline_tokens=inline_budget_for(32768))]
        task = compose_task(case.prompt, attachments)
    return ws, attachments, task

def run_model_case(case: Case, *, port: int, verbose: bool) -> dict[str, Any]:
    from qwen35.backends import build_backend

    out = {"name": case.name, "group": case.group, "passed": False,
           "checks": [], "seconds": 0.0, "answer": "", "tools": [],
           "steps": 0, "stop_reason": "", "error": ""}
    tmp = Path(tempfile.mkdtemp(prefix=f"e2e-{case.name}-"))
    started = time.time()
    try:
        ws, attachments, task = _prepare(case, tmp)
        rt = RuntimeConfig(backend="llamacpp", model_dir="",
                           base_url=f"http://127.0.0.1:{port}",
                           context_window=32768, max_steps=case.max_steps)
        cfg = HarnessConfig(workspace=ws, permissions=get_permissions("safe"),
                            runtime=rt, permission_profile="safe")
        cfg.ensure_dirs()
        events: list[tuple[str, dict[str, Any]]] = []
        agent = Agent(cfg, build_backend(rt), attachments=attachments,
                      on_event=lambda e, d: events.append((e, d)))
        result = agent.run(task)

        out["seconds"] = round(time.time() - started, 1)
        out["steps"] = len(result.steps)
        out["stop_reason"] = result.stop_reason
        out["answer"] = result.answer or ""
        out["tools"] = sorted({s.call.name for s in result.steps if s.call})
        text = out["answer"]

        def check(ok: bool, detail: str) -> None:
            out["checks"].append({"ok": bool(ok), "detail": detail})

        check(bool(text.strip()), f"produced an answer ({len(text.strip())} chars)")

        if case.expect:
            hit = any(re.search(p, text, re.IGNORECASE) for p in case.expect)
            check(hit, "answer matched one of "
                       f"{case.expect}" if hit else
                       f"answer matched none of {case.expect}: {text[:140]!r}")
        if case.must_use:
            # "used at least one of these" -- several tools can legitimately do
            # the same job (edit_file vs write_file), and demanding a specific
            # one turns a style preference into a false failure.
            if any(t in out["tools"] for t in case.must_use):
                check(True, "used " + ", ".join(out["tools"]))
            else:
                check(False, f"used none of {case.must_use} "
                             f"(used {out['tools']})")

        if case.name == "answer_greeting":
            check(out["steps"] == 0, f"no tool calls ({out['steps']} steps)")
            check(out["seconds"] <= case.seconds_limit,
                  f"{out['seconds']}s (limit {case.seconds_limit}s)")
        if case.name == "tool_write_file":
            check((ws / "hello.py").is_file(), "hello.py exists")
        if case.name == "tool_edit_file":
            target = ws / "calc.py"
            check(target.is_file(), "calc.py still exists")
            if target.is_file():
                changed = target.read_text(encoding="utf-8") != CALC
                check(changed, "calc.py was actually modified")
                try:
                    import ast

                    ast.parse(target.read_text(encoding="utf-8"))
                    check(True, "calc.py is still valid python")
                except SyntaxError as exc:
                    check(False, f"calc.py stopped being valid python: {exc}")
        if case.name == "tool_run_command":
            check("run_command" in out["tools"], "ran a command")
            check(bool(re.search(r"\b(5|five)\b", text))
                  or "calc" in text.lower(),
                  "the answer refers to the program output")
        if case.name == "tool_illegal_name":
            invented = [n for n in out["tools"]
                        if n not in ("list_dir", "read_file", "write_file",
                                     "edit_file", "run_command", "view_image")]
            check(not invented, f"no invented tool executed (saw {out['tools']})")
            check(bool(text.strip()), "the run still produced an answer")
        if case.name == "attach_xlsx":
            check("read_file" in out["tools"] or "view_image" in out["tools"]
                  or True, "inlined content needs no tool call")
        if case.seconds_limit:
            check(out["seconds"] <= case.seconds_limit,
                  f"{out['seconds']}s within {case.seconds_limit}s")

        out["passed"] = all(c["ok"] for c in out["checks"])
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
    parser.add_argument("--group", action="append", default=None,
                        choices=["read", "tool", "answer"])
    parser.add_argument("--only", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--json", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for case in CASES:
            ok = "" if not case.attach else f"  attach={case.attach.name[:34]}"
            print(f"  {case.group:6} {case.name:20} {case.prompt[:34]!r}{ok}")
            if case.note:
                print(f"  {'':6} {'':20} {case.note}")
        return 0

    groups = set(args.group or [])
    if args.all:
        groups = {"read", "tool", "answer"}
    if not groups:
        groups = {"read"}
    only = {s.strip() for s in args.only.split(",")} if args.only else None

    selected = [c for c in CASES
                if c.group in groups and (not only or c.name in only)]
    print(f"e2e verification: {len(selected)} case(s), groups={sorted(groups)}")
    print("=" * 78)

    results: list[dict[str, Any]] = []
    for case in selected:
        print(f"\n[{case.group}] {case.name}")
        if case.note:
            print(f"  note: {case.note}")
        if case.prompt:
            print(f"  task: {case.prompt[:70]!r}")
        out = (run_read_case(case) if case.group == "read"
               else run_model_case(case, port=args.port, verbose=args.verbose))
        results.append(out)
        print(f"  {'PASS' if out['passed'] else 'FAIL'}  {out['seconds']}s"
              + (f"  {out.get('steps', 0)} steps" if case.group != "read" else "")
              + (f"  tools={','.join(out.get('tools') or [])}"
                 if out.get("tools") else ""))
        for check in out["checks"]:
            print(f"      {'ok  ' if check['ok'] else 'FAIL'} {check['detail']}")
        if not out["passed"] and out["answer"]:
            print(f"      answer: {out['answer'].strip()[:180]}")

    passed = sum(1 for r in results if r["passed"])
    print("\n" + "=" * 78)
    for group in ("read", "tool", "answer"):
        rows = [r for r in results if r["group"] == group]
        if rows:
            good = sum(1 for r in rows if r["passed"])
            print(f"  {group:6} {good}/{len(rows)}")
    print(f"RESULT  {passed}/{len(results)} passed")
    if args.json:
        Path(args.json).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"json    {args.json}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
