"""
Prompt construction.

The important idea here: **do not hand-write the tool documentation.**

Qwen3.5's own chat template already knows how to render a tool list, and it
renders it in exactly the format the model was trained on::

    <|im_start|>system
    # Tools

    You have access to the following functions:

    <tools>
    {"type": "function", "function": {...}}
    </tools>

    If you choose to call a function ONLY reply in the following format ...
    <IMPORTANT>
    Reminder:
    - Function calls MUST follow the specified format ...
    </IMPORTANT>

    <our optional system content goes here>
    <|im_end|>

So the harness passes structured tool definitions through the backend, and keeps
its own system message to a short block of *operating rules* that the template
appends after the trained tool block.  Re-deriving the tool prose by hand would
put the model off its training distribution for no benefit.

The system prefix is built to be **byte-stable across turns** so that
llama.cpp's prefix cache keeps hitting.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "ToolSpec",
    "build_tool_schema",
    "system_rules",
    "few_shot_block",
    "render_tools_block",
    "build_system",
    "looks_conversational",
    "conversational_system",
    "conversational_example",
]


@dataclass(frozen=True)
class ToolSpec:
    """A tool, in both the OpenAI schema form and the GBNF signature form."""

    name: str
    description: str
    params: tuple[str, ...]
    required: tuple[str, ...]
    types: dict[str, str] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)

    def schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        for p in self.params:
            entry: dict[str, Any] = {
                "type": self.types.get(p, "string"),
            }
            if p in self.defaults:
                entry["default"] = self.defaults[p]
            props[p] = entry
        return {
            "type": "function",
            "function": {
                # Descriptions carry the load: Hammer showed small models are
                # misled by names and must be steered by prose.
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(self.required),
                },
            },
        }


def build_tool_schema(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [t.schema() for t in tools]


def system_rules(
    *,
    workspace_label: str = ".",
    workspace_root: str = "",
    max_read_lines: int = 100,
    permission_note: str = "",
) -> str:
    """The harness's own system content.

    Kept deliberately short.  Every line here must earn its tokens, because a
    0.8 B model has to hold all of it in mind at once, and because a stable
    prefix is what makes prefix caching work.  The workspace path is part of the
    prefix: it is fixed for the whole session, and stating it stops the model
    from guessing about its location and reaching for ``cd ..``.
    """
    where = f"`{workspace_label}`"
    lines = [
        "# How to work here",
        f"You are a coding and analysis agent. Your workspace is the folder {where}. "
        "You can read and write there and nowhere else.",
        # Deliberately NOT printing the full path here: telling the model its
        # absolute location made it prepend that path to every single tool
        # argument, which is both longer and unnecessary.  The ledger carries the
        # absolute path when the model actually needs to reason about location.
        "All paths you use are RELATIVE to that folder, e.g. `calc.py` or "
        "`src/main.py`. Do not write the full path.",
        "Paths outside the workspace, and `..`, are refused.",
        "Commands run in that folder already, so you never need to change "
        "directory.",
        "",
        "Rules:",
        # First, and loudest, on purpose.  Without this a 0.8 B model treats every
        # message as a work order and starts exploring the repository: measured at
        # 3-4 wasted tool calls and 60-80 seconds to answer "你好".
        "- If the message is a greeting, thanks, or a general question that needs "
        "no file access, answer it directly in plain prose with NO tool call. "
        "Do not explore the project to answer a greeting.",
        "- One tool call per message. Never more.",
        "- If you are not sure which files exist, call list_dir once before "
        "reading anything. Never guess a filename.",
        "- Look before you write: read a file before changing it.",
        "- Prefer `edit_file` for small changes; use `write_file` for new files.",
        "- Do the smallest next step, then look at the result.",
        "- If a tool returns an error, change your approach. Do not repeat the "
        "same call.",
        "- When the task is finished, reply in plain prose with your answer and "
        "no tool call. That ends the run.",
        f"- Read at most {max_read_lines} lines at a time.",
        "",
        "About run_command:",
        "- Commands already run in the workspace root. To run a script, name it "
        "directly: `python subdir/script.py`.",
        "- There is no shell. No pipes, no `&&`, no `;`, no redirection, no "
        "variables. The one exception is `cd subdir && <command>`.",
        "- To create or change a file use write_file or edit_file, never `>`.",
        "- To run tests, run the test file or the test runner directly, e.g. "
        "`python test_calc.py` or `python -m unittest discover`.",
        "",
        "Style:",
        "- Be brief. Short reasoning, then act.",
        "- Never invent file contents; read them.",
        "- If you cannot do something, say so plainly instead of guessing.",
    ]
    if permission_note:
        lines += ["", permission_note]
    return "\n".join(lines)


def few_shot_block() -> str:
    """Worked examples, shown once, as the first thing the model reads.

    Demonstrations beat instructions for a 0.8 B model, which is why the fix for
    "it explores the repository when you say hello" is a demonstration rather than
    another rule.  Example 1 is deliberately first: it is the pattern the model
    most needs to copy, and ordering measurably matters at this size.

    The examples ship as part of the stable prefix so they stay cacheable.
    """
    return (
        "Example 1 - a greeting needs NO tools. Just answer:\n"
        "  user: 你好\n"
        "  assistant: 你好！我是这个工作区里的编码助手。有什么需要我做的吗？\n"
        "  (no tool call was made, because no file access was needed)\n"
        "\n"
        "Example 2 - inspect, then edit:\n"
        "  <tool_call>\n"
        "  <function=read_file>\n"
        "  <parameter=path>\n"
        "  app.py\n"
        "  </parameter>\n"
        "  </function>\n"
        "  </tool_call>\n"
        "  ... tool returns line-numbered text ...\n"
        "  <tool_call>\n"
        "  <function=edit_file>\n"
        "  <parameter=path>\n"
        "  app.py\n"
        "  </parameter>\n"
        "  <parameter=find>\n"
        "  return x\n"
        "  </parameter>\n"
        "  <parameter=replace>\n"
        "  return x + 1\n"
        "  </parameter>\n"
        "  </function>\n"
        "  </tool_call>\n"
        "\n"
        "Example 3 - finish:\n"
        "  The bug was an off-by-one in the loop bound; I fixed it and the test "
        "now passes. No tool call is needed because the task is done."
    )


# --------------------------------------------------------------------------
# Replicating the model's own tool block
# --------------------------------------------------------------------------
# llama-server refuses `tools` together with a custom `grammar`:
#
#     HTTP 400: "Cannot use custom grammar constraints with tools."
#
# That is a real fork in the road: the tools field gets us the model's
# training-distribution prompt, the grammar gets us decode-time guarantees, and
# the server will not give us both at once.
#
# The way out is that the tools field is doing nothing we cannot do ourselves:
# Qwen3.5's chat template renders it by string concatenation, and the template's
# `{% if tools %}` branch is reproduced below *verbatim*.  Sending that text as
# the system message instead of as the `tools` field produces a byte-identical
# prompt, leaves the server's tool parser out of the picture, and frees the
# `grammar` field for constrained decoding.
#
# The text below is copied from `chat_template.jinja` in Qwen/Qwen3.5-0.8B.
_TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
_TOOLS_FOOTER = (
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format "
    "with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal "
    "with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)


def render_tools_block(tools: Sequence[dict[str, Any]]) -> str:
    """Reproduce the chat template's tool block exactly.

    Jinja's ``tojson`` uses ``json.dumps`` with default separators, so the JSON
    lines below match what the template would have emitted.
    """
    if not tools:
        return ""
    parts = [_TOOLS_HEADER]
    for tool in tools:
        parts.append("\n")
        parts.append(json.dumps(tool, ensure_ascii=False))
    parts.append(_TOOLS_FOOTER)
    return "".join(parts)


def build_system(
    tools: Sequence[dict[str, Any]],
    *,
    workspace_label: str = ".",
    workspace_root: str = "",
    max_read_lines: int = 100,
    permission_note: str = "",
    embed_tools: bool = True,
) -> str:
    """Assemble the complete system message.

    ``embed_tools=True`` reproduces the template's tool block ourselves (so the
    ``grammar`` field stays free).  ``embed_tools=False`` leaves the tool block
    to the server's ``tools`` field.
    """
    rules = system_rules(
        workspace_label=workspace_label,
        workspace_root=workspace_root,
        max_read_lines=max_read_lines,
        permission_note=permission_note,
    )
    if embed_tools and tools:
        return f"{render_tools_block(tools)}\n\n{rules}"
    return rules


# --------------------------------------------------------------------------
# Conversational routing
# --------------------------------------------------------------------------
# A 0.8 B model has no notion of "this is just chit-chat".  Asked "你好" it
# treated the greeting as a work order: measured at 3-4 tool calls, 4 steps and
# 60-200 seconds, sometimes returning an *empty* answer after burning the whole
# token budget on reasoning.  Prompting alone made it better but not reliable, so
# the harness also routes the obvious cases.
#
# The test is deliberately narrow.  It is only consulted for the FIRST step, and a
# wrong guess costs one cheap step rather than a wrong answer -- the model may
# still call a tool, and the loop carries on normally.
_CONVERSATIONAL_MARKERS = (
    "你好", "您好", "在吗", "谢谢", "多谢", "再见", "你是谁", "你叫什么",
    "介绍一下你", "介绍一下自己", "早上好", "下午好", "晚上好", "晚安",
    "hi", "hello", "hey", "thanks", "thank you", "who are you", "good morning",
    "good evening", "bye",
)
#: Anything that smells like work.  Presence of any of these disables the
#: shortcut, which is the safe direction to fail in.
_ACTION_MARKERS = (
    # English
    "fix", "run", "test", "create", "write", "edit", "delete", "remove", "read",
    "add", "make", "build", "install", "refactor", "explain", "summar", "list",
    "show", "find", "search", "open", "check", "review", "rename", "move",
    "debug", "implement", "generate", "convert", "update", "change", "help me",
    "look", "see", "inspect", "analy", "why", "how",
    # Chinese
    "修", "跑", "运行", "测试", "创建", "新建", "写", "改", "删", "移除", "读",
    "添加", "做", "构建", "安装", "重构", "解释", "总结", "列", "显示", "找",
    "搜索", "打开", "检查", "审查", "重命名", "移动", "调试", "实现", "生成",
    "转换", "更新", "修改", "补", "看", "查", "帮", "如何", "怎么", "为什么",
)

_CODE_ISH = re.compile(r"[\\/`<>{}()\[\]|$]")
_FILE_EXT = re.compile(r"\.[A-Za-z0-9]{1,5}\b")

# --------------------------------------------------------------------------
# "The answer is already in the prompt"
# --------------------------------------------------------------------------
# When the full text of an attached file has been inlined, a tool call is pure
# overhead -- and a measured liability.  A real run with a 973-token spreadsheet
# already inlined called `list_dir` twice, tripped the duplicate breaker, and
# returned NO answer at all.  The prompt was not the problem: the rules say
# "If you are not sure which files exist, call list_dir once before reading
# anything", which is exactly what the model did.
#
# So the harness routes it, the same way greetings are routed (prompting alone
# proved unreliable there too).  These verbs mean the user wants the workspace
# *changed*, so tools stay available; anything else is a content question that
# the inlined text already answers.
_CHANGE_MARKERS = (
    # English
    "write", "create", "make a", "save", "export", "convert", "edit", "change",
    "modify", "update", "delete", "remove", "rename", "move", "run", "test",
    "install", "build", "fix", "refactor", "implement", "generate", "append",
    "replace", "add a", "add the", "copy to",
    # Chinese.  Bare 改/写/删 are included on purpose: the cost of guessing wrong
    # here is asymmetric.  Keeping tools when they were not needed costs one
    # wasted step; removing them when the user wanted a file changed makes the
    # task impossible.
    "写", "创建", "新建", "保存", "导出", "转换", "改", "修改", "更新", "删", "移除",
    "重命名", "移动", "运行", "跑", "测试", "安装", "构建", "修复", "重构", "实现",
    "生成", "加到", "追加", "替换", "复制到", "做一", "弄", "整理成",
)


_REQUEST_HEAD = "# Request"
_CONTENT_HEADS = ("# Extracted content", "# Files to read first", "# Attached files")


def extract_request(task: str) -> str:
    """The user's own words, with the attachment scaffolding removed.

    ``wants_changes`` must only ever look at what the *user* asked for.  Checking
    the whole composed message made the harness's own boilerplate match a marker:
    the attachment card says "already been **convert**ed to text", so every
    attachment question looked like a request to modify the workspace and the
    no-tool fast path never fired.
    """
    text = task or ""
    if _REQUEST_HEAD not in text:
        return text
    body = text.split(_REQUEST_HEAD, 1)[1]
    for head in _CONTENT_HEADS:
        if head in body:
            body = body.split(head, 1)[0]
    return body.strip()


def wants_changes(text: str) -> bool:
    """Does this request ask for the workspace to be modified?

    Fails safe: an unrecognised verb keeps the tools available, so the worst case
    is a normal tool-using run rather than a run that cannot write a file.
    """
    low = extract_request(text).lower()
    return any(marker in low for marker in _CHANGE_MARKERS)


def answers_from_prompt(task: str, attachments: Sequence[Any]) -> bool:
    """True when the attached text is already in the message and nothing needs
    changing -- i.e. the first step should offer no tools at all."""
    inlined = [a for a in (attachments or [])
               if getattr(a, "inlined", False) and getattr(a, "text", "")]
    if not inlined:
        return False
    return not wants_changes(task)


def looks_conversational(text: str, *, max_chars: int = 32) -> bool:
    """Is this message obvious small talk that needs no file access at all?"""
    t = (text or "").strip()
    if not t or len(t) > max_chars:
        return False
    low = t.lower()
    if _CODE_ISH.search(t) or _FILE_EXT.search(t):
        return False
    if any(marker in low for marker in _ACTION_MARKERS):
        return False
    return any(marker in low for marker in _CONVERSATIONAL_MARKERS)


def conversational_system(*, workspace_label: str = ".") -> str:
    """A deliberately tiny prompt for messages that need no tools.

    Two reasons it is tiny rather than the full system prompt:

    * the tool block is ~1 500 tokens and takes **12 seconds** to prefill on a
      four-core CPU, which is absurd for answering "你好";
    * describing six tools to a 0.8 B model is an invitation to use them.  A
      conversational turn is run with an empty tool list, so the model *cannot*
      call one -- a hard guarantee rather than a hint.
    """
    return (
        "# You are a helpful assistant\n"
        f"You are working in the folder `{workspace_label}`.\n"
        "This message is a greeting or a general question. Reply directly, in "
        "the user's language, in one or two short sentences.\n"
        "Do not list files. Do not describe tools. Do not show your reasoning. "
        "Just reply."
    )


def conversational_example() -> str:
    return (
        "Example:\n"
        "  user: 你好\n"
        "  assistant: 你好！我是这个工作区里的编码助手，可以读写文件、运行命令、"
        "看图。有什么需要我做的吗？"
    )
