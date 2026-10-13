"""
Parsing the model's turns.

Qwen3.5 was trained on a very specific XML envelope (see the model's
``chat_template.jinja``)::

    <tool_call>
    <function=read_file>
    <parameter=path>
    src/main.py
    </parameter>
    </function>
    </tool_call>

``<tool_call>``, ``</tool_call>``, ``<tool_response>`` and `` thinking`` are each a
*single* token in this vocabulary, which makes the envelope cheap to emit and
cheap to detect.

The grammar in :mod:`qwen35.grammar` makes malformed output impossible when it is
active.  The transformers backend has no grammar, so this parser is deliberately
tolerant -- and when it cannot recover, it produces a *precise* correction
message rather than a shrug.  Feedback quality is what lets a weak model recover;
silence or vagueness makes it repeat the same mistake.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ToolCall",
    "ParsedTurn",
    "parse_turn",
    "strip_think",
    "format_correction",
    "detect_degeneration",
]

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
THINK_CLOSE = "</think>"

# ---------------------------------------------------------------- envelopes
# Tolerant patterns: real models drop a newline, use = or a space, or quote it.
_TC_BLOCK = re.compile(
    r"<tool_call>(.*?)(?:</tool_call>|(?=<\|im_end\|>)|$)",
    re.DOTALL | re.IGNORECASE,
)
_FUNC = re.compile(
    r"<function\s*[=:]?\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?\s*>",
    re.IGNORECASE,
)
_PARAM = re.compile(
    r"<\s*parameter\s*[=:]?\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?\s*>\s*"
    r"(.*?)"
    # Tolerant of every near-miss a 0.8 B model actually produces.  Observed in
    # practice: "</parameter>", "</parameter >", "</parameter/>",
    # "</parameter />", and "</ parameter>" (whitespace straight after "</").
    # Missing one of these silently loses the argument, which is far worse than
    # accepting a slightly sloppy delimiter.
    r"\s*</\s*parameter\s*/?\s*>",
    re.DOTALL | re.IGNORECASE,
)
# a bare JSON body, for when the model ignores the XML format entirely
_JSON_BODY = re.compile(r"^\s*\{.*\}\s*$", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        """Stable identity for duplicate detection."""
        import hashlib
        import json

        blob = json.dumps({"n": self.name, "a": self.args}, sort_keys=True,
                          ensure_ascii=False, default=str)
        return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()[:16]

    def describe(self) -> str:
        bits = ", ".join(f"{k}={_short(v)}" for k, v in self.args.items())
        return f"{self.name}({bits})"


def _short(v: Any, limit: int = 60) -> str:
    s = str(v).replace("\n", "\\n")
    return s if len(s) <= limit else s[: limit - 1] + "\u2026"


@dataclass
class ParsedTurn:
    """One assistant turn, decomposed."""

    raw: str = ""
    thinking: str = ""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    format_error: str = ""
    extra_calls: int = 0

    @property
    def has_tool_call(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_final_answer(self) -> bool:
        return not self.tool_calls and bool(self.text.strip())

    @property
    def is_empty(self) -> bool:
        return not self.tool_calls and not self.text.strip()


def strip_think(raw: str) -> tuple[str, str]:
    """Split ``thinking ...  response`` off the front.  Returns (thinking, rest)."""
    if THINK_CLOSE in raw:
        head, _, tail = raw.partition(THINK_CLOSE)
        thinking = head
        if " thinking" in thinking:
            thinking = thinking.split(" thinking", 1)[1]
        return thinking.strip(), tail.lstrip("\n")
    if " thinking" in raw:
        # opened but never closed (truncated generation)
        head, _, _ = raw.partition(" thinking")
        return "", head.strip()
    return "", raw


def _coerce(value: str) -> Any:
    """Turn a raw parameter body into a Python value where that is obvious."""
    v = value.strip()
    if v.startswith("{") or v.startswith("["):
        import json

        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return v
    return v


def parse_turn(raw: str) -> ParsedTurn:
    """Decompose a raw completion into thinking, prose and tool calls."""
    turn = ParsedTurn(raw=raw or "")
    if not raw:
        turn.format_error = "the model produced no output at all"
        return turn

    thinking, body = strip_think(raw)
    turn.thinking = thinking

    blocks = list(_TC_BLOCK.finditer(body))
    text_before = body[: blocks[0].start()] if blocks else body
    text_after = body[blocks[-1].end():] if blocks else ""

    # anything the model says outside the tool_call blocks is prose
    turn.text = (text_before + "\n" + text_after).strip()

    if not blocks:
        if TOOL_CALL_OPEN in body:
            # opened a call and never closed it, and no </tool_call> anywhere
            turn.format_error = (
                "you opened <tool_call> but never closed it with </tool_call>"
            )
        return turn

    for block_index, m in enumerate(blocks):
        inner = m.group(1) or ""

        fm = _FUNC.search(inner)
        if not fm:
            if block_index == 0:
                # maybe the model emitted only JSON inside the envelope
                if _JSON_BODY.match(inner):
                    turn.format_error = (
                        "the tool call was JSON. This model uses XML: "
                        "<function=NAME> with <parameter=KEY> value </parameter>"
                    )
                else:
                    turn.format_error = (
                        "no <function=NAME> line was found inside <tool_call>"
                    )
            continue

        name = fm.group(1)
        args: dict[str, Any] = {}
        for pm in _PARAM.finditer(inner):
            args[pm.group(1)] = _coerce(pm.group(2))

        if block_index == 0:
            turn.tool_calls.append(ToolCall(name=name, args=args))
        else:
            turn.extra_calls += 1

    if turn.extra_calls:
        # the harness runs one action per step; say so explicitly
        turn.format_error = (
            turn.format_error
            or f"you emitted {turn.extra_calls + 1} tool calls; only the first was "
               "used. Issue exactly one tool call per message."
        )

    return turn


# ---------------------------------------------------------------- feedback
def _legal_template(tools: list[tuple[str, tuple[str, ...]]]) -> str:
    lines = ["<tool_call>", "<function=NAME>", "<parameter=KEY>", "value",
             "</parameter>", "</function>", "</tool_call>"]
    body = "\n".join(lines)
    if tools:
        listing = "; ".join(
            f"{n}({', '.join(p)})" if p else f"{n}()" for n, p in tools
        )
        return (
            "The only legal format is exactly this, with nothing after it:\n"
            f"{body}\n\n"
            f"Available tools: {listing}"
        )
    return f"The only legal format is exactly this:\n{body}"


def format_correction(
    turn: ParsedTurn,
    tools: list[tuple[str, tuple[str, ...]]] | None = None,
    reason: str = "",
) -> str:
    """Build a precise, actionable correction message.

    ToolScan's finding is that weak models *do* recover from malformed calls --
    but only when the feedback names the exact problem.  So this enumerates the
    legal template and quotes the offending output.
    """
    tools = tools or []
    problem = reason or turn.format_error or "the previous message could not be parsed"

    excerpt = (turn.raw or "").strip()
    if len(excerpt) > 400:
        excerpt = excerpt[:400] + "\u2026"

    parts = [
        "Your last message could not be understood.",
        f"Problem: {problem}",
        "",
        _legal_template(tools),
        "",
        "Your message was:",
        excerpt or "(empty)",
        "",
        "Reply with exactly one tool call in that format, or answer in plain "
        "prose if the task is finished.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------- degeneration
_WORD = re.compile(r"\S+")


def detect_degeneration(text: str, *, min_repeats: int = 4,
                        window: int = 24) -> bool:
    """Detect a model looping *inside a single completion*.

    A 0.8 B model can degenerate before it ever emits a tool call.  Observed in
    practice on this model, with thinking enabled: it produced

        **[list_dir](https://docs.python.org/3/library/files.html#list_dir) tool**

    seven times, padded with "---", until it hit the token cap.  Qwen's own model
    card warns the 0.8 B is prone to thinking loops, so this is expected
    behaviour rather than a bug in the prompt.

    Repetition is measured at three granularities, because a looping model does
    not always repeat at a fixed stride.
    """
    if not text:
        return False

    words = _WORD.findall(text)
    stripped = text.strip()

    # 1. character-level: the same opening chunk recurs
    if len(stripped) > 200:
        probe = stripped[:120]
        if stripped.count(probe) >= 3:
            return True

    # 2. raw substring repetition at a coarse stride, which catches loops that
    #    insert separators (e.g. "---") between repeats
    if len(stripped) > 300:
        for size in (60, 100, 160):
            probe = stripped[:size]
            if probe and stripped.count(probe) >= 3:
                return True

    if len(words) < 8:
        return False

    # 3. word n-gram repetition
    for n in (3, 5):
        if len(words) < n * min_repeats:
            continue
        seen: dict[tuple[str, ...], int] = {}
        for i in range(len(words) - n + 1):
            gram = tuple(words[i: i + n])
            seen[gram] = seen.get(gram, 0) + 1
            if seen[gram] >= min_repeats:
                return True

    # 4. a single token repeated
    run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            if run >= 6:
                return True
        else:
            run = 1
    return False
