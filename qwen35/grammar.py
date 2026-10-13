"""
GBNF grammar synthesis.

This is the single biggest reliability lever available for a 0.8 B model.

A 0.8 B model *will* eventually emit `<function read_file(...)>`, or JSON, or
forget ``</tool_call>``.  Arguing with it in the prompt does not work.  Instead we
compile the tool interface into a grammar and let the decoder enforce it, so
these failure modes become *impossible* rather than *unlikely*:

* the tool name must be one of the tools that actually exist
* the parameter names must belong to that tool
* the envelope is exactly the XML the model was trained on
* the value can be any text except the closing delimiter

Everything is derived from the tool registry, so the grammar can never drift out
of sync with the tools.  See :func:`build_grammar`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = ["build_grammar", "validate_grammar", "GrammarError", "ToolSignature",
           "nullable_rules", "derivable_rules"]


class GrammarError(Exception):
    pass


@dataclass(frozen=True)
class ToolSignature:
    """Minimal view of a tool that the grammar needs."""

    name: str
    params: tuple[str, ...]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", self.name):
            raise GrammarError(f"tool name {self.name!r} is not a valid identifier")
        for p in self.params:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", p):
                raise GrammarError(f"parameter {p!r} of {self.name!r} is invalid")


# The literal that terminates a parameter value, straight from the chat template.
VALUE_TERMINATOR = "\n</parameter>"
# What the value rule must NOT contain.  Note this is the delimiter *prefix*,
# without the trailing '>'.  A 0.8 B model reliably produces near-misses like
# "</parameter >" or "</parameter />"; if the value rule only forbade the exact
# full delimiter, it would happily swallow those as value text, the parameter
# would never close, and generation would run to the token cap with the
# arguments silently lost.  Forbidding the prefix instead means the decoder is
# forced to emit the exact delimiter the grammar expects.
VALUE_AVOID = "\n</parameter"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
THINK_CLOSE = " response"


def _gbnf_string(text: str) -> str:
    """Render a Python string as a GBNF double-quoted literal."""
    out = ['"']
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20:
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _class_char(ch: str) -> str:
    """Escape one character for use inside a GBNF character class."""
    if ch == "\n":
        return "\\n"
    if ch == "\r":
        return "\\r"
    if ch == "\t":
        return "\\t"
    if ch in ("]", "^", "-", "\\"):
        return "\\" + ch
    if ord(ch) < 0x20 or ord(ch) == 0x7F:
        return f"\\x{ord(ch):02x}"
    return ch


class _RuleGen:
    """Allocates unique rule names and collects definitions."""

    def __init__(self) -> None:
        self.rules: dict[str, str] = {}
        self._n = 0
        self._cache: dict[str, str] = {}

    def fresh(self, hint: str = "r") -> str:
        self._n += 1
        safe = re.sub(r"[^A-Za-z0-9]", "", hint)[:24] or "r"
        return f"{safe}{self._n}"

    def define(self, name: str, body: str) -> str:
        if name in self.rules and self.rules[name] != body:
            raise GrammarError(f"rule {name} defined twice with different bodies")
        self.rules[name] = body
        return name

    # -- the workhorse -----------------------------------------------------
    def not_containing(self, literal: str, hint: str = "nc",
                       allow_empty: bool = True) -> str:
        """Return a rule name matching strings that never contain ``literal``.

        Built as a character-level DFA, one GBNF rule per state.  State ``i``
        means "the last ``i`` characters matched a prefix of the literal"; the
        accepting state (the literal fully matched) is simply **never emitted**,
        so completing the literal is impossible rather than merely discouraged.

        Each state consumes **exactly one character** and the continuation is
        made *optional* (``(next)?``), which is what makes the whole rule able to
        terminate.  An earlier version ended every state with a bare recursive
        reference -- ``state_0 ::= "<" state_1 | [^<] state_0`` -- which can never
        finish, because after any character the parser still requires another one.
        GBNF has no way to express "and stop here" (an empty literal ``""`` is a
        syntax error to llama.cpp), so optionality is the mechanism.  The symptom
        was subtle and severe: the plain-answer branch could not accept
        end-of-input, so with no tools offered the model generated until the token
        cap and returned *nothing*.  :func:`derivable_rules` now guards this.

        Correctness also requires the literal's first character to appear nowhere
        else in it, so a fresh partial match can only restart at state 1.
        """
        key = (literal, allow_empty)
        if key in self._cache:
            return self._cache[key]
        if not literal:
            raise GrammarError("cannot build a rule for an empty literal")

        first = literal[0]
        if literal.count(first) != 1:
            raise GrammarError(
                f"literal {literal!r} cannot be used for delimiter avoidance: "
                f"its first character {first!r} recurs, so restarts are ambiguous"
            )

        m = len(literal)
        states = [self.fresh(f"{hint}s{i}") for i in range(m)]

        for i in range(m):
            ch = literal[i]
            alts: list[str] = []

            if i + 1 < m:
                # advance one character along the literal; the continuation is
                # optional so that the rule can always terminate
                alts.append(f"{_gbnf_string(ch)} ({states[i + 1]})?")
            # i == m - 1 deliberately has NO advance transition: completing the
            # literal is the one thing that must not be possible.

            if i == 0:
                alts.append(f"[^{_class_char(ch)}] ({states[0]})?")
            else:
                # a character that is neither the expected one nor a fresh start
                alts.append(
                    f"[^{_class_char(ch)}{_class_char(first)}] ({states[0]})?"
                )
                # the first character begins a brand-new partial match
                alts.append(f"{_gbnf_string(first)} ({states[1]})?")

            self.define(states[i], " | ".join(alts))

        whole = self.fresh(f"{hint}all")
        repeat = "*" if allow_empty else "+"
        self.define(whole, f"({states[0]}){repeat}")
        self._cache[key] = whole
        return whole


def build_grammar(
    tools: Sequence[ToolSignature],
    *,
    thinking: bool = False,
    allow_plain_answer: bool = True,
) -> str:
    """Compile the tool interface into a GBNF grammar.

    Parameters
    ----------
    tools:
        The tools the model may call.  Their names and parameter names become
        literal alternatives in the grammar.
    thinking:
        When ``True`` the prompt was seeded with `` thinking``, so the grammar
        expects reasoning text followed by ``</think>`` before the body.
    allow_plain_answer:
        When ``True`` the model may answer in prose instead of calling a tool
        (the run then ends).  When ``False`` every turn must be a tool call.
    """
    if not tools and not allow_plain_answer:
        raise GrammarError("no tools and no plain answers leaves nothing to generate")

    g = _RuleGen()

    # -- values ------------------------------------------------------------
    value_rule = g.not_containing(VALUE_AVOID, hint="val")

    # -- one rule per parameter, per tool ---------------------------------
    tool_alternatives: list[str] = []
    for tool in tools:
        param_alts: list[str] = []
        for pname in tool.params:
            rname = g.fresh("p")
            g.define(
                rname,
                f'{_gbnf_string(f"<parameter={pname}>" + chr(10))}'
                f" {value_rule}"
                f' {_gbnf_string(chr(10) + "</parameter>" + chr(10))}',
            )
            param_alts.append(rname)

        params_body = " | ".join(param_alts) if param_alts else None
        if params_body is None:
            params_ref = '""'
        else:
            pr = g.fresh("params")
            # order-free and tolerant of repeats: small models re-emit keys
            g.define(pr, f"({params_body})*")
            params_ref = pr

        call_rule = g.fresh("call")
        g.define(
            call_rule,
            f'{_gbnf_string(TOOL_CALL_OPEN + chr(10))}'
            f' {_gbnf_string(f"<function={tool.name}>" + chr(10))}'
            f" {params_ref}"
            f' {_gbnf_string("</function>" + chr(10) + TOOL_CALL_CLOSE)}',
        )
        tool_alternatives.append(call_rule)

    # -- body --------------------------------------------------------------
    body_alts: list[str] = list(tool_alternatives)
    if allow_plain_answer:
        # non-nullable: a nullable body would make `root` nullable, and
        # llama.cpp then treats the grammar as satisfied from the start and
        # stops masking entirely (see not_containing's docstring)
        answer_rule = g.not_containing(TOOL_CALL_OPEN, hint="ans",
                                       allow_empty=False)
        body_alts.append(answer_rule)

    body = g.fresh("body")
    g.define(body, " | ".join(body_alts))

    # -- root --------------------------------------------------------------
    if thinking:
        reasoning = g.not_containing(THINK_CLOSE, hint="think")
        g.define(
            "root",
            f"{reasoning}"
            f' {_gbnf_string(THINK_CLOSE + chr(10) + chr(10))}'
            f" {body}",
        )
    else:
        g.define("root", body)

    lines = [f"{name} ::= {body_}" for name, body_ in g.rules.items()]
    # `root` must come first for readability; GBNF does not require it
    lines.sort(key=lambda ln: 0 if ln.startswith("root ::=") else 1)
    text = "\n".join(lines) + "\n"

    # Fail fast rather than shipping a grammar llama.cpp will silently ignore.
    problems = validate_grammar(text)
    if problems:
        raise GrammarError(
            "generated an invalid grammar: " + "; ".join(problems)
        )
    return text


# --------------------------------------------------------------------------
# Static validation
# --------------------------------------------------------------------------
_RULE_DEF = re.compile(r"^([A-Za-z][A-Za-z0-9-]*)\s*::=\s*(.+)$")
_REF = re.compile(r"(?<![\"\[\w-])([A-Za-z][A-Za-z0-9-]*)(?![\"\w-])")
_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9-]*")

_UNESCAPE = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}


def _unescape_literal(raw: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            if nxt in _UNESCAPE:
                out.append(_UNESCAPE[nxt])
                i += 2
                continue
            if nxt == "x" and i + 3 < len(raw):
                try:
                    out.append(chr(int(raw[i + 2:i + 4], 16)))
                    i += 4
                    continue
                except ValueError:
                    pass
            if nxt in "uU":
                width = 4 if nxt == "u" else 8
                try:
                    out.append(chr(int(raw[i + 2:i + 2 + width], 16)))
                    i += 2 + width
                    continue
                except ValueError:
                    pass
        out.append(raw[i])
        i += 1
    return "".join(out)


def _split_alternatives(body: str) -> list[list[tuple[str, object, str]]]:
    """Parse a GBNF rule body into alternatives of terms.

    Each term is ``(kind, payload, suffix)`` where suffix is one of ``""``,
    ``"*"``, ``"+"``, ``"?"``.  GBNF allows the repeat operators on a bare rule
    reference (``c*``) as well as on a parenthesised group (``(a b)*``), and
    missing the bare form made the nullability analysis silently wrong.
    """
    alts: list[list[tuple[str, object, str]]] = [[]]
    i = 0
    n = len(body)

    def trailing_suffix(idx: int) -> str:
        return body[idx] if idx < n and body[idx] in "*+?" else ""

    def push(kind: str, payload: object) -> None:
        nonlocal i
        suffix = trailing_suffix(i)
        if suffix:
            i += 1
        alts[-1].append((kind, payload, suffix))

    while i < n:
        ch = body[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "|":
            alts.append([])
            i += 1
            continue
        if ch == '"':
            j = i + 1
            buf: list[str] = []
            while j < n:
                if body[j] == "\\" and j + 1 < n:
                    buf.append(body[j:j + 2])
                    j += 2
                    continue
                if body[j] == '"':
                    break
                buf.append(body[j])
                j += 1
            raw = "".join(buf)
            i = j + 1
            push("lit", _unescape_literal(raw))
            continue
        if ch == "[":
            j = i + 1
            while j < n:
                if body[j] == "\\":
                    j += 2
                    continue
                if body[j] == "]":
                    break
                j += 1
            i = j + 1
            push("class", None)
            continue
        if ch == "(":
            depth = 1
            j = i + 1
            while j < n and depth:
                if body[j] == "(":
                    depth += 1
                elif body[j] == ")":
                    depth -= 1
                j += 1
            inner = body[i + 1:j - 1]
            suffix = ""
            if j < n and body[j] in "*+?":
                suffix = body[j]
                j += 1
            alts[-1].append(("group", inner, suffix))
            i = j
            continue
        m = _IDENT.match(body, i)
        if m:
            name = m.group(0)
            i = m.end()
            push("ref", name)
            continue
        i += 1
    return alts


def nullable_rules(defined: dict[str, str]) -> dict[str, bool]:
    """Which rules can derive the empty string?

    This is not academic.  A nullable ``root`` makes llama.cpp consider the
    grammar satisfied before the first token, so it stops masking and the model
    free-runs -- returning HTTP 200 with plausible-looking output and many tool
    calls in one message.  :func:`validate_grammar` fails loudly on it.
    """
    table: dict[str, bool] = {name: False for name in defined}
    parsed = {name: _split_alternatives(body) for name, body in defined.items()}

    def term_nullable(term: tuple[str, object, str]) -> bool:
        kind, payload, suffix = term
        if suffix in ("*", "?"):
            return True
        if kind == "lit":
            inner = payload == ""
        elif kind == "class":
            inner = False
        elif kind == "group":
            inner = any(
                all(term_nullable(t) for t in seq)
                for seq in _split_alternatives(str(payload))
            )
        elif kind == "ref":
            inner = table.get(str(payload), False)
        else:
            inner = False
        return inner  # for suffix "" or "+"

    changed = True
    while changed:
        changed = False
        for name, seqs in parsed.items():
            if table[name]:
                continue
            if any(all(term_nullable(t) for t in seq) for seq in seqs):
                table[name] = True
                changed = True
    return table


def derivable_rules(defined: dict[str, str]) -> dict[str, bool]:
    """Which rules can derive at least one *finite* string?

    A rule whose every alternative ends in another rule reference can never stop
    consuming.  End-of-input then becomes unreachable, so generation runs to the
    token cap and the model returns nothing -- silently, with HTTP 200.

    This is not hypothetical: an earlier ``not_containing`` emitted exactly such a
    rule (``state_0 ::= "<" state_1 | [^<] state_0``), which made the plain-answer
    branch of the grammar impossible to finish.  A conversational run, which
    offers no tools and therefore *only* the answer branch, produced 300 empty
    tokens on every attempt.
    """
    parsed = {name: _split_alternatives(body) for name, body in defined.items()}
    table: dict[str, bool] = {name: False for name in defined}

    def term_derivable(term: tuple[str, object, str]) -> bool:
        kind, payload, suffix = term
        if suffix in ("?", "*"):
            return True                     # may be skipped entirely
        if kind in ("lit", "class"):
            return True                     # consumes characters, then it is done
        if kind == "group":
            return any(
                all(term_derivable(t) for t in seq)
                for seq in _split_alternatives(str(payload))
            )
        if kind == "ref":
            return table.get(str(payload), False)
        return True

    changed = True
    while changed:
        changed = False
        for name, seqs in parsed.items():
            if table[name]:
                continue
            if any(all(term_derivable(t) for t in seq) for seq in seqs):
                table[name] = True
                changed = True
    return table


def validate_grammar(text: str) -> list[str]:
    """Structural check.  Returns a list of problems (empty == fine).

    Catches the mistakes this generator could plausibly make, each of which
    llama.cpp accepts happily and then ignores or hangs on:

    * a missing ``root``, a reference to an undefined rule, a duplicate rule;
    * a **nullable** ``root`` (the grammar looks already satisfied, so masking
      stops and the model free-runs);
    * a rule that can never **terminate** (every alternative ends in another rule
      reference, so end-of-input is unreachable and generation runs to the cap).
    """
    problems: list[str] = []
    defined: dict[str, str] = {}
    bodies: list[tuple[str, str]] = []

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        m = _RULE_DEF.match(line)
        if not m:
            problems.append(f"line {lineno}: not a rule definition: {line[:60]!r}")
            continue
        name, body = m.group(1), m.group(2)
        if name in defined:
            problems.append(f"line {lineno}: rule {name!r} defined twice")
        defined[name] = body
        bodies.append((name, body))

    if "root" not in defined:
        problems.append("no 'root' rule")

    # strip string literals and char classes before hunting for references
    for name, body in bodies:
        stripped = re.sub(r'"(?:[^"\\]|\\.)*"', ' "" ', body)
        stripped = re.sub(r"\[(?:[^\]\\]|\\.)*\]", " [] ", stripped)
        for ref in _REF.findall(stripped):
            if ref not in defined:
                problems.append(f"rule {name!r} references undefined rule {ref!r}")

    if defined:
        table = nullable_rules(defined)
        if table.get("root"):
            problems.append(
                "root is nullable: llama.cpp will treat the grammar as already "
                "satisfied and stop constraining generation (the model will "
                "emit several tool calls in one message). Make every top-level "
                "alternative require at least one character."
            )
        derivable = derivable_rules(defined)
        stuck = sorted(name for name, ok in derivable.items() if not ok)
        if stuck:
            problems.append(
                "these rules can never terminate, so end-of-input is "
                "unreachable and generation runs to the token cap: "
                + ", ".join(stuck[:8])
            )
    return problems

