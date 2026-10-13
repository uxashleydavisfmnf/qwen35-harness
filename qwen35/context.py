"""
Context management for a 262 144-token window.

Three facts shape this module.

**1. Long context is genuinely cheap on this model.**  ``Qwen3.5-0.8B`` has 24
layers, of which only 6 are ``full_attention``; the other 18 are linear-attention
(Gated DeltaNet) layers whose state is constant-size.  The KV cache therefore
grows on 6 layers instead of 24, at roughly 12 KiB/token in fp16, so the full
262 144-token window costs about 3 GiB rather than the ~12 GiB an equivalent
all-attention model would need.  That is why the harness is willing to run with
a large window at all.

**2. Cheap is not free.**  On this CPU box, prefill is the dominant cost, so the
window is a *budget* (default 32 K) that the operator can raise toward 262 144,
not a target to fill.

**3. Masking beats summarizing.**  Lindenbauer et al. ("The Complexity Trap",
arXiv:2508.21433) found that masking old observations halves cost while matching
or slightly beating LLM summarization, and SWE-agent's ablation found that
keeping only the last 5 observations *improved* solve rate (18.0% vs 15.0% for
full history).  Asking a 0.8 B model to summarize its own history would be both
expensive and unreliable, so this module never does that.  It elides.

One thing this module *adds* rather than elides: a **ledger**.  A 0.8 B model
loses track of what it has already done.  The harness, which knows exactly,
maintains a compact running state block (files touched, commands run, failures)
and re-injects it fresh each step.  It sits in the volatile suffix, so the
cacheable prefix stays byte-identical.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = ["Message", "ContextManager", "estimate_tokens", "Ledger"]


# --------------------------------------------------------------------------
# Token estimation
# --------------------------------------------------------------------------
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
                  r"\uac00-\ud7af]")
_WORDISH = re.compile(r"[A-Za-z0-9_]+")

# Characters per BPE token for Latin prose and code in the Qwen vocabulary.
# Deliberately a little pessimistic: underestimating the prompt is what causes a
# context overflow, overestimating only costs a slightly early compaction.
_CHARS_PER_TOKEN = 3.7


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate.

    Counting *words* is not enough: a single 400-character identifier is one
    "word" but roughly a hundred tokens, and getting that wrong by 50x would make
    the whole context budget meaningless.  So the estimate is dominated by
    character count, with CJK counted near one token per character (true for this
    vocabulary) and a small per-word boundary cost.

    The agent loop refines this with the exact ``prompt_tokens`` the backend
    reports, via :meth:`ContextManager.calibrate`, so it self-corrects per model
    and per language.
    """
    if not text:
        return 0
    n_cjk = len(_CJK.findall(text))
    rest = _CJK.sub(" ", text)
    n_words = len(_WORDISH.findall(rest))
    estimate = n_cjk * 1.0 + len(rest) / _CHARS_PER_TOKEN + n_words * 0.25
    return max(1, int(math.ceil(estimate)))


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"

# what a message *is*, which drives compaction priority
KIND_TASK = "task"            # the operator's request -- never dropped
KIND_ACTION = "action"        # assistant tool call
KIND_OBSERVATION = "observation"
KIND_NOTE = "note"            # the ledger / corrections -- regenerated
KIND_ANSWER = "answer"


@dataclass
class Message:
    role: str
    content: str = ""
    kind: str = ""
    thinking: str = ""
    image_b64: str = ""
    image_mime: str = ""
    visual_tokens: int = 0
    elided: bool = False
    stub: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def pinned(self) -> bool:
        return self.kind == KIND_TASK

    def text_tokens(self) -> int:
        total = estimate_tokens(self.content)
        if self.thinking:
            total += estimate_tokens(self.thinking)
        if not self.elided:
            total += self.visual_tokens
        return total

    def for_model(self) -> dict[str, Any]:
        """Render for a chat backend.

        The image part comes **first**, before the text.  That is the order the
        model was trained on and the order that produced a correct description in
        direct testing; putting the text first left the model apparently unaware
        of the picture.
        """
        if self.image_b64 and not self.elided:
            return {
                "role": self.role,
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{self.image_mime};base64,"
                                             f"{self.image_b64}"},
                    },
                    {"type": "text", "text": self.content},
                ],
            }
        return {"role": self.role, "content": self.content}


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------
@dataclass
class Ledger:
    """Harness-side record of what actually happened.

    Re-injected as a compact block each step so the model does not have to
    remember.  This is the cheapest reliability win available for a
    sub-billion-parameter model: perfect recall of *its own actions*, for free.

    It also carries an anti-hallucination signal.  A weak model will happily
    conclude "the tests pass" after its commands were refused and nothing ran at
    all, so the ledger states plainly which commands actually succeeded and says
    so when none have.
    """

    root: str = ""
    files_read: dict[str, tuple[int, int]] = field(default_factory=dict)
    files_written: list[str] = field(default_factory=list)
    commands: list[tuple[str, int]] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    steps: int = 0
    last_write_step: int = 0
    last_command_step: int = 0

    def record_read(self, rel: str, start: int, end: int) -> None:
        self.files_read[rel] = (start, end)

    def record_write(self, rel: str, verb: str = "wrote") -> None:
        entry = f"{rel}"
        if entry not in self.files_written:
            self.files_written.append(entry)
        self.last_write_step = self.steps

    def record_command(self, cmd: str, exit_code: int) -> None:
        self.commands.append((cmd, exit_code))
        self.last_command_step = self.steps
        if len(self.commands) > 40:
            del self.commands[:-40]

    def record_refusal(self, cmd: str) -> None:
        """A command the harness refused.  It did NOT run."""
        self.refusals.append(cmd)
        if len(self.refusals) > 8:
            del self.refusals[:-8]

    def record_error(self, message: str) -> None:
        self.errors.append(message)
        if len(self.errors) > 12:
            del self.errors[:-12]

    def render(self, limit_chars: int = 1000) -> str:
        if not any((self.root, self.files_read, self.files_written, self.commands,
                    self.refusals, self.errors, self.notes)):
            return ""
        lines = ["# Current state (maintained by the harness, not by you)"]
        if self.root:
            lines.append(
                f"Workspace root: {self.root}. Commands already run there. Use "
                "RELATIVE paths in tool arguments (e.g. `calc.py`) -- never paste "
                "this path and never `cd` to it."
            )
        if self.files_written:
            lines.append("Files you changed: " + ", ".join(self.files_written[-8:]))
        if self.files_read:
            recent = list(self.files_read.items())[-6:]
            lines.append("Files you read: "
                         + ", ".join(f"{p} (lines {a}-{b})" for p, (a, b) in recent))
        if self.commands:
            recent = self.commands[-4:]
            lines.append("Commands that actually ran: "
                         + "; ".join(f"`{c}` -> exit {e}" for c, e in recent))
            if not any(e == 0 for _c, e in self.commands):
                lines.append(
                    "NOTE: no command has succeeded yet, so nothing is verified."
                )
        else:
            lines.append("NOTE: you have not successfully run any command yet.")
        if self.refusals:
            lines.append(
                "Commands the harness REFUSED (they did not run, so they prove "
                "nothing): " + "; ".join(f"`{c}`" for c in self.refusals[-4:])
            )
        # Push for re-verification.  A small model happily edits a file, sees the
        # edit succeed, and declares victory without ever re-running the tests --
        # observed live.  Only nags when a command has been run at least once, so
        # pure authoring tasks are not pestered.
        if (self.last_write_step and self.last_command_step
                and self.last_write_step > self.last_command_step):
            lines.append(
                "NOTE: you changed files AFTER your last command, so the change "
                "is not verified yet. Run the tests again before you finish."
            )
        if self.notes:
            lines.append("Notes: " + "; ".join(self.notes[-4:]))
        if self.errors:
            lines.append("Recent failures to avoid repeating:")
            for e in self.errors[-3:]:
                lines.append(f"  - {e}")

        text = "\n".join(lines)
        if len(text) > limit_chars:
            text = text[:limit_chars] + "\n(ledger truncated)"
        return text


# --------------------------------------------------------------------------
# Context manager
# --------------------------------------------------------------------------
@dataclass
class ContextStats:
    tokens: int
    budget: int
    fraction: float
    elided_observations: int
    dropped_thinking: int
    dropped_images: int
    dropped_turns: int


class ContextManager:
    """Assembles, budgets and compacts the conversation."""

    def __init__(
        self,
        *,
        context_window: int,
        usable_fraction: float = 0.85,
        compact_at: float = 0.55,
        summarize_at: float = 0.75,
        hard_at: float = 0.92,
        keep_recent_turns: int = 6,
        reserve_for_reply: int = 768,
    ) -> None:
        self.context_window = max(2048, int(context_window))
        self.usable_fraction = min(1.0, max(0.3, usable_fraction))
        self.compact_at = compact_at
        self.summarize_at = summarize_at
        self.hard_at = hard_at
        self.keep_recent_turns = max(2, int(keep_recent_turns))
        self.reserve_for_reply = reserve_for_reply

        self.messages: list[Message] = []
        self.ledger = Ledger()
        self._calibration = 1.0
        self._stats = ContextStats(0, self.budget, 0.0, 0, 0, 0, 0)

    # -- budget ------------------------------------------------------------
    @property
    def budget(self) -> int:
        raw = int(self.context_window * self.usable_fraction) - self.reserve_for_reply
        return max(1024, raw)

    def calibrate(self, estimated: int, actual: int) -> None:
        """Learn the estimator's bias from the backend's own token count."""
        if estimated <= 0 or actual <= 0:
            return
        ratio = actual / float(estimated)
        if 0.4 <= ratio <= 2.5:
            # exponential moving average, slow so one odd turn cannot skew it
            self._calibration = 0.7 * self._calibration + 0.3 * ratio

    def usage(self) -> int:
        total = 0
        for m in self.messages:
            total += m.text_tokens()
        total += estimate_tokens(self._ledger_text())
        return int(total * self._calibration)

    @property
    def stats(self) -> ContextStats:
        used = self.usage()
        self._stats = ContextStats(
            tokens=used,
            budget=self.budget,
            fraction=used / max(1, self.budget),
            elided_observations=self._stats.elided_observations,
            dropped_thinking=self._stats.dropped_thinking,
            dropped_images=self._stats.dropped_images,
            dropped_turns=self._stats.dropped_turns,
        )
        return self._stats

    # -- mutation ----------------------------------------------------------
    def add(self, message: Message) -> Message:
        self.messages.append(message)
        return message

    def add_task(self, text: str, *, image_b64: str = "",
                 image_mime: str = "", visual_tokens: int = 0) -> Message:
        """The user's request.  May carry an attached picture directly.

        A single attached image is inlined as a content part for the same reason
        text attachments are inlined: otherwise the model has to reproduce a
        hash-bearing path (``.qwen35/attachments/1-tft-9f3c2a11.jpg``) exactly in
        order to look at a picture that was handed to it.  Measured: it called
        ``view_image``, got the path wrong, retried the same wrong call, and the
        duplicate breaker ended the run with no answer.
        """
        return self.add(Message(role=ROLE_USER, content=text, kind=KIND_TASK,
                                image_b64=image_b64, image_mime=image_mime,
                                visual_tokens=visual_tokens))

    def add_observation(self, text: str, *, image_b64: str = "",
                        image_mime: str = "", visual_tokens: int = 0,
                        meta: dict[str, Any] | None = None) -> Message:
        return self.add(Message(
            role=ROLE_TOOL, content=text, kind=KIND_OBSERVATION,
            image_b64=image_b64, image_mime=image_mime,
            visual_tokens=visual_tokens, meta=meta or {},
        ))

    def add_note(self, text: str) -> Message:
        """Append a correction or nudge.

        **Append-only, on purpose.**  An earlier version removed the previous note
        and re-appended, which rewrote the message list in the middle.  Two things
        broke:

        * every rewrite invalidated the server's prefix cache from the earliest
          removed message onward, so a retry re-prefilled the whole conversation
          — the single most expensive thing this harness does on a CPU;
        * worse, :meth:`build_payload` filtered notes out of the payload
          altogether, so the carefully-worded correction from
          :func:`qwen35.parse.format_correction` was **never delivered**.  The
          model was retried on an unchanged prompt and only the one-line ledger
          error hinted at what had gone wrong.

        Notes are now ordinary transcript entries: kept in order, sent to the
        model, and elided like anything else once they are old.
        """
        return self.add(Message(role=ROLE_USER, content=text, kind=KIND_NOTE))

    def add_action(self, text: str, *, thinking: str = "",
                   tool_summary: str = "") -> Message:
        return self.add(Message(
            role=ROLE_ASSISTANT, content=text, kind=KIND_ACTION,
            thinking=thinking, meta={"tool": tool_summary} if tool_summary else {},
        ))

    def add_answer(self, text: str, *, thinking: str = "") -> Message:
        return self.add(Message(role=ROLE_ASSISTANT, content=text,
                                kind=KIND_ANSWER, thinking=thinking))

    # -- ledger ------------------------------------------------------------
    def _ledger_text(self) -> str:
        return self.ledger.render()

    def ledger_message(self) -> str:
        return self._ledger_text()

    # -- compaction --------------------------------------------------------
    def compact(self, force: bool = False) -> ContextStats:
        """Bring the conversation back inside budget.

        Escalation order, cheapest first, **stopping as soon as the budget is
        satisfied**.  Running every stage unconditionally would throw away turns
        that a cheaper stage had already saved.

        1. drop reasoning from old turns (the chat template discards it anyway,
           so this costs nothing in fidelity)
        2. elide old observations to one line each
        3. drop old images (they cost visual tokens on every single prefill)
        4. drop whole old turns, keeping the pinned task and the recent tail
        """
        stats = self.stats
        if not force and stats.fraction < self.compact_at:
            return stats

        def ratio() -> float:
            return self.usage() / max(1, self.budget)

        def satisfied() -> bool:
            # once within the entry threshold there is nothing left to do
            return ratio() < self.compact_at

        self._drop_old_thinking()
        if satisfied():
            return self.stats

        self._mask_observations()
        if satisfied():
            return self.stats

        self._drop_old_images()
        if satisfied():
            return self.stats

        self._drop_old_turns()
        return self.stats

    def _assistant_indices(self) -> list[int]:
        return [i for i, m in enumerate(self.messages)
                if m.role == ROLE_ASSISTANT]

    def _protected_start(self) -> int:
        """Index from which nothing may be dropped (the recent tail)."""
        ai = self._assistant_indices()
        if len(ai) <= self.keep_recent_turns:
            return 0
        return ai[-self.keep_recent_turns]

    def _drop_old_thinking(self) -> None:
        cutoff = self._protected_start()
        dropped = 0
        for i, m in enumerate(self.messages):
            if i < cutoff and m.thinking:
                m.thinking = ""
                dropped += 1
        self._stats.dropped_thinking += dropped

    def _mask_observations(self) -> None:
        cutoff = self._protected_start()
        masked = 0
        for i, m in enumerate(self.messages):
            if i >= cutoff or m.elided:
                continue
            # Corrections are observations too: once the turn they referred to is
            # far behind, the correction is noise and can be elided with it.
            if m.kind not in (KIND_OBSERVATION, KIND_NOTE):
                continue
            if m.role not in (ROLE_TOOL, ROLE_USER):
                continue
            m.stub = self._stub_for(m)
            m.content = m.stub
            m.elided = True
            m.image_b64 = ""
            m.visual_tokens = 0
            masked += 1
        self._stats.elided_observations += masked

    def _stub_for(self, m: Message) -> str:
        """One-line summary of an observation, SWE-agent style.

        Kept deliberately terse.  The first line is often a long file listing, so
        it is clipped hard -- a verbose stub defeats the purpose of masking, and
        an earlier 190-character version measurably failed to bring the budget
        back under the threshold on its own.
        """
        first = (m.content or "").splitlines()[0] if m.content else ""
        first = " ".join(first.split())
        if len(first) > 60:
            first = first[:59] + "\u2026"
        if m.kind == KIND_NOTE:
            return f"[earlier instruction elided: {first}]"
        tool = m.meta.get("tool", "tool")
        return f"[{tool} result elided: {first}]"

    def _drop_old_images(self) -> None:
        cutoff = self._protected_start()
        dropped = 0
        for i, m in enumerate(self.messages):
            if i < cutoff and m.image_b64:
                m.image_b64 = ""
                m.visual_tokens = 0
                m.content = (m.content + "\n[image dropped to save context; "
                                         "call view_image again if needed]").strip()
                dropped += 1
        self._stats.dropped_images += dropped

    def _drop_old_turns(self) -> None:
        """Last resort: remove whole old turns, never the pinned task."""
        keep_recent = self._protected_start()
        kept: list[Message] = []
        dropped = 0
        removed_tool_groups = 0
        for i, m in enumerate(self.messages):
            if i >= keep_recent or m.pinned or m.kind in (KIND_TASK, KIND_NOTE):
                kept.append(m)
                continue
            # dropping an assistant action without its observation (or vice
            # versa) leaves an incoherent transcript, so drop the pair together
            if m.role == ROLE_TOOL:
                removed_tool_groups += 1
            dropped += 1

        # a tool message must always follow the assistant call that produced it
        cleaned: list[Message] = []
        prev_role = None
        for m in kept:
            if m.role == ROLE_TOOL and prev_role != ROLE_ASSISTANT:
                dropped += 1
                continue
            cleaned.append(m)
            prev_role = m.role

        self.messages = cleaned
        self._stats.dropped_turns += dropped

    # -- rendering ---------------------------------------------------------
    def build_payload(self, *, system: str, tools: list[dict[str, Any]] | None,
                      with_ledger: bool = True) -> list[dict[str, Any]]:
        """Render the conversation into backend messages."""
        out: list[dict[str, Any]] = [{"role": ROLE_SYSTEM, "content": system}]

        ledger = self.ledger_message() if with_ledger else ""

        # Notes (corrections, nudges) ARE part of the transcript now.  Filtering
        # them out here silently discarded every repair message the harness built.
        body = list(self.messages)
        rendered: list[dict[str, Any]] = []
        for m in body:
            item = m.for_model()
            if m.role == ROLE_ASSISTANT and m.thinking:
                # keep the trained shape: reasoning, then the visible content
                item["content"] = f" thinking\n{m.thinking}\n</think>\n\n{m.content}"
            rendered.append(item)

        # the ledger rides at the very end so the cacheable prefix is untouched
        if ledger:
            rendered.append({"role": ROLE_USER, "content": ledger})

        out.extend(rendered)
        return out

    def transcript(self) -> list[dict[str, Any]]:
        """Full fidelity record, for persistence (not for the model)."""
        return [
            {
                "role": m.role,
                "kind": m.kind,
                "content": m.content,
                "thinking": m.thinking,
                "elided": m.elided,
                "visual_tokens": m.visual_tokens,
                "has_image": bool(m.image_b64),
                "meta": m.meta,
            }
            for m in self.messages
        ]
