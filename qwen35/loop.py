"""
The agent loop.

Interleaved ReAct: one thought, one action, one observation, repeat.  ReWOO-style
plan-then-execute was rejected deliberately -- it wins on token efficiency by
deferring all observation, which is exactly wrong for a 0.8 B model whose first
plan will often be wrong and which then has no observation to correct it with.

What makes a 0.8 B model work here is that almost every reliability mechanism
lives in the harness rather than in the prompt:

===================  =========================================================
Mechanism            Where it lives
===================  =========================================================
Valid tool names     GBNF grammar (decode time, not prompt time)
Valid XML envelope   GBNF grammar
One action per turn  grammar + parser
Bad edit reverted    :func:`tools.validate_syntax` restores the file
Duplicate calls      harness refuses to re-run; tells the model to change tack
Loops                degeneration detector on raw text + repeated-action hash
Small observations   one-line masking of old tool output
Lost state           harness ledger, re-injected every step
Recovery             precise correction message quoting the legal template
===================  =========================================================

The loop is also strict about budget: steps, wall-clock, commands and context all
have hard ceilings, because the failure mode of a weak agent is not being wrong,
it is never stopping.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .backends.base import BackendError, ChatBackend
from .config import HarnessConfig, ModelProfile, QWEN35_08B
from .context import (KIND_ACTION, KIND_ANSWER, KIND_OBSERVATION, KIND_TASK,
                      ContextManager, Message, estimate_tokens)
from .grammar import ToolSignature, build_grammar
from .parse import (ParsedTurn, ToolCall, detect_degeneration,
                    format_correction, parse_turn)
from .permissions import CommandScreener
from .prompt import (ToolSpec, answers_from_prompt, build_system,
                     build_tool_schema, conversational_example,
                     conversational_system, few_shot_block,
                     looks_conversational, system_rules, wants_changes)
from .session import SessionRecorder
from .tools import ToolContext, ToolResult, build_specs, call_tool
from .workspace import WorkspaceGuard

__all__ = ["Agent", "RunResult", "Step"]


@dataclass
class Step:
    index: int
    thinking: str = ""
    text: str = ""
    call: ToolCall | None = None
    result: ToolResult | None = None
    note: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cached_in: int = 0
    seconds: float = 0.0
    grammar: bool = False


@dataclass
class RunResult:
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    stop_reason: str = "finished"
    tokens_in: int = 0
    tokens_out: int = 0
    cached_in: int = 0
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return self.stop_reason in ("finished", "max_steps")

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.call is not None)

    @property
    def cache_hit_ratio(self) -> float:
        """Share of prefill tokens the server served from its prefix cache."""
        if self.tokens_in <= 0:
            return 0.0
        return min(1.0, self.cached_in / self.tokens_in)


class Agent:
    """Runs a task to completion inside one workspace."""

    def __init__(
        self,
        cfg: HarnessConfig,
        backend: ChatBackend,
        *,
        profile: ModelProfile = QWEN35_08B,
        tool_names: list[str] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        recorder: SessionRecorder | None = None,
        attachments: list[Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.backend = backend
        self.profile = profile
        self.guard = WorkspaceGuard(cfg.workspace)
        self.screener = CommandScreener(self.guard, cfg.permissions)
        self.tool_specs: list[ToolSpec] = build_specs(tool_names, cfg.permissions)
        self.on_event = on_event or (lambda _e, _d: None)
        self.recorder = recorder
        self._stop = False
        self._conversational = False
        #: attachments were supplied, so a run that reads nothing is suspicious
        self._attachments_pending = bool(attachments)
        #: the attachment objects themselves, so the loop can tell whether the
        #: answer is already sitting in the prompt
        self._attachments = list(attachments or [])
        self._answer_from_prompt = False
        self._inlined_image = False
        self._saved_specs: list[ToolSpec] = []

        self.ctx = ToolContext(
            guard=self.guard,
            perms=cfg.permissions,
            profile=profile,
            screener=self.screener,
            allow_vision=cfg.permissions.allow_vision and profile.supports_vision,
            attachment_map=self._attachment_map(attachments or []),
        )
        self.runtime = cfg.runtime
        self.manager = ContextManager(
            context_window=self.runtime.context_window,
            usable_fraction=self.runtime.usable_fraction,
            compact_at=self.runtime.compact_at,
            summarize_at=self.runtime.summarize_at,
            hard_at=self.runtime.hard_at,
            keep_recent_turns=self.runtime.keep_recent_turns,
            reserve_for_reply=self.runtime.max_tokens,
        )
        # telling the model where it is prevents "cd .." style guessing
        self.manager.ledger.root = str(self.guard.root)

    # ------------------------------------------------------------------
    def _attachment_map(self, attachments: list[Any]) -> dict[str, str]:
        """Map each attached original to its converted text, for redirection."""
        out: dict[str, str] = {}
        for att in attachments:
            out_rel = getattr(att, "out_rel", "")
            source = getattr(att, "source", None)
            if not out_rel or source is None:
                continue
            out[Path(source).name] = out_rel
            try:
                rel = Path(os.path.realpath(source)).relative_to(
                    self.guard.root).as_posix()
                out[rel] = out_rel
            except (ValueError, OSError):
                pass
        return out

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------
    def _tool_prompt_mode(self) -> str:
        """Decide whether the tool block is embedded (grammar) or sent as tools.

        llama-server cannot combine the two, so exactly one path is taken.
        """
        mode = (self.runtime.tool_prompt or "auto").lower()
        if mode in ("grammar", "tools"):
            return mode
        # auto: the grammar is the stronger guarantee for a 0.8 B model
        if (self.runtime.use_grammar or "auto").lower() != "off" \
                and self.backend.supports_grammar:
            return "grammar"
        return "tools"

    def _system_prompt(self, tools: list[dict[str, Any]]) -> str:
        if self._conversational:
            # Short on purpose: the tool block costs ~12 s of prefill on this CPU
            # and advertising six tools to a 0.8 B model is an invitation to use
            # them on a greeting.
            return (conversational_system(workspace_label=self.guard.root.name or ".")
                    + "\n\n" + conversational_example())
        note = self._permission_note()
        body = build_system(
            tools if self._tool_prompt_mode() == "grammar" else [],
            workspace_label=self.guard.root.name or ".",
            workspace_root=str(self.guard.root),
            max_read_lines=100,
            permission_note=note,
            embed_tools=True,
        )
        return f"{body}\n\n{few_shot_block()}"

    def _permission_note(self) -> str:
        p = self.cfg.permissions
        allowed, denied = [], []
        (allowed if p.read else denied).append("read files in the workspace")
        (allowed if p.write else denied).append("create and overwrite files")
        (allowed if p.edit else denied).append("edit files")
        (allowed if p.run_command else denied).append("run allowed commands")
        denied.append("delete or move files")
        denied.append("touch anything outside the workspace")
        if not p.allow_network_commands:
            denied.append("use the network")
        if not p.allow_package_install:
            denied.append("install packages")
        lines = ["# Your permissions (enforced by the harness, not by you)"]
        if allowed:
            lines.append("Allowed: " + "; ".join(allowed) + ".")
        lines.append("Refused: " + "; ".join(denied) + ".")
        return "\n".join(lines)

    def _grammar(self, *, thinking: bool) -> str | None:
        mode = (self.runtime.use_grammar or "auto").lower()
        if mode == "off":
            return None
        if self._tool_prompt_mode() == "tools":
            # the server rejects grammar together with tools
            return None
        if mode == "auto" and not self.backend.supports_grammar:
            return None
        sigs = [ToolSignature(name=s.name, params=s.params) for s in self.tool_specs]
        try:
            return build_grammar(sigs, thinking=thinking, allow_plain_answer=True)
        except Exception as exc:  # noqa: BLE001 - a broken grammar must not stop a run
            # NEVER fail silently.  Returning None here disables constrained
            # decoding for the whole step, and the model is then free to invent
            # tool names -- which looks exactly like a bug in the parser.  A
            # `grep` call that no grammar could have produced is what led here.
            self.on_event("grammar_unavailable",
                          {"error": f"{type(exc).__name__}: {exc}"})
            self.manager.ledger.record_error(
                f"constrained decoding UNAVAILABLE this step: {exc}")
            return None

    def _use_thinking(self, step_index: int, retrying: bool,
                      *, last_truncated: bool = False) -> bool:
        """Decide whether to let the model reason before answering.

        Three rules, each of which was measured rather than assumed:

        * a greeting never needs reasoning -- thinking there cost 60-200 seconds
          and often produced an *empty* answer after burning the whole token
          budget, so this shortcut applies in every mode;
        * reasoning that already consumed the budget and produced nothing usable
          is not worth repeating;
        * otherwise ``on`` thinks, ``off`` does not, and ``auto`` thinks when
          planning and whenever the previous step went wrong.
        """
        if not self.profile.thinking_capable:
            return False
        mode = (self.runtime.thinking or "auto").lower()
        if mode == "off":
            return False
        if step_index == 0 and self._conversational:
            return False
        if last_truncated:
            return False
        if mode == "on":
            return True
        return step_index == 0 or retrying

    # ------------------------------------------------------------------
    def run(self, task: str, *, history: list[Message] | None = None) -> RunResult:
        started = time.monotonic()
        result = RunResult()

        # A single attached picture is inlined straight into the message, exactly
        # as inlined text is.  Otherwise the model must reproduce a hash-bearing
        # path (`.qwen35/attachments/1-attached-9f3c2a11.jpg`) just to look at a
        # picture that was handed to it -- measured: it called view_image with the
        # wrong path, repeated the same call, and the duplicate breaker ended the
        # run with no answer.
        image_kwargs: dict[str, Any] = {}
        if (len(self._attachments) == 1 and not wants_changes(task)
                and self.ctx.allow_vision):
            only = self._attachments[0]
            if getattr(only, "kind", "") == "image" and getattr(only, "images", None):
                try:
                    from .tools import prepare_image_b64

                    b64, mime, visual, description = prepare_image_b64(
                        self.ctx, only.images[0])
                    image_kwargs = {"image_b64": b64, "image_mime": mime,
                                    "visual_tokens": visual}
                    self.manager.ledger.notes.append(
                        "The attached picture is included in this message "
                        f"({description}). Describe it directly -- you do not need "
                        "to call view_image."
                    )
                    self._inlined_image = True
                except Exception as exc:  # noqa: BLE001 - fall back to the tool
                    self.on_event("inline_image_failed", {"error": str(exc)})

        self.manager.add_task(task, **image_kwargs)
        if history:
            self.manager.messages = list(history) + self.manager.messages

        # Small talk is routed away from the tool loop.  A 0.8 B model otherwise
        # treats "你好" as a work order and tours the repository; see
        # qwen35.prompt.looks_conversational.
        self._conversational = looks_conversational(task)
        if self._conversational:
            # Hard guarantee rather than a hint.  With no tools in the schema and
            # none in the grammar the model *cannot* call one, so a greeting is
            # answered in a single cheap generation instead of a repository tour.
            # Prompting alone measured 1/4; this measures deterministic.
            self.tool_specs = []
            self.manager.ledger.notes.append(
                "This message is a greeting or general question: reply directly "
                "and briefly, with no tool call."
            )
        else:
            # The full text of an attached file is already in this message and
            # nothing asks for the workspace to change, so the first step offers
            # NO tools -- the same hard guarantee that fixed greetings.
            #
            # Prompting was not enough here either: with a 973-token spreadsheet
            # fully inlined, the model still called list_dir twice (the rules say
            # "call list_dir once before reading anything"), tripped the
            # duplicate breaker and returned no answer at all.
            self._answer_from_prompt = (
                answers_from_prompt(task, self._attachments)
                or self._inlined_image
            )
            if self._answer_from_prompt:
                self.manager.ledger.notes.append(
                    "The full text of the attached file is already in this "
                    "message. Answer from it directly in plain prose, with NO "
                    "tool call. Do not list or read the workspace."
                )

        if self.recorder:
            self.recorder.record("task", {"text": task,
                                          "conversational": self._conversational})

        system = self._system_prompt(build_tool_schema(self.tool_specs))
        tools_schema = build_tool_schema(self.tool_specs)
        # exactly one of {tools field, grammar} is used -- the server forbids both
        send_tools = (
            tools_schema
            if (self._tool_prompt_mode() == "tools" and self.backend.supports_tools)
            else None
        )

        parse_failures = 0
        refusals = 0
        action_counts: dict[str, int] = {}
        consecutive_same = 0
        last_key = ""

        self.on_event("start", {
            "workspace": str(self.guard.root),
            "tools": [s.name for s in self.tool_specs],
            "backend": self.backend.describe(),
            "context_window": self.runtime.context_window,
            "max_steps": self.runtime.max_steps,
        })

        step_index = 0
        last_truncated = False
        illegal_names = 0
        suspended_tools = False
        while step_index < self.runtime.max_steps:
            if self._stop:
                result.stop_reason = "interrupted"
                break

            self.manager.compact()
            stats = self.manager.stats
            if stats.fraction >= 1.0:
                self.on_event("context_full", {"tokens": stats.tokens,
                                               "budget": stats.budget})

            retrying = parse_failures > 0

            # First step, answer already in the prompt: offer no tools.  With an
            # empty tool list the grammar has no call branch at all, so the model
            # *cannot* call one -- a hard guarantee rather than a hint.
            if self._answer_from_prompt and step_index == 0 and self.tool_specs:
                self._saved_specs = list(self.tool_specs)
                self.tool_specs = []
                suspended_tools = True

            if self._conversational:
                thinking = False
                max_tokens = min(self.runtime.max_tokens, 300)
            elif suspended_tools:
                # Answering from text already in the prompt: skip the reasoning
                # budget.  With thinking on, a long think block plus a summary
                # overran max_tokens, the completion was cut off, and the loop
                # correctly treated that as a failure and retried -- turning a
                # one-step answer into six steps.
                thinking = False
                max_tokens = min(self.runtime.max_tokens, 1200)
            else:
                thinking = self._use_thinking(step_index, retrying,
                                              last_truncated=last_truncated)
                max_tokens = self.runtime.max_tokens
            last_truncated = False        # consumed; only the failure below re-sets it
            grammar = self._grammar(thinking=thinking)

            messages = self.manager.build_payload(
                system=system,
                tools=send_tools,
            )

            estimated = estimate_tokens(
                "".join(str(m.get("content", "")) for m in messages)
            )

            self.on_event("thinking_start", {"step": step_index + 1,
                                            "grammar": bool(grammar),
                                            "thinking": thinking})
            t0 = time.monotonic()
            try:
                gen = self.backend.generate(
                    messages,
                    tools=send_tools,
                    grammar=grammar,
                    temperature=self.runtime.temperature,
                    top_p=self.runtime.top_p,
                    top_k=self.runtime.top_k,
                    max_tokens=max_tokens,
                    # `</tool_call>` as a stop string is what actually guarantees
                    # one action per step.  llama.cpp releases grammar masking
                    # once the grammar is satisfied, so a grammar alone lets the
                    # model keep emitting further tool calls to the token cap --
                    # measured at ~22 s and 400 wasted tokens per step versus
                    # 1.3 s and 24 tokens with the stop string in place.
                    stop=tuple(self.profile.stop_tokens) + ("</tool_call>",),
                    thinking=thinking,
                    repeat_penalty=self.runtime.repeat_penalty,
                    seed=self.runtime.seed,
                )
            except BackendError as exc:
                result.stop_reason = f"backend error: {exc}"
                self.on_event("error", {"message": str(exc)})
                break
            elapsed = time.monotonic() - t0

            if gen.prompt_tokens:
                self.manager.calibrate(estimated, gen.prompt_tokens)
            result.tokens_in += gen.prompt_tokens
            result.tokens_out += gen.completion_tokens
            result.cached_in += gen.cached_tokens

            raw = gen.text or ""
            self.on_event("raw", {"step": step_index + 1, "text": raw,
                                  "seconds": elapsed})

            # --- degeneration: the model looped inside one completion ------
            if detect_degeneration(raw):
                self.on_event("degenerate", {"step": step_index + 1})
                self.manager.add_note(
                    "Your previous reply repeated itself until it was cut off. "
                    "Reply with ONE short tool call and nothing else."
                )
                parse_failures += 1
                step_index += 1
                if parse_failures > self.runtime.max_parse_retries + 2:
                    result.stop_reason = "degenerated"
                    break
                continue

            turn = parse_turn(raw)
            self._record_turn_meta(turn)

            # The no-tool step is over.  If the model answered, the run ends
            # below; if it somehow did not, tools come back so the task can still
            # be carried out.
            if suspended_tools:
                self.tool_specs = self._saved_specs
                suspended_tools = False

            # --- format failure or truncation: precise correction ----------
            # A completion that ran into the token cap without producing a tool
            # call is NOT a considered final answer -- it is the model running
            # out of room mid-thought.  Treating it as an answer would silently
            # end the run on a half-finished sentence.
            problem = turn.format_error
            if not problem and not turn.has_tool_call and gen.finish_reason == "length":
                problem = (
                    "your reply hit the length limit before you either called a "
                    "tool or wrote a finished answer"
                )
            if not problem and turn.is_empty:
                # An empty turn happens when truncated reasoning is the only
                # thing the model produced.  It is never a valid answer.
                problem = "your reply was empty"

            if not turn.has_tool_call and problem:
                parse_failures += 1
                # Remember that the verbose path failed, so the retry takes the
                # direct one instead of burning the same tokens again.
                last_truncated = bool(gen.finish_reason == "length"
                                      or turn.is_empty)
                self.manager.ledger.record_error(problem)
                self.on_event("format_error", {
                    "step": step_index + 1,
                    "problem": problem,
                    "retry": parse_failures,
                    "finish_reason": gen.finish_reason,
                })
                if parse_failures > self.runtime.max_parse_retries:
                    result.stop_reason = f"unparseable output: {problem}"
                    self.manager.add_answer(raw)
                    break
                correction = format_correction(
                    turn,
                    [(s.name, s.params) for s in self.tool_specs],
                    reason=problem,
                )
                if gen.finish_reason == "length":
                    correction += (
                        "\n\nBe brief: one short tool call only, no preamble and "
                        "no repetition."
                    )
                self.manager.add_note(correction)
                step_index += 1
                continue

            parse_failures = 0

            # --- plain prose: the run is finished --------------------------
            if not turn.has_tool_call:
                answer = turn.text.strip() or raw.strip()
                self.manager.add_answer(answer, thinking=turn.thinking)
                result.answer = answer
                result.stop_reason = "finished"
                self.on_event("answer", {"text": answer})
                break

            call = turn.tool_calls[0]

            # Defence in depth: the grammar should make an illegal tool name
            # impossible, so if one arrives the grammar was not applied.  Treat it
            # as a format failure with a precise correction rather than executing
            # a tool that does not exist.
            legal_names = {s.name for s in self.tool_specs}
            if call.name not in legal_names:
                # A separate counter: parse_failures is reset just above for the
                # *successful* parse path, so reusing it here would never
                # accumulate and the run would grind to max_steps instead.
                illegal_names += 1
                problem = (
                    f"there is no tool called {call.name!r}. The only tools are: "
                    + ", ".join(sorted(legal_names))
                )
                self.manager.ledger.record_error(problem)
                self.on_event("illegal_tool", {"name": call.name,
                                               "legal": sorted(legal_names),
                                               "count": illegal_names})
                if illegal_names > self.runtime.max_parse_retries:
                    result.stop_reason = f"illegal tool name: {call.name}"
                    break
                self.manager.add_note(format_correction(
                    turn, [(s.name, s.params) for s in self.tool_specs],
                    reason=problem))
                step_index += 1
                continue
            step = Step(
                index=step_index + 1,
                thinking=turn.thinking,
                text=turn.text,
                call=call,
                tokens_in=gen.prompt_tokens,
                tokens_out=gen.completion_tokens,
                cached_in=gen.cached_tokens,
                seconds=elapsed,
                grammar=gen.grammar_used,
            )

            # --- duplicate action breaker ---------------------------------
            # Counted *totally*, not just consecutively: a model that alternates
            # between two pointless calls never trips a consecutive-only check.
            key = call.key()
            if key == last_key:
                consecutive_same += 1
            else:
                consecutive_same = 1
                last_key = key
            action_counts[key] = action_counts.get(key, 0) + 1
            total_same = action_counts[key]

            # --- escalate duplicate refusals ------------------------------
            # Order matters: the "conclude now" exit must be reachable BEFORE the
            # hard stop.  With the hard stop first, a model that had already
            # verified its work ground on to a `looping` stop with no answer at
            # all -- which is exactly what happened before this was reordered.
            hard_cap = max(self.runtime.loop_stop_at,
                           self.runtime.loop_warn_at + 3)
            if total_same >= hard_cap:
                step.note = "stopped: repeated the same action too many times"
                result.steps.append(step)
                result.stop_reason = "looping"
                self.on_event("loop_stop", {"call": call.describe(),
                                            "repeats": total_same})
                break

            if total_same >= self.runtime.loop_warn_at:
                refusals += 1
                step.note = f"duplicate action ({total_same}x, refusal {refusals})"
                self._refuse_duplicate(call, total_same,
                                       final_warning=refusals >= 2)
                result.steps.append(step)
                step_index += 1
                if refusals >= 3:
                    # Nothing is changing, so concluding beats grinding on.
                    result.stop_reason = "finished"
                    self.on_event("forced_finish", {"repeats": total_same})
                    break
                continue

            # --- execute ------------------------------------------------
            self.on_event("tool_call", {"step": step.index,
                                        "call": call.describe()})
            self.manager.add_action(
                self._render_action(call, turn.text), thinking=turn.thinking,
                tool_summary=call.name,
            )

            tool_result = call_tool(call.name, self.ctx, call.args)
            step.result = tool_result
            observation = tool_result.for_model(self.cfg.permissions.max_output_bytes)

            self._update_ledger(call, tool_result)

            if tool_result.ok:
                self.on_event("tool_ok", {"call": call.describe(),
                                          "chars": len(observation)})
            else:
                self.on_event("tool_error", {"call": call.describe(),
                                             "text": observation[:300],
                                             "exit": tool_result.meta.get("exit"),
                                             "refused": tool_result.meta.get("refused", False)})
                self.manager.ledger.record_error(
                    f"{call.name} -> {observation.splitlines()[0][:160]}"
                    if observation else call.name
                )

            self.manager.add_observation(
                observation,
                image_b64=tool_result.image_b64,
                image_mime=tool_result.image_mime,
                visual_tokens=int(tool_result.meta.get("visual_tokens", 0)),
                meta={"tool": call.name},
            )

            # multimodal: the picture rides in a following user turn
            if tool_result.image_b64:
                self.manager.add_observation(
                    f"Here is the image from {call.name} "
                    f"({tool_result.meta.get('path', '')}). Look at it and answer "
                    "the user's question directly -- you do not need another tool "
                    "call for this.",
                    image_b64=tool_result.image_b64,
                    image_mime=tool_result.image_mime,
                    visual_tokens=int(tool_result.meta.get("visual_tokens", 0)),
                    meta={"tool": call.name, "image": True},
                )
                # the bare attachment must not look like a fresh instruction
                self.manager.messages[-1].role = "user"

            if self.recorder:
                self.recorder.record("step", {
                    "index": step.index,
                    "thinking": turn.thinking,
                    "call": {"name": call.name, "args": call.args},
                    "ok": tool_result.ok,
                    "observation": observation[:2000],
                    "seconds": elapsed,
                    # Prefill dominates the cost on CPU, so record it: a run whose
                    # prompt tokens dwarf its completion tokens is paying for
                    # context, not for reasoning.
                    "prompt_tokens": gen.prompt_tokens,
                    "completion_tokens": gen.completion_tokens,
                })

            result.steps.append(step)
            step_index += 1

        else:
            result.stop_reason = "max_steps"
            self.on_event("max_steps", {"steps": self.runtime.max_steps})

        result.elapsed = time.monotonic() - started
        if not result.answer and result.steps:
            # Prefer the most recent actual prose the model produced rather than
            # only the final step, which is often an empty tool-call turn.
            for st in reversed(result.steps):
                if st.text and st.text.strip():
                    result.answer = st.text.strip()
                    break
        if not result.answer and result.stop_reason in ("finished", "looping"):
            # The model verified its work and then fell silent.  Say what the
            # harness knows to be true, clearly labelled as harness-authored so
            # it is never mistaken for the model's own claim.
            result.answer = self._summarise()
        if not result.answer and result.stop_reason == "max_steps":
            result.answer = self._summarise()
        if not result.answer and result.stop_reason.startswith("illegal tool"):
            result.answer = self._summarise()
        # Append a concrete next step whenever the harness had to write the
        # answer for the model -- otherwise the user is left with a description
        # of the failure and nothing to do about it.
        if (result.answer.startswith("[harness summary") or not result.answer):
            tip = self._diagnose()
            if tip:
                result.answer = (result.answer + " " if result.answer else "") + \
                    f"[next step] {tip}"
        self.on_event("done", {
            "stop_reason": result.stop_reason,
            "steps": len(result.steps),
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "elapsed": result.elapsed,
        })
        if self.recorder:
            self.recorder.record("done", {
                "stop_reason": result.stop_reason,
                "answer": result.answer,
                "steps": len(result.steps),
            })
            self.recorder.close()
        return result

    def _summarise(self) -> str:
        """A factual, harness-authored account of what actually happened.

        Used when the model stops without saying anything.  Every clause is
        derived from the ledger -- files really changed, commands that really ran
        with their real exit codes -- so it cannot overstate what was achieved.

        It is also **never empty**.  An earlier version returned ``""`` when the
        run had only read files (nothing written, no commands run, nothing
        refused), which the forced-conclusion path then published as the final
        answer: the evaluation suite recorded a task whose entire result was an
        empty string.  Silence is not an acceptable outcome.
        """
        led = self.manager.ledger
        bits: list[str] = []
        if led.files_written:
            bits.append("changed " + ", ".join(led.files_written[-4:]))
        succeeded = [(c, e) for c, e in led.commands if e == 0]
        if succeeded:
            bits.append("verified with "
                        + "; ".join(f"`{c}` (exit 0)" for c, _ in succeeded[-2:]))
        elif led.commands:
            last = led.commands[-1]
            bits.append(f"the last command failed (`{last[0]}` exit {last[1]})")
        if led.refusals:
            bits.append(f"{len(led.refusals)} action(s) were refused and never ran")
        if led.files_read:
            bits.append("read " + ", ".join(list(led.files_read)[-4:]))
        if not bits:
            return ("[harness summary, not the model's own words] The run ended "
                    "without a written conclusion, and without reading, changing "
                    "or running anything.")
        return ("[harness summary, not the model's own words] The run ended "
                "without a written conclusion. What actually happened: "
                + "; ".join(bits) + ".")

    def _diagnose(self) -> str:
        """A concrete next step, derived from how the run actually failed.

        A summary that only says "3 action(s) were refused" tells the user
        nothing actionable, and one real run ended exactly that way with no hint
        at all.  This turns the ledger into advice.
        """
        led = self.manager.ledger
        tips: list[str] = []

        refused_calls = [r for r in led.refusals if "(" in r]
        if refused_calls:
            tips.append("the refused call(s) were: "
                        + "; ".join(refused_calls[-3:]))
        if led.errors and any("no tool called" in e for e in led.errors):
            tips.append("the model invented a tool name; try again, or ask for a "
                        "smaller step")
        if not led.files_read and self.tool_specs:
            tips.append("nothing was read at all - check that the file is "
                        "attached and that .qwen35/attachments/ is not empty")
        if self._attachments_pending:
            tips.append("the attachment was not read; ask the question again, or "
                        "attach a smaller file so it can be pasted inline")
        return " ".join(tips)

    # ------------------------------------------------------------------
    def _render_action(self, call: ToolCall, text: str) -> str:
        """What the transcript shows as the assistant's turn."""
        lines = [f"<tool_call>", f"<function={call.name}>"]
        for key, value in call.args.items():
            lines.append(f"<parameter={key}>")
            lines.append(str(value))
            lines.append("</parameter>")
        lines.append("</function>")
        lines.append("</tool_call>")
        rendered = "\n".join(lines)
        return f"{text}\n{rendered}".strip() if text.strip() else rendered

    def _refuse_duplicate(self, call: ToolCall, repeats: int,
                          *, final_warning: bool = False) -> None:
        """Tell the model, in-band, that repeating itself is not working.

        A 0.8 B model that has finished the job often keeps re-running its
        verification command instead of concluding.  The first warning suggests
        alternatives; the second is directive, because a hint is not enough.
        """
        self.manager.add_action(self._render_action(call, ""), tool_summary=call.name)
        if self.recorder:
            # Refused steps used to vanish from the transcript entirely, which
            # made a real failure ("3 action(s) were refused") impossible to
            # diagnose after the fact: the log showed step 2 then jumped to
            # "done" at step 5 with nothing in between.
            self.recorder.record("refused", {
                "index": None,
                "call": {"name": call.name, "args": call.args},
                "repeats": repeats,
                "final_warning": final_warning,
                "reason": "duplicate action",
            })
        if final_warning:
            message = (
                f"You have now made this exact call {repeats} times. It did not "
                "work the first time and it will not work now.\n"
                "STOP CALLING TOOLS. Reply with your final answer as plain "
                "prose, with no tool call at all. That ends the run. If you have "
                "not actually verified anything yet, say so honestly."
            )
        else:
            refusals = len(self.manager.ledger.refusals)
            extra = ""
            if refusals:
                extra = (
                    f" Your last {refusals} command(s) were REFUSED by the "
                    "harness and never ran, so they produced no result and "
                    "proved nothing. Read the refusal message and change the "
                    "command to match it."
                )
            message = (
                f"You have already made this exact call {repeats} times and the "
                "result is identical, so it is shown above only once. You do not "
                "need it again." + extra + "\n"
                "Do something different: read a DIFFERENT file, change your "
                "arguments -- or, if you already have what you need, reply with "
                "a plain-prose answer and no tool call, which ends the run."
            )
        self.manager.add_observation(message, meta={"tool": call.name,
                                                    "duplicate": True})
        self.manager.ledger.record_error(f"repeated {call.describe()} {repeats}x")
        # Count it as refused too: it did not run, and the final summary should
        # be able to say so rather than coming out empty.
        self.manager.ledger.record_refusal(call.describe())

    def _update_ledger(self, call: ToolCall, res: ToolResult) -> None:
        led = self.manager.ledger
        led.steps += 1
        if call.name == "read_file" and res.ok:
            rel = str(call.args.get("path", ""))
            led.record_read(rel, int(res.meta.get("start", 0) or 0),
                            int(res.meta.get("end", 0) or 0))
        elif call.name in ("write_file", "edit_file") and res.ok:
            led.record_write(str(call.args.get("path", "")))
        elif call.name == "run_command":
            command = str(call.args.get("command", ""))[:80]
            if res.meta.get("refused"):
                # A refused command never ran, so it must not be recorded as a
                # success.  Recording it with an exit code of 0 previously made
                # the ledger tell the model its commands had worked.
                led.record_refusal(command)
            else:
                led.record_command(command, int(res.meta.get("exit", 0) or 0))

    def _record_turn_meta(self, turn: ParsedTurn) -> None:
        if turn.extra_calls:
            self.manager.ledger.record_error(
                f"emitted {turn.extra_calls + 1} tool calls in one message"
            )
