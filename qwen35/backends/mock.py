"""
A deterministic backend.

This exists so the *harness* can be tested without the model: the agent loop,
the workspace confinement, the permission screener, the grammar, the parser and
the context compactor are all exercised against scripted completions.  Every
unit test in ``tests/`` runs on this, which means the security properties are
verifiable on a machine with no weights, no GPU and no network.

It is also useful for reproducing a failing trajectory: record a session, then
replay it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .base import BackendError, ChatBackend, GenerationResult

__all__ = ["MockBackend", "scripted_tool_call", "scripted_answer"]


def scripted_tool_call(name: str, **params: Any) -> str:
    """Build a completion in the model's native XML envelope."""
    body = [f"<tool_call>", f"<function={name}>"]
    for key, value in params.items():
        body.append(f"<parameter={key}>")
        body.append(str(value))
        body.append("</parameter>")
    body.append("</function>")
    body.append("</tool_call>")
    return "\n".join(body)


def scripted_answer(text: str, *, thinking: str = "") -> str:
    if thinking:
        return f"thinking...\n</think>\n\n{text}"
    return text


@dataclass
class MockBackend(ChatBackend):
    """Replays a fixed list of completions.

    ``script`` may be a list of strings, or a callable that receives the
    rendered messages and returns the next completion (for adaptive tests).
    """

    script: Sequence[str] | Callable[[list[dict[str, Any]]], str] = field(
        default_factory=list)
    name: str = "mock"
    supports_grammar: bool = True
    supports_vision: bool = True
    supports_tools: bool = True
    fail_after: int = -1

    _calls: int = 0
    _seen: list[list[dict[str, Any]]] = field(default_factory=list)

    def generate(self, messages: list[dict[str, Any]], *, tools=None,
                 grammar=None, temperature=0.7, top_p=0.8, top_k=20,
                 max_tokens=768, stop=(), thinking=False,
                 repeat_penalty=1.05, seed=None) -> GenerationResult:
        self._seen.append(messages)
        index = self._calls
        self._calls += 1

        if self.fail_after >= 0 and index >= self.fail_after:
            raise BackendError("mock backend: scripted failure")

        if callable(self.script):
            text = self.script(messages)
        elif index < len(self.script):
            text = self.script[index]
        else:
            text = scripted_answer(
                "I have run out of scripted steps. Stopping here."
            )

        return GenerationResult(
            text=text,
            prompt_tokens=sum(len(str(m.get("content", ""))) // 4 for m in messages),
            completion_tokens=len(text) // 4,
            # Simulate prefix-cache reuse: everything but the newest turn is
            # served from cache.  This makes the cache plumbing testable without
            # a real server.
            cached_tokens=(0 if index == 0 else
                           max(0, sum(len(str(m.get("content", ""))) // 4
                                      for m in messages[:-2]))),
            finish_reason="stop",
            grammar_used=bool(grammar) and self.supports_grammar,
            thinking_used=thinking,
            backend=self.name,
            extra={"index": index},
        )

    # -- test helpers ------------------------------------------------------
    @property
    def calls(self) -> int:
        return self._calls

    @property
    def seen(self) -> list[list[dict[str, Any]]]:
        return self._seen

    def last_prompt(self) -> str:
        if not self._seen:
            return ""
        return "\n".join(str(m.get("content", "")) for m in self._seen[-1])

    def health(self) -> tuple[bool, str]:
        return True, f"scripted, {len(self.script) if not callable(self.script) else '?'} steps"
