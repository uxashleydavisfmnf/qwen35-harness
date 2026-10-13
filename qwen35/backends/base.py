"""Backend interface shared by every inference path."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ChatBackend", "GenerationResult", "BackendError"]


class BackendError(Exception):
    """Raised when the backend cannot serve a request."""


@dataclass
class GenerationResult:
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: how many prompt tokens the server served from its prefix cache.  This is
    #: the number that matters on a CPU: prefill dominates, and a cache hit is
    #: the difference between 0.5 s and 12 s for the same prompt.
    cached_tokens: int = 0
    finish_reason: str = ""
    grammar_used: bool = False
    thinking_used: bool = False
    backend: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_ratio(self) -> float:
        if self.prompt_tokens <= 0:
            return 0.0
        return min(1.0, self.cached_tokens / self.prompt_tokens)


class ChatBackend(abc.ABC):
    """A source of completions.

    ``generate`` receives already-rendered messages and, where the backend can
    honour it, a GBNF grammar and the model's own tool schema.
    """

    name: str = "backend"
    supports_grammar: bool = False
    supports_vision: bool = False
    supports_tools: bool = False

    @abc.abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        grammar: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_tokens: int = 768,
        stop: tuple[str, ...] = (),
        thinking: bool = False,
        repeat_penalty: float = 1.05,
        seed: int | None = None,
    ) -> GenerationResult:
        raise NotImplementedError

    def health(self) -> tuple[bool, str]:
        """Return (ok, message).  Used by ``qwen35 doctor``."""
        return True, "no health check implemented"

    def close(self) -> None:
        return None

    def describe(self) -> str:
        caps = []
        if self.supports_tools:
            caps.append("tools")
        if self.supports_grammar:
            caps.append("grammar")
        if self.supports_vision:
            caps.append("vision")
        return f"{self.name} ({', '.join(caps) or 'text only'})"
