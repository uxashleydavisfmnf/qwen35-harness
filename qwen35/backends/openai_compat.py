"""
OpenAI-compatible HTTP backend -- the primary path (llama-server).

``llama-server`` speaks the OpenAI chat-completions dialect and adds a few
extras this harness relies on:

* ``grammar``               -- GBNF, the constrained-decoding win
* ``chat_template_kwargs``  -- drives Qwen3.5's native ``enable_thinking``
* ``cache_prompt``          -- prefix-cache reuse, which the byte-stable system
                               prompt is designed to exploit
* ``top_k`` / ``min_p`` / ``repeat_penalty`` -- sampler controls the OpenAI
                               schema does not cover

It also works unchanged against Ollama, LM Studio, vLLM and any hosted
OpenAI-compatible endpoint; the extra fields are simply ignored there.

Standard library only (``urllib``), so the harness has no hard dependencies.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from .base import BackendError, ChatBackend, GenerationResult

__all__ = ["OpenAICompatBackend"]


class OpenAICompatBackend(ChatBackend):
    name = "openai-compat"
    supports_grammar = True
    supports_vision = True
    supports_tools = True

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        model: str = "local",
        api_key: str = "",
        timeout: float = 600.0,
        native_extras: bool = True,
        label: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        # when False, only strict OpenAI fields are sent (for strict servers)
        self.native_extras = native_extras
        self.name = label or f"openai-compat@{self.base_url}"

    # -- plumbing ----------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json",
             "Accept": "application/json",
             "User-Agent": "qwen35-harness/1.0"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(),
                                    method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:800]
            except Exception:
                pass
            raise BackendError(
                f"{url} returned HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise BackendError(
                f"cannot reach the model server at {url}: {exc.reason}. "
                "Start it with `qwen35 serve`, or point --base-url at a running "
                "server."
            ) from exc
        except TimeoutError as exc:
            raise BackendError(
                f"{url} timed out after {self.timeout:.0f}s"
            ) from exc

        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError(
                f"{url} returned a non-JSON body: {body[:300]!r}"
            ) from exc

    # -- health ------------------------------------------------------------
    def health(self) -> tuple[bool, str]:
        for path in ("/health", "/v1/models"):
            try:
                req = urllib.request.Request(f"{self.base_url}{path}",
                                             headers=self._headers())
                with urllib.request.urlopen(req, timeout=8) as resp:
                    body = resp.read().decode("utf-8", "replace")
                if path == "/health":
                    try:
                        status = json.loads(body).get("status", "ok")
                    except json.JSONDecodeError:
                        status = "ok"
                    return status in ("ok", "no slot available"), f"{path}: {status}"
                return True, f"{path}: reachable"
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                last = exc
        return False, f"{self.base_url} is not answering ({last})"

    # -- generation --------------------------------------------------------
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
        stream_callback: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if stop:
            payload["stop"] = list(stop)
        if seed is not None:
            payload["seed"] = int(seed)

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if self.native_extras:
            payload["top_k"] = int(top_k)
            payload["repeat_penalty"] = float(repeat_penalty)
            payload["cache_prompt"] = True
            payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
            if grammar:
                payload["grammar"] = grammar

        data = self._post("/v1/chat/completions", payload)

        choices = data.get("choices") or []
        if not choices:
            raise BackendError(
                f"the server returned no choices: {json.dumps(data)[:400]}"
            )
        message = choices[0].get("message") or {}
        text = message.get("content") or ""

        # llama-server (and vLLM/SGLang) parse the model's <tool_call> XML and
        # move it into `message.tool_calls`, leaving `content` empty.  Rebuild
        # the model's native XML so exactly one representation ever reaches the
        # parser, whatever the server decides to do with it.
        native_calls = message.get("tool_calls") or []
        if native_calls and "<tool_call>" not in text:
            text = (text + "\n" + self._render_native_calls(native_calls)).strip()

        # some servers put reasoning in a separate field
        reasoning = message.get("reasoning_content") or ""
        if reasoning and " response" not in text:
            text = f" thinking\n{reasoning}\n</think>\n\n{text}"

        usage = data.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return GenerationResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cached_tokens=int(details.get("cached_tokens") or 0),
            finish_reason=str(choices[0].get("finish_reason") or ""),
            grammar_used=bool(grammar) and self.native_extras,
            thinking_used=bool(thinking),
            backend=self.name,
            extra={"id": data.get("id", "")},
        )

    # -- convenience -------------------------------------------------------
    @staticmethod
    def _render_native_calls(calls: list[dict[str, Any]]) -> str:
        """Turn OpenAI-style tool_calls back into Qwen3.5's native XML.

        The harness's parser only has to understand one format, and it is the
        one the model was trained on.
        """
        blocks: list[str] = []
        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name") or ""
            if not name:
                continue
            raw_args = fn.get("arguments")
            args: dict[str, Any] = {}
            if isinstance(raw_args, str) and raw_args.strip():
                try:
                    parsed = json.loads(raw_args)
                    if isinstance(parsed, dict):
                        args = parsed
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
            elif isinstance(raw_args, dict):
                args = raw_args

            lines = ["<tool_call>", f"<function={name}>"]
            for key, value in args.items():
                lines.append(f"<parameter={key}>")
                if isinstance(value, (dict, list)):
                    lines.append(json.dumps(value, ensure_ascii=False))
                else:
                    lines.append(str(value))
                lines.append("</parameter>")
            lines.append("</function>")
            lines.append("</tool_call>")
            blocks.append("\n".join(lines))
        return "\n".join(blocks)

    def wait_until_ready(self, timeout: float = 120.0,
                         poll: float = 1.0) -> tuple[bool, str]:
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            ok, msg = self.health()
            if ok:
                return True, msg
            last = msg
            time.sleep(poll)
        return False, f"not ready after {timeout:.0f}s ({last})"
