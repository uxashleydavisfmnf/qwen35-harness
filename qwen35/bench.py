"""
Throughput and prefix-cache measurement.

On a CPU box, **prefill dominates everything**.  A 1 500-token system prompt costs
about twelve seconds cold and half a second cached, so the single most important
number for this harness is the prefix-cache hit rate — and until now nothing
measured it.

This module produces the numbers, in four parts:

``cold``
    A prompt with a unique nonce, so nothing can be cached.  Gives honest prefill
    tokens/second and time-to-first-token.
``cached``
    The identical request again.  If this is not near-instant, prefix caching is
    broken and every agent step is paying full price.
``grow``
    A realistic agent conversation: a byte-stable system prompt, then turns
    appended one at a time.  Reports the cache hit ratio **per step**, which is
    what the loop actually experiences.
``compact``
    The same conversation with its *middle* rewritten — exactly what observation
    masking does.  This is where a hybrid recurrent model hurts: llama.cpp cannot
    rewind into the middle of the context, so this measures how much the context
    checkpoints actually buy.
"""

from __future__ import annotations

import argparse
import json
import secrets
import statistics
import time
from pathlib import Path
from typing import Any

from .backends.openai_compat import OpenAICompatBackend
from .llama_runtime import load_server_key

__all__ = ["run_bench", "main"]

_FILLER = ("the quick brown fox jumps over the lazy dog and then sits down "
           "quietly beside the river bank while the sun sets slowly. ")
_SYSTEM = ("You are a coding assistant working inside a workspace. Answer "
           "briefly. " + _FILLER * 20)


def _text_of_tokens(approx_tokens: int) -> str:
    """Roughly ``approx_tokens`` tokens of filler (4 chars/token for English)."""
    return (_FILLER * (approx_tokens // 12 + 1))[: max(40, approx_tokens * 4)]


def _ask(backend: OpenAICompatBackend, messages: list[dict[str, Any]],
         *, max_tokens: int = 8) -> tuple[Any, float]:
    started = time.monotonic()
    result = backend.generate(messages, temperature=0.0, max_tokens=max_tokens,
                              thinking=False)
    return result, time.monotonic() - started


def run_bench(
    *,
    base_url: str = "http://127.0.0.1:8080",
    api_key: str = "",
    cold_tokens: int = 1500,
    grow_steps: int = 8,
    gen_tokens: int = 128,
    progress: Any = None,
) -> dict[str, Any]:
    """Collect the measurements.  Returns a JSON-serialisable report."""
    say = progress or (lambda _m: None)
    key = api_key or load_server_key()
    backend = OpenAICompatBackend(base_url, api_key=key)
    ok, message = backend.health()
    if not ok:
        raise RuntimeError(f"the model server is not answering at {base_url}: {message}")

    out: dict[str, Any] = {"base_url": base_url, "server": message}

    # ---- 1. cold prefill ------------------------------------------------
    say("measuring cold prefill...")
    nonce = secrets.token_hex(8)
    cold_messages = [
        {"role": "system", "content": f"[{nonce}] " + _text_of_tokens(cold_tokens)},
        {"role": "user", "content": "Say OK."},
    ]
    cold, cold_s = _ask(backend, cold_messages)
    out["cold"] = {
        "prompt_tokens": cold.prompt_tokens,
        "cached_tokens": cold.cached_tokens,
        "seconds": round(cold_s, 3),
        "prefill_tokens_per_second": round(cold.prompt_tokens / cold_s, 1)
        if cold_s > 0 else None,
    }

    # ---- 2. warm repeat -------------------------------------------------
    say("measuring a repeated prompt (prefix cache)...")
    warm, warm_s = _ask(backend, cold_messages)
    out["cached"] = {
        "prompt_tokens": warm.prompt_tokens,
        "cached_tokens": warm.cached_tokens,
        "seconds": round(warm_s, 3),
        "cache_hit_ratio": round(warm.cache_hit_ratio, 4),
        "speedup_vs_cold": round(cold_s / warm_s, 1) if warm_s > 0 else None,
    }

    # ---- 3. a growing conversation --------------------------------------
    say(f"measuring {grow_steps} agent-like steps...")
    messages: list[dict[str, Any]] = [{"role": "system", "content": _SYSTEM},
                                      {"role": "user", "content": "Start."}]
    grow: list[dict[str, Any]] = []
    for i in range(1, grow_steps + 1):
        messages.append({"role": "assistant", "content": f"step {i} reply. " + _text_of_tokens(120)})
        messages.append({"role": "user", "content": f"step {i} result. " + _text_of_tokens(220)})
        result, seconds = _ask(backend, messages)
        grow.append({
            "step": i,
            "prompt_tokens": result.prompt_tokens,
            "cached_tokens": result.cached_tokens,
            "cache_hit_ratio": round(result.cache_hit_ratio, 4),
            "seconds": round(seconds, 3),
        })
        say(f"  step {i}: {result.cached_tokens}/{result.prompt_tokens} cached "
            f"in {seconds:.2f}s")
    hits = [g["cache_hit_ratio"] for g in grow]
    out["grow"] = {
        "steps": grow,
        "final_prompt_tokens": grow[-1]["prompt_tokens"] if grow else 0,
        "mean_cache_hit_ratio": round(statistics.fmean(hits), 4) if hits else 0.0,
        "mean_step_seconds": round(statistics.fmean([g["seconds"] for g in grow]), 3)
        if grow else 0.0,
        "total_seconds": round(sum(g["seconds"] for g in grow), 3),
    }

    # ---- 4. compaction: rewrite the middle ------------------------------
    # This is the case the context checkpoints exist for.  Masking an old
    # observation changes the prompt somewhere in the middle, and a hybrid
    # recurrent model cannot rewind there -- so the cost of this request is the
    # real cost of compaction.
    if len(messages) >= 8:
        say("measuring the cost of compacting the middle of the transcript...")
        compacted = [dict(m) for m in messages]
        for index in range(3, max(4, len(compacted) // 2), 2):
            compacted[index] = {"role": compacted[index]["role"],
                                "content": "[earlier tool result elided]"}
        result, seconds = _ask(backend, compacted)
        out["compact"] = {
            "prompt_tokens": result.prompt_tokens,
            "cached_tokens": result.cached_tokens,
            "cache_hit_ratio": round(result.cache_hit_ratio, 4),
            "seconds": round(seconds, 3),
            "note": ("low cache hit here is expected for a hybrid recurrent "
                     "model; context checkpoints are what mitigate it"),
        }

    # ---- 5. generation throughput ---------------------------------------
    say(f"measuring generation of {gen_tokens} tokens...")
    result, seconds = _ask(
        backend,
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Count from one to two hundred in words, "
                                     "comma separated, nothing else."}],
        max_tokens=gen_tokens,
    )
    out["generation"] = {
        "completion_tokens": result.completion_tokens,
        "seconds": round(seconds, 3),
        "tokens_per_second": round(result.completion_tokens / seconds, 1)
        if seconds > 0 and result.completion_tokens else None,
    }
    return out


def _print_report(report: dict[str, Any]) -> None:
    print()
    print("=" * 74)
    print("  qwen35 benchmark — prefill, prefix cache and generation")
    print("=" * 74)
    print(f"  server: {report['base_url']}  ({report.get('server', '')})")

    cold = report.get("cold", {})
    if cold:
        print(f"\n  cold prefill")
        print(f"    {cold['prompt_tokens']:,} tokens in {cold['seconds']}s "
              f"= {cold['prefill_tokens_per_second']} tok/s")

    cached = report.get("cached", {})
    if cached:
        print(f"\n  repeated prompt (prefix cache)")
        print(f"    {cached['cached_tokens']:,}/{cached['prompt_tokens']:,} cached "
              f"({cached['cache_hit_ratio'] * 100:.1f}%)  "
              f"{cached['seconds']}s  "
              f"{cached.get('speedup_vs_cold')}x faster than cold")

    grow = report.get("grow", {})
    if grow:
        print(f"\n  growing conversation ({len(grow['steps'])} agent-like steps)")
        print(f"    {'step':>5} {'prompt':>8} {'cached':>8} {'hit':>7} {'sec':>7}")
        for step in grow["steps"]:
            print(f"    {step['step']:>5} {step['prompt_tokens']:>8,} "
                  f"{step['cached_tokens']:>8,} "
                  f"{step['cache_hit_ratio'] * 100:>6.1f}% {step['seconds']:>7.2f}")
        print(f"    mean hit {grow['mean_cache_hit_ratio'] * 100:.1f}%, "
              f"mean step {grow['mean_step_seconds']}s, "
              f"total {grow['total_seconds']}s")

    compact = report.get("compact", {})
    if compact:
        print(f"\n  after compacting the middle of the transcript")
        print(f"    {compact['cached_tokens']:,}/{compact['prompt_tokens']:,} cached "
              f"({compact['cache_hit_ratio'] * 100:.1f}%)  {compact['seconds']}s")

    gen = report.get("generation", {})
    if gen:
        print(f"\n  generation")
        print(f"    {gen['completion_tokens']} tokens in {gen['seconds']}s "
              f"= {gen['tokens_per_second']} tok/s")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qwen35 bench",
        description="Measure prefill, prefix-cache and generation on the "
                    "running server.",
    )
    parser.add_argument("--base-url", dest="base_url",
                        default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", dest="api_key", default="")
    parser.add_argument("--cold-tokens", dest="cold_tokens", type=int, default=1500)
    parser.add_argument("--grow-steps", dest="grow_steps", type=int, default=8)
    parser.add_argument("--gen-tokens", dest="gen_tokens", type=int, default=128)
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args(argv)

    try:
        report = run_bench(
            base_url=args.base_url, api_key=args.api_key,
            cold_tokens=args.cold_tokens, grow_steps=args.grow_steps,
            gen_tokens=args.gen_tokens,
            progress=lambda m: print(f"  {m}"),
        )
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1

    _print_report(report)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  json written to {args.json_out}")
    return 0
