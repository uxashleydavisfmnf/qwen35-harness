"""Backend selection and construction."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from ..config import RuntimeConfig, discover_gguf
from .base import BackendError, ChatBackend, GenerationResult
from .mock import MockBackend, scripted_answer, scripted_tool_call
from .openai_compat import OpenAICompatBackend
from .transformers_local import TransformersBackend

__all__ = [
    "ChatBackend",
    "GenerationResult",
    "BackendError",
    "MockBackend",
    "OpenAICompatBackend",
    "TransformersBackend",
    "scripted_tool_call",
    "scripted_answer",
    "build_backend",
    "autodetect_backend",
]


def _is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "")


def resolve_api_key(cfg: RuntimeConfig) -> str:
    """Pick the API key to use, resolving the stored one for local servers.

    This belongs here rather than in each front end.  `qwen35 serve` writes a
    random key to ``~/.qwen35/server.json`` and the server rejects unauthenticated
    requests -- so *every* caller that builds a config by hand (the evaluation
    suite, a library user, a future front end) used to get an opaque HTTP 401.
    Only the CLI and the GUI happened to populate the key themselves.
    """
    if cfg.api_key:
        return cfg.api_key
    if not _is_local_url(cfg.base_url):
        return ""            # a remote endpoint must supply its own key
    from ..llama_runtime import load_server_key

    return load_server_key()


def build_backend(cfg: RuntimeConfig, *, allow_mock: bool = False) -> ChatBackend:
    """Construct the backend named by ``cfg.backend``.

    ``auto`` probes in order of preference: a running local server, then GGUF
    weights (if ``llama-server`` is available), then an in-process transformers
    model, then a clear error rather than a silent fallback to something fake.
    """
    choice = (cfg.backend or "auto").lower()

    if choice == "mock":
        if not allow_mock:
            raise BackendError(
                "the mock backend must be requested explicitly (--backend mock)"
            )
        return MockBackend()

    if choice == "auto":
        choice = autodetect_backend(cfg)
        if choice == "none":
            raise BackendError(
                "no usable backend found.\n"
                "  - start a server and retry:  qwen35 serve\n"
                "  - or download weights:       qwen35 fetch\n"
                "  - or install llama.cpp:      qwen35 install-runtime\n"
                "  - or use transformers:       --backend transformers"
            )

    if choice in ("llamacpp", "llama.cpp", "llama-server", "server"):
        return OpenAICompatBackend(
            cfg.base_url,
            model=cfg.model_id,
            api_key=resolve_api_key(cfg),
            label=f"llama-server@{cfg.base_url}",
        )

    if choice in ("openai", "openai-compat", "remote", "ollama", "lmstudio", "vllm"):
        if not cfg.base_url:
            raise BackendError("--base-url is required for a remote backend")
        key = resolve_api_key(cfg)
        return OpenAICompatBackend(
            cfg.base_url,
            model=cfg.model_id,
            api_key=key,
            # llama.cpp-only extra fields; harmless for a local server, rejected
            # by some hosted endpoints when a key is in play
            native_extras=not (key and not _is_local_url(cfg.base_url)),
            label=f"{choice}@{cfg.base_url}",
        )

    if choice in ("transformers", "hf", "local"):
        model_dir = cfg.model_dir or None
        if model_dir and not Path(model_dir).exists():
            model_dir = None
        return TransformersBackend(
            model_dir,
            model_id=cfg.model_id,
            max_context=cfg.context_window if cfg.context_window > 32768 else 262144,
        )

    raise BackendError(
        f"unknown backend {cfg.backend!r}; choose from: auto, llamacpp, openai, "
        "transformers, mock"
    )


def autodetect_backend(cfg: RuntimeConfig) -> str:
    """Return the best backend name, or ``"none"``."""
    # 1. an already-running local server
    if cfg.base_url:
        probe = OpenAICompatBackend(cfg.base_url, model=cfg.model_id,
                                    api_key=resolve_api_key(cfg))
        ok, _ = probe.health()
        if ok:
            return "llamacpp" if _is_local_url(cfg.base_url) else "openai"

    # 2. GGUF weights + a llama-server binary
    model_dir = Path(cfg.model_dir) if cfg.model_dir else None
    if model_dir and model_dir.is_dir():
        weights, _mmproj = discover_gguf(model_dir)
        if weights is not None and _find_llama_server():
            return "llamacpp"

    # 3. in-process transformers, if the library is present
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401

        if model_dir and model_dir.is_dir():
            if any(model_dir.glob("*.safetensors")) or any(
                model_dir.glob("*.bin")
            ):
                return "transformers"
        elif os.environ.get("QWEN35_ALLOW_HUB_DOWNLOAD"):
            return "transformers"
    except ImportError:
        pass

    return "none"


def _find_llama_server() -> str | None:
    """Locate llama-server on PATH or in the harness's runtime directory."""
    import shutil

    for name in ("llama-server", "llama-server.exe"):
        hit = shutil.which(name)
        if hit:
            return hit
    from ..llama_runtime import installed_server

    found = installed_server()
    return str(found) if found else None
