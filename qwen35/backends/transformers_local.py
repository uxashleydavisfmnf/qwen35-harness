"""
In-process backend using ``transformers``.

This is the zero-setup path: ``torch`` and ``transformers`` are commonly already
present, so the harness can run the model with no separate server, no GGUF and
no downloads beyond the weights themselves.

Trade-off, stated plainly: this backend has **no grammar support**.  Constrained
decoding through ``transformers`` would need a custom logits processor and is not
implemented here, so the harness falls back to its parse-and-repair loop.  Use
``llama-server`` when you want the grammar-level guarantees.

Vision works: images are passed as ``image_url`` data URLs and the processor
handles them.
"""

from __future__ import annotations

import base64
import gc
import io
import os
import threading
from pathlib import Path
from typing import Any

from .base import BackendError, ChatBackend, GenerationResult

__all__ = ["TransformersBackend"]


class TransformersBackend(ChatBackend):
    name = "transformers"
    supports_grammar = False           # documented limitation
    supports_vision = True
    supports_tools = True

    def __init__(
        self,
        model_dir: str | os.PathLike[str] | None = None,
        *,
        model_id: str = "Qwen/Qwen3.5-0.8B",
        dtype: str = "auto",
        device: str = "cpu",
        trust_remote_code: bool = False,
        max_context: int = 262144,
        threads: int | None = None,
    ) -> None:
        self.model_path = str(model_dir) if model_dir else model_id
        self.dtype = dtype
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.max_context = max_context
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()
        self.name = f"transformers:{Path(self.model_path).name}"

        if threads:
            os.environ.setdefault("OMP_NUM_THREADS", str(threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    # -- loading -----------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise BackendError(
                "the transformers backend needs torch and transformers "
                f"(pip install torch transformers): {exc}"
            ) from exc

        try:
            processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )
        except Exception as exc:
            raise BackendError(
                f"could not load the processor for {self.model_path}: {exc}"
            ) from exc

        torch_dtype = {
            "auto": "auto",
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }.get(self.dtype, "auto")

        model = None
        errors: list[str] = []
        # Qwen3.5 is a vision-language conditional-generation model; fall back
        # through the causal-LM classes so older transformers still works for
        # text-only use.
        for loader_name in ("AutoModelForImageTextToText",
                            "AutoModelForVision2Seq",
                            "AutoModelForCausalLM"):
            loader = getattr(__import__("transformers", fromlist=[loader_name]),
                             loader_name, None)
            if loader is None:
                continue
            try:
                model = loader.from_pretrained(
                    self.model_path,
                    dtype=torch_dtype,
                    trust_remote_code=self.trust_remote_code,
                    low_cpu_mem_usage=True,
                )
                break
            except Exception as exc:  # try the next loader
                errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")

        if model is None:
            raise BackendError(
                "could not load the model with any known loader:\n  "
                + "\n  ".join(errors[-3:])
            )

        model.eval()
        if self.device and self.device != "cpu":
            try:
                model.to(self.device)
            except Exception:
                pass

        self._processor = processor
        self._model = model

    # -- health ------------------------------------------------------------
    def health(self) -> tuple[bool, str]:
        try:
            import transformers  # noqa: F401
            import torch
        except ImportError as exc:
            return False, f"transformers/torch unavailable: {exc}"
        where = Path(self.model_path)
        if where.exists():
            return True, f"local weights at {where}"
        return True, f"will fetch {self.model_path} from the Hub on first use"

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
    ) -> GenerationResult:
        with self._lock:
            self._ensure_loaded()
            try:
                import torch
            except ImportError as exc:  # pragma: no cover
                raise BackendError(f"torch is required: {exc}") from exc

            assert self._processor is not None and self._model is not None

            # images are inlined as data URLs; the processor wants raw bytes
            prepared = self._prepare_messages(messages)

            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if tools:
                template_kwargs["tools"] = tools
            # Qwen3.5's template accepts enable_thinking
            template_kwargs["enable_thinking"] = bool(thinking)

            try:
                text = self._processor.apply_chat_template(
                    prepared, **template_kwargs
                )
            except TypeError:
                template_kwargs.pop("enable_thinking", None)
                text = self._processor.apply_chat_template(
                    prepared, **template_kwargs
                )

            try:
                inputs = self._processor(text=[text], return_tensors="pt",
                                         padding=True)
            except Exception:
                tok = self._processor.tokenizer
                inputs = tok([text], return_tensors="pt", padding=True)

            if self.device and self.device != "cpu":
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}

            if seed is not None:
                torch.manual_seed(int(seed))

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(max_tokens),
                "do_sample": temperature > 0,
                "pad_token_id": getattr(self._processor.tokenizer, "pad_token_id", None),
            }
            if temperature > 0:
                gen_kwargs.update(temperature=float(temperature),
                                  top_p=float(top_p), top_k=int(top_k))
                if repeat_penalty and repeat_penalty != 1.0:
                    gen_kwargs["repetition_penalty"] = float(repeat_penalty)

            prompt_len = int(inputs["input_ids"].shape[-1])
            if prompt_len + max_tokens > self.max_context:
                raise BackendError(
                    f"prompt is {prompt_len} tokens and the reply budget is "
                    f"{max_tokens}, which exceeds the {self.max_context}-token "
                    "model limit. Lower --context or shorten the task."
                )

            with torch.inference_mode():
                output = self._model.generate(**inputs, **gen_kwargs)

            new_tokens = output[0][prompt_len:]
            decoded = self._processor.tokenizer.decode(
                new_tokens, skip_special_tokens=False
            )

            # trim at the first stop token, keeping any  response marker so the
            # parser can still split reasoning out
            for token in ("<|im_end|>", "<|endoftext|>"):
                if token in decoded:
                    decoded = decoded.split(token)[0]

            return GenerationResult(
                text=decoded.strip(),
                prompt_tokens=prompt_len,
                completion_tokens=int(new_tokens.shape[-1]),
                finish_reason="stop",
                grammar_used=False,
                thinking_used=bool(thinking),
                backend=self.name,
            )

    # -- helpers -----------------------------------------------------------
    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI image_url parts into the processor's image form."""
        out: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                out.append(dict(msg))
                continue
            parts: list[dict[str, Any]] = []
            for part in content:
                ptype = part.get("type")
                if ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        _, _, b64 = url.partition(",")
                        try:
                            raw = base64.b64decode(b64)
                        except Exception:
                            continue
                        try:
                            from PIL import Image

                            parts.append({"type": "image",
                                          "image": Image.open(io.BytesIO(raw)).convert("RGB")})
                        except Exception:
                            continue
                elif ptype == "text":
                    parts.append({"type": "text", "text": part.get("text", "")})
            out.append({"role": msg.get("role"), "content": parts})
        return out

    def close(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
