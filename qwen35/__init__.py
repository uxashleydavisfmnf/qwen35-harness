"""
qwen35 -- a lightweight agent harness built specifically for Qwen3.5-0.8B.

Why this exists rather than "just use a big framework": Qwen3.5-0.8B is a 0.8 B
vision-language model with a 262 144-token window and a hybrid Gated-DeltaNet +
gated-attention layout.  General-purpose harnesses assume a model that can follow
long instructions and recover from its own mistakes.  A 0.8 B model cannot, so
this harness moves the reliability burden from the prompt into the runtime:

* **Grammar-constrained decoding** makes malformed or hallucinated tool calls
  impossible rather than unlikely (:mod:`qwen35.grammar`).
* **Validate-and-revert** means a broken edit never lands (:mod:`qwen35.tools`).
* **Observation masking + a harness ledger** keep a small model oriented without
  ever asking it to summarize itself (:mod:`qwen35.context`).
* **Strict confinement** limits it to one operator-chosen workspace that can
  never be a filesystem root, with an argv-based, allowlisted command policy that
  never spawns a shell (:mod:`qwen35.workspace`, :mod:`qwen35.permissions`).
* **Native format fidelity** uses the model's own chat template for tools and for
  ``enable_thinking``, instead of re-inventing the prompt (:mod:`qwen35.prompt`).

Quick start::

    python -m qwen35 doctor                 # what is available here
    python -m qwen35 install-runtime        # fetch llama.cpp (CPU build)
    python -m qwen35 fetch                  # download the GGUF weights
    python -m qwen35 serve                  # start llama-server
    python -m qwen35 run --workspace ./myproject "add tests for utils.py"
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
