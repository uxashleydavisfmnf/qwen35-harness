"""
Configuration, model profiles and permission profiles for the Qwen3.5-0.8B harness.

All hard numbers in ``QWEN35_08B`` were verified against the model's published
``config.json`` / ``tokenizer_config.json`` (Qwen/Qwen3.5-0.8B, created 2026-02-28).
Nothing here is guessed: if you re-derive a value, cite the file it came from.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

# --------------------------------------------------------------------------
# Ground truth: token ids from Qwen/Qwen3.5-0.8B tokenizer_config.json
# --------------------------------------------------------------------------
TOK_ENDOFTEXT = 248044        # <|endoftext|>   (also text_config.eos_token_id)
TOK_IM_START = 248045         # <|im_start|>
TOK_IM_END = 248046           # <|im_end|>      (tokenizer_config eos_token)
TOK_VISION_START = 248053
TOK_VISION_END = 248054
TOK_IMAGE_PAD = 248056
TOK_VIDEO_PAD = 248057
TOK_TOOL_CALL = 248058        # <tool_call>     -> a SINGLE token
TOK_TOOL_CALL_END = 248059    # </tool_call>    -> a SINGLE token
TOK_TOOL_RESPONSE = 248066    # <tool_response> -> a SINGLE token
TOK_TOOL_RESPONSE_END = 248067
TOK_THINK = 248068            #  thinking
TOK_THINK_END = 248069        # </think>

VISION_PATCH_SIZE = 16
VISION_MERGE_SIZE = 2
# One visual token covers (16*2) x (16*2) = 32x32 pixels -> tokens = W*H/1024
VISION_PIXELS_PER_TOKEN = (VISION_PATCH_SIZE * VISION_MERGE_SIZE) ** 2

DriveScope = Literal["workspace"]


@dataclass(frozen=True)
class ModelProfile:
    """Everything the harness needs to know about the served model."""

    name: str
    context_window: int
    supports_vision: bool
    stop_tokens: tuple[str, ...]
    # hybrid-attention geometry, used purely to *report* KV economics
    num_hidden_layers: int = 24
    full_attention_layers: int = 6
    num_key_value_heads: int = 2
    head_dim: int = 256
    mtp_layers: int = 1
    thinking_capable: bool = True
    native_tool_format: str = "xml"

    # -- derived -----------------------------------------------------------
    def kv_bytes_per_token(self, bytes_per_element: float = 2.0) -> float:
        """KV bytes per token for the *full attention* layers only.

        The 18 linear-attention (Gated DeltaNet) layers keep a constant-size
        recurrent state, so they contribute O(1) memory instead of O(n).  Only
        the 6 ``full_attention`` layers grow a KV cache.  That is why a 262144
        token window is genuinely affordable on a 16 GB CPU box.
        """
        return (
            self.full_attention_layers
            * self.num_key_value_heads
            * self.head_dim
            * 2  # K and V
            * bytes_per_element
        )

    def kv_bytes(self, tokens: int, bytes_per_element: float = 2.0) -> float:
        return self.kv_bytes_per_token(bytes_per_element) * tokens

    def approx_visual_tokens(self, width: int, height: int) -> int:
        if width <= 0 or height <= 0:
            return 0
        return max(1, (width * height) // VISION_PIXELS_PER_TOKEN)

    @property
    def effective_context_fraction(self) -> float:
        """Full-attention layers as a fraction of all layers.

        Used in the README/`doctor` output to explain the cache economics.
        """
        return self.full_attention_layers / max(1, self.num_hidden_layers)


QWEN35_08B = ModelProfile(
    name="Qwen3.5-0.8B",
    context_window=262144,
    supports_vision=True,
    stop_tokens=("<|im_end|>", "<|endoftext|>"),
    num_hidden_layers=24,
    full_attention_layers=6,
    num_key_value_heads=2,
    head_dim=256,
    mtp_layers=1,
    thinking_capable=True,
    native_tool_format="xml",
)


# --------------------------------------------------------------------------
# Permission profiles -- "not too much privilege, but the basics are granted"
# --------------------------------------------------------------------------
@dataclass
class Permissions:
    """Capability grant handed to the model.

    A 0.8B model is not a trustworthy principal: it misreads intent, loops, and
    occasionally invents destructive commands.  The default posture is therefore
    *write where it was told, run what is provably safe, touch nothing else*.
    """

    # filesystem
    read: bool = True
    write: bool = True           # create + overwrite, inside the workspace only
    edit: bool = True            # surgical find/replace
    delete: bool = False         # never by default; no tool exposes it anyway
    move: bool = False
    # process execution
    run_command: bool = True
    allow_network_commands: bool = False
    allow_package_install: bool = False
    allow_shell_metachars: bool = False
    # limits
    max_read_bytes: int = 400_000
    max_write_bytes: int = 512_000
    # Tool output handed back to the model.  SWE-agent's viewer ablation found a
    # tuned ~100-line window (18.0%) beat both a 30-line window (14.3%) and the
    # full file (12.7%), so this is deliberately small.
    max_output_bytes: int = 6_000
    max_command_seconds: int = 30
    max_commands_per_run: int = 40
    # misc
    allow_vision: bool = True
    history_lines: int = 200

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES: dict[str, Permissions] = {
    # read-and-analyse only, no writes at all
    "readonly": Permissions(write=False, edit=False, run_command=False),
    # THE DEFAULT.
    #
    # This used to be called "safe" with much tighter limits, and there was a
    # second, wider "standard" above it.  In practice "safe" was simply too small
    # to do real work -- reading a 400 KB file, 30-second commands, 40 commands
    # per run -- so people immediately switched to "standard" and the smaller tier
    # only got in the way.  The two are now one profile, and it is the default.
    #
    # Network and package installation are allowed because a coding agent without
    # them cannot follow a README.  The containment is not "no network"; it is
    # that everything a child process writes -- packages, caches, temp files,
    # `$HOME` -- lands inside `.qwen35/` in the workspace, as the current user,
    # with no elevation.  Delete the workspace and every trace is gone.
    "standard": Permissions(
        max_read_bytes=2_000_000,
        max_write_bytes=4_000_000,
        max_output_bytes=20_000,
        max_command_seconds=120,
        max_commands_per_run=200,
        allow_network_commands=True,
        allow_package_install=True,
        allow_shell_metachars=False,   # still no pipes / && / redirection
    ),
    # escape hatch for power users; still cannot leave the workspace on disk
    "trusted": Permissions(
        max_read_bytes=20_000_000,
        max_write_bytes=20_000_000,
        max_output_bytes=1_000_000,
        max_command_seconds=600,
        max_commands_per_run=1000,
        allow_package_install=True,
        allow_network_commands=True,
        allow_shell_metachars=True,
    ),
}

#: The profile used when none is named.
DEFAULT_PROFILE = "standard"

#: Names kept working for older scripts and configs.  "safe" was merged into
#: "standard"; silently resolving it keeps bookmarks, launcher flags and saved
#: sessions from breaking, while making clear it is no longer a separate tier.
_PROFILE_ALIASES = {
    "safe": "standard",       # merged: safe was too weak to be useful
    "default": "standard",
}


def get_permissions(profile: str) -> Permissions:
    name = _PROFILE_ALIASES.get((profile or "").lower(), (profile or "").lower())
    if name not in PROFILES:
        raise KeyError(
            f"unknown permission profile {profile!r}; "
            f"choose from {sorted(set(PROFILES) | set(_PROFILE_ALIASES))}")
    # return a copy so callers cannot mutate the shared table
    return Permissions(**PROFILES[name].to_dict())


#: Everything ``--profile`` will accept, aliases included.
#:
#: The aliases have to be in the *choices* as well as the resolver.  They were
#: not, and the launcher passes ``--profile safe``, so the GUI died at startup
#: with "invalid choice: 'safe'" -- an error about a profile the user never typed.
PROFILE_CHOICES: list[str] = sorted(set(PROFILES) | set(_PROFILE_ALIASES))
#: Only the real tiers, for a picker a human reads.  An alias is a compatibility
#: shim, not a choice worth showing twice.
PROFILE_TIERS: list[str] = sorted(PROFILES)


# --------------------------------------------------------------------------
# Runtime configuration
# --------------------------------------------------------------------------
@dataclass
class RuntimeConfig:
    """How the harness talks to the model."""

    backend: str = "auto"                 # auto | llamacpp | transformers | openai | mock
    model_dir: str = ""
    model_id: str = "Qwen/Qwen3.5-0.8B"
    gguf_path: str = ""
    mmproj_path: str = ""
    base_url: str = "http://127.0.0.1:8080"
    api_key: str = ""
    # generation
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repeat_penalty: float = 1.05
    max_tokens: int = 768                # per step; small model, small budget
    # Measured on Qwen3.5-0.8B across five representative tasks: thinking ON
    # produced correct tool arguments 5/5 times, thinking OFF only 3/5 (it
    # dropped the arguments from write_file and read_file).  Argument fidelity
    # compounds over an agent loop, so ON is the default; `--thinking off` is
    # roughly twice as fast.  The degeneration detector covers the loop risk
    # Qwen's model card warns about.
    thinking: str = "on"                 # on | off | auto
    seed: int | None = None
    # context
    context_window: int = 32768          # working window (<=262144)
    usable_fraction: float = 0.85
    compact_at: float = 0.55
    summarize_at: float = 0.75
    hard_at: float = 0.92
    keep_recent_turns: int = 6
    # agent loop
    max_steps: int = 24
    # mini-swe-agent ships 3 as its consecutive-format-error limit; ToolScan
    # found weak models do recover, but only from *precise* feedback.
    max_parse_retries: int = 3
    # A repeated *identical* action returns an identical result -- there is never
    # a reason for the second one.  Measured: allowing two identical reads let a
    # "what is in this file?" question run to 16 steps, and since prefill is the
    # dominant cost on CPU (56k input tokens vs 2.4k output), each wasted step is
    # expensive.  Warn on the second, conclude by the fourth.
    loop_warn_at: int = 2
    loop_stop_at: int = 4
    use_grammar: str = "auto"            # auto | on | off
    # How the tool interface reaches the model.  llama-server rejects `tools`
    # together with `grammar`, so "grammar" embeds the model's own tool block in
    # the system prompt and keeps constrained decoding; "tools" does the
    # opposite.  "auto" prefers the grammar (it is the stronger guarantee).
    tool_prompt: str = "auto"            # auto | grammar | tools
    # speculative decoding with the model's own MTP head
    mtp: bool = False
    mtp_draft_max: int = 6
    # images: Qwen-VL warns that grounding needs >= 1024 visual tokens
    image_max_edge: int = 1280
    image_min_tokens: int = 1024
    image_max_tokens: int = 1600
    image_tokens: int = 1024
    # KV cache
    kv_cache_type: str = "q8_0"          # f16 | q8_0 | q4_0
    flash_attention: bool = True
    no_mmproj_offload: bool = True       # CPU-only: never try to offload mmproj
    threads: int = 0                     # 0 = auto (physical cores)
    # Context checkpoints -- measured, and therefore OFF by default.
    #
    # The theory was sound: Qwen3.5 is a *hybrid* model, 18 of its 24 layers are
    # linear attention whose state is a running summary rather than a shiftable
    # KV cache, so llama.cpp cannot rewind into the middle of the context.  When
    # the harness compacts (which rewrites the middle of the prompt) the server
    # should, in principle, restore a saved state instead of re-prefilling.
    #
    # Measured on this machine, three ways, same 2 329-token compacted prompt:
    #
    #     no checkpoints          26.2% cached   17.2 s   ~1196 MB
    #     -ctxcp 16 -cms 512      12.4% cached   19.4 s    1847 MB
    #     -ctxcp  4 -cms 2048      0.0% cached   21.1 s    1577 MB
    #
    # Checkpoints made it *worse* every time and cost 350-650 MB.  Cold prefill,
    # ordinary prefix reuse and generation were identical across all three, so
    # there is no compensating benefit.  llama.cpp's checkpoint feature targets
    # long-context branching across slots, not "the middle of my prompt changed".
    #
    # Kept configurable for future experimentation; the honest default is off.
    ctx_checkpoints: int = 0
    checkpoint_min_step: int = 512

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessConfig:
    workspace: Path
    permissions: Permissions
    runtime: RuntimeConfig
    permission_profile: str = "safe"
    state_dir_name: str = ".qwen35"
    transcript_dir_name: str = "sessions"

    # -- helpers -----------------------------------------------------------
    @property
    def state_dir(self) -> Path:
        return self.workspace / self.state_dir_name

    @property
    def transcript_dir(self) -> Path:
        return self.state_dir / self.transcript_dir_name

    def ensure_dirs(self) -> None:
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace),
            "permission_profile": self.permission_profile,
            "permissions": self.permissions.to_dict(),
            "runtime": self.runtime.to_dict(),
        }


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
CONFIG_FILENAME = "harness.json"


def config_path(workspace: Path) -> Path:
    return workspace / ".qwen35" / CONFIG_FILENAME


def save_config(cfg: HarnessConfig) -> Path:
    p = config_path(cfg.workspace)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    return p


def load_config(workspace: Path) -> dict[str, Any] | None:
    p = config_path(workspace)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def default_model_dir() -> Path:
    """Prefer a roomy non-system drive for multi-GB weights."""
    env = os.environ.get("QWEN35_MODEL_DIR")
    if env:
        return Path(env)
    for drive in ("D:", "E:", "F:"):
        p = Path(drive + os.sep)
        if p.exists():
            return p / "qwen35-models"
    return Path.home() / ".qwen35" / "models"


def discover_gguf(model_dir: Path) -> tuple[Path | None, Path | None]:
    """Find a weights GGUF and an optional vision projector (mmproj) in a dir."""
    if not model_dir.is_dir():
        return None, None
    ggufs = sorted(model_dir.glob("*.gguf"))
    weights: Path | None = None
    mmproj: Path | None = None
    for g in ggufs:
        low = g.name.lower()
        if "mmproj" in low:
            # prefer F16, then anything
            if mmproj is None or "f16" in low:
                mmproj = g
            continue
        # prefer a mid-size quant: best quality/speed tradeoff on CPU
        if weights is None:
            weights = g
        elif "q4_k_m" in low and "q4_k_m" not in weights.name.lower():
            weights = g
    return weights, mmproj
