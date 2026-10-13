# qwen35 — a lightweight harness built specifically for Qwen3.5-0.8B

An agent harness for one model. Not a framework, not a wrapper around a framework:
a purpose-built runtime for a **0.8 B vision-language model** whose weaknesses are
known and engineered around.

The design premise is that a 0.8 B model cannot be made reliable by *prompting*
better. It has to be made reliable by moving the guarantees out of the model and
into the runtime — into the decoder, the tool layer, the context manager and the
permission system.

---

## 中文速览

为 **Qwen3.5-0.8B** 专门构建的轻量级 agent harness。核心思路：0.8B 模型靠提示词
是不可靠的，必须把保证从模型搬到运行时。

* **权限克制**：默认 `safe` 配置——可读、可写（仅限工作区）、可运行白名单命令；
  禁删除、禁网络、禁装包、禁提权。命令以 argv 方式执行，**从不经过 shell**。
* **工作区强制限制**：只能选择一个**真实项目目录**。盘符根目录、家目录本身、
  `C:\Windows`、`Program Files`、`AppData` 一律拒绝；`..`、符号链接逃逸、
  绝对路径外部访问、NTFS 数据流、保留设备名全部拦截。
* **压榨模型**：
  * GBNF 文法约束解码 → 工具名和参数名在解码期就被限制为合法值，格式错误不可能发生；
  * 参数容错解析 → `</parameter>`、`</parameter >`、`</ parameter>` 都能还原；
  * 校验后回滚 → 写入会让文件语法出错时**自动还原**并说明原因；
  * 观察掩码 + 状态台账 → 不靠模型记忆，由 harness 注入"你做过什么"；
  * `</tool_call>` 停止串 → 每步只产生一个动作（实测 1.3 s / 24 token，否则 22 s / 400 token）。
* **长上下文**：模型原生 262 144 token。24 层里只有 6 层是全注意力，其余 18 层是
  线性注意力（Gated DeltaNet），KV cache 只按 6 层增长，约 12 KiB/token，
  262K 全窗口仅需 ~3.0 GiB（q8 量化后 ~1.6 GiB）。
* **多模态**：`view_image` 工具，图片按视觉 token 预算缩放后送入，实测可正常识别。
* 113 个单元测试全部通过（无需模型、无需网络即可验证全部安全性质）。

---

## Verified model facts

Every number below was read from the published model files, not assumed.

| Property | Value | Source |
|---|---|---|
| Model id | `Qwen/Qwen3.5-0.8B` (created 2026-02-28) | HF API |
| Architecture | `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5` | `config.json` |
| Context window | **262 144** tokens | `max_position_embeddings` |
| Layers | 24 = **18 linear-attention (Gated DeltaNet) + 6 full-attention** | `layer_types` |
| Attention | 8 heads, 2 KV heads (GQA), `head_dim` 256, output gate | `text_config` |
| Vision | yes — patch 16, merge 2, depth 12, hidden 768 → 1024 | `vision_config` |
| Position | M-RoPE, interleaved, `mrope_section [11,11,10]`, partial rotary 0.25 | `rope_parameters` |
| MTP head | 1 layer → self-speculative decoding available | `mtp_num_hidden_layers` |
| Tool syntax | **XML**, `<tool_call>` / `</tool_call>` / `<tool_response>` / ` thinking` are **single tokens** | `chat_template.jinja`, `tokenizer_config.json` |
| Thinking | native, via `enable_thinking` | `chat_template.jinja` |

### Why 262 K is affordable here

Only the 6 full-attention layers keep a growing KV cache; the 18 linear layers hold
a **constant-size recurrent state**. So the cache costs

```
6 layers × 2 KV heads × 256 head_dim × 2 (K,V) × 2 bytes = 12 288 B/token
```

| Context | KV f16 | KV q8_0 | If all 24 layers were cached |
|---|---|---|---|
| 32 768 | 0.38 GiB | 0.20 GiB | 1.50 GiB |
| 131 072 | 1.50 GiB | 0.80 GiB | 6.00 GiB |
| 262 144 | **3.00 GiB** | **1.59 GiB** | 12.00 GiB |

`qwen35 doctor` prints this table for the machine it runs on.

---

## Quick start

```powershell
cd qwen35-harness

# 0. the easy path: a real window (Tkinter -- no browser, no extra install)
python -m qwen35 gui --workspace .\myproject

# ...or do it from the terminal:
# 1. what is available here
python -m qwen35 doctor

# 2. fetch the llama.cpp CPU build (~19 MB) and the weights (~700 MB total)
python -m qwen35 install-runtime
python -m qwen35 fetch --model-dir D:\qwen35-models

# 3. start the server (generates an API key; CPU-only flags are probed, not assumed)
python -m qwen35 serve --model-dir D:\qwen35-models --context 32768

# 4. in another shell: run one task, then chat
python -m qwen35 run --workspace .\myproject "add tests for utils.py"
python -m qwen35 chat --workspace .\myproject
```

`--workspace` may be omitted: the CLI then offers a validated picker, and the GUI
opens a folder chooser. Neither will guess.

### The window

`qwen35 gui` opens a standalone application window (Tkinter, so it ships with
CPython and needs no `pip install`):

* **drag files straight onto the window**, or use the file picker — both are
  wired to the ingestion pipeline above, and the attachment list shows exactly
  what was extracted (`✓ report.pdf — pdf, 12 pages, ~9 800 tokens`)
* workspace picker that **enforces the same confinement as the CLI** and explains
  refusals in a dialog rather than silently accepting
* live transcript: every step, tool call, observation, retry and the final answer
* permission profile, context size, step budget and thinking mode, all switchable
  per run
* a preview of **the picture the model is actually looking at** when it calls
  `view_image`
* stop button, server status, and session recording to the same JSONL transcripts

Drag-and-drop uses `tkinterdnd2` when available and degrades to the file picker
when it is not — the API used is the OS drop target, so dropping from Explorer
works exactly as you would expect.

One rule keeps it responsive: **the worker thread never touches a widget or a Tk
variable**; it posts events to a queue that the main thread drains. Reading a
`StringVar` off-thread raises `RuntimeError: main thread is not in main loop` --
which is exactly the bug the end-to-end run in `scripts/gui_smoke.py` caught, and
which `tests/test_gui.py` now guards against.

```powershell
python scripts\gui_smoke.py --workspace .\demo-workspace   # opens the real window and runs a real task
```

### Commands

| Command | Purpose |
|---|---|
| `gui` | standalone window with drag-and-drop attachments |
| `ingest` | convert files and print exactly what the model will receive (`--show`) |
| `run --attach a.pdf --attach b.xlsx` | attach files to a one-shot task |
| `chat` (`/attach`) | attach files to the next message |
| `doctor` | environment report, KV-cost table, workspace-refusal self-test |
| `workspaces` | show candidate workspaces and prove which paths are refused |
| `tools` | list tools and the generated grammar for each thinking mode |
| `grammar` | print the GBNF grammar (`--thinking on|off`) |
| `install-runtime` | download the llama.cpp Windows CPU build |
| `fetch` | download GGUF weights + vision projector (via a HF mirror) |
| `serve` | launch `llama-server` with probed, model-appropriate flags |
| `run` | run one task |
| `chat` | interactive session |

---

## Permissions: basic capability, tightly scoped

The default profile is `safe`:

| | |
|---|---|
| **Allowed** | read files in the workspace; create/overwrite files; surgical edits; run allowlisted programs |
| **Refused** | delete or move anything; touch anything outside the workspace; network; package installation; any privilege change |

| Profile | read | write | run | network | install |
|---|---|---|---|---|---|
| `safe` (default) | ✅ | ✅ | ✅ allowlisted | ❌ | ❌ |
| `readonly` | ✅ | ❌ | ❌ | ❌ | ❌ |
| `standard` | ✅ | ✅ | ✅ | ❌ | ❌ |
| `trusted` | ✅ | ✅ | ✅ | ✅ | ✅ |

Even `trusted` cannot delete files, and `cmd`, `powershell`, `reg`, `schtasks`,
`diskpart`, `format`, `certutil`, `runas`, `ssh`, `robocopy`, `xcopy`, `rm`, `del`
and friends are **permanently blocked in every profile** — those are the programs
that change machine security state, and a 0.8 B model gets no access to them.

### How commands are contained

* **No shell, ever.** The command line is parsed into an argv vector and executed
  with `shell=False`. Pipes, `&&`, `;`, redirection and variable expansion are
  rejected — and would be inert anyway, because no shell is ever spawned.
  Quoted metacharacters are allowed, so `python -c "print(1 > 0)"` works.
* **`cd subdir && <command>` is honoured, safely.** The directory is resolved
  *inside* the workspace and used as the child's working directory; only the real
  command is screened. Escaping with `cd ..` is refused.
* **Credentials never reach the child.** The environment is rebuilt from a small
  allowlist; anything matching `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `AWS_*`,
  `OPENAI_*`, `GITHUB_*`, `DSH_*`, … is dropped.
* **`~` is redirected into the workspace.** `HOME`, `USERPROFILE`, `TEMP`,
  `PIP_CACHE_DIR`, `npm_config_cache` and friends point at
  `<workspace>/.qwen35/sandbox-home`, so accidental writes land in the sandbox.
* **Timeouts and budgets.** Per-command timeout, per-run command count, and output
  caps are enforced.
* **The model server is API-key protected.** `serve` generates a random key and
  stores it in `~/.qwen35/server.json`; without it `/v1/*` returns 401. This
  matters because `llama-server` otherwise accepts requests from any local process
  — and, with permissive CORS, potentially from a web page.

### The honest boundary

The allowlist includes language runtimes (`python`, `node`), because a coding
harness that cannot run code is useless. **`run_command` therefore grants the
model code execution with your user's privileges.** Confinement bounds *where the
harness's own tools* may write; it does not virtualise the filesystem for a child
process. Use the `readonly` profile to remove execution entirely. This is stated
plainly rather than papered over.

### What may never be a workspace

Filesystem roots, the home directory itself, anything *above* the home directory,
`C:\Windows`, `System32`, `Program Files*`, `ProgramData`, `AppData`/`LocalAppData`
(except the OS temp directory, which exists for scratch work), and UNC paths. The
harness also refuses to select *any* ancestor of those. `qwen35 workspaces` proves
this on your machine.

Inside a workspace, these are refused: `..` traversal, absolute paths that resolve
outside, symlink/junction escapes (checked on the **real** path, so a link cannot
become a hole), NTFS alternate data streams, reserved device names (`CON`, `NUL`,
`COM1`…), control characters, and POSIX-absolute arguments like `/etc/passwd`.

---

## How the model is squeezed

Nine mechanisms, in rough order of leverage.

### 1. Grammar-constrained decoding (`grammar.py`)

The tool interface is compiled into GBNF, so the decoder cannot emit a tool name
that does not exist, a parameter that does not belong to that tool, or a
malformed envelope. Failure modes become impossible rather than unlikely.

Two non-obvious problems had to be solved, both found by testing against the real
server rather than by reasoning:

* **llama.cpp releases grammar masking once the grammar is satisfied.** The model
  then free-runs: measured **400 wasted tokens and ~22 s per step**, emitting up
  to 16 tool calls in one message. Enforcing *one action per step* therefore also
  requires a `</tool_call>` stop string. With it: **1.3 s and 24 tokens**.
* **A nullable `root` silently disables the grammar entirely.** If any top-level
  alternative can match the empty string, llama.cpp treats the grammar as already
  satisfied and stops masking — HTTP 200, plausible-looking output, no constraint.
  `validate_grammar()` now performs a fixpoint nullability analysis and
  `build_grammar()` refuses to emit a nullable root. This bug is invisible without
  the check.

`write_file` still writes HTML and C++ (full of `<` and `<<`): the value rule is a
KMP-style DFA that forbids the delimiter *prefix* `\n</parameter` without
forbidding `<`.

### 2. Native format fidelity (`prompt.py`)

The system prompt reproduces the tool block from the model's own
`chat_template.jinja` **verbatim**, rather than paraphrasing tool docs. That is the
distribution the model was trained on.

This also resolves a hard conflict: `llama-server` rejects `tools` together with a
custom `grammar` (`"Cannot use custom grammar constraints with tools."`). Since the
template builds its tool block by string concatenation, the harness emits that text
itself, leaves the `tools` field unset, and keeps constrained decoding. `--tool-prompt`
switches between the two strategies.

### 3. Validate-and-revert (`tools.py`)

After every `write_file`/`edit_file`, the result is syntax-checked (Python via
`ast`, JSON, TOML, YAML, CSV). A write that would break the file is **reverted** and
the model is told what broke. SWE-agent measured edit success at 90.5% on a first
attempt but 57.2% after one prior failure, with cascading failed edits causing
23.4% of unsolved runs: for a weak model, prevention beats recovery.

The check matches what `import` actually accepts — including **UTF-8 BOMs**, which
`ast.parse` rejects but CPython's tokenizer accepts. Rejecting the BOM deadlocked
every edit to a BOM'd file; that was a real bug caught by end-to-end testing.

Overwrites are also backed up to `.qwen35/backups/` and a drastic shrink is
flagged, because this model sometimes emits truncated file content.

### 4. Observation masking, never summarization (`context.py`)

Old tool output is elided to a one-line stub; reasoning is dropped from old turns
(the chat template discards it anyway, so it costs nothing); old images are
released; whole turns are dropped only as a last resort. Escalation stops as soon
as the budget is satisfied.

Summarizing would mean asking a 0.8 B model to compress its own history — slow and
unreliable. Masking is deterministic harness logic. Lindenbauer et al. found
masking halves cost while matching summarization, and SWE-agent found that keeping
only the last 5 observations *beat* full history (18.0% vs 15.0%).

### 5. A harness-maintained ledger (`context.py`)

The model does not have to remember what it did; the harness knows exactly. A
compact state block — workspace root, files read (with line ranges), files changed,
commands that actually ran with exit codes, refusals, recent failures — is re-injected
each step, in the volatile suffix so the cacheable prefix stays byte-identical.

It also provides an anti-hallucination signal. A weak model will happily conclude
"the tests pass" after its commands were *refused* and nothing ran. So refused
commands are recorded separately and the ledger says so:

> `NOTE: no command has succeeded yet, so nothing is verified.`

(Recording refusals as exit code 0 — the original behaviour — actively *lied* to
the model. Fixed, with a regression test.)

### 6. Precise, bounded correction (`parse.py`)

Malformed output is not a shrug. The correction message quotes the exact legal
template, lists the available tools and shows what was produced. Retries are
bounded (3, as in mini-swe-agent). Truncation counts as a format failure too: a
completion that hits the token cap without a tool call is not a considered answer.

The parser is tolerant of every near-miss observed in practice:
`</parameter>`, `</parameter >`, `</parameter/>`, `</parameter />`, `</ parameter>`.

### 7. Loop breaking (`loop.py`)

Actions are hashed by name+arguments and counted **totally**, not just
consecutively — a model alternating between two pointless calls would otherwise
never trip a consecutive-only check. Repeats are refused (not re-executed) with an
escalating message; repeated refusals lead to a forced conclusion rather than
grinding to `max_steps`.

Degeneration *inside a single completion* is detected separately at three
granularities. Qwen's own card warns the 0.8 B is prone to thinking loops; observed
live, it emitted a markdown link seven times until it hit the cap.

### 8. Tuned information flow

Six tools, deliberately few and dissimilar in name. Observations capped at ~100
lines / 6 KB — SWE-agent's ablation found a tuned 100-line window (18.0%) beat both
30 lines (14.3%) and the full file (12.7%). Directory listings cap at 50 entries
with an explicit "narrow your query" hint. Silent success is stated explicitly
("ran successfully and produced no output"), because silence makes a weak model
re-run the command.

### 9. Thinking mode — measured, not assumed

Thinking is **on** by default for work. On five representative tasks, thinking
produced correct tool arguments **5/5**; without it only **3/5** — it dropped the
arguments from `write_file` and `read_file` entirely. Argument fidelity compounds
over a loop, so quality wins. `--thinking off` is roughly twice as fast.

But thinking is actively harmful on small talk, and the harness says so:

* a greeting is answered with **no tools at all** (see below), so there is nothing
  to reason about;
* a retry after reasoning already consumed the budget and produced nothing usable
  runs *without* reasoning, instead of burning the same tokens twice.

### 10. Conversational routing — a hard guarantee, not a hint

Asked **"你好"**, a 0.8 B model treats the greeting as a work order. Measured
before the fix: 3-4 tool calls, 4 steps, **60-200 seconds**, and often an *empty*
answer after the token budget went on reasoning. Prompting it better (a rule plus a
worked example) moved the needle only to 1/4.

So the harness routes it. `looks_conversational()` is deliberately narrow — short
message, no action verb, no path or code token — and when it fires, the run is
started **with an empty tool list**. Nothing in the schema, nothing in the grammar,
therefore a tool call is *impossible* rather than discouraged. It also uses a tiny
system prompt, because the tool block costs ~12 s of prefill on a four-core CPU and
advertising six tools is an invitation to use them.

| | before | after |
|---|---|---|
| greeting / small talk | 0/6 passing, 152 s average | **6/6 passing, 3 s average** |
| output tokens for those 6 runs | 7 200 | **130** |

A wrong guess costs one cheap step (the model may still call a tool; the loop
carries on) and never a wrong answer.

### 11. Reading documents — the harness decomposes, not the model

A 0.8 B model will never parse a PDF. So `qwen35/ingest.py` converts files once,
into page-marked Markdown written **inside the workspace**:

| Input | How | Output |
|---|---|---|
| PDF | PyMuPDF, or a built-in reader | `<!-- page N -->` blocks, plus a **document map** for long files |
| Word (`.docx`) | python-docx, or a built-in OOXML reader | `#` headings, pipe tables, plus an **outline** when there are 4+ headings |
| **Word 97-2003 (`.doc`)** | a built-in **OLE2 + piece-table reader** (`legacy_word.py`) | the body text, read-only. No Word, no LibreOffice, no network |
| RTF (`.rtf`, or mislabelled `.doc`) | a built-in reader | prose, with `\uNNNN` and `\'hh` escapes decoded (UTF-8/GBK runs kept together) |
| Excel (`.xlsx`) | openpyxl, or a built-in OOXML reader | one pipe table per sheet |
| PowerPoint (`.pptx`) | python-pptx, or a built-in OOXML reader | `<!-- slide N -->` + speaker notes, in `sldIdLst` order |
| CSV / TSV | built-in | a pipe table |
| JSON / JSON Lines | built-in | a shape summary + pretty JSON, or a records table |
| HTML | built-in parser | prose, list markers and table cells; script/style dropped |
| Text / code | built-in | verbatim, with BOM/UTF-16/GBK detection |
| Images | copied into the workspace | handed to `view_image` |
| Scanned PDF | pages rendered at 150 DPI | handed to `view_image`, page by page |

Three decisions make it work rather than just exist:

* **The converted text goes into the workspace**, under `.qwen35/attachments/`.
  Not tidiness: the workspace guard refuses reads outside the workspace, so a file
  dropped from the Desktop would otherwise be unreachable to the very tools meant
  to read it — and `.qwen35` is skipped by `list_dir`, so the user's project stays
  clean.
* **The decision to inline is made in tokens, not characters**, and the budget is
  a third of the context window (2 500–12 000 tokens). Inlining costs *exactly
  the same prefill* as deferring — the text enters the context either way — so
  deferring only adds a model step and a chance to fail. Two real failures came
  from getting this wrong: a 2 909-byte Chinese spreadsheet (973 tokens, 3% of
  the window) was left on disk for no reason, and a 7 839-token `.doc` was too.
* **The extracted text goes LAST**, after the request. With the content between
  the card and the request — the middle of the prompt, where "Lost in the Middle"
  says attention degrades — the model ignored it and spent eight steps trying to
  open the original binary.
* **A card never names a tool it does not need.** The card used to say "the text
  is at the end of this message" and then "use `read_file(...)`" one line later.
  The model obeyed the tool suggestion and ignored the inlined text, burning a
  whole run. Naming a tool is an instruction to call it.

### 12. Licensing, stated plainly

`PyMuPDF` is **dual-licensed AGPL-3.0 / commercial**. It is an *optional*
accelerator here: `ingest.py` uses it when present and otherwise falls back to a
standard-library PDF reader that handles `(...)` literals, `<hex>` strings and the
`Tj`/`TJ`/`'`/`"` operators. The fallback is honest about being best-effort — it
cannot resolve `/Identity-H` CID fonts without a ToUnicode CMap.

Anyone *conveying* this harness together with PyMuPDF must satisfy the AGPL.
python-docx, openpyxl and python-pptx are MIT and carry no such caveat.

---

## What the small-model literature says, and what was taken from it

Researched before writing the ingestion layer, because the failure modes of
sub-1B agents are well documented and mostly not obvious.

| Source | Finding | Taken? |
|---|---|---|
| [TinyAgent 2409.00608](https://arxiv.org/abs/2409.00608) | Plan-first (LLMCompiler) lifted 1.1B plan success **12.71% → 78.89%** — but only *after* LoRA fine-tuning. Off-the-shelf 1.1B "produced wrong plans when prompted like a large model". | **No**, deliberately. The 12.7% untuned figure is the argument *against* plan-first here: we cannot fine-tune, so interleaved ReAct with grammar-constrained actions stays. |
| [ToolChoiceConfusion 2606.06284](https://arxiv.org/abs/2606.06284) | Causal minimal tool filtering: **100 → 1 visible tool per step**, ~90% fewer tokens, no accuracy loss. | **Partly.** The harness offers six tools and withholds unusable ones per profile; per-step filtering is the obvious next step and is noted as such. |
| [How Many Tools 2605.24660](https://arxiv.org/abs/2605.24660) | **~7 tools** shown per step matches fixed-50 coverage; 100 tools measurably hurts. | **Yes** — six tools, deliberately few and dissimilar. |
| [Hammer 2410.04587](https://arxiv.org/abs/2410.04587) | Small models fail at **irrelevance detection** — knowing to call *nothing*. 1.5B reaches 73 BFCL but only 72.2 on irrelevance. | **Yes** — conversational routing (§10) makes "no tool" a legal, sometimes *forced*, path. |
| [Lost in the Middle 2307.03172](https://arxiv.org/abs/2307.03172) | Retrieval is best at the **beginning or end**; the middle degrades. | **Yes** — attachment content moved to the end; the ledger rides last; `read_file` windows land as the newest observation. |
| [System One Harness](https://harnessrouter.ai/system-one-harness) | Finite action space, typed questions, **confidence gates**, refuse rather than guess. | **Partly.** Budgets, duplicate breakers and forced conclusion already refuse rather than guess; a per-class confidence gate is not implemented. |
| [Tokasaurus/…, Octopus v2 2404.01744](https://arxiv.org/abs/2404.01744) | One token per API collapses the output space. | **Equivalent** — the GBNF grammar enumerates legal tool names, so the model *selects* rather than writes. |
| [Apple AFM 2025](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates) | The framework owns schemas, constrained decoding and call sequencing; the model never orchestrates. | **Yes** — that is this harness's whole architecture. |
| [SEGMENT+ / RAPTOR](https://arxiv.org/abs/2401.18059) | Segment → per-segment notes → reason over notes; summarisation ladders. | **Partly.** The document map and DOCX outline are the harness-built index; no model-generated summaries (a 0.8 B summariser is not trustworthy). |

The consistent theme across every source: **the harness does the work**. Asking a
0.8 B model to plan, to self-critique, or to parse a binary format is where these
systems fail.

---

## Architecture

```
qwen35/
  cli.py             CLI: gui / ingest / doctor / serve / fetch / run / chat / ...
  gui.py             standalone Tkinter window: drag-and-drop, attachments, transcript
  ingest.py          PDF / Word / Excel / PPT / CSV / JSON / HTML / image -> Markdown
  config.py          verified model profile, permission profiles, runtime config
  workspace.py       workspace validation + path confinement (the security core)
  permissions.py     argv-based allowlisted command screening, env scrubbing
  grammar.py         GBNF synthesis + nullable-root and termination validators
  parse.py           tolerant XML tool-call parser, correction builder, loop detector
  prompt.py          verbatim chat-template tool block, operating rules, few-shots
  tools.py           6 tools: validate-and-revert, backups, argv execution
  context.py         token estimation, observation masking, harness ledger
  loop.py            interleaved ReAct loop with budgets and breakers
  session.py         JSONL transcripts for replay and audit
  llama_runtime.py   llama.cpp install, flag-probing launcher, ensure_server, KV economics
  backends/          mock (tests) | openai-compatible (llama-server) | transformers
```

Backends are pluggable. `mock` is deterministic and lets the entire harness —
loop, confinement, screening, grammar, parser, compactor — be tested with no
weights, no GPU and no network. The GUI is a thin shell over the same `Agent`, so
it inherits every one of those guarantees rather than reimplementing them.

---

## Testing

Two layers, because they catch different things.

**Unit tests** prove the internals are correct and need no model, no GPU and no
network:

```powershell
python -m unittest discover -s tests -v     # 165 tests
python -m qwen35 doctor                     # environment + confinement self-test
python -m qwen35 workspaces                 # show which paths are refused, and why
```

They cover workspace refusal (roots, home, system dirs, UNC, ancestors,
operator-protected paths), path confinement (`..`, symlinks, ADS, device names,
control characters), command screening (metacharacters, banned programs,
network/install gating, budgets, the `cd` affordance), environment scrubbing,
grammar validity, nullability *and termination*, the parser against every observed
delimiter variant, degeneration detection, validate-and-revert, backups, context
masking and the ledger, conversational routing, the thinking policy, the full agent
loop against the mock backend, and the GUI.

**The evaluation suite** proves the whole thing is *usable*. It runs representative
prompts through the real agent loop against the real model, in a fresh workspace
per task, and asserts on the outcome:

```powershell
python scripts\eval_suite.py                 # 13 tasks, ~20 minutes
python scripts\eval_suite.py --only greeting # one task
python scripts\eval_suite.py --list          # what is covered
```

The distinction is not academic. Every bug below was found by this layer and is
invisible to unit tests:

* **a greeting triggered a 13-step repository tour** (the motivating example);
* **`build_backend` returned HTTP 401** for any caller that built a config by hand,
  because only the CLI and GUI loaded the stored API key;
* **the grammar's plain-answer branch could never terminate** — every state ended
  in a bare recursive reference, so end-of-input was unreachable and a run with no
  tools generated to the token cap and returned *nothing*, with HTTP 200 and no
  error anywhere;
* **a task whose acceptance criterion was unachievable** — "make all tests pass" on
  a fixture with a deliberate bug. A broken contract, not a broken model;
* **an attached spreadsheet was ignored** while the model spent eight steps trying
  to open the original binary, because the extracted text sat in the middle of the
  prompt.

That last one is worth stating plainly: an evaluation suite tests the *tests* too.

### Measured results

| What | Result |
|---|---|
| Unit tests | **284 passing** (no model, no network) |
| End-to-end verification | **17/17** — 读文件 6/6, 调用工具 5/5, 回答 6/6 (`scripts/e2e_verify.py`) |
| Evaluation suite | **13 tasks**, all passing across the runs above |
| Greeting | answered in **~1.3 s**, zero tool calls (was 60-200 s and 3-4 calls) |
| Attached spreadsheet | **0 steps, 40 s** (was 6 steps, 210 s, sometimes no answer) |
| Attached `.doc` (7 839 tokens) | **0 steps, 42 s** (was 5 steps, 337 s, failing) |
| Attached PDF | **0 steps, 40 s** (was 10 steps, 206 s) |
| Attached picture | **0 steps, 42 s**, described correctly (was 6 steps, failing) |
| Grammar | 2 907 B / 51 rules (non-thinking), 3 562 B / 61 rules (thinking) |
| One action per step | 1.3 s / 24 tokens with the stop string vs 22 s / 400 tokens without |
| Server hardening | `/v1/*` returns 401 without the generated key |

**The attachment fast path is the biggest single win.** When the full text of an
attached file is already in the prompt and the user is not asking for the
workspace to change, the first step offers **no tools at all** — the grammar has
no call branch, so the model cannot call one. Measured over one real spreadsheet:
6 steps and 210 s became 0 steps and 40 s, and answers that used to end in "no
conclusion" now land. It is the same mechanism that fixed greetings, and for the
same reason: for a 0.8 B model, a named tool is an instruction to call it.

The routing reads **only the user's own words** (the `# Request` section), never
the harness's scaffolding. That distinction was itself a bug worth recording: the
attachment card says "already been **convert**ed to text", which matched the
"convert" change-marker, so every attachment question looked like a request to
modify the workspace and the fast path never fired.

One earlier full run scored 12/13: `read-pdf` returned an **empty answer**. That was
not model variance — the forced-conclusion path published a `_summarise()` that
returned `""` when a run had only read files. Silence is never an acceptable
result, so the summary can no longer be empty and `tests/test_harness.py` now
asserts it.

### Honest limitations

* **Task success is variable.** The evaluation suite passes 10/10, but this is a
  0.8 B model: repeat runs sometimes derail (inventing a subdirectory, re-reading
  the same file, truncating content, misunderstanding a describe-only request as a
  request to edit). The harness bounds the damage and reports it honestly rather
  than pretending otherwise.
* **Speed is dominated by prefill, not reasoning.** On this four-core CPU, the
  first prefill of the ~1 500-token system prompt costs ~12 s and generation runs
  at ~18 tokens/s, so a real task takes 1-3 minutes. Input tokens dwarf output
  tokens in every trace. A GPU, more cores, or a smaller `--context` all help more
  than prompt tuning does.
* **`run_command` is capability-equivalent to your shell** for allowlisted
  programs.
* **The `transformers` backend has no grammar support**, so it falls back to
  parse-and-repair.
* **Tolerant parsing accepts sloppy delimiters by design.**
* MTP self-speculative decoding is wired up (`--mtp`) but not benchmarked here.

---

## Licence

MIT. GGUF weights and llama.cpp binaries are fetched from their own upstream
releases under their own licences.
