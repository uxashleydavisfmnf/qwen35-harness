"""
A standalone-window GUI for the qwen35 harness.

Built on Tkinter so the deployment stays self-contained: it ships with CPython,
needs no ``pip install``, and works with no network.  A browser-based UI would
have been easier to write but would not be "its own window", and pulling in a web
view or Electron runtime for a 0.8 B model's harness would be absurd.

Threading model
---------------
Tkinter is not thread-safe, so exactly one rule is enforced here: **the worker
thread never touches a widget**.  The agent runs in a background thread and pushes
events onto a ``queue.Queue``; the main thread drains that queue on a timer and
does all the drawing.  That keeps the window responsive during a 60-second
generation, which matters a lot when the model is running on four CPU cores.

The GUI is a thin shell over the same :class:`~qwen35.loop.Agent` the CLI uses --
same tools, same permissions, same workspace confinement, same grammar.  Anything
the GUI can do, ``qwen35 run`` can do, and vice versa.
"""

from __future__ import annotations

import base64
import io
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter import font as tkfont
from tkinter import scrolledtext

from .config import (DEFAULT_PROFILE, PROFILE_CHOICES, PROFILE_TIERS, PROFILES,
                     QWEN35_08B, HarnessConfig, RuntimeConfig,
                     default_model_dir, discover_gguf, get_permissions)
from .ingest import (SUPPORTED_EXTENSIONS, Attachment, IngestError,
                     compose_task, ingest_many, inline_budget_for)
from .llama_runtime import (ServerStatus, ensure_server, load_server_key,
                            probe_server)
from .session import SessionRecorder
from .workspace import WorkspaceError, validate_workspace

__all__ = ["HarnessGUI", "main"]

# --------------------------------------------------------------------------
# Palette -- one place, so the window looks deliberate
# --------------------------------------------------------------------------
BG = "#1e1f26"
PANEL = "#262832"
FIELD = "#2f3240"
FG = "#e6e6ea"
MUTED = "#9aa0b4"
ACCENT = "#7aa2f7"
OK = "#9ece6a"
WARN = "#e0af68"
ERR = "#f7768e"
ANSWER = "#c0caf5"
TASK = "#bb9af7"

CONTEXT_CHOICES = ("32768", "65536", "131072", "262144")
THINKING_CHOICES = ("on", "off", "auto")
#: Human labels for the picker.  Only the three real tiers appear; "safe" is a
#: compatibility alias for standard, and offering it as a separate choice would
#: suggest a difference that no longer exists.
PROFILE_LABELS = {
    "readonly": "readonly - 只读：不写文件、不跑命令、不联网",
    "standard": "standard - 默认：读写工作区、跑命令、联网、装包（仅工作区）",
    "trusted": "trusted - 额外允许 shell 通道（管道/&&）；仍出不了工作区",
}


class HarnessGUI:
    """One window, one workspace, one task at a time."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        workspace: str | Path,
        model_dir: str | Path | None = None,
        context: int = 32768,
        profile: str = DEFAULT_PROFILE,
        port: int = 8080,
        auto_start_server: bool = True,
        manage_server: bool = True,
    ) -> None:
        self.root = root
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.port = int(port)
        self.auto_start_server = auto_start_server
        self.manage_server = manage_server

        self.events: queue.Queue[tuple[str, Any, Any]] = queue.Queue()
        self.agent: Any = None
        self.worker: threading.Thread | None = None
        self.server_status: ServerStatus | None = None
        self._spinner = 0
        self._busy = False
        self._image_ref: Any = None          # keep PhotoImage alive
        self._last_image: Any = None
        self._last_elapsed = 0.0
        #: files the user dropped or picked, already converted for the model
        self.attachments: list[Attachment] = []
        #: every wrapping row, so the initial layout can be triggered once the
        #: window has a real width (a FlowFrame cannot lay out before then)
        self._flows: list[FlowFrame] = []
        #: the last run's token accounting, for the 上下文用量 button
        self._last_stats: dict[str, Any] | None = None
        self._bench_running = False
        #: set by _on_close so the 120 ms pump stops re-arming itself
        self._closing = False
        self._pump_id: str | None = None
        self._reflow_id: str | None = None
        #: true while a server start is in flight, so two clicks cannot race
        self._starting_server = False
        self._pending_paths: list[str] = []
        self._dnd_ready = False

        self.workspace = self._validate_initial_workspace(workspace)

        root.title("qwen35 - Qwen3.5-0.8B harness")
        root.configure(bg=BG)
        root.geometry("1180x780")
        # The old floor was 880x560, chosen when every row was laid out once and
        # clipped below it -- which also meant the wrapping rows never had a
        # reason to wrap, because the window could not get narrow.  With every
        # row now adaptive the floor can drop to a genuinely small window.
        root.minsize(520, 420)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        # Also catch a direct destroy (tests, or the window being torn down some
        # other way): without this the 120 ms pump re-arms itself one last time
        # and fires against a destroyed Tcl command.
        root.bind("<Destroy>", self._on_root_destroy, add="+")

        self._init_fonts()
        self._build_styles()
        self._build_ui(
            context=context, profile=profile if profile in PROFILES else "safe"
        )
        self._pump()
        if self.manage_server and self.auto_start_server:
            self._start_server_async()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _validate_initial_workspace(self, workspace: str | Path) -> Path:
        try:
            return validate_workspace(str(workspace), create=True)
        except WorkspaceError as exc:
            fallback = Path.cwd()
            try:
                ws = validate_workspace(str(fallback))
            except WorkspaceError:
                ws = Path.home() / "qwen35-workspace"
                ws.mkdir(parents=True, exist_ok=True)
            self._pending_notice = f"工作区被拒绝（{exc}），已改用 {ws}"
            return ws

    def _init_fonts(self) -> None:
        family = "Consolas"
        available = set(tkfont.families())
        for candidate in ("Cascadia Mono", "Consolas", "Courier New"):
            if candidate in available:
                family = candidate
                break
        self.mono = tkfont.Font(family=family, size=10)
        self.ui = tkfont.Font(family="Microsoft YaHei UI" if
                              "Microsoft YaHei UI" in available else "Segoe UI",
                              size=10)
        self.ui_bold = tkfont.Font(font=self.ui, weight="bold")

    def _build_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=self.ui)
        style.configure("Panel.TLabel", background=PANEL, foreground=FG, font=self.ui)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=self.ui)
        style.configure("Head.TLabel", background=BG, foreground=ACCENT,
                        font=self.ui_bold)
        style.configure("TButton", font=self.ui, padding=(10, 5))
        style.configure("Go.TButton", font=self.ui_bold, padding=(14, 6))
        style.configure("TCombobox", font=self.ui)
        style.configure("TSpinbox", font=self.ui)
        style.configure("TCheckbutton", background=BG, foreground=FG, font=self.ui)

    def _build_ui(self, *, context: int, profile: str) -> None:
        # ---- header ---------------------------------------------------
        # A wrapping row: the title, the model blurb, the server state and the
        # restart button move to a second line rather than being clipped.
        header = self._flow(self.root, gap=8, row_gap=4)
        header.pack(fill="x", padx=14, pady=(12, 6))

        header.add(ttk.Label(header, text="qwen35", style="Head.TLabel"))
        subtitle = ttk.Label(header,
                             text="Qwen3.5-0.8B  ·  多模态  ·  262144 上下文",
                             style="Muted.TLabel")
        # wraplength so the text folds instead of forcing the row wider than the
        # window; without it this label alone set a floor on the whole layout.
        subtitle.configure(wraplength=260)
        header.add(subtitle, stretch=True)
        self.server_label = ttk.Label(header, text="服务：未启动",
                                      style="Muted.TLabel")
        header.add(self.server_label)
        header.add(ttk.Button(header, text="重启服务",
                              command=self._start_server_async))

        # ---- workspace row --------------------------------------------
        ws_row = self._flow(self.root, gap=8, row_gap=4)
        ws_row.pack(fill="x", padx=14, pady=(0, 8))

        ws_row.add(ttk.Label(ws_row, text="工作区", style="Muted.TLabel"))
        self.ws_var = tk.StringVar(value=str(self.workspace))
        ws_entry = tk.Entry(ws_row, textvariable=self.ws_var, font=self.mono,
                            bg=FIELD, fg=FG, insertbackground=FG, width=20,
                            relief="flat", highlightthickness=1,
                            highlightbackground=PANEL, highlightcolor=ACCENT)
        ws_row.add(ws_entry, stretch=True)
        ws_row.add(ttk.Button(ws_row, text="浏览…", command=self._choose_workspace))
        ws_row.add(ttk.Button(ws_row, text="校验", command=self._apply_workspace))

        # ---- options row ----------------------------------------------
        # A FlowFrame, not a packed row: five label+control pairs do not fit a
        # narrow window, and a fixed row simply clips them off the right edge.
        # Each pair is one unit so a wrapped row never splits a label from its
        # control.
        opts = self._flow(self.root, gap=14, row_gap=6)
        opts.pack(fill="x", padx=14, pady=(0, 8))

        def _pair(label: str, widget_factory) -> None:
            group = ttk.Frame(opts, style="TFrame")
            ttk.Label(group, text=label, style="Muted.TLabel").pack(side="left")
            widget_factory(group).pack(side="left", padx=(6, 0))
            opts.add(group)

        self.profile_var = tk.StringVar(value=profile)
        _pair("权限", lambda parent: ttk.Combobox(
            parent, textvariable=self.profile_var, width=34, state="readonly",
            values=[PROFILE_LABELS.get(p, p) for p in PROFILE_TIERS]))

        self.context_var = tk.StringVar(value=str(context))
        _pair("上下文", lambda parent: ttk.Combobox(
            parent, textvariable=self.context_var, width=9, state="readonly",
            values=list(CONTEXT_CHOICES)))

        self.steps_var = tk.StringVar(value="20")
        _pair("最大步数", lambda parent: ttk.Spinbox(
            parent, from_=1, to=200, width=5, textvariable=self.steps_var))

        self.thinking_var = tk.StringVar(value="on")
        _pair("思考", lambda parent: ttk.Combobox(
            parent, textvariable=self.thinking_var, width=6, state="readonly",
            values=list(THINKING_CHOICES)))

        self.verbose_var = tk.BooleanVar(value=False)
        verbose_group = ttk.Frame(opts, style="TFrame")
        ttk.Checkbutton(verbose_group, text="显示模型原文",
                        variable=self.verbose_var).pack(side="left")
        opts.add(verbose_group)

        # ---- one button per capability --------------------------------
        # The window is meant to be the whole interface: anything the CLI can do
        # that a person might want should be reachable without opening a
        # terminal, and its output should land in the transcript rather than in a
        # console that a GUI user never sees.  Each button below maps to exactly
        # one command -- and the row wraps, so none of them disappears when the
        # window is narrow.
        actions = self._flow(self.root, gap=6, row_gap=6)
        actions.pack(fill="x", padx=14, pady=(0, 8))
        for label, command, tip in (
            ("环境自检", self._act_doctor, "Python / 模型 / KV 开销 / 权限报告"),
            ("工具与文法", self._act_tools, "模型能用哪些工具，约束文法长什么样"),
            ("会话记录", self._act_sessions, "列出已保存的会话，可打开目录"),
            ("上下文用量", self._act_context, "上一次运行占用了多少 token"),
            ("定时任务", self._act_schedule, "按分钟 / 小时周期自动工作"),
            ("性能实测", self._act_bench, "预填充 / 前缀缓存 / 生成吞吐（需要服务）"),
        ):
            button = ttk.Button(actions, text=label, command=command)
            actions.add(button)
            _tip(button, tip)

        # ---- transcript -----------------------------------------------
        body = ttk.Frame(self.root, style="TFrame")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 6))

        self.transcript = scrolledtext.ScrolledText(
            body, wrap="word", font=self.mono, bg=PANEL, fg=FG,
            insertbackground=FG, relief="flat", padx=12, pady=10,
            state="disabled", highlightthickness=0,
            # width=30, not Tk's default of 80: a ScrolledText asks for
            # 80 characters (~700 px) by default, and that one request set the
            # whole window's minimum width to ~880 px.  The window could not be
            # made narrow, so none of the wrapping rows ever had a reason to
            # wrap.  It is packed with expand=True, so it still grows.
            width=30, height=12,
        )
        self.transcript.pack(side="left", fill="both", expand=True)

        # image the model is currently looking at
        side = ttk.Frame(body, style="Panel.TFrame", width=260)
        side.pack(side="right", fill="y", padx=(10, 0))
        side.pack_propagate(False)
        ttk.Label(side, text="模型看到的图片", style="Panel.TLabel").pack(pady=(8, 4))
        self.image_label = tk.Label(side, bg=PANEL, fg=MUTED, font=self.ui,
                                    text="（尚未查看图片）", wraplength=240)
        self.image_label.pack(fill="both", expand=True, padx=8, pady=8)

        for tag, colour in (
            ("task", TASK), ("step", ACCENT), ("ok", OK), ("warn", WARN),
            ("err", ERR), ("answer", ANSWER), ("meta", MUTED), ("head", FG),
        ):
            self.transcript.tag_configure(tag, foreground=colour)
        self.transcript.tag_configure("task", font=self.ui_bold)
        self.transcript.tag_configure("answer", font=self.ui_bold)
        self.transcript.tag_configure("meta", font=self.ui)

        # ---- attachments ----------------------------------------------
        self._build_attachments()

        # ---- input ----------------------------------------------------
        # The row the user flagged: this used to be a fixed side column, so a
        # narrow window squeezed the text box to nothing while the four buttons
        # kept their width.  As a wrapping row the text box takes the slack, and
        # the buttons move underneath when there is no longer room beside it.
        bottom = self._flow(self.root, gap=8, row_gap=6)
        bottom.pack(fill="x", padx=14, pady=(0, 4))

        self.input = tk.Text(bottom, height=4, width=24, font=self.ui, bg=FIELD,
                             fg=FG, insertbackground=FG, relief="flat",
                             wrap="word", padx=10, pady=8, highlightthickness=1,
                             highlightbackground=PANEL, highlightcolor=ACCENT)
        bottom.add(self.input, stretch=True)
        self.input.bind("<Return>", self._on_return)
        self.input.bind("<Shift-Return>", lambda _e: None)

        self.send_button = ttk.Button(bottom, text="发送  (Enter)", width=13,
                                      style="Go.TButton", command=self._on_send)
        bottom.add(self.send_button)
        self.stop_button = ttk.Button(bottom, text="停止", width=13,
                                      command=self._on_stop, state="disabled")
        bottom.add(self.stop_button)
        clear_button = ttk.Button(bottom, text="清空", width=13,
                                  command=self._clear)
        bottom.add(clear_button)
        _tip(clear_button, "开新对话：清空窗口内容、附件和上一次的运行统计。"
                           "工作区里的文件不会动。")
        bottom.add(ttk.Button(bottom, text="打开工作区", width=13,
                              command=self._open_workspace))

        # ---- status bar -----------------------------------------------
        self.status_var = tk.StringVar(value="就绪")
        status = ttk.Label(self.root, textvariable=self.status_var,
                           style="Muted.TLabel")
        status.pack(fill="x", padx=14, pady=(0, 10))

        self._banner()
        self.input.focus_set()

        # Lay the wrapping rows out once the window has a real width.  A
        # FlowFrame cannot compute rows before that (winfo_width() is 1), and
        # without this first pass a row would sit empty until the user happened
        # to resize the window.  The id is kept so a window destroyed before the
        # idle callback runs does not leave a dangling Tcl command.
        self._reflow_id = self.root.after_idle(self._reflow_all)

    # ------------------------------------------------------------------
    # Attachments: drag-and-drop, file picker, conversion
    # ------------------------------------------------------------------
    def _flow(self, parent: tk.Misc, **kwargs: Any) -> FlowFrame:
        """Create a wrapping row and remember it for the initial layout."""
        frame = FlowFrame(parent, **kwargs)
        self._flows.append(frame)
        return frame

    def _reflow_all(self) -> None:
        for frame in self._flows:
            try:
                frame.reflow_now()
            except tk.TclError:
                pass

    def _build_attachments(self) -> None:
        frame = ttk.Frame(self.root, style="TFrame")
        frame.pack(fill="x", padx=14, pady=(0, 6))

        # The hint text is long, so this row wraps: on a narrow window the three
        # buttons move below the label instead of being pushed off the edge.
        head = self._flow(frame, gap=8, row_gap=4)
        head.pack(fill="x")
        head.add(ttk.Label(head, text="附件", style="Muted.TLabel"))
        self.attach_hint = ttk.Label(
            head,
            text="把文件拖进这个窗口，或点「选择文件…」"
                 "（PDF · Word · Excel · PPT · 图片 · 文本 · CSV · JSON）",
            style="Muted.TLabel", wraplength=420)
        head.add(self.attach_hint, stretch=True)
        head.add(ttk.Button(head, text="选择文件…", command=self._choose_files))
        head.add(ttk.Button(head, text="移除选中",
                            command=self._remove_selected_attachments))
        head.add(ttk.Button(head, text="清空", command=self._clear_attachments))

        self.attach_list = tk.Listbox(
            frame, height=3, width=30, font=self.ui, bg=PANEL, fg=FG,
            selectbackground=ACCENT, selectforeground=BG, relief="flat",
            highlightthickness=1, highlightbackground=PANEL,
            highlightcolor=ACCENT, activestyle="none", exportselection=False,
        )
        self.attach_list.pack(fill="x", pady=(4, 0))
        self._pending_paths: list[str] = []
        self._refresh_attachments()

    def register_dnd(self) -> bool:
        """Enable OS drag-and-drop, if tkinterdnd2 is available.

        Called after construction rather than inside ``__init__`` so tests can
        build a plain ``tk.Tk()`` window without the tkdnd extension.
        """
        drop = getattr(self.root, "drop_target_register", None)
        bind = getattr(self.root, "dnd_bind", None)
        if drop is None or bind is None:
            self.attach_hint.configure(
                text="点「选择文件…」添加附件（本机未安装拖放扩展 tkinterdnd2）")
            return False
        try:
            from tkinterdnd2 import DND_FILES

            for widget in (self.root, self.transcript, self.attach_list, self.input):
                try:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind("<<Drop>>", self._on_drop)
                except Exception:  # noqa: BLE001 - per-widget, keep going
                    continue
            self._dnd_ready = True
            return True
        except Exception as exc:  # noqa: BLE001
            self.attach_hint.configure(text=f"拖放不可用：{exc}")
            return False

    def _on_drop(self, event: Any) -> str:
        """Handle a file drop.  The path list arrives brace-quoted."""
        raw = getattr(event, "data", "") or ""
        try:
            paths = list(self.root.tk.splitlist(raw))
        except Exception:  # noqa: BLE001
            paths = [raw]
        cleaned: list[str] = []
        for item in paths:
            text = str(item).strip().strip("{}").strip()
            if text:
                cleaned.append(text)
        if cleaned:
            self._add_files(cleaned)
        return "break"

    def _choose_files(self) -> None:
        patterns = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        chosen = filedialog.askopenfilenames(
            title="选择要交给模型的文件",
            filetypes=[
                ("支持的文件", patterns),
                ("PDF", "*.pdf"),
                ("Word", "*.docx"),
                ("Excel", "*.xlsx *.xlsm"),
                ("PowerPoint", "*.pptx"),
                ("图片", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                ("文本 / 数据", "*.txt *.md *.csv *.json *.log *.yaml"),
                ("全部文件", "*.*"),
            ],
        )
        if chosen:
            self._add_files(list(chosen))

    def _add_files(self, paths: list[str]) -> None:
        files = [p for p in paths if Path(p).is_file()]
        skipped = [p for p in paths if not Path(p).is_file()]
        if skipped:
            self._append("\n跳过（不是文件）：" + ", ".join(Path(p).name for p in skipped)
                         + "\n", "warn")
        if not files:
            return
        added = 0
        for p in files:
            key = str(Path(p).resolve())
            if key in self._pending_paths:
                continue
            self._pending_paths.append(key)
            added += 1
        self._refresh_attachments()
        if not added:
            return
        self.status_var.set(f"正在转换 {added} 个文件…")
        # Read the context size HERE, on the main thread: a worker must never
        # touch a Tk variable (that was a real crash: "main thread is not in
        # main loop").
        try:
            context = int(self.context_var.get())
        except (tk.TclError, ValueError):
            context = 32768
        threading.Thread(target=self._ingest_worker, args=(files, context),
                         daemon=True, name="qwen35-ingest").start()

    def _ingest_worker(self, files: list[str], context: int) -> None:
        """Background thread.  Conversion can be slow for a big PDF."""
        try:
            from .ingest import inline_budget_for

            results = ingest_many(
                files, self.workspace,
                inline_tokens=inline_budget_for(context))
            self.events.put(("attachments", results, None))
        except Exception as exc:  # noqa: BLE001
            self.events.put(("attachment_error", exc, files))

    def _on_attachments(self, results: list[Attachment]) -> None:
        existing = {str(a.source.resolve()) for a in self.attachments}
        for att in results:
            key = str(att.source.resolve())
            if key in existing:
                self.attachments = [a for a in self.attachments
                                    if str(a.source.resolve()) != key]
            self.attachments.append(att)
        self._refresh_attachments()
        good = [a for a in results if a.kind != "unsupported"]
        bad = [a for a in results if a.kind == "unsupported"]
        if good:
            self._append("\n已转换附件：\n", "meta")
            for a in good:
                self._append(f"  ✓ {a.summary_line()}\n", "ok")
                if a.out_rel:
                    self._append(f"      -> {a.out_rel}\n", "meta")
                for w in a.warnings:
                    self._append(f"      注意：{w}\n", "warn")
        for a in bad:
            self._append(f"  ✗ {a.source.name}：{'; '.join(a.warnings)}\n", "err")
        self.status_var.set(f"附件就绪：{len(self.attachments)} 个")

    def _refresh_attachments(self) -> None:
        self.attach_list.delete(0, "end")
        by_path = {str(a.source.resolve()): a for a in self.attachments}
        if not self._pending_paths:
            self.attach_list.insert("end", "  （还没有附件；可直接把文件拖进来）")
            self.attach_list.itemconfig("end", foreground=MUTED)
            return
        for key in self._pending_paths:
            att = by_path.get(key)
            name = Path(key).name
            if att is None:
                self.attach_list.insert("end", f"  转换中…  {name}")
                self.attach_list.itemconfig("end", foreground=MUTED)
            elif att.kind == "unsupported":
                reason = att.warnings[0] if att.warnings else "不支持"
                self.attach_list.insert("end", f"  ✗ {name} — {reason[:64]}")
                self.attach_list.itemconfig("end", foreground=ERR)
            else:
                self.attach_list.insert("end", f"  ✓ {att.summary_line()}")
                self.attach_list.itemconfig("end", foreground=OK)

    def _remove_selected_attachments(self) -> None:
        selection = list(self.attach_list.curselection())
        if not selection or not self._pending_paths:
            return
        remove = {self._pending_paths[i] for i in selection
                  if 0 <= i < len(self._pending_paths)}
        self._pending_paths = [p for p in self._pending_paths if p not in remove]
        self.attachments = [a for a in self.attachments
                            if str(a.source.resolve()) not in remove]
        self._refresh_attachments()

    def _clear_attachments(self) -> None:
        self._pending_paths = []
        self.attachments = []
        self._refresh_attachments()

    def _usable_attachments(self) -> list[Attachment]:
        return [a for a in self.attachments if a.kind != "unsupported"]

    # ------------------------------------------------------------------
    # Transcript helpers (main thread only)
    # ------------------------------------------------------------------
    def _append(self, text: str, tag: str = "meta") -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", text, tag)
        self.transcript.see("end")
        self.transcript.configure(state="disabled")

    def _write(self, text: str, tag: str = "meta") -> None:
        self._append(text, tag)

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    def _start_server_async(self, force: bool = False) -> None:
        if self._starting_server:
            # guard the *start*, not just a running task: two clicks used to
            # launch two starts at once, and the loser saved a key the winner's
            # server had never seen
            self.status_var.set("模型服务正在启动中…")
            return
        if self._busy:
            return
        self._starting_server = True
        # Read the Tk variable HERE, on the main thread.  Tk variables are not
        # thread-safe: touching one from the worker raises
        # "RuntimeError: main thread is not in main loop".
        context = int(self.context_var.get())
        self.server_label.configure(text="服务：启动中…", foreground=WARN)
        self.status_var.set("正在启动模型服务…" + ("（会先停掉占用端口的服务）"
                                                 if force else ""))

        def work() -> None:
            try:
                status = ensure_server(
                    model_dir=self.model_dir,
                    context=context,
                    port=self.port,
                    state_dir=None,
                    force_restart=force,
                    progress=lambda m: self.events.put(("server_progress", m, None)),
                )
            except Exception as exc:  # noqa: BLE001
                # Keep the frames: "main thread is not in main loop" and friends
                # say nothing about *where* the offending call was.
                frames = traceback.format_exc().strip().splitlines()[-5:]
                status = ServerStatus(
                    False, "failed",
                    f"{type(exc).__name__}: {exc}\n" + "\n".join(frames),
                )
            self.events.put(("server", status, None))

        threading.Thread(target=work, daemon=True, name="qwen35-server").start()

    def _on_server_status(self, status: ServerStatus) -> None:
        self._starting_server = False
        self.server_status = status
        if status.ok:
            verb = "复用" if status.state == "reused" else "就绪"
            self.server_label.configure(
                text=f"服务：{verb}  {status.base_url}", foreground=OK)
            self.status_var.set(f"模型服务{verb}")
            if status.state == "started" and status.notes:
                self._append("模型服务已启动：" +
                             "; ".join(status.notes) + "\n", "meta")
        else:
            self.server_label.configure(text="服务：启动失败", foreground=ERR)
            self.status_var.set("模型服务启动失败")
            self._append(f"\n模型服务启动失败：{status.message}\n", "err")
            self._append("可以在终端运行 -Action serve 查看详细日志，"
                         "或检查 D:\\qwen\\state\\logs。\n", "meta")

    def _stop_owned_server(self) -> None:
        if self.server_status and self.server_status.we_own_it:
            try:
                self.server_status.server.stop()
            except Exception:  # noqa: BLE001
                pass
            self.server_status = None

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------
    def _choose_workspace(self) -> None:
        chosen = filedialog.askdirectory(
            title="选择工作区（盘符根目录、系统目录会被拒绝）",
            initialdir=str(self.workspace) if self.workspace.is_dir() else None,
            mustexist=False,
        )
        if chosen:
            self.ws_var.set(chosen)
            self._apply_workspace()

    def _apply_workspace(self) -> None:
        raw = self.ws_var.get().strip().strip('"')
        try:
            ws = validate_workspace(raw, create=True)
        except WorkspaceError as exc:
            self.ws_var.set(str(self.workspace))
            messagebox.showerror(
                "工作区被拒绝",
                f"{raw}\n\n{exc}\n\n"
                "工作区必须是真实项目目录：不能是盘符根目录、家目录本身，"
                "也不能是系统目录或其内部。",
            )
            return
        self.workspace = ws
        self.ws_var.set(str(ws))
        self._append(f"\n工作区已切换：{ws}\n\n", "meta")

    def _open_workspace(self) -> None:
        try:
            import os

            os.startfile(str(self.workspace))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            messagebox.showinfo("工作区", str(self.workspace))

    # ------------------------------------------------------------------
    # Running a task
    # ------------------------------------------------------------------
    def _on_return(self, _event: tk.Event) -> str:
        self._on_send()
        return "break"        # stop Tk inserting a newline

    # ------------------------------------------------------------------
    # One button per capability.  Each writes into the transcript so the window
    # stays self-contained -- a GUI user has no console to read.
    # ------------------------------------------------------------------
    def _act_head(self, title: str) -> None:
        self._append(f"\n{'─' * 72}\n", "meta")
        self._append(f"{title}\n", "head")

    def _act_doctor(self) -> None:
        """Python / model / KV / permissions, without leaving the window."""
        self._act_head("环境自检")
        import platform
        import shutil as _shutil

        profile = self._selected_profile()
        perms = get_permissions(profile)
        lines = [
            f"Python          {platform.python_version()}  ({sys.executable})",
            f"平台            {platform.platform()}",
            f"工作区          {self.workspace}",
            f"权限档位        {profile}",
            f"  读上限        {perms.max_read_bytes:,} 字节",
            f"  写上限        {perms.max_write_bytes:,} 字节",
            f"  命令超时      {perms.max_command_seconds} 秒",
            f"  每轮命令数    {perms.max_commands_per_run}",
            f"  联网          {'允许' if perms.allow_network_commands else '禁止'}",
            f"  装包          {'允许（仅工作区）' if perms.allow_package_install else '禁止'}",
            f"  提权          {'禁止（永久）' if True else ''}",
        ]
        profile_obj = QWEN35_08B
        layers = profile_obj.num_hidden_layers
        full = profile_obj.full_attention_layers
        lines += [
            f"模型            {profile_obj.name}",
            f"  层数          {layers}（{layers - full} 线性注意力 + {full} 全注意力）",
            f"  上下文        {profile_obj.context_window:,} token",
            f"  每 token KV   {full * 2 * 256 * 2 * 2:,} 字节（只有全注意力层要缓存）",
        ]
        binary = self._server_binary()
        lines.append(f"llama-server    {binary or '（未安装）'}")
        model_dir = Path(self.model_dir) if self.model_dir else default_model_dir()
        weights, mmproj = discover_gguf(Path(model_dir))
        lines += [
            f"模型目录        {model_dir}",
            f"  权重          {weights.name if weights else '（缺失）'}",
            f"  视觉投影      {mmproj.name if mmproj else '（缺失）'}",
            f"服务状态        {'就绪' if self.server_status and self.server_status.ok else '未就绪'}",
        ]
        for line in lines:
            self._append(f"  {line}\n", "meta")

    def _server_binary(self) -> str:
        try:
            from .llama_runtime import installed_server

            found = installed_server()
            return str(found) if found else ""
        except Exception:  # noqa: BLE001
            return ""

    def _act_tools(self) -> None:
        """Which tools the model has, and the grammar that constrains it."""
        self._act_head("工具与约束文法")
        from .grammar import build_grammar
        from .grammar import ToolSignature
        from .prompt import build_system  # noqa: F401  (kept for parity checks)
        from .tools import build_specs

        perms = get_permissions(self._selected_profile())
        specs = build_specs(None, perms)
        self._append(f"  共 {len(specs)} 个工具：\n", "meta")
        for spec in specs:
            params = ", ".join(spec.params) or "（无参数）"
            self._append(f"    {spec.name:<12} {params}\n", "meta")
        for thinking in (False, True):
            try:
                sigs = [ToolSignature(name=s.name, params=s.params) for s in specs]
                grammar = build_grammar(sigs, thinking=thinking,
                                        allow_plain_answer=True)
                rules = grammar.count("\n") + 1
                label = "思考开启" if thinking else "思考关闭"
                self._append(
                    f"  文法（{label}）：{len(grammar):,} 字节 / {rules} 条规则\n",
                    "meta")
            except Exception as exc:  # noqa: BLE001
                self._append(f"  文法生成失败：{exc}\n", "err")

    def _act_sessions(self) -> None:
        """List the recorded sessions, and offer to open the folder."""
        self._act_head("会话记录")
        folder = Path(self.workspace) / ".qwen35" / "sessions"
        files = sorted(folder.glob("*.jsonl")) if folder.is_dir() else []
        if not files:
            self._append(f"  还没有会话记录。\n  位置：{folder}\n", "meta")
            return
        self._append(f"  {len(files)} 个会话（最新在后）：\n", "meta")
        for path in files[-12:]:
            size = path.stat().st_size
            self._append(f"    {path.name[:58]:<58} {size / 1024:6.1f} KB\n", "meta")
        self._append(f"  位置：{folder}\n", "meta")
        if messagebox.askyesno("打开会话目录", f"要在资源管理器里打开吗？\n{folder}"):
            self._open_path(folder)

    def _act_context(self) -> None:
        """How much of the window the last run actually used."""
        self._act_head("上下文用量")
        stats = getattr(self, "_last_stats", None)
        if not stats:
            self._append("  还没跑过任务，所以没有数据。\n", "meta")
            return
        budget = stats.get("budget") or 0
        used = stats.get("tokens") or 0
        share = (used / budget * 100) if budget else 0.0
        for key, value in (
            ("窗口预算", f"{budget:,} token"),
            ("上次占用", f"{used:,} token（{share:.1f}%）"),
            ("压缩阈值", "55% 开始遮蔽旧观察"),
            ("输入合计", f"{stats.get('tokens_in', 0):,} token"),
            ("输出合计", f"{stats.get('tokens_out', 0):,} token"),
            ("缓存命中", f"{stats.get('cached_in', 0):,} token"),
            ("步数", str(stats.get("steps", 0))),
            ("耗时", f"{stats.get('elapsed', 0):.1f} 秒"),
        ):
            self._append(f"  {key:<10} {value}\n", "meta")

    def _act_schedule(self) -> None:
        """List recurring tasks, and add one if asked."""
        self._act_head("定时任务")
        try:
            from .schedule import ScheduleStore, format_interval, parse_interval
        except ImportError as exc:
            self._append(f"  调度模块不可用：{exc}\n", "err")
            return

        store = ScheduleStore()
        if store.schedules:
            for schedule in store.schedules:
                state = "开" if schedule.enabled else "关"
                self._append(
                    f"  [{state}] {schedule.id}  每 "
                    f"{format_interval(schedule.every_seconds):<4} "
                    f"运行 {schedule.runs} 次 / 失败 {schedule.failures} 次\n"
                    f"        {schedule.task[:56]}\n", "meta")
        else:
            self._append("  还没有定时任务。\n", "meta")

        if not messagebox.askyesno("定时任务", "要新增一个定时任务吗？"):
            return
        task = simpledialog.askstring(
            "定时任务", "每次要做什么？", parent=self.root,
            initialvalue="检查工作区里的 TODO，把结果整理成 todo.md")
        if not task:
            return
        every = simpledialog.askstring(
            "定时任务", "多久一次？（分钟 m / 小时 h，例如 30m、2h）",
            parent=self.root, initialvalue="30m")
        if not every:
            return
        try:
            seconds = parse_interval(every)
        except ValueError as exc:
            messagebox.showerror("间隔无效", str(exc))
            return
        schedule = store.add(task, seconds, workspace=str(self.workspace),
                             profile=self._selected_profile())
        self._append(
            f"  已添加 {schedule.id}：每 {format_interval(schedule.every_seconds)}"
            f"运行一次。\n  在终端执行 `start.bat -Action watch` 开始定时工作。\n",
            "ok")

    def _probe_server(self) -> tuple[bool, str]:
        """Ask the server whether it is actually there, right now.

        The cached ``server_status`` is not enough.  It records what was true when
        the server was started, and a llama-server can be stopped from outside the
        window -- by ``stop.bat``, by a reboot, by anything.  Trusting the cache
        produced exactly the failure this method exists to prevent:

            性能实测失败：the model server is not answering at
            http://127.0.0.1:8080 (<urlopen error [WinError 10061] ...>)

        which reported a *connection* error for what was really a stale label.
        A short timeout keeps this cheap on the UI thread.
        """
        from .llama_runtime import probe_server

        try:
            return probe_server(self._base_url(), load_server_key(), timeout=3.0)
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            return "down", f"{type(exc).__name__}: {exc}"

    def _mark_server_down(self) -> None:
        self.server_status = None
        try:
            self.server_label.configure(text="服务：未运行")
        except tk.TclError:
            pass

    def _offer_restart(self, what: str, message: str) -> None:
        """Explain that the server is gone, and offer to bring it back."""
        self._mark_server_down()
        self._append(f"  模型服务没有在运行：{message}\n", "warn")
        self._append("  点右上角「重启服务」可以拉起来（首次约 20 秒）。\n", "meta")
        if messagebox.askyesno(
                "模型服务未运行",
                f"{what}需要模型服务，但现在连不上 {self._base_url()}。\n\n"
                f"{message}\n\n现在启动吗？（首次约 20 秒）"):
            self._start_server_async()

    def _act_bench(self) -> None:
        """Measure prefill / cache / generation on the running server."""
        self._act_head("性能实测")
        if getattr(self, "_bench_running", False):
            self._append("  上一次实测还在进行中。\n", "warn")
            return
        self._bench_running = True
        self._append("  检查模型服务…\n", "meta")

        def worker() -> None:
            try:
                state, message = self._probe_server()
                # NOTE: compare against "up".  probe_server returns a (state,
                # message) tuple, and an earlier version of this fix wrote
                # `if not up:` -- but `up` is the *string* "down", which is
                # truthy, so every dead server was treated as healthy.
                if state == "unauthorized":
                    self.events.put(("server_conflict", message, None))
                    return
                if state != "up":
                    self.events.put(("server_down", ("性能实测", message), None))
                    return
                self.events.put(("bench_note",
                                 "  正在测量（冷预填充 + 前缀缓存 + 生成，"
                                 "约 1 分钟）…\n", None))
                from .bench import run_bench

                report = run_bench(base_url=self._base_url(), grow_steps=4,
                                   cold_tokens=1200, gen_tokens=48)
                self.events.put(("bench", report, None))
            except Exception as exc:  # noqa: BLE001
                self.events.put(("bench_error", exc, None))

        threading.Thread(target=worker, daemon=True, name="qwen35-bench").start()

    def _on_bench(self, report: dict[str, Any]) -> None:
        self._bench_running = False
        cold = report.get("cold") or {}
        cached = report.get("cached") or {}
        grow = report.get("grow") or {}
        gen = report.get("generation") or {}
        if cold:
            self._append(
                f"  冷预填充   {cold.get('prompt_tokens', 0):,} token / "
                f"{cold.get('seconds')} 秒 = "
                f"{cold.get('prefill_tokens_per_second')} token/秒\n", "meta")
        if cached:
            self._append(
                f"  前缀缓存   {cached.get('cached_tokens', 0):,}/"
                f"{cached.get('prompt_tokens', 0):,} 命中"
                f"（{cached.get('cache_hit_ratio', 0) * 100:.1f}%），"
                f"比冷启动快 {cached.get('speedup_vs_cold')} 倍\n", "meta")
        if grow:
            self._append(
                f"  增长对话   平均命中 "
                f"{grow.get('mean_cache_hit_ratio', 0) * 100:.1f}%，"
                f"平均每步 {grow.get('mean_step_seconds')} 秒\n", "meta")
        if gen:
            self._append(
                f"  生成       {gen.get('tokens_per_second')} token/秒\n", "meta")

    def _open_path(self, path: Path) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(str(path))          # noqa: S606 - fixed local path
            else:
                import subprocess as _sp

                _sp.Popen(["xdg-open", str(path)])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("打开失败", str(exc))

    def _base_url(self) -> str:
        if self.server_status and getattr(self.server_status, "base_url", ""):
            return self.server_status.base_url
        return f"http://127.0.0.1:{self.port}"

    def _on_send(self) -> None:
        if self._busy:
            return
        task = self.input.get("1.0", "end").strip()
        attachments = self._usable_attachments()
        if not task and not attachments:
            return
        if not self.server_status or not self.server_status.ok:
            messagebox.showwarning(
                "模型服务未就绪",
                "模型服务还没起来。点右上角「重启服务」等待就绪，"
                "或先在终端运行 -Action serve。",
            )
            return

        self.input.delete("1.0", "end")
        self._append(f"\n{'─' * 72}\n", "meta")
        if attachments:
            self._append("附件  " + "、".join(a.title for a in attachments) + "\n",
                         "meta")
        self._append(f"任务  {task or '（仅附件）'}\n", "task")

        full_task = self._compose_task(task)
        options = self._snapshot_options()   # main thread: reads Tk variables
        self._set_busy(True)
        self.worker = threading.Thread(target=self._worker,
                                       args=(full_task, options),
                                       daemon=True, name="qwen35-agent")
        self.worker.start()

    def _compose_task(self, task: str) -> str:
        """Attach the converted files around the user's request.

        The conversion happens in the harness, not in the model: a 0.8 B model
        cannot parse a PDF.  By the time it sees this the document is page-marked
        Markdown -- and the extracted text is placed *after* the request, because
        that is the position a small model actually attends to.
        """
        files = self._usable_attachments()
        if not files:
            return task
        return compose_task(task, files)

    # -- thread-safe option handling -----------------------------------
    @staticmethod
    def _profile_from_label(raw: str) -> str:
        raw = (raw or "").strip()
        return raw.split(" - ", 1)[0].strip() if " - " in raw else raw

    def _selected_profile(self) -> str:
        """Main thread only -- reads a Tk variable."""
        return self._profile_from_label(self.profile_var.get())

    def _snapshot_options(self) -> dict[str, Any]:
        """Read every Tk variable ONCE, on the main thread.

        Tk variables are not thread-safe.  Reading one from the agent thread
        raises ``RuntimeError: main thread is not in main loop``, which is
        exactly what an earlier version did -- the window appeared to start the
        model server and then failed for no visible reason.  The worker therefore
        only ever receives plain Python values.
        """
        return {
            "profile": self._selected_profile(),
            "context": int(self.context_var.get()),
            "steps": int(self.steps_var.get()),
            "thinking": self.thinking_var.get(),
            # snapshot the attachment list too, so the worker never reads shared
            # mutable state that the main thread could change under it
            "attachments": list(self._usable_attachments()),
        }

    def _make_config(self, options: dict[str, Any]) -> HarnessConfig:
        perms = get_permissions(options["profile"])
        weights, projector = discover_gguf(self.model_dir)
        rt = RuntimeConfig(
            backend="llamacpp",
            model_dir=str(self.model_dir),
            gguf_path=str(weights) if weights else "",
            mmproj_path=str(projector) if projector else "",
            base_url=f"http://127.0.0.1:{self.port}",
            api_key=(self.server_status.api_key if self.server_status
                     else load_server_key()),
            context_window=int(options["context"]),
            max_steps=int(options["steps"]),
            thinking=str(options["thinking"]),
        )
        cfg = HarnessConfig(workspace=self.workspace, permissions=perms,
                            runtime=rt, permission_profile=options["profile"])
        cfg.ensure_dirs()
        return cfg

    def _worker(self, task: str, options: dict[str, Any]) -> None:
        """Background thread.  Never touches a widget or a Tk variable."""
        recorder = None
        try:
            from .backends import build_backend
            from .loop import Agent

            # Probe first, for the same reason the bench does: a cached "ready"
            # goes stale the moment the server is stopped from outside, and the
            # run would then die with an opaque connection error instead of
            # "the server is not running, shall I start it?".
            state, message = self._probe_server()
            if state == "unauthorized":
                self.events.put(("server_conflict", message, None))
                return
            if state != "up":
                self.events.put(("server_down", ("运行任务", message), None))
                return

            cfg = self._make_config(options)
            backend = build_backend(cfg.runtime)
            recorder = SessionRecorder.open(cfg.workspace, task)
            agent = Agent(
                cfg, backend,
                tool_names=None,
                on_event=lambda e, d: self.events.put(("event", e, d)),
                recorder=recorder,
                attachments=options.get("attachments") or [],
            )
            self.agent = agent
            result = agent.run(task)
            self.events.put(("result", result, str(recorder.path) if recorder else ""))
        except Exception as exc:  # noqa: BLE001
            self.events.put(("fatal", exc, traceback.format_exc()))
        finally:
            if recorder is not None:
                recorder.close()

    def _on_stop(self) -> None:
        if self.agent is not None:
            self.agent.stop()
            self.status_var.set("已请求停止；当前这一步结束后停止")

    def _clear(self) -> None:
        """Start a new conversation.

        This used to wipe the transcript text and nothing else, which was
        misleading: the attachments stayed attached, so the *next* message still
        carried the files from the conversation the user thought they had just
        ended.  "New conversation" now means all of it -- transcript, banner,
        attachments, image preview and the previous run's accounting.
        """
        if self._busy:
            messagebox.showwarning(
                "任务正在运行",
                "先点「停止」，等这一步结束再开新对话。")
            return
        has_content = bool(self.transcript.get("1.0", "end").strip())
        has_files = bool(self.attachments)
        if (has_content or has_files) and not messagebox.askyesno(
                "开始新对话",
                "会清空窗口内容、附件列表和上一次的运行统计。\n"
                "工作区里的文件不会被动。\n\n确定吗？"):
            return

        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")

        # attachments: the list, and the converted copies this conversation
        # produced.  They are derived data -- re-dropping the file regenerates
        # them -- so leaving them behind would just accumulate in .qwen35/.
        removed = self._remove_ingested_files()
        self._pending_paths = []
        self.attachments = []
        self._refresh_attachments()
        self._set_image(None)
        self._last_stats = None
        self.status_var.set("新对话" + (f"（清理了 {removed} 个转换文件）"
                                       if removed else ""))
        self._banner()

    def _remove_ingested_files(self) -> int:
        """Delete the converted copies for the current attachments.

        Only what *this* conversation ingested, and only inside the workspace's
        own ``.qwen35/attachments`` directory: the originals the user dragged in
        are never touched.  Session transcripts are kept -- they are the record.
        """
        folder = Path(self.workspace) / ".qwen35" / "attachments"
        if not folder.is_dir():
            return 0
        removed = 0
        for attachment in self.attachments:
            targets: list[Path] = []
            if attachment.out_rel:
                targets.append(Path(self.workspace) / attachment.out_rel)
            for image in attachment.images:
                targets.append(Path(self.workspace) / image)
            for target in targets:
                try:
                    resolved = target.resolve()
                    resolved.relative_to(folder.resolve())
                except (OSError, ValueError):
                    continue          # never delete outside the attachment store
                if resolved.is_file():
                    try:
                        resolved.unlink()
                        removed += 1
                    except OSError:
                        pass
        # the per-attachment page folders (scanned PDFs) go too
        for child in folder.iterdir():
            if child.is_dir() and not any(child.iterdir()):
                try:
                    child.rmdir()
                except OSError:
                    pass
        return removed

    def _banner(self) -> None:
        """The greeting at the top of a fresh conversation."""
        self._append("qwen35 harness — Qwen3.5-0.8B\n", "head")
        self._append(f"工作区  {self.workspace}\n", "meta")
        self._append(f"模型    {self.model_dir}\n", "meta")
        self._append("提示：Enter 发送，Shift+Enter 换行。"
                     "模型只能在工作区内读写，盘符根目录会被拒绝。\n"
                     "「清空」= 开新对话：窗口、附件、运行统计一起重置。\n\n",
                     "meta")
        notice = getattr(self, "_pending_notice", "")
        if notice:
            self._append(notice + "\n\n", "warn")
        self.input.focus_set()

    def _clear_attachments(self) -> None:
        self._pending_paths = []
        self.attachments = []
        self._refresh_attachments()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.send_button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")

    # ------------------------------------------------------------------
    # Event pump -- the only place widgets are touched
    # ------------------------------------------------------------------
    def _drain(self) -> int:
        """Process every queued message once.  Returns how many were handled.

        Split out from :meth:`_pump` so tests can drive rendering
        deterministically without racing the ``after`` timer.
        """
        handled = 0
        while True:
            try:
                kind, a, b = self.events.get_nowait()
            except queue.Empty:
                return handled
            handled += 1
            try:
                if kind == "event":
                    self._on_event(a, b or {})
                elif kind == "server":
                    self._on_server_status(a)
                elif kind == "server_progress":
                    self.status_var.set(str(a))
                elif kind == "result":
                    self._on_result(a, b)
                elif kind == "fatal":
                    self._on_fatal(a, b)
                elif kind == "attachments":
                    self._on_attachments(a)
                elif kind == "bench":
                    self._on_bench(a)
                elif kind == "bench_note":
                    self._append(a, "meta")
                elif kind == "server_down":
                    self._bench_running = False
                    self._set_busy(False)
                    self.agent = None
                    self._offer_restart(*a)
                elif kind == "server_conflict":
                    self._bench_running = False
                    self._set_busy(False)
                    self.agent = None
                    self._mark_server_down()
                    self._append(f"  端口上有一个服务，但不接受已保存的密钥：{a}\n",
                                 "warn")
                    # Offer to do the cleanup instead of telling the user to go
                    # and run taskkill: the harness owns this lifecycle, and
                    # making them leave the window to fix it is a cop-out.
                    if messagebox.askyesno(
                            "端口被占用",
                            f"{self._base_url()} 上有一个 llama-server，但它不接受"
                            "已保存的密钥。\n\n"
                            "这通常是上一次运行留下的进程。\n"
                            "要停掉它并重新启动模型服务吗？"):
                        self._start_server_async(force=True)
                elif kind == "bench_error":
                    self._bench_running = False
                    self._append(f"\n性能实测失败：{a}\n", "err")
                elif kind == "attachment_error":
                    self._append(f"\n附件转换失败：{a}\n", "err")
                    self.status_var.set("附件转换失败")
            except Exception:  # a rendering bug must not kill the pump
                self._append(f"\n[界面错误] {traceback.format_exc()}\n", "err")

    def _pump(self) -> None:
        self._drain()

        if self._busy:
            self._spinner = (self._spinner + 1) % 4
            dots = "." * (self._spinner + 1)
            self.status_var.set(f"运行中{dots}  {self._last_elapsed:.0f}s")

        # Remember the id so it can be cancelled on close: an after() that
        # outlives its widget fires against a destroyed Tcl command and prints
        # `invalid command name "..._pump"` on the way out.
        if not self._closing:
            self._pump_id = self.root.after(120, self._pump)

    def _on_event(self, event: str, data: dict[str, Any]) -> None:
        if event == "start":
            tools = ", ".join(data.get("tools", []))
            self._append(f"backend  {data.get('backend', '')}\n", "meta")
            self._append(f"工具     {tools}\n", "meta")
            self._append(
                f"上下文   {data.get('context_window', 0):,} tokens"
                f"   权限 {self._selected_profile()}\n", "meta")
        elif event == "thinking_start":
            flags = []
            if data.get("grammar"):
                flags.append("文法")
            if data.get("thinking"):
                flags.append("思考")
            self._append(f"\n步骤 {data['step']}"
                         f"{' [' + '·'.join(flags) + ']' if flags else ''}\n",
                         "step")
        elif event == "raw":
            if self.verbose_var.get():
                text = str(data.get("text", "")).strip()
                self._append(f"  模型原文: {text[:600]}\n", "meta")
        elif event == "tool_call":
            self._append(f"  → {data.get('call', '')}\n", "step")
        elif event == "tool_ok":
            self._append(f"    ok ({data.get('chars', 0)} 字符)\n", "ok")
        elif event == "tool_error":
            first = self._first_line(data.get("text", ""))
            exit_code = data.get("exit")
            if exit_code is not None and exit_code != 0 and not data.get("refused"):
                self._append(f"    exit {exit_code}: {first}\n", "warn")
            else:
                self._append(f"    {first}\n", "err")
        elif event == "format_error":
            self._append(f"    ? {self._first_line(data.get('problem', ''))}"
                         f" (重试 {data.get('retry')})\n", "warn")
        elif event == "degenerate":
            self._append("    ~ 模型开始重复，已要求重说\n", "warn")
        elif event == "forced_finish":
            self._append("    ~ 模型不肯结束，harness 代为收尾\n", "warn")
        elif event == "loop_stop":
            self._append(f"    ! 同一动作重复 {data.get('repeats')} 次，停止\n", "err")
        elif event == "context_full":
            self._append(f"    ! 上下文已满 "
                         f"({data.get('tokens')}/{data.get('budget')})\n", "warn")
        elif event == "error":
            self._append(f"    !! {data.get('message')}\n", "err")
        elif event == "answer":
            pass  # rendered with the result, so it is not shown twice
        self._maybe_show_image(event, data)

    def _maybe_show_image(self, event: str, data: dict[str, Any]) -> None:
        """Preview the picture when the agent calls view_image."""
        if event != "tool_call":
            return
        call = str(data.get("call", ""))
        if not call.startswith("view_image"):
            return
        start = call.find("path=")
        if start < 0:
            return
        raw = call[start + 5:].split(",")[0].strip()
        path = self.workspace / raw
        if path.is_file():
            self._set_image(path)

    def _set_image(self, path: Path | None) -> None:
        if path is None:
            self._image_ref = None
            self.image_label.configure(image="", text="（尚未查看图片）")
            return
        try:
            from PIL import Image, ImageTk

            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((240, 240))
                photo = ImageTk.PhotoImage(im)
            self._image_ref = photo          # Tk needs the reference kept
            self.image_label.configure(image=photo, text="")
        except Exception as exc:  # noqa: BLE001
            self.image_label.configure(image="", text=f"{path.name}\n({exc})")

    def _on_result(self, result: Any, session_path: str) -> None:
        self._set_busy(False)
        self.agent = None
        if result.answer:
            self._append("\n" + "─" * 72 + "\n", "meta")
            self._append(f"{result.answer}\n", "answer")
        self._append(
            f"\n[{result.stop_reason} | {len(result.steps)} 步 | "
            f"入 {result.tokens_in:,} / 出 {result.tokens_out:,} tokens | "
            f"{result.elapsed:.1f}s]\n", "meta")
        if session_path:
            self._append(f"会话记录  {session_path}\n", "meta")
        self._append("\n", "meta")
        self.status_var.set(f"完成：{result.stop_reason}  ({result.elapsed:.1f}s)")
        self._last_elapsed = 0.0
        # Feeds the 上下文用量 button: the window budget plus what the run that
        # just finished actually used.  Average prompt size per step is the
        # honest "how full is my window" number, since every step re-sends the
        # whole conversation.
        steps = list(result.steps)
        # getattr everywhere: the event-rendering tests drive this with a stub
        # result, and a UI callback must never be the thing that crashes a run
        # that already finished successfully.
        per_step = [getattr(s, "tokens_in", 0) for s in steps]
        self._last_stats = {
            "budget": int(self.context_var.get()),
            "tokens": (sum(per_step) // len(per_step)) if per_step
                      else getattr(result, "tokens_in", 0),
            "tokens_in": getattr(result, "tokens_in", 0),
            "tokens_out": getattr(result, "tokens_out", 0),
            "cached_in": getattr(result, "cached_in", 0),
            "steps": len(steps),
            "elapsed": getattr(result, "elapsed", 0.0),
        }

    def _on_fatal(self, exc: BaseException, tb: str) -> None:
        self._set_busy(False)
        self.agent = None
        self._append(f"\n运行失败：{type(exc).__name__}: {exc}\n", "err")
        first_frames = "\n".join(tb.strip().splitlines()[-6:])
        self._append(first_frames + "\n", "meta")
        self.status_var.set("运行失败")

    @staticmethod
    def _first_line(text: Any, limit: int = 160) -> str:
        s = str(text or "").strip()
        line = s.splitlines()[0] if s else ""
        return line[:limit]

    # ------------------------------------------------------------------
    def _on_root_destroy(self, event: tk.Event) -> None:
        if event.widget is not self.root:
            return
        self._closing = True
        for attr in ("_pump_id", "_reflow_id"):
            pending = getattr(self, attr, None)
            if pending is not None:
                try:
                    self.root.after_cancel(pending)
                except tk.TclError:
                    pass
                setattr(self, attr, None)

    def _on_close(self) -> None:
        if self._busy and not messagebox.askokcancel(
                "仍有任务在跑", "任务还在运行。确定要退出吗？"):
            return
        self._closing = True          # stops _pump from re-arming itself
        if self._pump_id is not None:
            try:
                self.root.after_cancel(self._pump_id)
            except tk.TclError:
                pass
            self._pump_id = None
        if self.agent is not None:
            try:
                self.agent.stop()
            except Exception:  # noqa: BLE001
                pass
        self._stop_owned_server()
        self.root.destroy()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
class FlowFrame(ttk.Frame):
    """A container that lays its children out in rows and **wraps**.

    Neither ``pack`` nor ``grid`` reflows: a row of six buttons is laid out once
    and then either clips off the right edge when the window is narrow or leaves
    a ragged gap when it is wide.  Tkinter has no flexbox, so this recomputes the
    layout itself on ``<Configure>``: children keep their natural size and move
    onto the next line when there is no room left.

    Two details that matter in practice:

    * the reflow is **debounced** (~60 ms) because ``<Configure>`` fires
      continuously while a window edge is dragged, and re-placing six widgets on
      every event makes the drag visibly stutter;
    * the frame's height is set explicitly, since ``place``-managed children do
      not contribute to a parent's requested size -- without it the row would be
      allocated zero height and nothing would appear.
    """

    def __init__(self, master: tk.Misc, *, gap: int = 6, row_gap: int = 6,
                 **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self._items: list[tk.Widget] = []
        self._gap = gap
        self._row_gap = row_gap
        self._width = 0
        self._pending: str | None = None
        self._last_height = 0
        self.bind("<Configure>", self._on_configure)
        # A pending after() callback outlives the widget and then fires against a
        # destroyed Tcl command, which surfaces as
        # `invalid command name "..._reflow"` on stderr.  Cancel it on destroy.
        self.bind("<Destroy>", self._on_destroy)

    # -- api -----------------------------------------------------------
    def add(self, widget: tk.Widget, *, stretch: bool = False) -> tk.Widget:
        """Add a child.  ``stretch`` lets it share leftover space on its row."""
        self._items.append(widget)
        setattr(widget, "_flow_stretch", stretch)
        return widget

    def reflow_now(self) -> None:
        """Lay out immediately (used once at startup, and by tests)."""
        if self._pending is not None:
            try:
                self.after_cancel(self._pending)
            except tk.TclError:
                pass
            self._pending = None
        self._reflow()

    # -- internals -----------------------------------------------------
    def _on_configure(self, event: tk.Event) -> None:
        if abs(event.width - self._width) < 4:
            return                      # ignore height-only churn
        self._width = event.width
        if self._pending is not None:
            try:
                self.after_cancel(self._pending)
            except tk.TclError:
                pass
        self._pending = self.after(60, self._reflow)

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is not self:
            return                      # a child was destroyed, not the row
        if self._pending is not None:
            try:
                self.after_cancel(self._pending)
            except tk.TclError:
                pass
            self._pending = None

    def _rows(self, available: int) -> list[list[tk.Widget]]:
        """Group the children into rows that fit ``available`` pixels."""
        rows: list[list[tk.Widget]] = []
        row: list[tk.Widget] = []
        used = 0
        for child in self._items:
            width = child.winfo_reqwidth()
            needed = width if not row else used + self._gap + width
            if row and needed > available:
                rows.append(row)
                row, used = [child], width
            else:
                row.append(child)
                used = needed
        if row:
            rows.append(row)
        return rows

    def _reflow(self) -> None:
        self._pending = None
        available = self.winfo_width()
        if available <= 1:
            return
        y = 0
        for row in self._rows(available):
            heights = [c.winfo_reqheight() for c in row]
            row_height = max(heights) if heights else 0
            # leftover space is shared by any stretchy children, so a wrapped row
            # still looks deliberate rather than leaving a hole on the right
            spent = sum(c.winfo_reqwidth() for c in row) + self._gap * (len(row) - 1)
            stretchy = [c for c in row if getattr(c, "_flow_stretch", False)]
            extra = max(0, available - spent) // len(stretchy) if stretchy else 0
            x = 0
            for child in row:
                width = child.winfo_reqwidth()
                if getattr(child, "_flow_stretch", False):
                    width += extra
                # A child whose natural width exceeds the row (a long label, a
                # wide combobox) would otherwise be placed past the right edge
                # and clipped by the parent anyway -- but silently, and the test
                # for overflow would be checking a lie.  Clamp it.
                width = max(1, min(width, available))
                child.place(x=x, y=y, width=width, height=row_height)
                x += width + self._gap
            y += row_height + self._row_gap
        total = max(0, y - self._row_gap)
        if total != self._last_height:
            self._last_height = total
            self.configure(height=total)


def _tip(widget: tk.Widget, text: str) -> None:
    """A minimal hover tooltip.

    Tkinter has none built in, and six new buttons with no explanation would be
    worse than none.  Kept deliberately small: one Toplevel, destroyed on leave.
    """
    state: dict[str, tk.Toplevel | None] = {"window": None}

    def show(_event: tk.Event) -> None:
        if state["window"] is not None:
            return
        try:
            x = widget.winfo_rootx() + 12
            y = widget.winfo_rooty() + widget.winfo_height() + 6
            window = tk.Toplevel(widget)
            window.wm_overrideredirect(True)
            window.wm_geometry(f"+{x}+{y}")
            tk.Label(window, text=text, justify="left", background="#2b2f36",
                     foreground="#d8dee9", relief="solid", borderwidth=1,
                     font=("Microsoft YaHei UI", 9), padx=8, pady=4).pack()
            state["window"] = window
        except tk.TclError:
            state["window"] = None

    def hide(_event: tk.Event) -> None:
        window = state["window"]
        state["window"] = None
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass

    widget.bind("<Enter>", show, add="+")
    widget.bind("<Leave>", hide, add="+")
    widget.bind("<ButtonPress>", hide, add="+")


def _make_root() -> tk.Tk:
    """Prefer a Tk root that accepts OS file drops.

    ``TkinterDnD.Tk`` is a normal Tk root with the tkdnd extension loaded, so
    everything else works identically when it is unavailable.
    """
    try:
        from tkinterdnd2 import TkinterDnD

        return TkinterDnD.Tk()
    except Exception:  # noqa: BLE001 - plain Tkinter still works, minus drops
        return tk.Tk()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="qwen35 gui",
        description="Standalone-window GUI for the qwen35 harness.",
    )
    parser.add_argument("--workspace", "-w", default=None,
                        help="workspace directory (root/system dirs are refused)")
    parser.add_argument("--model-dir", dest="model_dir", default=None)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default=DEFAULT_PROFILE)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--attach", action="append", default=None,
                        help="attach a file at startup (repeatable)")
    parser.add_argument("--no-server", dest="no_server", action="store_true",
                        help="do not start the model server automatically")
    parser.add_argument("--no-manage", dest="no_manage", action="store_true",
                        help="never stop the server on exit")
    args = parser.parse_args(argv)

    workspace = args.workspace or (Path.cwd() if _cwd_is_usable() else None)
    if workspace is None:
        workspace = Path.home() / "qwen35-workspace"
        workspace.mkdir(parents=True, exist_ok=True)

    try:
        root = _make_root()
    except tk.TclError as exc:
        print(f"cannot open a window: {exc}")
        print("This machine has no usable display for Tkinter.")
        return 1

    gui = HarnessGUI(
        root,
        workspace=workspace,
        model_dir=args.model_dir,
        context=args.context,
        profile=args.profile,
        port=args.port,
        auto_start_server=not args.no_server,
        manage_server=not args.no_manage,
    )
    gui.register_dnd()
    if args.attach:
        gui._add_files([str(Path(p)) for p in args.attach])
    root.mainloop()
    return 0


def _cwd_is_usable() -> bool:
    try:
        validate_workspace(str(Path.cwd()))
        return True
    except WorkspaceError:
        return False
