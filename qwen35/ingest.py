"""
File ingestion: turn common file formats into something a 0.8 B model can read.

The design rule is that **the harness does the decomposing, not the model**.  A
PDF, a spreadsheet or a slide deck is not text, and a sub-billion-parameter model
will never parse one.  So each dropped file is converted, once, into a clean
page-marked Markdown document written *inside the workspace*, and the model is
handed a short card telling it what the file is and how to read it with the tool
it already has.

That last part matters more than the conversion itself.  Dumping a 40-page PDF
into the prompt would consume the entire context and still fail.  Instead the
converted text lives at ``<workspace>/.qwen35/attachments/N-slug.md`` and the
model pulls it in windows with ``read_file`` — so a long document is handled with
the harness's existing context discipline rather than in spite of it.  Small
documents are inlined, because making the model spend a step on a two-line file
is just waste.

Nothing here needs a network, and only PDF/images need a third-party library
(PyMuPDF, which ships with the runtime).  DOCX/XLSX/PPTX are read through
python-docx/openpyxl/python-pptx when present and through a built-in ZIP+XML
reader otherwise, so the harness never hard-fails on a missing package.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
import zlib
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator

from .context import estimate_tokens

__all__ = [
    "Attachment",
    "IngestError",
    "ingest",
    "ingest_many",
    "build_preamble",
    "compose_task",
    "inline_budget_for",
    "detect_kind",
    "decode_text",
    "SUPPORTED_EXTENSIONS",
]

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
STATE_DIR = ".qwen35"
ATTACH_DIR = "attachments"
#: Anything at or below this many *tokens* is pasted straight into the prompt.
#:
#: Token-based, not character-based.  The old 1 400-*character* limit was
#: language-dependent: 1 400 characters of English is ~380 tokens, but 1 400
#: characters of Chinese is ~1 400 tokens.  A 2 909-byte Chinese procurement
#: spreadsheet (~973 tokens, 3% of a 32 K window) therefore missed the cut and
#: was left on disk -- forcing the model to fetch it with read_file, a round trip
#: the 0.8 B model then botched, producing no answer at all after 52 seconds.
#:
#: 2 500 tokens is 8% of a 32 K window and 1% of this model's 262 K, so inlining
#: at this size is free.  A single wasted step costs far more than the tokens.
INLINE_TOKEN_BUDGET = 2_500
#: Absolute character ceiling, so one pathological long line cannot blow up.
INLINE_CHAR_LIMIT = 20_000


def inline_budget_for(context_window: int) -> int:
    """How many tokens of attachment text to paste rather than defer.

    **Inlining is not more expensive than deferring.**  The text enters the
    context either way, so the prefill cost is identical; the only difference is
    that a deferred file costs the model an extra step to fetch it -- and a
    0.8 B model asked to reproduce a 60-character Chinese path sometimes fails
    and burns the whole run, which is exactly what happened with a 7 839-token
    ``.doc``.

    So the budget is generous, bounded only by what the window can hold
    alongside the conversation: a third of it, between 2 500 and 12 000 tokens.
    """
    if context_window <= 0:
        return INLINE_TOKEN_BUDGET
    return max(INLINE_TOKEN_BUDGET, min(12_000, int(context_window * 0.35)))
#: Guard against a pathological file; the stored copy is windowed by read_file.
MAX_STORED_CHARS = 3_000_000
MAX_TABLE_ROWS = 200
MAX_TABLE_COLS = 14
MAX_PDF_PAGES_TEXT = 400
MAX_RENDERED_PAGES = 20
RENDER_DPI = 150


class IngestError(Exception):
    """Raised when a file cannot be converted at all."""


# --------------------------------------------------------------------------
# Kinds
# --------------------------------------------------------------------------
_KIND_BY_EXT: dict[str, str] = {}


def _register(kind: str, *exts: str) -> None:
    for e in exts:
        _KIND_BY_EXT[e] = kind


_register("text", ".txt", ".md", ".markdown", ".rst", ".log", ".ini", ".cfg",
          ".conf", ".toml", ".yaml", ".yml", ".json", ".jsonl", ".ndjson",
          ".xml", ".html", ".htm", ".css", ".scss", ".less", ".tex", ".rtf")
_register("code", ".py", ".pyw", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
          ".java", ".kt", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".cs",
          ".go", ".rs", ".rb", ".php", ".pl", ".lua", ".r", ".jl", ".swift",
          ".scala", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".sql",
          ".vue", ".svelte", ".dart", ".m", ".mm", ".f90", ".asm", ".dockerfile")
_register("csv", ".csv", ".tsv")
_register("pdf", ".pdf")
_register("docx", ".docx", ".docm")
_register("xlsx", ".xlsx", ".xlsm", ".xltx")
_register("pptx", ".pptx", ".pptm")
_register("image", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif",
          ".tiff", ".ico", ".heic")
_register("archive", ".zip", ".tar", ".gz", ".7z", ".rar")
_register("doc", ".doc", ".dot", ".rtf")
_register("legacy-office", ".xls", ".ppt", ".pps")

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_KIND_BY_EXT)

_TEXTUAL_KINDS = frozenset({"text", "code", "csv"})


def detect_kind(path: Path) -> str:
    name = path.name.lower()
    if name in ("dockerfile", "makefile", "cmakelists.txt", "readme"):
        return "text"
    suffix = path.suffix.lower()
    if suffix in _KIND_BY_EXT:
        return _KIND_BY_EXT[suffix]
    return "binary"


#: Magic numbers, checked when the name is no help.  A file with no extension --
#: a screenshot dropped straight out of an object store, a download that lost its
#: name -- is otherwise unusable, which is a real failure: one GUI run reported
#: "no attachment could be converted" for a JPEG that was perfectly readable.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image"),
    (b"\xff\xd8\xff", "image"),
    (b"GIF87a", "image"),
    (b"GIF89a", "image"),
    (b"BM", "image"),
    (b"II*\x00", "image"),          # TIFF, little endian
    (b"MM\x00*", "image"),          # TIFF, big endian
    (b"%PDF", "pdf"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "legacy-office"),
    (b"{\\rtf", "doc"),
    (b"<?xml", "text"),
)


def sniff_kind(raw: bytes) -> str:
    """Guess a kind from the leading bytes.  ``""`` when nothing matches.

    Deliberately conservative: it only ever *adds* a kind where the name gave
    none, so a file called ``notes.txt`` is still text.
    """
    if not raw:
        return ""
    head = raw[:16]
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            return kind
    if head.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image"
    if head.startswith(b"PK\x03\x04"):
        # an Office file or a plain zip; the extension decides which, and the
        # converters fall back to raw XML/ZIP reading anyway
        return "archive"
    if raw.startswith(b"\x1f\x8b"):
        return "archive"
    return ""


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------
_ENCODING_HINTS = ("utf-8-sig", "utf-8", "utf-16", "gb18030", "big5", "shift_jis",
                   "euc-kr", "cp1252", "latin-1")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
                  r"\uac00-\ud7af]")


def decode_text(raw: bytes) -> tuple[str, str]:
    """Decode bytes to text, with Chinese encodings handled deliberately.

    ``charset_normalizer`` is not used for the decision: it maps a GBK-encoded
    Chinese file to ``cp949`` (Korean) because that also decodes the bytes, and
    the result is silent mojibake.  The order below is chosen so the common cases
    are unambiguous, and each candidate is scored rather than merely accepted.
    """
    if not raw:
        return "", "empty"
    for bom, enc in ((b"\xef\xbb\xbf", "utf-8-sig"),
                     (b"\xff\xfe\x00\x00", "utf-32-le"),
                     (b"\x00\x00\xfe\xff", "utf-32-be"),
                     (b"\xff\xfe", "utf-16-le"),
                     (b"\xfe\xff", "utf-16-be")):
        if raw.startswith(bom):
            try:
                # Decoding the BOM bytes as utf-16-le/be yields U+FEFF as a
                # character rather than consuming it, so strip it explicitly.
                return raw.decode(enc).lstrip("\ufeff"), enc
            except UnicodeDecodeError:
                break

    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    # Not UTF-8.  Score the CJK candidates; gb18030 decodes almost any byte
    # sequence, so "it decoded" is not evidence -- the *content* has to look
    # like text.
    best: tuple[float, str, str] | None = None
    for enc in ("gb18030", "big5", "shift_jis", "cp1252", "latin-1"):
        try:
            candidate = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        score = _text_score(candidate)
        if best is None or score > best[0]:
            best = (score, enc, candidate)
    if best is not None:
        return best[2], best[1]

    return raw.decode("utf-8", errors="replace"), "utf-8(replace)"


def _text_score(text: str) -> float:
    """How much does this look like real text rather than an accident?

    Private-use-area characters are the tell for a wrong CJK decode: ``gb18030``
    decodes Big5 and Shift-JIS bytes without raising, but the result lands in the
    PUA.  Accepting "it decoded" is not enough; the *content* has to look like
    language.
    """
    if not text:
        return 0.0
    sample = text[:4000]
    printable = sum(1 for ch in sample if ch.isprintable() or ch in "\n\r\t")
    cjk = len(_CJK.findall(sample))
    control = sum(1 for ch in sample if ord(ch) < 32 and ch not in "\n\r\t")
    pua = sum(1 for ch in sample if 0xE000 <= ord(ch) <= 0xF8FF)
    replacement = sample.count("\ufffd")
    return (printable / len(sample)
            + min(cjk, 400) / 400.0 * 0.35
            - control / 50.0
            - pua / 10.0
            - replacement / 10.0)


# --------------------------------------------------------------------------
# Attachment
# --------------------------------------------------------------------------
@dataclass
class Attachment:
    """One ingested file, plus how to describe it to the model."""

    source: Path
    kind: str
    text: str = ""
    unit: str = "lines"           # pages | sheets | slides | rows | lines
    units: int = 0
    warnings: list[str] = field(default_factory=list)
    #: workspace-relative paths of images the model can view
    images: list[str] = field(default_factory=list)
    #: workspace-relative path of the converted Markdown, if it was stored
    out_rel: str = ""
    stored_lines: int = 0
    inlined: bool = False
    size_bytes: int = 0

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text) if self.text else 0

    @property
    def title(self) -> str:
        return self.source.name

    def summary_line(self) -> str:
        bits = [f"{self.kind}"]
        if self.units:
            bits.append(f"{self.units} {self.unit}")
        if self.text:
            bits.append(f"~{self.tokens:,} tokens")
        if self.images:
            bits.append(f"{len(self.images)} image(s)")
        return f"{self.title} — " + ", ".join(bits)

    def card(self) -> str:
        """The model-facing description.  Short, factual, actionable."""
        if self.kind == "unsupported":
            lines = [f"### {self.title}", "- this file could NOT be read:"]
            lines += [f"  - {w}" for w in self.warnings]
            return "\n".join(lines)

        lines = [f"### {self.title}",
                 f"- type: {self.kind}"
                 + (f", {self.units} {self.unit}" if self.units else "")
                 + f", {self.size_bytes:,} bytes"]
        if self.inlined and self.text:
            # Deliberately NO path and NO read_file suggestion here.
            #
            # The card used to say "the text is at the end of this message" and
            # then, one line later, "to read more of it use read_file(...)".  The
            # model obeyed the tool suggestion and ignored the text: a real run
            # spent eight steps on list_dir/read_file while the full answer sat
            # inlined in the prompt, and produced nothing at all.  Naming a tool
            # is an instruction to call it; when the content is already here,
            # there is nothing to call.
            lines.append("- the FULL extracted text is included at the END of "
                         "this message")
            lines.append("- answer from that text directly - no tool call is "
                         "needed to read this file")
        elif self.out_rel:
            lines.append(
                f"- it is {self.stored_lines:,} lines and too large to include "
                f"here; the extracted text is saved at: {self.out_rel}"
            )
            lines.append(
                "- read it with read_file, copying the path exactly: "
                f'read_file(path="{self.out_rel}", start_line=1, max_lines=100)'
            )
        for img in self.images[:4]:
            lines.append(f'- an image is available: view_image(path="{img}")')
        if len(self.images) > 4:
            lines.append(f"- ...and {len(self.images) - 4} more page images "
                         f"in the same folder")
        for w in self.warnings:
            lines.append(f"- note: {w}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Table / text helpers
# --------------------------------------------------------------------------
def _md_cell(value: Any) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()
    if len(s) > 120:
        s = s[:119] + "…"
    return s


def _md_table(rows: Iterable[Iterable[Any]]) -> str:
    """Render rows as a pipe table, capped so one sheet cannot flood the file."""
    out: list[str] = []
    width = 0
    for i, row in enumerate(rows):
        cells = [_md_cell(c) for c in row]
        if not any(cells):
            continue
        if len(cells) > MAX_TABLE_COLS:
            cells = cells[:MAX_TABLE_COLS] + ["…"]
        if i == 0:
            width = len(cells)
            out.append("| " + " | ".join(cells) + " |")
            out.append("|" + "|".join([" --- "] * width) + "|")
            continue
        if len(out) - 2 >= MAX_TABLE_ROWS:
            out.append(f"| … (truncated at {MAX_TABLE_ROWS} rows) |"
                       + " |" * (width - 1))
            break
        if len(cells) < width:
            cells += [""] * (width - len(cells))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _read_csv_text(text: str, delimiter: str) -> str:
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows: list[list[str]] = []
    for i, row in enumerate(reader):
        rows.append(row)
        if i > MAX_TABLE_ROWS + 5:
            break
    return _md_table(rows)


# --------------------------------------------------------------------------
# Converters
# --------------------------------------------------------------------------
def _convert_text(path: Path, raw: bytes) -> tuple[str, str, int, list[str]]:
    text, enc = decode_text(raw)
    notes = [] if enc.startswith("utf-8") or enc == "ascii" else [f"decoded as {enc}"]
    return text, "lines", text.count("\n") + 1, notes


def _convert_csv(path: Path, raw: bytes) -> tuple[str, str, int, list[str]]:
    text, enc = decode_text(raw)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    if path.suffix.lower() == ".csv" and text.count(";") > text.count(","):
        delimiter = ";"
    body = _read_csv_text(text, delimiter)
    return (f"<!-- table: {path.name} (delimiter {delimiter!r}) -->\n\n{body}",
            "rows", text.count("\n") + 1, [])


def _convert_json(path: Path, raw: bytes) -> tuple[str, str, int, list[str]]:
    text, enc = decode_text(raw)
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return _convert_jsonl(path, text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return (f"<!-- {path.name}: not valid JSON ({exc}) -->\n\n"
                f"```json\n{text[:20000]}\n```", "lines", text.count("\n") + 1,
                [f"invalid JSON: {exc}"])
    pretty = json.dumps(data, indent=2, ensure_ascii=False)
    shape = _json_shape(data)
    return (f"<!-- json structure: {shape} -->\n\n```json\n{pretty}\n```",
            "lines", pretty.count("\n") + 1, [])


def _convert_jsonl(path: Path, text: str) -> tuple[str, str, int, list[str]]:
    """Newline-delimited JSON: show the schema of the first rows, not all of them."""
    records: list[Any] = []
    bad = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
        if len(records) >= 5:
            break
    warnings = [f"{bad} line(s) were not valid JSON"] if bad else []
    total = sum(1 for line in text.splitlines() if line.strip())
    if records and isinstance(records[0], dict):
        keys = list(records[0].keys())
        body = _md_table([[k for k in keys]]
                         + [[_json_scalar(r.get(k)) for k in keys]
                            for r in records if isinstance(r, dict)])
        head = (f"<!-- JSON Lines: {total} records, "
                f"first row schema: {_json_shape(records[0])} -->")
    else:
        body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
        head = f"<!-- JSON Lines: {total} records -->"
    return f"{head}\n\n{body}", "records", total, warnings


def _json_scalar(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)[:120]
    return "" if value is None else str(value)


def _convert_html(path: Path, raw: bytes) -> tuple[str, str, int, list[str]]:
    """Strip markup down to readable text -- the model does not need the tags."""
    text, enc = decode_text(raw)
    parser = _HtmlText()
    try:
        parser.feed(text)
    except Exception as exc:  # noqa: BLE001
        return text, "lines", text.count("\n") + 1, [f"HTML parse issue: {exc}"]

    body = re.sub(r"[ \t]+", " ", "".join(parser.parts))
    body = re.sub(r"\n\s*\n\s*\n+", "\n\n", body).strip()
    head = f"<!-- html: {path.name}"
    if parser.title:
        head += f" | title: {parser.title.strip()[:120]}"
    head += " -->"
    notes = [] if enc.startswith("utf-8") else [f"decoded as {enc}"]
    return f"{head}\n\n{body}", "lines", body.count("\n") + 1, notes


class _HtmlText(HTMLParser):
    """Minimal HTML-to-text: keep prose, list markers and table cells only."""

    SKIP = {"script", "style", "head", "noscript", "svg", "template", "iframe"}
    BLOCK = {"p", "div", "br", "section", "article", "header", "footer", "tr",
             "blockquote", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: str = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in ("td", "th"):
            self.parts.append("| ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        chunk = data.strip()
        if not chunk:
            return
        # Check the title BEFORE the skip counter: <title> lives inside <head>,
        # which is skipped, so the title was being discarded with the rest of it.
        if self._in_title:
            self.title += chunk + " "
            return
        if self._skip:
            return
        self.parts.append(chunk + " ")


def _json_shape(data: Any, depth: int = 0) -> str:
    if depth > 3:
        return "…"
    if isinstance(data, dict):
        keys = list(data)[:12]
        inner = ", ".join(f"{k}: {_json_shape(data[k], depth + 1)}" for k in keys)
        more = f", +{len(data) - 12} more" if len(data) > 12 else ""
        return "{" + inner + more + "}"
    if isinstance(data, list):
        return f"[{len(data)} items]" + (f" of {_json_shape(data[0], depth+1)}"
                                         if data else "")
    return type(data).__name__


def _pdf_engine() -> Any:
    """Return the PyMuPDF module, or ``None`` when it is not installed.

    **Licensing note.**  PyMuPDF is dual-licensed AGPL-3.0 / commercial.  Using it
    here is fine for a private deployment, but anything that *conveys* the
    harness together with PyMuPDF must satisfy the AGPL.  The harness therefore
    treats it as an optional accelerator and falls back to a standard-library
    reader (:func:`_pdf_text_stdlib`) rather than depending on it.  openpyxl,
    python-docx and python-pptx are MIT and carry no such caveat.
    """
    try:
        import pymupdf  # type: ignore

        return pymupdf
    except ImportError:
        try:
            import fitz  # type: ignore

            return fitz
        except ImportError:
            return None


_PDF_PDFDOC_ENCODINGS = (b"UniGB-UTF16", b"UniKS-UTF16", b"UniJIS-UTF16",
                         b"UniCNS-UTF16", b"Identity-H", b"UTF-16")


def _pdf_content_streams(raw: bytes) -> Iterator[bytes]:
    """Yield decompressed content streams, scanning linearly.

    A backtracking regex over a PDF container can hang for minutes on a large
    file with an embedded CJK font; ``bytes.find`` over the same data is
    instantaneous, so the scan below is deliberately linear and bounded.
    """
    pos = 0
    size = len(raw)
    while pos < size:
        start = raw.find(b"stream", pos)
        if start < 0:
            return
        header = raw[max(0, start - 3000):start]
        body = start + 6
        if raw[body:body + 2] == b"\r\n":
            body += 2
        elif raw[body:body + 1] in (b"\n", b"\r"):
            body += 1
        end = raw.find(b"endstream", body)
        if end < 0:
            return
        pos = end + 9
        data = raw[body:end]
        if b"FlateDecode" in header:
            try:
                data = zlib.decompress(data)
            except zlib.error:
                try:
                    data = zlib.decompressobj().decompress(data)
                except zlib.error:
                    continue
        yield data


def _pdf_decode_bytes(buf: bytes, context: bytes, hex_string: bool) -> str:
    """Turn a PDF string's bytes into text, guessing the encoding."""
    if not buf:
        return ""
    if hex_string:
        for marker in _PDF_PDFDOC_ENCODINGS:
            if marker in context:
                try:
                    return buf.decode("utf-16-be")
                except UnicodeDecodeError:
                    break
        # a pure-ASCII hex string is usually plain text
        if all(b < 0x80 or b in (0x0a, 0x0d) for b in buf):
            return buf.decode("latin-1")
        try:
            return buf.decode("utf-16-be")
        except UnicodeDecodeError:
            return buf.decode("latin-1", "replace")
    if buf.startswith(b"\xfe\xff"):
        try:
            return buf[2:].decode("utf-16-be")
        except UnicodeDecodeError:
            pass
    return buf.decode("latin-1")


def _pdf_stream_text(content: bytes) -> str:
    """Extract the text-showing operations from one content stream."""
    pieces: list[str] = []
    pending: list[str] = []
    gaps: list[bool] = []          # True where a TJ array asked for a space
    index = 0
    size = len(content)

    def flush() -> None:
        if not pending:
            return
        chunk = ""
        for text, gap in zip(pending, gaps + [False] * len(pending)):
            if gap and chunk and not chunk.endswith(" "):
                chunk += " "
            chunk += text
        if chunk.strip():
            pieces.append(chunk)
        pending.clear()
        gaps.clear()

    while index < size:
        byte = content[index:index + 1]
        if byte == b"(":
            depth = 1
            cursor = index + 1
            buf = bytearray()
            while cursor < size and depth:
                char = content[cursor:cursor + 1]
                if char == b"\\":
                    nxt = content[cursor + 1:cursor + 2]
                    escaped = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b",
                               b"f": b"\f", b"(": b"(", b")": b")", b"\\": b"\\"}
                    if nxt in escaped:
                        buf += escaped[nxt]
                        cursor += 2
                        continue
                    if nxt.isdigit():
                        digits = b""
                        probe = cursor + 1
                        while (probe < size and len(digits) < 3
                               and content[probe:probe + 1].isdigit()):
                            digits += content[probe:probe + 1]
                            probe += 1
                        buf.append(int(digits, 8) & 0xFF)
                        cursor = probe
                        continue
                    buf += nxt
                    cursor += 2
                    continue
                if char == b"(":
                    depth += 1
                elif char == b")":
                    depth -= 1
                    if depth == 0:
                        break
                buf += char
                cursor += 1
            pending.append(_pdf_decode_bytes(bytes(buf), content[:index], False))
            gaps.append(False)
            index = cursor + 1
            continue
        if byte == b"<" and content[index + 1:index + 2] != b"<":
            close = content.find(b">", index)
            if close < 0:
                break
            hexdigits = re.sub(rb"[^0-9A-Fa-f]", b"", content[index + 1:close])
            if len(hexdigits) % 2:
                hexdigits += b"0"
            try:
                decoded = bytes.fromhex(hexdigits.decode("ascii"))
            except (ValueError, UnicodeDecodeError):
                decoded = b""
            pending.append(_pdf_decode_bytes(decoded, content[:index], True))
            gaps.append(False)
            index = close + 1
            continue
        if byte.isdigit() or byte == b"-":
            # inside a TJ array a large negative number means "insert a space"
            cursor = index
            while cursor < size and content[cursor:cursor + 1] in b"-0123456789.":
                cursor += 1
            try:
                if float(content[index:cursor]) < -100:
                    if gaps:
                        gaps[-1] = True
            except ValueError:
                pass
            index = cursor
            continue
        if byte in b"Tj'\"":
            flush()
            index += 1
            continue
        if content[index:index + 2] == b"TJ":
            flush()
            index += 2
            continue
        index += 1

    flush()
    return " ".join(pieces)


def _pdf_text_stdlib(raw: bytes) -> tuple[str, int, list[str]]:
    """Best-effort PDF text with only the standard library.

    Handles ``(...)`` literals, ``<hex>`` strings and the ``Tj``/``TJ``/``'``/``"``
    operators.  It cannot resolve ``/Identity-H`` CID fonts without a ToUnicode
    CMap, so some documents will extract poorly -- which is exactly why PyMuPDF
    is preferred when it is available.
    """
    pages: list[str] = []
    for i, content in enumerate(_pdf_content_streams(raw), 1):
        text = _pdf_stream_text(content).strip()
        text = re.sub(r"[ \t]{2,}", " ", text)
        if text:
            pages.append(f"<!-- page {i} (approximate) -->\n\n{text}")
    warnings = [
        "PyMuPDF is not installed (it is AGPL-licensed), so the built-in "
        "best-effort PDF reader was used; page numbers and some fonts may be "
        "inaccurate. Install PyMuPDF for reliable extraction."
    ]
    return "\n\n".join(pages), len(pages), warnings


def _convert_pdf(path: Path, raw: bytes) -> tuple[str, str, int, list[str], dict]:
    engine = _pdf_engine()
    if engine is None:
        text, pages, warnings = _pdf_text_stdlib(raw)
        if not text.strip():
            raise IngestError(
                f"{path.name}: no text could be extracted without PyMuPDF. "
                "Install it (`pip install PyMuPDF`) or export the PDF as text."
            )
        return text, "pages", pages, warnings, {"total": pages, "scanned": False}

    warnings: list[str] = []
    parts: list[str] = []
    with engine.open(stream=raw, filetype="pdf") as doc:
        total = doc.page_count
        if total > MAX_PDF_PAGES_TEXT:
            warnings.append(f"only the first {MAX_PDF_PAGES_TEXT} of {total} "
                            "pages were converted")
        textless = 0
        bodies: list[str] = []
        for i in range(min(total, MAX_PDF_PAGES_TEXT)):
            page = doc[i]
            body = page.get_text("text") or ""
            # "This page is a scan" is a claim about *images*, not about length.
            # A one-line note is a text page; a full-page bitmap with a stray page
            # number is not.  Length alone got this wrong in both directions.
            looks_scanned = len(body.strip()) < 12 and bool(page.get_images(full=True))
            if looks_scanned:
                textless += 1
            body = re.sub(r"[ \t]+\n", "\n", body).strip()
            bodies.append(body)
            parts.append(f"<!-- page {i + 1} -->\n\n{body}")

        # A page map, so a small model heads for the right page instead of
        # reading the document front to back.  This is the "segment then reason
        # over notes" pattern with no model call: the harness builds the index.
        if len(bodies) > 6:
            rows: list[list[str]] = [["page", "starts with"]]
            for i, body in enumerate(bodies, 1):
                first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
                rows.append([str(i), first[:90]])
            parts.insert(0,
                         "<!-- document map: read only the pages you need; each "
                         "page is marked with a `<!-- page N -->` comment -->\n\n"
                         + _md_table(rows))
        text = "\n\n".join(parts)

        # A PDF whose pages carry no extractable text is a scan (or outlined
        # text).  Per-page emptiness is the signal; a whole-document character
        # threshold would misjudge a legitimate one-page note.
        scanned = total > 0 and (textless / total) >= 0.6
        if scanned:
            warnings.append(
                "this PDF has no usable text layer (it is a scan), so its pages "
                "are exposed as images for the vision model instead"
            )
        meta = {"total": total, "scanned": scanned, "textless": textless}
    return text, "pages", total, warnings, meta


def _convert_docx(path: Path, raw: bytes) -> tuple[str, str, int, list[str]]:
    warnings: list[str] = []
    try:
        import docx  # type: ignore

        document = docx.Document(io.BytesIO(raw))
        parts: list[str] = []
        headings: list[tuple[int, str]] = []
        # Walk the body so tables appear where they actually sit in the text.
        from docx.table import Table  # type: ignore
        from docx.text.paragraph import Paragraph  # type: ignore

        for child in document.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                para = Paragraph(child, document)
                style = (para.style.name or "").lower()
                text = para.text.strip()
                if not text:
                    continue
                if style.startswith("heading"):
                    level = "".join(c for c in style if c.isdigit()) or "2"
                    lvl = min(6, max(1, int(level)))
                    headings.append((lvl, text))
                    parts.append("#" * lvl + " " + text)
                elif style.startswith("list"):
                    parts.append(f"- {text}")
                else:
                    parts.append(text)
            elif tag == "tbl":
                table = Table(child, document)
                rows = [[c.text for c in r.cells] for r in table.rows]
                parts.append(_md_table(rows))
        # An outline lets the model see the shape of a long document at a glance.
        if len(headings) >= 4:
            outline = "<!-- outline -->\n" + "\n".join(
                f"{'  ' * (lvl - 1)}- {text}" for lvl, text in headings)
            parts.insert(0, outline)
        text = "\n\n".join(parts)
        return text, "lines", text.count("\n") + 1, warnings
    except ImportError:
        warnings.append("python-docx not available; used the built-in OOXML reader")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"python-docx failed ({type(exc).__name__}); fell back to "
                        "the built-in OOXML reader")

    text = _docx_via_zip(raw)
    return text, "lines", text.count("\n") + 1, warnings


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_via_zip(raw: bytes) -> str:
    """Minimal, dependency-free DOCX text reader."""
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        try:
            xml = zf.read("word/document.xml")
        except KeyError as exc:
            raise IngestError("not a DOCX file (word/document.xml missing)") from exc
    root = ET.fromstring(xml)

    # Walk only the *direct* children of <w:body>.  Iterating the whole tree
    # visits the <w:p> inside every table cell too, so the table text was emitted
    # twice -- once as paragraphs and once as the table.
    body = root.find(_W + "body")
    nodes = list(body) if body is not None else list(root)

    parts: list[str] = []
    for node in nodes:
        tag = node.tag
        if tag == _W + "p":
            line = "".join(t.text or "" for t in node.iter(_W + "t")).strip()
            if line:
                parts.append(line)
        elif tag == _W + "tbl":
            rows: list[list[str]] = []
            for tr in node.findall(_W + "tr"):
                cells = ["".join(t.text or "" for t in tc.iter(_W + "t"))
                         for tc in tr.findall(_W + "tc")]
                rows.append(cells)
            if rows:
                parts.append(_md_table(rows))
    return "\n\n".join(parts)


def _convert_xlsx(path: Path, raw: bytes) -> tuple[str, str, int, list[str]]:
    warnings: list[str] = []
    try:
        import openpyxl  # type: ignore

        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        parts: list[str] = []
        count = 0
        for name in wb.sheetnames:
            ws = wb[name]
            count += 1
            rows = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                rows.append(list(row))
                if i >= MAX_TABLE_ROWS + 5:
                    break
            parts.append(f"<!-- sheet: {name} -->\n\n" + _md_table(rows))
        wb.close()
        text = "\n\n".join(parts)
        return text, "sheets", count, warnings
    except ImportError:
        warnings.append("openpyxl not available; used the built-in XLSX reader")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"openpyxl failed ({type(exc).__name__}); used the "
                        "built-in XLSX reader")

    text, sheets = _xlsx_via_zip(raw)
    return text, "sheets", sheets, warnings


_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _xlsx_via_zip(raw: bytes) -> tuple[str, int]:
    """Minimal, dependency-free XLSX reader (shared strings + first values)."""
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = set(zf.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(_S + "si"):
                shared.append("".join(t.text or "" for t in si.iter(_S + "t")))
        subtitle = "xl/workbook.xml"
        sheets: list[str] = []
        if subtitle in names:
            wbroot = ET.fromstring(zf.read(subtitle))
            for sh in wbroot.iter(_S + "sheet"):
                sheets.append(sh.get("name") or "sheet")
        sheet_files = sorted(n for n in names
                             if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        parts: list[str] = []
        for idx, sf in enumerate(sheet_files):
            label = sheets[idx] if idx < len(sheets) else f"sheet{idx + 1}"
            rows: list[list[str]] = []
            root = ET.fromstring(zf.read(sf))
            for row in root.iter(_S + "row"):
                cells: list[str] = []
                for c in row.findall(_S + "c"):
                    ctype = c.get("t")
                    value_node = c.find(_S + "v")
                    if ctype == "s" and value_node is not None:
                        try:
                            cells.append(shared[int(value_node.text or "0")])
                        except (ValueError, IndexError):
                            cells.append("")
                    elif ctype == "inlineStr":
                        cells.append("".join(t.text or "" for t in c.iter(_S + "t")))
                    else:
                        cells.append(value_node.text if value_node is not None else "")
                rows.append(cells)
                if len(rows) > MAX_TABLE_ROWS + 5:
                    break
            parts.append(f"<!-- sheet: {label} -->\n\n" + _md_table(rows))
        return "\n\n".join(parts), len(sheet_files)


def _convert_pptx(path: Path, raw: bytes) -> tuple[str, str, int, list[str]]:
    warnings: list[str] = []
    try:
        from pptx import Presentation  # type: ignore

        prs = Presentation(io.BytesIO(raw))
        parts: list[str] = []
        count = 0
        for i, slide in enumerate(prs.slides, 1):
            count += 1
            chunks: list[str] = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    text = shape.text_frame.text.strip()
                    if text:
                        chunks.append(text)
                elif getattr(shape, "has_table", False):
                    rows = [[c.text for c in r.cells] for r in shape.table.rows]
                    chunks.append(_md_table(rows))
            notes = ""
            try:
                if slide.has_notes_slide:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
            except Exception:  # noqa: BLE001
                notes = ""
            body = "\n\n".join(chunks)
            section = f"<!-- slide {i} -->\n\n{body}"
            if notes:
                section += f"\n\n> speaker notes: {notes}"
            parts.append(section)
        text = "\n\n".join(parts)
        return text, "slides", count, warnings
    except ImportError:
        warnings.append("python-pptx not available; used the built-in OOXML reader")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"python-pptx failed ({type(exc).__name__}); used the "
                        "built-in OOXML reader")

    text, slides = _pptx_via_zip(raw)
    return text, "slides", slides, warnings


_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _pptx_slide_order(zf: zipfile.ZipFile) -> list[str]:
    """Slide parts in *presentation* order, which is not filename order.

    ``sldIdLst`` carries relationship ids that are not sequential -- a 26-slide
    deck used rId7, rId9, rId10 ... because other ids belong to the notes master
    -- so sorting by filename silently reorders the deck.
    """
    import posixpath

    names = set(zf.namelist())
    if "ppt/presentation.xml" not in names:
        return []
    if "ppt/_rels/presentation.xml.rels" not in names:
        return []
    import xml.etree.ElementTree as ET

    rels: dict[str, str] = {}
    for rel in ET.fromstring(zf.read("ppt/_rels/presentation.xml.rels")):
        rid, target = rel.get("Id"), rel.get("Target")
        if rid and target:
            rels[rid] = posixpath.normpath(posixpath.join("ppt", target))
    order: list[str] = []
    for sld in ET.fromstring(zf.read("ppt/presentation.xml")).iter(_P + "sldId"):
        target = rels.get(sld.get(_R + "id") or "")
        if target and target in names:
            order.append(target)
    return order


def _pptx_via_zip(raw: bytes) -> tuple[str, int]:
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        order = _pptx_slide_order(zf)
        if not order:
            order = sorted(
                (n for n in zf.namelist()
                 if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                key=lambda n: int(re.findall(r"\d+", n)[-1]),
            )
        parts: list[str] = []
        for i, sf in enumerate(order, 1):
            root = ET.fromstring(zf.read(sf))
            lines: list[str] = []
            for para in root.iter(_A + "p"):
                text = "".join(t.text or "" for t in para.iter(_A + "t")).strip()
                if text:
                    lines.append(text)
            parts.append(f"<!-- slide {i} -->\n\n" + "\n".join(lines))
        return "\n\n".join(parts), len(order)


def _convert_doc(path: Path, raw: bytes) -> tuple[str, str, int, list[str]]:
    """Word 97-2003 (``.doc``), plus RTF which often carries the same extension.

    ``.doc`` is still the default in a lot of Chinese offices, so refusing it
    ("save as .docx first") makes the harness useless for the documents people
    actually have.  The OLE2 + piece-table reader in :mod:`qwen35.legacy_word` is
    pure Python and handles the real thing.
    """
    warnings: list[str] = []

    # Some files named .doc are really RTF or HTML; sniff rather than trust the
    # extension, exactly as the other converters do.
    if raw.lstrip()[:5] == b"{\\rtf" or raw[:5] == b"{\\rtf":
        text = _rtf_to_text(raw)
        if text.strip():
            warnings.append("this is an RTF file saved with a .doc extension")
            return text, "lines", text.count("\n") + 1, warnings
    if raw.lstrip()[:6].lower() in (b"<html>", b"<!docty"):
        text, unit, units, _w = _convert_html(path, raw)
        warnings.append("this is an HTML file saved with a .doc extension")
        return text, unit, units, warnings

    from .legacy_word import DocParseError, extract_doc_text

    try:
        text = extract_doc_text(raw)
    except DocParseError as exc:
        raise IngestError(
            f"{path.name}: this .doc could not be parsed ({exc}). If it opens in "
            "Word, save it as .docx and attach that instead."
        ) from exc

    lines = text.count("\n") + 1
    warnings.append("converted from the legacy .doc format (text only; images, "
                    "fonts and layout are not preserved)")
    return text, "lines", lines, warnings


_RTF_DEST = re.compile(r"\{\\\*?\\(?:fonttbl|colortbl|stylesheet|info|pict|"
                       r"header|footer|footnote|generator)")


def _rtf_to_text(raw: bytes) -> str:
    """A small RTF reader: enough for prose, not a full implementation.

    Handles the escapes that actually matter for Chinese documents --
    ``\\uNNNN?`` Unicode escapes, ``\\'hh`` code-page bytes and destination groups
    that must be skipped.
    """
    src = raw.decode("latin-1", "replace")

    # ``\'hh`` bytes are in the document's ANSI code page, and consecutive ones
    # form a multi-byte character.  Decoding each byte on its own as cp1252 (the
    # obvious reading) turns UTF-8 Chinese into mojibake, so runs are buffered
    # and decoded together.
    hexbuf: list[int] = []

    def flush_hex() -> None:
        if not hexbuf:
            return
        blob = bytes(hexbuf)
        hexbuf.clear()
        for encoding in ("utf-8", "gb18030", "cp1252"):
            try:
                out.append(blob.decode(encoding))
                return
            except UnicodeDecodeError:
                continue
        out.append(blob.decode("cp1252", "replace"))

    out: list[str] = []
    i = 0
    depth = 0
    skip_depth = -1
    while i < len(src):
        ch = src[i]
        if ch == "\\":
            nxt = src[i + 1:i + 2]
            if nxt == "u":
                flush_hex()
                digits = ""
                j = i + 2
                if src[j:j + 1] == "-":
                    digits, j = "-", j + 1
                while j < len(src) and src[j].isdigit() and len(digits) < 7:
                    digits += src[j]
                    j += 1
                try:
                    code = int(digits)
                    if code < 0:
                        code += 65536
                    out.append(chr(code))
                except ValueError:
                    pass
                # the escaped char after \uNNNN is a fallback; skip it
                if src[j:j + 1] == " ":
                    j += 1
                elif src[j:j + 1] not in ("", "\\", "{", "}"):
                    j += 1
                i = j
                continue
            if nxt == "'":
                try:
                    hexbuf.append(int(src[i + 2:i + 4], 16))
                except ValueError:
                    pass
                i += 4
                continue
            flush_hex()
            # control words we translate, the rest we ignore
            word = re.match(r"\\([a-z]+)(-?\d+)?", src[i:])
            if word:
                name = word.group(1)
                if name in ("par", "line", "sect", "page"):
                    out.append("\n")
                elif name == "tab":
                    out.append("\t")
                i += word.end()
                continue
            i += 2
            continue
        if ch == "{":
            flush_hex()
            depth += 1
            # a group starting with \* or a known destination is skipped
            probe = src[i + 1:i + 40]
            if probe.startswith("\\*") or _RTF_DEST.match(src[i:]):
                if skip_depth < 0:
                    skip_depth = depth
            i += 1
            continue
        if ch == "}":
            flush_hex()
            if skip_depth >= 0 and depth <= skip_depth:
                skip_depth = -1
            depth -= 1
            i += 1
            continue
        if skip_depth < 0:
            flush_hex()
            if ch not in "\r":
                out.append(ch)
        i += 1
    flush_hex()

    text = "".join(out)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _convert_archive(path: Path, raw: bytes) -> tuple[str, str, int, list[str]]:
    warnings: list[str] = []
    if path.suffix.lower() != ".zip":
        return (f"<!-- {path.name}: archive listing not supported for "
                f"{path.suffix} -->", "lines", 0, ["only .zip can be listed"])
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        infos = zf.infolist()
        lines = [f"<!-- zip archive with {len(infos)} entries -->", "",
                 "| entry | size | compressed |", "| --- | --- | --- |"]
        for info in infos[:MAX_TABLE_ROWS]:
            lines.append(f"| {_md_cell(info.filename)} | {info.file_size:,} | "
                         f"{info.compress_size:,} |")
        if len(infos) > MAX_TABLE_ROWS:
            lines.append(f"| … {len(infos) - MAX_TABLE_ROWS} more | | |")
    if len(infos) > 40:
        warnings.append("the archive holds many files; extract the one you need "
                        "instead of reading this listing")
    return "\n".join(lines), "entries", len(infos), warnings


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
def _slug(name: str, limit: int = 40) -> str:
    """A filesystem-safe stem that **keeps non-ASCII letters**.

    Stripping to ``[A-Za-z0-9._-]`` turned a Chinese filename into the empty
    string, so ``最新慧采采购.xlsx`` was stored as ``01-file-4c7efa3e.md``.  That
    name tells the model nothing about which file it came from, and it is hard to
    reproduce exactly.  It mattered: the 0.8 B model was handed that path, quoted
    it correctly in its reasoning, and then emitted ``read_file(path=".")``
    anyway -- burning the whole run.  Chinese filenames are perfectly valid on
    Windows, so the name is preserved.
    """
    s = re.sub(r"[^\w.\-]+", "-", name, flags=re.UNICODE).strip("-._")
    return s[:limit] or "file"


def _looks_like_text(raw: bytes) -> bool:
    """Would this pass for a text file?  Used only when the name says nothing."""
    sample = raw[:4096]
    if not sample or b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        pass
    text, _enc = decode_text(sample)
    return _text_score(text) > 0.9


def _dispatch_convert(path: Path, raw: bytes, kind: str) -> tuple[
        str, str, int, list[str], dict[str, Any]]:
    """Route a file to its converter.  Returns (text, unit, units, warnings, meta)."""
    suffix = path.suffix.lower()

    if kind in _TEXTUAL_KINDS:
        if kind == "csv":
            text, unit, units, warnings = _convert_csv(path, raw)
        elif suffix in (".json", ".jsonl", ".ndjson"):
            text, unit, units, warnings = _convert_json(path, raw)
        elif suffix in (".html", ".htm"):
            text, unit, units, warnings = _convert_html(path, raw)
        else:
            text, unit, units, warnings = _convert_text(path, raw)
        return text, unit, units, warnings, {}

    if kind == "pdf":
        return _convert_pdf(path, raw)
    if kind == "docx":
        text, unit, units, warnings = _convert_docx(path, raw)
        return text, unit, units, warnings, {}
    if kind == "xlsx":
        text, unit, units, warnings = _convert_xlsx(path, raw)
        return text, unit, units, warnings, {}
    if kind == "pptx":
        text, unit, units, warnings = _convert_pptx(path, raw)
        return text, unit, units, warnings, {}
    if kind == "image":
        return "", "image", 1, [], {}
    if kind == "archive":
        text, unit, units, warnings = _convert_archive(path, raw)
        return text, unit, units, warnings, {}
    if kind == "doc":
        text, unit, units, warnings = _convert_doc(path, raw)
        return text, unit, units, warnings, {}
    if kind == "legacy-office":
        raise IngestError(
            f"{path.name}: the legacy binary format ({path.suffix}) cannot be "
            "read. Open it in Word/Excel and save as .docx/.xlsx, or export to PDF."
        )
    raise IngestError(
        f"{path.name}: unrecognised format ({path.suffix or 'no extension'}). "
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def ingest(
    source: str | Path,
    workspace: str | Path,
    *,
    index: int = 1,
    store: bool = True,
    inline_limit: int = INLINE_CHAR_LIMIT,
    inline_tokens: int = INLINE_TOKEN_BUDGET,
) -> Attachment:
    """Convert one file into a model-readable attachment.

    The converted text and any images are written **inside the workspace**, under
    ``.qwen35/attachments/``.  That is not tidiness: the workspace guard refuses
    to read anything outside the workspace, so a file dropped from the Desktop
    would otherwise be unreachable to the very tools that are supposed to read it.
    """
    path = Path(source)
    if not path.is_file():
        raise IngestError(f"not a file: {path}")
    raw = path.read_bytes()
    if not raw:
        raise IngestError(f"{path.name} is empty")

    kind = detect_kind(path)
    warnings: list[str] = []
    images: list[str] = []
    meta: dict[str, Any] = {}

    # No usable extension?  Look at the bytes.  A file with no name at all is
    # otherwise rejected even when it is a perfectly ordinary image or PDF.
    if kind == "binary" or not path.suffix:
        sniffed = sniff_kind(raw)
        if sniffed and sniffed != kind:
            warnings.append(
                f"no usable file extension; treated as {sniffed} "
                "based on its contents")
            kind = sniffed
        elif not path.suffix and _looks_like_text(raw):
            kind = "text"

    # A converter can still meet something it did not expect -- a file named
    # .xlsx that is not a zip raises BadZipFile, for instance.  Turning that into
    # IngestError keeps one bad file from aborting a whole drop.
    try:
        # NOTE: extend, never rebind.  Assigning to `warnings` here silently threw
        # away the "no usable file extension" note added just above.
        text, unit, units, convert_warnings, meta = _dispatch_convert(path, raw, kind)
        warnings.extend(convert_warnings)
    except IngestError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise IngestError(
            f"{path.name}: could not be read as {kind} "
            f"({type(exc).__name__}: {exc}). The file may be corrupt, or its "
            "extension may not match its contents."
        ) from exc
    if len(text) > MAX_STORED_CHARS:
        text = text[:MAX_STORED_CHARS]
        warnings.append(f"truncated at {MAX_STORED_CHARS:,} characters")

    att = Attachment(source=path, kind=kind, text=text, unit=unit, units=units,
                     warnings=list(warnings), size_bytes=len(raw))

    if not store:
        att.inlined = att.tokens <= inline_tokens and len(text) <= inline_limit
        return att

    dest_dir = Path(workspace) / STATE_DIR / ATTACH_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    # images always need a copy inside the workspace (view_image can only reach
    # paths the guard allows)
    if kind == "image":
        # An extension-less image (an object-store path, a nameless download)
        # would otherwise be stored without one, which is needlessly awkward.
        rel = _copy_into(path, dest_dir, index,
                         suffix=path.suffix.lower() or _image_suffix(raw))
        att.images = [rel]
        att.warnings.append("this is a picture: use view_image to look at it")
        return att

    # a scanned PDF: render pages so the vision encoder can read them
    if kind == "pdf" and meta.get("scanned"):
        folder = dest_dir / f"{index}-{_slug(path.stem)}-pages"
        folder.mkdir(parents=True, exist_ok=True)
        rendered = _render_pdf(raw, folder)
        att.images = [f"{STATE_DIR}/{ATTACH_DIR}/{folder.name}/{p.name}"
                      for p in rendered]
        if not rendered:
            att.warnings.append("page rendering failed; no images available")

    if text.strip():
        # The name is stable and unique: a short hash of the source path means two
        # different files both called "report.pdf" never overwrite each other, and
        # re-attaching the same file is idempotent.  The *readable* part comes
        # first so the model can recognise which attachment it is looking at.
        tag = hashlib.sha1(str(path.resolve()).encode("utf-8", "replace")).hexdigest()[:8]
        out = dest_dir / f"{index:02d}-{_slug(path.stem)}-{tag}.md"
        header = (f"<!-- source: {path.name} | kind: {kind} | "
                  f"{units} {unit} | original {len(raw):,} bytes -->\n\n")
        out.write_text(header + text, encoding="utf-8")
        att.out_rel = f"{STATE_DIR}/{ATTACH_DIR}/{out.name}"
        att.stored_lines = (header + text).count("\n") + 1
        att.inlined = att.tokens <= inline_tokens and len(text) <= inline_limit
    elif not att.images:
        raise IngestError(f"{path.name}: nothing extractable was found")

    return att


_IMAGE_SUFFIX_BY_MAGIC = (
    (b"\x89PNG", ".png"), (b"\xff\xd8\xff", ".jpg"), (b"GIF8", ".gif"),
    (b"BM", ".bmp"), (b"RIFF", ".webp"), (b"II*\x00", ".tiff"),
    (b"MM\x00*", ".tiff"),
)


def _image_suffix(raw: bytes) -> str:
    for magic, suffix in _IMAGE_SUFFIX_BY_MAGIC:
        if raw.startswith(magic):
            return suffix
    return ".png"


def _copy_into(path: Path, dest_dir: Path, index: int,
               suffix: str = "") -> str:
    """Copy a file into the attachment store, de-duplicating by content."""
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    ext = suffix or path.suffix.lower()
    target = dest_dir / f"{index}-{_slug(path.stem)}-{digest}{ext}"
    if not target.exists():
        shutil.copyfile(path, target)
    return f"{STATE_DIR}/{ATTACH_DIR}/{target.name}"


def _render_pdf(raw: bytes, folder: Path) -> list[Path]:
    engine = _pdf_engine()
    if engine is None:
        return []
    out: list[Path] = []
    with engine.open(stream=raw, filetype="pdf") as doc:
        for i in range(min(doc.page_count, MAX_RENDERED_PAGES)):
            page = doc[i]
            pix = page.get_pixmap(dpi=RENDER_DPI)
            target = folder / f"page-{i + 1:03d}.png"
            pix.save(str(target))
            out.append(target)
    return out


def ingest_many(paths: Iterable[str | Path], workspace: str | Path,
                *, inline_limit: int = INLINE_CHAR_LIMIT,
                inline_tokens: int = INLINE_TOKEN_BUDGET) -> list[Attachment]:
    """Ingest several files, keeping per-file failures out of the way."""
    out: list[Attachment] = []
    for i, p in enumerate(paths, 1):
        try:
            out.append(ingest(p, workspace, index=i, inline_limit=inline_limit,
                              inline_tokens=inline_tokens))
        except IngestError as exc:
            failed = Attachment(source=Path(p), kind="unsupported")
            failed.warnings.append(str(exc))
            out.append(failed)
    return out


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def compose_task(task: str, attachments: list[Attachment],
                 *, inline_limit: int = INLINE_CHAR_LIMIT) -> str:
    """Build the whole message: metadata, the request, then the content.

    **Order is deliberate.**  "Lost in the Middle" (arXiv:2307.03172) found that
    models retrieve evidence at the *beginning or end* of a long context and
    degrade badly in between.  An earlier version put the extracted text between
    the attachment card and the request -- the worst possible position -- and the
    model simply ignored it, spending eight steps trying to open the original
    ``.xlsx`` binary instead.  So the extracted text now goes **last**, directly
    before generation.

    The instruction not to open the original matters too: a 0.8 B model sees
    ``scores.xlsx`` in the directory and reaches for it, because nothing told it
    that the content had already been handed over.
    """
    usable = [a for a in attachments if a.kind != "unsupported"]
    failed = [a for a in attachments if a.kind == "unsupported"]
    if not usable and not failed:
        return task

    parts: list[str] = [
        "# Attached files",
        "The user attached the files listed below. They have **already been "
        "converted to text** for you. Answer from that text. The original files "
        "are binary formats you cannot open, so do not try to read them "
        "directly.",
        "\n\n".join(a.card() for a in usable + failed),
    ]

    request = (task or "").strip() or "请阅读这些附件，并总结要点。"
    parts.append("# Request\n" + request)

    inlined = [a for a in usable if a.inlined and a.text]
    pending = [a for a in usable if not a.inlined and a.out_rel]
    if inlined:
        body = "\n\n".join(f"<!-- {a.title} -->\n{a.text}" for a in inlined)
        parts.append("# Extracted content — answer from this\n" + body)
    if pending:
        # These are too big to paste.  The exact call is spelled out because a
        # 0.8 B model asked to reproduce a path from prose will sometimes emit
        # `read_file(path=".")` instead -- the single most expensive failure this
        # harness has, since the run then has no content at all.
        lines = [f"- read_file(path=\"{a.out_rel}\", start_line=1, max_lines=100)"
                 f"   # {a.title}" for a in pending]
        parts.append(
            "# Files to read first\n"
            "These are too large to paste, so read them with the tool before "
            "answering. Copy the path exactly:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def build_preamble(attachments: list[Attachment],
                   *, inline_limit: int = INLINE_CHAR_LIMIT) -> str:
    """Attachment block without a request.  Prefer :func:`compose_task`."""
    return compose_task("", attachments, inline_limit=inline_limit)


def describe_summary(attachments: list[Attachment]) -> str:
    """One line per attachment, for the GUI list."""
    return "\n".join(f"  {a.summary_line()}" for a in attachments)
