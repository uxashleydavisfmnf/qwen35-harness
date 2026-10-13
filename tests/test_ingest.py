"""
Ingestion tests.

The point of this module is that a 0.8 B model can be handed a PDF, a
spreadsheet or a slide deck and still do something useful.  That only works if the
conversion is correct *and* the converted file lands somewhere the model's own
tools can reach it, so both halves are tested here.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen35.config import QWEN35_08B, get_permissions
from qwen35.ingest import (SUPPORTED_EXTENSIONS, Attachment, IngestError,
                           _docx_via_zip, _pptx_via_zip, _xlsx_via_zip,
                           build_preamble, compose_task, decode_text,
                           detect_kind, ingest, ingest_many)
from qwen35.permissions import CommandScreener
from qwen35.tools import ToolContext, call_tool
from qwen35.workspace import WorkspaceGuard


def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _ole2_container(streams: dict[str, bytes]) -> bytes:
    """Build a minimal OLE2 compound file holding the given streams.

    Hand-writing the container keeps the test independent of Word and of any
    conversion tool, and it exercises the reader's sector-chain and directory
    logic rather than a mock of it.  Streams are padded past the 4 096-byte mini
    stream cutoff so everything lives in the main FAT.
    """
    SECTOR = 512
    ENDOFCHAIN = 0xFFFFFFFE
    FREESECT = 0xFFFFFFFF
    FATSECT = 0xFFFFFFFD

    padded = {name: data + b"\x00" * ((-len(data)) % SECTOR)
              for name, data in streams.items()}

    placement: dict[str, tuple[int, int, int]] = {}
    cursor = 2                                   # 0 = FAT, 1 = directory
    for name, data in padded.items():
        count = max(1, len(data) // SECTOR)
        placement[name] = (cursor, count, len(streams[name]))
        cursor += count
    total = cursor

    fat = [FREESECT] * max(total, 128)
    fat[0] = FATSECT
    fat[1] = ENDOFCHAIN
    for start, count, _size in placement.values():
        for i in range(count):
            fat[start + i] = ENDOFCHAIN if i == count - 1 else start + i + 1

    header = bytearray(512)
    header[0:8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    struct.pack_into("<H", header, 0x18, 0x003E)     # minor version
    struct.pack_into("<H", header, 0x1A, 0x0003)     # major version
    struct.pack_into("<H", header, 0x1C, 0xFFFE)     # little endian marker
    struct.pack_into("<H", header, 0x1E, 9)          # 2**9 = 512
    struct.pack_into("<H", header, 0x20, 6)          # 2**6 = 64
    struct.pack_into("<I", header, 0x2C, 1)          # one FAT sector
    struct.pack_into("<I", header, 0x30, 1)          # directory at sector 1
    struct.pack_into("<I", header, 0x38, 4096)       # mini stream cutoff
    struct.pack_into("<I", header, 0x3C, ENDOFCHAIN)  # no mini FAT
    struct.pack_into("<I", header, 0x40, 0)
    struct.pack_into("<I", header, 0x44, ENDOFCHAIN)  # no DIFAT sectors
    struct.pack_into("<I", header, 0x48, 0)
    for i in range(109):
        struct.pack_into("<I", header, 0x4C + 4 * i, 0 if i == 0 else FREESECT)

    fat_sector = struct.pack(f"<{len(fat)}I", *fat)[:SECTOR].ljust(SECTOR, b"\xff")

    def entry(name: str, kind: int, start: int, size: int) -> bytes:
        raw = bytearray(128)
        encoded = name.encode("utf-16-le") + b"\x00\x00"
        raw[0:len(encoded)] = encoded
        struct.pack_into("<H", raw, 0x40, len(encoded))
        raw[0x42] = kind
        raw[0x43] = 1                              # black
        struct.pack_into("<I", raw, 0x44, FREESECT)  # left sibling
        struct.pack_into("<I", raw, 0x48, FREESECT)  # right sibling
        struct.pack_into("<I", raw, 0x4C, FREESECT)  # child
        struct.pack_into("<I", raw, 0x74, start)
        struct.pack_into("<Q", raw, 0x78, size)
        return bytes(raw)

    directory = bytearray(entry("Root Entry", 5, ENDOFCHAIN, 0))
    for name, (start, _count, size) in placement.items():
        directory += entry(name, 2, start, size)
    directory = bytes(directory).ljust(SECTOR, b"\x00")

    body = bytearray()
    for start, count, _size in sorted(placement.values()):
        assert start == 2 + len(body) // SECTOR
        body += padded[[n for n, v in placement.items() if v[0] == start][0]]
    return bytes(header) + fat_sector + directory + bytes(body)


class IngestCase(unittest.TestCase):
    """Base: a source directory and a separate workspace."""

    def setUp(self):
        self._src = tempfile.TemporaryDirectory()
        self._ws = tempfile.TemporaryDirectory()
        self.src = Path(self._src.name)
        self.ws = Path(self._ws.name)

    def tearDown(self):
        self._src.cleanup()
        self._ws.cleanup()

    def make(self, name: str, data: bytes | str) -> Path:
        p = self.src / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            p.write_bytes(data)
        else:
            p.write_text(data, encoding="utf-8")
        return p


# ==========================================================================
# Kind detection and encodings
# ==========================================================================
class TestDetectKind(unittest.TestCase):
    def test_common_extensions(self):
        cases = {".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx", ".pptx": "pptx",
                 ".png": "image", ".jpg": "image", ".csv": "csv", ".md": "text",
                 ".py": "code", ".json": "text", ".zip": "archive",
                 ".doc": "doc", ".rtf": "doc", ".xls": "legacy-office",
                 ".xyz": "binary"}
        for ext, kind in cases.items():
            with self.subTest(ext=ext):
                self.assertEqual(detect_kind(Path("f" + ext)), kind)

    def test_extensionless_names(self):
        self.assertEqual(detect_kind(Path("Dockerfile")), "text")

    def test_supported_set_is_not_empty(self):
        self.assertIn(".pdf", SUPPORTED_EXTENSIONS)
        self.assertIn(".docx", SUPPORTED_EXTENSIONS)


class TestDecodeText(unittest.TestCase):
    def test_utf8(self):
        text, enc = decode_text("hello 世界".encode("utf-8"))
        self.assertEqual(text, "hello 世界")
        self.assertEqual(enc, "utf-8")

    def test_utf8_bom(self):
        text, enc = decode_text("hello".encode("utf-8-sig"))
        self.assertEqual(text, "hello")
        self.assertEqual(enc, "utf-8-sig")

    def test_gbk_chinese_is_not_mojibake(self):
        """The regression that matters: GBK must not decode as Korean.

        charset_normalizer maps these bytes to cp949, which 'succeeds' and
        produces garbage.  The scorer has to prefer the Chinese reading.
        """
        raw = "中文测试内容，这是一段话。".encode("gbk")
        text, enc = decode_text(raw)
        self.assertIn("中文测试内容", text)
        self.assertIn(enc, ("gb18030", "gbk"))

    def test_utf16_with_bom(self):
        text, enc = decode_text("hi there".encode("utf-16"))
        self.assertEqual(text, "hi there")
        self.assertTrue(enc.startswith("utf-16"))

    def test_empty(self):
        self.assertEqual(decode_text(b"")[0], "")

    def test_binary_does_not_crash(self):
        text, _enc = decode_text(bytes(range(256)) * 4)
        self.assertIsInstance(text, str)


# ==========================================================================
# Text-ish formats
# ==========================================================================
class TestTextFormats(IngestCase):
    def test_plain_text(self):
        p = self.make("a.txt", "line one\nline two\n")
        att = ingest(p, self.ws)
        self.assertEqual(att.kind, "text")
        self.assertIn("line one", att.text)
        self.assertTrue(att.out_rel)

    def test_csv_becomes_a_markdown_table(self):
        p = self.make("d.csv", "name,score\nalice,3\nbob,5\n")
        att = ingest(p, self.ws)
        self.assertEqual(att.kind, "csv")
        self.assertIn("| name | score |", att.text)
        self.assertIn("| alice | 3 |", att.text)
        self.assertEqual(att.unit, "rows")

    def test_json_reports_structure(self):
        p = self.make("c.json", json.dumps({"a": 1, "b": [1, 2, 3]}))
        att = ingest(p, self.ws)
        self.assertIn("json structure", att.text)
        self.assertIn('"a": 1', att.text)

    def test_invalid_json_is_reported_not_fatal(self):
        p = self.make("bad.json", "{not json")
        att = ingest(p, self.ws)
        self.assertTrue(any("invalid JSON" in w for w in att.warnings))

    def test_jsonl_becomes_records(self):
        p = self.make("r.jsonl",
                      "\n".join(json.dumps({"i": i, "v": i * 2}) for i in range(4)))
        att = ingest(p, self.ws)
        self.assertEqual(att.unit, "records")
        self.assertIn("| i | v |", att.text)

    def test_html_is_stripped_to_text(self):
        p = self.make("p.html",
                      "<html><head><title>T</title><style>x{}</style></head><body>"
                      "<h1>Head</h1><p>Para.</p><ul><li>one</li><li>two</li></ul>"
                      "<script>var x=1;</script>"
                      "<table><tr><td>a</td><td>b</td></tr></table></body></html>")
        att = ingest(p, self.ws)
        self.assertIn("Head", att.text)
        self.assertIn("- one", att.text)
        self.assertIn("| a | b", att.text)
        self.assertNotIn("var x", att.text)      # script body dropped
        self.assertNotIn("x{}", att.text)        # style body dropped
        self.assertIn("title: T", att.text)

    def test_table_rows_are_capped(self):
        rows = "\n".join(f"r{i},{i}" for i in range(400))
        p = self.make("big.csv", "a,b\n" + rows)
        att = ingest(p, self.ws)
        self.assertIn("truncated", att.text)
        self.assertLess(att.text.count("\n"), 260)

    def test_zip_lists_entries(self):
        p = self.src / "a.zip"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("one.txt", "hello")
            zf.writestr("two/three.txt", "world")
        att = ingest(p, self.ws)
        self.assertEqual(att.kind, "archive")
        self.assertIn("one.txt", att.text)


# ==========================================================================
# Office formats
# ==========================================================================
@unittest.skipUnless(_have("docx"), "python-docx not installed")
class TestDocx(IngestCase):
    def _docx(self, name="d.docx", headings=("Report",), body=True, table=True):
        import docx

        doc = docx.Document()
        for h in headings:
            doc.add_heading(h, level=1)
        if body:
            doc.add_paragraph("Body text.")
        if table:
            t = doc.add_table(rows=2, cols=2)
            t.cell(0, 0).text = "A"
            t.cell(0, 1).text = "B"
            t.cell(1, 0).text = "1"
            t.cell(1, 1).text = "2"
        p = self.src / name
        doc.save(str(p))
        return p

    def test_headings_and_table(self):
        att = ingest(self._docx(), self.ws)
        self.assertEqual(att.kind, "docx")
        self.assertIn("# Report", att.text)
        self.assertIn("Body text.", att.text)
        self.assertIn("| A | B |", att.text)

    def test_table_text_appears_exactly_once(self):
        """Regression: the ZIP fallback walked the whole tree and duplicated it."""
        att = ingest(self._docx(), self.ws)
        self.assertEqual(att.text.count("| A | B |"), 1)

    def test_long_document_gets_an_outline(self):
        att = ingest(self._docx(headings=("One", "Two", "Three", "Four"),
                                body=False, table=False), self.ws)
        self.assertIn("<!-- outline -->", att.text)
        self.assertIn("- One", att.text)

    def test_short_document_gets_no_outline(self):
        att = ingest(self._docx(headings=("Only",)), self.ws)
        self.assertNotIn("<!-- outline -->", att.text)

    def test_zip_fallback_matches_the_library(self):
        p = self._docx()
        fallback = _docx_via_zip(p.read_bytes())
        self.assertIn("Report", fallback)
        self.assertIn("| A | B |", fallback)
        self.assertEqual(fallback.count("| A | B |"), 1)


@unittest.skipUnless(_have("openpyxl"), "openpyxl not installed")
class TestXlsx(IngestCase):
    def _xlsx(self):
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Data"
        ws.append(["name", "score"])
        ws.append(["alice", 3])
        ws.append(["bob", 5])
        second = wb.create_sheet("Other")
        second.append(["k", "v"])
        second.append(["x", 9])
        p = self.src / "b.xlsx"
        wb.save(str(p))
        return p

    def test_sheets_and_values(self):
        att = ingest(self._xlsx(), self.ws)
        self.assertEqual(att.kind, "xlsx")
        self.assertEqual(att.units, 2)
        self.assertIn("<!-- sheet: Data -->", att.text)
        self.assertIn("| alice | 3 |", att.text)
        self.assertIn("<!-- sheet: Other -->", att.text)

    def test_zip_fallback_matches(self):
        text, sheets = _xlsx_via_zip(self._xlsx().read_bytes())
        self.assertEqual(sheets, 2)
        self.assertIn("| alice | 3 |", text)
        self.assertIn("Data", text)


@unittest.skipUnless(_have("pptx"), "python-pptx not installed")
class TestPptx(IngestCase):
    def _pptx(self):
        from pptx import Presentation

        prs = Presentation()
        s1 = prs.slides.add_slide(prs.slide_layouts[5])
        s1.shapes.title.text = "Slide One"
        s2 = prs.slides.add_slide(prs.slide_layouts[5])
        s2.shapes.title.text = "Slide Two"
        p = self.src / "d.pptx"
        prs.save(str(p))
        return p

    def test_slides(self):
        att = ingest(self._pptx(), self.ws)
        self.assertEqual(att.kind, "pptx")
        self.assertEqual(att.units, 2)
        self.assertIn("<!-- slide 1 -->", att.text)
        self.assertIn("Slide One", att.text)
        self.assertIn("<!-- slide 2 -->", att.text)

    def test_zip_fallback_matches(self):
        text, slides = _pptx_via_zip(self._pptx().read_bytes())
        self.assertEqual(slides, 2)
        self.assertIn("Slide One", text)


# ==========================================================================
# PDF
# ==========================================================================
def _pymupdf():
    try:
        import pymupdf  # type: ignore
        return pymupdf
    except ImportError:
        try:
            import fitz  # type: ignore
            return fitz
        except ImportError:
            return None


@unittest.skipUnless(_pymupdf() is not None, "PyMuPDF not installed")
class TestPdf(IngestCase):
    def _pdf(self, pages=("Page one text",), images_only=False):
        mu = _pymupdf()
        doc = mu.open()
        for text in pages:
            page = doc.new_page()
            if images_only:
                pix = mu.Pixmap(mu.csRGB, mu.IRect(0, 0, 300, 200))
                pix.set_rect(pix.irect, (255, 255, 255))
                page.insert_image(mu.Rect(20, 20, 280, 180), pixmap=pix)
            else:
                page.insert_text((72, 100), text, fontsize=14)
        p = self.src / "d.pdf"
        doc.save(str(p))
        doc.close()
        return p

    def test_text_pdf_is_extracted_and_not_called_a_scan(self):
        att = ingest(self._pdf(("PDF hello\npage one",)), self.ws)
        self.assertEqual(att.kind, "pdf")
        self.assertIn("<!-- page 1 -->", att.text)
        self.assertIn("PDF hello", att.text)
        self.assertEqual(att.images, [],
                         "a text PDF must not be rendered to images")
        self.assertFalse(any("scan" in w for w in att.warnings))

    def test_short_page_is_still_a_text_page(self):
        """Regression: a <20-character page was misread as a scan."""
        att = ingest(self._pdf(("Hi",)), self.ws)
        self.assertEqual(att.images, [])
        self.assertIn("Hi", att.text)

    def test_long_pdf_gets_a_document_map(self):
        pages = tuple(f"Section {i}\n" + "word " * 30 for i in range(1, 9))
        att = ingest(self._pdf(pages), self.ws)
        self.assertIn("document map", att.text)
        self.assertIn("| 1 | Section 1 |", att.text)
        self.assertIn("<!-- page 8 -->", att.text)

    def test_short_pdf_gets_no_map(self):
        att = ingest(self._pdf(("One", "Two")), self.ws)
        self.assertNotIn("document map", att.text)

    def test_scanned_pdf_is_rendered_for_the_vision_model(self):
        att = ingest(self._pdf(("", ""), images_only=True), self.ws)
        self.assertTrue(any("scan" in w for w in att.warnings))
        self.assertEqual(len(att.images), 2)
        for rel in att.images:
            full = self.ws / rel
            self.assertTrue(full.is_file(), rel)
            self.assertGreater(full.stat().st_size, 1000)


# ==========================================================================
# Images, failures, and the workspace integration
# ==========================================================================
class TestImagesAndFailures(IngestCase):
    def test_image_is_copied_into_the_workspace(self):
        from PIL import Image

        p = self.src / "pic.png"
        Image.new("RGB", (64, 48), "red").save(p)
        att = ingest(p, self.ws)
        self.assertEqual(att.kind, "image")
        self.assertEqual(len(att.images), 1)
        self.assertTrue((self.ws / att.images[0]).is_file())
        self.assertTrue(any("view_image" in w for w in att.warnings))

    def test_empty_file_is_rejected(self):
        p = self.make("empty.txt", "")
        with self.assertRaises(IngestError):
            ingest(p, self.ws)

    def test_legacy_office_gets_a_useful_message(self):
        p = self.make("old.doc", b"\xd0\xcf\x11\xe0" + b"\x00" * 64)
        with self.assertRaises(IngestError) as ctx:
            ingest(p, self.ws)
        self.assertIn("docx", str(ctx.exception))

    def test_unknown_format_is_rejected(self):
        p = self.make("thing.zzz", b"\x01\x02\x03")
        with self.assertRaises(IngestError) as ctx:
            ingest(p, self.ws)
        self.assertIn("unrecognised", str(ctx.exception))

    def test_missing_file_is_rejected(self):
        with self.assertRaises(IngestError):
            ingest(self.src / "nope.txt", self.ws)

    def test_ingest_many_survives_a_bad_file(self):
        self.make("good.txt", "fine")
        self.make("bad.zzz", b"\x00\x01")
        atts = ingest_many([self.src / "good.txt", self.src / "bad.zzz"], self.ws)
        self.assertEqual(len(atts), 2)
        self.assertEqual(atts[0].kind, "text")
        self.assertEqual(atts[1].kind, "unsupported")
        self.assertTrue(atts[1].warnings)

    def test_two_files_with_the_same_name_do_not_collide(self):
        a = self.src / "d1" / "report.txt"
        b = self.src / "d2" / "report.txt"
        a.parent.mkdir(parents=True)
        b.parent.mkdir(parents=True)
        a.write_text("first", encoding="utf-8")
        b.write_text("second", encoding="utf-8")
        atts = ingest_many([a, b], self.ws)
        self.assertNotEqual(atts[0].out_rel, atts[1].out_rel)
        self.assertIn("first", (self.ws / atts[0].out_rel).read_text(encoding="utf-8"))
        self.assertIn("second", (self.ws / atts[1].out_rel).read_text(encoding="utf-8"))


class TestConvertedFileIsReachableByTheModel(IngestCase):
    """The whole pipeline is useless if the model's own tools cannot read it."""

    def setUp(self):
        super().setUp()
        self.guard = WorkspaceGuard(self.ws)
        perms = get_permissions("safe")
        self.ctx = ToolContext(guard=self.guard, perms=perms,
                               profile=QWEN35_08B,
                               screener=CommandScreener(self.guard, perms))

    def test_read_file_can_read_the_converted_text(self):
        p = self.make("note.txt", "alpha\nbeta\n")
        att = ingest(p, self.ws)
        result = call_tool("read_file", self.ctx, {"path": att.out_rel})
        self.assertTrue(result.ok, result.text)
        self.assertIn("alpha", result.text)

    def test_view_image_can_reach_a_copied_picture(self):
        from PIL import Image

        p = self.src / "shot.png"
        Image.new("RGB", (240, 160), "blue").save(p)
        att = ingest(p, self.ws)
        result = call_tool("view_image", self.ctx, {"path": att.images[0]})
        self.assertTrue(result.ok, result.text)
        self.assertGreater(result.meta.get("visual_tokens", 0), 0)

    def test_a_rendered_scan_page_can_be_viewed(self):
        mu = _pymupdf()
        if mu is None:
            self.skipTest("PyMuPDF not installed")
        doc = mu.open()
        page = doc.new_page()
        pix = mu.Pixmap(mu.csRGB, mu.IRect(0, 0, 300, 200))
        pix.set_rect(pix.irect, (250, 250, 250))
        page.insert_image(mu.Rect(20, 20, 280, 180), pixmap=pix)
        pdf = self.src / "scan.pdf"
        doc.save(str(pdf))
        doc.close()

        att = ingest(pdf, self.ws)
        self.assertTrue(att.images)
        result = call_tool("view_image", self.ctx, {"path": att.images[0]})
        self.assertTrue(result.ok, result.text)

    def test_attachments_stay_out_of_the_project_listing_but_are_readable(self):
        """Converted files live under .qwen35 so the user's project stays clean."""
        self.make("keep.txt", "x")
        p = self.make("doc.txt", "content")
        att = ingest(p, self.ws)
        self.assertTrue(att.out_rel.startswith(".qwen35/"))
        listing = call_tool("list_dir", self.ctx, {"path": "."})
        self.assertNotIn("attachments", listing.text)
        self.assertTrue(call_tool("read_file", self.ctx,
                                  {"path": att.out_rel}).ok)


@unittest.skipUnless(_have("docx"), "python-docx not installed")
class TestLegacyWord(IngestCase):
    """``.doc`` (Word 97-2003) must be readable without Word or LibreOffice."""

    def _build_doc(self) -> bytes:
        """Make a real OLE2 .doc by driving Word's own structures.

        A tiny writer is enough: header, one FAT sector, a directory, the
        WordDocument stream and a 0Table stream carrying the piece table.
        """
        from qwen35.legacy_word import extract_doc_text

        body = "明辨是非 知荣明辱\r扣好军旅人生第一粒扣子\r\rHello from a legacy file.\r"
        text_bytes = body.encode("utf-16-le")

        # Streams must exceed the 4 096-byte mini-stream cutoff, otherwise they
        # live in the mini stream -- whose reader is deliberately out of scope
        # here.  So the FIB is padded to 4 KiB and the text starts after it.
        text_off = 0x1000
        fib = bytearray(text_off)
        struct.pack_into("<H", fib, 0x00, 0xA5EC)      # wIdent
        struct.pack_into("<H", fib, 0x02, 0x00C1)      # nFib (Word 97)
        struct.pack_into("<H", fib, 0x0A, 0x0000)      # fWhichTblStm = 0Table
        word = bytes(fib) + text_bytes

        # Pcdt: 0x02, lcb, then PlcPcd = (n+1) CPs followed by n 8-byte PCDs.
        plc = struct.pack("<II", 0, len(body)) + struct.pack("<HII", 0, text_off, 0)
        table = b"\x02" + struct.pack("<I", len(plc)) + plc
        struct.pack_into("<II", fib, 0x01A2, 0, len(table))
        word = bytes(fib) + text_bytes

        return _ole2_container({
            "WordDocument": word,
            "0Table": table.ljust(0x1000, b"\x00"),
        })

    def test_a_hand_built_doc_round_trips(self):
        raw = self._build_doc()
        p = self.src / "sample.doc"
        p.write_bytes(raw)
        att = ingest(p, self.ws)
        self.assertEqual(att.kind, "doc")
        self.assertIn("明辨是非", att.text)
        self.assertIn("Hello from a legacy file.", att.text)
        self.assertTrue(any(".doc" in w for w in att.warnings))

    def test_a_broken_doc_gives_a_clean_error(self):
        p = self.make("broken.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 900)
        with self.assertRaises(IngestError) as ctx:
            ingest(p, self.ws)
        self.assertIn("could not be parsed", str(ctx.exception))

    def test_a_file_that_is_really_rtf_is_sniffed(self):
        p = self.make("really.rtf",
                      rb"{\rtf1\ansi\deff0 {\fonttbl{\f0 Times;}}"
                      rb"\f0 Hello \'e4\'b8\'ad\'e6\'96\'87 \u26085?\par Second line.}")
        att = ingest(p, self.ws)
        self.assertIn("Hello", att.text)
        self.assertIn("Second line", att.text)
        self.assertIn("中文", att.text)

    def test_non_word_legacy_formats_still_explain_themselves(self):
        p = self.make("old.xls", b"\xd0\xcf\x11\xe0" + b"\x00" * 64)
        with self.assertRaises(IngestError) as ctx:
            ingest(p, self.ws)
        self.assertIn("xlsx", str(ctx.exception))


class TestContentSniffing(IngestCase):
    """A file whose name says nothing must still be usable.

    Found by a real GUI run: an image handed over as a bare object-store path
    (no extension) reported "no attachment could be converted", even though it
    was an ordinary JPEG.
    """

    def test_the_magic_numbers_are_recognised(self):
        from qwen35.ingest import sniff_kind

        cases = {
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 20: "image",
            b"\xff\xd8\xff\xdb\x00\x43\x00": "image",
            b"GIF89a" + b"\x00" * 10: "image",
            b"%PDF-1.7\n": "pdf",
            b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 8: "legacy-office",
            b"{\\rtf1\\ansi": "doc",
            b"PK\x03\x04" + b"\x00" * 12: "archive",
            b"just some prose here": "",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw[:8]):
                self.assertEqual(sniff_kind(raw), expected)

    def test_an_extensionless_jpeg_is_ingested(self):
        from PIL import Image

        p = self.src / "no_extension_at_all"
        Image.new("RGB", (48, 32), "orange").save(p, format="JPEG")
        att = ingest(p, self.ws)
        self.assertEqual(att.kind, "image")
        self.assertEqual(len(att.images), 1)
        # and it gets a usable extension on the copy
        self.assertTrue(att.images[0].lower().endswith((".jpg", ".jpeg")),
                        att.images[0])
        self.assertTrue((self.ws / att.images[0]).is_file())
        self.assertTrue(any("no usable file extension" in w for w in att.warnings))

    def test_an_extensionless_text_file_is_ingested(self):
        p = self.src / "nameless"
        p.write_text("just some ordinary prose\nsecond line\n", encoding="utf-8")
        att = ingest(p, self.ws)
        self.assertEqual(att.kind, "text")
        self.assertIn("ordinary prose", att.text)

    def test_a_named_file_keeps_its_extension_based_kind(self):
        """Sniffing must never *override* a clear name."""
        att = ingest(self.make("notes.txt", "plain text"), self.ws)
        self.assertEqual(att.kind, "text")


class TestInliningAndNaming(IngestCase):
    """Two design faults behind one real failed run.

    A user attached ``最新慧采采购.xlsx`` (973 tokens) and asked for a summary.  The
    harness pasted nothing, handed over the path ``.qwen35/attachments/
    01-file-4c7efa3e.md``, and the 0.8 B model -- which had quoted that path
    correctly in its own reasoning -- then called ``read_file(path=".")`` and
    burned five steps without producing an answer.

    Both halves were the harness's fault: a character-based inline limit that is
    language-dependent, and a slug function that deleted the Chinese filename.
    """

    def test_a_chinese_filename_survives_into_the_path(self):
        from qwen35.ingest import _slug

        self.assertEqual(_slug("最新慧采采购"), "最新慧采采购")
        self.assertEqual(_slug("季度 report 2026"), "季度-report-2026")
        self.assertEqual(_slug("!!!"), "file")

    def test_a_chinese_attachment_keeps_a_readable_path(self):
        p = self.make("最新慧采采购.csv", "商品,数量\n遥控器,1\n")
        att = ingest(p, self.ws)
        self.assertIn("最新慧采采购", att.out_rel)
        self.assertNotIn("file-", att.out_rel)

    def test_a_small_chinese_document_is_inlined(self):
        """973 tokens out of a 32 K window must not cost a round trip."""
        rows = "\n".join(f"商品名称{i},编号{1000 + i},不在池,{i * 13}"
                         for i in range(40))
        p = self.make("采购.csv", "商品,编号,状态,价格\n" + rows)
        att = ingest(p, self.ws)
        self.assertGreater(att.tokens, 400)
        self.assertLess(att.tokens, 2500)
        self.assertTrue(att.inlined, f"{att.tokens} tokens should have been inlined")
        self.assertIn("商品名称1", compose_task("总结", [att]))

    def test_a_large_document_is_still_left_on_disk(self):
        big = "\n".join(f"line {i} " + "x" * 60 for i in range(600))
        att = ingest(self.make("big.txt", big), self.ws)
        self.assertFalse(att.inlined)
        composed = compose_task("summarise", [att])
        self.assertIn("# Files to read first", composed)
        self.assertIn(att.out_rel, composed)

    def test_the_read_instruction_spells_out_the_exact_call(self):
        big = "\n".join(f"line {i} " + "x" * 60 for i in range(600))
        att = ingest(self.make("big.txt", big), self.ws)
        composed = compose_task("summarise", [att])
        self.assertIn(f'read_file(path="{att.out_rel}"', composed)
        self.assertIn("Copy the path exactly", composed)

    def test_the_inline_decision_uses_tokens_not_characters(self):
        """1 400 CJK characters is ~1 400 tokens; 1 400 English characters is ~380.

        The old character limit therefore treated the two languages completely
        differently for no principled reason.
        """
        cjk = "这是一个中文测试句子。" * 100      # ~1200 chars, ~1200 tokens
        eng = "this is an english test sentence. " * 40   # ~1400 chars, ~350 tokens
        a = ingest(self.make("cjk.txt", cjk), self.ws)
        b = ingest(self.make("eng.txt", eng), self.ws)
        self.assertTrue(b.inlined)
        # both are plausible; the point is the decision follows token count
        self.assertEqual(a.inlined, a.tokens <= 2500)


class TestBinaryRedirect(IngestCase):
    """A model that reaches for the original binary must be sent to the text.

    This was the single most common evaluation failure: the model saw
    ``scores.xlsx`` in the listing and spent eight steps trying to open it, while
    its contents sat unconsumed in the prompt.  The harness now answers that
    attempt with the converted path -- precise feedback at the exact moment of
    failure, which is when a weak model actually recovers.
    """

    def setUp(self):
        super().setUp()
        from qwen35.backends.mock import MockBackend, scripted_answer
        from qwen35.config import HarnessConfig, RuntimeConfig
        from qwen35.loop import Agent

        self.guard = WorkspaceGuard(self.ws)
        perms = get_permissions("safe")
        self.cfg = HarnessConfig(workspace=self.ws, permissions=perms,
                                 runtime=RuntimeConfig(context_window=8192))

    def _agent(self, attachments):
        from qwen35.backends.mock import MockBackend, scripted_answer
        from qwen35.loop import Agent

        return Agent(self.cfg, MockBackend(script=[scripted_answer("ok")]),
                     attachments=attachments)

    def _xlsx(self) -> Path:
        """The attachment is a file *in the project*, which is the case where a
        model actually tries to open the original."""
        try:
            import openpyxl
        except ImportError:
            self.skipTest("openpyxl not installed")
        wb = openpyxl.Workbook()
        wb.active.append(["name", "score"])
        wb.active.append(["bob", 5])
        p = self.ws / "scores.xlsx"
        wb.save(str(p))
        return p

    def test_attachment_map_is_built(self):
        att = ingest(self._xlsx(), self.ws)
        agent = self._agent([att])
        self.assertEqual(agent.ctx.attachment_map.get("scores.xlsx"), att.out_rel)

    def test_reading_the_original_binary_points_at_the_converted_text(self):
        att = ingest(self._xlsx(), self.ws)
        agent = self._agent([att])
        result = call_tool("read_file", agent.ctx, {"path": "scores.xlsx"})
        self.assertFalse(result.ok)
        self.assertIn(att.out_rel, result.text)
        self.assertIn("already been extracted", result.text)

    def test_the_converted_file_is_then_readable(self):
        att = ingest(self._xlsx(), self.ws)
        agent = self._agent([att])
        result = call_tool("read_file", agent.ctx, {"path": att.out_rel})
        self.assertTrue(result.ok, result.text)
        self.assertIn("bob", result.text)

    def test_reading_a_directory_points_at_the_attachment(self):
        """A model that lost the path must be redirected, not sent to a listing
        that skips .qwen35 and dead-ends."""
        att = ingest(self._xlsx(), self.ws)
        agent = self._agent([att])
        result = call_tool("read_file", agent.ctx, {"path": "."})
        self.assertFalse(result.ok)
        self.assertIn("is a directory", result.text)
        self.assertIn(att.out_rel, result.text)
        self.assertIn("read_file(path=", result.text)

    def test_reading_a_directory_without_attachments_still_works(self):
        agent = self._agent([])
        result = call_tool("read_file", agent.ctx, {"path": "."})
        self.assertFalse(result.ok)
        self.assertIn("use list_dir", result.text)

    def test_listing_marks_the_original_as_already_converted(self):
        att = ingest(self._xlsx(), self.ws)
        agent = self._agent([att])
        listing = call_tool("list_dir", agent.ctx, {"path": "."})
        self.assertIn("already converted to text", listing.text)
        self.assertIn(att.out_rel, listing.text)

    def test_without_an_attachment_there_is_no_redirect(self):
        self._xlsx()
        agent = self._agent([])
        result = call_tool("read_file", agent.ctx, {"path": "scores.xlsx"})
        self.assertFalse(result.ok)
        self.assertIn("looks binary", result.text)

    def test_images_are_annotated_for_view_image_not_redirected(self):
        from PIL import Image

        p = self.ws / "shot.png"
        Image.new("RGB", (32, 32), "red").save(p)
        att = ingest(p, self.ws)
        agent = self._agent([att])
        listing = call_tool("list_dir", agent.ctx, {"path": "."})
        self.assertIn("view_image", listing.text)
        self.assertNotIn("already converted to text", listing.text)


# ==========================================================================
# Prompt construction
# ==========================================================================
class TestPreamble(IngestCase):
    def test_small_file_is_inlined(self):
        att = ingest(self.make("small.txt", "tiny content"), self.ws)
        self.assertTrue(att.inlined)
        composed = compose_task("它说了什么？", [att])
        self.assertIn("# Extracted content", composed)
        self.assertIn("tiny content", composed)

    def test_large_file_is_pointed_at_not_inlined(self):
        big = "\n".join(f"line {i}" for i in range(3000))
        att = ingest(self.make("big.txt", big), self.ws)
        self.assertFalse(att.inlined)
        composed = compose_task("summarise", [att])
        self.assertNotIn("# Extracted content", composed)
        self.assertIn(att.out_rel, composed)
        self.assertIn("read_file", composed)

    def test_the_request_comes_before_the_extracted_content(self):
        """Ordering is the fix, not a detail.

        "Lost in the Middle" (arXiv:2307.03172): models attend to the start and
        the end of a long context and degrade in between.  The extracted text used
        to sit in the middle, and the model ignored it -- it spent eight steps
        trying to open the original binary instead.
        """
        att = ingest(self.make("s.txt", "THE ANSWER IS 42"), self.ws)
        composed = compose_task("答案是什么？", [att])
        self.assertLess(composed.index("# Request"), composed.index("THE ANSWER IS 42"))
        self.assertLess(composed.index("# Attached files"),
                        composed.index("# Request"))

    def test_an_inlined_card_never_mentions_a_path_or_a_tool(self):
        """The fix for a real failed run.

        The card used to say "the text is at the end of this message" and then,
        one line later, "to read more of it use read_file(...)".  The model
        obeyed the tool suggestion and ignored the inlined text: eight steps of
        list_dir/read_file and no answer at all.  Naming a tool is an instruction
        to call it.
        """
        att = ingest(self.make("t.csv", "a,b\n1,2\n"), self.ws)
        self.assertTrue(att.inlined)
        card = att.card()
        self.assertNotIn("read_file", card)
        self.assertNotIn(att.out_rel, card)
        self.assertIn("no tool call is needed", card)
        # ...and the content really is in the composed prompt
        composed = compose_task("summarise", [att])
        self.assertIn("| a | b |", composed)

    def test_a_large_card_gives_the_path_and_the_exact_call(self):
        big = "\n".join(f"line {i} " + "x" * 60 for i in range(600))
        att = ingest(self.make("big.txt", big), self.ws)
        self.assertFalse(att.inlined)
        card = att.card()
        self.assertIn(att.out_rel, card)
        self.assertIn("read_file", card)

    def test_a_mislabelled_file_gives_a_clean_error(self):
        """A .xlsx that is not a zip must not leak BadZipFile to the caller."""
        p = self.make("fake.xlsx", "this is not a spreadsheet")
        with self.assertRaises(IngestError) as ctx:
            ingest(p, self.ws)
        self.assertIn("could not be read", str(ctx.exception))

    def test_the_model_is_told_not_to_open_the_original(self):
        att = ingest(self.make("s.txt", "x"), self.ws)
        composed = compose_task("hi", [att])
        self.assertIn("already been converted", composed)
        self.assertIn("do not try to read them directly", composed)

    def test_card_mentions_images_for_a_picture(self):
        from PIL import Image

        p = self.src / "x.png"
        Image.new("RGB", (32, 32), "green").save(p)
        att = ingest(p, self.ws)
        composed = compose_task("what is it", [att])
        self.assertIn("view_image", composed)

    def test_unsupported_files_are_reported_not_hidden(self):
        att = Attachment(source=Path("weird.zzz"), kind="unsupported")
        att.warnings.append("unrecognised format")
        composed = compose_task("read it", [att])
        self.assertIn("weird.zzz", composed)
        self.assertIn("could NOT be read", composed)

    def test_empty_input_gives_the_request_unchanged(self):
        self.assertEqual(compose_task("hello", []), "hello")

    def test_missing_request_gets_a_default(self):
        att = ingest(self.make("s.txt", "x"), self.ws)
        self.assertIn("请阅读这些附件", compose_task("", [att]))

    def test_summary_line_is_informative(self):
        att = ingest(self.make("s.csv", "a,b\n1,2\n"), self.ws)
        line = att.summary_line()
        self.assertIn("s.csv", line)
        self.assertIn("csv", line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
