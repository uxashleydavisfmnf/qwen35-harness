"""
Legacy Word (``.doc``, Word 97-2003) text extraction, in pure Python.

Why this exists: ``.doc`` is still everywhere in Chinese offices, and a harness
that answers "save it as .docx first" is not usable for the documents people
actually have.  A real test file (a 33 KB military-education handout) was refused
outright before this module was written.

The format is two layers:

1. **OLE2 / Compound File Binary** -- a small filesystem in one file
   (:class:`Ole2File`).  We only need to read named streams.
2. **The Word FIB + piece table** -- ``WordDocument`` holds the header and the
   character data; the piece table in the ``1Table``/``0Table`` stream says where
   each run of text lives and whether it is 8-bit or UTF-16.

Nothing here needs Word, LibreOffice or a network connection.

References: [MS-CFB] (compound file), [MS-DOC] §2.3.1 (FIB), §2.9.1 (Clx),
§2.9.3 (PlcPcd).  Character positions are "CP"s; the piece table maps CP ranges to
file offsets, which is why the text is not simply a contiguous block.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterator

__all__ = ["Ole2File", "extract_doc_text", "DocParseError"]


class DocParseError(Exception):
    """The container or the Word structures could not be understood."""


# --------------------------------------------------------------------------
# OLE2 / Compound File Binary
# --------------------------------------------------------------------------
_FREESECT = 0xFFFFFFFF
_ENDOFCHAIN = 0xFFFFFFFE
_FATSECT = 0xFFFFFFFD
_DIFSECT = 0xFFFFFFFC
_NOSTREAM = 0xFFFFFFFF

_DIR_ENTRY_SIZE = 128
_TYPE_STORAGE = 1
_TYPE_STREAM = 2
_TYPE_ROOT = 5


class Ole2File:
    """Minimal reader for the parts of [MS-CFB] a ``.doc`` needs."""

    def __init__(self, data: bytes) -> None:
        if len(data) < 512 or data[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            raise DocParseError("not an OLE2 compound file")
        self.data = data
        self.sector_size = 1 << struct.unpack_from("<H", data, 0x1E)[0]
        self.mini_sector_size = 1 << struct.unpack_from("<H", data, 0x20)[0]
        if self.sector_size not in (512, 4096):
            raise DocParseError(f"implausible sector size {self.sector_size}")

        self.num_fat_sectors = struct.unpack_from("<I", data, 0x2C)[0]
        self.first_dir_sector = struct.unpack_from("<I", data, 0x30)[0]
        self.mini_cutoff = struct.unpack_from("<I", data, 0x38)[0]
        self.first_mini_fat = struct.unpack_from("<I", data, 0x3C)[0]
        self.num_mini_fat = struct.unpack_from("<I", data, 0x40)[0]
        self.first_difat = struct.unpack_from("<I", data, 0x44)[0]
        self.num_difat = struct.unpack_from("<I", data, 0x48)[0]

        self._fat = self._read_fat()
        self._minifat = self._read_minifat()
        self._entries = self._read_directory()
        self._mini_stream = self._read_mini_stream()

    # -- sector plumbing ---------------------------------------------------
    def _sector(self, index: int) -> bytes:
        if index in (_FREESECT, _ENDOFCHAIN):
            return b""
        start = 512 + index * self.sector_size
        end = start + self.sector_size
        if start >= len(self.data):
            return b""
        return self.data[start:min(end, len(self.data))]

    def _chain(self, start: int, fat: list[int]) -> Iterator[int]:
        """Walk a sector chain, with a hard cap so a corrupt file cannot hang."""
        seen: set[int] = set()
        current = start
        for _ in range(len(fat) + 8):
            if current in (_FREESECT, _ENDOFCHAIN, _NOSTREAM) or current >= len(fat):
                return
            if current in seen:            # a cycle would loop forever
                return
            seen.add(current)
            yield current
            current = fat[current]

    def _read_fat(self) -> list[int]:
        entries: list[int] = []
        # The first 109 FAT sector numbers live in the header's DIFAT.
        difat = list(struct.unpack_from("<109I", self.data, 0x4C))
        # ...and any overflow continues in DIFAT sectors.
        current = self.first_difat
        for _ in range(self.num_difat + 1):
            if current in (_FREESECT, _ENDOFCHAIN) or current >= 0xFFFFFFF0:
                break
            block = self._sector(current)
            if len(block) < self.sector_size:
                break
            count = self.sector_size // 4 - 1
            difat.extend(struct.unpack_from(f"<{count}I", block, 0))
            current = struct.unpack_from("<I", block, self.sector_size - 4)[0]

        for sector in difat[: self.num_fat_sectors]:
            if sector in (_FREESECT, _ENDOFCHAIN):
                continue
            block = self._sector(sector)
            if len(block) < self.sector_size:
                break
            entries.extend(struct.unpack_from(f"<{self.sector_size // 4}I", block, 0))
        if not entries:
            raise DocParseError("the FAT could not be read")
        return entries

    def _read_minifat(self) -> list[int]:
        out: list[int] = []
        for sector in self._chain(self.first_mini_fat, self._fat):
            block = self._sector(sector)
            out.extend(struct.unpack_from(f"<{len(block) // 4}I", block, 0))
        return out

    def _read_directory(self) -> list[dict]:
        entries: list[dict] = []
        for sector in self._chain(self.first_dir_sector, self._fat):
            block = self._sector(sector)
            for offset in range(0, len(block) - _DIR_ENTRY_SIZE + 1,
                                _DIR_ENTRY_SIZE):
                raw = block[offset:offset + _DIR_ENTRY_SIZE]
                name_len = struct.unpack_from("<H", raw, 0x40)[0]
                if name_len < 2 or name_len > 64:
                    continue
                name = raw[:name_len - 2].decode("utf-16-le", "replace")
                entries.append({
                    "name": name,
                    "type": raw[0x42],
                    "start": struct.unpack_from("<I", raw, 0x74)[0],
                    "size": struct.unpack_from("<Q", raw, 0x78)[0],
                })
        if not entries:
            raise DocParseError("the directory is empty")
        return entries

    def _read_mini_stream(self) -> bytes:
        """The mini stream holds every object smaller than ``mini_cutoff``."""
        root = next((e for e in self._entries if e["type"] == _TYPE_ROOT), None)
        if root is None:
            return b""
        chunks = [self._sector(s) for s in self._chain(root["start"], self._fat)]
        return b"".join(chunks)[:root["size"]]

    # -- public ------------------------------------------------------------
    def names(self) -> list[str]:
        return [e["name"] for e in self._entries if e["type"] == _TYPE_STREAM]

    def stream(self, name: str) -> bytes:
        entry = next((e for e in self._entries
                      if e["type"] == _TYPE_STREAM and e["name"] == name), None)
        if entry is None:
            raise DocParseError(f"no stream named {name!r}")
        size = entry["size"]
        if size < self.mini_cutoff:
            # small: read it out of the mini stream, in mini sectors
            out = bytearray()
            for index in self._chain(entry["start"], self._minifat):
                start = index * self.mini_sector_size
                out += self._mini_stream[start:start + self.mini_sector_size]
                if len(out) >= size:
                    break
            return bytes(out[:size])
        chunks = [self._sector(s) for s in self._chain(entry["start"], self._fat)]
        return b"".join(chunks)[:size]


# --------------------------------------------------------------------------
# Word 97-2003 text
# --------------------------------------------------------------------------
def _piece_text(word: bytes, table: bytes) -> str:
    """Read the text via the piece table ([MS-DOC] §2.9.1)."""
    if len(word) < 0x01AA:
        raise DocParseError("the FIB is truncated")
    # FibRgFcLcb97.fcClx / lcbClx
    fc_clx, lcb_clx = struct.unpack_from("<II", word, 0x01A2)
    if not lcb_clx or fc_clx + lcb_clx > len(table):
        raise DocParseError("no usable piece table")

    clx = table[fc_clx:fc_clx + lcb_clx]
    # Skip any Prc entries (0x01) to reach the Pcdt (0x02).
    pos = 0
    while pos < len(clx) and clx[pos] == 0x01:
        if pos + 3 > len(clx):
            break
        cb_grpprl = struct.unpack_from("<H", clx, pos + 1)[0]
        pos += 3 + cb_grpprl
    if pos >= len(clx) or clx[pos] != 0x02:
        raise DocParseError("the piece table marker is missing")
    lcb = struct.unpack_from("<I", clx, pos + 1)[0]
    plc = clx[pos + 5:pos + 5 + lcb]
    if len(plc) < 4:
        raise DocParseError("the piece table is empty")

    # PlcPcd = (n+1) CPs, then n PCDs of 8 bytes.
    n = (len(plc) - 4) // 12
    if n <= 0:
        raise DocParseError("the piece table has no pieces")
    cps = struct.unpack_from(f"<{n + 1}I", plc, 0)
    pieces: list[str] = []
    for i in range(n):
        pcd = plc[4 * (n + 1) + 8 * i: 4 * (n + 1) + 8 * i + 8]
        fc = struct.unpack_from("<I", pcd, 2)[0]
        length = cps[i + 1] - cps[i]
        if length <= 0 or length > 8_000_000:
            continue
        if fc & 0x40000000:
            # "compressed": one byte per character, in the code page
            offset = (fc & ~0x40000000) // 2
            raw = word[offset:offset + length]
            pieces.append(raw.decode("cp1252", "replace"))
        else:
            offset = fc
            raw = word[offset:offset + length * 2]
            pieces.append(raw.decode("utf-16-le", "replace"))
    return "".join(pieces)


def _clean(text: str) -> str:
    """Turn Word's control characters into readable text.

    Field codes get special care: a Word field is ``\\x13 <instruction> \\x14
    <result> \\x15``, so the *result* is worth keeping and the *instruction* is
    not.  Without this the tail of a real document read
    ``— PAGE  \\* MERGEFORMAT 1—``.
    """
    out: list[str] = []
    in_instruction = False
    for ch in text:
        code = ord(ch)
        if ch == "\x13":            # field begin
            in_instruction = True
            continue
        if ch == "\x14":            # instruction ends, the result starts
            in_instruction = False
            continue
        if ch == "\x15":            # field end
            in_instruction = False
            continue
        if in_instruction:
            continue
        if ch in "\r\n":
            out.append("\n")
        elif ch == "\x07":          # cell / row mark
            out.append("\t")
        elif ch in ("\x0c", "\x0b", "\x0e", "\x0f"):
            out.append("\n")
        elif ch == "\x08" or ch == "\x01" or ch == "\x02" or ch == "\x05":
            # footnote / picture / drawing anchors
            continue
        elif code < 32 and ch not in "\t\n":
            out.append(" ")
        elif code == 0xFFFD:
            continue
        else:
            out.append(ch)
    text = "".join(out)
    # collapse the runs of blank lines Word leaves behind
    lines = [ln.rstrip() for ln in text.split("\n")]
    cleaned: list[str] = []
    for line in lines:
        if line.strip() or (cleaned and cleaned[-1].strip()):
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def extract_doc_text(data: bytes) -> str:
    """Extract the body text of a Word 97-2003 ``.doc`` file."""
    ole = Ole2File(data)
    try:
        word = ole.stream("WordDocument")
    except DocParseError as exc:
        raise DocParseError("this .doc has no WordDocument stream") from exc

    # FibBase.fWhichTblStm (bit 9 of the flags at 0x0A) picks the table stream.
    flags = struct.unpack_from("<H", word, 0x0A)[0] if len(word) >= 0x0C else 0
    table_name = "1Table" if flags & 0x0200 else "0Table"
    try:
        table = ole.stream(table_name)
    except DocParseError:
        try:
            table = ole.stream("1Table" if table_name == "0Table" else "0Table")
        except DocParseError as exc:
            raise DocParseError("this .doc has no table stream") from exc

    try:
        text = _piece_text(word, table)
    except DocParseError:
        # Very old (Word 6/95) files have no piece table: the text is simply the
        # contiguous range fcMin..fcMac from the FIB.
        fc_min, fc_mac = struct.unpack_from("<II", word, 0x18)
        if not (0 < fc_min < fc_mac <= len(word)):
            raise
        text = word[fc_min:fc_mac].decode("cp1252", "replace")

    cleaned = _clean(text)
    if not cleaned.strip():
        raise DocParseError("no text could be extracted from this .doc")
    return cleaned
