"""A file's bytes into the text every later step indexes into (T2.2).

The declared encoding comes from the sidecar and is applied strictly, so a byte it
cannot read stops the file rather than turning into a guess. Line ends become "\\n"
and the text is put in NFC, so offsets from the parsers are offsets into one
canonical string. Two signs of a file decoded the wrong way are refused outright:
U+FFFD, which a lossy conversion leaves behind, and Cyrillic letters, which Greek
read in a Cyrillic code page turns into.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

CODECS = {"utf-8": "utf-8", "utf-8-sig": "utf-8-sig", "windows-1253": "cp1253"}
SUSPECT = re.compile("[\ufffd\u0400-\u04ff\u0500-\u052f]")


class DecodeError(ValueError):
    """A file that cannot be read as its sidecar declares, or reads as text no writer meant."""


def decode(data: bytes, encoding: str, name: str = "the file") -> str:
    codec = CODECS.get(encoding)
    if codec is None:
        raise DecodeError(f"{name}: the sidecar declares {encoding!r}, which is none of {', '.join(CODECS)}")
    try:
        text = data.decode(codec)
    except UnicodeDecodeError as error:
        raise DecodeError(f"{name}: byte {error.start} is not {encoding}: {data[error.start:error.start + 4]!r}") from None
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    bad = SUSPECT.search(text)
    if bad:
        what = "a replacement character" if bad.group() == "\ufffd" else f"the Cyrillic letter {bad.group()!r}"
        raise DecodeError(f"{name}: {what} at {bad.start()}, a sign of a wrong encoding upstream")
    return text


def load(file: Path, meta: dict) -> str:
    """A corpus file as its sidecar declares it."""
    return decode(file.read_bytes(), meta["encoding"], meta.get("doc_id", file.name))
