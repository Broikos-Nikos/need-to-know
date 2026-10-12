"""Decoding (T2.2): the declared encoding applied strictly, line ends and NFC made canonical, and the two signs of
a wrong encoding refused; the legacy file is written strictly too (DR-F4)."""

import json
import unicodedata

import pytest

from needtoknow.corpus import assemble
from needtoknow.corpus.plan import header
from needtoknow.ingest.decode import DecodeError, decode, load
from needtoknow.settings import repo_root

GREEK = "Ο πελάτης επιστρέφει τη συσκευή εντός 14 ημερών."


def test_each_declared_encoding_is_read_as_declared():
    assert decode(GREEK.encode("utf-8"), "utf-8") == GREEK
    assert decode(b"\xef\xbb\xbf" + GREEK.encode("utf-8"), "utf-8-sig") == GREEK
    assert decode(GREEK.encode("cp1253"), "windows-1253") == GREEK


def test_line_ends_and_accents_come_out_in_one_canonical_form():
    decomposed = unicodedata.normalize("NFD", GREEK)
    assert decomposed != GREEK
    assert decode(("a\r\nb\rc\n" + decomposed).encode("utf-8"), "utf-8") == "a\nb\nc\n" + GREEK


@pytest.mark.parametrize("data, encoding, says", [
    (b"caf\xe9", "utf-8", "byte 3 is not utf-8"),
    (GREEK.encode("utf-8"), "latin-1", "declares 'latin-1'"),
    ("a lossy \ufffd here".encode("utf-8"), "utf-8", "replacement character at 8"),
    ("Привет".encode("utf-8"), "utf-8", "Cyrillic letter 'П' at 0"),
    # Greek written in Windows-1253 and read in Windows-1251 turns into Cyrillic; declared as utf-8 it fails
    # outright, and the Cyrillic guard catches the case where the wrong code page was applied upstream.
    (GREEK.encode("cp1253").decode("cp1251").encode("utf-8"), "utf-8", "Cyrillic letter"),
])
def test_a_wrong_encoding_is_refused_with_its_place(data, encoding, says):
    with pytest.raises(DecodeError, match=says):
        decode(data, encoding, "the file")


def test_every_sample_decodes_and_the_legacy_one_reads_as_greek():
    for meta_path in sorted((repo_root() / "samples").glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        text = load(meta_path.with_name(meta_path.name[: -len(".meta.json")] + "." + meta["format"]), meta)
        assert text and "\r" not in text and text == unicodedata.normalize("NFC", text)
        if meta["encoding"] == "windows-1253":
            assert "ΕΝΟΤΗΤΑ 1" in text


def test_the_legacy_file_refuses_a_character_its_code_page_lacks(monkeypatch, tmp_path):
    # DR-F4: written with errors="replace", the character would have become "?" without a word.
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    base = tmp_path / "corpus"
    (base / "plan").mkdir(parents=True)
    (base / "narrative").mkdir()
    doc = {"id": "RET_MAN_001", "type": "MAN", "type_name": "Εγχειρίδιο", "owner": "returns", "language": "el", "format": "txt",
           "target_mb": 0.01, "version": "1.0",
           "sections": [{"label": "returns", "topic_en": "Scope", "topic_el": "Πεδίο", "facts": [], "canary": None}]}
    text = "\n".join(header(doc, 1, doc["sections"][0]) + ["", GREEK + " Ένα σύμβολο που λείπει: ≠.", ""])
    (base / "narrative" / "RET_MAN_001.txt").write_bytes(text.encode("utf-8"))
    (base / "plan" / "catalogue.json").write_bytes(json.dumps([doc], ensure_ascii=False).encode("utf-8"))
    (base / "plan" / "facts.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="RET_MAN_001: '≠' at \\d+ has no place in windows-1253"):
        assemble.assemble()
    assert not (base / "full").exists()
