"""The parsers (T2.1): every block resolves back to its span, nothing visible is lost or invented, and tables,
headings and audiences come out in the shape each format writes them."""

import html
import json
import re

import pytest

from needtoknow.corpus.mutate import read
from needtoknow.ingest.parse import Block, parse, resolve
from needtoknow.settings import repo_root

SAMPLES = sorted((repo_root() / "samples").glob("*.meta.json"))


def sample(meta_path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    file = meta_path.with_name(meta_path.name[: -len(".meta.json")] + "." + meta["format"])
    return read(file, meta), meta


def visible_words(text: str, fmt: str) -> list[str]:
    """The words a reader sees in the source, markup taken away."""
    if fmt == "md":
        lines = [line for line in text.split("\n") if not re.fullmatch(r"\|(?:\s*:?-+:?\s*\|)+", line.strip())]
        text = "\n".join(re.sub(r"^#{1,6}[ \t]+", "", line) for line in lines).replace("|", " ")
    elif fmt == "html":
        body = text[text.index("<body>"):]
        text = html.unescape(re.sub(r"<[^>]+>", " ", body))
    else:
        text = text.replace(" | ", " ")
    return text.split()


@pytest.mark.parametrize("meta_path", SAMPLES, ids=lambda p: p.name.split(".")[0])
def test_every_block_resolves_and_nothing_visible_is_lost(meta_path):
    text, meta = sample(meta_path)
    fmt = meta["format"]
    blocks = parse(text, fmt)
    assert blocks
    for block in blocks:
        assert resolve(text, fmt, block) == block.text, block
    assert all(a.end <= b.start for a, b in zip(blocks, blocks[1:]))
    words = [w for b in blocks for w in (b.text.replace(" | ", " ").split())]
    assert words == visible_words(text, fmt)


@pytest.mark.parametrize("meta_path", SAMPLES, ids=lambda p: p.name.split(".")[0])
def test_every_table_has_one_header_row_and_rows_as_wide_as_it(meta_path):
    text, meta = sample(meta_path)
    blocks = parse(text, meta["format"])
    tables, width = 0, None
    for block in blocks:
        if block.kind != "row":
            width = None
            continue
        if block.header:
            tables, width = tables + 1, len(block.cells)
        assert width is not None and len(block.cells) == width, block
    # Each sample carries its register, the directory, forms, a glossary and its revision history.
    assert tables == 5


@pytest.mark.parametrize("meta_path", SAMPLES, ids=lambda p: p.name.split(".")[0])
def test_markup_headings_and_html_audiences(meta_path):
    text, meta = sample(meta_path)
    blocks = parse(text, meta["format"])
    sections = len(re.findall(r"(?m)^(?:## )?(?:SECTION|ΕΝΟΤΗΤΑ) \d+  ", text)) or text.count("<section ")
    headings = [b for b in blocks if b.kind == "heading"]
    if meta["format"] == "txt":
        assert headings == []  # a plain text heading is found by its wording, at T2.3
    else:
        assert [b.level for b in headings] == [1] + [2] * sections
    if meta["format"] == "html":
        audiences = re.findall(r'<section data-audience="([^"]*)">', text)
        inside = [b for b in blocks if b.audience is not None]
        assert inside and {html.unescape(a) for a in audiences} == {b.audience for b in inside}


def test_html_entities_and_inline_tags_are_read_as_a_reader_sees_them():
    text = '<body><section data-audience="Legal &amp; Finance"><h2>Fees &amp; terms</h2><p>Pay <b>within</b> the &lt;limit&gt;.</p></section></body>'
    blocks = parse(text, "html")
    assert [(b.kind, b.text, b.audience) for b in blocks] == [
        ("heading", "Fees & terms", "Legal & Finance"), ("paragraph", "Pay within the <limit>.", "Legal & Finance")]
    assert all(resolve(text, "html", b) == b.text for b in blocks)


def test_markdown_tables_headings_and_paragraphs_split_where_the_markup_says():
    text = "# Title #\nDocument line\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\nAfter the table\n"
    blocks = parse(text, "md")
    assert [(b.kind, b.text, b.header) for b in blocks] == [
        ("heading", "Title", False), ("paragraph", "Document line", False), ("row", "A | B", True),
        ("row", "1 | 2", False), ("row", "3 | 4", False), ("paragraph", "After the table", False)]
    assert all(text[b.start:b.end] == b.text for b in blocks)


def test_an_empty_text_and_a_carriage_return_and_an_unknown_format():
    assert parse("", "txt") == [] and parse("\n\n", "md") == [] and parse("<body></body>", "html") == []
    with pytest.raises(ValueError, match="line ends"):
        parse("one\r\ntwo", "txt")
    with pytest.raises(ValueError, match="no parser"):
        parse("text", "docx")
    assert parse("a | b\nc | d", "txt") == [Block("row", "a | b", 0, 5, cells=("a", "b"), header=True),
                                           Block("row", "c | d", 6, 11, cells=("c", "d"))]
