"""Each format's text into blocks, every block with the character offsets it came from (T2.1).

A block is a heading, a paragraph or a table row. Its text is what a reader sees:
markdown's markers are dropped and html is unescaped. Its start and end index the
decoded text, so `resolve` gives the block back from its span: exactly in txt and
markdown, and with tags dropped and entities unescaped in html. Headings found by
their wording rather than their markup, "SECTION 3" or a numbered line, are left to
the heading step (T2.3); audience lines to the segment step (T2.4).

The parsers expect "\\n" line ends, which decoding produces (T2.2), so an offset is a
plain index into one string. Html text outside every block, in a div or straight in a
section, would never reach the index, so a file holding any is refused (DR-F14).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser

PARSER_VERSION = 1
CELL = " | "


@dataclass(frozen=True, slots=True)
class Block:
    kind: str                       # "heading", "paragraph" or "row"
    text: str
    start: int
    end: int
    level: int = 0                  # a heading's level, where the markup gives one
    cells: tuple[str, ...] = ()     # a row's cells
    header: bool = False            # a table's header row
    audience: str | None = None     # html: the data-audience of the enclosing section


def parse(text: str, fmt: str) -> list[Block]:
    if "\r" in text:
        raise ValueError("carriage returns in the text: decode and normalise line ends first (T2.2)")
    if fmt == "txt":
        return parse_txt(text)
    if fmt == "md":
        return parse_md(text)
    if fmt == "html":
        return parse_html(text)
    raise ValueError(f"no parser for {fmt!r}")


def runs(text: str) -> list[tuple[int, int]]:
    """Spans of consecutive lines that are not blank, without the final line end."""
    return [(m.start(), m.end()) for m in re.finditer(r"[^\n]*\S[^\n]*(?:\n[^\n]*\S[^\n]*)*", text)]


def parse_txt(text: str) -> list[Block]:
    """Paragraphs are runs of lines between blank lines. A run whose every line holds " | " is a table: one row
    per line, the first its header, as a plain text table writes its column names first."""
    blocks = []
    for start, end in runs(text):
        lines = text[start:end].split("\n")
        if len(lines) > 1 and all(CELL in line for line in lines):
            at = start
            for k, line in enumerate(lines):
                blocks.append(Block("row", line, at, at + len(line), cells=tuple(line.split(CELL)), header=k == 0))
                at += len(line) + 1
        else:
            blocks.append(Block("paragraph", text[start:end], start, end))
    return blocks


SEPARATOR = re.compile(r"\|(?:\s*:?-+:?\s*\|)+")


def parse_md(text: str) -> list[Block]:
    """ATX headings, pipe tables (the row before the separator line is the header) and paragraphs."""
    blocks = []
    for start, end in runs(text):
        lines, at, paragraph = text[start:end].split("\n"), start, None
        for k, line in enumerate(lines):
            heading = re.match(r"(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$", line)
            is_row = line.startswith("|") and line.rstrip().endswith("|")
            if (heading or is_row) and paragraph is not None:
                blocks.append(Block("paragraph", text[paragraph:at - 1], paragraph, at - 1))
                paragraph = None
            if heading:
                inner = at + heading.start(2)
                blocks.append(Block("heading", heading.group(2), inner, inner + len(heading.group(2)), level=len(heading.group(1))))
            elif is_row:
                if not SEPARATOR.fullmatch(line.strip()):
                    body = re.match(r"\|[ \t]?(.*?)[ \t]?\|[ \t]*$", line)
                    inner = at + body.start(1)
                    following = lines[k + 1] if k + 1 < len(lines) else ""
                    blocks.append(Block("row", body.group(1), inner, inner + len(body.group(1)),
                                        cells=tuple(c.strip() for c in body.group(1).split("|")),
                                        header=bool(SEPARATOR.fullmatch(following.strip()))))
            elif paragraph is None:
                paragraph = at
            at += len(line) + 1
        if paragraph is not None:
            blocks.append(Block("paragraph", text[paragraph:end], paragraph, end))
    return blocks


class _Html(HTMLParser):
    BLOCKS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr"}
    UNSEEN = {"title", "script", "style"}  # text no reader of the page sees as body text

    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.text, self.blocks, self.audiences = text, [], []
        self.lines = [0] + [m.end() for m in re.finditer("\n", text)]
        self.open = None  # (tag, start of content, pieces, cells, header)
        self.unseen = 0

    def at(self) -> int:
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def handle_starttag(self, tag, attrs):
        self.unseen += tag in self.UNSEEN
        if tag == "section":
            self.audiences.append(dict(attrs).get("data-audience"))
        elif tag in self.BLOCKS and self.open is None:
            self.open = (tag, self.at() + len(self.get_starttag_text()), [], [], False)
        elif tag in ("td", "th") and self.open and self.open[0] == "tr":
            self.open[3].append([])
            if tag == "th":
                self.open = self.open[:4] + (True,)

    def handle_data(self, data):
        if self.open:
            self.open[2].append(data)
            if self.open[0] == "tr" and self.open[3]:
                self.open[3][-1].append(data)
        elif data.strip() and not self.unseen:
            raise ValueError(f"text outside every block at {self.at()}: {data.strip()[:60]!r}; the html parser reads "
                             f"h1 to h6, p, li and tr")

    def handle_endtag(self, tag):
        self.unseen -= tag in self.UNSEEN and self.unseen > 0
        if tag == "section" and self.audiences:
            self.audiences.pop()
        elif self.open and tag == self.open[0]:
            kind, start, pieces, cells, header = self.open
            end, audience = self.at(), (self.audiences[-1] if self.audiences else None)
            if kind == "tr":
                row = tuple("".join(c).strip() for c in cells)
                self.blocks.append(Block("row", CELL.join(row), start, end, cells=row, header=header, audience=audience))
            else:
                text = "".join(pieces).strip()
                if text:
                    level = int(kind[1]) if kind[0] == "h" else 0
                    self.blocks.append(Block("heading" if level else "paragraph", text, start, end, level=level, audience=audience))
            self.open = None


def parse_html(text: str) -> list[Block]:
    """Headings h1 to h6, paragraphs, list items and table rows, each with the data-audience of its section."""
    parser = _Html(text)
    parser.feed(text)
    parser.close()
    return parser.blocks


def resolve(text: str, fmt: str, block: Block) -> str:
    """The block's text as its span in the source gives it back."""
    span = text[block.start:block.end]
    if fmt == "html":
        if block.kind == "row":
            return CELL.join(html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", span, re.S))
        return html.unescape(re.sub(r"<[^>]+>", "", span)).strip()
    return span
