"""The instrument page (T6.2): its fonts draw every character the narrative, the golden questions and the page's own
words use, so no Greek falls back to another face, the house style's fourth rule for an instrument; each ships with its
licence and keeps its copyright and licence records; and the page writes what the server sends, the documents' text
and the model's answer among it, as text, never through the HTML parser."""

import re

import pytest
from fontTools.ttLib import TTFont

from needtoknow.settings import repo_root

FONTS = repo_root() / "web" / "public" / "fonts"
# Every way a script hands a string to the HTML parser, or writes a page of its own.
MARKUP = re.compile(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|createContextualFragment|DOMParser"
                    r"|srcdoc|setHTMLUnsafe|parseHTMLUnsafe")


def page_sources() -> list:
    web = repo_root() / "web"
    return [web / "index.html", *sorted(path for path in (web / "src").rglob("*") if path.is_file())]


def shown_characters() -> set[str]:
    root = repo_root()
    texts = [path.read_text(encoding="utf-8") for path in sorted((root / "narrative").glob("*.txt"))]
    texts += [path.read_text(encoding="utf-8") for path in sorted((root / "eval" / "golden").glob("*.jsonl"))]
    texts += [path.read_text(encoding="utf-8") for path in page_sources()]
    return {character for text in texts for character in text if ord(character) > 32}


@pytest.mark.parametrize("font", ["manrope-var.woff2", "robotomono-var.woff2"])
def test_the_page_fonts_draw_every_character_the_corpus_and_the_page_use(font):
    characters = shown_characters()
    assert any(0x370 <= ord(character) <= 0x3FF for character in characters)  # Greek is there to draw
    drawn = TTFont(FONTS / font).getBestCmap()
    missing = sorted(character for character in characters if ord(character) not in drawn)
    assert not missing, (font, missing[:20])


def test_each_font_ships_with_its_licence_and_keeps_its_name_records():
    for font, licence in (("manrope-var.woff2", "Manrope-OFL.txt"), ("robotomono-var.woff2", "RobotoMono-LICENSE.txt")):
        assert (FONTS / licence).is_file(), licence
        assert {0, 13} <= {record.nameID for record in TTFont(FONTS / font)["name"].names}, font


def test_nothing_moves_or_waits_to_appear_when_the_reader_asks_for_less_motion():
    # T6.2b: slow motion's rows arrive with a movement and a delay each; with reduced motion set, the stages still take
    # their turns, and within one nothing moves and nothing is held back.
    style = (repo_root() / "web" / "src" / "style.css").read_text(encoding="utf-8")
    assert re.search(r"animation:|transition:", style)
    block = re.search(r"@media \(prefers-reduced-motion: reduce\) \{(.*?)\n\}", style, re.S)
    assert block, "no reduced motion block"
    for held in ("animation-duration: 0.01ms", "animation-delay: 0s", "transition-duration: 0.01ms",
                 "transition-delay: 0s"):
        assert f"{held} !important" in block.group(1), held


def test_the_page_writes_text_and_never_markup():
    # T6.2b: an excerpt is a document's text and an answer is a model's, and a section the role may read can hold
    # markup; the page writes both with textContent, so a tag in them shows as the characters it is made of.
    sources = [path for path in page_sources() if path.suffix in {".ts", ".js", ".html"}]
    assert any(path.suffix == ".ts" for path in sources)
    found = [f"{path.name}:{number}" for path in sources
             for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if MARKUP.search(line)]
    assert not found, found
