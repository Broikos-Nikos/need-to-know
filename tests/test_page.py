"""The instrument page (T6.2): its fonts draw every character the narrative and the golden questions use, so no Greek
falls back to another face, the house style's fourth rule for an instrument; and each ships with its licence and keeps
its copyright and licence records."""

import pytest
from fontTools.ttLib import TTFont

from needtoknow.settings import repo_root

FONTS = repo_root() / "web" / "public" / "fonts"


def corpus_characters() -> set[str]:
    root = repo_root()
    texts = [path.read_text(encoding="utf-8") for path in sorted((root / "narrative").glob("*.txt"))]
    texts += [path.read_text(encoding="utf-8") for path in sorted((root / "eval" / "golden").glob("*.jsonl"))]
    return {character for text in texts for character in text if ord(character) > 32}


@pytest.mark.parametrize("font", ["manrope-var.woff2", "robotomono-var.woff2"])
def test_the_page_fonts_draw_every_character_the_corpus_uses(font):
    characters = corpus_characters()
    assert any(0x370 <= ord(character) <= 0x3FF for character in characters)  # Greek is there to draw
    drawn = TTFont(FONTS / font).getBestCmap()
    missing = sorted(character for character in characters if ord(character) not in drawn)
    assert not missing, (font, missing[:20])


def test_each_font_ships_with_its_licence_and_keeps_its_name_records():
    for font, licence in (("manrope-var.woff2", "Manrope-OFL.txt"), ("robotomono-var.woff2", "RobotoMono-LICENSE.txt")):
        assert (FONTS / licence).is_file(), licence
        assert {0, 13} <= {record.nameID for record in TTFont(FONTS / font)["name"].names}, font
