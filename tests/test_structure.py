"""Headings and heading paths (T2.3): plain text headings found by their wording, the levels each format gives,
and every paragraph and row under the headings above it."""

import html
import json
import re

import pytest

from needtoknow.ingest.decode import load
from needtoknow.ingest.parse import parse
from needtoknow.ingest.structure import heading_level, structure
from needtoknow.settings import repo_root

SAMPLES = sorted((repo_root() / "samples").glob("*.meta.json"))


@pytest.mark.parametrize("meta_path", SAMPLES, ids=lambda p: p.name.split(".")[0])
def test_every_sample_gets_its_title_its_sections_and_a_path_for_everything_under_them(meta_path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    text = load(meta_path.with_name(meta_path.name[: -len(".meta.json")] + "." + meta["format"]), meta)
    elements = structure(parse(text, meta["format"]), meta["format"])
    if meta["format"] == "html":
        sections = [html.unescape(s) for s in re.findall(r"<h2>([^<]+)</h2>", text)]
    else:
        sections = re.findall(r"(?m)^(?:## )?((?:SECTION|ΕΝΟΤΗΤΑ) \d+  .+)$", text)
    headings = [e for e in elements if e.kind == "heading"]
    assert [e.level for e in headings] == [1] + [2] * len(sections)
    assert [e.text for e in headings[1:]] == sections
    first = elements.index(headings[1])
    for e in elements[first:]:
        assert len(e.path) == 2 and e.path[0] == headings[0].text and e.path[1] in sections, e
    if meta["format"] != "html":
        assert all(text[e.start:e.end] == e.text for e in elements)


@pytest.mark.parametrize("line, level", [
    ("SECTION 3  Returns at the counter", 2), ("ΕΝΟΤΗΤΑ 12  Επιστροφές", 2), ("ANNEX B: Stock list", 2),
    ("2 Scope", 2), ("2.1 Who may approve", 3), ("2.1.4 Exceptions", 4), ("RETURNS AT THE COUNTER", 2),
    ("ΕΠΙΣΤΡΟΦΕΣ ΣΤΟ ΚΑΤΑΣΤΗΜΑ", 2), ("ΚΟΙΝΟ: ΑΝΘΡΩΠΙΝΟ ΔΥΝΑΜΙΚΟ", 0), ("CLASSIFICATION: INTERNAL", 0),
    ("1. Check the parcel against its authorisation.", 0), ("Audience: Returns", 0), ("Κοινό: Επιστροφές", 0),
    ("A | B | C", 0), ("SKU", 0), ("30 days after the purchase the advisor checks the receipt and the box", 0),
    ("The returns desk checks each parcel against its authorisation and records the result before moving on", 0),
])
def test_a_plain_text_line_reads_as_a_heading_only_by_its_wording(line, level):
    assert heading_level(line) == level


def test_a_heading_inside_a_plain_text_paragraph_splits_it_with_offsets_kept():
    text = "Handbook: Returns\nDocument RET_MAN_001\n\nSECTION 1  Scope\nAudience: Returns\n\nText one.\n\n2.1 Detail\nText two.\n"
    elements = structure(parse(text, "txt"), "txt")
    assert [(e.kind, e.text, e.path) for e in elements] == [
        ("heading", "Handbook: Returns", ("Handbook: Returns",)),
        ("paragraph", "Document RET_MAN_001", ("Handbook: Returns",)),
        ("heading", "SECTION 1  Scope", ("Handbook: Returns", "SECTION 1  Scope")),
        ("paragraph", "Audience: Returns", ("Handbook: Returns", "SECTION 1  Scope")),
        ("paragraph", "Text one.", ("Handbook: Returns", "SECTION 1  Scope")),
        ("heading", "2.1 Detail", ("Handbook: Returns", "SECTION 1  Scope", "2.1 Detail")),
        ("paragraph", "Text two.", ("Handbook: Returns", "SECTION 1  Scope", "2.1 Detail")),
    ]
    assert all(text[e.start:e.end] == e.text for e in elements)


def test_a_later_heading_of_the_same_level_replaces_the_earlier_one_in_the_path():
    text = "# Title\n\n## One\n\nFirst.\n\n### One point one\n\nDeeper.\n\n## Two\n\nSecond.\n"
    paths = [(e.text, e.path) for e in structure(parse(text, "md"), "md") if e.kind == "paragraph"]
    assert paths == [("First.", ("Title", "One")), ("Deeper.", ("Title", "One", "One point one")), ("Second.", ("Title", "Two"))]


def test_a_numbered_line_is_a_heading_only_at_the_head_of_its_run_and_before_a_new_sentence():
    # DR-F9: wording is a guess, and a wrapped sentence can start with a number and a capital.
    def headings(text):
        return [e.text for e in structure(parse(text.replace("~", "\n"), "txt"), "txt") if e.kind == "heading"]
    assert headings("Title~~2.1 Detail~Text two.~") == ["Title", "2.1 Detail"]
    assert headings("Title~~2.1 Detail~") == ["Title", "2.1 Detail"]
    assert headings("Title~~The contract was signed on~15 Μαΐου από τη Διεύθυνση~and runs to 2027.~") == ["Title"]
    assert headings("Title~~15 Μαΐου από τη Διεύθυνση~ενέκρινε τη νέα πολιτική.~") == ["Title"]
    assert headings("Title~~Text one.~SECTION 2  Pay~Text two.~") == ["Title", "SECTION 2  Pay"]
