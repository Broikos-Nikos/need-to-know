"""Access segments (T2.4): every element takes its section's audience, default deny for names no label knows,
and a coverage report by unit against the registered shares (KB3-K5, RS-F10, MA-F4)."""

import json
import re
from collections import Counter

import pytest

from needtoknow.cli import main
from needtoknow.corpus.plan import header
from needtoknow.ingest.decode import load
from needtoknow.ingest.parse import parse
from needtoknow.ingest.segments import Coverage, labels_named, segments
from needtoknow.ingest.structure import Element, structure
from needtoknow.policy import UNLABELLED
from needtoknow.settings import repo_root

SAMPLES = sorted((repo_root() / "samples").glob("*.meta.json"))


def segmented(text: str, fmt: str, meta: dict, unknown=None):
    return segments(structure(parse(text, fmt), fmt), meta, unknown)


@pytest.mark.parametrize("meta_path", SAMPLES, ids=lambda p: p.name.split(".")[0])
def test_every_element_of_a_sample_takes_its_sections_audience(meta_path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    text = load(meta_path.with_name(meta_path.name[: -len(".meta.json")] + "." + meta["format"]), meta)
    unknown = []
    elements = segmented(text, meta["format"], meta, unknown)
    assert unknown == []
    named = [labels_named(m, []) for m in re.findall(r"(?:Audience|Κοινό): ([^<\n]+)", text)]
    sections = [e for e in elements if e.kind == "heading" and e.level == 2]
    assert [e.labels for e in sections] == named
    section = None
    for e in elements:
        if e.kind == "heading" and e.level == 2:
            section = sections.index(e)
        expected = named[section] if section is not None else (meta["default_audience"],)
        assert e.labels == expected, e
        # AC-F4: a path keeps only headings every reader of the element may read.
        assert e.path[-1:] == (e.text,) if e.kind == "heading" else True
        assert all(h in {s.text for s in elements if s.kind == "heading" and set(e.labels) <= set(s.labels)} for h in e.path), e
    assert all(e.kind == "audience" for e in elements if re.match(r"(?:Audience|Κοινό):", e.text))


def test_names_in_either_language_ids_two_labels_and_names_no_label_knows():
    unknown = []
    assert labels_named("Returns; Legal", unknown) == ("returns", "legal")
    assert labels_named("Εξυπηρέτηση Πελατών", unknown) == ("customer-care",)
    assert labels_named("customer-care", unknown) == ("customer-care",)
    assert unknown == []
    assert labels_named("Returns; Marketing", unknown) == (UNLABELLED,) and unknown == ["Marketing"]
    assert labels_named("", unknown) == (UNLABELLED,)


def test_a_subsection_inherits_and_a_section_with_an_unknown_audience_is_unlabelled():
    text = ("# Manual\n\nDocument line\n\n## SECTION 1  Open\n\nAudience: Sales\n\nFor sales.\n\n### 1.1 Detail\n\nStill sales.\n\n"
            "## SECTION 2  Closed\n\nAudience: Board\n\nFor nobody named.\n")
    unknown = []
    labels = {e.text: e.labels for e in segmented(text, "md", {"default_audience": "returns"}, unknown)}
    assert labels["Document line"] == ("returns",)
    assert labels["SECTION 1  Open"] == labels["For sales."] == labels["1.1 Detail"] == labels["Still sales."] == ("sales",)
    assert labels["SECTION 2  Closed"] == labels["For nobody named."] == (UNLABELLED,) and unknown == ["Board"]


def test_without_a_sidecar_default_the_text_above_the_first_audience_is_unlabelled():
    labels = [e.labels for e in segmented("Title\nDocument line\n", "txt", {})]
    assert labels == [(UNLABELLED,), (UNLABELLED,)]


def test_coverage_counts_paragraphs_and_groups_of_twelve_rows_by_first_label():
    rows = [Element("row", "h", 0, 1, header=True, labels=("legal", "finance"))]
    rows += [Element("row", f"r{k}", 0, 1, labels=("legal", "finance")) for k in range(12)]
    more = [Element("row", f"s{k}", 0, 1, labels=("hr",)) for k in range(13)]
    coverage = Coverage()
    coverage.add([Element("paragraph", "p", 0, 1, labels=("sales",)), *rows, Element("audience", "Audience: Sales", 0, 1, labels=("sales",)),
                  Element("row", "h", 0, 1, header=True, labels=("hr",)), *more])
    # Twelve rows under a header are one unit, not two: the header is repeated in each group, never counted alone.
    assert dict(coverage.first) == {"sales": 1, "legal": 1, "hr": 2}
    assert dict(coverage.readable) == {"sales": 1, "legal": 1, "finance": 1, "hr": 2}


def test_the_coverage_command_reports_shares_and_fails_on_a_name_no_label_knows(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    base = tmp_path / "corpus"
    (base / "plan").mkdir(parents=True)
    (base / "narrative").mkdir()
    doc = {"id": "SAL_PRO_001", "type": "PRO", "type_name": "Procedure", "owner": "sales", "language": "en", "format": "txt",
           "target_mb": 0.01, "version": "1.0",
           "sections": [{"label": "sales", "topic_en": "Scope", "topic_el": "Πεδίο", "facts": [], "canary": None}]}
    narrative = "\n".join(header(doc, 1, doc["sections"][0]) + ["", "The team checks every case and records it.", ""])
    (base / "narrative" / "SAL_PRO_001.txt").write_bytes(narrative.encode("utf-8"))
    (base / "plan" / "catalogue.json").write_bytes(json.dumps([doc]).encode("utf-8"))
    (base / "plan" / "facts.json").write_bytes(b"[]")
    assert main(["corpus", "assemble"]) == 0
    capsys.readouterr()
    assert main(["corpus", "coverage"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"sales\s+first 0\.\d+\s+registered 0\.14", out) and "no label knows: none" in out
    (base / "narrative" / "SAL_PRO_001.txt").write_bytes(narrative.replace("Audience: Sales", "Audience: Board").encode("utf-8"))
    assert main(["corpus", "assemble"]) == 0
    capsys.readouterr()
    assert main(["corpus", "coverage"]) == 1
    assert "no label knows: Board" in capsys.readouterr().out



# The red team's probes of tick 40 (AC-F1 to AC-F4): can a document's own text open a Legal or Human Resources
# line to the sales role?
EN_DASH = chr(0x2013)


def labelled(text: str, fmt: str, needle: str):
    elements = segmented(text, fmt, {"default_audience": "sales"}, [])
    return next(e for e in elements if needle in e.text)


def test_a_later_audience_line_can_only_narrow_its_section():
    # AC-F1: "Audience: All staff" halfway down a Legal section opened the rest to everyone.
    text = ("# Policy: Sales\n\nDocument line\n\n## SECTION 1  Claims\n\nAudience: Legal\n\nSettled by counsel.\n\n"
            "Audience: All staff\n\nSECRET settlement ceiling.\n")
    assert labelled(text, "md", "SECRET").labels == (UNLABELLED,)
    narrowed = ("# Policy: Sales\n\nDocument line\n\n## SECTION 1  Desk\n\nAudience: Sales; Returns\n\nFor both.\n\n"
                "Audience: Returns\n\nFor returns alone.\n")
    assert labelled(narrowed, "md", "returns alone").labels == ("returns",)


@pytest.mark.parametrize("line", ["AUDIENCE - Human Resources", "audience: human resources", "ΚΟΙΝΟ: ΑΝΘΡΩΠΙΝΟ ΔΥΝΑΜΙΚΟ",
                                  "Audience " + EN_DASH + " Human Resources"])
def test_an_audience_line_is_read_whatever_its_case_accents_and_separator(line):
    # AC-F2: these were read as text, and the section fell back to the document's owner, sales.
    text = f"Policy: Sales\nDocument line\n\nSECTION 1  Pay\n{line}\nClassification: Internal\n\nSECRET salary bands.\n"
    assert labelled(text, "txt", "SECRET").labels == ("hr",)


def test_html_data_audience_and_audience_line_take_what_both_allow():
    # AC-F3: the line "All staff" overrode data-audience="Legal".
    text = ('<body><h1>Policy: Sales</h1><section data-audience="Legal"><h2>SECTION 1  Claims</h2>'
            '<p class="audience">Audience: All staff</p><p>SECRET settlement ceiling.</p></section></body>')
    notes = Counter()
    elements = segments(structure(parse(text, "html"), "html"), {"default_audience": "sales"}, [], notes)
    assert next(e for e in elements if "SECRET" in e.text).labels == (UNLABELLED,)
    assert notes == {"html data-audience and audience line disagree": 1}


def test_a_heading_path_never_carries_a_heading_its_reader_may_not_read():
    # AC-F4: an all staff subsection under a Legal heading carried that heading in its path.
    text = ("# Policy: Sales\n\nDocument line\n\n## SECTION 1  SECRET merger\n\nAudience: Legal\n\nTerms.\n\n"
            "### 1.1 Shop notice\n\nAudience: All staff\n\nShops stay open.\n")
    notice = labelled(text, "md", "Shops stay open")
    assert notice.labels == ("general",) and notice.path == ("1.1 Shop notice",)
    assert labelled(text, "md", "Terms").path == ("SECTION 1  SECRET merger",)
