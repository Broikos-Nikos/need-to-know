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
from needtoknow.ingest.units import Unit
from needtoknow.policy import UNLABELLED
from needtoknow.settings import repo_root

SAMPLES = sorted((repo_root() / "samples").glob("*.meta.json"))


def segmented(text: str, fmt: str, meta: dict, unknown=None):
    return segments(structure(parse(text, fmt), fmt), meta, unknown)


@pytest.mark.parametrize("meta_path", SAMPLES, ids=lambda p: p.name.split(".")[0])
def test_a_row_is_made_twice_on_its_way_and_rows_under_the_same_headings_share_one_path(meta_path, monkeypatch):
    # PF-F1: structure and segments copied every row three times after making it, and every element built its own
    # path tuple: over the corpus, 70 of 120 s of ingestion. Now structure makes each row with its path and segments
    # makes it once more with its labels and safe path.
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    text = load(meta_path.with_name(meta_path.name[: -len(".meta.json")] + "." + meta["format"]), meta)
    made = Counter()
    real = Element.__init__

    def counted(self, kind, *args, **kwargs):
        made[kind] += 1
        real(self, kind, *args, **kwargs)

    monkeypatch.setattr(Element, "__init__", counted)
    elements = segmented(text, meta["format"], meta)
    rows = [e for e in elements if e.kind == "row"]
    assert len(rows) > 40 and made["row"] == 2 * len(rows)
    shared = {}
    for row in rows:
        shared.setdefault((row.path, row.labels), set()).add(id(row.path))
    assert all(len(ids) == 1 for ids in shared.values())


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


def test_coverage_counts_units_by_first_label_by_every_reader_and_by_layer():
    def unit(labels, layer):
        return Unit("id", "DOC", "key", layer, "text", labels, (), 0, 1, "en")

    coverage = Coverage()
    coverage.add([unit(("sales",), "prose"), unit(("legal", "finance"), "table"), unit(("legal", "finance"), "table")])
    assert dict(coverage.first) == {"sales": 1, "legal": 2}
    assert dict(coverage.readable) == {"sales": 1, "legal": 2, "finance": 2}
    assert dict(coverage.layers) == {"prose": 1, "table": 2}


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


@pytest.mark.parametrize("fmt, text", [
    ("md", "# Policy: Sales@@Document line@@## SECTION 1  Contracts@@Audience: Legal~The register lists every supplier.~"),
    ("txt", "Policy: Sales~Document line@@SECTION 1  Contracts~Audience: Legal~The register lists every supplier.~"),
])
def test_an_audience_line_heading_a_paragraph_is_read_as_if_a_blank_line_followed_it(fmt, text):
    # DR-F8: with no blank line after it, the line was text, and the section took the document's sales.
    text = text.replace("@@", "~~").replace("~", "\n")
    elements = segmented(text, fmt, {"default_audience": "sales", "format": fmt}, [])
    register = next(e for e in elements if "register" in e.text)
    assert register.labels == ("legal",) and register.text == "The register lists every supplier."
    assert text[register.start:register.end] == register.text
    assert next(e for e in elements if e.text.startswith("Audience")).kind == "audience"


def test_an_audience_line_inside_a_paragraph_narrows_the_whole_paragraph():
    # DR-F8: a line further down a paragraph was ignored; read now as a later line is (AC-F1).
    text = ("# Policy: Sales@@Document line@@## SECTION 1  Claims@@Audience: Legal@@Settled by counsel.~"
            "Audience: All staff~SECRET settlement ceiling.~").replace("@@", "~~").replace("~", "\n")
    notes = Counter()
    elements = segments(structure(parse(text, "md"), "md"), {"default_audience": "sales", "format": "md"}, [], notes)
    assert next(e for e in elements if "SECRET" in e.text).labels == (UNLABELLED,)
    assert notes == {"audience line inside a paragraph, read as narrowing only": 1}


def test_a_wrapped_plain_text_line_never_widens_who_may_read():
    # DR-F9: "15 Μαΐου ..." read as a numbered heading closed the legal section; the rest went to everyone.
    text = ("Πολιτική συμβάσεων~~ΕΝΟΤΗΤΑ 3  Συμβάσεις~Κοινό: Νομικό Τμήμα~~"
            "Η σύμβαση με τον προμηθευτή υπογράφηκε στις~15 Μαΐου από τη Διεύθυνση και ισχύει~"
            "έως το 2027 με ρήτρα αποχώρησης.~").replace("~", "\n")
    elements = segmented(text, "txt", {"default_audience": "general", "format": "txt"}, [])
    assert [e.text for e in elements if e.kind == "heading"] == ["Πολιτική συμβάσεων", "ΕΝΟΤΗΤΑ 3  Συμβάσεις"]
    assert all(e.labels == ("legal",) for e in elements if e.kind == "paragraph")


def test_a_heading_guessed_from_its_wording_keeps_the_audience_in_force():
    # DR-F9: a numbered heading with no audience line of its own may be a wrapped sentence, so it never widens.
    text = ("Πολιτική συμβάσεων~~ΕΝΟΤΗΤΑ 3  Συμβάσεις~Κοινό: Νομικό Τμήμα~~Όροι.~~"
            "15 Μαΐου από τη Διεύθυνση~Η ρήτρα ισχύει έως το 2027.~").replace("~", "\n")
    elements = segmented(text, "txt", {"default_audience": "general", "format": "txt"}, [])
    guessed = next(e for e in elements if e.text.startswith("15"))
    assert guessed.kind == "heading" and guessed.inferred and guessed.labels == ("legal",)
    assert next(e for e in elements if "2027" in e.text).labels == ("legal",)


def test_an_unknown_html_audience_is_reported_as_written():
    # DR-F13: it was reported folded, lower case and without accents.
    unknown = []
    segmented('<body><h1>T</h1><section data-audience="Διοικητικό Συμβούλιο"><h2>S</h2><p>x</p></section></body>', "html",
              {"default_audience": "general", "format": "html"}, unknown)
    assert unknown == ["Διοικητικό Συμβούλιο"]
