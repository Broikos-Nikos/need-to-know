"""A plain text file cut by its parts (T7.6): its keyword sections split it where a heading puts back the title's path
and audience, each part cut after the file's head gives the units the whole cut gives, and a scan cuts only the parts
a change touched, ending where a build ends."""

import json

import pytest

from needtoknow.corpus.plan import LABEL_EL
from needtoknow.ingest import manifest, pipeline
from needtoknow.ingest.manifest import build, opened, scan

META = {"doc_id": "LOG_CAT_999", "title": "Catalogue and annex pack: Logistics", "owner": "logistics", "language": "en",
        "version": "1.0", "type": "CAT", "default_audience": "logistics", "classification": "internal", "format": "txt",
        "encoding": "utf-8"}
TEXT = f"""Catalogue and annex pack: Logistics
Document LOG_CAT_999, version 1.0, the Company, internal use

A preamble under the title, which the default audience reads.

SECTION 1  Courier pickups
Audience: Logistics
Classification: Internal

The courier pickup schedule is the main link between the warehouse and the customer.

2.1 Weekend cutoffs
Pickups on Saturday close an hour before the weekday cutoff.

Code | Courier | Cutoff
C1 | Courier One | 15:00
C2 | Courier Two | 16:00

The returns desk keeps a log of every parcel it opens.
SECTION 3  Lifted from a paragraph
Audience: Returns
Every return is checked within 3 working days of its arrival.

SECTION 4  Finance and logistics
Audience: Finance; Logistics

Invoices for courier services are approved once a month by the finance lead.
Audience: Finance
This sentence follows a narrowing inside its paragraph.

  SECTION 5  An indented heading

A section without an audience line of its own reads as the title's does.

RETURNS AND REPAIRS

A heading in capitals keeps the audience in force under it.

SECTION 6 covers the repairs process.

That line ends like a sentence, so it is no heading at all.

ΕΝΟΤΗΤΑ 7  Ελληνικό τμήμα
Κοινό: {LABEL_EL["legal"]}

Το νομικό τμήμα ελέγχει κάθε σύμβαση πριν από την υπογραφή της.
"""


def rows(units):
    return [(u.unit_id, u.labels, u.path, u.text, u.start, u.end) for u in units]


def by_parts(text: str, meta: dict) -> list:
    """The file cut part by part, each part after the head, as a scan cuts the parts a change touched."""
    from dataclasses import replace
    starts = pipeline.parts(text, meta["format"])
    head = text[:starts[0]]
    out = [u for u in pipeline.text_units(head, meta, []) if u.start < len(head)]
    for a, b in zip(starts, [*starts[1:], len(text)]):
        out += [replace(u, start=u.start + a - len(head), end=u.end + a - len(head))
                for u in pipeline.text_units(head + text[a:b], meta, []) if u.start >= len(head)]
    return out


def test_a_plain_text_file_splits_at_the_keyword_headings_that_open_a_run():
    starts = pipeline.parts(TEXT, "txt")
    assert [TEXT[s:TEXT.index("\n", s)].strip() for s in starts] == [
        "SECTION 1  Courier pickups", "SECTION 4  Finance and logistics", "SECTION 5  An indented heading",
        "ΕΝΟΤΗΤΑ 7  Ελληνικό τμήμα"]
    assert pipeline.parts(TEXT, "md") == [] and pipeline.parts("SECTION 1  Title\n\nText.\n", "txt") == []


def test_each_part_cut_after_the_head_gives_the_units_the_whole_cut_gives():
    whole = pipeline.text_units(TEXT, META, [])
    assert rows(by_parts(TEXT, META)) == rows(whole)
    labels = {u.text.split()[0]: u.labels for u in whole if u.layer == "prose"}
    assert labels["A"] == ("logistics",) and labels["Invoices"] == ("finance",)  # narrowed inside its paragraph
    assert labels["Το"] == ("legal",)


@pytest.fixture
def sectioned(monkeypatch, tmp_path):
    monkeypatch.setenv("NTK_DATA_DIR", str(tmp_path))
    full = tmp_path / "corpus" / "full"
    full.mkdir(parents=True)
    (full / "LOG_CAT_999.txt").write_bytes(TEXT.encode("utf-8"))
    (full / "LOG_CAT_999.meta.json").write_bytes(json.dumps(META, ensure_ascii=False).encode("utf-8"))
    cut = []
    real = pipeline.text_units
    monkeypatch.setattr(pipeline, "text_units", lambda text, meta, unknown=None: cut.append(len(text)) or real(text, meta, unknown))
    build()
    cut.clear()
    return full / "LOG_CAT_999.txt", full / "LOG_CAT_999.meta.json", cut


def manifest_rows():
    with opened() as db:
        return sorted(db.execute('SELECT unit_id, doc_id, section_key, layer, labels, path, text, start, "end", language '
                                 "FROM units"))


def edit(file, old: str, new: str) -> None:
    text = file.read_bytes().decode("utf-8")
    assert text.count(old) == 1, old
    file.write_bytes(text.replace(old, new).encode("utf-8"))


def as_built() -> list:
    """The manifest's units after the scan, then the same after a build that cuts every file whole."""
    scanned = manifest_rows()
    summary = build()
    assert summary["units_added"] == summary["units_removed"] == 0
    return scanned


def test_an_edit_in_one_part_cuts_that_part_alone_and_ends_where_a_build_ends(sectioned):
    file, _, cut = sectioned
    edit(file, "courier services are approved", "courier services are checked")
    summary = scan()
    assert summary["units_added"] == summary["units_removed"] == 1
    head = TEXT.index("SECTION 1")
    part = TEXT.index("  SECTION 5") - TEXT.index("SECTION 4") - len("approved") + len("checked")
    assert cut == [head + part]  # the head and the one part, never the file
    assert as_built() == manifest_rows()


def test_the_parts_after_a_change_move_by_its_length(sectioned):
    file, _, cut = sectioned
    old, new = "The courier pickup schedule is", "The courier pickup schedule, read every morning, is"
    edit(file, old, new)
    scan()
    assert cut == [TEXT.index("SECTION 4") + len(new) - len(old)]  # the head and the first part
    assert as_built() == manifest_rows()


@pytest.mark.parametrize("old, new", [
    ("  SECTION 5  An indented heading", "SECTION 4A  Inserted\nAudience: Sales\n\nA new part.\n\n  SECTION 5  An indented heading"),
    ("SECTION 4  Finance and logistics\nAudience: Finance; Logistics\n\nInvoices for courier services are approved once a "
     "month by the finance lead.\nAudience: Finance\nThis sentence follows a narrowing inside its paragraph.\n\n", ""),
    ("Το νομικό τμήμα ελέγχει", "Το νομικό τμήμα πάντα ελέγχει"),
])
def test_a_part_inserted_removed_or_edited_last_ends_where_a_build_ends(sectioned, old, new):
    file, _, cut = sectioned
    edit(file, old, new)
    scan()
    assert max(cut, default=0) < len(file.read_bytes().decode("utf-8"))  # a part removed needs no cut at all
    assert as_built() == manifest_rows()


def test_a_change_to_the_head_or_to_the_default_audience_cuts_the_file_whole(sectioned):
    file, sidecar, cut = sectioned
    edit(file, "A preamble under the title", "A longer preamble under the title")
    scan()
    assert cut == [len(TEXT) + len("longer ")]
    assert as_built() == manifest_rows()
    cut.clear()
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    meta["default_audience"] = "sales"
    sidecar.write_bytes(json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    edit(file, "An indented heading", "An indented heading, renamed")
    scan()
    assert cut == [len(file.read_bytes().decode("utf-8"))]
    with opened() as db:
        labels = {text.split()[0]: json.loads(l) for text, l in db.execute("SELECT text, labels FROM units WHERE layer = 'prose'")}
    assert labels["A"] == ["sales"]
    assert as_built() == manifest_rows()


def test_a_part_that_shares_a_section_with_a_part_kept_is_cut_whole(sectioned):
    file, _, cut = sectioned
    text = file.read_bytes().decode("utf-8")
    again = ("\nSECTION 1  Courier pickups\nAudience: Logistics\n\nThe courier pickup schedule is the main link between "
             "the warehouse and the customer.\n")  # a unit the first section holds too: the whole cut numbers it
    file.write_bytes((text + again).encode("utf-8"))
    scan()
    assert cut[-1] == len(file.read_bytes().decode("utf-8"))  # its section found among the parts kept: cut whole
    assert as_built() == manifest_rows()


def test_a_scan_keeps_each_files_parts_and_a_build_too(sectioned):
    with opened() as db:
        kept = db.execute('SELECT ordinal, start, "end" FROM parts WHERE doc_id = ? ORDER BY ordinal', ("LOG_CAT_999",)).fetchall()
    starts = pipeline.parts(TEXT, "txt")
    assert kept == [(k, a, b) for k, (a, b) in enumerate(zip([0, *starts], [*starts, len(TEXT)]))]
    assert manifest.parts_of(TEXT, META)[0][2] != manifest.parts_of(TEXT, {**META, "default_audience": "sales"})[0][2]


def test_a_sidecars_language_reaches_every_unit_alone_or_with_an_edit_as_a_build_from_nothing_gives_it(sectioned):
    # DR-F42: a unit's language is outside its id, and a scan or a build rewrote a surviving unit's offsets alone, so a
    # language corrected in a sidecar reached only the units an edit made.
    file, sidecar, _ = sectioned

    def relabel(language: str) -> None:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        sidecar.write_bytes(json.dumps({**meta, "language": language}, ensure_ascii=False).encode("utf-8"))

    def languages() -> set:
        return {row[-1] for row in manifest_rows()}

    relabel("el")
    scan()
    assert languages() == {"el"}
    edit(file, "courier services are approved", "courier services are checked")
    relabel("en")
    scan()
    assert languages() == {"en"}
    scanned = manifest_rows()
    for leftover in manifest.manifest_path().parent.glob(manifest.manifest_path().name + "*"):
        leftover.unlink()
    build()
    assert manifest_rows() == scanned
