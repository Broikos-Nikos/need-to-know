"""Units (T2.6): a paragraph or a sentence packed piece, row groups under a repeated header with content defined
boundaries, never across a segment, short pieces merged, ids content addressed (KB1-F10, KB1-F8, AC-F4)."""

import json

import pytest

from needtoknow.ingest.decode import load
from needtoknow.ingest.parse import parse
from needtoknow.ingest.segments import segments
from needtoknow.ingest.structure import structure
from needtoknow.ingest.units import MAX_ROWS, MAX_TOKENS, MIN_TOKENS, estimate_tokens, make_unit, units
from needtoknow.settings import repo_root

SAMPLES = sorted((repo_root() / "samples").glob("*.meta.json"))
SENTENCE = "The returns desk checks each parcel against its authorisation and records the result in Dock."


def cut(text: str, fmt: str = "md", meta: dict | None = None, language: str = "en"):
    meta = meta or {"default_audience": "sales"}
    return units(segments(structure(parse(text, fmt), fmt), meta), "DOC_001", language)


def table(rows: int, extra: str | None = None, at: int = 0) -> str:
    lines = [f"| R{k:04d} | Attica | {k} | shop manager | {30000 + k} |" for k in range(rows)]
    if extra is not None:
        lines.insert(at, extra)
    return "# T\n\n## SECTION 1  Shops\n\nAudience: All staff\n\n| Shop | Region | Unit | Manager | Extension |\n|---|---|---|---|---|\n" + "\n".join(lines) + "\n"


@pytest.mark.parametrize("meta_path", SAMPLES, ids=lambda p: p.name.split(".")[0])
def test_the_samples_cut_into_units_within_their_segments_and_limits(meta_path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    text = load(meta_path.with_name(meta_path.name[: -len(".meta.json")] + "." + meta["format"]), meta)
    elements = segments(structure(parse(text, meta["format"]), meta["format"]), meta)
    cut_units = units(elements, meta["doc_id"], meta["language"])
    assert cut_units and len({u.unit_id for u in cut_units}) == len(cut_units)
    for u in cut_units:
        assert estimate_tokens(u.text, u.language) <= MAX_TOKENS or u.layer == "prose", u
        same = [e for e in elements if e.labels == u.labels and e.path == u.path and e.kind in ("paragraph", "row")]
        assert same, u  # every unit comes from one run of elements sharing labels and path
        assert not u.text.startswith(("Audience:", "Κοινό:", "Classification:", "Διαβάθμιση:")), u
    tables = [u for u in cut_units if u.layer == "table"]
    assert tables and all(u.text.split("\n")[0] == next(e.text for e in elements if e.header and e.path == u.path) for u in tables)
    assert units(elements, meta["doc_id"], meta["language"]) == cut_units  # a replay gives the same units


def test_a_long_paragraph_splits_at_sentence_ends_and_a_short_one_joins_the_next():
    long = " ".join([SENTENCE] * 20)
    text = f"# T\n\n## SECTION 1  Desk\n\nAudience: Returns\n\nShort note.\n\n{long}\n\n## SECTION 2  Other\n\nAudience: Legal\n\nTiny.\n"
    pieces = [u for u in cut(text) if u.path[-1:] == ("SECTION 1  Desk",)]
    assert len(pieces) >= 2 and all(estimate_tokens(u.text, "en") <= MAX_TOKENS + estimate_tokens("Short note.", "en") + 1 for u in pieces)
    assert pieces[0].text.startswith("Short note.\n\n" + SENTENCE)
    assert " ".join(u.text.replace("Short note.\n\n", "") for u in pieces).split() == long.split()
    other = [u for u in cut(text) if u.path[-1:] == ("SECTION 2  Other",)]
    assert [u.text for u in other] == ["Tiny."] and estimate_tokens("Tiny.", "en") < MIN_TOKENS  # alone in its segment


def test_a_short_piece_never_joins_across_a_segment():
    text = "# T\n\n## SECTION 1  A\n\nAudience: Sales\n\nShort.\n\n## SECTION 2  B\n\nAudience: Legal\n\n" + " ".join([SENTENCE] * 3) + "\n"
    by_path = {u.path[-1]: u for u in cut(text)}
    assert by_path["SECTION 1  A"].text == "Short." and by_path["SECTION 1  A"].labels == ("sales",)
    assert "Short." not in by_path["SECTION 2  B"].text


def test_row_groups_repeat_the_header_respect_the_limits_and_lose_no_row():
    groups = [u for u in cut(table(200)) if u.layer == "table"]
    assert all(u.text.split("\n")[0] == "Shop | Region | Unit | Manager | Extension" for u in groups)
    rows = [line for u in groups for line in u.text.split("\n")[1:]]
    assert len(rows) == 200 and len(set(rows)) == 200
    assert all(len(u.text.split("\n")) - 1 <= MAX_ROWS and estimate_tokens(u.text, "en") <= MAX_TOKENS for u in groups)


def test_inserting_a_row_reshapes_one_or_two_groups_not_all_that_follow():
    before = {u.unit_id for u in cut(table(200)) if u.layer == "table"}
    after = {u.unit_id for u in cut(table(200, "| X0001 | Crete | 7 | team leader | 40001 |", 20)) if u.layer == "table"}
    assert len(before - after) <= 2 and len(after - before) <= 3


def test_ids_change_with_the_unit_and_with_nothing_else():
    text = "# T\n\n## SECTION 1  A\n\nAudience: Sales\n\n" + SENTENCE + "\n\n## SECTION 2  B\n\nAudience: Legal\n\n" + SENTENCE + " Again.\n"
    first = {u.path[-1]: u.unit_id for u in cut(text)}
    edited = {u.path[-1]: u.unit_id for u in cut(text.replace(" Again.", " Twice."))}
    relabelled = {u.path[-1]: u.unit_id for u in cut(text.replace("Audience: Sales", "Audience: Sales; Returns"))}
    assert first["SECTION 1  A"] == edited["SECTION 1  A"] and first["SECTION 2  B"] != edited["SECTION 2  B"]
    assert first["SECTION 1  A"] != relabelled["SECTION 1  A"] and first["SECTION 2  B"] == relabelled["SECTION 2  B"]
    # The labels alone change the id, with text and path held still.
    path = ("Handbook", "SECTION 1  A")
    assert (make_unit("DOC_001", "prose", SENTENCE, (("sales",), path), 0, 1, "en").unit_id
            != make_unit("DOC_001", "prose", SENTENCE, (("sales", "returns"), path), 0, 1, "en").unit_id)


def test_a_units_path_holds_no_heading_its_readers_may_not_read():
    # AC-F4: the path travels with the unit into its embedding prefix and its citation.
    text = ("# Policy: Sales\n\nDocument line\n\n## SECTION 1  SECRET merger\n\nAudience: Legal\n\nTerms.\n\n"
            "### 1.1 Shop notice\n\nAudience: All staff\n\nShops stay open.\n")
    notice = next(u for u in cut(text) if "Shops stay open" in u.text)
    assert notice.labels == ("general",) and notice.path == ("1.1 Shop notice",)


def test_the_pieces_of_a_long_paragraph_carry_their_own_offsets():
    # DR-F12: every piece carried the whole paragraph's span, so nothing could order or cite them.
    long = " ".join([SENTENCE] * 20)
    text = f"# T@@## SECTION 1  Desk@@Audience: Returns@@{long}~"
    text = text.replace("@@", "\n\n").replace("~", "\n")
    pieces = [u for u in cut(text) if u.path[-1:] == ("SECTION 1  Desk",)]
    assert len(pieces) >= 2 and all(text[u.start:u.end] == u.text for u in pieces)
    assert [u.start for u in pieces] == sorted(u.start for u in pieces) and len({u.start for u in pieces}) == len(pieces)


def test_a_groups_length_is_counted_as_rows_come_and_its_boundaries_read_as_the_digest_says():
    # T3.16: table_groups joined the group again for every row only to measure it, and is_boundary read the hash
    # through hex; both now count, and must cut exactly the groups the join and the hex cut.
    import random

    from needtoknow.ingest import units as cutter
    from needtoknow.ingest.structure import Element
    newline = chr(10)

    def joined(header, rows, language):
        head = header.text if header else ""
        groups, current = [], []

        def close():
            if current:
                groups.append((newline.join(([head] if head else []) + [r.text for r in current]), current[0].start,
                               current[-1].end))
                current.clear()

        for row in rows:
            candidate = newline.join(([head] if head else []) + [r.text for r in current] + [row.text])
            if current and (len(current) >= MAX_ROWS or estimate_tokens(candidate, language) > MAX_TOKENS):
                close()
            current.append(row)
            if int(cutter.digest(row.text, size=8), 16) % cutter.ROWS_PER_UNIT == 0:
                close()
        close()
        return groups

    rng = random.Random(73)
    headers = (None, Element("row", "Shop | Region", 0, 0, header=True), Element("row", "H" * 700, 0, 0, header=True))
    for case in range(300):
        width = rng.choice((5, 40, 120, 400, 1100))
        rows = [Element("row", "x" * rng.randint(1, width) + str(n), 10 * n, 10 * n + 5) for n in range(rng.randint(1, 90))]
        header = rng.choice(headers)
        for language in ("en", "el"):
            assert cutter.table_groups(header, rows, language) == joined(header, rows, language), (case, language)
