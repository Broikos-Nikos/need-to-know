"""Turn the writers' narrative into the corpus: a title block, the sections, and the tables a real manual carries.

Every table has named columns. Each department's documents carry that department's
register under the owner's audience: a sales log, a stock list, a return
authorisation log, a ticket log, a contract register, cost centres, network sites or
a training roster. Every document also carries a page of the shop directory and a
short glossary, which everyone reads, so they sit under the general audience, and it
closes with its forms and its revision history, dated in order up to its version.

The rows of the register and the directory are budgeted so that each label's share
of units meets `config/labels.toml` (N43). A unit is a prose section or a group of
ROWS_PER_UNIT table rows. Tables carry no policy value, so they can never contradict
a fact, and every date of something done falls on or before LAST_DAY; only a
contract renewal may lie ahead.
"""

from __future__ import annotations

import html
import json
import random
import shutil
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from needtoknow.corpus import dump_json, exclusive, narrative_dir
from needtoknow.corpus.plan import DEPARTMENTS, LABEL_EL, LABEL_EN, SEED, world
from needtoknow.policy import load_labels
from needtoknow.settings import data_dir

ROWS_PER_UNIT = 12
# The scale of a plan: units per planned megabyte, set so that the plan's 280 MB come out
# at about the 300 MB the owner settled on (N39). `ntk corpus assemble` prints the result.
UNITS_PER_MB = 1250
FIRST_DAY, LAST_DAY = date(2023, 1, 1), date(2026, 9, 30)

TITLES = {
    "directory": ("Shop directory", "Κατάλογος καταστημάτων", ["Shop", "Region", "Unit", "Manager", "Extension"],
                  ["Κατάστημα", "Περιφέρεια", "Μονάδα", "Υπεύθυνος", "Εσωτερικό"]),
    "glossary": ("Glossary", "Γλωσσάριο", ["Term", "Meaning"], ["Όρος", "Σημασία"]),
    "forms": ("Form definitions", "Ορισμοί εντύπων", ["Form", "Field", "Required", "Entered in"],
              ["Έντυπο", "Πεδίο", "Υποχρεωτικό", "Καταχωρείται στο"]),
    "revisions": ("Revision history", "Ιστορικό αναθεωρήσεων", ["Revision", "Date", "Role", "Change"],
                  ["Αναθεώρηση", "Ημερομηνία", "Ρόλος", "Αλλαγή"]),
    "sales": ("Shop sales log", "Ημερολόγιο πωλήσεων", ["Date", "Shop", "Product", "Units sold", "Channel"],
              ["Ημερομηνία", "Κατάστημα", "Προϊόν", "Τεμάχια", "Κανάλι"]),
    "logistics": ("Stock list", "Κατάσταση αποθέματος", ["SKU", "Product", "Bin", "Units on hand", "Reorder"],
                  ["Κωδικός", "Προϊόν", "Θέση", "Τεμάχια σε απόθεμα", "Αναπαραγγελία"]),
    "returns": ("Return authorisation log", "Ημερολόγιο εξουσιοδοτήσεων επιστροφής",
                ["RMA", "Received", "Product", "Reason code", "Status"], ["RMA", "Παραλαβή", "Προϊόν", "Κωδικός αιτίας", "Κατάσταση"]),
    "customer-care": ("Ticket log", "Ημερολόγιο αιτημάτων", ["Ticket", "Opened", "Channel", "Category", "Team"],
                      ["Αίτημα", "Άνοιγμα", "Κανάλι", "Κατηγορία", "Ομάδα"]),
    "legal": ("Contract register", "Μητρώο συμβάσεων", ["Contract", "Counterparty", "Signed", "Renewal", "Owner"],
              ["Σύμβαση", "Αντισυμβαλλόμενος", "Υπογραφή", "Ανανέωση", "Υπεύθυνος"]),
    "finance": ("Cost centres", "Κέντρα κόστους", ["Cost centre", "Department", "Region", "Approver", "Ledger code"],
                ["Κέντρο κόστους", "Τμήμα", "Περιφέρεια", "Εγκρίνων", "Κωδικός Ledger"]),
    "network-ops": ("Network sites", "Σημεία δικτύου", ["Site", "Region", "Node type", "Commissioned", "Status"],
                    ["Σημείο", "Περιφέρεια", "Τύπος κόμβου", "Ενεργοποίηση", "Κατάσταση"]),
    "hr": ("Training roster", "Πρόγραμμα εκπαιδεύσεων", ["Course", "Title", "Department", "Session", "Seats"],
           ["Μάθημα", "Τίτλος", "Τμήμα", "Συνεδρία", "Θέσεις"]),
}
ROLES = {"en": ["shop manager", "team leader", "regional manager", "process owner", "quality reviewer"],
         "el": ["διευθυντής καταστήματος", "επικεφαλής ομάδας", "περιφερειακός διευθυντής", "υπεύθυνος διαδικασίας", "ελεγκτής ποιότητας"]}
LEGAL_ROLES = {"en": ["legal counsel", "contract manager", "compliance officer"],
               "el": ["νομικός σύμβουλος", "υπεύθυνος συμβάσεων", "υπεύθυνος κανονιστικής συμμόρφωσης"]}
CHANGES = {"en": ["wording clarified in section {s}", "table layout updated", "references to Atlas updated",
                  "approval step moved to Relay", "examples added to section {s}", "obsolete form removed"],
           "el": ["διευκρίνιση διατύπωσης στην ενότητα {s}", "ενημέρωση διάταξης πίνακα", "ενημέρωση αναφορών στο Atlas",
                  "μεταφορά βήματος έγκρισης στο Relay", "προσθήκη παραδειγμάτων στην ενότητα {s}", "κατάργηση παλαιού εντύπου"]}
FIELDS = {"en": ["customer number", "contract reference", "device serial", "reason code", "approver", "date received", "channel"],
          "el": ["αριθμός πελάτη", "αριθμός σύμβασης", "σειριακός αριθμός συσκευής", "κωδικός αιτίας", "εγκρίνων", "ημερομηνία παραλαβής", "κανάλι"]}
NEED = {"en": ["mandatory", "optional"], "el": ["υποχρεωτικό", "προαιρετικό"]}
TERMS = {"en": [("RMA", "a return authorisation recorded in Dock"), ("First line", "the advisors who take the first contact"),
                ("Second line", "the team that resolves escalated cases"), ("Cut off", "the last pickup time agreed with a courier"),
                ("Depot", "a regional store of spare parts"), ("Ticket", "a customer request recorded in Relay")],
         "el": [("RMA", "εξουσιοδότηση επιστροφής καταχωρημένη στο Dock"), ("Πρώτη γραμμή", "οι σύμβουλοι που δέχονται την πρώτη επαφή"),
                ("Δεύτερη γραμμή", "η ομάδα που επιλύει κλιμακωμένες υποθέσεις"), ("Ώρα αποκοπής", "η τελευταία ώρα παραλαβής που έχει συμφωνηθεί με μεταφορέα"),
                ("Αποθήκη ανταλλακτικών", "περιφερειακή αποθήκη ανταλλακτικών"), ("Αίτημα", "αίτημα πελάτη καταχωρημένο στο Relay")]}
CHANNELS = {"en": ["shop", "online", "phone", "partner"], "el": ["κατάστημα", "ηλεκτρονικό", "τηλεφωνικό", "συνεργάτης"]}
YES_NO = {"en": ["yes", "no"], "el": ["ναι", "όχι"]}
RETURN_STATUS = {"en": ["received", "inspected", "refunded", "repaired", "rejected"],
                 "el": ["παραλήφθηκε", "ελέγχθηκε", "επιστράφηκαν χρήματα", "επισκευάστηκε", "απορρίφθηκε"]}
CATEGORIES = {"en": ["billing", "coverage", "device fault", "number porting", "contract change", "complaint"],
              "el": ["χρέωση", "κάλυψη", "βλάβη συσκευής", "φορητότητα αριθμού", "αλλαγή σύμβασης", "παράπονο"]}
TEAMS = {"en": ["first line", "second line", "billing desk", "technical desk"],
         "el": ["πρώτη γραμμή", "δεύτερη γραμμή", "γραφείο χρεώσεων", "τεχνικό γραφείο"]}
NODES = {"en": ["mobile mast", "fibre cabinet", "exchange", "microwave link", "equipment room"],
         "el": ["σταθμός βάσης", "καμπίνα οπτικής ίνας", "κέντρο", "μικροκυματική ζεύξη", "αίθουσα εξοπλισμού"]}
SITE_STATUS = {"en": ["in service", "under maintenance", "retired"], "el": ["σε λειτουργία", "σε συντήρηση", "εκτός λειτουργίας"]}
COURSES = {"en": ["Induction", "Data protection", "Health and safety", "Sales conversations", "Complaint handling",
                  "Fibre installation", "Leadership basics"],
           "el": ["Εισαγωγική εκπαίδευση", "Προστασία δεδομένων", "Υγεία και ασφάλεια", "Συζητήσεις πωλήσεων",
                  "Διαχείριση παραπόνων", "Εγκατάσταση οπτικής ίνας", "Βασικές αρχές ηγεσίας"]}


def day(rng: random.Random, first: date = FIRST_DAY, last: date = LAST_DAY) -> date:
    return first + timedelta(days=rng.randrange((last - first).days + 1))


def shop(rng: random.Random) -> str:
    return f"{rng.choice('ABCDEFGHJKLMNP')}{rng.choice('ABCDEFGHJKLMNP')}{rng.randint(100, 999)}"


def register_rows(owner: str, lang: str, rng: random.Random, w: dict, n: int) -> list[str]:
    """n rows of a department's register; logs run in date order."""
    products, regions = [p[lang] for p in w["products"]], w["regions"][lang]
    departments = [(LABEL_EL if lang == "el" else LABEL_EN)[d] for d in DEPARTMENTS]
    if owner == "sales":
        rows = sorted((day(rng), shop(rng), rng.choice(products), rng.randint(1, 40), rng.choice(CHANNELS[lang])) for _ in range(n))
        return [f"{d.isoformat()} | {s} | {p} | {u} | {c}" for d, s, p, u, c in rows]
    if owner == "logistics":
        return [f"{rng.choice('DVAXTRS')}{rng.choice('DVAXTRS')}{rng.randint(10000, 99999)} | {rng.choice(products)} | "
                f"{rng.choice('ABCDEFGH')}{rng.randint(1, 60):02d} | {rng.randint(0, 900)} | {rng.choice(YES_NO[lang])}" for _ in range(n)]
    if owner == "returns":
        start, rows = rng.randint(100000, 800000), sorted((day(rng), rng.choice(products), rng.randint(1, 9), rng.choice(RETURN_STATUS[lang])) for _ in range(n))
        return [f"RMA{start + k} | {d.isoformat()} | {p} | RC{r} | {s}" for k, (d, p, r, s) in enumerate(rows)]
    if owner == "customer-care":
        start, rows = rng.randint(1000000, 8000000), sorted((day(rng), rng.choice(CHANNELS[lang]), rng.choice(CATEGORIES[lang]), rng.choice(TEAMS[lang])) for _ in range(n))
        return [f"T{start + k} | {d.isoformat()} | {c} | {g} | {t}" for k, (d, c, g, t) in enumerate(rows)]
    if owner == "legal":
        parties = w["partners"]["couriers"] + w["partners"]["repair_centres"] + [w["partners"]["insurer"]]
        rows = sorted((day(rng, FIRST_DAY, date(2026, 6, 30)), rng.choice(parties), rng.choice(LEGAL_ROLES[lang])) for _ in range(n))
        return [f"K{1000 + k} | {p} | {d.isoformat()} | {(d + timedelta(days=365 * rng.randint(1, 3))).isoformat()} | {o}"
                for k, (d, p, o) in enumerate(rows)]
    if owner == "finance":
        return [f"CC{rng.randint(100, 999)} | {rng.choice(departments)} | {rng.choice(regions)} | {rng.choice(ROLES[lang])} | "
                f"L{rng.randint(10000, 99999)}" for _ in range(n)]
    if owner == "network-ops":
        rows = sorted((day(rng), rng.choice(regions), rng.choice(NODES[lang]), rng.choice(SITE_STATUS[lang])) for _ in range(n))
        return [f"S{rng.randint(1000, 9999)} | {r} | {t} | {d.isoformat()} | {s}" for d, r, t, s in rows]
    if owner == "hr":
        rows = sorted((day(rng), rng.randrange(len(COURSES[lang])), rng.choice(departments), rng.randint(8, 30)) for _ in range(n))
        return [f"TR{100 + c} | {COURSES[lang][c]} | {dep} | {d.isoformat()} | {seats}" for d, c, dep, seats in rows]
    raise ValueError(f"no register for {owner!r}")


def directory_rows(lang: str, rng: random.Random, w: dict, n: int) -> list[str]:
    return [f"{shop(rng)} | {rng.choice(w['regions'][lang])} | {rng.randint(1, 240)} | {rng.choice(ROLES[lang])} | "
            f"{rng.randint(30000, 89999)}" for _ in range(n)]


def small_tables(doc: dict, w: dict) -> list[tuple[str, str, list[str]]]:
    """The forms, the glossary and the revision history, as (kind, audience, rows): small, and fixed per document."""
    lang, rng = doc["language"], random.Random(f"{SEED}:{doc['id']}:small")
    forms = []
    for number in rng.sample(range(10, 100), rng.randint(2, 5)):
        for field in rng.sample(FIELDS[lang], rng.randint(4, len(FIELDS[lang]))):
            forms.append(f"F{number} | {field} | {rng.choice(NEED[lang])} | {rng.choice(list(w['systems'].values()))}")
    glossary = [f"{term} | {meaning}" for term, meaning in rng.sample(TERMS[lang], rng.randint(3, len(TERMS[lang])))]
    major, minor = (int(part) for part in doc["version"].split("."))
    versions = [f"{a}.{b}" for a in range(1, major + 1) for b in range(10) if (a, b) <= (major, minor)]
    dates, when = [], day(rng, date(2024, 1, 1))
    for _ in versions:
        dates.append(when)
        when -= timedelta(days=rng.randint(20, 120))
    sections = len(doc.get("sections") or []) or 1
    revisions = [f"{v} | {d.isoformat()} | {rng.choice(ROLES[lang])} | {rng.choice(CHANGES[lang]).format(s=rng.randint(1, sections))}"
                 for v, d in zip(versions, reversed(dates))]
    return [("forms", doc["owner"], forms), ("glossary", "general", glossary), ("revisions", doc["owner"], revisions)]


def units_of(rows: list[str]) -> int:
    return -(-len(rows) // ROWS_PER_UNIT)


def budgets(docs: list[dict], shares: dict[str, float], w: dict) -> dict[str, dict[str, int]]:
    """Rows of each document's register and directory, so that every label's share of units meets its share.

    The plan's sizes set the scale, UNITS_PER_MB units per planned megabyte. General's
    units come from the directory in every document, a department's from its own
    register, each spread over the documents in proportion to their planned size.
    """
    total = UNITS_PER_MB * sum(d["target_mb"] for d in docs)
    units = Counter(s["label"] for d in docs for s in d["sections"])
    for d in docs:
        for _kind, audience, rows in small_tables(d, w):
            units[audience] += units_of(rows)
    out = {d["id"]: {"register": 0, "directory": 0} for d in docs}

    def spread(rows: float, group: list[dict], table: str) -> None:
        weight = sum(d["target_mb"] for d in group)
        for d in group:
            out[d["id"]][table] = round(rows * d["target_mb"] / weight)

    spread(max(0.0, shares["general"] * total - units["general"]) * ROWS_PER_UNIT, docs, "directory")
    for label, share in shares.items():
        owned = [d for d in docs if d["owner"] == label]
        if label != "general" and owned:
            spread(max(0.0, share * total - units[label]) * ROWS_PER_UNIT, owned, "register")
    return out


def header_lines(lang: str, number: int, title: str, label: str) -> list[str]:
    if lang == "el":
        return [f"ΕΝΟΤΗΤΑ {number}  {title}", f"Κοινό: {LABEL_EL[label]}", "Διαβάθμιση: Εσωτερικό"]
    return [f"SECTION {number}  {title}", f"Audience: {LABEL_EN[label]}", "Classification: Internal"]


def parse_sections(text: str) -> list[tuple[list[str], str]]:
    """The writer's sections as (three header lines, body)."""
    lines = text.replace("\r\n", "\n").split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith(("SECTION ", "ΕΝΟΤΗΤΑ "))]
    out = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        out.append((lines[start:start + 3], "\n".join(lines[start + 3:end]).strip()))
    return out


def render(fmt: str, title: str, meta_line: str, sections: list[tuple[list[str], str, list[str] | None]]) -> str:
    """The file in its format. A section with columns is a table: its body holds one row per line, cells split by " | "."""
    if fmt == "txt":
        parts = [title, meta_line, ""]
        for head, body, columns in sections:
            parts += head + [""] + ([" | ".join(columns)] if columns else []) + [body, ""]
        return "\n".join(parts)
    if fmt == "md":
        parts = [f"# {title}", "", meta_line, ""]
        for head, body, columns in sections:
            parts += [f"## {head[0]}", "", head[1], head[2], ""]
            if columns:
                parts += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
                parts += [f"| {line} |" for line in body.split("\n") if line]
            else:
                parts += [body]
            parts += [""]
        return "\n".join(parts)
    parts = ["<!doctype html>", "<html><head><meta charset=\"utf-8\"><title>" + html.escape(title) + "</title></head><body>",
             f"<h1>{html.escape(title)}</h1>", f"<p>{html.escape(meta_line)}</p>"]
    for head, body, columns in sections:
        audience = head[1].split(":", 1)[1].strip()
        parts.append(f"<section data-audience=\"{html.escape(audience)}\"><h2>{html.escape(head[0])}</h2>")
        parts.append(f"<p class=\"audience\">{html.escape(head[1])}</p><p class=\"classification\">{html.escape(head[2])}</p>")
        if columns:
            parts.append("<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr>"
                         + "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in line.split(" | ")) + "</tr>"
                                   for line in body.split("\n") if line) + "</table>")
        else:
            parts += [f"<p>{html.escape(p)}</p>" for p in body.split("\n\n") if p.strip()]
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts)


def assemble(profile: str = "full") -> dict:
    """A fresh build of the whole corpus, swapped in only once every document is written (OP-F1).

    Nothing is built while any document lacks its narrative, so the corpus in place and
    its journal stay as they were (MT-F1). The files go to a sibling folder first, so a
    build killed halfway leaves the previous corpus and its journal as they were. The
    swap is three renames, and a kill between two of them is finished or undone by
    `settle` before the next build.
    """
    base = data_dir() / "corpus"
    docs = json.loads((base / "plan" / "catalogue.json").read_text(encoding="utf-8"))
    shares = {label: entry["target_share"] for label, entry in load_labels().items()}
    narrative = narrative_dir(base)
    missing = [doc["id"] for doc in docs if not (narrative / f"{doc['id']}.txt").is_file()]
    report = {"documents": len(docs), "written": 0, "missing": missing, "bytes": 0, "units": Counter(), "registered": shares}
    if missing:
        return report
    w = world()
    place = places(base, profile)
    budget = budgets(docs, shares, w)
    legacy = legacy_id(docs)
    with exclusive(base):
        settle(place)
        place["building"].mkdir(parents=True)
        for doc in docs:
            source = narrative / f"{doc['id']}.txt"
            size, units = assemble_doc(doc, source, w, place["building"], legacy, budget[doc["id"]])
            report["bytes"] += size
            report["units"] += units
            report["written"] += 1
        # The old corpus goes aside with its journal, since the new one has had no changes yet.
        if place["corpus"].exists():
            place["corpus"].rename(place["retired"])
        if place["journal"].exists():
            place["journal"].rename(place["retired journal"])
        place["building"].rename(place["corpus"])
        settle(place)
    return report


def places(base: Path, profile: str) -> dict[str, Path]:
    return {"corpus": base / profile, "journal": base / f"{profile}.journal.jsonl", "building": base / f"{profile}.building",
            "retired": base / f"{profile}.retired", "retired journal": base / f"{profile}.retired.journal.jsonl"}


def settle(place: dict[str, Path]) -> None:
    """Make the corpus and its journal agree again after a kill at any step of a build or a swap.

    With no corpus in place the swap stopped halfway, and the old corpus comes back,
    its journal first. Whatever is then left aside, and any unfinished build, is deleted.
    """
    if not place["corpus"].exists() and place["retired"].exists():
        if place["retired journal"].exists():
            place["retired journal"].replace(place["journal"])
        place["retired"].rename(place["corpus"])
    if place["retired"].exists():
        shutil.rmtree(place["retired"])
    place["retired journal"].unlink(missing_ok=True)
    if place["building"].exists():
        shutil.rmtree(place["building"])


def legacy_id(docs: list[dict]) -> str | None:
    """One Greek text file keeps the legacy Windows-1253 encoding, when the catalogue has one."""
    return next((d["id"] for d in docs if d["language"] == "el" and d["format"] == "txt"), None)


def assemble_doc(doc: dict, source: Path, w: dict, out: Path, legacy: str | None,
                 budget: dict[str, int]) -> tuple[int, Counter]:
    """One document and its sidecar into `out`; returns the bytes written and the units by label."""
    lang, rng = doc["language"], random.Random(f"{SEED}:{doc['id']}")
    sections = [(head, body, None) for head, body in parse_sections(source.read_text(encoding="utf-8"))]
    units = Counter(s["label"] for s in doc["sections"])
    owner_name = LABEL_EL[doc["owner"]] if lang == "el" else LABEL_EN[doc["owner"]]
    title = f"{doc['type_name']}: {owner_name}"
    meta_line = (f"Έγγραφο {doc['id']}, έκδοση {doc['version']}, η Εταιρεία, εσωτερική χρήση" if lang == "el"
                 else f"Document {doc['id']}, version {doc['version']}, the Company, internal use")
    tables = [(doc["owner"], doc["owner"], register_rows(doc["owner"], lang, rng, w, budget["register"])),
              ("directory", "general", directory_rows(lang, rng, w, budget["directory"]))] + small_tables(doc, w)
    prefix = "Παράρτημα" if lang == "el" else "Annex"
    for letter, (kind, audience, rows) in enumerate(t for t in tables if t[2]):
        en, el, columns_en, columns_el = TITLES[kind]
        head = header_lines(lang, len(sections) + 1, f"{prefix} {chr(65 + letter)}: {el if lang == 'el' else en}", audience)
        sections.append((head, "\n".join(rows), columns_el if lang == "el" else columns_en))
        units[audience] += units_of(rows)
    text = render(doc["format"], title, meta_line, sections)
    encoding = "windows-1253" if doc["id"] == legacy else "utf-8"
    path = out / f"{doc['id']}.{doc['format']}"
    try:
        data = text.encode("cp1253" if encoding == "windows-1253" else "utf-8")
    except UnicodeEncodeError as error:  # never a silent "?" in the legacy file (DR-F4)
        raise ValueError(f"{doc['id']}: {error.object[error.start]!r} at {error.start} has no place in {encoding}") from None
    path.write_bytes(data)
    meta = {"doc_id": doc["id"], "title": title, "owner": doc["owner"], "language": lang, "version": doc["version"],
            "type": doc["type"], "default_audience": doc["owner"], "classification": "internal",
            "format": doc["format"], "encoding": encoding}
    dump_json(out / f"{doc['id']}.meta.json", meta)
    return path.stat().st_size, units


def sample_ids(docs: list[dict]) -> list[str]:
    """Six documents a stranger can open: each format in each language, mixing departments where one does,
    and the Greek text file in its legacy encoding."""
    def mixed(d: dict) -> bool:
        return len({s["label"] for s in d["sections"]} - {"general"}) > 1
    by_id = {d["id"]: d for d in docs}
    chosen = [legacy_id(docs)]
    owners = {by_id[chosen[0]]["owner"]} if chosen[0] else set()
    for lang, fmt in [("en", "txt"), ("en", "md"), ("el", "md"), ("en", "html"), ("el", "html")]:
        # A department not shown yet comes first, so the six samples show six owners where they can.
        pool = sorted((d for d in docs if (d["language"], d["format"]) == (lang, fmt)),
                      key=lambda d: (d["owner"] in owners, not mixed(d), d["id"]))
        chosen.append(pool[0]["id"])
        owners.add(pool[0]["owner"])
    return [c for c in chosen if c]


def samples(out: Path) -> list[Path]:
    """The six sample files, small: the writers' full text, then each table cut to a page of rows."""
    base = data_dir() / "corpus"
    docs = json.loads((base / "plan" / "catalogue.json").read_text(encoding="utf-8"))
    w, by_id = world(), {d["id"]: d for d in docs}
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*"):
        if stale.name.split(".", 1)[0] in by_id:  # a file of a document, never one added by hand (MT-F4)
            stale.unlink()
    legacy = legacy_id(docs)
    for doc_id in sample_ids(docs):
        rng = random.Random(f"{SEED}:{doc_id}:sample")
        page = {"register": rng.randint(12, 24), "directory": rng.randint(12, 24)}
        assemble_doc(by_id[doc_id], narrative_dir(base) / f"{doc_id}.txt", w, out, legacy, page)
    return sorted(out.iterdir())
