"""Turn the writers' narrative into the corpus: a title block, the sections, and annexes up to each file's size.

Annexes are the structured part real manuals are mostly made of: directories, stock
lists, revision histories, form definitions, glossaries. They are generated from the
world file and a seed per document, carry no policy value, so they can never
contradict a fact, and sit under the owner's audience like the rest of the file.
"""

from __future__ import annotations

import html
import json
import random
import shutil
from pathlib import Path

from needtoknow.corpus.plan import LABEL_EL, LABEL_EN, SEED, world
from needtoknow.settings import data_dir

ANNEXES = {
    "en": [("Shop directory", "Shop {code} | {region} | unit {unit} | managed by the {role} | extension {ext}"),
           ("Stock list", "SKU {sku} | {product} | bin {bin} | {stock} units on hand | reorder flag {flag}"),
           ("Revision history", "Revision {rev} | {date} | {role} | {change}"),
           ("Form definitions", "Form {form} | field {field} | {need} | entered in {system}"),
           ("Glossary", "{term}: {meaning}")],
    "el": [("Κατάλογος καταστημάτων", "Κατάστημα {code} | {region} | μονάδα {unit} | υπεύθυνος: {role} | εσωτερικό {ext}"),
           ("Κατάσταση αποθέματος", "Κωδικός {sku} | {product} | θέση {bin} | {stock} τεμάχια σε απόθεμα | σήμανση αναπαραγγελίας {flag}"),
           ("Ιστορικό αναθεωρήσεων", "Αναθεώρηση {rev} | {date} | {role} | {change}"),
           ("Ορισμοί εντύπων", "Έντυπο {form} | πεδίο {field} | {need} | καταχωρείται στο {system}"),
           ("Γλωσσάριο", "{term}: {meaning}")],
}
ROLES = {"en": ["shop manager", "team leader", "regional manager", "process owner", "quality reviewer"],
         "el": ["διευθυντής καταστήματος", "επικεφαλής ομάδας", "περιφερειακός διευθυντής", "υπεύθυνος διαδικασίας", "ελεγκτής ποιότητας"]}
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


def annex_row(kind: int, lang: str, rng: random.Random, w: dict, sections: int) -> str:
    template = ANNEXES[lang][kind][1]
    products = [p[lang] for p in w["products"]]
    return template.format(
        code=f"{rng.choice('ABCDEFGHJKLMNP')}{rng.choice('ABCDEFGHJKLMNP')}{rng.randint(100, 999)}",
        region=rng.choice(w["regions"][lang]), unit=rng.randint(1, 240), role=rng.choice(ROLES[lang]),
        ext=rng.randint(30000, 89999), sku=f"{rng.choice('DVAXTRS')}{rng.choice('DVAXTRS')}{rng.randint(10000, 99999)}",
        product=rng.choice(products), bin=f"{rng.choice('ABCDEFGH')}{rng.randint(1, 60):02d}", stock=rng.randint(0, 900),
        flag=rng.choice(["Y", "N"]), rev=f"{rng.randint(1, 4)}.{rng.randint(0, 40)}",
        date=f"20{rng.randint(23, 26)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        change=rng.choice(CHANGES[lang]).format(s=rng.randint(1, max(1, sections))),
        form=f"F{rng.randint(10, 99)}", field=rng.choice(FIELDS[lang]), need=rng.choice(NEED[lang]),
        system=rng.choice(list(w["systems"].values())),
        term=(t := rng.choice(TERMS[lang]))[0], meaning=t[1])


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


def render(fmt: str, title: str, meta_line: str, sections: list[tuple[list[str], str, bool]]) -> str:
    if fmt == "txt":
        parts = [title, meta_line, ""]
        for head, body, _table in sections:
            parts += head + ["", body, ""]
        return "\n".join(parts)
    if fmt == "md":
        parts = [f"# {title}", "", meta_line, ""]
        for head, body, table in sections:
            parts += [f"## {head[0]}", "", head[1], head[2], ""]
            if table:
                rows = [line.split(" | ") for line in body.split("\n") if line]
                width = max(len(r) for r in rows)
                parts += ["| " + " | ".join(f"c{i + 1}" for i in range(width)) + " |", "|" + "---|" * width]
                parts += ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in rows]
            else:
                parts += [body]
            parts += [""]
        return "\n".join(parts)
    parts = ["<!doctype html>", "<html><head><meta charset=\"utf-8\"><title>" + html.escape(title) + "</title></head><body>",
             f"<h1>{html.escape(title)}</h1>", f"<p>{html.escape(meta_line)}</p>"]
    for head, body, table in sections:
        audience = head[1].split(":", 1)[1].strip()
        parts.append(f"<section data-audience=\"{html.escape(audience)}\"><h2>{html.escape(head[0])}</h2>")
        parts.append(f"<p class=\"audience\">{html.escape(head[1])}</p><p class=\"classification\">{html.escape(head[2])}</p>")
        if table:
            parts.append("<table>" + "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in line.split(" | ")) + "</tr>"
                                             for line in body.split("\n") if line) + "</table>")
        else:
            parts += [f"<p>{html.escape(p)}</p>" for p in body.split("\n\n") if p.strip()]
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts)


def assemble(profile: str = "full") -> dict:
    base = data_dir() / "corpus"
    docs = json.loads((base / "plan" / "catalogue.json").read_text(encoding="utf-8"))
    w = world()
    out = base / profile
    if out.exists():  # a fresh build: no file a mutation moved or added survives it
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (base / f"{profile}.journal.jsonl").unlink(missing_ok=True)
    report = {"written": 0, "missing": [], "bytes": 0}
    legacy = legacy_id(docs)
    for doc in docs:
        source = base / "narrative" / f"{doc['id']}.txt"
        if not source.is_file():
            report["missing"].append(doc["id"])
            continue
        report["bytes"] += assemble_doc(doc, source, w, out, legacy)
        report["written"] += 1
    return report


def legacy_id(docs: list[dict]) -> str | None:
    """One Greek text file keeps the legacy Windows-1253 encoding, when the catalogue has one."""
    return next((d["id"] for d in docs if d["language"] == "el" and d["format"] == "txt"), None)


def assemble_doc(doc: dict, source: Path, w: dict, out: Path, legacy: str | None,
                 cap_bytes: int | None = None, rows: tuple[int, int] = (400, 1600)) -> int:
    """One document and its sidecar into `out`; returns the bytes written."""
    lang, rng = doc["language"], random.Random(f"{SEED}:{doc['id']}")
    sections = [(head, body, False) for head, body in parse_sections(source.read_text(encoding="utf-8"))]
    owner_name = LABEL_EL[doc["owner"]] if lang == "el" else LABEL_EN[doc["owner"]]
    title = f"{doc['type_name']}: {owner_name}"
    meta_line = (f"Έγγραφο {doc['id']}, έκδοση {doc['version']}, η Εταιρεία, εσωτερική χρήση" if lang == "el"
                 else f"Document {doc['id']}, version {doc['version']}, the Company, internal use")
    target = int(doc["target_mb"] * 1024 * 1024)
    if cap_bytes is not None:
        target = min(target, cap_bytes)
    size = len(render(doc["format"], title, meta_line, sections).encode("utf-8"))
    number, letter = len(sections), 0
    while size < target:
        kind = letter % len(ANNEXES[lang])
        name = ANNEXES[lang][kind][0]
        prefix = "Παράρτημα" if lang == "el" else "Annex"
        lines = [annex_row(kind, lang, rng, w, len(sections)) for _ in range(rng.randint(*rows))]
        number += 1
        head = header_lines(lang, number, f"{prefix} {chr(65 + letter % 26)}{letter // 26 or ''}: {name}", doc["owner"])
        body = "\n".join(lines)
        sections.append((head, body, True))
        size += len(body.encode("utf-8")) + 200
        letter += 1
    text = render(doc["format"], title, meta_line, sections)
    encoding = "windows-1253" if doc["id"] == legacy else "utf-8"
    path = out / f"{doc['id']}.{doc['format']}"
    path.write_bytes(text.encode("cp1253" if encoding == "windows-1253" else "utf-8", errors="replace"))
    meta = {"doc_id": doc["id"], "title": title, "owner": doc["owner"], "language": lang, "version": doc["version"],
            "type": doc["type"], "default_audience": doc["owner"], "classification": "internal",
            "format": doc["format"], "encoding": encoding}
    (out / f"{doc['id']}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return path.stat().st_size


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


def samples(out: Path, cap_kb: int = 40) -> list[Path]:
    """The six sample files, small: the writers' full text, then a few annex rows up to the cap."""
    base = data_dir() / "corpus"
    docs = json.loads((base / "plan" / "catalogue.json").read_text(encoding="utf-8"))
    w, by_id = world(), {d["id"]: d for d in docs}
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*"):
        stale.unlink()
    legacy = legacy_id(docs)
    for doc_id in sample_ids(docs):
        assemble_doc(by_id[doc_id], base / "narrative" / f"{doc_id}.txt", w, out, legacy, cap_kb * 1024, rows=(12, 24))
    return sorted(out.iterdir())
