"""Realistic changes to the assembled corpus, each written to a journal with the effect it must have.

`ntk corpus mutate --count K --seed N` changes files under NTK_DATA_DIR/corpus/full and
appends to corpus/full.journal.jsonl. The change bench later replays the journal
against the index and compares what happened with what the journal says.
`ntk corpus assemble` rebuilds the untouched corpus and clears the journal.

A change is journaled before it is made (OP-F2). Its new bytes go to a staging
folder first; then its journal line, marked pending, names the moves and deletes
that make it; then those run; then a second line marks it applied. Every step can
run twice, so the next run finishes whatever a crash interrupted, and the corpus
never holds a change its journal does not name.
"""

from __future__ import annotations

import html
import json
import os
import random
import re
import shutil
from pathlib import Path

from needtoknow.corpus import exclusive, json_bytes
from needtoknow.corpus.plan import LABEL_EL, LABEL_EN, MIXES
from needtoknow.settings import data_dir

KINDS = ["value_update", "insert_paragraph", "relabel", "rename", "re_encode",
         "annex_refresh", "delete_paragraph", "new_file", "delete_file"]
CODECS = {"windows-1253": "cp1253", "utf-8-sig": "utf-8-sig"}
EXTRA = {"en": "Team leaders review this step at the weekly meeting and record any exception in Relay.",
         "el": "Οι επικεφαλής ομάδων εξετάζουν αυτό το βήμα στην εβδομαδιαία συνάντηση και καταγράφουν κάθε εξαίρεση στο Relay."}
NEW_FILE = {"en": ("Circular compilation: All staff", "This circular confirms that the procedures in force remain unchanged "
                   "and asks every team to read the latest versions in the document library."),
            "el": ("Συλλογή εγκυκλίων: Όλο το προσωπικό", "Η εγκύκλιος επιβεβαιώνει ότι οι διαδικασίες που ισχύουν δεν αλλάζουν "
                   "και ζητά από κάθε ομάδα να διαβάσει τις τελευταίες εκδόσεις στη βιβλιοθήκη εγγράφων.")}


def corpus_files(full: Path) -> dict[str, tuple[Path, Path, dict]]:
    """doc_id to (file, sidecar, sidecar contents), wherever the file now lives."""
    found = {}
    for meta_path in sorted(full.rglob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stem = meta_path.name[: -len(".meta.json")]
        file = meta_path.with_name(f"{stem}.{meta['format']}")
        if file.is_file():
            found[meta["doc_id"]] = (file, meta_path, meta)
    return found


def read(file: Path, meta: dict) -> str:
    return file.read_bytes().decode(CODECS.get(meta["encoding"], "utf-8"))


def encoded(meta: dict, text: str) -> bytes:
    return text.encode(CODECS.get(meta["encoding"], "utf-8"))


def as_written(text: str, fmt: str) -> str:
    return html.escape(text) if fmt == "html" else text


def replace_value(sentence: str, old: str, new: str) -> str:
    """The value as a whole token, so the 1 in "P1" or "C1" is never mistaken for it."""
    return re.sub(rf"(?<![\w.,]){re.escape(old)}(?![\w.,])", new, sentence, count=1)


def paragraphs(text: str, fmt: str) -> list[str]:
    """Narrative paragraphs: blocks of prose, never headers, audience lines or table rows."""
    if fmt == "html":
        return [m.group(1) for m in re.finditer(r"<p>(.*?)</p>", text) if len(m.group(1)) > 200]
    return [b for b in re.split(r"\n\s*\n", text) if len(b) > 200 and " | " not in b and not b.lstrip().startswith(("#", "SECTION", "ΕΝΟΤΗΤΑ"))]


def entries(journal: Path) -> list[dict]:
    """The journal's changes in order, each with its latest state.

    A last line a crash cut short is left out: as a pending line it named a change
    never begun, and as an applied line it leaves its change pending, to be finished.
    """
    if not journal.is_file():
        return []
    changes: dict[int, dict] = {}
    for line in journal.read_bytes().decode("utf-8").split("\n")[:-1]:
        if line:
            record = json.loads(line)
            changes.setdefault(record["seq"], {}).update(record)
    return [changes[seq] for seq in sorted(changes)]


def append(journal: Path, record: dict) -> None:
    with journal.open("ab") as handle:
        handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))


def mutate(count: int, seed: int = 1) -> list[dict]:
    base = data_dir() / "corpus"
    full, journal = base / "full", base / "full.journal.jsonl"
    facts = json.loads((base / "plan" / "facts.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)
    made = []
    with exclusive(base):
        recover(base)
        last = max((e["seq"] for e in entries(journal)), default=0)
        for seq in range(last + 1, last + count + 1):
            kind = KINDS[(seq - 1) % len(KINDS)]
            files = corpus_files(full)
            doc_id = rng.choice(sorted(files))
            entry, effects = plan(kind, doc_id, files[doc_id], facts, rng, full, seq)
            # "kind" is what was done; "requested" differs when a document could not take the change asked for.
            entry.update(seq=seq, seed=seed, requested=kind, ops=stage(base, seq, effects))
            append(journal, entry | {"state": "pending"})
            commit(base, entry["ops"])
            append(journal, {"seq": seq, "state": "applied"})
            made.append(entry | {"state": "applied"})
        if (base / "full.staging").exists():
            (base / "full.staging").rmdir()  # empty by now: every staged file was moved into place
    return made


def recover(base: Path) -> list[int]:
    """Finish what a crash interrupted, before anything else writes the corpus.

    A line cut short is cut off, a pending change is made from its journal line, and
    bytes staged for a change whose line was never written are deleted.
    """
    journal = base / "full.journal.jsonl"
    if journal.is_file():
        data = journal.read_bytes()
        if not data.endswith(b"\n"):
            with journal.open("r+b") as handle:
                handle.truncate(data.rfind(b"\n") + 1)
    finished = []
    for entry in entries(journal):
        if entry.get("state") == "pending":
            commit(base, entry["ops"])
            append(journal, {"seq": entry["seq"], "state": "applied", "finished_after_a_crash": True})
            finished.append(entry["seq"])
    if (base / "full.staging").exists():
        shutil.rmtree(base / "full.staging")
    return finished


def stage(base: Path, seq: int, effects: list[tuple]) -> list[list[str]]:
    """New bytes go to the staging folder, so the journal line names only moves and deletes."""
    staging = base / "full.staging"
    ops = []
    for n, (op, *paths) in enumerate(effects):
        if op == "write":
            target, data = paths
            staging.mkdir(exist_ok=True)
            staged = staging / f"{seq}.{n}"
            staged.write_bytes(data)
            paths = [staged, target]
            op = "move"
        ops.append([op, *(p.relative_to(base).as_posix() for p in paths)])
    return ops


def commit(base: Path, ops: list[list[str]]) -> None:
    """Moves and deletes, each safe to run again: a source already gone was moved before."""
    for op, *paths in ops:
        if op == "move":
            source, target = base / paths[0], base / paths[1]
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
        elif op == "delete":
            (base / paths[0]).unlink(missing_ok=True)
        else:
            raise ValueError(f"unknown journal op {op!r}")


def instead(kind: str, note: str, doc_id: str, found: tuple[Path, Path, dict], facts: list[dict],
            rng: random.Random, full: Path, seq: int) -> tuple[dict, list[tuple]]:
    entry, effects = plan(kind, doc_id, found, facts, rng, full, seq)
    return entry | {"note": note}, effects


def plan(kind: str, doc_id: str, found: tuple[Path, Path, dict], facts: list[dict], rng: random.Random,
         full: Path, seq: int) -> tuple[dict, list[tuple]]:
    """A change worked out in full and written nowhere yet: its journal entry, and the writes, moves and deletes that make it."""
    file, meta_path, meta = found
    fmt, lang = meta["format"], meta["language"]
    text = read(file, meta)
    rel = lambda p: str(p.relative_to(full)).replace("\\", "/")
    entry = {"doc_id": doc_id, "file": rel(file), "kind": kind}
    if kind == "value_update":
        here = [f for f in facts if doc_id in f.get("placements", []) and f["values"][0].isdigit()
                and f["kind"] != "superseded" and as_written(f[lang], fmt) in text]
        if not here:
            return instead("insert_paragraph", "no numeric fact here, a paragraph instead", doc_id, found, facts, rng, full, seq)
        f = rng.choice(here)
        old_value = f["values"][0]
        new_value = str(int(old_value) + (1 if int(old_value) < 10 else max(1, round(int(old_value) * 0.2))))
        new_sentence = replace_value(f[lang], old_value, new_value)
        entry.update(fact=f["id"], old_value=old_value, new_value=new_value, sentence=new_sentence,
                     expected="only the units holding the sentence are embedded again; the fact now answers the new value")
        return entry, [("write", file, encoded(meta, text.replace(as_written(f[lang], fmt), as_written(new_sentence, fmt), 1)))]
    if kind in ("insert_paragraph", "delete_paragraph"):
        blocks = [b for b in paragraphs(text, fmt) if not any(as_written(f[lang], fmt) in b for f in facts) and "CANARY_" not in b]
        if not blocks:  # a short new file, say: change its encoding instead
            return instead("re_encode", "no free paragraph here, a re encode instead", doc_id, found, facts, rng, full, seq)
        target = rng.choice(blocks)
        if kind == "insert_paragraph":
            extra = as_written(EXTRA[lang], fmt)
            changed = text.replace(target, f"{target}</p>\n<p>{extra}" if fmt == "html" else f"{target}\n\n{extra}", 1)
            entry.update(paragraph=EXTRA[lang], expected="one new unit in that section; no other unit changes")
            return entry, [("write", file, encoded(meta, changed))]
        changed = text.replace(f"<p>{target}</p>\n", "", 1) if fmt == "html" else text.replace(f"{target}\n\n", "", 1)
        if changed == text:  # the last paragraph of a file has no blank line after it
            changed = text.replace(f"\n\n{target}", "", 1)
        assert changed != text, f"{doc_id}: the paragraph to delete was not found"
        entry.update(paragraph_start=target[:80], expected="that paragraph's units are deleted; no other unit changes")
        return entry, [("write", file, encoded(meta, changed))]
    if kind == "relabel":
        line_prefix = "Κοινό: " if lang == "el" else "Audience: "
        names = LABEL_EL if lang == "el" else LABEL_EN
        end = "</p>" if fmt == "html" else "\n"
        candidates = [(label, name) for label, name in names.items() if label in MIXES and f"{line_prefix}{name}{end}" in text]
        if not candidates:
            return instead("rename", "no section to relabel, a rename instead", doc_id, found, facts, rng, full, seq)
        label, name = rng.choice(candidates)
        added = names[MIXES[label]]
        changed = text.replace(f"{line_prefix}{name}{end}", f"{line_prefix}{name}; {added}{end}", 1)
        if fmt == "html":
            changed = changed.replace(f'data-audience="{html.escape(name)}"', f'data-audience="{html.escape(name)}; {html.escape(added)}"', 1)
        entry.update(from_labels=[label], to_labels=[label, MIXES[label]],
                     expected="no unit is embedded again; one section's labels change in the payload, and its new readers see it at the next query")
        return entry, [("write", file, encoded(meta, changed))]
    if kind == "rename":
        new_file = full / "moved" / f"{doc_id}_{seq}.{fmt}"
        entry.update(file=rel(new_file), before_file=rel(file), expected="no unit is embedded again; the document's path is updated")
        return entry, [("move", file, new_file), ("move", meta_path, new_file.with_name(f"{doc_id}_{seq}.meta.json"))]
    if kind == "re_encode":
        new_encoding = "utf-8" if meta["encoding"] != "utf-8" else ("windows-1253" if lang == "el" else "utf-8-sig")
        try:
            text.encode("cp1253" if new_encoding == "windows-1253" else "utf-8")
        except UnicodeEncodeError:
            new_encoding = "utf-8-sig"
        new_meta = meta | {"encoding": new_encoding}
        entry.update(encoding=new_encoding, expected="nothing changes: the decoded text is the same")
        return entry, [("write", file, encoded(new_meta, text)), ("write", meta_path, json_bytes(new_meta))]
    if kind == "annex_refresh":
        rows = [m for m in re.finditer(r"(?m)^.*\| (\d+) (units on hand|τεμάχια σε απόθεμα) \|.*$", text)]
        if not rows:
            return instead("re_encode", "no stock list here, a re encode instead", doc_id, found, facts, rng, full, seq)
        chosen = rng.sample(rows, min(len(rows), max(1, len(rows) // 20)))
        changed = text
        for m in sorted(chosen, key=lambda m: m.start(), reverse=True):
            line = m.group(0).replace(f"| {m.group(1)} {m.group(2)}", f"| {rng.randint(0, 900)} {m.group(2)}", 1)
            changed = changed[: m.start()] + line + changed[m.end():]
        entry.update(rows_changed=len(chosen), expected="only the row groups holding the changed rows are embedded again")
        return entry, [("write", file, encoded(meta, changed))]
    if kind == "new_file":
        new_id = f"GEN_NEW_{seq:03d}"
        title, body = NEW_FILE[lang]
        head = ([f"ΕΝΟΤΗΤΑ 1  Ανακοίνωση", f"Κοινό: {LABEL_EL['general']}", "Διαβάθμιση: Εσωτερικό"] if lang == "el"
                else ["SECTION 1  Notice", f"Audience: {LABEL_EN['general']}", "Classification: Internal"])
        new_file = full / f"{new_id}.txt"
        new_meta = {"doc_id": new_id, "title": title, "owner": "general", "language": lang, "version": "1.0", "type": "CIR",
                    "default_audience": "general", "classification": "internal", "format": "txt", "encoding": "utf-8"}
        entry.update(doc_id=new_id, file=rel(new_file), expected="its units are added; nothing else changes")
        # The sidecar goes in last: a file is a document from the moment its sidecar describes it.
        return entry, [("write", new_file, "\n".join([title, new_id, ""] + head + ["", body, ""]).encode("utf-8")),
                       ("write", full / f"{new_id}.meta.json", json_bytes(new_meta))]
    if kind == "delete_file":
        entry.update(facts_lost=[f["id"] for f in facts if f.get("placements") == [doc_id]],
                     expected="all its units leave both indexes; facts planted only here can no longer be answered")
        return entry, [("delete", file), ("delete", meta_path)]
    raise ValueError(f"unknown change kind {kind!r}")
