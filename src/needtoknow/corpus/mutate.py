"""Realistic changes to the assembled corpus, each written to a journal with the effect it must have.

`ntk corpus mutate --count K --seed N` edits files under NTK_DATA_DIR/corpus/full in
place and appends to corpus/full.journal.jsonl. The change bench later replays the
journal against the index and compares what happened with what the journal says.
`ntk corpus assemble` rebuilds the untouched corpus and clears the journal.
"""

from __future__ import annotations

import html
import json
import random
import re
from pathlib import Path

from needtoknow.corpus import dump_json
from needtoknow.corpus.plan import LABEL_EL, LABEL_EN, MIXES
from needtoknow.settings import data_dir

KINDS = ["value_update", "insert_paragraph", "relabel", "rename", "re_encode",
         "annex_refresh", "delete_paragraph", "new_file", "delete_file"]
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
    codec = {"windows-1253": "cp1253", "utf-8-sig": "utf-8-sig"}.get(meta["encoding"], "utf-8")
    return file.read_bytes().decode(codec)


def write(file: Path, meta: dict, text: str) -> None:
    codec = {"windows-1253": "cp1253", "utf-8-sig": "utf-8-sig"}.get(meta["encoding"], "utf-8")
    file.write_bytes(text.encode(codec))


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


def mutate(count: int, seed: int = 1) -> list[dict]:
    base = data_dir() / "corpus"
    full = base / "full"
    facts = json.loads((base / "plan" / "facts.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)
    journal = base / "full.journal.jsonl"
    done = sum(1 for _ in journal.open(encoding="utf-8")) if journal.is_file() else 0
    entries = []
    for k in range(count):
        kind = KINDS[(done + k) % len(KINDS)]
        files = corpus_files(full)
        doc_id = rng.choice(sorted(files))
        entry = apply(kind, doc_id, files[doc_id], facts, rng, full, done + k + 1)
        # "kind" is what was done; "requested" differs when a document could not take the change asked for.
        entry.update(seq=done + k + 1, seed=seed, requested=kind)
        entries.append(entry)
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entries


def apply(kind: str, doc_id: str, found: tuple[Path, Path, dict], facts: list[dict], rng: random.Random, full: Path, seq: int) -> dict:
    file, meta_path, meta = found
    fmt, lang = meta["format"], meta["language"]
    text = read(file, meta)
    rel = lambda p: str(p.relative_to(full)).replace("\\", "/")
    entry = {"doc_id": doc_id, "file": rel(file), "kind": kind}
    if kind == "value_update":
        here = [f for f in facts if doc_id in f.get("placements", []) and f["values"][0].isdigit()
                and f["kind"] != "superseded" and as_written(f[lang], fmt) in text]
        if not here:
            return apply("insert_paragraph", doc_id, found, facts, rng, full, seq) | {"note": "no numeric fact here, a paragraph instead"}
        f = rng.choice(here)
        old_value = f["values"][0]
        new_value = str(int(old_value) + (1 if int(old_value) < 10 else max(1, round(int(old_value) * 0.2))))
        new_sentence = replace_value(f[lang], old_value, new_value)
        write(file, meta, text.replace(as_written(f[lang], fmt), as_written(new_sentence, fmt), 1))
        entry.update(fact=f["id"], old_value=old_value, new_value=new_value, sentence=new_sentence,
                     expected="only the units holding the sentence are embedded again; the fact now answers the new value")
    elif kind in ("insert_paragraph", "delete_paragraph"):
        blocks = [b for b in paragraphs(text, fmt) if not any(as_written(f[lang], fmt) in b for f in facts) and "CANARY_" not in b]
        if not blocks:  # a short new file, say: change its encoding instead
            return apply("re_encode", doc_id, found, facts, rng, full, seq) | {"note": "no free paragraph here, a re encode instead"}
        target = rng.choice(blocks)
        if kind == "insert_paragraph":
            extra = as_written(EXTRA[lang], fmt)
            addition = f"{target}</p>\n<p>{extra}" if fmt == "html" else f"{target}\n\n{extra}"
            write(file, meta, text.replace(target, addition, 1))
            entry.update(paragraph=EXTRA[lang], expected="one new unit in that section; no other unit changes")
        else:
            removed = text.replace(f"<p>{target}</p>\n", "", 1) if fmt == "html" else text.replace(f"{target}\n\n", "", 1)
            if removed == text:  # the last paragraph of a file has no blank line after it
                removed = text.replace(f"\n\n{target}", "", 1)
            assert removed != text, f"{doc_id}: the paragraph to delete was not found"
            write(file, meta, removed)
            entry.update(paragraph_start=target[:80], expected="that paragraph's units are deleted; no other unit changes")
    elif kind == "relabel":
        line_prefix = "Κοινό: " if lang == "el" else "Audience: "
        names = LABEL_EL if lang == "el" else LABEL_EN
        end = "</p>" if fmt == "html" else "\n"
        candidates = [(label, name) for label, name in names.items() if label in MIXES and f"{line_prefix}{name}{end}" in text]
        if not candidates:
            return apply("rename", doc_id, found, facts, rng, full, seq) | {"note": "no section to relabel, a rename instead"}
        label, name = rng.choice(candidates)
        added = names[MIXES[label]]
        changed = text.replace(f"{line_prefix}{name}{end}", f"{line_prefix}{name}; {added}{end}", 1)
        if fmt == "html":
            changed = changed.replace(f'data-audience="{html.escape(name)}"', f'data-audience="{html.escape(name)}; {html.escape(added)}"', 1)
        write(file, meta, changed)
        entry.update(from_labels=[label], to_labels=[label, MIXES[label]],
                     expected="no unit is embedded again; one section's labels change in the payload, and its new readers see it at the next query")
    elif kind == "rename":
        target_dir = full / "moved"
        target_dir.mkdir(exist_ok=True)
        new_file = target_dir / f"{doc_id}_{seq}.{fmt}"
        file.rename(new_file)
        meta_path.rename(new_file.with_name(f"{doc_id}_{seq}.meta.json"))
        entry.update(file=rel(new_file), before_file=rel(file), expected="no unit is embedded again; the document's path is updated")
    elif kind == "re_encode":
        new_encoding = "utf-8" if meta["encoding"] != "utf-8" else ("windows-1253" if lang == "el" else "utf-8-sig")
        try:
            text.encode("cp1253" if new_encoding == "windows-1253" else "utf-8")
        except UnicodeEncodeError:
            new_encoding = "utf-8-sig"
        meta["encoding"] = new_encoding
        write(file, meta, text)
        dump_json(meta_path, meta)
        entry.update(encoding=new_encoding, expected="nothing changes: the decoded text is the same")
    elif kind == "annex_refresh":
        rows = [m for m in re.finditer(r"(?m)^.*\| (\d+) (units on hand|τεμάχια σε απόθεμα) \|.*$", text)]
        if not rows:
            return apply("re_encode", doc_id, found, facts, rng, full, seq) | {"note": "no stock list here, a re encode instead"}
        chosen = rng.sample(rows, min(len(rows), max(1, len(rows) // 20)))
        changed = text
        for m in sorted(chosen, key=lambda m: m.start(), reverse=True):
            line = m.group(0).replace(f"| {m.group(1)} {m.group(2)}", f"| {rng.randint(0, 900)} {m.group(2)}", 1)
            changed = changed[: m.start()] + line + changed[m.end():]
        write(file, meta, changed)
        entry.update(rows_changed=len(chosen), expected="only the row groups holding the changed rows are embedded again")
    elif kind == "new_file":
        new_id = f"GEN_NEW_{seq:03d}"
        title, body = NEW_FILE[lang]
        head = ([f"ΕΝΟΤΗΤΑ 1  Ανακοίνωση", f"Κοινό: {LABEL_EL['general']}", "Διαβάθμιση: Εσωτερικό"] if lang == "el"
                else ["SECTION 1  Notice", f"Audience: {LABEL_EN['general']}", "Classification: Internal"])
        new_file = full / f"{new_id}.txt"
        new_file.write_bytes("\n".join([title, new_id, ""] + head + ["", body, ""]).encode("utf-8"))
        new_meta = {"doc_id": new_id, "title": title, "owner": "general", "language": lang, "version": "1.0", "type": "CIR",
                    "default_audience": "general", "classification": "internal", "format": "txt", "encoding": "utf-8"}
        dump_json(full / f"{new_id}.meta.json", new_meta)
        entry.update(doc_id=new_id, file=rel(new_file), expected="its units are added; nothing else changes")
    elif kind == "delete_file":
        lost = [f["id"] for f in facts if f.get("placements") == [doc_id]]
        file.unlink()
        meta_path.unlink()
        entry.update(facts_lost=lost, expected="all its units leave both indexes; facts planted only here can no longer be answered")
    return entry
