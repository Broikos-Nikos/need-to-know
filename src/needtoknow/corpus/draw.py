"""Another draw of the corpus (T5.5e, RS-F6): the same documents, sections and writers' prose, with the facts'
values and the annexes drawn again from a seed, so a retrieval figure can be read across draws of the generator and
not only on the one corpus every other number comes from.

`ntk corpus draw --seed S --every N` makes `NTK_DATA_DIR/draws/S/` a data folder of its own: every Nth document of
the catalogue in its `corpus/full/`, each with its sidecar, the narrative they were assembled from in
`corpus/narrative/`, and the draw's plan in `corpus/plan/`, the catalogue cut to those documents and the facts they
plant, each with the value it holds there and its placements cut to them (DR-F29). With NTK_DATA_DIR set to it, the
ingest, the embedder and the probes read the draw as they read the corpus; the index commands do not, since they
reach the one server the config names whatever the data folder, and a sync from a draw would delete the served
index's points, which the sync refuses without --force (MT-F18): `bench/draws.py` builds each draw's own index, in
Qdrant's local mode. A single fact or a regional variant takes
a new value from its own template under the seed, written into every sentence that plants it; a versioned fact and
the one it superseded keep theirs, since their pair is what their items ask about. The annexes, each document's
register, directory and small tables, are assembled under the seed, sized as in the whole corpus. The base seed keeps
every value and every annex, so its draw is the corpus's own documents, byte for byte.

What a draw reports is counted over its own documents: how many facts they plant, how many of those took a new value,
and how many planted sentences were written again.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from needtoknow.corpus import dump_json, narrative_dir, read_plan
from needtoknow.corpus.assemble import assemble_doc, budgets, legacy_id
from needtoknow.corpus.mutate import as_written
from needtoknow.corpus.plan import SEED, TEMPLATES, num, world
from needtoknow.policy import load_labels
from needtoknow.settings import data_dir


class DrawError(RuntimeError):
    """A draw that could not write a fact's new value where the plan put it."""


def swap(sentence: str, old: str, new: str) -> str:
    """The value as a whole token: no letter, digit, point or comma before it, and no digit after it or after a
    following point or comma, so the 7 of 7.5 or of 17 is never taken for it, while a full stop or a Greek ordinal's
    ending, 17η, may follow."""
    return re.sub(rf"(?<![\w.,]){re.escape(old)}(?!\d|[.,]\d)", lambda _: new, sentence, count=1)


def redraw(facts: list[dict], seed: int) -> list[dict]:
    """The facts with each single fact's and regional variant's value drawn again from its template under the seed."""
    if seed == SEED:
        return [dict(f) for f in facts]
    out = []
    for f in facts:
        g = dict(f)
        if f["kind"] in ("single", "variant"):
            values = TEMPLATES[int(f["family"][1:3])][5]
            raw = values(random.Random(f"{seed}:{f['id']}"), int(f["id"][4:6]))
            value = raw if isinstance(raw, tuple) else num(raw)
            g["values"] = list(value)
            for lang, old, new in (("en", f["values"][0], value[0]), ("el", f["values"][1], value[1])):
                g[lang] = swap(f[lang], old, new)
                if old != new and g[lang] == f[lang]:
                    raise DrawError(f"{f['id']}: {old!r} is not a whole token of its {lang} sentence")
        out.append(g)
    return out


def draw_dir(seed: int) -> Path:
    """The data folder a draw makes, which NTK_DATA_DIR names to work on it."""
    return data_dir() / "draws" / str(seed)


def draw(seed: int, every: int) -> dict:
    """Every `every`th document of the plan, its values and annexes drawn under `seed`; what was written."""
    if every < 1:  # 0 stopped in a traceback and a negative step read the catalogue from its end (DR-F29)
        raise DrawError(f"every {every}: a draw takes every Nth document of the catalogue, N at least 1")
    base = data_dir() / "corpus"
    catalogue = read_plan(base, "catalogue.json")
    facts = read_plan(base, "facts.json")
    drawn = redraw(facts, seed)
    docs = catalogue[::every]
    held = {doc["id"] for doc in docs}
    placed = [(f, g) for f, g in zip(facts, drawn) if held & set(f["placements"])]
    root = draw_dir(seed)
    out, plan, sources = (root / "corpus" / name for name in ("full", "plan", "narrative"))
    for folder in (out, plan, sources):
        folder.mkdir(parents=True, exist_ok=True)
    w = world()
    shares = {label: entry["target_share"] for label, entry in load_labels().items()}
    budget = budgets(catalogue, shares, w, seed)
    legacy = legacy_id(catalogue)
    narrative = narrative_dir(base)
    report = {"seed": seed, "documents": len(docs), "facts": len(placed),
              "values": sum(f["values"] != g["values"] for f, g in placed), "sentences": 0, "bytes": 0,
              "folder": root}
    for doc in docs:
        lang = doc["language"]
        text = (narrative / f"{doc['id']}.txt").read_text(encoding="utf-8")
        for f, g in placed:
            if doc["id"] in f["placements"] and f[lang] != g[lang]:
                if f[lang] not in text:
                    raise DrawError(f"{doc['id']}: the planted sentence of {f['id']} is not in its narrative")
                text = text.replace(f[lang], g[lang])
                report["sentences"] += 1
        source = sources / f"{doc['id']}.txt"
        source.write_bytes(text.encode("utf-8"))
        size, _ = assemble_doc(doc, source, w, out, legacy, budget[doc["id"]], seed)
        written = (out / f"{doc['id']}.{doc['format']}").read_bytes().decode(
            "cp1253" if doc["id"] == legacy else "utf-8")
        for _, g in placed:
            if doc["id"] in g["placements"] and as_written(g[lang], doc["format"]) not in written:
                raise DrawError(f"{doc['id']}: {g['id']}'s sentence is not in the assembled file as its format writes it")
        report["bytes"] += size
    dump_json(plan / "catalogue.json", docs)
    # The facts these documents plant, placed only in them, so the plan agrees with itself (DR-F29).
    dump_json(plan / "facts.json", [g | {"placements": [d for d in g["placements"] if d in held]} for _, g in placed])
    return report
