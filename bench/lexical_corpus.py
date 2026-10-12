"""The lexical analyser over the whole corpus (T3.2): what it keeps, how fast it runs, and what its hashes collide.

    uv run python bench/lexical_corpus.py

Needs a manifest (`ntk ingest build`); no GPU. Reads every unit's passage and analyses it
twice: plain, lower case words and nothing else, as the baseline; then with the analyser.
Prints for each the seconds, the mean terms per passage and the distinct terms; for the
analyser also the share of words dropped as stopwords, the terms by the stemmer that made
them, the distinct term ids against the distinct terms, which counts hash collisions, and
the seconds to turn every passage into its sparse vector at the measured average length;
then RS-F3's question and planted sentence, word by word and term by term.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter

from needtoknow import lexical
from needtoknow.ingest.manifest import opened
from needtoknow.ingest.units import passage

QUESTION = "πόσες ημέρες δέχεστε επιστροφές συσκευών πελάτες"
PLANTED = "πελάτης μπορεί επιστρέψει συσκευή 30 ημερών"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    with opened() as db:
        passages = [passage(json.loads(path), text) for path, text in db.execute("SELECT path, text FROM units ORDER BY unit_id")]
    print(f"passages {len(passages)}, characters {sum(map(len, passages))}")

    began = time.perf_counter()
    plain = Counter()
    count = 0
    for text in passages:
        words = re.findall(r"\w+", text.lower())
        plain.update(words)
        count += len(words)
    print(f"plain words: {time.perf_counter() - began:.1f} s; mean {count / len(passages):.1f} a passage; distinct {len(plain)}")

    began = time.perf_counter()
    analysed = Counter()
    for text in passages:
        analysed.update(lexical.terms(text))
    seconds = time.perf_counter() - began
    total = sum(analysed.values())
    print(f"analyser terms: {seconds:.1f} s; mean {total / len(passages):.2f} a passage; distinct {len(analysed)}; "
          f"stopwords dropped {1 - total / count:.1%} of the words")
    scripts = Counter()
    for term, n in analysed.items():
        kind = "not stemmed, has a digit" if not term.isalpha() else \
            "greek" if any("α" <= c <= "ω" for c in term) else "english"
        scripts[kind] += n
    print("terms by stemmer: " + ", ".join(f"{kind} {n / total:.1%}" for kind, n in scripts.most_common()))
    ids = {lexical.term_id(term) for term in analysed}
    print(f"distinct term ids {len(ids)} for {len(analysed)} distinct terms: {len(analysed) - len(ids)} collisions")

    avgdl = total / len(passages)  # what lexical.measure gives, without analysing everything a third time
    began = time.perf_counter()
    nonzero = sum(len(lexical.document(text, avgdl)[0]) for text in passages)
    print(f"sparse vectors at avgdl {avgdl:.2f}: {time.perf_counter() - began:.1f} s; "
          f"mean {nonzero / len(passages):.1f} nonzero values")

    shared_words = set(QUESTION.split()) & set(PLANTED.split())
    shared_terms = set(lexical.terms(QUESTION)) & set(lexical.terms(PLANTED))
    print(f"RS-F3: words shared {len(shared_words)} of {len(QUESTION.split())}; terms shared {len(shared_terms)} of "
          f"{len(lexical.terms(QUESTION))}: {' '.join(sorted(shared_terms))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
