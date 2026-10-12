"""The lexical side (T3.2): a passage as BM25 terms, hashed into a sparse vector that Qdrant
scores with its IDF modifier.

The analyser, in this order: lower case; accents stripped and final sigma folded, so a
question typed without accents finds accented text (`03-retrieval-rag/10` section 5);
words as runs of letters and digits, so a code such as RMA133420 stays whole; the
stopwords of both languages dropped, Lucene's Greek list and its English one, because
half the corpus is in each and a function word of either would otherwise carry weight
(KB1-F15, `03-retrieval-rag/12` section 5); and each word stemmed by its own script, not
by its unit's language: Snowball's Greek stemmer for a word with a Greek letter, its
English stemmer for any other word of letters, none for a word with a digit (KB1-F15). A
stemmed BM25 is what made fusion pay in Greek in the base's M9c (same doc, section 5).

A unit is analysed as its passage, the same text the dense side embeds. A term's id is
four bytes of its blake2b hash, the u32 that Qdrant's sparse indices are; terms that
collide share one weight. A passage's value for a term is BM25's term frequency part,
tf (k1 + 1) / (tf + k1 (1 - b + b dl / avgdl)); Qdrant multiplies it by the term's IDF
over the collection, and a question sends 1 for each of its terms. avgdl is measured
once per index and frozen with the analyser's version (KB1-F15): vectors written on a
moving average would drift from every vector written before them. The version is this
file's own digest and the stemmer's release, not a number bumped by hand, so a changed
analyser can never write vectors beside its predecessor's (DR-F7's shape).

IDF counts restricted text too, a side channel stated rather than hidden (architecture
section 7, KB1-F4). The encoder reads one passage and no label, so moving the lexical
side to one collection per role view, should P6's attack work, changes nothing here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import snowballstemmer

from needtoknow.settings import data_dir

ANALYSER_VERSION = (hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
                    + " snowballstemmer " + importlib.metadata.version("snowballstemmer"))
K1 = 1.2
B = 0.75
WORD = re.compile(r"\w+")
# Combining marks, the accents NFD splits off Greek and Latin letters, and final sigma.
FOLD = {**dict.fromkeys(range(0x300, 0x370)), ord("ς"): "σ"}
# Lucene's English stop set, and its Greek list, which is already lower case, without
# accents and with σ for ς (lucene/analysis/common, EnglishAnalyzer and el/stopwords.txt).
STOPWORDS = frozenset("""
a an and are as at be but by for if in into is it no not of on or such that the their then there these they
this to was will with
ο η το οι τα του τησ των τον την και κι κ ειμαι εισαι ειναι ειμαστε ειστε στο στον στη στην μα αλλα απο για
προσ με σε ωσ παρα αντι κατα μετα θα να δε δεν μη μην επι ενω εαν αν τοτε που πωσ ποιοσ ποια ποιο ποιοι ποιεσ
ποιων ποιουσ αυτοσ αυτη αυτο αυτοι αυτων αυτουσ αυτεσ αυτα εκεινοσ εκεινη εκεινο εκεινοι εκεινεσ εκεινα εκεινων
εκεινουσ οπωσ ομωσ ισωσ οσο οτι
""".split())
GREEK = snowballstemmer.stemmer("greek")
ENGLISH = snowballstemmer.stemmer("english")


class LexicalError(RuntimeError):
    """Frozen statistics that are missing, or that another analyser wrote."""


@lru_cache(maxsize=1 << 18)
def stem(word: str) -> str:
    if not word.isalpha():
        return word
    if any("α" <= letter <= "ω" for letter in word):
        return GREEK.stemWord(word)
    return ENGLISH.stemWord(word)


def terms(text: str) -> list[str]:
    """The text's terms in order: folded, split, stopwords dropped, each word stemmed by its script."""
    folded = unicodedata.normalize("NFD", text.lower()).translate(FOLD)
    return [stem(word) for word in WORD.findall(folded) if word not in STOPWORDS]


@lru_cache(maxsize=1 << 20)
def term_id(term: str) -> int:
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest(), "little")


def document(text: str, avgdl: float) -> tuple[list[int], list[float]]:
    """A passage's sparse vector: sorted term ids and BM25's saturated, length normalised term frequency."""
    counts = Counter(terms(text))
    by_id: Counter = Counter()
    for term, tf in counts.items():
        by_id[term_id(term)] += tf
    norm = K1 * (1 - B + B * sum(counts.values()) / avgdl)
    indices = sorted(by_id)
    return indices, [by_id[i] * (K1 + 1) / (by_id[i] + norm) for i in indices]


def query(text: str) -> tuple[list[int], list[float]]:
    """A question's sparse vector: each of its terms once, at 1, for Qdrant to weigh by IDF."""
    indices = sorted({term_id(term) for term in terms(text)})
    return indices, [1.0] * len(indices)


def measure(passages) -> dict:
    """The statistics an index is built with: the analyser's version, k1, b, and the mean terms per passage."""
    count = total = 0
    for text in passages:
        count += 1
        total += len(terms(text))
    return {"analyser": ANALYSER_VERSION, "k1": K1, "b": B, "passages": count, "avgdl": total / count if count else 0.0}


def stats_path() -> Path:
    return data_dir() / "index" / "lexical.json"


def freeze(stats: dict, path: Path | None = None) -> None:
    path = path or stats_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(stats, indent=2) + "\n").encode("utf-8"))


def frozen(path: Path | None = None) -> dict:
    """The index's frozen statistics; refused when missing or written by another analyser, k1 or b."""
    path = path or stats_path()
    if not path.is_file():
        raise LexicalError(f"{path} does not exist: the index build measures and freezes the statistics")
    stats = json.loads(path.read_bytes().decode("utf-8"))
    mine = {"analyser": ANALYSER_VERSION, "k1": K1, "b": B}
    if {key: stats.get(key) for key in mine} != mine:
        raise LexicalError(f"{path} was written with {', '.join(f'{k} {stats.get(k)}' for k in mine)}; this code "
                           f"has {', '.join(f'{k} {v}' for k, v in mine.items())}: rebuild the sparse side")
    return stats
