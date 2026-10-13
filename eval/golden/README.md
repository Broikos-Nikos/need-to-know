# The golden items

`test.jsonl` and `dev.jsonl` hold the questions the answers are scored on, one JSON object
a line, in the format `needtoknow.eval.golden` describes. `eval/protocol.toml` version 6
fixes how many of each kind every split holds, and `uv run ntk eval check` holds each item
to its category's rules and each split to those counts before anything is scored.

## The test set

35 items, written once and never tuned on: the test split decides nothing and runs at the
baseline and at each release.

* 29 items over the 31 test families of `eval/families.json`, each family asked once: 6 in
  the asker's own label in English, 5 in Greek, 3 across languages, 4 from a file that also
  holds sections the asker may not read, 4 that the asker may not read at all and must be
  refused, 4 where a value changed and the current one is asked, 1 on the right regional
  variant, one asked by the administrator across two departments, and one that needs two
  documents.
* 6 items that belong to no family: 2 cells of table rows, 2 facts the corpus does not hold,
  and 2 near misses, absent but with a present fact beside them, scored like the absent ones.
* 16 of the 35 are in Greek.

## How they were written, and what that leaves

* From the plan's registry of facts, by the project's loop, not by a model under test.
* The protocol registers that each item is drafted from the registry and rewritten without
  the planted sentence in view. The test items were drafted with the planted sentences in
  view; what holds them instead is the cap the protocol also registers and `ntk eval check`
  enforces on every item: no question holds more than one of its planted sentence's three
  rarest terms, counted the way the lexical search counts them. The dev items are drafted
  from the registry's topic and value alone.
* Every value that changed did so on one date, 1 July 2025, its former value dating from 1
  March 2024. A superseded item therefore tests choosing the later of two dated values, not
  reasoning over many dates.
* A value a model may know without the documents, such as the 14 days of withdrawal from a
  distance contract, is asked only by an item whose right answer is a refusal.
* The documents are synthetic, and the facts sit early in short paragraphs about one thing,
  so a retrieval figure measured here is an upper bound for real manuals.
