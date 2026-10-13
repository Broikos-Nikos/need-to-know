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

## The dev set

153 items: two or three for each dev family an item can ask, in the test's category
shares, with 8 table cells, 8 absent facts and 8 near misses; every development choice and
experiment is made on it. Two dev families carry no item: each holds only a value a model
may know without the documents, under the label every role reads, so no item could ask it
(N58). Twenty items have a paraphrase twin: the same question for the same role and
language, worded away from the original, with no more of the planted sentence's rarest
terms and at most half of the original's words. The twenty were drawn by a seed in the
test's category shares, `uv run ntk eval twins` draws them again, and the gap between twin
and original is reported (N59).

## How they were written, and what that leaves

* From the plan's registry of facts, by the project's loop, not by a model under test.
* The protocol registers that each item is drafted from the registry and rewritten without
  the planted sentence in view. The test items were drafted with the planted sentences in
  view; what holds them instead is the cap the protocol also registers and `ntk eval check`
  enforces on every item: no question holds more than one of its planted sentence's three
  rarest terms, counted the way the lexical search counts them. The dev items are drafted
  from `uv run ntk eval draft --split dev`, which shows each planted sentence with its value
  and its three rarest terms blanked, lists those terms' stems so a question can name the
  subject, and says where each fact sits.
* A table item's row is looked up in the manifest by the check: its key in one unit of the
  whole corpus, a table of the named document under a label the asker reads, the cell under
  the named column the expected value. Only three registers key their rows that way, the
  ticket log, the stock list and the return authorisation log; the cost centres, sites,
  courses, contracts, shops and dated sales reuse a key across documents with other values,
  so a question on one has no single answer. The dev rows were drawn from those three by a
  fixed seed, four in each language, and `uv run ntk eval rows` draws them again.
* Every absent fact and near miss names in `probes.json` the regular expressions its
  absence rests on, in both languages, and the check confirms that no prose unit of the
  corpus matches one: a regenerated narrative that answered one in passing fails it. The
  narrative already mentions a courier C3 and gift card refunds, so neither is asked.
* Every value that changed did so on one date, 1 July 2025, its former value dating from 1
  March 2024. A superseded item therefore tests choosing the later of two dated values, not
  reasoning over many dates.
* A value a model may know without the documents, such as the 14 days of withdrawal from a
  distance contract, is asked only by an item whose right answer is a refusal.
* `seeds.md` holds ten questions written from public documents of the domain, EU consumer
  and electronic communications law and the Greek complaint procedure, beside what each
  document and this corpus say. They are not scored. Where the two meet they agree, save one
  planted value that breaks the EU's 24 month cap on a consumer's commitment; and the corpus
  says nothing of porting, which the law regulates in detail.
* The documents are synthetic, and the facts sit early in short paragraphs about one thing,
  so a retrieval figure measured here is an upper bound for real manuals.
