# The decisions

Code comments, docstrings and commit messages cite decisions like `N51`. These are the
ones cited in this repository, each as the project made it: what it decided, on what
evidence, and what would reverse it. The project's working record holds every decision
in full, with the sessions around them; this page is its public part. A decision's
numbers are what was measured when it was made, and where a command in this repository
measures the same thing, the entry names it. Citations like `03-retrieval-rag/10` name
chapters of the author's research notes, which are not published.

`tests/test_decisions.py` fails if the repository cites a decision this page lacks.

## N7. Units are paragraphs; the sentence variant is an experiment

**Call.** A unit is a paragraph, split at sentence ends past 256 tokens, never packed
across paragraphs, so an insertion does not shift every unit after it, and never across
an access segment; its heading path is prefixed. Table rows are grouped under their
header.

**Evidence.** In the research notes, embedding sentences and returning their paragraph
beat embedding paragraphs by 2 to 12 points of recall@1, at four times the vectors;
fixed token windows lost by 9 to 18 points (`03-retrieval-rag/10` section 2).

**Amended at the full build.** The full profile came to 501,031 units, 1,641 of prose
and 499,390 groups of rows. Cut into sentences and rows it would be about 4.33 million
points, whose int8 codes and graph fit in 16 GB: Qdrant holds 1.08 GB of its own memory
for this build (`bench/full_build.py --serving`). The exact search every role takes
(N51) would not: it reads every float32 vector its filter passes, 17.7 GB at that size.
So the sentence variant runs on the full profile and searches through the graph, by
label.

**Reverses if.** The sentence variant wins on the evaluation's probes with accuracy no
worse.

## N13. deepseek-flash is the model under test, behind one small interface

**Call.** One provider is evaluated, DeepSeek's `deepseek-flash`, the model the project is
tuned for. The answer layer reaches it through `complete`, which sends `thinking` as a top
level field of the request, disabled unless asked or configured otherwise. Model ids and
prices live in `config/providers.toml`. Adapters for OpenAI's Responses API, Anthropic's
Messages API and Gemini's generateContent share only the post and its retries, since
reasoning switches, JSON output, stop reasons and token counts differ by provider; each
is tested on replies shaped as its reference documents them and claims nothing more, as
no key for them exists here.

**Evidence.** DeepSeek's documentation shows `thinking` inside `extra_body`, the OpenAI
SDK's syntax, which the SDK unwraps before it sends; a raw request that keeps the wrapper
leaves thinking on, the API's default. On 2026-10-13, `bench/deepseek_call.py` asked one
question three ways: thinking off at the top level spent 0 reasoning tokens, on spent 34,
and off inside `extra_body` spent 41. The research notes warn that reasoning and caching
controls are not portable and should not be abstracted before a second provider is used
(`09-production/05`).

**Reverses if.** The owner opens the model comparison across providers; then the other
adapters get keys and runs.

## N23. No company is named; the documents say "the Company"

**Call.** The generated documents call their author "the Company", "η Εταιρεία" in
Greek, the defined term real policies use, so no name exists to leak. The repository,
its history and its page name no company, real or invented; department names, internal
system names and product nouns stay, because the processes need them. A check kept
outside the repository, so its word list is never published, scans the tracked tree,
every commit message and the built page before every push.

**Reverses if.** The author asks for a named company.

## N31. The filter gate measures every role's filter as it is sent

**Call.** `policy.readable()` is the one source of a role's labels, for the query path
and the bench alike. The gate is every role's filter exactly as sent, with in domain and
off topic questions apart, against an exact scan of the float32 vectors computed outside
the engine; `legal` alone and `hr` alone are measured as bench only views, reported and
never gated.

**Evidence.** Every role reads `general`, so the filters sent pass 22 to 46 percent of
units and admin's all of them; nobody sends `hr` alone. In the research notes, off topic
questions were the ones filtered search lost (`03-retrieval-rag/11` section 7).

**Reverses if.** A role is defined without `general`.

## N39. The corpus: one plan in code, prose by language models, annexes by code

**Call.** A plan in code, `needtoknow.corpus.plan` at a fixed seed, fixes 164 facts in
103 families, each one exact sentence in English and in Greek, and a catalogue of 160
documents, 97 of them Greek, with 900 sections. Language model agents wrote the prose of
each document around its fact sentences, copying each verbatim and adding no policy
value; the prose is committed (N45). Code adds the annexes a real manual carries,
directories, registers, forms and glossaries, to bring each file to its size, about 280
MB in all. `ntk corpus check` confirms every planted fact where the plan put it. The
corpus is 160 files and 300 MB.

**What it costs.** 103 families cannot carry a test and a dev set one item per family,
so the evaluation's protocol makes the family the unit.

**Reverses if.** The full 400 documents planned at first are wanted; the plan scales
with its catalogue.

## N43. The corpus is revised in code before the parsers, and no prose is bought

**Call.** Every table has named columns and a header; each document has one glossary,
each term once; revisions are dated in order up to a version date; annex rows are
budgeted per label so that the share of units meets `config/labels.toml`. No prose is
bought until the evaluation's dev items show the retrieval has no headroom.

**Evidence.** A review of the corpus at the end of P1 found the annexes to be 1,164 row
shapes, one glossary written 602,893 times, and label shares by unit following annex
bytes rather than the registered shares.

**Reverses if.** The dense search's annex intrusion shows annex units taking a real
share of a prose question's top 10. Measured at P3: 0.7 percent of the top 10, where
tables are 99.7 percent of all units (`bench/filter_bench.py --report`).

## N44. The audits page is written from the finding queue, never by hand

**Call.** `docs/AUDITS.md` is generated from the project's finding queue, and a check
holds every row to that queue and every finding id the history cites to a row, so the
page cannot drift from what was found and fixed.

**Reverses if.** The queue moves into the repository, and the page with it.

## N45. The writers' narrative is committed

**Call.** `narrative/`, the prose the writers produced, 160 files and 1.7 MB, is
committed: it is the one source of the corpus that code cannot make again, and the
corpus rebuilt from it is byte identical. The generated corpus is not committed.

**Reverses if.** The narrative grows past what a repository should carry; it would move
to a release asset whose sha256 sits in config.

## N47. Sentences are split by explicit rules, not Punkt

**Call.** `needtoknow.ingest.sentences`: a sentence ends at ".", "?", "!", and at ";"
after a Greek letter or as U+037E, with a space after and a capital or digit to start
the next, never after a listed abbreviation, a single capital, or inside a number.

**Evidence.** Punkt's trained models are a separate download; the corpus's punctuation
is clean, and its hard cases are few and named. With the rules, 207 of 207 fact
placements come out whole as one sentence of their paragraph (`bench/parse_corpus.py`).

**Reverses if.** Split errors are measured to cost retrieval, or text whose punctuation
cannot be trusted arrives.

## N48. The watcher polls a fingerprint instead of listening for filesystem events

**Call.** `ntk ingest watch` reads every file's path, size and modification time each
interval, waits one quiet interval after they move, then runs the scan, which decides
what changed.

**Evidence.** Filesystem events need a dependency or per platform code and arrive in
bursts that need debouncing anyway; a look at the corpus's 320 files costs 9 ms.

**Reverses if.** The polling interval is what keeps a change from reaching queries in
time.

## N50. Lexical weights are counted over the asking role's view

**Call.** The sparse index stores each passage's BM25 term frequencies without Qdrant's
IDF modifier. The sync keeps every term's document frequency per label set, and a
question's term weights are BM25's IDF, ln(1 + (N-n+0.5)/(n+0.5)), with N and n
summed over the label sets the role may read. One collection, the fusion in the engine,
and a change to the policy takes effect at the next query.

**Evidence.** Weighted over the whole collection, ranks told a role which words were
common in text it may not read: the hidden-frequent word's unit ranked lower in 82 to
88 percent of matched pairs, against 45 to 50 for null pairs; over the view the ranks
depend on readable text alone (`bench/idf_attack.py` and `--global-idf`). A collection
per role would have given the same weights at three times the storage, with a rebuild
for every policy change.

**Reverses if.** The view's frequency table cannot be kept in step with the index.

## N51. Every role's dense search is exact

**Call.** The dense search scores every vector the role's filter passes, in float32;
Qdrant ignores quantization on that path. Naming an `ef` asks for the graph instead, for
the benches and experiments.

**Evidence.** Through Qdrant's graph, no role's filter reached 0.97 overlap@10 with an
exact scan at any ef from 64 to 512, and at 512 the means ran from 0.680 to 0.949;
single label views held. Exact, every role held 1.000 on every probe, at a median of 26
to 83 ms a search (`bench/filter_bench.py`).

**Measured again, 2026-10-13.** The probes are the first 30 words of 360 units drawn per
label, half of them dev, which sit closer to the indexed text than a question does. On
the index rebuilt fresh after the decision, the graph's dev means at ef 512 ran from
0.688 to 0.959; they move when the graph is built again. The exact path held 1.000 on
every probe, at a median of 24 to 38 ms a search and 77 for admin, with the machine 78
to 89 percent busy, this bench and its server included: the bench prints that share
beside its medians, since they move with it.

**Reverses if.** Exact search passes 200 ms for a role, as a corpus several times larger
would make it; then each label is searched through the graph and the lists merged.

## N53. The watcher looks every half second, and a scan pauses the collector

**Call.** `ntk ingest watch` looks every 0.5 s; a scan runs with Python's cyclic
collector paused and under a lock it waits for; the watcher loads the encoder when it
starts.

**Evidence.** A change waits two intervals before its scan, and a look costs 9 ms. In a
watcher holding the encoder, full collections during a cut walked 368,000 objects, and
the largest document's change took 6.6 to 7.4 s. With these, it reaches a query in 4.6
s, and a one sentence edit in 1.6 s (`bench/revocation.py`, `bench/freshness.py`).

**Measured again, 2026-10-13.** Both benches now repeat each change five times in one
watcher and print the median, the largest and the count. The largest document's
widening reached a query at a median of 4.8 s and at most 4.8 s, its narrowing at 4.1 s
and at most 4.1 s; the one sentence edit at 1.4 s and at most 1.5 s, its undoing at 1.5
s. The largest document's margin under 5 s is 0.2 s, and its sync read more slowly with
each run, from 0.40 to 0.73 s.

**Reverses if.** A look at the corpus folder costs more than a few percent of a core,
or the largest document's change passes 5 s again; then a scan cuts only the sections
that changed.

## N56. The split's family is every fact that shares a planted section

**Call.** The evaluation's unit is the fact family as `eval/protocol.toml` version 4
defines it: a family of the plan's registry, its versions and regional variants, joined
to every fact planted in a section beside one of them, transitively. The plan's 103
registry families make 81 such families; 31 are drawn for the test set, one item each,
and the other 50 carry the dev items. `ntk eval split` draws them by the registered seed
and writes `eval/families.json`.

**Evidence.** The registry gives each product sibling a family of its own and plants
siblings, and some facts of other topics, in one section, so an excerpt about one shows
the others' sentences. Split by the registry's families, a group of two such families
would straddle test and dev with chance 2 x 0.388 x 0.612, about 0.47, and a dev item's
excerpt would then show a test fact. Joined, no excerpt can. The price is a smaller test
set: about 15.4 points of Wilson half width at 0.7 accuracy with 31 items, against 13.7
with 40.

**Reverses if.** The corpus is planned again with one fact to a section.

## N57. The golden categories and their counts, fixed before any item

**Call.** `eval/protocol.toml` version 5 counts every item before one is written. The test
holds 35: family items that ask each of the 31 test families once, 6 in the asker's own
label in English, 5 in Greek, 3 cross lingual, 4 in a mixed file, 4 with no access, 4 on
a superseded value, 1 on a regional variant, and one administrator item and one multi hop
item using two families each; and items that belong to no family, 2 on a table row, 2 on
a fact the corpus does not hold, 2 near misses scored like them. Dev holds two or three
items for each of its 50 families in the test's shares, within 0.05, and 8 of each own
kind; each split is between 0.45 and 0.55 Greek. A fact whose value a model may know
without the documents is asked only by a no access item, whose right answer is a refusal.

**Evidence.** A count left to the writing can be chosen once items exist, and the items a
careful model refuses are the easiest. Three test families hold nothing but such a
default, so three of the four no access items are theirs. Every planted fact is one self
contained sentence in one unit, so no item can honestly need an answer spanning two units,
and none is asked for.

**Reverses if.** The writing finds a category its split's families cannot fill; then a new
version, dated, before any item is scored.

## N58. A dev family that no rule lets an item ask carries no item

**Call.** `eval/protocol.toml` version 6 exempts from the dev items a family whose every
fact is a deliberate default under the general label; every other dev family carries two
or three items. Two families are exempt: a yearly badge renewal and a yearly code of
conduct confirmation.

**Evidence.** Version 5 lets only a refusal ask a value a model may know without the
documents, and no role can be refused what is under the general label, which every role
reads. Such a family could be asked by no item at all, while version 5 asked two or three
of each.

**Reverses if.** The defaults rule changes, or a general fact stops being readable by every
role.
