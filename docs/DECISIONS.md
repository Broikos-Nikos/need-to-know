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

## N33. The split's balance is checked as it is built, and no test score moves it

**Call.** The baseline reports dev less test over fact families, with its interval, as a
diagnostic that changes nothing. The split is balanced when it is built, on properties
that carry no outcome: the families' placements, document size, language and salience
when the split is drawn, and each item's stemmed overlap when it is written, each as a
standardized mean difference under 0.1. A failing split is redrawn before any item
exists, an unbalanced item is rewritten and never moved, and after the first scored run
no family changes split (`eval/protocol.toml`, `[splits.balance]` and `[baseline]`).

**Evidence.** The rule this replaces redrew the split at the baseline when dev and test
disagreed beyond noise. Here the baseline is the first release and its test score is
published, and the dev items are read during development: a redraw would let a test
outcome choose the split, and would move dev families whose answers had been read into
the new test split.

**Reverses if.** Development stops reading dev items before the baseline and the
baseline stops being published; the redraw would then fit again, as a new protocol
version.

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

**Measured on questions, 2026-10-13.** This call first reversed on annex units taking a
real share of a prose question's top 10. Measured at P3 on units' first 30 words: 0.7
percent of the top 10, where tables are 99.7 percent of all units
(`bench/filter_bench.py --report`). On the dev probes' questions: 14.7 percent of the
dense top 10 (`--questions dev --report`), and through the whole search, fused and
reranked as an answer's excerpts are, 13.0 percent of the top 10 and 15.6 of the shown
excerpts' units for each probe's narrowest reader, 18.4 and 24.6 for admin. The share is
real; the cost it stood for is not. With the table units set aside from the same hits,
recall@1 is the same for every role, and recall@10 moves from 0.964 to 0.969, and from
0.918 to 0.943 for admin (`bench/probe_recall.py`). The annexes stay as written.

**Measured again, 2026-10-18,** after ties came to be ordered by unit id: the table units
take 13.0 percent of the top 10 and 15.7 of the shown excerpts' units for each probe's
narrowest reader, 18.5 and 24.6 for admin, and every recall above is as it was
(`bench/probe_recall.py`, recorded in `docs/claims/`).

**Reverses if.** Table units cost what their share stood for: recall with them set aside
passes recall as served by more than 0.01 at 1 or at 10, or the dev items whose shown
excerpts hold table rows score below those whose excerpts hold none by more than the
smallest effect the protocol can see, 0.12 (N60). As first written the condition had no
margin, and the baseline met it on noise: the answerable prose items shown table rows
scored 0.952 over 21, those shown none 0.958 over 72 (`bench/baseline.py
eval/runs/baseline`).

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
common in text it may not read. Asked as sales, hr and returns, the hidden-frequent
word's unit ranked lower in 74 of 88 matched pairs (Wilson 0.75 to 0.90), against 60 of
126 null pairs (Wilson 0.39 to 0.56); weighted over the view, in 53 of 88 (Wilson 0.50 to
0.70), its null 61 of 126 (Wilson 0.40 to 0.57)
(`bench/idf_attack.py --role sales,hr,returns`, and `--global-idf`). The view's gap over
its null is one this statistic cannot read: a null pair matches how often the role sees
a word, not how its units sit in the role's text. The view's claim rests on construction
instead, ranks from text the role may read and one statistic over every unit, the
average passage length frozen at build, and on a test: builds that differ only in text a
role may not read, rewritten or dropped, give every role the same weights, fused order
and scores, and reranked order (`tests/test_noninterference.py`). Units of exactly equal
score are put in order by the engine; Qdrant's local mode reads every point's score to
do it, so the test sorts those stably, and how the server orders them is not measured
here. A collection per role would have given the same weights at three times the
storage, with a rebuild for every policy change.

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

**Asked questions, 2026-10-13.** On the 194 dev probes' questions instead, shorter and
worded apart from the text (`bench/filter_bench.py --questions dev`), no role reached
0.97 at any ef: the graph's dev means at ef 512 ran from 0.533 to 0.678 over the nine
roles, and admin's fell from 0.959 to 0.622. Of the two single label views, hr alone
still reached it at ef 64; outside counsel, legal alone, which reached it at ef 256 on
unit prefixes, ended at 0.943. The exact path held 1.000 on every question, with no unit
a case may not read in 21,340 searches, at a median of 25 to 44 ms a search and 80 for
admin, the machine 85 to 98 percent busy.

**Reverses if.** Exact search passes 200 ms for a role, as a corpus several times larger
would make it; then each label is searched through the graph, at an ef its questions
hold 0.97 with, and the lists merged.

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

**Measured again, 2026-10-18.** The freshness bench edits the largest plain text file
that plants a number now, NET_MAN_009 at 8.8 MB, where the figures above came from the
first such file: the edit reached a query at a median of 3.2 s and at most 3.5 s, its
undoing at 3.1 s and at most 3.2 s, the scan of the file taking 1.5 to 1.7 s of it, and
the first edit embedded one unit of the 501,031 (`bench/freshness.py`).

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

## N59. Twenty dev items get a paraphrase twin, drawn by a seed

**Call.** Twenty dev items each get a twin: the same question for the same role, in the
same language, with the same expectation, worded away from the original. A twin shares no
more of its planted sentence's three rarest terms than its original does, and at most half
of its original's terms. The twenty are drawn by a committed seed in the test's category
shares; `ntk eval twins` draws them again. The gap between twins and originals is reported.

**Evidence.** The protocol registers twenty twins, and the review that asked for them cited
a measurement in which questions reworded away from a note found it first 0.556 of the
time against 0.963. A twin adds an item to its family, so only an item whose families hold
two may take one; no versioned or variant family has room, and the table items, absent
facts and near misses have counts registered exactly, so none of them takes a twin.

**Reverses if.** A family is allowed a fourth item, or the registered number changes.

## N60. The smallest visible effect is a simulation's, at a stated discordance

**Call.** `eval/protocol.toml` version 7 records the smallest effect the dev units can show
as 0.12, at an assumed discordance of 0.2 per item, the share of items a change flips, and
names `bench/power.py`, which prints the figure: the smallest true effect the registered
test adopts in 80 percent of simulated experiments.

**Evidence.** The bench, on the dev split's 80 units: 0.08 at a discordance of 0.10, 0.10
at 0.15, 0.12 at 0.20 and 0.14 at 0.30. The previous figure, 0.10, was an estimate for 63
families that no command produced, and holds only at the lower discordance.

**Measured at the baseline, 2026-10-13.** What a baseline measures is the noise, not the
discordance a change brings: verdicts differ between two repeats of one configuration for
0.0044 of the dev items and 0.0190 of the test items (`bench/baseline.py
eval/runs/baseline`). An experiment's discordance is that floor and what its change flips,
so 0.2 stays as the assumption, the side that asks a larger effect.

**Reverses if.** The first experiment measures its own discordance, which then replaces the
assumption, or the dev units change.

## N61. The closed book control asks for the model's best guess

**Call.** The control KB1-F7 asks for, the questions with no documents, is asked under
`prompt.GUESS`, rules that want the model's best guess from its own knowledge in the
answer's reply form (`ntk eval run --guess`), both splits at the protocol's three repeats.
The run under the answer rules stays beside it as what it is, the check that the rules
hold with no documents.

**Evidence.** Under the answer rules, which say never to use other knowledge and never to
guess, the model refused 105 of 105 answers on test and 459 of 459 on dev: a model that
knew every planted value would have done the same (DR-F27). Asked for its best guess, it
got 0.074 of the 27 answerable test items and 0.043 of the 101 dev ones right, none on G's
value a law contradicts, and hit a deliberate default on T21, D105 and D142
(`bench/baseline.py eval/runs/baseline`). A model allowed to decline what it is unsure of,
as closed book factuality benchmarks grade, would answer fewer of these, not more, so the
best guess is the bound: the most the model's training could have supplied without
retrieval. It errs high, since the scorer credits a guessed range that holds the value.

**Reverses if.** The bound grows past a few items, when a run that may decline is needed to
tell what the model holds with confidence from what it guesses well; or the answer rules
change.

## N62. Each draw is built in a data folder of its own and searched in Qdrant's local mode

**Call.** `ntk corpus draw` makes each draw a data folder of its own, and `bench/draws.py`
builds it there and indexes it in Qdrant's local mode, in the bench's own memory, at one
document in four of the catalogue.

**Evidence.** A second collection on the running server would search as the served index
does, but the collection's name is one constant across the index code, and `ntk index
sync` run with NTK_DATA_DIR on a draw would sync the draw's manifest into the served
collection, deleting every point the draw does not explain. Local mode reaches no server.
It scores every point the filter passes, as the exact path does (N51), and differs from
the server only in the order of equal scores (AC-F12). One document in eight planted too
few dev facts to read a spread across draws; one in four doubles them for twice the build.

**Reverses if.** The draws differ by more than their probes' intervals and the server's
exact path is needed to rule local mode out as the cause, or a draw must be searched
through the filtered graph.

## N63. Until dev can show a gain, an experiment is adopted on cost alone

**Call.** Protocol version 8 states beside every decision its reach, the best p any
candidate could get against the incumbent's dev units (`stats.reach`), and registers that
until dev can show a gain an experiment is adopted on cost alone. P8 opens by growing dev
in the categories the baseline misses, as a dated version, before any experiment.

**Evidence.** The registered test, a sign flip over family differences, sees only the
families that change, and with k of them it cannot go below 2 in 2^k. The baseline left 4
of the 80 dev families below a perfect score, so the best experiment there can be, every
one of them made perfect and none worse, reaches p 0.125 (`bench/baseline.py
eval/runs/baseline`); the probe arm's 7 misses of 194 sit in 5 of 50 families, p 0.0625
at best (`bench/probe_recall.py --report`). Any gain would have read too small to tell,
however large.

**Weighed.** Counting items instead of families reaches no further: the dev originals below
a perfect score are 4 of 133, p 0.125 again (`bench/baseline.py eval/runs/baseline`), and
the protocol made the family the unit in its version 3, since items about one fact are not
independent. Growing dev now would write and answer new items before any experiment needs
them.

**Reverses if.** Dev grows until the reach falls below alpha, when the arm that reads a
gain can adopt again; or the incumbent changes.

*Corrected 2026-10-18, tick 161:* as first written this entry said counting items would
reach alpha with 7 of them, a count no command printed that took in three paraphrase
twins, which are no units; and the probe arm's 5 families came from an audit's scratch
work (MA-F19). Both now come from the commands named.

## N64. The server takes the role at sign in, and its trace shows each search's own scores

**Call.** `ntk serve` listens on 127.0.0.1 alone and answers only requests addressed to that
name or to localhost. A role signs in once, `POST /session`, a role of the policy taken on
the caller's word as `ntk --as` takes it, and gets a token; every query sends the token in a
header and never a role: a body that names one is refused, and so is a query without a
session. Each search answers with its stage trace: the role's filter, with the units it
passes and holds back counted; each search's candidates with its own scores, asked again on
their own since the engine returns only their fusion, in the engine's order; the fusion; the
reranker's order; the excerpts; each stage's seconds; and the text of every unit named.

**Evidence.** The research notes resolve a user's roles at query time from the identity
provider and make the retrieval API the store's only path (`03-retrieval-rag/11` sections 4
and 6). A token in a header, unlike a cookie, is attached by no browser on its own, so a page
served elsewhere cannot ask as the signed in role, and the host check refuses one that
reaches the loopback through a name of its own; a page can hold a token a role, to ask the
same question as several. KB1-F4's first condition kept the raw lexical scores in the
server while the weights read the whole collection; N50 made them a function of the role's
view, and the noninterference test now holds each search's own candidates and scores as the
trace shows them, and the count of units the filter passes. The count of units held back is
the one figure of the trace that moves when text the role may not read changes: it says how
much the filter keeps from the role, never what. Units of equal score keep the engine's
order, the order the fusion ranks them in: sorted by unit id, tied units showed ranks the
fusion had not given them. So kept, the engine's fused scores match reciprocal rank fusion
of the two searches as traced within 2.3e-09 over 2,000 fused units on the running index
(`uv run python bench/served.py`).

**Reverses if.** The noninterference test fails on the trace; an attack on the side scores
separates words frequent in hidden text from the null; or a deployment signs its users in
through an identity provider, which replaces the sign in and keeps the rest.

## N65. A document is shown by its id, which names its owner and its kind

**Call.** Citations, excerpts, the served trace and a document's view name a document by its
id, as `SAL_CIR_017`: the owning department's code, the kind of document and a number. A
role that may read any part of a document learns those three, even where the document's
title stands in a section the role may not read, which AC-F4 keeps from it; it learns none
of the document's other text.

**Evidence.** The policy labels text, not documents: each section names its audience. An id
that named nothing would stand in every prompt, for the model and the role alike, mapped
back for the operator, so every prompt would change, the registered baseline's with them,
and the baseline would be asked again, to hide a document's kind from a role that already
reads part of it. The research notes ask that citations and titles come from the segment
the role may read (`03-retrieval-rag/11` section 4); an id is the document's address, not
its text.

**Reverses if.** A document's owner or kind becomes something a role may not know, or a
deployment's ids carry people or subjects: then ids are mapped to opaque ones at the prompt
and at the interface, under a new protocol version, and the baseline is asked again.

*Amended 2026-10-20, tick 173 (AC-F17):* the documents number their sections across
audiences, so a role reading sections 1, 8, 10 and 12 of a document learns from headings it
may read that the others exist and are someone else's. The numbers are the documents' own
text, as the ids are their addresses, and stay; a deployment whose section counts were
sensitive would number sections within each audience.

## N66. The change feed is polled, keyed by time, and shows only what a role may read

**Call.** `GET /events?after=<time>` answers a session's role with the finished scans after
that time that added units it may read, each with the documents and how many, keyed by the
time the scan finished. A scan that added none is left out whole, a scan's own number is
never served, and a removal shows as nothing. The page asks it again every few seconds.

**Evidence.** A scan's units are the manifest's rows of its run, with their labels; a
removed unit leaves no row, so no role's share of a removal can be told, and none is shown.
The scans are numbered across every audience, so a role seeing scans 120 and 123 would learn
that two changes it may not see happened between: the time it finished stands in for the
number. A browser's `EventSource` cannot send the session's header, so a stream would carry
its token in the address; a poll sends it as every other route does.

**Reverses if.** The manifest keeps a removed unit's labels, when a role's share of a removal
can be shown; or the page needs changes faster than a poll brings them.

## N67. The page sets every word in faces that draw Greek, and finds its server by an open route

**Call.** The page is set in Manrope and Roboto Mono, subsets of Latin and Greek, and not in
the house's display face, Teko, which draws no Greek. It is live when the address that
serves it answers `GET /roles`, the one route that asks no session, as `ntk serve` does
where it serves the built page beside its routes; anywhere else it replays.

**Evidence.** Half of the corpus and of the golden questions is Greek, and a face whose Greek
falls back to another face is the house style's own fourth warning for an instrument.
`tests/test_page.py` holds both files to every character the narrative and the golden
questions use, and fails on a Manrope cut without Greek. A probe of a route that wants a
session answers 401, which a browser reports as an error on every load; the open route
answers 200 live and 404 on a static host, and brings the role names the sign in needs.

**Reverses if.** A display face with Greek and a licence for the web is chosen for the
house, or the page needs to find a server it is not served by.

## N68. The served path asks the provider every time

**Call.** `POST /ask` reads and keeps no cached answer. The answer cache stays for `ntk ask`,
the operator's.

**Evidence.** The cache is keyed by the prompt as sent, the reader's mark in it, so two people
share an entry only when they read the same labels and were shown the same text (AC-F11). But
an answer read back comes in milliseconds where a call takes seconds, so the trace's seconds,
and the response's time on any clock, told an asker that someone with the same access had
asked the same before, which the served answer was written not to tell (KB5-K12). The base
agrees on both counts: show cache hits to the operator, not to the asker, and a shared cache
leaks by its timing. `tests/test_api.py` asks one question twice, as two sessions of one
role, of a provider that takes its time: both reach the provider and both take that time,
and the test fails with the cache read.

**What stays open.** The provider's own prefix cache is shared within one label set, which the
reader's mark parts, and a repeat may come back from it faster. Where the existence of a
question is itself sensitive, a deployment keys by user.

**Reverses if.** The served path needs the cache's saving more than an asker's privacy among
people of the same access, or the cache comes to be keyed by user.
