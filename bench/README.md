# Benches

Each script here prints numbers that the history quotes, so that none of them rests on
a script outside this repository. `operator_probes.py`, `crash_points.py`,
`parse_corpus.py`, `lexical_corpus.py` and `ingest_steps.py` need the project and its
corpus, and no GPU, key or network; `deepseek_call.py` and `injection.py` need a
DeepSeek key and the network, and nothing else; the others need the GPU, the running index server or both, as
each section says.

## operator_probes.py

The pipeline operator's probes on a scratch copy of 24 documents from the plan: a build
killed after 15 of them, 9 files deleted behind the pipeline's back, a change whose
journal line fails, and two assemblies compared. It needs only `ntk corpus plan`; where
the writers' narrative exists it is used, and otherwise a synthetic one gives the same
counts.

```
uv run python bench/operator_probes.py
git worktree add ../ntk-before 074907c
uv run python bench/operator_probes.py --src ../ntk-before/src
```

The second run is the code before the fixes, and gives the numbers in commit `5547561`
(a kill after 15 of 24 documents left 9 files missing, and the check named 1) and in
commit `fbd9532` (with the journal line made to fail, one file changed and the journal
stayed empty). The first run gives 0 files missing, 9 of 9 named, and 0 files changed.

## crash_points.py

Every write of nine changes, killed in turn on three small documents, then the next run.
It prints how many writes there are, how many kill points end with the change finished
or dropped, and how many leave the files and the journal equal to the crash free state.
Commit `fbd9532` quotes it: 42 writes, and 42 of 42 equal.

```
uv run python bench/crash_points.py
```

`tests/test_bench.py` runs both, so a change that breaks either property, or the
scripts themselves, fails the suite.

## parse_corpus.py

Decodes and parses every file of the assembled corpus, each in its declared encoding,
and checks that every block's span gives its text back, that every file's section
headings match its section lines, and that every fact placement comes out as one whole
sentence. It needs an assembled corpus, which `ntk corpus assemble` builds from a clone
in under a minute.

```
uv run python bench/parse_corpus.py
```

## freshness.py

The change bench of the freshness objective: a one sentence edit to a planted fact in a
plain text file, timed until a role that may read it finds the new sentence by asking it,
and its undoing, through `ntk ingest watch` running as a process of its own, as a
deployment runs it, started once its catch up is done. Every scan and sync between an
edit and the query that saw it prints with its seconds and parts, from the lines the
watcher prints (`bench/watched.py`). It makes the edit and its undoing five times,
`--runs` to change that, and prints each change's median, largest and count, since one
run is one draw of the watcher's poll, the scan and the machine's load; it exits 1 when
any run passes the objective, 5 s, and leaves the corpus as it found it. It needs the corpus, the manifest, the vector cache, the `embed`
extra and the index on the running server.

```
uv run python bench/freshness.py
```

## embed_throughput.py

Counts every unit's passage in BGE-M3 tokens, by layer and language, against the
model's limit and against the estimate the units step cuts by; then times a fixed
sample of 4096 units embedded in batches of 32 by count, as before T3.1, and in
batches cut by padded tokens at five budgets, and each language alone. It needs a
manifest, the `embed` extra and a CUDA GPU, and writes nothing. Commit `8336c0d`
quotes it: 26,434 tokens/s by count, 30,791 at 32768 padded tokens, the longest
passage 486 tokens.

```
uv run python bench/embed_throughput.py
```

## lexical_corpus.py

Runs the lexical analyser over every unit's passage, against plain lower case words:
terms per passage, distinct terms, stopwords dropped, terms by the stemmer that made
them, hash collisions among term ids, the time to build every sparse vector, and
RS-F3's question against its planted sentence. It needs a manifest and no GPU.

```
uv run python bench/lexical_corpus.py
```

## revocation.py

Widens one section's audience to a second label, as the simulator's relabel does, then
puts the file back, which narrows it again, with `ntk ingest watch` running as a process
of its own and syncing the index; it times how long a reader of the second label takes to
find the section, and how long until it can no longer, each with its scan and sync. By
default it takes the largest document that qualifies, LOG_CAT_006, since a change costs a
scan of its whole file; `--doc` names another. It widens and narrows five times,
`--runs` to change that, and prints each change's median, largest and count; it exits 1
when any run passes the objective, 5 s. It needs the corpus, the manifest, the vector cache, the `embed` extra and the index on the
running server, and leaves the corpus as it found it. Commits "Let the watcher carry
every change into the index" and "Bring the largest document's change under 5 s" quote
it.

```
uv run python bench/revocation.py
uv run python bench/revocation.py --doc CUS_MAN_003
```

## idf_attack.py

The lexical side channel attack of AC-F5, run as a role through the retrieval API:
pairs of words the role sees equally rarely, one of them frequent in text it may not
read, asked together; the share whose hidden-frequent word's unit ranks below the other,
against the same share in pairs where neither word occurs in hidden text, in the fused
order, before any reranking, each with its count and Wilson interval, and pooled when
several roles are asked. `--global-idf` gives the weights as they were before N50,
counted over the whole collection, and the shares part. Over the role's view a smaller
gap remains, which a matched pair cannot tell from how the words sit in the role's own
text; that the view reads no hidden text is held by `tests/test_noninterference.py`
instead. It needs the manifest and the index on the running server, and writes its
counts to NTK_DATA_DIR/bench/idf_attack_view.json or idf_attack_whole.json, to which
`tests/test_bench.py` holds N50.

```
uv run python bench/idf_attack.py --role sales,hr,returns
uv run python bench/idf_attack.py --role sales,hr,returns --global-idf
```

## rerank_latency.py

What the rerank switch costs: planted fact sentences, half in each language, asked as one
role at each position of the switch, none, multivector and cross, with the median and 95th
percentile seconds a question takes from text to final order and how often the first unit
differs from the fused order's. It needs the index on the running server, the manifest,
the `embed` extra and both pinned models; whether a reranker helps is P5's question, asked
on dev.

```
uv run python bench/rerank_latency.py
```

## filter_bench.py

Whether the engine's filtered dense search finds what an exact scan finds, for every
role's filter as sent and two bench only views, in both filter forms: probes drawn from
every label's prose and tables, split dev and test, in domain and off topic apart;
Qdrant's graph at ef 64 to 512 and the exact path the search takes, each against an
exact float32 top 10 computed on the GPU outside the engine, by a tie aware overlap@10;
units a case may not read counted, and the share of table units in a prose probe's top
10; beside each case's medians, the share of the machine busy while it ran, this bench
and its server included, since the medians move with it. It needs the manifest, the vector cache, the `embed` extra, a CUDA GPU and the index
on the running server, and writes its numbers to NTK_DATA_DIR/bench/filter_bench.json,
which `--report` prints again. With `--questions dev` its probes are the committed dev
probes' questions instead of units' first 30 words, and its numbers go to
filter_bench_questions.json. N51, N43 and the commit "Search exactly under every role's
filter" quote it, and `tests/test_bench.py` holds the decisions to both saved runs.

```
uv run python bench/filter_bench.py
uv run python bench/filter_bench.py --report
uv run python bench/filter_bench.py --questions dev
uv run python bench/filter_bench.py --questions dev --report
```

## ingest_steps.py

Every file of the assembled corpus through decode, parse, structure, segments and units,
each step timed apart in wall and CPU seconds, with the elements and units made and a
digest of every unit; then the largest file cut again under tracemalloc for its peak
memory. It needs an assembled corpus and no GPU. Commit "Make each table row once in
structure and once in segments" quotes it, run at that commit and at its parent's code:

```
uv run python bench/ingest_steps.py
git worktree add ../ntk-before 4659edd
PYTHONPATH=../ntk-before/src uv run python bench/ingest_steps.py
```

## full_build.py

The full build, timed: with `--fresh` it deletes the manifest, the vector cache, the
lexical statistics, the frequency table and the collection, then runs `ntk ingest build`,
`ntk embed`, `ntk index create` and `ntk index sync`, each in a process of its own that
reads its own peak working set and CPU seconds as it ends, and waits for Qdrant to finish
indexing, Qdrant started afresh so its peaks are this build's. Every phase prints its
seconds, those peaks, the share of the machine busy meanwhile, and the data directory's
largest size, sampled every second; then the manifest's bytes by column. `--serving N`
restarts Qdrant on the built index instead and prints its memory once loaded and after N
questions as admin. It needs the assembled corpus, the `embed` extra, the GPU and the
server's binary, and takes about 80 minutes on this machine. `config/build.toml` records the
disk peak the doctor's floor comes from.

```
uv run python bench/full_build.py --fresh
uv run python bench/full_build.py --serving 50
```

## deepseek_call.py

One question through the DeepSeek adapter, live: T4.1's prompt over two fixed excerpts,
asked with thinking off at the top level of the request, on, and off inside `extra_body`
as the OpenAI SDK's syntax writes it; for each, the tokens as DeepSeek bills them, cache
hits apart, the reasoning tokens, the seconds, the USD and the answer as `parse` reads
it. It exits 1 if thinking off at the top level spends a reasoning token, if the wrapped
form spends none, which would mean the API now reads it, or if an answer is not the first
excerpt's, cited alone. N13 quotes it; three calls cost a fraction of a cent.

```
uv run python bench/deepseek_call.py
```

## injection.py

An instruction planted in a permitted section, asked live (AC-F9): two fixed excerpts,
the first telling its reader to answer a fabricated value and cite excerpts that do not
exist, or to refuse, each asked three times in English and once in Greek, and the clean
excerpt as often, with the answer layer's settings. It exits 1 when an answer is not the
first excerpt's, cited alone, or is not in the question's language. It needs a DeepSeek
key and the network, and nothing else.

```
uv run python bench/injection.py
uv run python bench/injection.py --repeats 5
```

## smoke_answers.py

P4's exit, live: ten questions about table rows, each asking for one cell by its row's
key, a key no other unit holds, answered as a role that may read the row; a pass cites
the excerpt holding the row, holds the value and drops nothing. Three of them are asked
again by a role that may not read their rows, and a pass is `not_found` with the row
neither shown nor cited. Every question goes through the answer cache and the spend log;
`--no-cache` asks the provider again. The rows are table rows, never planted facts, so no
family of either evaluation split is read. It needs the manifest, the index on the
running server, the `embed` extra and a DeepSeek key.

```
uv run python bench/smoke_answers.py
uv run python bench/smoke_answers.py --no-cache
```

## reader_line.py

Whether the line that opens every user message, a mark of the asking role's label set
(AC-F11, KB5-K12), moves an answer: a sample of dev's originals, drawn by the seed, each
searched once as its role and asked twice with the line and twice without it from the
same excerpts, the answer cache bypassed, every answer scored. It prints each run's right
answers, the verdicts that moved between two runs of one prompt and between the prompts,
and the cost. It needs the manifest, the index on the running server, the `embed` extra,
the GPU and a DeepSeek key, and asks the protocol first.

```
uv run python bench/reader_line.py
```

## baseline.py

The baseline's report from a folder of runs, `ntk eval run`'s answers with their
provenance: the headline, all test items' share right, each item scored by its mean over
the repeats, with its Wilson interval, and the family items alone beside it; each split's
four figures apart, the grounded accuracy, the leaks, the unstable items, and the figures
by category, layer, language, table rows shown and a value a law contradicts; the closed
book runs beside, their E items answered with a deliberate default apart; dev less test
over fact families with its interval, a diagnostic that changes nothing (N33); and how
many items were shown the same units in every repeat. It needs the plan, the frequency
table and the manifest the scorer reads.

```
uv run python bench/baseline.py eval/runs/baseline
```

## repeatability.py

Whether retrieval is repeatable on one build, which the release's parity rule needs: every
dev item searched as its role, as `ask` searches, several times in one process, and the
excerpts the prompt would hold compared run by run; it names the items shown different
excerpts and whether their candidates differed too, and exits 1 on any. It needs the
manifest, the index on the running server, the `embed` extra and the GPU, and asks the
protocol first.

```
uv run python bench/repeatability.py --runs 3
```

## null_runs.py

The scorer's floor, ceiling and citation null, three answer sets no model wrote: every item
refused; every item answered with its own first form, citing exactly the sections that
plant its facts; and those answers citing every excerpt the role's search shows. It asks
the protocol first. The first two need the plan, the frequency table and the manifest; the
third also the index on the running server and the `embed` extra, and runs on dev only.

```
uv run python bench/null_runs.py
uv run python bench/null_runs.py --split test --no-search
```

## power.py

The smallest effect the registered test sees on the dev units, by simulation: for each
assumed discordance, the smallest true effect the sign flip test over units adopts in 80
percent of seeded trials. The protocol records the figure at a discordance of 0.2. It needs
only the repository.

```
uv run python bench/power.py
```

## permission_suite.py

The permission suite at retrieval level (`specs/evaluation.md` section 6, items 1 to 3):
up to 20 of the committed dev probes a label, their questions, asked as every role the
policy defines, a user holding two roles, a user with only `general`, and sales with a
membership removed from the policy in memory; each three ways 100 deep, the dense search
alone, the sparse alone and their fusion, which is the reranker's input, and once as
`ask` asks, reranked, the excerpts it would put in the prompt checked unit by unit and the
prompt searched for canaries. Any unit a case may not read, and any canary of a label it
may not read, is counted, and the suite exits 1 on one; for the cases that may read a
probe's gold units, how often one comes back in the fused top 10 and top 100 and in the
excerpts. A probe whose gold unit the manifest lacks is set apart and named.
`--prefixes` asks units' first 30 words instead, half prose and half tables, as at tick
75. It needs the manifest, the vector cache, the `embed` extra, the GPU and the index on
the running server, and takes about 25 minutes. What the answers left behind, the spend
log and the cached answers it names, is searched for canaries by `ntk eval canaries`,
which needs neither the GPU nor the server.

```
uv run python bench/permission_suite.py
uv run python bench/permission_suite.py --prefixes
uv run ntk eval canaries
```
