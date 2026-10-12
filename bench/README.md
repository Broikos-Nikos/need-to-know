# Benches

Each script here prints numbers that the history quotes, so that none of them rests on
a script outside this repository. Both run on any machine with the project installed,
with no GPU, no key and no network.

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
watcher prints (`bench/watched.py`). It exits 1 past the objective, 5 s, and leaves the
corpus as it found it. It needs the corpus, the manifest, the vector cache, the `embed`
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
scan of its whole file; `--doc` names another. It exits 1 past the objective, 5 s. It
needs the corpus, the manifest, the vector cache, the `embed` extra and the index on the
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
against the same share in pairs where neither word occurs in hidden text. With weights
counted over the whole collection the shares part; with weights over the role's view
they must not. It needs the manifest and the index on the running server.

```
uv run python bench/idf_attack.py --role sales
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
10. It needs the manifest, the vector cache, the `embed` extra, a CUDA GPU and the index
on the running server, and writes its numbers to NTK_DATA_DIR/bench/filter_bench.json,
which `--report` prints again. N51 and the commit "Search exactly under every role's
filter" quote it.

```
uv run python bench/filter_bench.py
uv run python bench/filter_bench.py --report
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
