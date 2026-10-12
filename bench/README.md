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

Times a one sentence edit through the watcher into the manifest, and its undoing, on a
plain text file of the corpus, which it leaves as it found it. It needs an assembled
corpus and a manifest, which `ntk ingest build` makes.

```
uv run python bench/freshness.py
```
