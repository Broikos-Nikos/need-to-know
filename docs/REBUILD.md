# Rebuilding need-to-know

After the upload, on 2026-10-22, the build's data was deleted from the machine that made it: the assembled corpus,
the manifest, the index and its server's storage, the vector cache, the lexical statistics and the three draws. This
repository holds everything they were made from, and this page is the order to make them again in and the figures
that say a rebuild is the same build.

## What it needs

- Python 3.12 and [uv](https://docs.astral.sh/uv/); Node 20.19, or 22.12 or later, for the page; a CUDA GPU for the
  encoder and the reranker, an RTX 4060 with 8 GB here; and the free disk `uv run ntk doctor` asks for under the
  folder `NTK_DATA_DIR` names (see `.env.example`).
- The two models, pinned in `config/models.toml` by repository and revision, `BAAI/bge-m3` at
  `5617a9f61b028005a4858fdac845db406aefb181` and `BAAI/bge-reranker-v2-m3` at
  `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. They load only from the local Hugging Face cache; `ntk doctor` names
  the command that fetches each, `uv run hf download <repo> --revision <revision>`.
- The index's server, Qdrant `1.19.2`, pinned in `config/qdrant.toml` with the sha256 of its download and of its
  binary; `ntk qdrant start` names the download and the sha256 to check it by, checks the binary's own sha256 before
  every start, and writes the server's key into `.env`.
- A DeepSeek key in `.env` for answers alone. The index needs none, and the committed answers in
  `eval/runs/baseline/` need no asking again.

## The order

```
uv sync --extra embed
uv run ntk doctor
uv run ntk qdrant start
uv run ntk corpus plan
uv run ntk corpus assemble
uv run ntk corpus check
uv run ntk ingest build
uv run python bench/corpus_digest.py
uv run ntk embed
uv run ntk index create
uv run ntk index sync
uv run ntk stats
npm ci
npm run build
uv run ntk serve
```

The seeds are code, not settings: the plan's in `src/needtoknow/corpus/plan.py`, the split's in
`eval/protocol.toml`, the golden questions' in `src/needtoknow/eval/golden.py`, the draws' in the command
`bench/draws.py`'s section of the bench README gives. `uv run python bench/full_build.py --fresh` runs the build from
the manifest on, timed phase by phase.

## The figures a rebuild reproduces

- `bench/corpus_digest.py` prints a digest of the plan, of the assembled corpus byte for byte, and of the manifest's
  units: the same three lines as its record, `docs/claims/python-bench-corpus_digest.py.txt`. The build made on
  2026-10-12, as it stood on 2026-10-22, a fresh clone's assembly and ingest of 2026-10-22, and the rebuild from
  this page on the machine whose data was deleted gave the same lines.
- `ntk corpus check` finds every planted fact where the plan put it, and `ntk stats` prints the build's counts as its
  record, `docs/claims/ntk-stats.txt`, does. The three sizes on disk may differ: the index's by the engine's own
  layout, the manifest's and the vector cache's by the order their pages were written in.
- Every figure the README states is a command's output kept in `docs/claims/`. `uv run python bench/claims.py record
  "<command>"` writes a record again, and `git diff docs/claims` shows what moved. The vectors come from the GPU,
  which need not give an embedding the same last bits twice, so retrieval's figures are compared at the precision
  they are printed.

## What was kept

The repository, its committed answers and records, and the Hugging Face cache, which other projects on that machine
share: the two models were left in it.
