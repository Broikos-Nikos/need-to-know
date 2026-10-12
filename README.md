# need-to-know

A vector database RAG project. Retrieval over a large company document base in
which every section of a file says who may read it, so the same question asked by
different departments is meant to get different answers, and no department is
meant to see a paragraph written for another.

Under construction. Nothing here is measured yet, so nothing here claims a
number: each one arrives with the command in this repository that produces it.

## Run it

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). None of this needs a
GPU, a key or a network.

```
uv sync
uv run ntk doctor --offline
uv run ntk corpus plan
uv run ntk corpus assemble
uv run ntk corpus check
uv run pytest
```

* `ntk doctor` says, one line each, what is in place and what is missing.
* `ntk corpus plan` draws the facts and the catalogue of documents from a fixed seed.
* `ntk corpus assemble` builds the corpus from that plan and the writers' text in
  `narrative/`, into the folder that `NTK_DATA_DIR` names (see `.env.example`).
* `ntk corpus check` confirms that every planted fact sits where the plan put it.
* `ntk corpus mutate` changes the built corpus the way real documents change, and
  journals each change with the effect it must have.
* `bench/` holds the scripts behind the numbers in the history, and
  `docs/AUDITS.md` lists every audit finding the history cites.

The index and the answers, still to come, will also need `uv sync --extra embed`,
a CUDA GPU, the Qdrant binary named in `config/qdrant.toml` and a DeepSeek key.
`ntk doctor` checks each of them.
