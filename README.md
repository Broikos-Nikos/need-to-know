# need-to-know

A vector database RAG project. Retrieval over a company's document base in which
every section of a file says who may read it, so the same question asked by
different departments is meant to get different answers, and no department is
meant to see a paragraph written for another.

The documents are synthetic. A plan in code fixes every fact and the document and
section it goes in; language model agents wrote the prose around those facts, kept in
`narrative/`; and code adds the tables a real manual carries. The company is never
named, real or invented: the documents call it "the Company".

Under construction: retrieval, the answers and their evaluation are still to come.
The README claims no number yet; each will arrive with the command in this
repository that produces it.

## What works today

* A clone builds the whole corpus from the plan and `narrative/`, the same bytes
  every time, and checks every planted fact where the plan put it:
  `ntk corpus plan`, `ntk corpus assemble`, `ntk corpus check`.
* A build killed at any point, or a change to the corpus killed at any write,
  leaves the corpus and its change journal agreeing:
  `uv run python bench/operator_probes.py` and `uv run python bench/crash_points.py`.
* Every block the parsers read from the corpus resolves back to its span in the
  file: `uv run python bench/parse_corpus.py`.
* Every paragraph and table row knows who may read it, from its section's audience
  line, and each label's share of the corpus meets its registered share:
  `ntk corpus coverage`.
* Every unit is embedded as its passage, its heading path above its text, and kept
  in a cache under that passage, so a second run, a relabel or a rebuilt index
  embeds nothing again; a passage longer than the model takes is refused, never
  cut: `ntk embed`.
* Every audit finding the history cites, open or fixed, is in `docs/AUDITS.md`.

## Run it

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). `uv sync` fetches the
dependencies once; after that, none of this needs a GPU, a key or a network.

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
* `ntk corpus assemble` builds the corpus from that plan and `narrative/`, into the
  folder that `NTK_DATA_DIR` names (see `.env.example`).
* `ntk corpus check` confirms that every planted fact sits where the plan put it.
* `ntk corpus mutate` changes the built corpus the way real documents change, and
  journals each change with the effect it must have.
* `bench/` holds the scripts behind the numbers in the history, and
  `docs/AUDITS.md` lists every audit finding the history cites.

`ntk embed` needs `uv sync --extra embed` and a CUDA GPU, and the index and the
answers, still to come, will also need the Qdrant binary named in
`config/qdrant.toml` and a DeepSeek key.
`ntk doctor` checks each of them.
