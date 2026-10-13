# need-to-know

A vector database RAG project. Retrieval over a company's document base in which
every section of a file says who may read it, so the same question asked by
different departments is meant to get different answers, and no department is
meant to see a paragraph written for another.

The documents are synthetic. A plan in code fixes every fact and the document and
section it goes in; language model agents wrote the prose around those facts, kept in
`narrative/`; and code adds the tables a real manual carries. The company is never
named, real or invented: the documents call it "the Company".

Under construction: the page and the hardening are still to come. Every accuracy
figure below is this one corpus's; retrieval, which needs no language model, was
measured again on two more draws of it. Each number the README gives comes with the
command in this repository that produces it.

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
* Ingestion holds one file at a time, so its memory follows the largest file, not
  the corpus, at about 14 times that file's size; a file of a hundred megabytes or
  more would want cutting a section at a time: `uv run python bench/ingest_steps.py`.
* The full build, from the assembled corpus to a searchable index, runs phase by
  phase with the memory and disk each needs, about 80 minutes here, an hour of it
  embedding on the GPU: `uv run python bench/full_build.py --fresh`.
* Every unit is embedded as its passage, its heading path above its text, and kept
  in a cache under that passage, so a second run, a relabel or a rebuilt index
  embeds nothing again; a passage longer than the model takes is refused, never
  cut: `ntk embed`.
* A role's question is searched inside that role's filter, on the dense and the lexical
  side alike, and fused in the engine: no unit a role may not read came back in 5,760
  searches, nor reached the excerpts a prompt would hold, every role and three
  combinations of them asking up to 20 dev probes a label:
  `uv run python bench/permission_suite.py`. Asked three times on one build, every dev item
  is shown the same excerpts, ties in the fused list and the reranker ordered by unit id:
  `uv run python bench/repeatability.py`. No answer the cache holds, read for each role
  the spend log names, carries a canary of a label that role may not read:
  `uv run ntk eval canaries`.
* A role's question is answered by DeepSeek from the excerpts the role may read, its
  citations able to name only what it was shown, and an instruction planted in a
  document is quoted rather than followed: `uv run ntk ask "<question>" --as <role>` and
  `uv run python bench/injection.py`.
* The evaluation's fact families are split into test and dev by a seed registered before
  any question was written, with every draw that failed the balance check recorded:
  `uv run ntk eval split --check`.
* The golden questions, 35 test and 153 dev in `eval/golden/`, are each held to the rules
  of their category, the counts the protocol registered and the corpus they ask about:
  `uv run ntk eval check`, which reads the plan, the frequency table and the manifest a
  build makes. A dev question is drafted from `uv run ntk eval draft`, which hides the
  words that decide retrieval. `eval/protocol.toml` changes only by a new dated entry in
  `eval/protocol_history.toml`, which a test holds it to.
* A split's items are asked as their roles, three repeats with the answer cache off as the
  protocol registers, each answer written with the model that gave it as it comes, so a
  run cut short resumes without paying twice; with no excerpts, `--closed-book` asks under
  the answer rules and `--guess` for the model's best guess from its own knowledge (N61):
  `uv run ntk eval run --split dev --purpose development --out <file>`. Then
  `uv run ntk eval score <file> --purpose development` scores each repeat, then each item
  by its mean over them, and names the items whose verdict changed.
* The baseline, deepseek-flash at temperature 0, every item asked three times with the
  cache off and scored by its mean: all 35 test items 0.981, Wilson 0.869 to 0.997, the 29
  family items alone 0.977; dev 0.970 over its 133 originals, answerable 0.960 and refusals
  1.000; no leak on either split. The lowest categories ask across languages, C, 0.778 over
  3 test items and 0.857 over 14 dev ones, and two facts at once, F, 0.500 over 4 dev items.
  Of the wrong answers over the repeats, 15, all on dev, were not shown a section planting
  every fact they needed, and 6 were shown it and refused: T12 twice on test, D141 three
  times and D136 once on dev. With no documents the same model refused every time, 105 of
  105 answers on test and 459 of 459 on dev, as its rules require, which shows nothing of
  what it knows; asked
  instead for its best guess from its own knowledge, it got 0.074 of the 27 answerable
  test items and 0.043 of the 101 dev ones right, the most its training could have
  supplied without retrieval, none of them on G's value a law contradicts, and its guesses
  hit a deliberate default on three refusal items, T21, D105 and D142 (N61). Verdicts
  differ between two repeats for 0.0190 of the test items and 0.0044 of dev's, and dev less
  test over families is -0.018, -0.069 to +0.038, a diagnostic that changes nothing (N33).
  The answers and their provenance are in `eval/runs/baseline/`; three of the six runs
  record an unclean tree, edits to code those runs never load, and the two best guess runs
  were made later, at another commit: `uv run python bench/baseline.py eval/runs/baseline`.
* Retrieval is probed apart from the answers: two questions a fact, one in each language,
  made by code from each split's own facts and worded apart from the planted sentences,
  each with its gold units and the roles that may read them: `uv run ntk eval probes --check`.
  Asked by the narrowest role that may read each, the dev probes find a gold unit in the
  top 10 for 0.964 of the 194: 1.000 when the question is in its documents' language and
  0.915 across languages, where the first hit is right half as often, 0.500 against 0.973.
  Table rows, the padding that makes up most units, take 13.0 percent of that top 10 and
  15.6 percent of the excerpts' units shown, and setting them aside changes recall@1 for no
  role (N43). An upper bound, since templated questions sit near the planted text:
  `uv run python bench/probe_recall.py`.
* Retrieval holds on other draws of the corpus. One document in four of the catalogue,
  drawn three times, the corpus's own and twice with every single and regional value and
  every table drawn again under another seed, the writers' prose shared: in each, the 56
  dev probes find a gold unit first for 49 and in the top 10 and the excerpts for all 56.
  The figures belong to the method and the prose, not to one draw of values and tables
  (RS-F6, N62): `uv run python bench/draws.py`.
* Every audit finding the history cites, open or fixed, is in `docs/AUDITS.md`, and
  every decision the code cites is in `docs/DECISIONS.md`.

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

### Build the index and search it

```
uv sync --extra embed
uv run ntk qdrant start
uv run ntk ingest build
uv run ntk embed
uv run ntk index create
uv run ntk index sync
uv run ntk search "How many days does a customer have to return a device?" --as returns
uv run ntk ask "How many days does a customer have to return a device?" --as returns
uv run ntk ingest watch
```

* The build needs a CUDA GPU for the encoder and the Qdrant binary that
  `config/qdrant.toml` pins. `ntk qdrant start` says where to download it, the sha256
  to check and where to unpack it, and writes the server's key into `.env`.
* `ntk ingest build` cuts the corpus into units in the manifest; `ntk embed` embeds
  every unit's passage into a cache; `ntk index create` makes the collection and
  freezes the lexical statistics; `ntk index sync` brings the collection to the
  manifest. `ntk index create --replace` builds the collection again and keeps the
  vector cache.
* A search asks as a role, through that role's filter only; `ntk ingest watch` carries
  every change to the corpus into the manifest and the index.
* `uv run python bench/full_build.py --fresh` runs the whole build, timed phase by
  phase with its memory and disk.
* An answer comes from the excerpts the role may read, through DeepSeek with the key in
  `.env`, and the command prints its citations, what the call cost and how long each
  stage took.
  An answer is cached under the prompt as sent, which opens with a mark of the asking
  role's label set: a narrowed section is never answered from the cache, and a role shares
  a cached answer only with roles that read the same labels, a mark that moved no dev
  answer (`uv run python bench/reader_line.py`). Every question adds a line of numbers to
  the spend log in the data directory, the provider's cached token counts with them.
  `ntk doctor` checks each of these.
* `ntk` is the operator's tool: `--as` takes the role on the caller's word, to stand in
  for a user. A served path takes the role from the session, never from the request.
* The provider reads every excerpt the asking role may read, restricted sections
  included: it is a reader outside the role model, and choosing one is a decision about
  where the documents may go.
