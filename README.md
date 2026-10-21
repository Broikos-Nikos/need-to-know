# need-to-know

Ask a question as a department, and get an answer built only from what that
department may read. Every section of every document says who may read it, so sales
and legal asking the same question get different answers, and neither is shown a
paragraph written for the other.

Underneath it is retrieval augmented generation over a vector database: a dense and a
lexical search, each filtered by the asking role's labels inside the engine, fused and
reranked, then answered with citations from the excerpts the role may read.

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
  phase with the memory and disk each needs, 4912 s here, 3620 of them embedding on
  the GPU: `uv run python bench/full_build.py --fresh`. It holds 160
  documents of 314,698,549 bytes, cut into 501,031 units and as many vectors, the index
  taking 3,948,042,658 bytes of disk: `uv run ntk stats`.
* Every unit is embedded as its passage, its heading path above its text, and kept
  in a cache under that passage, so a second run, a relabel or a rebuilt index
  embeds nothing again; a passage longer than the model takes is refused, never
  cut: `ntk embed`.
* One paragraph edited in the largest file, 8.8 MB, reaches a question through
  `ntk ingest watch` in 3.7 s at the median and 3.8 s at most over five edits, and its
  undoing in 3.6 s and 3.7 s, with the processor 24 percent busy with other programs before
  the runs; the edit is embedded again as one unit of the 501,031 the first time, and read
  from the vector cache after: `uv run python bench/freshness.py`.
* A role's question is searched inside that role's filter, on the dense and the lexical
  side alike, and fused in the engine: no unit a role may not read came back in 5,760
  searches, nor reached the excerpts a prompt would hold, every role, sales and hr held
  together, general alone, and sales with returns removed, asking up to 20 dev probes a
  label:
  `uv run python bench/permission_suite.py`. The test suite asks the same of a small
  profile on CPU, with no GPU or server, so the build itself fails on a leak:
  `uv run pytest tests/test_permission_gate.py`. Asked three times on one build, every dev item
  is shown the same excerpts, ties in the fused list and the reranker ordered by unit id:
  `uv run python bench/repeatability.py`. No answer the cache holds, read for each role
  the spend log names, and no trace the page replays, read for the role it was asked as,
  carries a canary of a label that role may not read, written in any case or spacing:
  `uv run ntk eval canaries`.
* A role's question is answered by DeepSeek from the excerpts the role may read, its
  citations able to name only what it was shown, and an instruction planted in a
  document is quoted rather than followed: `uv run ntk ask "<question>" --as <role>` and
  `uv run python bench/injection.py`.
* The local server is the index's one client on the served path: `uv run ntk serve` listens
  on the loopback alone and answers only requests addressed to it there, a role signs in
  once, and every search sends the session's token, never a role (N64). Each search answers
  with its stage trace: the role's filter and how many units it passes and holds back, each
  search's own candidates and scores, their fusion, the reranker's order, the excerpts and
  each stage's seconds, every unit in it one the role may read; the leak gate asks every
  case through it too. `POST /ask` answers with the same trace, the excerpts as they were sent
  and the answer with its citations, and asks the provider every time: an answer read back
  from a cache comes in milliseconds, and its time would tell the asker that someone with the
  same access asked the same before (N68). The spend log keeps the tokens and the price. A
  dev golden question asked as its own role is
  scored beside its answer by the registered scorer; a test question is answered and never
  scored, `eval/protocol.toml` keeping the test split for the baseline and the release; a
  refusal's verdict reads the same whether its answer exists where the role may not read or
  nowhere, naming no category and no leak (AC-F16). `GET /doc/<id>` shows a role the segments
  of a document it may read and nothing of the others, a document it may read none of refused
  as an unknown one is; a document is named by its id, which says its owning department and
  kind, and its sections by their own numbers, whose gaps show where sections a role may not
  read stand, never their text (N65). `GET /stats` answers with the build's figures as `ntk
  stats` prints them, the same for every role, and `GET /events` with the watcher's changes as
  the role may see them: a scan shows only for the units it added that the role may read,
  keyed by the time it finished, and a change to text the role may not read shows as nothing,
  not even its time (N66). The models
  load as it starts, 33.4 s here with the processor 57 percent busy with other programs, so
  its first search took 3.13 s and the rest a median of 1.20 s and 3.21 s at the 95th
  percentile, the trace's own queries 0.184 s of it at the median; over 2,000 fused units the
  engine's scores match reciprocal rank fusion of the two searches as traced within 2.3e-09:
  `uv run python bench/served.py`.
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
  Of the wrong answers over the repeats, 15, all on dev, were not shown a unit holding every
  fact they needed, and 2 on test and 4 on dev were shown it and refused: T12 twice,
  D141 three times and D136 once. With no documents the same model refused every time, 105 of
  105 answers on test and 459 of 459 on dev, as its rules require, which shows nothing of
  what it knows; asked
  instead for its best guess from its own knowledge, it got 0.074 of the 27 answerable
  test items and 0.043 of the 101 dev ones right, the most its training could have
  supplied without retrieval, none of them on G's value a law contradicts, and its guesses
  hit a deliberate default on three refusal items, T21, D105 and D142 (N61). Verdicts
  differ between two repeats for 0.0190 of the test items and 0.0044 of dev's, and dev less
  test over families is -0.018, -0.069 to +0.038, a diagnostic that changes nothing (N33).
  At this baseline 4 of the 80 dev families have room, so the best experiment there can be
  reaches p 0.125 by the registered test: until dev grows, an experiment is adopted on
  cost alone (N63).
  A question cost 0.000213 USD on test and 0.000191 on dev at the mean, and took 1.91 s and
  1.87 s at the median and 2.53 s and 2.32 s at the 95th percentile, each answer's seconds
  read from the spend log line written for it on this machine.
  The answers and their provenance are in `eval/runs/baseline/`; three of the six runs
  record an unclean tree, edits to code those runs never load, and the two best guess runs
  were made later, at another commit: `uv run python bench/baseline.py eval/runs/baseline`.
* Retrieval is probed apart from the answers: two questions a fact, one in each language,
  made by code from each split's own facts and worded apart from the planted sentences,
  each with its gold units and the roles that may read them: `uv run ntk eval probes --check`.
  Asked by the narrowest role that may read each, the dev probes find a gold unit in the
  top 10 for 0.964 of the 194: 1.000 when the question is in its documents' language and
  0.915 across languages, where the first hit is right half as often, 0.500 against 0.973.
  Table rows, the padding that makes up most units, take 13.1 percent of that top 10 and
  15.8 percent of the excerpts' units shown, and setting them aside changes recall@1 for no
  role (N43). An upper bound, since templated questions sit near the planted text, and each
  fact sits early in a short paragraph on one topic, where a real manual buries it deeper
  (KB1-F9); and with most sections a single paragraph, nothing here tells paragraph units
  from section units (KB3-K6): `uv run python bench/probe_recall.py`. On the test split, asked once for the baseline,
  the 108 test probes find a gold unit in the top 10 for 0.944, 1.000 in their documents'
  language and 0.870 across, and every one of the 27 answerable test items, asked as its
  role, has each section it needs in the top 10:
  `uv run python bench/probe_recall.py --split test`.
* Retrieval holds on other draws of the corpus. One document in four of the catalogue,
  drawn three times, the corpus's own and twice with every single and regional value and
  every table drawn again under another seed, the writers' prose shared: in each, the 56
  dev probes find a gold unit first for 49 and in the top 10 and the excerpts for all 56.
  The figures belong to the method and the prose, not to one draw of values and tables
  (RS-F6, N62): `uv run python bench/draws.py`.
* Every number this list states is printed by a command its bullet names, the output
  recorded in `docs/claims/` and checked by the test suite:
  `uv run python bench/claims.py check`.
* Every audit finding the history cites, open or fixed, is in `docs/AUDITS.md`, and
  every decision the code cites is in `docs/DECISIONS.md`.

## What fails

* The lexical search's term weights leak when they are counted over the whole collection:
  asked as sales, hr and returns, a word frequent in text the role may not read ranked its
  unit lower in 72 of 88 matched pairs (Wilson 0.72 to 0.88), against 60 of 126 pairs of
  words absent from that text (Wilson 0.39 to 0.56). This build counts them over the role's
  own view (N50), where the gap falls to 51 of 88 (Wilson 0.48 to 0.68), inside what a
  matched pair cannot tell from how words sit in the role's own text:
  `uv run python bench/idf_attack.py` and `uv run python bench/idf_attack.py --global-idf`.
* A relabel of the largest document misses its 5 s objective: the widened section reached
  its new reader in 5.3 s at the median and 6.0 s at most over five runs, and narrowed
  back, it left that reader in 4.5 s at the median and 5.2 s at most, with the processor
  12 percent busy with other programs before them, the scan of that one file taking most
  of it (PF-F10): `uv run python bench/revocation.py`.

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
npm ci
npm run build
uv run ntk serve
uv run ntk ingest watch
```

* `ntk serve` and `ntk ingest watch` run until stopped, each in a terminal of its own.
* With `ntk serve` running, a role signs in once, and each search or question carries the
  token the sign in answered with, never a role. From a POSIX shell:

  ```
  curl -s -X POST http://127.0.0.1:6435/session -H "Content-Type: application/json" -d '{"role": "returns"}'
  curl -s -X POST http://127.0.0.1:6435/search -H "Authorization: Bearer <token>" -H "Content-Type: application/json" -d '{"question": "How many days does a customer have to return a device?"}'
  curl -s -X POST http://127.0.0.1:6435/ask -H "Authorization: Bearer <token>" -H "Content-Type: application/json" -d '{"question": "How many days does a customer have to return a device?"}'
  ```
* `npm ci` and `npm run build`, with Node 22 or later, build the page into `web/dist`, which
  `ntk serve` serves at its own address beside its routes. There a role searches, or asks and
  is answered and scored, or the same question goes to all nine roles side by side, and each
  stage of the trace shows with its seconds on the server;
  slow motion plays the trace one stage at a time, and with reduced motion set nothing moves.
  Below it the build's figures, live or as `ntk stats` recorded them, and the change benches'
  recorded runs, each scan and sync of each change, replayed at the speed they ran:
  `uv run python bench/page_data.py`.
  Live, it follows the change feed as the chosen role may see it, asked every five seconds.
  Served anywhere but the loopback, where no server can answer it, it replays three dev
  questions asked as every role through the app, with when, at which commit and with which
  models they were asked: `uv run python bench/page_traces.py`.
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
  the spend log in the data directory, the provider's cached token counts with them. A cache
  that cannot be read is set aside under a name that says so, and the question answered
  without it.
  `ntk doctor` checks each of these.
* `ntk` is the operator's tool: `--as` takes the role on the caller's word, to stand in
  for a user. `ntk serve` takes it from the session instead: a role signs in once, on the
  same word, where a deployment's identity provider would stand, and no query can name one.
* The provider reads every excerpt the asking role may read, restricted sections
  included: it is a reader outside the role model, and choosing one is a decision about
  where the documents may go.
