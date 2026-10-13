# The audits

Commit messages and code comments cite identifiers like `OP-F1`. This is what they refer to.

**157 findings, 123 closed, 34 open**, from 16 reviews. Each audit pass took one perspective and nothing else.

Written from the loop's own queue by its `tools/audits_md.py`, and held to that queue by the workspace gate `tools/check-audit-status.mjs`, which fails if a row says anything the queue does not.

A finding's numbers are what its audit measured when it filed it, by that audit's own probes, which stay with the loop's records rather than in this repository (MA-F9). Where a command in this repository measures the same thing, the row names it; run today, it gives today's figure, which a fixed finding has moved.

## `HS`, the hostile stranger

An audit pass that tried to break a first run. 17 findings, 15 closed.

| id | severity | status | finding |
|---|---|---|---|
| `HS-F1` | medium | fixed, tick 12 | A .env with a UTF-8 byte order mark hides its first key; read with utf-8-sig |
| `HS-F2` | medium | fixed, tick 12 | HTML answered with 200 prints a JSONDecodeError traceback; catch it and say what came back |
| `HS-F3` | medium | fixed, tick 12 | A refused key reports only HTTPError: inline comments and trailing spaces stay in values, and a 401 loses its status |
| `HS-F4` | medium | fixed, tick 12 | Greek printed into a pipe crashes on Windows (stdout cp1252); reconfigure stdout and stderr to UTF-8 in main |
| `HS-F5` | medium | fixed, tick 12 | NTK_DATA_DIR=~/x creates a folder named ~ inside the repository; expanduser and refuse a remaining ~ |
| `HS-F6` | medium | fixed, tick 11 | The name gate misses spaced, hyphenated and zero width split names and skips binaries while reporting ok |
| `HS-F7` | low | fixed, tick 12 | A model list item without an id raises KeyError |
| `HS-F8` | low | fixed, tick 12 | A configured model missing from the price table raises KeyError even offline; validate the catalogue on load |
| `HS-F9` | high | fixed, tick 77 | A build running while one file changes kills the watcher: its scan waits 5 s for the manifest and ends it, 'database is locked', and later changes wait for a restart |
| `HS-F10` | medium | fixed, tick 79 | A question of a megabyte crashes the search: one SQL variable per term, 'too many SQL variables' |
| `HS-F11` | medium | fixed, tick 79 | A stopped server or a missing GPU reaches ntk search and ntk index sync as a traceback |
| `HS-F12` | low | fixed, tick 79 | Out of range numbers are taken: --limit -5 prints 95 results, --limit 0 nothing, --interval -1 a traceback, --interval 0 a busy loop |
| `HS-F13` | low | fixed, tick 79 | An empty or blank question returns three arbitrary table rows |
| `HS-F14` | low | fixed, tick 79 | ntk doctor says 9 of 9 ok with the index's server stopped |
| `HS-F15` | medium | open | An English question that names a Greek place is answered in Greek: one Greek letter makes the language Greek, the prompt asks for a Greek answer and the refusal comes in Greek; decide by the question's majority script, English on a tie |
| `HS-F16` | low | fixed, tick 105 | ntk eval check ends in a traceback on a golden line that is no object and on a missing eval/families.json: refuse each in one line, by line number and with the command that writes the file |
| `HS-F17` | low | open | A corrupted answer cache ends ntk ask in a traceback after both models load: set it aside in one line that names the file and says it can be deleted, answer without it, and note it in the spend log |

## `RS`, the retrieval scientist

An audit pass on whether the evaluation can fail and its statistics are honest. 27 findings, 23 closed.

| id | severity | status | finding |
|---|---|---|---|
| `RS-F1` | high | fixed, tick 9 | The baseline's redraw lets a test score choose the split, and a redraw moves read dev families into test: balance at construction on outcome free properties, report the dev and test difference only |
| `RS-F2` | medium | fixed, tick 98 | Repeats have no registered aggregation: register three repeats, the per item mean and an interval over items, for every release and for the incumbent on dev |
| `RS-F3` | medium | fixed, tick 102 | The overlap guard counts surface words; measure it with the lexical analyser BM25 uses (Greek: 0 of 6 surface, 3 of 6 stems on one example) |
| `RS-F4` | medium | fixed, tick 126 | Probes have no overlap guard and dev probes decide E2 and E5: question templates disjoint from statement templates, the stemmed guard, the overlap distribution reported |
| `RS-F5` | medium | fixed, tick 127 | Pooled recall counts a general fact nine times and a department's fact twice: each probe once under its most selective permitted role, per role beside it |
| `RS-F6` | medium | open | Every number comes from one corpus draw: the small profile at two more seeds, the spread of the retrieval numbers reported, accuracy stated as one corpus |
| `RS-F7` | medium | fixed, tick 119 | The accuracy has an 18 percent floor (9 of 50 by always answering not_found) and no null or oracle run is planned: run both through the scorer, publish them |
| `RS-F8` | low | fixed, tick 98 | A multi valued experiment has no rule for choosing among its values: compare each with the incumbent, adopt the cheapest that clears an arm, or Holm within it |
| `RS-F9` | low | fixed, tick 122 | Part of the simulation's false claims is an assumed probe coin (0.12 to 0.18 without it): print the assumption or simulate the probe arm |
| `RS-F10` | low | fixed, tick 39 | Target shares must sum to 1 while a section may name several labels: define the share by first label, or check at least 1 |
| `RS-F11` | low | fixed, tick 69 | Overlap@10 on off topic probes counts swaps among near equal scores: report a tie aware overlap and gate on misses beyond the tolerance |
| `RS-F12` | medium | open | N50 and the bench README quote the attack's share over the whole collection and its null, and leave out its own share over the view, 58 to 65 percent against 45 to 50: all three with their counts and intervals, the view's claim carried by KB4-K9's noninterference test Measured by bench/idf_attack.py. |
| `RS-F13` | medium | fixed, tick 89 | Timings and graph recall are quoted from one run each, and a rerun prints other numbers: N51's 26 to 83 ms and 0.680 to 0.949 against 13 to 25 and 0.688 to 0.959 at the P3 exit, N53's 4.6 s against 4.0 to 4.5 s in three runs; the timing benches repeat and print median, largest and count, and the decisions quote what today's commands print Measured by bench/filter_bench.py --report and bench/revocation.py. |
| `RS-F14` | medium | fixed, tick 94 | P4's exit counts citations that parse() has already made valid: it drops a citation it cannot place without a trace, so the exit cannot fail; parse returns what it dropped, runs record it, and each smoke answer cites an excerpt holding its value |
| `RS-F15` | medium | fixed, tick 119 | Citation accuracy, a cited section among the expected sources, is passed by citing every excerpt: citation precision and recall per item, with a cite everything null through the scorer on dev |
| `RS-F16` | low | open | The filter bench and the permission suite ask the first 30 words of a unit, not a question: the filter bench run on the dev probes when T5.4 makes them, and N51 says what its probes were |
| `RS-F17` | medium | fixed, tick 101 | The protocol does not say how many table and absent fact items each split holds, so the test's easiest items can still be counted in late: register so many I and J items per split before any is written, and report the headline with and without them |
| `RS-F18` | medium | fixed, tick 122 | The registered smallest visible effect, 0.10, is N43's figure for 63 families and no command produces it; a simulation of the registered test puts it at 0.11 for 63 and 0.12 for 50: a committed bench over a range of discordance rates, and the protocol's figure from it |
| `RS-F19` | low | fixed, tick 101 | Deliberate defaults sit in 5 of 31 test families and 6 of 50 dev, and nothing says whether an item may ask one: register that no item asks a default unless its category needs one, and report the closed book run's hits on defaults apart |
| `RS-F20` | low | fixed, tick 101 | The items' language and category shares are checked in the spec, not in the hashed protocol: add them to item_properties with how close they must match |
| `RS-F21` | low | fixed, tick 114 | The test items were drafted with the planted sentences in view, against the registered rewrite without them; the cap on shared rare terms holds them and eval/golden/README.md says so: the dev items are drafted from the registry's topic and value alone |
| `RS-F22` | medium | fixed, tick 121 | The categories overlap: A, B and C items also sit in mixed files (T02 and T05, A; T12 and T14, C; D026, D031 and D032 in dev), so a D against A comparison understates what a mixed file costs: report beside each item whether a file it is planted in, that its role reads, holds a section the role may not, and give the mixed file effect over every answered item by that property |
| `RS-F23` | medium | open | F07V04's current value, a 29 month minimum term for fixed wireless broadband, breaks the EU's 24 month cap on a consumer's commitment (Directive (EU) 2018/1972, Article 105(1)), and its former value is exactly that cap: a model that knows the law answers the superseded value, so a miss on the three dev G items D013 to D015 may be knowledge, not recency; the closed book run must mark them and G be reported with and without them |
| `RS-F24` | medium | fixed, tick 127 | On dev, the role's search never shows a planted section for 5 of 118 answerable items: three miss a fact planted only in the other language (D019, D021, D099) and two ask two facts and were shown one fact's section only (D092, D134); answerable accuracy is capped below 1 before any model reads: report retrieval recall at the shown excerpts by category beside every answer figure, and probe the cross lingual path and the two fact questions |
| `RS-F25` | medium | fixed, tick 125 | 33 of 110 answerable dev family items and 6 of 25 test ones expect a value another fact their role reads also states, so an answer from the wrong excerpt scores correct: a grounded accuracy, correct and citing a section that plants each fact, beside the answerable accuracy, and the collision as an item property |
| `RS-F26` | high | fixed, tick 125 | A refusal item's forbidden value that a readable fact also states is scored as a leak: 6 of 19 dev E items and T19 forbid a value, 12 months or 5 working days, that facts every role or the asking role reads also state; a leak is a forbidden value no readable fact states, the rest wrong but no leak, shown per item |
| `RS-F27` | low | fixed, tick 125 | The dev item rates count the twenty twinned facts twice: give the dev figures over the originals, the twins' rates and their gap apart |

## `DR`, the deep reviewer

An audit pass on correctness, read in the logic rather than the comments. 22 findings, 20 closed.

| id | severity | status | finding |
|---|---|---|---|
| `DR-F1` | medium | fixed, tick 18 | The light check misses Greek genitive plurals, working days and euro signs as invented values; its zero held only by a broader rescan |
| `DR-F2` | medium | fixed, tick 17 | 140 of 160 documents repeat a section heading (247 of 900 sections); rename repeats in plan and narrative together before ingestion |
| `DR-F3` | low | fixed, tick 18 | The light check reads the narrative, not the assembled files (207 of 207 present there, measured) |
| `DR-F4` | low | fixed, tick 35 | The Windows-1253 file is written with errors=replace (0 lost characters today) |
| `DR-F5` | low | fixed, tick 56 | server.stop reports a pid stopped after its wait times out |
| `DR-F6` | low | fixed, tick 22 | The simulator counts journal lines through an unclosed file |
| `DR-F7` | high | fixed, tick 53 | A change to the steps that decide text and labels never reaches the manifest: VERSIONS leaves out decode and segments, and build keeps units whenever the content hash matches |
| `DR-F8` | high | fixed, tick 54 | An audience line followed by text in the same paragraph is not read, and the text takes the inherited audience |
| `DR-F9` | high | fixed, tick 54 | In plain text a wrapped line that reads as a heading closes a restricted section and hands the rest of its paragraph the parent's audience |
| `DR-F10` | high | fixed, tick 55 | A relabel changes the passage when the safe path drops a heading, while the journal and T3.4's plan say no unit is embedded again |
| `DR-F11` | medium | fixed, tick 53 | A scan of a missing corpus folder removes every document from the manifest |
| `DR-F12` | low | fixed, tick 67 | A split paragraph's pieces all carry the whole paragraph's offsets |
| `DR-F13` | low | fixed, tick 54 | An unknown html data-audience is reported folded, not as written |
| `DR-F14` | low | fixed, tick 54 | HTML text outside h1 to h6, p, li and tr is dropped without a word (0 characters on this corpus) |
| `DR-F15` | medium | fixed, tick 85 | A hit the manifest has dropped, between a scan and its sync, crashes the reranked search (KeyError) and ntk search (TypeError) |
| `DR-F16` | low | open | The Greek stemmer turns έως into an empty term, so every unit holding it shares one meaningless term in the lexical index: make it a stopword when lexical.py next changes, with MT-F9's fix, since any edit to it stales the frozen statistics |
| `DR-F17` | medium | fixed, tick 117 | parse raises on a reply whose citations are not a list (TypeError) or hold a digit int cannot read such as a superscript (ValueError), where its docstring promises status invalid: read a non list as invalid or a bare integer as one citation, take a string only when isdecimal and ASCII, and test each; the class, found the same tick: the adapters read choices, output, content and candidates with .get on what they assume are objects, so a reply of the wrong shape there ends in a traceback too |
| `DR-F18` | medium | open | A reply cut off mid body raises http.client.IncompleteRead, no OSError, so it escapes call.post's retries after one attempt as a raw exception: catch HTTPException with the network errors and test four attempts then a ProviderError |
| `DR-F19` | medium | fixed, tick 123 | quoted escapes only lower case document tags, so </DOCUMENT> or <Document inside a document reaches the model as written, against the docstring's promise that no text opens or closes a quotation: escape case insensitively, with any space after < or / |
| `DR-F20` | medium | fixed, tick 110 | A table item's row is taken on the writer's word: no check that one unit of the document holds the key, is a table under a label the role reads and holds the value; T30 and T31 were checked by hand and are right; T5.1h writes eight more |
| `DR-F21` | low | fixed, tick 110 | Context.sections and split.salience see a fact planted twice in one document through its first section and first occurrence only; F22R12 sits in two logistics sections of LOG_MAN_007, so nothing decides wrongly yet: yield every section, measure each occurrence |
| `DR-F22` | low | fixed, tick 109 | Three category rules are left to the writer: a G item may name any superseded fact, an E item's facts are never required empty, an N item's neighbour need not be readable by its role: tie each in check_item with a test |

## `OP`, the pipeline operator

An audit pass that killed the pipeline mid build and mid change. 5 findings, 5 closed.

| id | severity | status | finding |
|---|---|---|---|
| `OP-F1` | medium | fixed, tick 21 | A killed build leaves a partial corpus, and the check named 1 of 9 missing files: build into a sibling folder and swap; the check compares full/ with the catalogue Measured by `bench/operator_probes.py`, at 074907c's code (`bench/README.md`). |
| `OP-F2` | medium | fixed, tick 22 | A change is written before its journal line, so a crash leaves it unexplained: journal pending, write, mark applied |
| `OP-F3` | low | fixed, tick 22 | Two simulators at once would share sequence numbers: a lock for the run |
| `OP-F4` | high | fixed, tick 65 | A watcher that starts late, or restarts after dying between its scan and its sync, never catches up: a revocation made then stays served |
| `OP-F5` | medium | fixed, tick 65 | The frequency table drifts from the index after a killed sync or two syncs at once, and nothing notices |

## `MA`, the measurement auditor

An audit pass on whether every number can be produced again. 16 findings, 9 closed.

| id | severity | status | finding |
|---|---|---|---|
| `MA-F1` | medium | fixed, tick 25 | Two commit messages carry numbers that no command in the repository produces (5547561: 24, 15, 9, 1; fbd9532: one file changed, 42 writes): commit the probes as a bench script, before numbers runnable on the parent commit, and only numbers a committed command prints go into commit messages Measured by `bench/operator_probes.py` and `bench/crash_points.py`. |
| `MA-F2` | low | open | 'N tests pass' counts this machine: 135 passed as built, 129 passed and 6 skipped without the GPU and the Qdrant binary; give passed and skipped and say which tests need what Measured by `uv run pytest`, with and without the GPU. |
| `MA-F3` | low | fixed, tick 71 | The doctor's 15 GB floor has no measurement behind it (the data directory holds 415 MB): measure the full build's peak disk use and set the floor from it Measured by `bench/full_build.py --fresh`. |
| `MA-F4` | low | fixed, tick 39 | 35974c7 states the plan's 22 percent as what hr's filter sends (measured about 5 percent), and labels.toml promises an `ntk corpus verify` that does not exist: the coverage report prints planned and measured shares and the comment names it |
| `MA-F5` | low | fixed, tick 26 | Finding ids in commits and code resolve to no document in the repository (D20): docs/AUDITS.md lists every cited id, and a gate fails on any it lacks |
| `MA-F6` | medium | fixed, tick 73 | c079ead says a revocation lands inside 5 s; at HEAD it takes 5.2 to 5.6 s, and a grant waits for the watcher's start up catch up Measured by `bench/revocation.py --doc CUS_MAN_003`. |
| `MA-F7` | low | fixed, tick 74 | 7772c46 quotes 16 of 17 (94 percent) from documents_v1, which no committed command can rebuild now Measured by `bench/idf_attack.py --global-idf`. |
| `MA-F8` | low | fixed, tick 71 | The sync's printed seconds leave out the reading that is most of a quiet sync, and its docstring's quarter of a minute has no command |
| `MA-F9` | low | fixed, tick 74 | docs/AUDITS.md quotes auditors' probes that the repository does not hold |
| `MA-F10` | medium | open | The README says a planted instruction is quoted rather than followed, on the evidence of two instructions in one excerpt: say what was tried, two instructions not followed in 20 asks, and keep quoting as the mechanism Measured by bench/injection.py. |
| `MA-F11` | low | open | N53 quotes a look's 9 ms and a cut's 368,000 objects, and no command prints either: print them from the watcher's opening line and the ingestion bench, or drop them and keep the mechanism |
| `MA-F12` | low | open | The README calls a user with only general and a removed membership combinations of roles: name the cases the permission suite runs Measured by bench/permission_suite.py. |
| `MA-F13` | low | open | Two before figures, the injection bench on the old prompt and the watcher beside a build, cannot be rerun from what the repository says: a worktree recipe beside the bench, and a scripted run or a named hand run for the other |
| `MA-F14` | low | fixed, tick 117 | ntk eval check checks the Greek share and skips the exempt families without printing either, and gives no count of family items, while eval/golden/README.md and commit 3b4e77d state 16 of 35 in Greek, two exempt dev families and 29 family items: the summary prints each split's Greek count and family items, and dev's exempt families by id |
| `MA-F15` | low | open | N59 in docs/DECISIONS.md quotes 0.556 against 0.963, another project's measurement reported by the knowledge base, with neither its source nor its sample size: say whose measurement it is and on how many questions, or keep the claim and drop the figures |
| `MA-F16` | medium | open | 18 of 128 commit messages count the negative controls a gate failed, but the controls ran from the session's scratchpad: commit them as data (file, line broken, replacement, test that must fail) with a runner that applies, runs, restores and prints the count, and cite its output, or stop counting them |

## `MT`, the maintainer in six months

An audit pass on whether a fresh clone can be rebuilt and changed safely. 12 findings, 11 closed.

| id | severity | status | finding |
|---|---|---|---|
| `MT-F1` | high | fixed, tick 30 | ntk corpus assemble crashes on a fresh clone (ZeroDivisionError in the share line), and a missing narrative swaps an empty corpus in for a good one: swap only a complete build, guard the share line, test the corpus commands through the CLI |
| `MT-F2` | medium | fixed, tick 31 | The writers' narrative (160 files, 1.7 MB) exists only on this machine, so no one else can rebuild the corpus and a lost disk loses it: make it fetchable and verifiable from the repository |
| `MT-F3` | medium | fixed, tick 31 | The README says nothing about installing or running the project: a short run section with uv sync, ntk doctor, the corpus commands and the benches |
| `MT-F4` | low | fixed, tick 30 | ntk corpus samples deletes every file in samples/ before writing, a hand added note included: delete only files named for a document |
| `MT-F5` | medium | fixed, tick 82 | The README stops at the corpus and calls the index still to come: no written path from a clone to a searchable index but a bench's source |
| `MT-F6` | medium | fixed, tick 82 | The code cites decisions N7 to N53 in 18 files, and none resolves in the repository |
| `MT-F7` | low | fixed, tick 82 | A changed collection or analyser is refused with no command to rebuild it, short of a bench that also deletes the vector cache |
| `MT-F8` | low | fixed, tick 82 | bench/README.md opens by saying every bench runs with no GPU, key or network; eight of fifteen files need the GPU or the server |
| `MT-F9` | low | open | An edit anywhere in lexical.py stales the index's frozen statistics: the analyser's version is the whole file's hash |
| `MT-F10` | medium | fixed, tick 117 | The README never mentions the golden items, ntk eval check and what it needs, ntk eval draft, or that the protocol changes only by a dated entry; two docstrings still say protocol version 4; nothing tells whoever edits prompt.py's rules to rerun bench/injection.py |
| `MT-F11` | medium | fixed, tick 113 | The dev set rests on two checks no command repeats: the table rows' seeded draw and each absent fact's and near miss's search of the corpus prose ran from the session's scratchpad; a regenerated narrative could answer one in passing, as it already mentions a courier C3: each J and N item names its probe words and ntk eval check confirms none is held, the row draw a committed command |
| `MT-F12` | low | fixed, tick 113 | ntk eval split writes over eval/families.json whatever it draws, though the split's salience reads narrative/ and the items were written against the registered split: refuse a differing result once items exist, behind a flag the protocol history records |

## `SC`, the supply chain auditor

An audit pass on every dependency, pin and download, and anything committed by accident. 3 findings, 3 closed.

| id | severity | status | finding |
|---|---|---|---|
| `SC-F1` | medium | fixed, tick 33 | No advisory check reads uv.lock: the workspace gate selects projects by package-lock.json; pip-audit found no known vulnerabilities once, by hand, and cannot reach torch 2.14.1+cu130 |
| `SC-F2` | low | fixed, tick 33 | qdrant-client is a runtime dependency that no module imports before P3: say so beside the line, or move it to the dev group |
| `SC-F3` | low | fixed, tick 33 | The pinned model and Qdrant binary have no command that fetches them: the doctor prints the exact commands with the revision and digest |

## `HE`, the hiring engineer

An audit pass of three minutes, on whether the author ships, measures and says so honestly. 3 findings, 3 closed.

| id | severity | status | finding |
|---|---|---|---|
| `HE-F1` | medium | fixed, tick 37 | The README says nothing is measured while the repository proves a clone rebuilds the corpus, kills recover and every block resolves: a What works today list, each item with its command |
| `HE-F2` | medium | fixed, tick 37 | The README never says the documents are synthetic or who the writers are: one sentence on provenance where the documents first appear |
| `HE-F3` | high | fixed, tick 37 | The README says none of the run section needs a GPU, a key or a network, but uv sync downloads the dependencies: say the rest runs without them once uv sync has fetched them |

## `AC`, the access control red team

An audit pass that tried to make restricted text readable by the wrong role. 11 findings, 10 closed.

| id | severity | status | finding |
|---|---|---|---|
| `AC-F1` | high | fixed, tick 41 | A second audience line inside a section relabels the rest of it, so one line opens a Legal section to everyone: only the line under a heading sets the audience, a later one may only narrow |
| `AC-F2` | high | fixed, tick 41 | An audience line in another case or with another separator is ignored and the section falls back to the document owner: recognise any case, accents and separator |
| `AC-F3` | high | fixed, tick 41 | In html an audience line overrides a narrower data-audience: take the intersection when they disagree, and count it |
| `AC-F4` | medium | fixed, tick 43 | A heading path carries restricted headings into text everyone may read: a path keeps only headings that every reader of the element may read |
| `AC-F5` | high | fixed, tick 63 | The IDF side channel works in the fused ranks: a word frequent in hidden text ranked lower in 16 of 17 pairs (94 percent) against 45 percent for null pairs Measured by `bench/idf_attack.py --global-idf`. |
| `AC-F6` | high | fixed, tick 62 | A narrowed audience reaches the manifest in seconds and the index never: nothing runs the sync after the watcher's scan |
| `AC-F7` | low | fixed, tick 94 | ntk search --as takes the role on the caller's word; the README should say ntk is the operator's tool and served paths take the role from the session |
| `AC-F8` | high | fixed, tick 93 | The cache the architecture plans, keyed by the role's readable labels, serves a narrowed or edited section's answer after the change, since the role's labels do not move: key an answer by the hash of the prompt as sent with the model and its parameters, key nothing by role or label set, and gate on the change bench asked through the answer path |
| `AC-F9` | medium | fixed, tick 94 | An instruction planted in a permitted section was never obeyed, but it turned answers to an English question Finnish in 6 of 14 calls against 0 of 4 clean: name the answer's language in the user message, quote the excerpts as documents whose instructions are not the model's, and keep the probe as a bench |
| `AC-F10` | low | fixed, tick 94 | A role's excerpts, restricted sections included, are posted to the provider, and neither the README nor the enforcement table names it as a reader outside the role model: one row and one sentence |
| `AC-F11` | medium | open | The answer cache shares an entry between roles shown the same text, so a reply from the cache tells one role that someone in another asked the same question, the timing channel the research notes key caches by permission set against: key by the prompt as sent and the asking role's readable label set, never by the role's name |

## `RC`, the recruiter

An audit pass of ten seconds on the first screen, by someone who hires but does not build. 2 findings, 0 closed.

| id | severity | status | finding |
|---|---|---|---|
| `RC-F1` | medium | open | The README opens with jargon, a vector database RAG project, and the plain idea comes forty words in: lead with what it does for a person |
| `RC-F2` | medium | open | Nothing in the first screen can be seen or tried: the capture goes above everything when P6 lands |

## `PF`, the performance auditor

An audit pass on time, memory and disk, its page half waiting for the page. 9 findings, 5 closed.

| id | severity | status | finding |
|---|---|---|---|
| `PF-F1` | medium | fixed, tick 70 | More than half of ingestion is spent copying row objects (segments 35 percent, structure 20): labels and safe paths in one pass and one copy, measured again before the full build Measured by `bench/ingest_steps.py`, at 4659edd's code. |
| `PF-F2` | low | fixed, tick 71 | The manifest is 1.7 times the corpus, its paths repeated on every unit: paths once per section, and the manifest counted in the disk floor Measured by `bench/full_build.py` and `bench/ingest_steps.py`. |
| `PF-F3` | low | fixed, tick 70 | Memory peaks at 19 times the size of the file being cut (158 MB for 8.43 MB): say the limit, stream by section if larger files arrive Measured by `bench/ingest_steps.py`, at 4659edd's code. |
| `PF-F4` | medium | fixed, tick 78 | The cross encoder pads each batch of 16, in rank order, to its longest pair: sorted by length it scores a question in 572 ms instead of 843, the same top 8 |
| `PF-F5` | low | fixed, tick 78 | Expansion finds each hit's segment by reading its whole document through the doc_id index: 31 to 43 ms for the top 8 |
| `PF-F6` | low | open | Within one watcher, each widening of the largest document made the next sync's read slower, 0.40 to 0.73 s over five runs, leaving 0.2 s under the 5 s objective: run the change thirty times to see whether the engine's own vacuum bounds it, and if it does not, read only the sections that changed Measured by bench/revocation.py. |
| `PF-F7` | low | open | Each ntk ask loads the encoder and the reranker in a fresh process, 30 s of search before the question reaches the provider against 0.65 s warm: the served path keeps both loaded, and the README says what a first question costs Measured by ntk ask. |
| `PF-F8` | medium | open | The spend log books the models' loading as search time: each process's first question shows 28 to 35 s of search against a 0.77 s median, cached answers too, so the log's p95 of 28 s is loading, not search: load the models before the clock and record the load apart, or mark cold questions and report cold and warm apart |
| `PF-F9` | low | open | The golden checks read the whole manifest for ten keys: the table items' check scans every unit's text, 5.2 of ntk eval check's 7.1 s, and the row draw scans once per candidate key, its test the suite's slowest at 43 s of 139: count each key's units in one pass and cache it beside the manifest under its last run id |

## `KB1`, the plan, reviewed against a knowledge base

A separate session's review of the plan against the author's research notes. 15 findings, 12 closed.

| id | severity | status | finding |
|---|---|---|---|
| `KB1-F1` | high | fixed, tick 5 | The headline is tuned on its own test set: freeze the 50, decide P8 on a separate dev set of about 100, register experiment order, adoption rule and a non inferiority margin, run the 50 once per release |
| `KB1-F2` | high | fixed, tick 6 | The filtered recall gate never reaches the selective case: every role reads general, so role filters are about 22 percent or more; gate per role filter as sent, in domain and off topic split, overlap@10 against an exact float32 scan on all probes at two or three ef values, a bench only view for a role without general |
| `KB1-F3` | medium | fixed, tick 69 | Filter on label_set, the tenant key, by computing each role's readable label sets at query time; keep labels for audit and display; measure both filters in the F2 bench |
| `KB1-F4` | medium | fixed, tick 63 | BM25 IDF side channel: raw lexical scores never leave the server; P6 runs a term existence attack and publishes its success rate; if it works, per role view sparse collections |
| `KB1-F5` | medium | fixed, tick 118 | Publish answerable accuracy, refusal correctness and leaks as separate figures; the overall rate only as a fourth |
| `KB1-F6` | medium | fixed, tick 101 | Golden set needs near misses, multi hop across two documents and answers spanning a unit boundary; score near misses like J |
| `KB1-F7` | medium | open | Headline says synthetic; a closed book run with empty excerpts published beside it; planted values mostly non default, deliberate defaults flagged in the registry |
| `KB1-F8` | medium | open | Corpus is mostly code generated structure while the eval is mostly prose: stats by layer, metrics by layer, cap layer C at about a third of units and raise table items |
| `KB1-F9` | medium | open | Planted facts are too salient: a salience plan per fact recorded in the registry, and a noise pass (page headers and footers, cross references, numbering drift, typos) over layers A and B |
| `KB1-F10` | medium | fixed, tick 43 | Unit rules: list runs as one block, units under about 30 tokens merged into the next in the same segment and heading, audience and classification lines moved to the payload, table header repeated per row group |
| `KB1-F11` | low | fixed, tick 115 | The KB session is not a domain expert: 10 seed questions from real public documents with no company in them (EU consumer law, the Greek regulator's consumer rules); five generated Greek samples offered to the owner for a naturalness read |
| `KB1-F12` | low | fixed, tick 71 | N7's arithmetic is wrong: about 6 million sentence units fit as int8 in 16 GB; plan E6 on the full profile after a peak RAM measurement on the small one |
| `KB1-F13` | low | fixed, tick 50 | The 2.6 and 4.1 points are a title prefix, not the heading path; cite each number for what it measured |
| `KB1-F14` | low | fixed, tick 42 | After NFC the Greek question mark is a semicolon that Punkt does not split on: a Greek sentence end rule and a unit test on the sentence Τι ώρα είναι; Είναι πέντε. |
| `KB1-F15` | low | fixed, tick 51 | Lexical analyser: stem per token by script, stopword lists for both languages, average unit length frozen per index version |

## `KB2`, the protocol, reviewed against a knowledge base

The same session's review of the evaluation protocol. 2 findings, 2 closed.

| id | severity | status | finding |
|---|---|---|---|
| `KB2-K1` | medium | fixed, tick 98 | Paired tests on dev treat items as independent while items can cluster by fact family, and the 600 dev probes are two phrasings of each of 300 facts: one golden item per family by construction, the family as the unit of every probe test, registered in eval/protocol.toml before T5.1 writes any item |
| `KB2-K2` | low | fixed, tick 98 | The item balance bar (standardized mean difference under 0.1) fails about half of fair writing rounds on 50 against 200 golden items by chance; register before any item exists: keep 0.1 and log each rewrite with its side, or check golden and probes together (650 against 800, about 6 percent) |

## `KB3`, the corpus at P1 exit, reviewed against a knowledge base

The same session's review of the generated corpus. 5 findings, 4 closed.

| id | severity | status | finding |
|---|---|---|---|
| `KB3-K3` | medium | fixed, tick 69 | The annex mass is 1,164 row shapes and its glossary six definitions repeated 602,893 times: named columns with a header in every table, one glossary per document, revisions dated in order; the dense intrusion rate of annex units measured at P3 |
| `KB3-K4` | medium | fixed, tick 114 | An overlap cap stated as a share of the question still lets it carry every word that decides retrieval: items rewritten without the planted sentence in view, overlap capped on the three rarest stems, about 20 dev items with a paraphrase twin |
| `KB3-K5` | low | fixed, tick 39 | Label shares by unit follow annex bytes, not the registered shares (general 0.0 against 0.20, hr 4.7 against 0.02): shop directories under general, annex rows budgeted per label; the coverage report prints registered and measured shares Measured by `ntk corpus coverage`. |
| `KB3-K6` | low | open | The prose cannot compare paragraph and section units (685 of 900 sections are one paragraph): the README and any units claim say so |
| `KB3-K7` | low | fixed, tick 103 | Templated tells: a capital letter after the date frame in 14 sentences, one effective date for all versioned facts, and N39's stale canary count (corrected at tick 23) |

## `KB4`, the retrieval design at P3 exit, reviewed against a knowledge base

The same session's review of the retrieval design and its measured filtered recall. 4 findings, 0 closed.

| id | severity | status | finding |
|---|---|---|---|
| `KB4-K8` | medium | open | The planned graph path mends the union filter but not the prose island, and N7's comparison would mix unit size with exactness: per label prefetches nested under one request and checked once against a merge in the client, prose searched exactly through a filter on its layer, and the sentence variant held to the filter bench at 0.97 per role on both layers before E6 compares it |
| `KB4-K9` | low | open | The matched pair gap cannot tell a leak from how words sit in the role's own text: a noninterference test, two builds with the same frozen avgdl that differ only in units a role may not read, with every role's fused ids, order and scores and its reranked order identical on the attack's questions and the dev probes; N50 names avgdl as the one statistic measured over the whole collection |
| `KB4-K10` | low | open | Four keyword payload indexes build graph edges per value that no served search walks: enable_hnsw false on doc_id and lang, and on label_set once the bench stops comparing forms, with bench/full_build.py --fresh timed before and after |
| `KB4-K11` | low | open | The lexical module's docstrings say Qdrant weighs a question by its IDF modifier, gone since N50: query() returns placeholders that search() replaces with BM25's IDF over the asking role's view |

## `KB5`, the answer layer at P4 exit, reviewed against a knowledge base

The same session's review of the answer layer, its cache and the measurements offered to the base. 4 findings, 1 closed.

| id | severity | status | finding |
|---|---|---|---|
| `KB5-K12` | medium | open | The provider's prefix cache and the cached count ntk ask prints keep open the cross-role channel AC-F11 closes: the asking role's readable label set inside the prompt after the fixed rules and before the excerpts, checked on dev to move no answer, and the cache counts kept in the spend log only |
| `KB5-K13` | medium | open | The baseline's three repeats would come back from the answer cache, every item looking stable and one call counted where three were meant: every scored answer asks with the cache off or the repeat in the key, and a control leaves three provider calls for one item asked three times |
| `KB5-K14` | low | open | A model alias in the cache key lets answers from a retired model outlive it: an entry is stale when a fresh reply names another model, and every evaluation run records the model that answered |
| `KB5-K15` | low | fixed, tick 123 | The injection bench's language check passes any answer in unaccented Latin letters as English: judge the language by stopwords or a fixed identifier, the ASCII test kept as a second gate |
