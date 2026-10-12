# The audits

Commit messages and code comments cite identifiers like `OP-F1`. This is what they refer to.

**69 findings, 34 closed, 35 open**, from 12 reviews. Each audit pass took one perspective and nothing else.

Written from the loop's own queue by its `tools/audits_md.py`, and held to that queue by the workspace gate `tools/check-audit-status.mjs`, which fails if a row says anything the queue does not.

## `HS`, the hostile stranger

An audit pass that tried to break a first run. 8 findings, 8 closed.

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

## `RS`, the retrieval scientist

An audit pass on whether the evaluation can fail and its statistics are honest. 11 findings, 2 closed.

| id | severity | status | finding |
|---|---|---|---|
| `RS-F1` | high | fixed, tick 9 | The baseline's redraw lets a test score choose the split, and a redraw moves read dev families into test: balance at construction on outcome free properties, report the dev and test difference only |
| `RS-F2` | medium | open | Repeats have no registered aggregation: register three repeats, the per item mean and an interval over items, for every release and for the incumbent on dev |
| `RS-F3` | medium | open | The overlap guard counts surface words; measure it with the lexical analyser BM25 uses (Greek: 0 of 6 surface, 3 of 6 stems on one example) |
| `RS-F4` | medium | open | Probes have no overlap guard and dev probes decide E2 and E5: question templates disjoint from statement templates, the stemmed guard, the overlap distribution reported |
| `RS-F5` | medium | open | Pooled recall counts a general fact nine times and a department's fact twice: each probe once under its most selective permitted role, per role beside it |
| `RS-F6` | medium | open | Every number comes from one corpus draw: the small profile at two more seeds, the spread of the retrieval numbers reported, accuracy stated as one corpus |
| `RS-F7` | medium | open | The accuracy has an 18 percent floor (9 of 50 by always answering not_found) and no null or oracle run is planned: run both through the scorer, publish them |
| `RS-F8` | low | open | A multi valued experiment has no rule for choosing among its values: compare each with the incumbent, adopt the cheapest that clears an arm, or Holm within it |
| `RS-F9` | low | open | Part of the simulation's false claims is an assumed probe coin (0.12 to 0.18 without it): print the assumption or simulate the probe arm |
| `RS-F10` | low | fixed, tick 39 | Target shares must sum to 1 while a section may name several labels: define the share by first label, or check at least 1 |
| `RS-F11` | low | open | Overlap@10 on off topic probes counts swaps among near equal scores: report a tie aware overlap and gate on misses beyond the tolerance |

## `DR`, the deep reviewer

An audit pass on correctness, read in the logic rather than the comments. 6 findings, 5 closed.

| id | severity | status | finding |
|---|---|---|---|
| `DR-F1` | medium | fixed, tick 18 | The light check misses Greek genitive plurals, working days and euro signs as invented values; its zero held only by a broader rescan |
| `DR-F2` | medium | fixed, tick 17 | 140 of 160 documents repeat a section heading (247 of 900 sections); rename repeats in plan and narrative together before ingestion |
| `DR-F3` | low | fixed, tick 18 | The light check reads the narrative, not the assembled files (207 of 207 present there, measured) |
| `DR-F4` | low | fixed, tick 35 | The Windows-1253 file is written with errors=replace (0 lost characters today) |
| `DR-F5` | low | open | server.stop reports a pid stopped after its wait times out |
| `DR-F6` | low | fixed, tick 22 | The simulator counts journal lines through an unclosed file |

## `OP`, the pipeline operator

An audit pass that killed the pipeline mid build and mid change. 3 findings, 3 closed.

| id | severity | status | finding |
|---|---|---|---|
| `OP-F1` | medium | fixed, tick 21 | A killed build leaves a partial corpus, and the check named 1 of 9 missing files: build into a sibling folder and swap; the check compares full/ with the catalogue |
| `OP-F2` | medium | fixed, tick 22 | A change is written before its journal line, so a crash leaves it unexplained: journal pending, write, mark applied |
| `OP-F3` | low | fixed, tick 22 | Two simulators at once would share sequence numbers: a lock for the run |

## `MA`, the measurement auditor

An audit pass on whether every number can be produced again. 5 findings, 3 closed.

| id | severity | status | finding |
|---|---|---|---|
| `MA-F1` | medium | fixed, tick 25 | Two commit messages carry numbers that no command in the repository produces (5547561: 24, 15, 9, 1; fbd9532: one file changed, 42 writes): commit the probes as a bench script, before numbers runnable on the parent commit, and only numbers a committed command prints go into commit messages |
| `MA-F2` | low | open | 'N tests pass' counts this machine: 135 passed as built, 129 passed and 6 skipped without the GPU and the Qdrant binary; give passed and skipped and say which tests need what |
| `MA-F3` | low | open | The doctor's 15 GB floor has no measurement behind it (the data directory holds 415 MB): measure the full build's peak disk use and set the floor from it |
| `MA-F4` | low | fixed, tick 39 | 35974c7 states the plan's 22 percent as what hr's filter sends (measured about 5 percent), and labels.toml promises an `ntk corpus verify` that does not exist: the coverage report prints planned and measured shares and the comment names it |
| `MA-F5` | low | fixed, tick 26 | Finding ids in commits and code resolve to no document in the repository (D20): docs/AUDITS.md lists every cited id, and a gate fails on any it lacks |

## `MT`, the maintainer in six months

An audit pass on whether a fresh clone can be rebuilt and changed safely. 4 findings, 4 closed.

| id | severity | status | finding |
|---|---|---|---|
| `MT-F1` | high | fixed, tick 30 | ntk corpus assemble crashes on a fresh clone (ZeroDivisionError in the share line), and a missing narrative swaps an empty corpus in for a good one: swap only a complete build, guard the share line, test the corpus commands through the CLI |
| `MT-F2` | medium | fixed, tick 31 | The writers' narrative (160 files, 1.7 MB) exists only on this machine, so no one else can rebuild the corpus and a lost disk loses it: make it fetchable and verifiable from the repository |
| `MT-F3` | medium | fixed, tick 31 | The README says nothing about installing or running the project: a short run section with uv sync, ntk doctor, the corpus commands and the benches |
| `MT-F4` | low | fixed, tick 30 | ntk corpus samples deletes every file in samples/ before writing, a hand added note included: delete only files named for a document |

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

An audit pass that tried to make restricted text readable by the wrong role. 4 findings, 0 closed.

| id | severity | status | finding |
|---|---|---|---|
| `AC-F1` | high | open | A second audience line inside a section relabels the rest of it, so one line opens a Legal section to everyone: only the line under a heading sets the audience, a later one may only narrow |
| `AC-F2` | high | open | An audience line in another case or with another separator is ignored and the section falls back to the document owner: recognise any case, accents and separator |
| `AC-F3` | high | open | In html an audience line overrides a narrower data-audience: take the intersection when they disagree, and count it |
| `AC-F4` | medium | open | A heading path carries restricted headings into text everyone may read: a path keeps only headings that every reader of the element may read |

## `KB1`, the plan, reviewed against a knowledge base

A separate session's review of the plan against the author's research notes. 15 findings, 2 closed.

| id | severity | status | finding |
|---|---|---|---|
| `KB1-F1` | high | fixed, tick 5 | The headline is tuned on its own test set: freeze the 50, decide P8 on a separate dev set of about 100, register experiment order, adoption rule and a non inferiority margin, run the 50 once per release |
| `KB1-F2` | high | fixed, tick 6 | The filtered recall gate never reaches the selective case: every role reads general, so role filters are about 22 percent or more; gate per role filter as sent, in domain and off topic split, overlap@10 against an exact float32 scan on all probes at two or three ef values, a bench only view for a role without general |
| `KB1-F3` | medium | open | Filter on label_set, the tenant key, by computing each role's readable label sets at query time; keep labels for audit and display; measure both filters in the F2 bench |
| `KB1-F4` | medium | open | BM25 IDF side channel: raw lexical scores never leave the server; P6 runs a term existence attack and publishes its success rate; if it works, per role view sparse collections |
| `KB1-F5` | medium | open | Publish answerable accuracy, refusal correctness and leaks as separate figures; the overall rate only as a fourth |
| `KB1-F6` | medium | open | Golden set needs near misses, multi hop across two documents and answers spanning a unit boundary; score near misses like J |
| `KB1-F7` | medium | open | Headline says synthetic; a closed book run with empty excerpts published beside it; planted values mostly non default, deliberate defaults flagged in the registry |
| `KB1-F8` | medium | open | Corpus is mostly code generated structure while the eval is mostly prose: stats by layer, metrics by layer, cap layer C at about a third of units and raise table items |
| `KB1-F9` | medium | open | Planted facts are too salient: a salience plan per fact recorded in the registry, and a noise pass (page headers and footers, cross references, numbering drift, typos) over layers A and B |
| `KB1-F10` | medium | open | Unit rules: list runs as one block, units under about 30 tokens merged into the next in the same segment and heading, audience and classification lines moved to the payload, table header repeated per row group |
| `KB1-F11` | low | open | The KB session is not a domain expert: 10 seed questions from real public documents with no company in them (EU consumer law, the Greek regulator's consumer rules); five generated Greek samples offered to the owner for a naturalness read |
| `KB1-F12` | low | open | N7's arithmetic is wrong: about 6 million sentence units fit as int8 in 16 GB; plan E6 on the full profile after a peak RAM measurement on the small one |
| `KB1-F13` | low | open | The 2.6 and 4.1 points are a title prefix, not the heading path; cite each number for what it measured |
| `KB1-F14` | low | open | After NFC the Greek question mark is a semicolon that Punkt does not split on: a Greek sentence end rule and a unit test on the sentence Τι ώρα είναι; Είναι πέντε. |
| `KB1-F15` | low | open | Lexical analyser: stem per token by script, stopword lists for both languages, average unit length frozen per index version |

## `KB2`, the protocol, reviewed against a knowledge base

The same session's review of the evaluation protocol. 2 findings, 0 closed.

| id | severity | status | finding |
|---|---|---|---|
| `KB2-K1` | medium | open | Paired tests on dev treat items as independent while items can cluster by fact family, and the 600 dev probes are two phrasings of each of 300 facts: one golden item per family by construction, the family as the unit of every probe test, registered in eval/protocol.toml before T5.1 writes any item |
| `KB2-K2` | low | open | The item balance bar (standardized mean difference under 0.1) fails about half of fair writing rounds on 50 against 200 golden items by chance; register before any item exists: keep 0.1 and log each rewrite with its side, or check golden and probes together (650 against 800, about 6 percent) |

## `KB3`, the corpus at P1 exit, reviewed against a knowledge base

The same session's review of the generated corpus. 5 findings, 1 closed.

| id | severity | status | finding |
|---|---|---|---|
| `KB3-K3` | medium | open | The annex mass is 1,164 row shapes and its glossary six definitions repeated 602,893 times: named columns with a header in every table, one glossary per document, revisions dated in order; the dense intrusion rate of annex units measured at P3 |
| `KB3-K4` | medium | open | An overlap cap stated as a share of the question still lets it carry every word that decides retrieval: items rewritten without the planted sentence in view, overlap capped on the three rarest stems, about 20 dev items with a paraphrase twin |
| `KB3-K5` | low | fixed, tick 39 | Label shares by unit follow annex bytes, not the registered shares (general 0.0 against 0.20, hr 4.7 against 0.02): shop directories under general, annex rows budgeted per label; the coverage report prints registered and measured shares |
| `KB3-K6` | low | open | The prose cannot compare paragraph and section units (685 of 900 sections are one paragraph): the README and any units claim say so |
| `KB3-K7` | low | open | Templated tells: a capital letter after the date frame in 14 sentences, one effective date for all versioned facts, and N39's stale canary count (corrected at tick 23) |
