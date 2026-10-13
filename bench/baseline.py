"""The baseline report (T5.5d, `eval/protocol.toml` [baseline] and [reporting]): a release's runs read and published.

    uv run python bench/baseline.py eval/runs/baseline

Reads `test.jsonl` and `dev.jsonl` in the folder, each split asked as its roles with the cache off; their runs with no
documents under the answer rules, `test-closed.jsonl` and `dev-closed.jsonl`; and, where they are, their best guess
runs, `test-guess.jsonl` and `dev-guess.jsonl`, with each file's provenance, and prints:

* what made the runs: the test run's commit, protocol, model and manifest, then every field of the provenance on
  which the runs differ, named run by run, and the runs whose tree held changes (DR-F28);
* the headline: the share of all test items answered right, each item scored by its mean over the repeats, with its
  Wilson interval, and beside it the family items alone, since J and N items are the easiest (RS-F17);
* for each split the four figures apart (KB1-F5), the grounded accuracy, the leaks, the unstable items, and the
  figures by category, layer, language, table rows shown (KB1-F8, N43) and a value a law contradicts (RS-F23);
* the runs with no documents beside: under the answer rules, how many answers were refusals, the check that the rules
  hold, which shows nothing of what the model knows; and the best guess, the control KB1-F7 asks for: the answerable
  items the model got right from its own knowledge alone, apart the E items it answered with a deliberate default and
  G on the value a law contradicts (RS-F19, RS-F23, DR-F27);
* dev less test over fact families, with a percentile bootstrap interval of the difference, a diagnostic that changes
  nothing (N33);
* the release's parity on its own build: how many items were shown the same units in every repeat, the retrieval the
  protocol requires to be the same before a test run.

Needs the plan, the frequency table and the manifest the scorer reads; asks the protocol for each split's purpose.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from needtoknow.eval.golden import Context, golden_path, load_items
from needtoknow.eval.protocol import check_use
from needtoknow.eval.score import FAMILY, Sources, over_repeats, scorer_hash
from needtoknow.eval.stats import difference, unit_means, wilson


def read(path: Path) -> tuple[list[dict], dict]:
    answers = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sidecar = path.with_name(path.name + ".provenance.json")
    return answers, json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}


def share(rate: dict) -> str:
    return "no items" if rate["rate"] is None else f"{rate['rate']:.3f} of {rate['items']}"


FIELDS = ("commit", "protocol", "model", "settings", "manifest", "analyser")


def differences(made: dict[str, dict]) -> list[str]:
    """Each provenance field on which the runs differ, with its value run by run, then the runs whose tree held
    changes: the report names what made each of its numbers, not only what made the first run (DR-F28)."""
    lines = []
    for field in FIELDS:
        values: dict[str, list[str]] = {}
        for name, provenance in made.items():
            values.setdefault(json.dumps(provenance.get(field), sort_keys=True), []).append(name)
        if len(values) > 1:
            lines.append(f"{field} differs: " + "; ".join(
                f"{', '.join(names)} at {value.strip(chr(34))[:12] if field == 'commit' else value[:80]}"
                for value, names in values.items()))
    unclean = [name for name, provenance in made.items() if not provenance.get("clean")]
    if unclean:
        lines.append(f"tree not clean when made: {', '.join(unclean)}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    ctx = Context()
    sources = Sources(ctx)
    defaults = {fid for family in ctx.families.values() for fid in family.get("defaults") or {}}
    runs, items, parity, statuses = {}, {}, {}, {}
    for split in ("test", "dev"):
        check_use(split, "baseline")
        items[split] = load_items(golden_path(split))
        for book in ("", "-closed", "-guess"):
            if book == "-guess" and not (args.folder / f"{split}{book}.jsonl").is_file():
                continue
            answers, made = read(args.folder / f"{split}{book}.jsonl")
            if not book:
                shown: dict[str, set] = {}
                for answer in answers:
                    shown.setdefault(answer["id"], set()).add(json.dumps(answer.get("excerpts"), sort_keys=True))
                parity[split] = (sum(len(seen) == 1 for seen in shown.values()), len(shown))
            runs[split + book] = over_repeats(items[split], answers, ctx, sources)
            runs[split + book]["made"] = made
            runs[split + book]["usd"] = sum(answer.get("usd") or 0.0 for answer in answers)
            statuses[split + book] = Counter(answer["status"] for answer in answers)
    first = runs["test"]["made"]
    print(f"baseline at {first.get('commit', '?')[:12]}; protocol version {first.get('protocol_version')}; "
          f"{first.get('model')} at temperature {first.get('settings', {}).get('temperature')}; manifest run "
          f"{(first.get('manifest') or {}).get('run')}, {(first.get('manifest') or {}).get('units')} units; "
          f"{len(runs['test']['repeats'])} repeats; scorer {scorer_hash()[:16]}")
    made = {name: run["made"] for name, run in runs.items()}
    for line in differences(made) or [f"every run made at the same {', '.join(FIELDS)}"]:
        print(f"  {line}")
    test = runs["test"]
    scores = test["scores"]
    family = [item for item in items["test"] if item["category"] in FAMILY]
    low, high = wilson(sum(scores.values()), len(scores))
    family_rate = sum(scores[item["id"]] for item in family) / len(family)
    print(f"headline: all {len(scores)} test items, {sum(scores.values()) / len(scores):.3f} answered right, Wilson "
          f"{low:.3f} to {high:.3f}; the {len(family)} family items alone {family_rate:.3f}")
    for split in ("test", "dev"):
        run = runs[split]
        print(f"{split}: answerable {share(run['answerable'])}; refusals {share(run['refusals'])}; overall "
              f"{share(run['overall'])}; grounded {share(run['grounded'])}; leaks {run['leaks']}; unstable "
              f"{len(run['unstable'])}" + (f" ({', '.join(run['unstable'])})" if run["unstable"] else ""))
        for axis, groups in run["by"].items():
            print(f"  by {axis}: " + "; ".join(f"{key} {g['rate']:.3f} of {g['items']}" for key, g in groups.items()
                                               if g["rate"] is not None))
        misses = run["misses"]
        print(f"  wrong answers to answerable items over the repeats: {len(misses['not shown'])} not shown a section "
              f"planting every fact they needed" + (f" ({', '.join(misses['not shown'])})" if misses["not shown"]
                                                    else "") + f"; {len(misses['shown'])} shown every one"
              + (f" ({', '.join(misses['shown'])})" if misses["shown"] else ""))
        print(f"  verdicts that differ between two repeats: {run['discordance']:.4f} of the items, over every pair")
    for split in ("test", "dev"):
        counts = statuses[split + "-closed"]
        print(f"{split}, no documents under the answer rules: {counts['not_found']} of {sum(counts.values())} "
              "answers refused, which shows the rules hold and nothing of what the model knows")
        if split + "-guess" not in runs:
            print(f"{split}, the best guess with no documents: no {split}-guess.jsonl in {args.folder}")
            continue
        guess, counts = runs[split + "-guess"], statuses[split + "-guess"]
        on_defaults = [item["id"] for item in items[split] if item["category"] == "E" and guess["leaked"].get(item["id"])
                       and defaults & {e.get("fact_id") for e in (item.get("expect") or {}).get("forbidden") or []}]
        others = [item["id"] for item in items[split] if guess["leaked"].get(item["id"]) and item["id"] not in on_defaults]
        law = guess["by"].get("G, on a value a law contradicts", {}).get("yes")
        print(f"{split}, the best guess with no documents: answerable {share(guess['answerable'])} right from the "
              f"model's own knowledge alone; {counts['not_found']} of {sum(counts.values())} answers declined; E items "
              f"answered with a deliberate default {len(on_defaults)}" + (f" ({', '.join(on_defaults)})" if on_defaults
                                                                          else "")
              + f"; any other forbidden value {len(others)}" + (f" ({', '.join(others)})" if others else "")
              + (f"; G on the value a law contradicts {share(law)}" if law else ""))
    originals = [item for item in items["dev"] if not item.get("twin")]
    dev_units = unit_means(originals, runs["dev"]["verdicts"])
    test_units = unit_means(items["test"], runs["test"]["verdicts"])
    gap, low, high = difference(list(dev_units.values()), list(test_units.values()))
    print(f"dev less test over fact families, a diagnostic that changes nothing (N33): {gap:+.3f}, 95 percent "
          f"{low:+.3f} to {high:+.3f}, over {len(dev_units)} dev and {len(test_units)} test units")
    print("parity, the same units shown in every repeat: " + "; ".join(
        f"{split} {same} of {total} items" for split, (same, total) in parity.items()))
    print(f"cost: {sum(run['usd'] for run in runs.values()):.6f} USD over the {len(runs)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
