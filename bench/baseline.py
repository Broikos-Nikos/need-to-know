"""The baseline report (T5.5d, `eval/protocol.toml` [baseline] and [reporting]): a release's runs read and published.

    uv run python bench/baseline.py eval/runs/baseline

Reads `test.jsonl` and `dev.jsonl` in the folder, each split asked as its roles with the cache off, and their closed
book runs, `test-closed.jsonl` and `dev-closed.jsonl`, with each file's provenance, and prints:

* the headline: the share of all test items answered right, each item scored by its mean over the repeats, with its
  Wilson interval, and beside it the family items alone, since J and N items are the easiest (RS-F17);
* for each split the four figures apart (KB1-F5), the grounded accuracy, the leaks, the unstable items, and the
  figures by category, layer, language, table rows shown (KB1-F8, N43) and a value a law contradicts (RS-F23);
* the closed book runs beside: the answerable items a model got right from its own knowledge, and apart the E items
  it answered with a deliberate default, which a book was never needed for (RS-F19, KB1-F7);
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
from pathlib import Path

from needtoknow.eval.golden import Context, golden_path, load_items
from needtoknow.eval.protocol import check_use
from needtoknow.eval.score import FAMILY, Sources, over_repeats
from needtoknow.eval.stats import difference, unit_means, wilson


def read(path: Path) -> tuple[list[dict], dict]:
    answers = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sidecar = path.with_name(path.name + ".provenance.json")
    return answers, json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}


def share(rate: dict) -> str:
    return "no items" if rate["rate"] is None else f"{rate['rate']:.3f} of {rate['items']}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    ctx = Context()
    sources = Sources(ctx)
    defaults = {fid for family in ctx.families.values() for fid in family.get("defaults") or {}}
    runs, items, parity = {}, {}, {}
    for split in ("test", "dev"):
        check_use(split, "baseline")
        items[split] = load_items(golden_path(split))
        for book in ("", "-closed"):
            answers, made = read(args.folder / f"{split}{book}.jsonl")
            if not book:
                shown: dict[str, set] = {}
                for answer in answers:
                    shown.setdefault(answer["id"], set()).add(json.dumps(answer.get("excerpts"), sort_keys=True))
                parity[split] = (sum(len(seen) == 1 for seen in shown.values()), len(shown))
            runs[split + book] = over_repeats(items[split], answers, ctx, sources)
            runs[split + book]["made"] = made
            runs[split + book]["usd"] = sum(answer.get("usd") or 0.0 for answer in answers)
    first = runs["test"]["made"]
    print(f"baseline at {first.get('commit', '?')[:12]}{'' if first.get('clean') else ', tree not clean'}; protocol "
          f"version {first.get('protocol_version')}; {first.get('model')} at temperature "
          f"{first.get('settings', {}).get('temperature')}; manifest run {(first.get('manifest') or {}).get('run')}, "
          f"{(first.get('manifest') or {}).get('units')} units; {len(runs['test']['repeats'])} repeats")
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
    for split in ("test", "dev"):
        closed = runs[split + "-closed"]
        on_defaults = [item["id"] for item in items[split] if item["category"] == "E" and closed["leaked"].get(item["id"])
                       and defaults & {e.get("fact_id") for e in (item.get("expect") or {}).get("forbidden") or []}]
        others = [item["id"] for item in items[split] if closed["leaked"].get(item["id"]) and item["id"] not in on_defaults]
        print(f"{split}, closed book: answerable {share(closed['answerable'])} from the model's own knowledge; "
              f"refusals {share(closed['refusals'])}; E items answered with a deliberate default "
              f"{len(on_defaults)}" + (f" ({', '.join(on_defaults)})" if on_defaults else "")
              + f"; any other forbidden value {len(others)}" + (f" ({', '.join(others)})" if others else ""))
    originals = [item for item in items["dev"] if not item.get("twin")]
    dev_units = unit_means(originals, runs["dev"]["verdicts"])
    test_units = unit_means(items["test"], runs["test"]["verdicts"])
    gap, low, high = difference(list(dev_units.values()), list(test_units.values()))
    print(f"dev less test over fact families, a diagnostic that changes nothing (N33): {gap:+.3f}, 95 percent "
          f"{low:+.3f} to {high:+.3f}, over {len(dev_units)} dev and {len(test_units)} test units")
    print("parity, the same units shown in every repeat: " + "; ".join(
        f"{split} {same} of {total} items" for split, (same, total) in parity.items()))
    print(f"cost: {sum(run['usd'] for run in runs.values()):.6f} USD over the four runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
