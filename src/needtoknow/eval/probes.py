"""The retrieval probes (T5.4b, `specs/evaluation.md` section 5, RS-F4): two questions a fact, made by code.

Each split's probes come from its own families' facts, a superseded value left out: one
question in English and one in Greek, from a question template written for each of the
plan's 55 statement templates and filled with the same slots the plan filled, a channel,
a product, a region, a warehouse and its courier. A question template is worded apart
from its statement: it names the subject and asks for the value, and every probe keeps to
the overlap cap the golden items keep, at most one of its planted sentence's three rarest
terms, counted by `needtoknow.lexical.terms` (RS-F3, RS-F4). The share of probes sharing
none, one, or more is printed with them, so their distance from the planted text is a
number and not a claim.

A probe's gold units are those of the manifest that hold its fact's planted sentence; its
readers, every role that may read one; its selective reader, the reader with the fewest
labels, under whom recall counts it once (RS-F5). Templated
questions sit closer to the planted text than a person's would, so their recall is an
upper bound; their worth is in comparisons. `ntk eval probes` writes `eval/probes/`, and
`--check` makes them again and compares.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from needtoknow import lexical
from needtoknow.corpus.plan import TEMPLATES, variants, world
from needtoknow.eval.golden import Context, GoldenError, rare_terms
from needtoknow.ingest.manifest import manifest_path
from needtoknow.policy import readable
from needtoknow.settings import repo_root

SPLITS = ("dev", "test")
# One question a statement template and language, {x} and {y} the plan's slots, {X} {x} capitalised, {p} the place
# alone in a warehouse's name, since "the Sindos warehouse" would hand a question two of its rarest terms.
QUESTIONS = {
    0: ("During which hours is the front desk of our main offices staffed?",
        "Ποιες ώρες εξυπηρετεί επισκέπτες η είσοδος των κεντρικών μας γραφείων;"),
    1: ("How often does a staff pass need to be reissued?", "Κάθε πότε εκδίδεται ξανά το πάσο εισόδου των εργαζομένων;"),
    2: ("What number do I dial internally for computer support?",
        "Ποιο εσωτερικό νούμερο καλώ για βοήθεια με τους υπολογιστές;"),
    3: ("How often must we reaffirm that we follow the rules of behaviour at work?",
        "Κάθε πότε δηλώνουμε ξανά ότι τηρούμε τους κανόνες συμπεριφοράς στη δουλειά;"),
    4: ("Above what value must a present from a supplier be reported?",
        "Από ποια αξία και πάνω πρέπει να αναφέρουμε ένα δώρο που μας έκανε ένας συνεργάτης;"),
    5: ("When do our shops in {x} start serving customers on a regular working day?",
        "Τι ώρα ξεκινούν να εξυπηρετούν πελάτες τα καταστήματα στην περιοχή «{x}»;"),
    6: ("How big a discount may an advisor give without sign off in {x}?",
        "Πόση έκπτωση μπορεί να δώσει ένας σύμβουλος χωρίς έγκριση στο κανάλι «{x}»;"),
    7: ("What is the shortest commitment a customer can sign for {x}?",
        "Ποια είναι η μικρότερη δέσμευση για συμβόλαιο στην κατηγορία «{x}»;"),
    8: ("What is the most we put toward a handset when a customer signs up for {x}?",
        "Μέχρι πόσα χρήματα συνεισφέρουμε στο κινητό όταν ο πελάτης υπογράφει για την κατηγορία «{x}»;"),
    9: ("After how long into a contract can a customer get a new handset?",
        "Μετά από πόσο καιρό συμβολαίου δικαιούται ο πελάτης καινούργιο κινητό;"),
    10: ("From what order value do we have to check whether a customer can pay?",
         "Από ποια αξία παραγγελίας ελέγχουμε αν ο πελάτης μπορεί να πληρώσει;"),
    11: ("What commission rate applies to selling {x}?",
         "Τι ποσοστό προμήθειας παίρνουμε για πωλήσεις στην κατηγορία «{x}»;"),
    12: ("How soon after a signed order must a line for {x} be working?",
         "Σε πόσες ώρες από την υπογραφή πρέπει να λειτουργεί μια νέα γραμμή για την κατηγορία «{x}»;"),
    13: ("How long does a customer have to bring back something from the {x} category?",
         "Σε πόσες μέρες μπορεί ο πελάτης να φέρει πίσω κάτι από την κατηγορία «{x}»;"),
    14: ("Up to what amount can a {x} sign off a refund alone?",
         "Μέχρι ποιο ποσό μπορεί να εγκρίνει μόνος του επιστροφή χρημάτων ο ρόλος «{x}»;"),
    15: ("What fee do we charge when an opened device comes back in good shape?",
         "Τι χρέωση κρατάμε όταν μια ανοιγμένη συσκευή γυρίζει πίσω χωρίς φθορές;"),
    16: ("How soon after reaching the warehouse must a returned item be checked?",
         "Σε πόσες μέρες ελέγχεται ένα προϊόν που γύρισε πίσω αφού φτάσει στην αποθήκη;"),
    17: ("How long does it take for an approved refund to reach the customer?",
         "Σε πόσο χρόνο φτάνουν στον πελάτη τα χρήματα μιας εγκεκριμένης επιστροφής;"),
    18: ("What prefix does a return authorisation code begin with?",
         "Ποιο πρόθεμα έχουν οι κωδικοί εξουσιοδότησης επιστροφής;"),
    19: ("How many returns can one customer make in a year before a supervisor reviews them?",
         "Πόσες επιστροφές μπορεί να κάνει ένας πελάτης τον χρόνο χωρίς έλεγχο από προϊστάμενο;"),
    20: ("What is the turnaround at repair centre {x}?", "Σε πόσες μέρες ολοκληρώνει μια συσκευή το κέντρο {x};"),
    21: ("What time does courier {y} make its final collection at our {p} site?",
         "Τι ώρα περνά για τελευταία φορά ο {y} από την {x};"),
    22: ("How many working days does an ordinary shipment to {x} take?",
         "Σε πόσες μέρες φτάνει μια κανονική αποστολή στην περιοχή «{x}»;"),
    23: ("How soon after an order is fibre fitted in {x}?",
         "Σε πόσες μέρες από την παραγγελία στήνεται οπτική ίνα στην περιοχή «{x}»;"),
    24: ("What is the heaviest a single pallet may be?", "Πόσο βάρος σηκώνει το πολύ μια παλέτα;"),
    25: ("Above what value must a shipment travel with insurance?",
         "Από ποια αξία και πάνω μια αποστολή πρέπει να έχει ασφάλεια;"),
    26: ("During which hours does {x} accept deliveries on a working day?",
         "Ποιες ώρες δέχεται παραδόσεις η {x} τις καθημερινές;"),
    27: ("How regularly is inventory taken at {x}?", "Κάθε πόσο μετράμε τα αποθέματα στην {x};"),
    28: ("How fast must a {x} ticket get its first reply?",
         "Σε πόσες ώρες πρέπει να απαντήσουμε πρώτη φορά σε αίτημα {x};"),
    29: ("How soon must a callback we promised take place?",
         "Σε πόσα λεπτά πρέπει να γίνει μια κλήση που υποσχεθήκαμε στον πελάτη;"),
    30: ("What is the longest we may keep someone waiting on hold?",
         "Πόση ώρα το πολύ μπορεί να περιμένει στη γραμμή ένας πελάτης;"),
    31: ("How much compensation does a customer get per complete hour of a verified service interruption?",
         "Τι αποζημίωση παίρνει ο πελάτης για κάθε ολόκληρη ώρα που έμεινε χωρίς υπηρεσία;"),
    32: ("After how many failed contacts does a complaint go to the second line?",
         "Μετά από πόσες επαφές χωρίς λύση περνά ένα παράπονο στη δεύτερη γραμμή;"),
    33: ("How long does the second line have to settle a complaint about {x}?",
         "Σε πόσες μέρες κλείνει η δεύτερη γραμμή ένα παράπονο για την κατηγορία «{x}»;"),
    34: ("How many days does a customer have to back out of a contract made at a distance?",
         "Σε πόσες μέρες μπορεί ο πελάτης να υπαναχωρήσει από σύμβαση που έκανε χωρίς να έρθει στο κατάστημα;"),
    35: ("For how long do we hold on to {x}?", "Για πόσο καιρό μένουν στο αρχείο {x};"),
    36: ("How many days do we have to send a reasoned reply to a written complaint?",
         "Σε πόσες μέρες πρέπει να απαντήσουμε σε ένα γραπτό παράπονο πελάτη;"),
    37: ("How long in advance must a business customer tell us to end a contract?",
         "Πόση προειδοποίηση πρέπει να δώσει ένας επαγγελματικός πελάτης για να λύσει τη σύμβαση;"),
    38: ("What is the most we pay out for one incident under a business contract?",
         "Ποιο είναι το ανώτατο ποσό που πληρώνουμε για ένα περιστατικό σε επαγγελματική σύμβαση;"),
    39: ("How soon must a claim on device insurance be passed to {x}?",
         "Μέσα σε πόσες μέρες πρέπει να ενημερωθεί ο συνεργάτης {x} για μια ζημιά σε ασφαλισμένη συσκευή;"),
    40: ("How many days after we receive it do we settle a supplier's bill?",
         "Σε πόσες μέρες πληρώνουμε το τιμολόγιο ενός προμηθευτή;"),
    41: ("What charge applies each month when a business customer pays late?",
         "Πόσο επιβαρύνεται κάθε μήνα ένας επαγγελματικός πελάτης που πληρώνει με καθυστέρηση;"),
    42: ("From what refund amount does finance have to approve as well?",
         "Από ποιο ποσό επιστροφής χρημάτων χρειάζεται και η υπογραφή του Οικονομικού;"),
    43: ("Below what amount can an unpaid balance be cancelled after a year?",
         "Κάτω από ποιο ποσό μπορεί να σβηστεί μια απλήρωτη οφειλή μετά από έναν χρόνο;"),
    44: ("On which day of the month does a household get the bill for {x}?",
         "Ποια μέρα του μήνα βγαίνει ο λογαριασμός για την κατηγορία «{x}»;"),
    45: ("How quickly do we need someone on site for an {x} network problem?",
         "Σε πόσες ώρες πρέπει να βρίσκεται κάποιος στο σημείο για πρόβλημα δικτύου {x};"),
    46: ("When does planned work on the network take place?", "Ποιες ώρες γίνονται οι προγραμματισμένες εργασίες στο δίκτυο;"),
    47: ("What uptime does the core network aim for each month?", "Ποια διαθεσιμότητα έχει ως στόχο κάθε μήνα το βασικό δίκτυο;"),
    48: ("How many days of parts does the depot for {x} hold?",
         "Για πόσες ημέρες έχει απόθεμα ανταλλακτικών η αποθήκη της περιοχής «{x}»;"),
    49: ("How soon must a technician visit after a fault on {x}?",
         "Σε πόσες ώρες πρέπει να πάει τεχνικός σε βλάβη στην κατηγορία «{x}»;"),
    50: ("How long is the trial period for a new hire?", "Πόσο διαρκεί η δοκιμή ενός νέου υπαλλήλου;"),
    51: ("How many days of paid holiday does a full time employee get?",
         "Πόσες μέρες άδειας τον χρόνο δικαιούται όποιος δουλεύει με πλήρες ωράριο;"),
    52: ("How many days a week can office staff work from home?",
         "Πόσες μέρες την εβδομάδα μπορεί το προσωπικό γραφείου να δουλεύει από το σπίτι;"),
    53: ("How many hours of training must each employee complete yearly?",
         "Πόσες ώρες εκπαίδευσης χρειάζεται κάθε υπάλληλος τον χρόνο;"),
    54: ("How much notice must someone who quits give?", "Πόση προειδοποίηση δίνει ένας υπάλληλος που φεύγει από τη δουλειά;"),
}


def probes_path(split: str) -> Path:
    return repo_root() / "eval" / "probes" / f"{split}.jsonl"


def slots(fact_id: str, w: dict) -> tuple[str, str, str, str]:
    """The slots the plan filled for this fact: from its template's spec, by its variant's index."""
    template = int(fact_id[1:3])
    spec = TEMPLATES[template][6]
    index = int(fact_id[4:6])
    if fact_id[3] == "R":
        return w["regions"]["en"][index], w["regions"]["el"][index], "", ""
    return variants(spec, w)[index]


def fill(question: str, x: str, y: str) -> str:
    place = x.removeprefix("the ").removesuffix(" warehouse")
    out = question.replace("{X}", x[:1].upper() + x[1:]).replace("{x}", x).replace("{y}", y).replace("{p}", place)
    return out[:1].upper() + out[1:]


def gold_units(fact: dict, ctx: Context, db: sqlite3.Connection) -> list[tuple[str, list[str]]]:
    """The manifest's units holding the fact's planted sentence, with their labels."""
    found = []
    for doc_id in dict.fromkeys(fact["placements"]):
        sentence = fact[ctx.docs[doc_id]["language"]]
        found += [(unit, json.loads(labels)) for unit, labels in db.execute(
            "SELECT unit_id, labels FROM units WHERE doc_id = ? AND instr(text, ?) > 0", (doc_id, sentence))]
    return found


def generate(ctx: Context | None = None) -> dict[str, list[dict]]:
    """Both splits' probes, two a fact, in the families' order."""
    ctx = ctx or Context()
    w = world()
    path = manifest_path()
    if not path.is_file():
        raise GoldenError(f"no manifest at {path}: ntk ingest build makes it, and a probe's gold units come from it")
    roles = {role: readable(ctx.policy, role) for role in ctx.policy["roles"]}
    count = ctx.protocol["items"]["rare_terms"]
    out: dict[str, list[dict]] = {split: [] for split in SPLITS}
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        for family_id, family in sorted(ctx.families.items()):
            for fact_id in family["facts"]:
                fact = ctx.facts[fact_id]
                if fact["kind"] == "superseded":
                    continue
                gold = gold_units(fact, ctx, db)
                readers = sorted(r for r, reads in roles.items() if any(set(labels) & reads for _, labels in gold))
                selective = min(readers, key=lambda r: (len(roles[r]), r))  # the narrowest view that holds it
                x_en, x_el, y_en, y_el = slots(fact_id, w)
                template = int(fact_id[1:3])
                for lang, question in zip(("en", "el"), (fill(QUESTIONS[template][0], x_en, y_en),
                                                         fill(QUESTIONS[template][1], x_el, y_el))):
                    asked = set(lexical.terms(question))
                    shared = max(len([t for t in rare_terms(fact[ctx.docs[d]["language"]], ctx.df, count) if t in asked])
                                 for d in dict.fromkeys(fact["placements"]))
                    split = family["split"]
                    out[split].append({"id": f"{split[0].upper()}P{len(out[split]) + 1:03d}", "fact_id": fact_id,
                                       "family": family_id, "lang": lang, "question": question,
                                       "gold": [unit for unit, _ in gold], "readers": readers,
                                       "selective": selective, "shared_rare": shared})
    finally:
        db.close()
    return out


def problems(probes: dict[str, list[dict]], ctx: Context) -> list[str]:
    """Every probe that breaks the overlap cap or has no gold unit or reader."""
    cap = ctx.protocol["items"]["max_shared_rare_terms"]
    found = []
    for split, rows in probes.items():
        for probe in rows:
            if probe["shared_rare"] > cap:
                found.append(f"{probe['id']}: shares {probe['shared_rare']} of the rarest terms of {probe['fact_id']}")
            if not probe["gold"] or not probe["readers"]:
                found.append(f"{probe['id']}: no gold unit or no reader for {probe['fact_id']}")
    return found


def write(probes: dict[str, list[dict]]) -> None:
    for split, rows in probes.items():
        path = probes_path(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)).encode("utf-8"))


def load(split: str) -> list[dict]:
    path = probes_path(split)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.is_file() else []
