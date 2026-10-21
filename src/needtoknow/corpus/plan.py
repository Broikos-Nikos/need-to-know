"""The fact registry and the document catalogue, drawn from a seed, and the writers' assignments.

Facts come first: every fact is one exact sentence in English and in Greek, with its
audience label, its family, and where it is planted. The catalogue says what each
document is and what each of its sections must contain, so that several writers
working apart still describe one consistent Company. Writers add prose around the
sentences; they never change a value.
"""

from __future__ import annotations

import json
import math
import random
import tomllib
from pathlib import Path

from needtoknow.corpus import dump_json
from needtoknow.settings import data_dir, repo_root

SEED = 20261012
LABEL_EL = {"general": "Όλο το προσωπικό", "sales": "Πωλήσεις", "logistics": "Εφοδιαστική", "returns": "Επιστροφές",
            "customer-care": "Εξυπηρέτηση Πελατών", "legal": "Νομικό Τμήμα", "finance": "Οικονομικά",
            "network-ops": "Λειτουργία Δικτύου", "hr": "Ανθρώπινο Δυναμικό"}
LABEL_EN = {"general": "All staff", "sales": "Sales", "logistics": "Logistics", "returns": "Returns",
            "customer-care": "Customer Care", "legal": "Legal", "finance": "Finance",
            "network-ops": "Network Operations", "hr": "Human Resources"}


def world() -> dict:
    with open(repo_root() / "config" / "world.toml", "rb") as handle:
        return tomllib.load(handle)


def num(value) -> tuple[str, str]:
    """A number as English and Greek write it: 1.5 against 1,5."""
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text, text.replace(".", ",")


def pick(*choices):
    return lambda rng, i: rng.choice(choices)


def span(low, high, step=1):
    return lambda rng, i: rng.randrange(low, high + 1, step)


# (label, topic en, topic el, sentence en, sentence el, values, variants)
# A variant list makes one family per variant; "REGION" makes one family of 13 regional variants.
TEMPLATES = [
    ("general", "Working hours and access", "Ωράριο και πρόσβαση",
     "Head office reception is open from {v} on working days.", "Η υποδοχή των κεντρικών γραφείων λειτουργεί από {v} τις εργάσιμες ημέρες.",
     pick(("08:00 to 17:00", "08:00 έως 17:00"), ("08:30 to 17:30", "08:30 έως 17:30"), ("07:30 to 16:30", "07:30 έως 16:30")), None),
    ("general", "Working hours and access", "Ωράριο και πρόσβαση",
     "Staff access badges must be renewed every {v} months.", "Οι κάρτες πρόσβασης του προσωπικού ανανεώνονται κάθε {v} μήνες.",
     pick(12, 18, 24), None),
    ("general", "Internal support", "Εσωτερική υποστήριξη",
     "The internal IT helpdesk answers on extension {v}.", "Το εσωτερικό γραφείο υποστήριξης πληροφορικής απαντά στο εσωτερικό τηλέφωνο {v}.",
     span(2100, 2999), None),
    ("general", "Conduct and compliance", "Δεοντολογία και συμμόρφωση",
     "Every employee confirms the code of conduct once every {v} months.", "Κάθε εργαζόμενος επιβεβαιώνει τον κώδικα δεοντολογίας μία φορά κάθε {v} μήνες.",
     pick(12, 24), None),
    ("general", "Conduct and compliance", "Δεοντολογία και συμμόρφωση",
     "Gifts from suppliers worth more than {v} euros must be declared.", "Τα δώρα από προμηθευτές αξίας άνω των {v} ευρώ πρέπει να δηλώνονται.",
     pick(30, 50, 75, 100), None),
    ("general", "Internal support", "Εσωτερική υποστήριξη",
     "Shops in {x} open at {v} on weekdays.", "Τα καταστήματα στην περιοχή «{x}» ανοίγουν στις {v} τις καθημερινές.",
     pick("09:00", "09:30", "10:00"), "REGION"),
    ("sales", "Discounts and approvals", "Εκπτώσεις και εγκρίσεις",
     "In {x}, the maximum discount a sales advisor may grant without approval is {v} percent.",
     "Στο κανάλι «{x}», η μέγιστη έκπτωση που μπορεί να δώσει σύμβουλος πωλήσεων χωρίς έγκριση είναι {v} τοις εκατό.",
     span(5, 20), [("shops", "καταστήματα"), ("telesales", "τηλεπωλήσεις"), ("the online store", "ηλεκτρονικό κατάστημα")]),
    ("sales", "Contracts and terms", "Συμβόλαια και όροι",
     "Contracts for {x} run for a minimum of {v} months.", "Τα συμβόλαια για την κατηγορία «{x}» έχουν ελάχιστη διάρκεια {v} μηνών.",
     pick(12, 24), "PRODUCTS:postpaid,business,fibre,copper,fwa,tv"),
    ("sales", "Devices and subsidies", "Συσκευές και επιδοτήσεις",
     "The device subsidy on a new contract for {x} is capped at {v} euros.",
     "Η επιδότηση συσκευής σε νέο συμβόλαιο για την κατηγορία «{x}» δεν υπερβαίνει τα {v} ευρώ.",
     span(100, 400, 10), "PRODUCTS:postpaid,business"),
    ("sales", "Devices and subsidies", "Συσκευές και επιδοτήσεις",
     "Customers become eligible for a device upgrade after {v} months of their contract.",
     "Ο πελάτης δικαιούται αναβάθμιση συσκευής μετά από {v} μήνες συμβολαίου.", pick(12, 18, 20), None),
    ("sales", "Credit and eligibility", "Πίστωση και επιλεξιμότητα",
     "A credit check is required for any order above {v} euros.", "Απαιτείται πιστοληπτικός έλεγχος για κάθε παραγγελία άνω των {v} ευρώ.",
     span(300, 900, 50), None),
    ("sales", "Commissions", "Προμήθειες",
     "Advisors earn a commission of {v} percent on sales of {x}.", "Οι σύμβουλοι λαμβάνουν προμήθεια {v} τοις εκατό στις πωλήσεις της κατηγορίας «{x}».",
     span(2, 12), "PRODUCTS:prepaid,postpaid,business,fibre,tv,devices,accessories,insurance"),
    ("sales", "Activation", "Ενεργοποίηση",
     "A new line for {x} is activated within {v} hours of a signed order.",
     "Μια νέα σύνδεση για την κατηγορία «{x}» ενεργοποιείται εντός {v} ωρών από την υπογραφή της παραγγελίας.",
     pick(2, 4, 8, 24, 48), "PRODUCTS:prepaid,postpaid,business,tv"),
    ("returns", "Return eligibility", "Δικαίωμα επιστροφής",
     "Items in the category {x} may be returned within {v} days of purchase.",
     "Τα προϊόντα της κατηγορίας «{x}» επιστρέφονται εντός {v} ημερών από την αγορά.",
     pick(14, 21, 30), "PRODUCTS:devices,accessories,insurance"),
    ("returns", "Refund approval", "Έγκριση επιστροφής χρημάτων",
     "A {x} may approve refunds up to {v} euros.", "Ο ρόλος «{x}» εγκρίνει επιστροφές χρημάτων έως {v} ευρώ.",
     lambda rng, i: [rng.randrange(50, 151, 10), rng.randrange(200, 501, 25), rng.randrange(800, 2001, 100)][i],
     [("sales advisor", "σύμβουλος πωλήσεων"), ("shop manager", "διευθυντής καταστήματος"), ("regional manager", "περιφερειακός διευθυντής")]),
    ("returns", "Return eligibility", "Δικαίωμα επιστροφής",
     "Opened devices returned in good condition carry a restocking fee of {v} percent.",
     "Για συσκευές που επιστρέφονται ανοιγμένες σε καλή κατάσταση ισχύει τέλος επανεισαγωγής {v} τοις εκατό.", pick(5, 10, 15), None),
    ("returns", "Inspection of returned goods", "Έλεγχος επιστρεφόμενων ειδών",
     "Returned items are inspected within {v} working days of arrival at the warehouse.",
     "Τα επιστρεφόμενα είδη ελέγχονται εντός {v} εργάσιμων ημερών από την άφιξή τους στην αποθήκη.", span(2, 5), None),
    ("returns", "Refund approval", "Έγκριση επιστροφής χρημάτων",
     "Approved refunds reach the customer within {v} working days.", "Οι εγκεκριμένες επιστροφές χρημάτων φτάνουν στον πελάτη εντός {v} εργάσιμων ημερών.",
     span(5, 14), None),
    ("returns", "Return authorisation", "Εξουσιοδότηση επιστροφής",
     "Return authorisation codes start with the letters {v}.", "Οι κωδικοί εξουσιοδότησης επιστροφής ξεκινούν με τα γράμματα {v}.",
     pick("RA", "RX", "RT", "RN"), None),
    ("returns", "Return authorisation", "Εξουσιοδότηση επιστροφής",
     "A customer may make at most {v} returns in any twelve months without a supervisor's review.",
     "Ένας πελάτης μπορεί να κάνει έως {v} επιστροφές σε δώδεκα μήνες χωρίς έλεγχο από επόπτη.", span(3, 6), None),
    ("returns", "Repairs under warranty", "Επισκευές εντός εγγύησης",
     "Devices sent to repair centre {x} come back within {v} working days.",
     "Οι συσκευές που στέλνονται στο κέντρο επισκευής {x} επιστρέφουν εντός {v} εργάσιμων ημερών.", span(5, 15), [("R1", "R1"), ("R2", "R2")]),
    ("logistics", "Courier pickups", "Παραλαβές μεταφορέων",
     "At {x}, the last pickup for courier {y} is at {v}.", "Στην {x}, η τελευταία παραλαβή για τον μεταφορέα {y} γίνεται στις {v}.",
     pick("14:00", "15:00", "15:30", "16:00", "17:00", "18:00"), "WAREHOUSE_COURIER"),
    ("logistics", "Delivery times", "Χρόνοι παράδοσης",
     "Standard delivery to {x} takes {v} working days.", "Η τυπική παράδοση προς την περιοχή «{x}» διαρκεί {v} εργάσιμες ημέρες.",
     span(1, 4), "REGION"),
    ("logistics", "Delivery times", "Χρόνοι παράδοσης",
     "Fibre installation in {x} is scheduled within {v} working days of the order.",
     "Η εγκατάσταση οπτικής ίνας στην περιοχή «{x}» προγραμματίζεται εντός {v} εργάσιμων ημερών από την παραγγελία.",
     span(5, 15), "REGION"),
    ("logistics", "Handling and packing", "Διακίνηση και συσκευασία",
     "A single pallet may not exceed {v} kilograms.", "Μια παλέτα δεν μπορεί να υπερβαίνει τα {v} κιλά.", span(600, 1000, 50), None),
    ("logistics", "Handling and packing", "Διακίνηση και συσκευασία",
     "Shipments worth more than {v} euros travel insured.", "Οι αποστολές αξίας άνω των {v} ευρώ ταξιδεύουν ασφαλισμένες.",
     span(500, 2000, 100), None),
    ("logistics", "Warehouse operations", "Λειτουργία αποθηκών",
     "{X} receives goods from {v} on working days.", "Η {x} παραλαμβάνει εμπορεύματα από {v} τις εργάσιμες ημέρες.",
     pick(("07:00 to 15:00", "07:00 έως 15:00"), ("06:00 to 14:00", "06:00 έως 14:00"), ("08:00 to 16:00", "08:00 έως 16:00")), "WAREHOUSES"),
    ("logistics", "Warehouse operations", "Λειτουργία αποθηκών",
     "Stock counts at {x} take place every {v} weeks.", "Η απογραφή στην {x} γίνεται κάθε {v} εβδομάδες.", pick(2, 4, 6, 8), "WAREHOUSES"),
    ("customer-care", "Response times", "Χρόνοι απόκρισης",
     "Tickets of priority {x} receive a first response within {v} hours.", "Τα αιτήματα προτεραιότητας {x} λαμβάνουν πρώτη απάντηση εντός {v} ωρών.",
     lambda rng, i: [rng.choice((1, 2)), rng.choice((4, 6, 8)), rng.choice((12, 24)), rng.choice((48, 72))][i],
     [("P1", "P1"), ("P2", "P2"), ("P3", "P3"), ("P4", "P4")]),
    ("customer-care", "Callbacks and holding", "Επανακλήσεις και αναμονή",
     "A promised callback must happen within {v} minutes.", "Μια υποσχεμένη επανάκληση πρέπει να γίνει εντός {v} λεπτών.", span(30, 120, 15), None),
    ("customer-care", "Callbacks and holding", "Επανακλήσεις και αναμονή",
     "Callers should not be kept on hold for more than {v} minutes.", "Οι καλούντες δεν πρέπει να παραμένουν σε αναμονή περισσότερο από {v} λεπτά.",
     span(3, 10), None),
    ("customer-care", "Compensation", "Αποζημιώσεις",
     "Customers receive a credit of {v} euros for each full hour of a confirmed outage.",
     "Οι πελάτες λαμβάνουν πίστωση {v} ευρώ για κάθε πλήρη ώρα επιβεβαιωμένης διακοπής.", pick(0.5, 1.0, 1.5, 2.0), None),
    ("customer-care", "Escalation", "Κλιμάκωση",
     "A complaint is escalated to the second line after {v} unresolved contacts.",
     "Ένα παράπονο κλιμακώνεται στη δεύτερη γραμμή μετά από {v} επαφές χωρίς επίλυση.", span(2, 4), None),
    ("customer-care", "Escalation", "Κλιμάκωση",
     "Second line resolves an escalated complaint about {x} within {v} working days.",
     "Η δεύτερη γραμμή επιλύει ένα κλιμακωμένο παράπονο για την κατηγορία «{x}» εντός {v} εργάσιμων ημερών.",
     span(3, 10), "PRODUCTS:postpaid,fibre,tv,devices"),
    ("legal", "Consumer rights", "Δικαιώματα καταναλωτών",
     "Distance contracts may be cancelled within {v} days without giving a reason.",
     "Οι συμβάσεις από απόσταση μπορούν να καταγγελθούν εντός {v} ημερών χωρίς αιτιολογία.", pick(14), None),
    ("legal", "Data retention", "Διατήρηση δεδομένων",
     "{X} are kept for {v} years and then deleted.", "{X} φυλάσσονται για {v} έτη και στη συνέχεια διαγράφονται.",
     lambda rng, i: [rng.choice((1, 2)), rng.choice((5, 7, 10)), rng.choice((10, 15))][i],
     [("call recordings", "οι ηχογραφήσεις κλήσεων"), ("billing records", "τα αρχεία χρέωσης"), ("signed customer contracts", "οι υπογεγραμμένες συμβάσεις πελατών")]),
    ("legal", "Complaints", "Καταγγελίες",
     "Written complaints receive a reasoned answer within {v} days.", "Οι γραπτές καταγγελίες λαμβάνουν αιτιολογημένη απάντηση εντός {v} ημερών.",
     span(15, 30, 5), None),
    ("legal", "Business contracts", "Επαγγελματικές συμβάσεις",
     "Business customers must give {v} days of written notice to terminate.",
     "Οι επαγγελματίες πελάτες δίνουν έγγραφη ειδοποίηση {v} ημερών για τη λύση της σύμβασης.", span(30, 90, 15), None),
    ("legal", "Business contracts", "Επαγγελματικές συμβάσεις",
     "The Company's liability under a business contract is capped at {v} euros per incident.",
     "Η ευθύνη της Εταιρείας σε επαγγελματική σύμβαση περιορίζεται στα {v} ευρώ ανά συμβάν.", span(5000, 50000, 5000), None),
    ("legal", "Insurance claims", "Ασφαλιστικές απαιτήσεις",
     "A device insurance claim must be reported to partner {x} within {v} days of the incident.",
     "Μια απαίτηση ασφάλισης συσκευής δηλώνεται στον συνεργάτη {x} εντός {v} ημερών από το συμβάν.", span(2, 10), [("INS1", "INS1")]),
    ("finance", "Payments", "Πληρωμές",
     "Supplier invoices are paid within {v} days of receipt.", "Τα τιμολόγια προμηθευτών εξοφλούνται εντός {v} ημερών από την παραλαβή.",
     span(30, 90, 15), None),
    ("finance", "Payments", "Πληρωμές",
     "Late payments by business customers carry a fee of {v} percent per month.",
     "Οι καθυστερημένες πληρωμές επαγγελματικών πελατών επιβαρύνονται με {v} τοις εκατό τον μήνα.", pick(1.0, 1.5, 2.0, 2.5), None),
    ("finance", "Approvals", "Εγκρίσεις",
     "Refunds above {v} euros also need sign off from finance.", "Οι επιστροφές χρημάτων άνω των {v} ευρώ χρειάζονται επιπλέον έγκριση από το Οικονομικό.",
     span(1000, 3000, 250), None),
    ("finance", "Approvals", "Εγκρίσεις",
     "Unpaid balances below {v} euros may be written off after twelve months.",
     "Οφειλές κάτω των {v} ευρώ μπορούν να διαγραφούν μετά από δώδεκα μήνες.", span(10, 50, 5), None),
    ("finance", "Billing", "Χρεώσεις",
     "Consumer bills for {x} are issued on day {v} of each month.", "Οι λογαριασμοί ιδιωτών για την κατηγορία «{x}» εκδίδονται την {v}η ημέρα κάθε μήνα.",
     span(1, 28), "PRODUCTS:postpaid,fibre,copper,tv"),
    ("network-ops", "Incident response", "Απόκριση σε συμβάντα",
     "Severity {x} network incidents require an engineer on site within {v} hours.",
     "Τα συμβάντα δικτύου σοβαρότητας {x} απαιτούν τεχνικό επί τόπου εντός {v} ωρών.",
     lambda rng, i: [rng.choice((2, 3, 4)), rng.choice((6, 8, 12)), rng.choice((24, 36, 48))][i], [("S1", "S1"), ("S2", "S2"), ("S3", "S3")]),
    ("network-ops", "Maintenance", "Συντήρηση",
     "Planned maintenance runs between {v}.", "Οι προγραμματισμένες εργασίες συντήρησης γίνονται μεταξύ {v}.",
     pick(("01:00 and 05:00", "01:00 και 05:00"), ("00:00 and 04:00", "00:00 και 04:00"), ("02:00 and 06:00", "02:00 και 06:00")), None),
    ("network-ops", "Availability", "Διαθεσιμότητα",
     "The core network targets an availability of {v} percent each month.",
     "Ο κορμός του δικτύου στοχεύει σε διαθεσιμότητα {v} τοις εκατό κάθε μήνα.", pick(99.9, 99.95, 99.99), None),
    ("network-ops", "Spare parts", "Ανταλλακτικά",
     "The depot serving {x} keeps spare parts for {v} days of normal use.",
     "Η αποθήκη που εξυπηρετεί την περιοχή «{x}» τηρεί ανταλλακτικά για {v} ημέρες κανονικής χρήσης.", span(14, 45), "REGION"),
    ("network-ops", "Field visits", "Επισκέψεις πεδίου",
     "A fault on {x} gets a technician visit within {v} hours.", "Μια βλάβη στην κατηγορία «{x}» εξυπηρετείται με επίσκεψη τεχνικού εντός {v} ωρών.",
     pick(24, 48, 72), "PRODUCTS:fibre,copper,fwa,tv"),
    ("hr", "Joining", "Πρόσληψη",
     "New employees serve a probation period of {v} months.", "Οι νέοι εργαζόμενοι έχουν δοκιμαστική περίοδο {v} μηνών.", span(3, 6), None),
    ("hr", "Leave", "Άδειες",
     "Full time staff are entitled to {v} days of annual leave.", "Το προσωπικό πλήρους απασχόλησης δικαιούται {v} ημέρες ετήσιας άδειας.",
     span(22, 26), None),
    ("hr", "Ways of working", "Τρόπος εργασίας",
     "Office staff may work remotely up to {v} days a week.", "Το προσωπικό γραφείου μπορεί να εργάζεται εξ αποστάσεως έως {v} ημέρες την εβδομάδα.",
     span(1, 3), None),
    ("hr", "Training", "Εκπαίδευση",
     "Each employee completes at least {v} hours of training a year.", "Κάθε εργαζόμενος ολοκληρώνει τουλάχιστον {v} ώρες εκπαίδευσης τον χρόνο.",
     span(16, 40, 4), None),
    ("hr", "Leaving", "Αποχώρηση",
     "Employees who resign give {v} weeks of notice.", "Οι εργαζόμενοι που παραιτούνται δίνουν προειδοποίηση {v} εβδομάδων.", span(2, 6), None),
]

# Facts whose value changed: the old one lives in an older circular, the new one everywhere else.
VERSIONED_SHARE = 0.12
OLD_DATE, NEW_DATE = ("2024-03-01", "1 March 2024", "1 Μαρτίου 2024"), ("2025-07-01", "1 July 2025", "1 Ιουλίου 2025")

TYPES = [  # key, en, el, count, size range MB, language mix (share Greek), formats
    ("MAN", "Operations manual", "Εγχειρίδιο λειτουργιών", 10, (4, 10), 0.5, ["txt"]),
    ("POL", "Group policy", "Πολιτική ομίλου", 16, (0.5, 3), 0.0, ["txt", "md"]),
    ("PRO", "Procedure", "Διαδικασία", 48, (0.5, 2), 0.75, ["txt", "md"]),
    ("PLB", "Playbook", "Οδηγός πρακτικής", 16, (1, 4), 0.5, ["md", "html"]),
    ("CON", "Contract and clause library", "Βιβλιοθήκη συμβάσεων και όρων", 12, (1, 6), 0.5, ["txt"]),
    ("CIR", "Circular compilation", "Συλλογή εγκυκλίων", 32, (0.5, 1), 1.0, ["txt", "md"]),
    ("FAQ", "FAQ and case notes", "Συχνές ερωτήσεις και υποθέσεις", 16, (0.5, 2), 0.5, ["md", "html"]),
    ("CAT", "Catalogue and annex pack", "Κατάλογος και παραρτήματα", 10, (2, 10), 0.0, ["txt"]),
]
OWNER_FOR = {"MAN": ["returns", "logistics", "customer-care", "network-ops", "sales"], "CON": ["legal", "finance"],
             "CAT": ["sales", "logistics"], "POL": None, "PRO": None, "PLB": ["sales", "customer-care"],
             "CIR": None, "FAQ": ["customer-care", "sales", "returns"]}
DEPARTMENTS = ["sales", "logistics", "returns", "customer-care", "legal", "finance", "network-ops", "hr"]
MIXES = {"returns": "legal", "sales": "returns", "logistics": "finance", "customer-care": "legal", "legal": "finance",
         "finance": "legal", "network-ops": "logistics", "hr": "legal"}


def variants(spec, w):
    if spec is None:
        return [("", "", "", "")]
    if spec == "REGION":
        return None
    if spec == "WAREHOUSES":
        return [(h["en"], h["el"].replace("η αποθήκη", "αποθήκη"), "", "") for h in w["warehouses"]]
    if spec == "WAREHOUSE_COURIER":
        return [(h["en"], h["el"].replace("η αποθήκη", "αποθήκη"), c, c) for h in w["warehouses"] for c in ("C1", "C2")]
    if isinstance(spec, str) and spec.startswith("PRODUCTS:"):
        products = {p["key"]: p for p in w["products"]}
        return [(products[k]["en"], products[k]["el"], "", "") for k in spec.split(":")[1].split(",")]
    return [(en, el, "", "") for en, el in spec]


# The unit after a value of one, for each unit a template writes after {v} where a value can be one: "1 years" and
# "1 εργάσιμες ημέρες" were in the corpus, the first in legal's own question as the page replays it.
SINGULAR = {"working days": "working day", "hours": "hour", "years": "year", "days": "day", "euros": "euro",
            "εργάσιμες ημέρες": "εργάσιμη ημέρα", "ωρών": "ώρας", "έτη": "έτος", "ημέρες": "ημέρα"}


def render(template: str, x: str, y: str, value: str) -> str:
    if value == "1":
        for many, one in SINGULAR.items():
            template = template.replace("{v} " + many, "{v} " + one)
    out = template.replace("{X}", x[:1].upper() + x[1:]).replace("{x}", x).replace("{y}", y).replace("{v}", value)
    return out[:1].upper() + out[1:]


def make_facts(rng: random.Random, w: dict) -> list[dict]:
    facts = []
    for t_index, (label, topic_en, topic_el, en, el, values, spec) in enumerate(TEMPLATES):
        groups = variants(spec, w)
        if groups is None:  # one family of regional variants
            family = f"F{t_index:02d}"
            for r, (region_en, region_el) in enumerate(zip(w["regions"]["en"], w["regions"]["el"])):
                facts.append(fact(rng, label, topic_en, topic_el, en, el, values, r, region_en, region_el, "", "", family, f"{family}R{r:02d}"))
            continue
        for v_index, (x_en, x_el, y_en, y_el) in enumerate(groups):
            family = f"F{t_index:02d}V{v_index:02d}"
            facts.append(fact(rng, label, topic_en, topic_el, en, el, values, v_index, x_en, x_el, y_en, y_el, family, family))
    # Versions: some facts changed value on NEW_DATE; the old sentence is kept as its own entry.
    candidates = [f for f in facts if f["kind"] == "single" and f["values"][0].isdigit() and f["label"] != "legal"]
    for f in [f for f in candidates if rng.random() < VERSIONED_SHARE * len(facts) / len(candidates)]:
        old = dict(f)
        old_value = f["values"]
        new_value = shift(old_value)
        f.update(kind="versioned", values=new_value, effective=NEW_DATE[0],
                 en=f"From {NEW_DATE[1]}, " + lower_first(f["en_template_render"](new_value[0])),
                 el=f"Από την {NEW_DATE[2]}, " + lower_first(f["el_template_render"](new_value[1])))
        old.update(id=f["id"] + "OLD", kind="superseded", effective=OLD_DATE[0], superseded_by=f["id"],
                   en=f"From {OLD_DATE[1]}, " + lower_first(old["en"]), el=f"Από την {OLD_DATE[2]}, " + lower_first(old["el"]))
        facts.append(old)
    for f in facts:
        f.pop("en_template_render", None)
        f.pop("el_template_render", None)
    return facts


def fact(rng, label, topic_en, topic_el, en, el, values, index, x_en, x_el, y_en, y_el, family, fid) -> dict:
    raw = values(rng, index)
    value = raw if isinstance(raw, tuple) else num(raw)
    return {"id": fid, "family": family, "label": label, "kind": "variant" if fid != family else "single",
            "topic_en": topic_en, "topic_el": topic_el, "values": value,
            "en": render(en, x_en, y_en, value[0]), "el": render(el, x_el, y_el, value[1]),
            "en_template_render": lambda v, en=en, x=x_en, y=y_en: render(en, x, y, v),
            "el_template_render": lambda v, el=el, x=x_el, y=y_el: render(el, x, y, v)}


def shift(value: tuple[str, str]) -> tuple[str, str]:
    """A different whole number of the same size, for the value that replaced an older one."""
    number = int(value[0])
    return num(number + (1 if number < 10 else max(1, round(number * 0.2))))


def lower_first(text: str) -> str:
    """The sentence after a date frame goes on in lower case, unless its first word is an acronym such as SLA.

    A one letter word is no acronym: "A credit check" and "Ο πελάτης" kept their capital
    after "From 1 July 2025, " until KB3-K7.
    """
    first = text.split(" ", 1)[0]
    return text if len(first) > 1 and first.isupper() else text[:1].lower() + text[1:]


def make_catalogue(rng: random.Random, facts: list[dict]) -> list[dict]:
    docs = []
    for key, type_en, type_el, count, (low, high), greek_share, formats in TYPES:
        for n in range(1, count + 1):
            owners = OWNER_FOR[key] or DEPARTMENTS
            owner = owners[(n - 1) % len(owners)]
            lang = "el" if rng.random() < greek_share else "en"
            size = math.exp(rng.uniform(math.log(low), math.log(high)))
            docs.append({"id": f"{owner[:3].upper()}_{key}_{n:03d}", "type": key, "type_name": type_el if lang == "el" else type_en,
                         "owner": owner, "language": lang, "format": rng.choice(formats), "target_mb": round(size, 2),
                         "version": f"{rng.randint(1, 4)}.{rng.randint(0, 9)}", "sections": []})
    # Sections: topics of the owner's label, a general one in half the documents, another department in a third.
    topics = {}
    for f in facts:
        topics.setdefault(f["label"], {}).setdefault((f["topic_en"], f["topic_el"]), [])
    for doc in docs:
        plan = [doc["owner"]] * rng.randint(3, 6)
        if rng.random() < 0.5:
            plan.insert(0, "general")
        if rng.random() < 0.42 and doc["owner"] in MIXES:
            plan.append(MIXES[doc["owner"]])
        for label in plan:
            topic = rng.choice(sorted(topics[label]))
            doc["sections"].append({"label": label, "topic_en": topic[0], "topic_el": topic[1], "facts": [], "canary": None})
    place(rng, docs, facts)
    for doc in docs:
        for s in doc["sections"]:
            if s["label"] != "general" and rng.random() < 0.06:
                s["canary"] = f"CANARY_{rng.randrange(16**6):06X}_{s['label'].upper().replace('-', '')}"
    number_repeats(docs)
    return docs


def number_repeats(docs: list[dict]) -> None:
    """A heading drawn twice in one document becomes "..., part 2", so no file repeats a heading (DR-F2).

    It runs after every random draw and after placement, so the facts, canaries and
    writers' sections stay exactly where they were; only the repeated headings change.
    """
    for doc in docs:
        seen: dict[str, int] = {}
        for s in doc["sections"]:
            seen[s["topic_en"]] = seen.get(s["topic_en"], 0) + 1
            if seen[s["topic_en"]] > 1:
                s["topic_en"], s["topic_el"] = f"{s['topic_en']}, part {seen[s['topic_en']]}", f"{s['topic_el']}, μέρος {seen[s['topic_en']]}"


def place(rng: random.Random, docs: list[dict], facts: list[dict]) -> None:
    """Every fact into a section of its label and topic; a quarter of them into a second document too."""
    by_key = {}
    for doc in docs:
        for s in doc["sections"]:
            by_key.setdefault((s["label"], s["topic_en"]), []).append((doc, s))
    circulars = [(d, s) for d in docs if d["type"] == "CIR" for s in d["sections"]]
    for f in facts:
        if f["kind"] == "superseded":
            home = [(d, s) for d, s in circulars if s["label"] == f["label"]] or circulars
        else:
            home = by_key.get((f["label"], f["topic_en"])) or [(d, s) for d in docs for s in d["sections"] if s["label"] == f["label"]]
        homes = rng.sample(home, min(len(home), 2 if rng.random() < 0.25 and f["kind"] != "superseded" else 1))
        f["placements"] = []
        for doc, section in homes:
            section["facts"].append(f["id"])
            f["placements"].append(doc["id"])


def header(doc: dict, number: int, section: dict) -> list[str]:
    if doc["language"] == "el":
        return [f"ΕΝΟΤΗΤΑ {number}  {section['topic_el']}", f"Κοινό: {LABEL_EL[section['label']]}", "Διαβάθμιση: Εσωτερικό"]
    return [f"SECTION {number}  {section['topic_en']}", f"Audience: {LABEL_EN[section['label']]}", "Classification: Internal"]


def assignments(docs: list[dict], facts: list[dict], writers: int) -> list[list[dict]]:
    by_id = {f["id"]: f for f in facts}
    order = sorted(docs, key=lambda d: (d["language"], d["type"], d["id"]))
    slices = [[] for _ in range(writers)]
    for i, doc in enumerate(order):
        sections = []
        for n, s in enumerate(doc["sections"], 1):
            sections.append({"header": header(doc, n, s), "facts": [by_id[fid][doc["language"]] for fid in s["facts"]],
                             "canary": s["canary"], "topic": s["topic_el"] if doc["language"] == "el" else s["topic_en"]})
        slices[i % writers].append({"id": doc["id"], "language": doc["language"], "kind": doc["type_name"],
                                    "owner": LABEL_EL[doc["owner"]] if doc["language"] == "el" else LABEL_EN[doc["owner"]],
                                    "sections": sections})
    return slices


def build(writers: int = 4, seed: int = SEED) -> Path:
    rng = random.Random(seed)
    w = world()
    facts = make_facts(rng, w)
    docs = make_catalogue(rng, facts)
    out = data_dir() / "corpus" / "plan"
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "facts.json", facts)
    dump_json(out / "catalogue.json", docs)
    for k, part in enumerate(assignments(docs, facts, writers), 1):
        dump_json(out / f"writer-{k}.json", part)
    return out


def summary(out: Path) -> str:
    """The plan in one line, from the files it wrote."""
    facts = json.loads((out / "facts.json").read_text(encoding="utf-8"))
    docs = json.loads((out / "catalogue.json").read_text(encoding="utf-8"))
    sections = [s for d in docs for s in d["sections"]]
    mixed = sum(1 for d in docs if len({s["label"] for s in d["sections"]} - {"general"}) > 1)
    return (f"{len(facts)} facts in {len({f['family'] for f in facts})} families, "
            f"{sum(f['kind'] == 'superseded' for f in facts)} superseded; {len(docs)} documents, "
            f"{sum(d['language'] == 'el' for d in docs)} Greek, {len(sections)} sections, {mixed} mixing departments, "
            f"{sum(1 for s in sections if s['canary'])} canaries; about {round(sum(d['target_mb'] for d in docs))} MB planned")
