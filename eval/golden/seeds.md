# Seed questions from public documents

Ten questions as a customer of a telecoms provider asks them, each written from a public
document of the domain, beside what that document says and what this corpus says about the
same thing. They are not scored. The golden items are an employee's questions about the
company's own rules, drafted from a plan that a model's prose then surrounded; these are the
anchor the knowledge base asks such a set to have (`10-use-case-patterns/07` section 9): what
the real rules are, and how a real asker words a question about them.

The texts were read on 2026-10-13: the two EU directives in the copy legislation.gov.uk keeps
of each as adopted, the Greek procedure in the national register of administrative
procedures.

| # | Question, as asked | Source | What the source says | What the corpus says |
|---|---|---|---|---|
| 1 | I signed up for broadband on the website yesterday. Can I still cancel, and until when? | Directive 2011/83/EU, Article 9(1) | 14 days to withdraw from a distance contract, without giving any reason | F34V00, a test family, 14 days: agrees |
| 2 | Μπορεί ο πάροχος να με δεσμεύσει με συμβόλαιο για περισσότερο από δύο χρόνια; | Directive (EU) 2018/1972, Article 105(1) | a consumer contract may not "mandate a commitment period longer than 24 months" | the 12 and 24 month minimum terms agree; F07V04's current 29 months does not (RS-F23) |
| 3 | How long can my number be out of service when I switch to a new provider? | Directive (EU) 2018/1972, Article 106(5) | the number is activated "within one working day from the date agreed" | a sales page mentions a customer switching provider with their number; no time anywhere: not covered |
| 4 | Αν μου στείλουν αλλαγή στους όρους, μπορώ να λύσω το συμβόλαιο χωρίς χρέωση; | Directive (EU) 2018/1972, Article 105(4) | the end user may terminate without further costs; notice at least one month ahead; one month to act, which a state may extend to three | a legal page: terms that allow unilateral price changes give a clear right to end the contract without penalty, the notice period set in the template: agrees, with no period |
| 5 | Will you tell me before my contract renews by itself? | Directive (EU) 2018/1972, Article 105(3) | the provider informs prominently and in good time before an automatic prolongation, with advice on the best tariff | a sales page: contracts that renew automatically are checked before renewal, since the customer has the right to be told of changed terms: agrees in part |
| 6 | Σε πόσες μέρες πρέπει να μου απαντήσει η εταιρεία σε ένα παράπονο; | Greek procedure for electronic communications complaints, under law 4727/2020, articles 210 to 223 | the provider answers within 20 days, extended once by up to 10 days with the regulator's approval | F36V00, a dev family: written complaints receive a reasoned answer within 20 days: agrees |
| 7 | Can I go straight to the regulator with a complaint, or must I complain to you first? | the same procedure | the customer must first contact the provider, whose unique complaint number goes with the complaint | legal pages: recourse to the regulator stays open, and complaints from a regulator are logged; nothing on contacting the provider first: not covered |
| 8 | Αν αργήσει η μεταφορά του αριθμού μου, παίρνω κάποια αποζημίωση; | Directive (EU) 2018/1972, Article 106(8) | each state lays down rules on compensating end users for delays in porting | no compensation for porting; an outage credit exists, F31V00, which is another thing: not covered |
| 9 | What do I get in writing before I sign a mobile contract? | Directive (EU) 2018/1972, Article 102(3) | a concise, easily readable contract summary of the main elements, before the contract is concluded | a sales page: an approved discount's final price appears in the contract summary and the receipt: agrees that there is one |
| 10 | Μετά την αυτόματη ανανέωση, με πόση ειδοποίηση μπορώ να διακόψω; | Directive (EU) 2018/1972, Article 105(3) | after an automatic prolongation, the end user may terminate at any time with at most one month's notice | not covered |

Sources: Directive 2011/83/EU, <https://eur-lex.europa.eu/eli/dir/2011/83/oj>; Directive (EU)
2018/1972, the European Electronic Communications Code, <https://eur-lex.europa.eu/eli/dir/2018/1972/oj>;
the Greek procedure, <https://mitos.gov.gr/index.php/ΔΔ:Διαχείριση_αιτημάτων/καταγγελιών_για_ηλεκτρονικές_επικοινωνίες>.

## What they show

* Where the corpus and the rules meet, they agree: the 14 days of withdrawal, the 20 days to
  answer a complaint, the right to leave on a unilateral change, the contract summary.
* One planted value breaks a rule: F07V04 says contracts for fixed wireless broadband run for
  at least 29 months since 1 July 2025, past the 24 month cap on a consumer's commitment, and
  its former value is exactly 24. A model that knows the law would give the former value, so
  a miss on F07V04's three dev items may be knowledge rather than a failure to choose the
  current value; the closed book run reports which items a model answers without the
  documents (RS-F23).
* Porting, which the Code regulates in detail, is not in the plan at all, and neither are
  the complaint route to the regulator or the notice after a renewal: the corpus is
  narrower than the domain, and the evaluation says nothing about those topics.
* A customer asks in the first person, names the channel, and asks about a right; the golden
  items are an employee's questions about a rule. The system is built for the employee, so
  that is the right asker for the items, and the difference is said here rather than hidden.
