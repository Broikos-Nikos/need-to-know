# Publishing need-to-know

What the repository needs on GitHub, written down before it is created, so that creating it is a transcription
rather than a run of small decisions taken at the API.

## The About box

**Description**, 350 characters or fewer, opening with the README's own first sentence and carrying numbers the
README's claims hold (RC-F7: on GitHub the README starts under the list of files, so this box holds the first
screen's only sentence):

> Ask a question as a department, and get an answer built only from what that department may read. In 5,760
> searches no role was shown a unit it may not read, and the 35 test questions scored 0.981. Retrieval over a vector
> database, each question's path through its filter, both searches, their fusion and the answer drawn on the page.

**Homepage**: `https://broikos-nikos.github.io/need-to-know/`

**Topics**, ten, the most GitHub shows without a fold, each a category rather than a tool:

```
rag  retrieval-augmented-generation  vector-database  access-control  permissions
hybrid-search  reranking  information-retrieval  semantic-search  multilingual
```

`access-control` and `permissions` are the two that matter: a reader searching how to keep a retrieval system from
showing one department another's text is the reader this is for. `multilingual` because the documents and the
questions are in English and Greek, and one search serves both.

## Settings

- **Public**, no wiki, no projects, no discussions; issues on, so a reader who finds a leak can say so.
- **Pages**: source GitHub Actions, not a branch. The workflow tests, builds, gates and deploys, and a branch source
  would publish whatever was committed whether the gates passed or not.
- **No default community files.** The MIT LICENSE is committed already.

## The order, and why this order

1. **Create the repository** with the description, homepage and topics above, so the About box is never empty.
2. **Push `main` with its whole history**, every commit authored `Nikos Broikos <broikos.nikolaos@gmail.com>`, none
   co-authored, after the preflight passes in a fresh clone.
3. **Turn Pages on** with GitHub Actions as its source, and let the workflow run. Nothing is deployed by hand.
4. **Wait for the deploy, then open the live page in a browser** at 1280 and at 390 and look at it: a green deploy is
   not a working page.
5. **Put the live link in the README's first screen.**

## What this does not claim

The documents are synthetic: a plan in code fixes every fact, language model agents wrote the prose around them, and
the company they belong to is never named. The published page replays recorded runs, since no index or model runs
behind it; the README gives the command that recorded them and the one that serves the page live. Every figure is
this corpus's.
