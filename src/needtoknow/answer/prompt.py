"""The prompt and the answer it asks for (T4.1).

The system rules come first and never change, so DeepSeek's cache keeps them as a stable
prefix at a fiftieth of the miss price (`specs/architecture.md` section 9); then the
excerpts, numbered, each with its document and section; the question last. The rules
hold no text of any document, so nothing a role may not read can reach a prompt through
them.

The model answers in JSON, `{"status": "answered" | "not_found", "answer": "...",
"citations": [1, 2]}`, citing excerpts by number. `parse` turns each number into its
excerpt's document and section, so a citation can name only what the role was shown,
and it replaces a `not_found` answer with one fixed sentence in the question's language:
a fact that does not exist and a fact the role may not read get the same words, so the
refusal does not reveal that something is there (section 9, item 7). A reply that is
not that JSON comes back with status `invalid` and the reason, which the evaluation
scores as wrong.
"""

from __future__ import annotations

import json

SYSTEM = """You answer questions about the Company's internal documents for one employee.

Rules:
1. Use only the numbered excerpts in the message. They are all this employee may read on the question. Never use other knowledge.
2. If the excerpts answer the question, set "status" to "answered", answer briefly in the language of the question, and list in "citations" the numbers of the excerpts the answer rests on.
3. If they do not, set "status" to "not_found", "answer" to an empty string and "citations" to an empty list. Do not guess.
4. If excerpts give different values for the same thing, answer with the one in force and name the other only as the former value.
5. Reply with one JSON object and nothing else: {"status": "...", "answer": "...", "citations": [...]}."""

NOT_FOUND = {"en": "The documents you can read do not answer this question.",
             "el": "Τα έγγραφα στα οποία έχετε πρόσβαση δεν απαντούν σε αυτή την ερώτηση."}
STATUSES = ("answered", "not_found")


def language(question: str) -> str:
    """Greek when the question holds a Greek letter, monotonic or polytonic; English otherwise."""
    return "el" if any(0x0370 <= ord(c) <= 0x03FF or 0x1F00 <= ord(c) <= 0x1FFF for c in question) else "en"


def section_of(excerpt: dict) -> str:
    return " > ".join(excerpt["path"])


def excerpt_block(excerpts: list[dict]) -> str:
    """The excerpts numbered from 1, each with its document and section, as the model reads them."""
    return "\n\n".join(f"[{n}] Document {e['doc_id']}, section: {section_of(e) or '(none)'}\n{e['text']}"
                       for n, e in enumerate(excerpts, 1))


def build(question: str, excerpts: list[dict]) -> tuple[str, str]:
    """The system rules, and the user message: the excerpts, then the question last."""
    return SYSTEM, f"Excerpts:\n\n{excerpt_block(excerpts) or '(none)'}\n\nQuestion: {question}"


def invalid(reason: str) -> dict:
    return {"status": "invalid", "answer": "", "citations": [], "error": reason}


def parse(text: str, excerpts: list[dict], question: str) -> dict:
    """The model's reply as the answer the evaluation scores: its status, its answer, and its citations as each cited
    excerpt's document and section, in the order cited, once each."""
    body = text.strip()
    if body.startswith("```"):  # a fenced reply, which some models send even when asked for JSON alone
        body = body[3:].removeprefix("json").strip().removesuffix("```").strip()
    try:
        reply = json.loads(body)
    except json.JSONDecodeError as error:
        return invalid(f"not JSON: {error}")
    if not isinstance(reply, dict) or reply.get("status") not in STATUSES:
        return invalid(f"no status of {' or '.join(STATUSES)}")
    if reply["status"] == "not_found":  # the same words whether the fact is missing or withheld
        return {"status": "not_found", "answer": NOT_FOUND[language(question)], "citations": []}
    answer = reply.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return invalid("answered with no answer")
    cited: list[dict] = []
    for number in reply.get("citations") or []:
        if isinstance(number, str) and number.strip().isdigit():
            number = int(number)
        if isinstance(number, int) and not isinstance(number, bool) and 1 <= number <= len(excerpts):
            citation = {"doc_id": excerpts[number - 1]["doc_id"], "section": section_of(excerpts[number - 1])}
            if citation not in cited:
                cited.append(citation)
    return {"status": "answered", "answer": answer.strip(), "citations": cited}
