#!/usr/bin/env python3
"""Convert a grounding-style dataset into a neuron-db fact pack (JSONL: one {"scope","fact"} per
line), ready for `neuron import`, the MCP `NEURON_MCP_PRELOAD` boot hook, or the WASM loadmany op.

Designed for google/FACTS-grounding-public (examples.csv): each row's `context_document` is the
fact-bearing prose; the other columns (system_instruction, user_request, full_prompt) are task
framing and are dropped. One row -> one scope (facts-<i>), so each document's context-relative
facts stay isolated AND each scope stays well under the semantic-recall window — which also keeps
paraphrase recall sharp.

Generalizes to any (document, optional-id) dataset via --column.

  python facts_to_pack.py examples.csv > pack.jsonl
  python facts_to_pack.py data.csv --column body --min-words 4 --scope-prefix doc- > pack.jsonl
  python facts_to_pack.py examples.csv > pack.jsonl && neuron --db facts.db --max 200000 import pack.jsonl

Output is wire-safe for every surface: no tabs/newlines, no leading '#', no questions, no fragments.
"""
import csv, json, re, sys, argparse

# words that end in '.' but do NOT end a sentence (so we don't split SEC / medical / legal prose mid-fact)
ABBR = {"inc.", "corp.", "co.", "ltd.", "llc.", "plc.", "u.s.", "u.k.", "e.g.", "i.e.", "etc.",
        "vs.", "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "no.", "fig.", "al.", "approx.", "est.",
        "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec."}

def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text.strip())
    pieces = re.split(r"(?<=[.?!])\s+(?=[\"'(]?[A-Z0-9])", text)
    merged, i = [], 0
    while i < len(pieces):
        s = pieces[i]
        # keep merging while this piece ends in an abbreviation or a single-letter initial ("J.").
        # (A real decimal like "3.5" has no space after the dot, so it is never a split point — no
        # need to guard it, and guarding "<digit>." wrongly merges a year-ending sentence like "2023.")
        last = s.split()[-1].lower() if s.split() else ""
        while i + 1 < len(pieces) and (last in ABBR or re.search(r"\b[A-Z]\.$", s)):
            i += 1
            s = s + " " + pieces[i]
            last = s.split()[-1].lower() if s.split() else ""
        merged.append(s.strip())
        i += 1
    return [m for m in merged if m]

def clean_fact(sent: str) -> str:
    fact = sent.replace("\t", " ").replace("\n", " ").strip()
    if fact.startswith("#"):
        fact = fact.lstrip("#").strip()    # a leading '#' is a comment marker in a pack
    return fact

def main():
    ap = argparse.ArgumentParser(description="grounding dataset CSV -> neuron-db fact pack (JSONL)")
    ap.add_argument("infile", help="CSV with a prose column (FACTS examples.csv by default)")
    ap.add_argument("--column", default="context_document", help="the fact-bearing prose column")
    ap.add_argument("--min-words", type=int, default=4, help="drop fragments shorter than this")
    ap.add_argument("--scope-prefix", default="facts-", help="scope = <prefix><row index>")
    a = ap.parse_args()
    with open(a.infile, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if a.column not in cols:
            sys.exit(f"column '{a.column}' not found; available: {cols}")
        rows = facts = 0
        for i, row in enumerate(reader):
            doc = (row.get(a.column) or "").strip()
            if not doc:
                continue
            rows += 1
            scope = f"{a.scope_prefix}{i}"
            for sent in split_sentences(doc):
                fact = clean_fact(sent)
                if len(fact.split()) < a.min_words or fact.endswith("?"):
                    continue               # neuron-db drops questions + tiny fragments anyway
                sys.stdout.write(json.dumps({"scope": scope, "fact": fact}, ensure_ascii=False) + "\n")
                facts += 1
    sys.stderr.write(f"wrote {facts} fact(s) from {rows} document(s)\n")

if __name__ == "__main__":
    main()
