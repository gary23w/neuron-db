"""Conversation routing over a Neuron. Deterministic: a statement is stored and
acknowledged, a question is answered from memory or with an honest "i don't know".
Native arithmetic for +,-,*,/. (The trained gary-neuron adder model is a separate
project and is NOT used here -- neuron-db is model-free by design.)
"""
from __future__ import annotations
import re
from .neuron import Neuron, _content, _words, _isnum, COMMAND

QWORDS = {"what", "whats", "where", "who", "when", "how", "which"}
YNWORDS = {"am", "is", "are", "do", "does", "did", "can", "could", "will", "would", "was", "were", "have", "has"}
GREET = re.compile(r"^(hi+|hey+|hello+|yo|sup|howdy|good (morning|afternoon|evening))\b[\s!,.?]*$", re.I)
LAUGH = re.compile(r"^(lo+l+|lmao+|haha+|hehe+)\b", re.I)
THANKS = re.compile(r"^(thanks|thank you|thx|ty)\b", re.I)
BYE = re.compile(r"^(bye+|goodbye|later|gtg|good ?night)\b", re.I)
HOWRU = re.compile(r"how('s| is| are)? (it going|you( doing)?|things)", re.I)
SELFQ = re.compile(r"real\s+person|human|alive|robot|an?\s+ai\b|sentient|a\s+machine", re.I)
MATH = re.compile(r"(-?\d{1,12})\s*([+\-*/])\s*(-?\d{1,12})")
GREETS = ["hey. tell me things; i'll remember them.", "hi. what should i remember?",
          "ready. give me facts, ask them back later.", "memory online. go ahead."]
CHATR = ["noted.", "okay.", "got it.", "if it matters, say it like a fact -- i'll keep it."]
IDK = "i don't know right now."


def turn(n: Neuron, u: str) -> dict:
    """Returns {'reply', 'kind', 'wrote'}. kind in: ack recall idk smalltalk math self."""
    u = u.strip()
    uw = _words(u)
    first = u.split()[0].lower().strip("?,!") if u else ""
    questionish = ("?" in u) or (first in QWORDS)
    about = (uw & {"your", "you", "yours"}) and not (uw & {"my", "i", "im", "mine"})
    yn = (first in YNWORDS) or ("yes or no" in u.lower())
    if yn: questionish = True

    if about and re.search(r"what('s| is) your name|who are you", u.lower()):
        return _r("i'm a neuron -- an associative memory. you're the one i remember things about.", "self")
    if about and SELFQ.search(u):
        return _r("i'm a tiny memory, not a person. but i'll remember what you tell me.", "self")
    if GREET.match(u): return _r(GREETS[n.fact_count % len(GREETS)], "smalltalk")
    if LAUGH.match(u): return _r("glad that landed.", "smalltalk")
    if THANKS.match(u): return _r("anytime.", "smalltalk")
    if BYE.match(u): return _r("later. i'll remember.", "smalltalk")
    if questionish and HOWRU.search(u): return _r("running fine -- all of it yours.", "smalltalk")
    if COMMAND.match(u) and not (uw & {"my", "i", "im", "mine"}):
        return _r("i can't do that -- i remember facts and do arithmetic.", "smalltalk")

    # native arithmetic: + - * /  (word forms too)
    mm = u
    for word, sym in (("plus", "+"), ("minus", "-"), ("times", "*"), ("divided by", "/")):
        mm = re.sub(r"\b" + word + r"\b", sym, mm, flags=re.I)
    m = MATH.search(mm)
    if m and not re.search(r"=\s*-?\d", mm):
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        r = {"+": a + b, "-": a - b, "*": a * b, "/": (round(a / b, 6) if b else None)}.get(op)
        if r is not None:
            return _r(f"{a} {op} {b} = {r}", "math")

    if questionish:
        cw = _content(u)
        if not cw and re.search(r"\bwho am i\b", u.lower()): cw = {"name"}
        hit = n.recall(cw) if cw else None
        if yn and cw:
            if hit and hit["coverage"] >= 0.99: return _r("yes.", "recall")
            if hit: return _r(f'hmm -- what i remember is: {hit["fact"]}', "recall")
            return _r("not that i know.", "recall")
        if hit:
            v = hit["value"]
            conf = _isnum(v) or (v[:1].isupper() and len(hit["fact"].split()) <= 6)
            if conf and not hit["echo"]: return _r(f"{v}.", "recall")
            if hit["echo"] or len(hit["fact"].split()) > 6:
                q = hit["fact"].strip()
                return _r(f'what i remember: "{q if len(q) <= 140 else q[:137] + chr(8230)}"', "recall")
            return _r(f"{v}.", "recall")
        if cw:
            return _r(IDK, "idk")

    wrote = n.observe(u)
    if wrote:
        val = wrote[-1]["v"]
        r = f"noted -- {len(wrote)} facts." if len(wrote) > 1 else [f"got it -- {val}.", f"noted -- {val}.", f"okay -- {val}."][n.fact_count % 3]
        return {"reply": r, "kind": "ack", "wrote": len(wrote)}
    return _r(CHATR[len(u) % len(CHATR)], "smalltalk")


def _r(reply: str, kind: str) -> dict:
    return {"reply": reply, "kind": kind, "wrote": 0}
