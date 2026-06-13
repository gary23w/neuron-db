"""PlasticNeuron -- the store tier of a plastic memory: a neuron whose recall changes
with use, forms associations, and forgets, using only cheap O(1) scalar updates. No
neural net runs here; this is the substrate that decides WHAT enters the working set.

The thinking tier (the trained gary-neuron cortex + plastic hippocampus) runs on top of
the small working set this returns -- see docs/PLASTICITY.md. Plasticity is split so the
expensive neural part only ever sees a bounded window, never the whole store.

Mechanisms (all O(1) or O(neighbors), no background sweeps):
  * strength       each fact carries a weight, bumped on recall (Hebbian "use it")
  * decay          weight = w * 0.5 ** (age / half_life), computed LAZILY at read time
  * association    facts recalled close together get linked (fire together, wire together)
  * spreading      recall can return the hit PLUS its strongest associates (one hop)
  * consolidate    off-hot-path: merge duplicate-stem facts, prune decayed-to-nothing
"""
from __future__ import annotations
from typing import Optional
from .neuron import (Neuron, _content, _stem, _pick_value, REL_S, PETS, STOPVAL_S, _encode, ENT, LEADIN, _words, COMMAND, _sents)


class PlasticNeuron(Neuron):
    def __init__(self, max_facts: int = 500, half_life: float = 200.0, link_window: int = 3):
        super().__init__(max_facts)
        self.half_life = half_life          # in ticks; smaller = forgets faster
        self.link_window = link_window      # how many recent recalls a new recall links back to
        self.tick = 0
        self._w: dict[int, float] = {}      # episode id -> strength
        self._t: dict[int, int] = {}        # episode id -> last-touched tick
        self._links: dict[int, dict[int, float]] = {}  # id -> {id: weight}
        self._recent: list[int] = []        # recently activated ids (for co-activation linking)
        self._next_id = 0

    # ----- write -----
    def observe(self, text: str, entity: Optional[str] = None) -> list:
        wrote = super().observe(text, entity)   # returns the newly-appended episode dicts
        new_ids = []
        for e in wrote:                          # same dict objects that live in self.episodes
            e["_id"] = self._next_id; self._next_id += 1
            self._w[e["_id"]] = 1.0; self._t[e["_id"]] = self.tick
            new_ids.append(e["_id"])
        # facts stated together are structurally associated
        for i in range(len(new_ids)):
            for j in range(i + 1, len(new_ids)):
                self._link(new_ids[i], new_ids[j], 1.0)
        self.tick += 1
        return wrote

    # ----- plastic helpers -----
    def _eff(self, eid: int) -> float:
        age = self.tick - self._t.get(eid, self.tick)
        return self._w.get(eid, 1.0) * (0.5 ** (age / self.half_life))

    def _link(self, a: int, b: int, d: float):
        if a == b: return
        self._links.setdefault(a, {})[b] = self._links.setdefault(a, {}).get(b, 0.0) + d
        self._links.setdefault(b, {})[a] = self._links.setdefault(b, {}).get(a, 0.0) + d

    def reinforce(self, eid: int, amount: float = 1.0):
        self._w[eid] = self._eff(eid) + amount; self._t[eid] = self.tick

    # ----- read: same matching as Neuron, but ties broken by effective strength -----
    def recall(self, query) -> Optional[dict]:
        cue = _stem(_content(query)) if isinstance(query, str) else _stem(set(query))
        if not cue: return None
        pet_query = bool(cue & _stem({"pet", "animal"}))
        name_query = ("name" in cue) and not (cue & REL_S)
        if self._index is None or self._index_len != len(self.episodes): self._build_index()
        cand: set = set()
        for s in cue: cand.update(self._index.get(s, ()))
        if pet_query:
            for s in PETS: cand.update(self._index.get(s, ()))
        best = None; bk = (-1, -1.0, -1, -1, 0, -1)
        for i in sorted(cand):
            e = self.episodes[i]
            es = set(e["s"]); ov = len(cue & es); es_pet = bool(es & PETS)
            if ov < 1 and pet_query and es_pet: ov = 1
            if ov < 1: continue
            if (es & REL_S) - cue and not (pet_query and es_pet): continue
            if (cue & REL_S) - es and not (pet_query and es_pet): continue
            eid = e.get("_id", -1)
            selfp = 1 if (name_query and e.get("self")) else 0
            spec = -len(es - cue - STOPVAL_S)
            # overlap first (a clearly better match still wins), THEN learned strength,
            # then the original tie-breakers. This is the adaptive bit: a frequently-used
            # fact beats a merely-recent one on an otherwise-equal cue.
            sc = (ov, self._eff(eid), selfp, 1 if e.get("h", "") in cue else 0, spec, i)
            if sc > bk: bk = sc; best = e
        if best is None: return None
        bes = set(best["s"]); cov = len(cue & bes) / max(1, len(cue))
        if pet_query and (bes & PETS): cov = 1.0
        val, echo = _pick_value(best, cue, bool(cue & _stem({"many", "much", "number"})))
        # plastic side effects: reinforce the hit, wire it to recently-activated facts
        eid = best.get("_id", -1)
        if eid >= 0:
            self.reinforce(eid)
            for prev in self._recent[-self.link_window:]:
                self._link(eid, prev, 0.5)
            self._recent.append(eid); self._recent = self._recent[-32:]
            self.tick += 1
        return {"fact": best["t"], "value": val, "coverage": cov, "overlap": bk[0], "echo": echo, "strength": round(self._eff(eid), 3)}

    def recall_related(self, query, k: int = 3) -> list:
        """The hit plus its strongest associates -- one-hop spreading activation."""
        hit = self.recall(query)
        if hit is None: return []
        out = [hit]
        eid = next((e["_id"] for e in self.episodes if e["t"] == hit["fact"]), -1)
        nbrs = sorted(self._links.get(eid, {}).items(), key=lambda kv: -kv[1] * self._eff(kv[0]))
        by_id = {e.get("_id"): e for e in self.episodes}
        for nid, w in nbrs[:k]:
            e = by_id.get(nid)
            if e: out.append({"fact": e["t"], "value": e["v"], "link": round(w, 2), "strength": round(self._eff(nid), 3)})
        return out

    # ----- consolidation ("sleep"): off the hot path -----
    def consolidate(self, prune_below: float = 0.05) -> dict:
        merged = 0; pruned = 0
        # merge facts with identical stem-sets, keeping the strongest, summing strength
        by_key: dict[tuple, dict] = {}
        keep = []
        for e in self.episodes:
            key = tuple(e["s"])
            if key in by_key:
                a = by_key[key]; ai = a["_id"]; bi = e["_id"]
                self._w[ai] = self._eff(ai) + self._eff(bi); self._t[ai] = self.tick
                for nid, w in self._links.pop(bi, {}).items():
                    self._link(ai, nid, w)
                merged += 1
            else:
                by_key[key] = e; keep.append(e)
        # prune facts whose effective strength has decayed away
        survivors = []
        for e in keep:
            if self._eff(e["_id"]) >= prune_below or e.get("self"):
                survivors.append(e)
            else:
                pruned += 1; self._w.pop(e["_id"], None); self._t.pop(e["_id"], None); self._links.pop(e["_id"], None)
        self.episodes = survivors
        self._index = None
        return {"merged": merged, "pruned": pruned, "facts": len(self.episodes)}
