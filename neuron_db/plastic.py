"""PlasticNeuron -- the store tier of a plastic memory: recall changes with use, forms
associations, and forgets, via cheap O(1) scalar updates. No neural net runs here.
Decay only changes recall RANKING; it never deletes a fact. Only consolidate() removes
anything, and it protects pinned + self facts. Plain Neuron/NeuronDB do not decay."""
from __future__ import annotations
from typing import Optional
from .neuron import (Neuron, _content, _stem, _pick_value, REL_S, PETS, STOPVAL_S)


class PlasticNeuron(Neuron):
    def __init__(self, max_facts: int = 500, half_life=200.0, link_window: int = 3):
        super().__init__(max_facts)
        # half_life in ticks; None/0/inf => NO decay (permanent strength).
        self.half_life = half_life
        self.link_window = link_window
        self.tick = 0
        self._w: dict = {}
        self._t: dict = {}
        self._links: dict = {}
        self._recent: list = []
        self._pinned: set = set()
        self._next_id = 0

    def observe(self, text: str, entity: Optional[str] = None) -> list:
        wrote = super().observe(text, entity)
        new_ids = []
        for e in wrote:
            e["_id"] = self._next_id; self._next_id += 1
            self._w[e["_id"]] = 1.0; self._t[e["_id"]] = self.tick
            new_ids.append(e["_id"])
        for i in range(len(new_ids)):
            for j in range(i + 1, len(new_ids)):
                self._link(new_ids[i], new_ids[j], 1.0)
        self.tick += 1
        return wrote

    def _eff(self, eid: int) -> float:
        w = self._w.get(eid, 1.0)
        if not self.half_life or self.half_life == float("inf"):
            return w
        age = self.tick - self._t.get(eid, self.tick)
        return w * (0.5 ** (age / self.half_life))

    def pin(self, eid: int): self._pinned.add(eid)

    def _link(self, a: int, b: int, d: float):
        if a == b: return
        self._links.setdefault(a, {})[b] = self._links.setdefault(a, {}).get(b, 0.0) + d
        self._links.setdefault(b, {})[a] = self._links.setdefault(b, {}).get(a, 0.0) + d

    def reinforce(self, eid: int, amount: float = 1.0):
        self._w[eid] = self._eff(eid) + amount; self._t[eid] = self.tick

    def recall(self, query) -> Optional[dict]:
        cue = _stem(_content(query)) if isinstance(query, str) else _stem(set(query))
        if not cue: return None
        pet_query = bool(cue & _stem({"pet", "animal"}))
        name_query = ("name" in cue) and not (cue & REL_S)
        if self._index is None or self._index_len != len(self.episodes): self._build_index()
        cand = set()
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
            sc = (ov, self._eff(eid), selfp, 1 if e.get("h", "") in cue else 0, spec, i)
            if sc > bk: bk = sc; best = e
        if best is None: return None
        bes = set(best["s"]); cov = len(cue & bes) / max(1, len(cue))
        if pet_query and (bes & PETS): cov = 1.0
        val, echo = _pick_value(best, cue, bool(cue & _stem({"many", "much", "number"})))
        eid = best.get("_id", -1)
        if eid >= 0:
            self.reinforce(eid)
            for prev in self._recent[-self.link_window:]:
                self._link(eid, prev, 0.5)
            self._recent.append(eid); self._recent = self._recent[-32:]
            self.tick += 1
        return {"fact": best["t"], "value": val, "coverage": cov, "overlap": bk[0], "echo": echo, "strength": round(self._eff(eid), 3)}

    def recall_related(self, query, k: int = 3) -> list:
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

    # ----- neurotransmitter-style spreading activation -----
    def recall_spreading(self, query, hops: int = 2, k: int = 10,
                         reuptake: float = 0.6, fan: int = 6) -> list:
        """Recall the way the brain does it: chemical signalling, not a forward pass.

        A cue triggers transmitter RELEASE at the neurons it matches -- graded
        (excitatory) by how well it matches and how strong that synapse already
        is. A relation conflict releases INHIBITORY transmitter instead, gating
        the neuron off (a dog-name cue must not fire the brother-name memory).
        Activation then SPREADS across synapses to connected neurons, losing a
        `reuptake` fraction at every hop, so it fades with distance. Only the
        strongest `fan` synapses of each neuron fire (release is sparse). After
        `hops` steps the most-activated neurons ARE the recalled set.

        It is multi-hop, so it surfaces facts that share NO word with the query
        but are wired to something that does -- association, not string match.
        And it is all scalar arithmetic over a sparse graph: microseconds, no
        cortex, no GPU. This is a pure read; it does not mutate the store.
        """
        cue = _stem(_content(query)) if isinstance(query, str) else _stem(set(query))
        if not cue: return []
        if self._index is None or self._index_len != len(self.episodes): self._build_index()
        pet_query = bool(cue & _stem({"pet", "animal"}))
        cand: set = set()
        for s in cue: cand.update(self._index.get(s, ()))
        if pet_query:
            for s in PETS: cand.update(self._index.get(s, ()))
        act: dict = {}                          # eid -> transmitter level
        for i in cand:
            e = self.episodes[i]; es = set(e["s"]); ov = len(cue & es)
            es_pet = bool(es & PETS)
            if ov < 1 and pet_query and es_pet: ov = 1
            if ov < 1: continue
            # inhibitory gate: a relation mismatch suppresses release entirely
            if (es & REL_S) - cue and not (pet_query and es_pet): continue
            if (cue & REL_S) - es and not (pet_query and es_pet): continue
            eid = e.get("_id", -1)
            if eid < 0: continue
            # excitatory release, graded by match quality x synaptic strength
            act[eid] = act.get(eid, 0.0) + ov * max(self._eff(eid), 1e-6)
        if not act: return []
        seeds = set(act)
        by_id = {e.get("_id"): e for e in self.episodes}
        # propagate: each hop carries (1-reuptake) of a neuron's signal onward
        frontier = dict(act)
        for _ in range(max(0, hops)):
            nxt: dict = {}
            for src, a in frontier.items():
                if a <= 1e-9: continue
                nbrs = self._links.get(src)
                if not nbrs: continue
                top = sorted(nbrs.items(), key=lambda kv: -kv[1] * self._eff(kv[0]))[:fan]
                norm = sum(w for _, w in top) or 1.0
                for nid, w in top:
                    sig = a * (1.0 - reuptake) * (w / norm) * min(1.0, self._eff(nid))
                    if sig <= 1e-9: continue
                    nxt[nid] = nxt.get(nid, 0.0) + sig
            for nid, s in nxt.items():
                act[nid] = act.get(nid, 0.0) + s
            frontier = nxt
            if not frontier: break
        ranked = sorted(act.items(), key=lambda kv: -kv[1])[:k]
        out = []
        for eid, a in ranked:
            e = by_id.get(eid)
            if e is None: continue
            out.append({"fact": e["t"], "value": e["v"], "activation": round(a, 4),
                        "seed": eid in seeds, "strength": round(self._eff(eid), 3)})
        return out

    def consolidate(self, prune_below: float = 0.05) -> dict:
        merged = 0; pruned = 0
        by_key = {}; keep = []
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
        survivors = []
        for e in keep:
            if self._eff(e["_id"]) >= prune_below or e.get("self") or e["_id"] in self._pinned:
                survivors.append(e)
            else:
                pruned += 1; self._w.pop(e["_id"], None); self._t.pop(e["_id"], None); self._links.pop(e["_id"], None)
        self.episodes = survivors
        self._index = None
        return {"merged": merged, "pruned": pruned, "facts": len(self.episodes)}
