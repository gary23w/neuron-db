"""NeuronRouter -- chain many neurons into one memory.

A single neuron recalls best when it holds distinct, not-too-many facts (see
BENCHMARKS.md: cue collisions grow with size). The router sidesteps that ceiling by
spreading facts across many small neuron *shards* and fanning a query out across them,
then returning the single best-matching value.

This is the answer to "can we chain neurons instead of stuffing everything into a giant
context?" Yes: the model never sees the whole store -- it asks a question and gets one
value back, no matter how many shards there are.

    r = NeuronRouter(per_shard=128)
    for fact in many_facts: r.observe(fact)     # auto-spills into new shards
    r.recall("what is the north gate code?")    # fan-out, best value -> one answer
"""
from __future__ import annotations
from typing import Optional
from .neuron import Neuron


class NeuronRouter:
    def __init__(self, per_shard: int = 128):
        self.per_shard = per_shard
        self.shards: list[Neuron] = [Neuron(max_facts=10 ** 9)]  # shards don't self-truncate; router controls size

    # --- write: fill the current shard, spill to a new one when full ---
    def observe(self, text: str) -> int:
        if self.shards[-1].fact_count >= self.per_shard:
            self.shards.append(Neuron(max_facts=10 ** 9))
        return self.shards[-1].observe(text)

    # --- read: fan out across shards, keep the strongest hit ---
    def recall(self, query: str) -> Optional[dict]:
        best = None; bk = (-1, -1, -1.0)  # (exact, overlap, coverage)
        for sh in self.shards:
            hit = sh.recall(query)
            if hit is None:
                continue
            sc = (hit.get("exact", 0), hit["overlap"], hit["coverage"])
            if sc > bk:
                bk = sc; best = hit
        return best

    def get(self, query: str) -> Optional[str]:
        hit = self.recall(query)
        return hit["value"] if hit else None

    @property
    def fact_count(self) -> int:
        return sum(s.fact_count for s in self.shards)

    @property
    def shard_count(self) -> int:
        return len(self.shards)

    def dump(self) -> list:
        return [s.dump() for s in self.shards]

    @classmethod
    def load(cls, blobs: list, per_shard: int = 128) -> "NeuronRouter":
        r = cls(per_shard)
        r.shards = [Neuron.load(b, max_facts=10 ** 9) for b in blobs] or [Neuron(max_facts=10 ** 9)]
        return r
