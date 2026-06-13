from __future__ import annotations
from typing import Union, List
from .plastic import PlasticNeuron
from .neuron import _sents

class ChunkedReader:
    """Normalize an arbitrarily large context to a cortex-sized input: chunk -> store
    selects the relevant chunk -> cortex generates over only that small window."""
    def __init__(self, bridge, top_k: int = 1):
        self.bridge = bridge
        self.top_k = top_k

    def answer(self, context: Union[str, List[str]], query: str, top_k: int = None) -> str:
        k = top_k or self.top_k
        chunks = list(context) if isinstance(context, (list, tuple)) else _sents(context, cap=10**9)
        store = PlasticNeuron(max_facts=10**9)
        for c in chunks:
            store.observe(c)
        if k <= 1:
            hit = store.recall(query)
            ws = [hit["fact"]] if hit else []
        else:
            ws = [r["fact"] for r in store.recall_related(query, k=k)]
        if not ws:
            return "i don't know right now."
        return self.bridge.generate(ws, query)
