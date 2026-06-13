"""bridge.py — the optional model tier: wire the trained gary-neuron brain on top of the
store, so a neuron-db can *think* over what it recalls instead of only returning a value.

neuron-db ships MODEL-FREE on purpose (zero dependency, microsecond recall). The trained
gary-neuron cortex lives in its own project. This bridge connects the two WITHOUT bundling
weights into this repo:

    pip install neuron-db[model]            # numpy + tokenizers
    export NEURON_MODEL_DIR=/path/to/gary-neuron-chat   # a checkout with cortex.npz etc.

    from neuron_db.plastic import PlasticNeuron
    from neuron_db.bridge import GaryNeuronBridge

    store = PlasticNeuron(); store.observe("the launch is on Friday")
    brain = GaryNeuronBridge()              # loads the cortex from NEURON_MODEL_DIR
    brain.think(store, "when do we launch?")  # store retrieves the working set; cortex answers

If the model isn't present, importing the class is fine; constructing it raises a clear
message telling you where to get the model. Nothing here forces a dependency on the core.

The model dir must contain: cortex.npz (or brain.npz), petite_vocab.json, petite_merges.txt,
and gpt_numpy.py — i.e. a gary-neuron-chat checkout (github / HF gary23w/gary-neuron-chat).
"""
from __future__ import annotations
import os
from typing import Optional


class GaryNeuronBridge:
    """The hippocampus/neocortex tier over a neuron-db store. Loads the cortex lazily."""

    def __init__(self, model_dir: Optional[str] = None, max_new: int = 16):
        self.model_dir = model_dir or os.environ.get("NEURON_MODEL_DIR")
        self.max_new = max_new
        if not self.model_dir or not os.path.isdir(self.model_dir):
            raise RuntimeError(
                "gary-neuron model not found. neuron-db is model-free by design; the trained "
                "cortex lives in the gary-neuron-chat project. Set NEURON_MODEL_DIR to a "
                "checkout (with cortex.npz, petite_vocab.json, petite_merges.txt, gpt_numpy.py) "
                "and `pip install numpy tokenizers`.")
        import sys, numpy as np
        sys.path.insert(0, self.model_dir)
        try:
            import gpt_numpy as G
            from tokenizers import ByteLevelBPETokenizer
        except Exception as e:
            raise RuntimeError(f"model deps missing ({e}). `pip install numpy tokenizers` and "
                               "ensure gpt_numpy.py is in NEURON_MODEL_DIR.")
        self._G = G; self._np = np
        self.tok = ByteLevelBPETokenizer(f"{self.model_dir}/petite_vocab.json",
                                         f"{self.model_dir}/petite_merges.txt")
        src = f"{self.model_dir}/cortex.npz"
        if not os.path.exists(src):
            src = f"{self.model_dir}/brain.npz"            # a saved brain works too
        z = np.load(src, allow_pickle=True)
        self.P = {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith("P/")}
        E = self.P["lnf.weight"].shape[0]
        L = max(int(k.split(".")[1]) for k in self.P if k.startswith("blocks.")) + 1
        self.CFG = dict(E=E, H=4, L=L, BLK=self.P["pos.weight"].shape[0])

    def working_set(self, store, query: str, k: int = 3) -> list:
        """The bounded window the store hands the model — its strongest relevant facts."""
        if hasattr(store, "recall_related"):
            return [r["fact"] for r in store.recall_related(query, k=k)]
        hit = store.recall(query)
        return [hit["fact"]] if hit else []

    def think(self, store, query: str, k: int = 3) -> str:
        """Retrieve a working set from the store, then let the cortex answer over it."""
        facts = self.working_set(store, query, k=k)
        prompt = "".join(f"U: {f}\nG: noted.\n" for f in facts) + f"U: {query}\nG:"
        ids = self.tok.encode(prompt).ids[-self.CFG["BLK"]:]
        out = []
        for _ in range(self.max_new):
            x = self._np.array([ids[-self.CFG["BLK"]:]])
            lg, _ = self._G.forward(self.P, x, self.CFG)
            nx = int(lg[0, -1].argmax())
            if nx == 0:
                break
            piece = self.tok.decode([nx])
            if "\n" in piece:
                break
            ids.append(nx); out.append(nx)
        return self.tok.decode(out).strip()
