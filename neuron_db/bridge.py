"""bridge.py - the optional model tier: wire the trained gary-neuron brain on top of the
store, so a neuron-db can *think* over what it recalls instead of only returning a value.

neuron-db ships MODEL-FREE on purpose. The trained cortex is published at
gary23w/gary-neuron-emergent and is loaded on demand. Nothing here forces a dependency on
the core; importing the class is always safe.
"""
from __future__ import annotations
import os
from typing import Optional


class GaryNeuronBridge:
    HF_REPO = "gary23w/gary-neuron-emergent"

    def __init__(self, model_dir: Optional[str] = None, max_new: int = 16):
        self.model_dir = model_dir or os.environ.get("NEURON_MODEL_DIR")
        self.max_new = max_new
        if not self.model_dir or not os.path.isdir(self.model_dir):
            try:
                from huggingface_hub import snapshot_download
                self.model_dir = snapshot_download(self.HF_REPO)
            except Exception:
                raise RuntimeError(
                    "gary-neuron model not found. neuron-db is model-free by design. Either set "
                    "NEURON_MODEL_DIR to a local checkout, or `pip install huggingface_hub numpy "
                    f"tokenizers` to auto-download {self.HF_REPO}. gpt_numpy.py must be importable "
                    "(it ships with gary-neuron-chat).")
        import sys, numpy as np
        sys.path.insert(0, self.model_dir)
        try:
            import gpt_numpy as G
            from tokenizers import ByteLevelBPETokenizer
        except Exception as e:
            raise RuntimeError(f"model deps missing ({e}). `pip install numpy tokenizers` and "
                               "ensure gpt_numpy.py is importable.")
        self._G = G; self._np = np
        self.tok = ByteLevelBPETokenizer(f"{self.model_dir}/petite_vocab.json",
                                         f"{self.model_dir}/petite_merges.txt")
        src = f"{self.model_dir}/cortex.npz"
        if not os.path.exists(src):
            src = f"{self.model_dir}/brain.npz"
        z = np.load(src, allow_pickle=True)
        self.P = {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith("P/")}
        E = self.P["lnf.weight"].shape[0]
        L = max(int(k.split(".")[1]) for k in self.P if k.startswith("blocks.")) + 1
        self.CFG = dict(E=E, H=4, L=L, BLK=self.P["pos.weight"].shape[0])

    def working_set(self, store, query: str, k: int = 3) -> list:
        if hasattr(store, "recall_related"):
            return [r["fact"] for r in store.recall_related(query, k=k)]
        hit = store.recall(query)
        return [hit["fact"]] if hit else []

    def think(self, store, query: str, k: int = 3) -> str:
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
