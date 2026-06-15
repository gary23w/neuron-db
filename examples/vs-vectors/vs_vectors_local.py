#!/usr/bin/env python3
"""Head-to-head harness (dense-vector side, LOCAL real model). Same frozen dataset and identical
retrieval-identity scoring as vs_vectors.rs. Uses a real pretrained sentence embedder
(all-MiniLM-L6-v2, 384-d) run locally on CPU via transformers+torch — NO network per query, so
this isolates the EMBEDDING ARCHITECTURE from network RTT (the honest, reproducible vector
baseline the design called for). A hosted API (text-embedding-3-small, 1536-d) would add network
RTT (~50-150ms) ON TOP of this forward-pass cost and 4x the footprint.

Latency reported BOTH ways: end-to-end (query forward-pass + cosine) and search-only (cosine alone).
No-answer class scored by a disclosed cosine gate (abstain if top-1 < THRESHOLD).

Run: python vs_vectors_local.py <facts.tsv> <queries.tsv> [out.tsv]
"""
import sys, time, math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME, DIM, K, ABSTAIN_T = "sentence-transformers/all-MiniLM-L6-v2", 384, 3, 0.45
torch.set_num_threads(1)  # pin to one core for a fair, stable latency number

_tok = _model = None
def _load():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME); _model.eval()

def embed(texts):
    _load()
    enc = _tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = _model(**enc).last_hidden_state                     # [B,T,H]
    mask = enc["attention_mask"].unsqueeze(-1).float()
    summed = (out * mask).sum(1); counts = mask.sum(1).clamp(min=1e-9)
    v = (summed / counts).numpy()                                  # mean pooling
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
    return v.astype(np.float32)

def main():
    if len(sys.argv) < 3: sys.exit("usage: vs_vectors_local.py <facts.tsv> <queries.tsv> [out.tsv]")
    facts_path, queries_path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    ids, fact_text = [], []
    for l in open(facts_path, encoding="utf-8"):
        if not l.strip(): continue
        fid, t = l.rstrip("\n").split("\t", 1); ids.append(int(fid)); fact_text.append(t)
    queries = []
    for l in open(queries_path, encoding="utf-8"):
        if not l.strip(): continue
        p = l.rstrip("\n").split("\t")
        queries.append((p[0], p[1] if len(p) > 1 else "NONE", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "?"))

    t0 = time.time()
    fvecs = np.vstack([embed(fact_text[i:i+64]) for i in range(0, len(fact_text), 64)])
    ingest_s = time.time() - t0
    footprint = len(fact_text) * DIM * 4
    print(f"#facts {len(fact_text)} #vec_bytes {footprint} #bytes_per_fact {footprint/len(fact_text):.1f} "
          f"#ingest_embed_s {ingest_s:.2f} #model {MODEL_NAME} #dim {DIM} #abstain_t {ABSTAIN_T}")

    rows = []
    for (q, gold_id, _gv, klass) in queries:
        t = time.time(); qv = embed([q])[0]; embed_ms = (time.time() - t) * 1000.0
        t = time.time()
        sims = fvecs @ qv
        top = np.argsort(-sims)[:K]
        search_ms = (time.time() - t) * 1000.0
        g = int(gold_id) if gold_id != "NONE" else -1
        top_ids = [ids[i] for i in top]
        hit1 = int(g >= 0 and top_ids and top_ids[0] == g)
        hit3 = int(g >= 0 and g in top_ids)
        abstain = int(len(top) > 0 and float(sims[top[0]]) < ABSTAIN_T)
        e2e_ns = int((embed_ms + search_ms) * 1e6); s_ns = int(search_ms * 1e6)
        rows.append((klass, "vector", e2e_ns, s_ns, hit1, hit3, abstain))
        print(f"{klass}\tvector\t{e2e_ns}\t{s_ns}\t{hit1}\t{hit3}\t{abstain}")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"#facts {len(fact_text)} #vec_bytes {footprint} #bytes_per_fact {footprint/len(fact_text):.1f} "
                    f"#ingest_embed_s {ingest_s:.2f} #dim {DIM}\n")
            for r in rows: f.write("\t".join(str(x) for x in r) + "\n")

if __name__ == "__main__":
    main()
