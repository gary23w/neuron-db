#!/usr/bin/env python3
"""Head-to-head harness (dense-vector side): the mainstream LLM-memory pattern. Embeds every fact
with a real production model (OpenAI text-embedding-3-small, 1536-d), then answers each query by
embedding the query and cosine-searching the fact vectors. Scored IDENTICALLY to the neuron-db side
(vs_vectors.rs) by retrieval identity: a hit = the gold fact id is in the top-k.

Honest latency: we report BOTH end-to-end (query-embed API round-trip + cosine) and search-only
(cosine alone). A real ANN/HNSW index makes search sub-linear but CANNOT remove the query-embed
round-trip — that floor (~50-150ms hosted) is intrinsic to vectors and is the real basis of the
latency gap, not the cosine loop. Footprint reported separately by the caller (N*1536*4 here).
No-answer class scored symmetrically: top-1 cosine < THRESHOLD => abstain (a disclosed, practitioner
-typical relevance gate), mirroring neuron-db's built-in abstention.

Run:  python vs_vectors_vec.py <facts.tsv> <queries.tsv> [out.tsv]   (needs OPENAI_API_KEY)
"""
import json, os, sys, time, urllib.request, urllib.error

MODEL, DIM, K, ABSTAIN_T = "text-embedding-3-small", 1536, 3, 0.35

def embed(inputs, key):
    body = json.dumps({"model": MODEL, "input": inputs}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/embeddings", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return [d["embedding"] for d in json.loads(r.read())["data"]]

def embed_all(texts, key, batch=256):
    out = []
    for i in range(0, len(texts), batch):
        for attempt in range(5):
            try: out.extend(embed(texts[i:i+batch], key)); break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 4: time.sleep(2 * (attempt + 1)); continue
                raise
    return out

def dot(a, b):  # OpenAI embeddings are L2-normalized => dot == cosine
    s = 0.0
    for i in range(len(a)): s += a[i] * b[i]
    return s

def main():
    if len(sys.argv) < 3: sys.exit("usage: vs_vectors_vec.py <facts.tsv> <queries.tsv> [out.tsv]")
    facts_path, queries_path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    key = os.environ.get("OPENAI_API_KEY")
    if not key: sys.exit("set OPENAI_API_KEY")

    ids, fact_text = [], []
    for l in open(facts_path, encoding="utf-8"):
        if not l.strip(): continue
        fid, t = l.rstrip("\n").split("\t", 1)
        ids.append(int(fid)); fact_text.append(t)
    queries = []
    for l in open(queries_path, encoding="utf-8"):
        if not l.strip(): continue
        p = l.rstrip("\n").split("\t")
        queries.append((p[0], p[1] if len(p) > 1 else "NONE", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "?"))

    t0 = time.time(); fvecs = embed_all(fact_text, key); ingest_s = time.time() - t0
    footprint = len(fact_text) * DIM * 4
    print(f"#facts {len(fact_text)} #vec_bytes {footprint} #bytes_per_fact {footprint/len(fact_text):.1f} "
          f"#ingest_embed_s {ingest_s:.2f} #model {MODEL} #abstain_t {ABSTAIN_T}")

    rows = []
    for (q, gold_id, _gv, klass) in queries:
        t = time.time(); qv = embed([q], key)[0]; embed_ms = (time.time() - t) * 1000.0
        t = time.time()
        scored = sorted(((dot(qv, fvecs[i]), i) for i in range(len(fvecs))), reverse=True)[:K]
        search_ms = (time.time() - t) * 1000.0
        top = [(s, ids[i]) for s, i in scored]
        g = int(gold_id) if gold_id != "NONE" else -1
        hit1 = int(g >= 0 and top and top[0][1] == g)
        hit3 = int(g >= 0 and any(fid == g for _, fid in top))
        abstain = int(bool(top) and top[0][0] < ABSTAIN_T)
        e2e_ns = int((embed_ms + search_ms) * 1e6); s_ns = int(search_ms * 1e6)
        rows.append((klass, "vector", e2e_ns, s_ns, hit1, hit3, abstain))
        print(f"{klass}\tvector\t{e2e_ns}\t{s_ns}\t{hit1}\t{hit3}\t{abstain}")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"#facts {len(fact_text)} #vec_bytes {footprint} #bytes_per_fact {footprint/len(fact_text):.1f} "
                    f"#ingest_embed_s {ingest_s:.2f}\n")
            for r in rows: f.write("\t".join(str(x) for x in r) + "\n")

if __name__ == "__main__":
    main()
