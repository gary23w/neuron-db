#!/usr/bin/env python3
"""Join both engines' frozen-dataset results into the honest per-class accuracy + latency tables.
Reads ndb_lexical.tsv, ndb_blended.tsv, ndb_blended_nocorpus.tsv (neuron-db) and vec_results.tsv
(dense vectors). Scoring is identical and retrieval-identity based (gold fact id in top-k); the
no-answer class is scored on abstention (abstain = correct). value_exact is a neuron-db-only bonus.
"""
import os, statistics as st
from collections import defaultdict

CLASSES = ["exact-id", "exact-lex", "paraphrase", "distractor", "none"]

def load(path, engine):
    """returns dict[class] -> list of row dicts"""
    rows = defaultdict(list)
    head = {}
    if not os.path.exists(path): return rows, head
    for l in open(path, encoding="utf-8-sig"):
        l = l.rstrip("\n")
        if not l.strip(): continue
        if l.startswith("#"):
            toks = l.split()
            for i in range(0, len(toks) - 1, 2):
                if toks[i].startswith("#"):
                    head[toks[i].lstrip("#")] = toks[i + 1]
            continue
        p = l.split("\t")
        klass = p[0]
        if engine == "ndb":
            rows[klass].append(dict(lat=int(p[2]), hit1=int(p[3]), hit3=int(p[4]), val=int(p[5]), abst=int(p[6])))
        else:  # vector
            rows[klass].append(dict(e2e=int(p[2]), search=int(p[3]), hit1=int(p[4]), hit3=int(p[5]), abst=int(p[6])))
    return rows, head

def pct(xs, q):
    xs = sorted(xs);
    return xs[min(int(len(xs) * q), len(xs) - 1)] if xs else 0

def acc_table(name, rows):
    print(f"\n## {name} — per-class accuracy")
    print(f"{'class':<12} {'n':>3} {'hit@1':>6} {'hit@3':>6} {'val-exact':>9}")
    for c in CLASSES:
        r = rows.get(c, [])
        if not r: continue
        n = len(r)
        if c == "none":
            ab = 100.0 * sum(x['abst'] for x in r) / n
            print(f"{c:<12} {n:>3} {'—':>6} {'—':>6} {'—':>9}   abstain(correct)={ab:.0f}%  false-positive={100-ab:.0f}%")
        else:
            h1 = 100.0 * sum(x['hit1'] for x in r) / n
            h3 = 100.0 * sum(x['hit3'] for x in r) / n
            v = (100.0 * sum(x.get('val', 0) for x in r) / n) if 'val' in r[0] else None
            vs = f"{v:.0f}%" if v is not None else "—"
            print(f"{c:<12} {n:>3} {h1:>5.0f}% {h3:>5.0f}% {vs:>9}")

def main():
    lab = os.path.dirname(os.path.abspath(__file__))
    lex, lh = load(os.path.join(lab, "ndb_lexical.tsv"), "ndb")
    bl, bh = load(os.path.join(lab, "ndb_blended.tsv"), "ndb")
    blnc, _ = load(os.path.join(lab, "ndb_blended_nocorpus.tsv"), "ndb")
    vec, vh = load(os.path.join(lab, "vec_results.tsv"), "vector")

    acc_table("neuron-db LEXICAL (default, no semantic)", lex)
    acc_table("neuron-db BLENDED (semantic re-rank, NO corpus — only fact text)", blnc)
    acc_table("neuron-db BLENDED (semantic re-rank, + generic corpus)", bl)
    if vec: acc_table("DENSE VECTORS (text-embedding-3-small, cosine)", vec)

    print("\n## Latency (per-query)")
    all_lat = [x['lat'] for c in lex for x in lex[c]]
    print(f"neuron-db lexical  : p50 {pct(all_lat,0.5)/1000:.1f} us   p95 {pct(all_lat,0.95)/1000:.1f} us   (in-process, no network, no model)")
    all_bl = [x['lat'] for c in bl for x in bl[c]]
    print(f"neuron-db blended  : p50 {pct(all_bl,0.5)/1000:.1f} us   p95 {pct(all_bl,0.95)/1000:.1f} us   (adds int8 semantic re-rank)")
    if vec:
        e2e = [x['e2e'] for c in vec for x in vec[c]]
        sr = [x['search'] for c in vec for x in vec[c]]
        print(f"vectors end-to-end : p50 {pct(e2e,0.5)/1e6:.1f} ms   p95 {pct(e2e,0.95)/1e6:.1f} ms   (query-embed RTT + cosine) <- production cost")
        print(f"vectors search-only: p50 {pct(sr,0.5)/1e6:.2f} ms   p95 {pct(sr,0.95)/1e6:.2f} ms   (cosine alone; embed RTT excluded)")
        ndb_us = pct(all_lat, 0.5) / 1000.0
        vec_ms = pct(e2e, 0.5) / 1e6
        print(f"   => neuron-db is ~{vec_ms*1000/ndb_us:,.0f}x faster end-to-end at the median ({ndb_us:.1f} us vs {vec_ms:.1f} ms)")

    print("\n## Footprint")
    nfacts = int(lh.get("facts", 0)); src = int(lh.get("src_text_bytes", 0))
    db = int(lh.get("db_bytes", 0)); sem = int(lh.get("sem_bytes", 0))
    print(f"facts: {nfacts}  | shared source text (both sides): {src} B ({src/max(nfacts,1):.0f} B/fact)")
    print(f"neuron-db on-disk (db+wal+shm, incl ~36KB fixed SQLite overhead): {db} B ({db/max(nfacts,1):.0f} B/fact)")
    print(f"neuron-db semantic space (feature ON): {sem} B")
    if vh:
        vb = int(vh.get("vec_bytes", 0))
        print(f"dense vectors ALONE (no text, no ANN graph): {vb} B ({vb/max(nfacts,1):.0f} B/fact)")
        print(f"   => dense vectors are {vb/max(db,1):.1f}x the size of neuron-db's entire store at this N;")
        print(f"      structural floor: 6144 B/fact (1536xf32) vs ~48 B/fact retrieval state => ~128x asymptotic")
        ie = vh.get("ingest_embed_s")
        if ie: print(f"   vector ingest: embedded all facts in {ie}s (every fact a model call); neuron-db ingest is local CPU")

if __name__ == "__main__":
    main()
