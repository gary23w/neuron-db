# vs-vectors — a fair head-to-head: neuron-db vs dense vectors

A reproducible benchmark comparing neuron-db (lexical-episodic) against a real pretrained
dense-vector embedder on the **same frozen dataset**, scored by the **same retrieval-identity
rule**. Full write-up and the honest scorecard: **[../../docs/guide/VS_VECTORS.md](../../docs/guide/VS_VECTORS.md)**.

The methodology was adversarially reviewed for fairness before running: per-class scoring (never
blended), abstentions count as misses, a symmetric no-answer set, the pure-paraphrase class (dense
vectors' strength) included in force, and the embedding cost counted in every vector latency number.

## Files

- `vs_gen.py` — generates the frozen dataset: `facts.tsv`, `queries.tsv`, `corpus.txt`.
- `../../rust/neuron-core/examples/vs_vectors.rs` — the **neuron-db side** (in-process, microseconds).
- `vs_vectors_local.py` — the **dense-vector side**, a real local model (`all-MiniLM-L6-v2`, 384-d)
  via `torch`+`transformers`. Local on purpose: isolates the embedding architecture from network RTT.
- `vs_vectors_vec.py` — the same, using a **hosted** API (`text-embedding-3-small`, 1536-d). Needs
  `OPENAI_API_KEY`; a hosted query adds ~50–150 ms RTT on top of the local cost.
- `vs_report.py` — joins both engines' TSV into the per-class accuracy + latency + footprint tables.
- `facts.tsv` / `queries.tsv` / `corpus.txt` — the exact frozen inputs used for the published run.
- `SCORECARD.txt` — the saved output of the published run.

## Run it

```sh
cd examples/vs-vectors
python vs_gen.py                                              # writes facts.tsv, queries.tsv, corpus.txt

# neuron-db side (three modes), from the repo root:
cargo run --release --features "sqlite semantic" --example vs_vectors -- \
    examples/vs-vectors/facts.tsv examples/vs-vectors/queries.tsv lexical \
    > examples/vs-vectors/ndb_lexical.tsv
cargo run --release --features "sqlite semantic" --example vs_vectors -- \
    examples/vs-vectors/facts.tsv examples/vs-vectors/queries.tsv blended examples/vs-vectors/corpus.txt \
    > examples/vs-vectors/ndb_blended.tsv
cargo run --release --features "sqlite semantic" --example vs_vectors -- \
    examples/vs-vectors/facts.tsv examples/vs-vectors/queries.tsv blended \
    > examples/vs-vectors/ndb_blended_nocorpus.tsv

# dense-vector side (local model) + report:
python vs_vectors_local.py facts.tsv queries.tsv vec_results.tsv
python vs_report.py
```

Stdlib + `torch`/`transformers` (local model) only; no API key needed for the local baseline.
