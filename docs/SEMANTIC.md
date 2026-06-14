# Semantic space & the book test

This documents neuron-db's **fuzzy-semantic** tier — how it resolves meaning without a
model or a dictionary — and the "ultimate test": ingesting whole books, the storage and
recall metrics at that scale, and an honest account of what forms a knowledge base and what
doesn't.

## 1. The problem

Lexical recall (stems + morphology + a curated synonym map) can't bridge open-vocabulary
paraphrase: *"the thing I use to get online"* shares no words with *"the wifi password
is …"*. Resolving that is what makes natural language work, and the brain doesn't do it with
a dictionary — it grounds meaning in **distributed experience**: a word means what its
contexts mean ("you shall know a word by the company it keeps").

## 2. The mechanism: a corpus-distributional semantic space

We build a continuous **semantic space** with *Random Indexing* (`src/semantic.rs`) — a
cheap, incremental alternative to word2vec/LSA, pure Rust, **no model, no dependencies**:

- every word owns a fixed **sparse random index vector** (256-dim, 12 nonzeros, derived
  deterministically from the word — never stored);
- a word's dense **context vector** is the running sum of the index vectors of the words it
  **co-occurs** with (a ±5-word window);
- words used in similar contexts end up **near each other** in the space, so cosine
  similarity captures meaning even when no characters are shared.

A query and a fact are each embedded as the normalized sum of their words' context vectors;
similarity is a dot product. It is wired into `NeuronDB` (feature `semantic`) as a **recall
fallback**: when lexical recall misses, the scope's facts are ranked in the semantic space
and the best is returned if it clears a similarity threshold. The lexical fast path is
untouched — the fallback only fires on a miss.

See the space itself — the 3D demo (`docs/demos/semantic-3d.html`) trains it in your browser
and projects the 256-D word vectors to 3 of the top-8 principal components (PCA computed in
the WASM core); words cluster by meaning and you can re-map the X/Y/Z axes to fly through
different dimensions. Clusters and nearest-neighbour links there are computed in the **true
256-D** space, not the projection.

This directly mirrors the neuroscience: meaning is **distributed** (co-occurrence vectors),
**contextual** (the query's other words shape its embedding), and **integrative** (cues sum
into one concept vector). It is also self-strengthening: the more text it sees, the better
the space — which is exactly why a book is the right test.

Verified (`tests/semantic_tier.rs`): with a small networking corpus trained in,
`recall("what is the thing I use to get online?")` returns the value of *"the wifi password
is vekam73"* — a paraphrase with **zero shared content words**. Lexical recall is still
preferred when it matches, and a truly unrelated query still abstains.

## 3. The book test — can it remember a book?

Harness: `examples/book_ingest.rs`. Corpus: 5 public-domain Project Gutenberg books
(Moby-Dick, Pride and Prejudice, Sherlock Holmes, Frankenstein, The Picture of Dorian Gray).

### Storage & ingestion (thesis-scale)

| metric | value |
|---|---|
| corpus | **598,684 words** → **29,123 sentence-facts** |
| ingest throughput | **~54,000 sentences/s (~1.1M words/s)** — whole corpus in ~0.5s |
| sqlite on disk | **8.7 MB** (~298 bytes / sentence) |
| semantic space | **25.8 MB** (23,919 vocab words, 545,540 tokens, dim 256) |
| total resident | **~34.5 MB** for 600k words |

Storage scales linearly: ~58 MB per million words (store + space). The store itself is tiny
(books are mostly the sentence text); the semantic space dominates at ~1 KB/word (256 × f32)
and is the obvious target for shrinking (lower dim or int8 quantization).

### Recall at scale (29,123 facts in ONE scope)

| path | latency |
|---|---|
| lexical recall, selective cue (proper nouns) | **~3 ms** |
| lexical recall, broad cue (common words) | **~3 ms** |
| semantic fallback (paraphrase) | **~90 ms** |
| reopen + first recall (load + index 29k facts) | **0.29 s** |

Lexical recall stays in **single-digit milliseconds** even over 29k facts in one scope.
The **semantic fallback is O(N)** — it embeds and scores every fact in the scope — so it
costs ~90 ms over 29k facts. That's the headline provisioning finding: fuzzy recall needs an
**embedding cache + an approximate-nearest-neighbour index** to stay fast at book scale; the
lexical path already is. In normal per-entity use (small scopes) the fallback is sub-ms.

### The semantic space actually learned meaning

Nearest words, learned **only** from the books (no external model):

```
whale   -> head, white, ship, boat, sea, sperm
ship    -> boat, sea, water, like, pequod        (pequod = the ship in Moby-Dick)
science -> subject, enthralled, course           (Frankenstein's vocabulary)
```

The maritime cluster (whale/ship/boat/sea/sperm/pequod) is unmistakable. Fuzzy recall over
the library:

```
"a gigantic sea creature of the ocean" -> "That sea beast Leviathan, which God ... Created"
"a horrifying dreadful fiend"          -> "It is so dreadful to think of our dear Arthur ..."
```

— matches grounded in meaning, not shared spelling.

## 4. Does it form a knowledge base and "hop" over a book?

Honest answer, two parts:

- **Associative / semantic knowledge: yes.** Ingesting the books builds (a) a recallable
  store of every sentence and (b) a distributional semantic space — so you can ask by cue or
  by meaning and get the right passage. That is a real, queryable knowledge base.
- **Multi-hop graph traversal over raw prose: not directly.** `recall_chain` walks a chain
  of *relations* (`owner → manager → timezone`), which needs structured `subject relation
  value` facts. A novel is prose, not triples, so there is nothing to hop along until those
  facts are **extracted**. To make a book hop-able you run an extraction pass first (an LLM
  or pattern IE turning sentences into facts), then `recall_chain` traverses them.

  Importantly, **there is no 3-hop limit.** `recall_chain` traverses an arbitrary-length
  path (the stress sim runs 4 hops; the engine is unbounded — each hop is one µs recall, see
  `COMPARISON.md`/`SYNAPSE.md`). The limiter for a book is **information extraction**, not
  hop count.

## 5. Where this leaves the "world-wide LLM memory" problem

For an LLM, this gives a memory that is **cheap, flat-cost, and now fuzzy**: lexical +
morphological + synonym + distributional recall, with unbounded server-side multi-hop over
*structured* facts. The remaining work to make it thesis-scale and fully semantic:

1. **Index the semantic fallback** (cache fact embeddings + ANN) so fuzzy recall is ms, not
   O(N), at book scale. *(highest leverage — turns the 90 ms above into sub-ms)*
2. **Shrink the space** (lower dim / int8) to cut the ~1 KB/word footprint.
3. **Persist the semantic space** (it is resident-only today; reopen reloads the store but
   must re-train the space).
4. **A fact-extraction step** so prose corpora become hop-able knowledge graphs.

## Reproduce

```sh
# download a few Project Gutenberg .txt files into a folder, then:
cargo run --release --features "sqlite semantic" --example book_ingest -- <folder>
cargo test --features "sqlite semantic" --test semantic_tier
```
