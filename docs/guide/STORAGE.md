# Storage capacity: neuron-db vs commercial database stores

The question: for a fixed amount of disk, how many more "memories" can neuron-db hold than
the databases people normally reach for when they give an LLM long-term memory?

Short answer: about 33x to 257x more than a float32 vector store at common embedding sizes,
and roughly 130x at the 1536-dim default. It is still about 5x denser than a binary-quantized
vector store that has already given up most of its recall accuracy. The reason is structural.
A vector store spends 1.5 to 12 KB per item on a dense embedding so it can retrieve by
content. neuron-db adds no embedding bytes at all. Its retrieval index is stems plus a few
scalars, and that fits inside the same bytes as the text.

This page shows the measurement, the comparison, and the caveats, so the number holds up
when someone pushes on it.

## Measurement

The 48-bytes/fact figure below was measured on the original **legacy-python** prototype
(now removed; preserved on the `legacy-python` branch), which exposed a `dump()` serializer:

```python
# legacy-python (removed) — preserved on the legacy-python branch
from neuron_db.plastic import PlasticNeuron
n = PlasticNeuron(half_life=1e9, max_facts=10**9)
for i in range(20000):
    n.observe(f"the server{i} holds value alpha{i}")
print(len(n.dump().encode()) / 20000, "bytes/fact serialized")
```

In the current Rust crate the equivalent store is `neuron_core::plastic::PlasticNeuron`
(see `rust/neuron-core/examples/plastic_adaptive.rs`); it persists facts more tightly than
the prototype, so 48 bytes/fact is the conservative upper bound here.

We loaded 20,000 plain-language facts (average source sentence 39 bytes) into a
`PlasticNeuron`, ran recall so the strength, timestamp, and link state actually exists, then
measured the serialized footprint from `dump()`, which is the bytes you persist:

```
neuron-db serialized state:  48.0 bytes / fact   (text + stem index + plastic state)
```

48 bytes per fact works out to roughly 22.4 million facts per GiB.

## The comparison

Every store below is sized for the same job: hold a short fact and be able to retrieve it by
content. For a vector DB that means text plus a dense embedding plus an ANN index. For
neuron-db it means text plus stem index plus scalar plasticity. Per-row metadata is charged
at about 28 bytes across the board.

| store | retrieval | bytes/item | items / GiB | vs neuron-db |
|---|---|--:|--:|--:|
| neuron-db (serialized) | exact-stem associative + plastic | 48 | 22,351,461 | 1.0x |
| Vector DB, binary quant, 1536-d | semantic ANN, heavy recall loss | 259 | 4,145,097 | 5.4x |
| Vector DB, int8 SQ, 1536-d | semantic ANN, minor recall loss | 1,603 | 669,816 | 33.4x |
| Vector DB, f32, 768-d | semantic ANN | 3,139 | 342,061 | 65.3x |
| Vector DB, f32, 1536-d (no index) | semantic | 6,211 | 172,876 | 129.3x |
| pgvector / Qdrant, f32 1536-d + HNSW | semantic ANN | 9,283 | 115,667 | 193.2x |
| Vector DB, f32, 3072-d (OpenAI large) | semantic ANN | 12,355 | 86,907 | 257.2x |

The last column reads as "neuron-db fits this many more facts in the same disk."

### How this maps to real products

pgvector (Postgres `vector(1536)`) stores 1536 x 4 = 6,144 bytes per vector, plus the row,
plus an HNSW index that usually adds 30 to 60 percent. The "+ HNSW" row is the realistic
on-disk figure, about 193x heavier than neuron-db per item.

Pinecone, Weaviate, Milvus, and Qdrant all keep the dense vector as the primary object, so
storage scales as `dim x 4`. They land in the same range as the f32 rows. Redis with
RediSearch is the same `dim x 4` but in RAM, which makes the density gap more expensive
there, not less.

Quantization is the fair counter-argument, so it is in the table. int8 scalar quantization
(1 byte per dim) and binary quantization (1 bit per dim) shrink the vector a lot. int8 is
still 33x heavier than neuron-db. Binary, still 5.4x heavier, throws away a large share of
recall accuracy, which is usually the reason you reached for vectors in the first place.

## Why the gap exists

Semantic retrieval pays for meaning by storing a coordinate in a 384 to 3072 dimensional
space for every item. neuron-db pays for cue-and-association retrieval by storing a few
discrete stems and scalar weights, which is one to three orders of magnitude smaller.

## Caveats

48 bytes is the serialized figure. In live Python RAM the same store is about 1.4 KB per
fact because of interpreter object overhead in dict, set, and str headers, not because of
information. The Rust port stores it far tighter. The persisted number is the fair
apples-to-apples comparison against a vector DB's on-disk size.

The multiplier depends on fact length. These facts are short, 39 bytes. For long documents
the text dominates both stores and the ratio shrinks toward `1 + embedding/text`.
neuron-db's advantage is largest where memory is many small facts, which is exactly the
LLM-memory case: preferences, entities, events, and settings.

It is a different capability. neuron-db does exact and cue associative recall with scalar
plasticity. It is not cosine-similarity semantic search, it is not ACID rows, and it does
not join. "The thing I use to get online" will not match "wifi password" without an
embedding. The honest framing is scalar-first. neuron-db handles the cheap, high-volume
lookup-and-adapt path at much higher density, and you add a vector tier only for the queries
that truly need meaning matching, paying for vectors only there.

Vector DBs also buy things neuron-db does not have, including fuzzy semantic matching,
cross-lingual recall, and similarity ranking. This is a storage-density comparison for
associative memory, not a claim that neuron-db replaces a vector DB everywhere.

## Bottom line

If your LLM memory is a large, growing pile of small facts that you mostly retrieve by name
or association, neuron-db holds around two orders of magnitude more of them per GiB than a
float32 vector store, with O(1) updates and microsecond recall, and no model, GPU, or index
to maintain. Reach for vectors only on the slice of queries that actually need semantic
similarity.
