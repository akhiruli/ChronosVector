# INT8 quantized storage

ChronosVector supports storing vectors as `int8_t` quantities plus a per-vector `float` scale factor instead of the default `float32` storage. This cuts memory 3.6–3.9× and speeds up queries 2.6–4.5× on the workloads we've measured — at the cost of a small, workload-dependent recall drop.

**As of v0.2, INT8 is compiled into the library by default.** Users still choose between `CHRONOSV_DTYPE_FLOAT32` (the runtime default) and `CHRONOSV_DTYPE_INT8` at engine construction time.

## When to use INT8

Enable `cfg.storage_dtype = CHRONOSV_DTYPE_INT8` when:

- You're storing **text or document embeddings** (BERT, MiniLM, E5, BGE, OpenAI embeddings, other unit-normalized models). Recall loss is minimal (<1%); memory + speed wins are large.
- You're storing **any embedding where the vectors are unit-normalized** (magnitude 1.0). Symmetric quantization has a tight, predictable dynamic range in this case.
- Your workload is **memory-bound at query time** (large corpus that spills out of L2/L3 cache). INT8's 4× smaller footprint translates directly into faster queries at the corpus sizes where cache pressure matters.
- Your workload is **query-heavy relative to ingest**. INT8 ingest is ~2.5× slower than FP32 due to per-vector quantization on the append path.

Stay on FP32 when:

- You're doing **final ranking** (as opposed to candidate generation) and can't tolerate any recall drop.
- Your embeddings are **not unit-normalized** and have wide value ranges — INT8's per-vector symmetric quantization loses precision to the outlier dimensions.
- You have **all-positive embeddings** (word frequencies, non-negative feature vectors) — symmetric quantization wastes half its resolution on values that never appear.
- Your workload is **ingest-heavy** and per-append latency matters more than per-query latency.

## Measured recall

Two reference datasets, both run through the harness at `tests/int8_recall/`:

| Dataset | Dim | Corpus | Metric | FP32 Recall@10 | INT8 Recall@10 | Drop | Notes |
|---|---|---|---|---|---|---|---|
| **BERT / MiniLM** (20-newsgroups sentences) | 384 | 50,000 | cosine | 0.9982 | **0.9915** | -0.66 pp | Unit-normalized; representative of text/RAG workloads |
| **SIFT-1M** (visual descriptors) | 128 | 1,000,000 | euclidean | 0.9992 | **0.9683** | -3.09 pp | Raw CV features with outlier dimensions |

**Reading these numbers:** BERT-family embeddings fall in the "essentially free" band; SIFT falls in the "great tradeoff, use rerank for final ranking" band. Both are shipping-quality outcomes.

## Measured performance

Same runs as above:

| Dataset | FP32 query latency | INT8 query latency | Speedup |
|---|---|---|---|
| BERT (50k × 384) | 2,202 µs | 492 µs | **4.5×** |
| SIFT-1M (1M × 128) | 12,913 µs | 4,922 µs | **2.6×** |

| Dataset | FP32 hot memory | INT8 hot memory | Reduction |
|---|---|---|---|
| BERT (50k × 384) | 96.8 MiB | 25.0 MiB | **3.9× smaller** |
| SIFT-1M (1M × 128) | 524 MiB | 144 MiB | **3.6× smaller** |

Speedup is largely driven by memory bandwidth: at the corpus sizes where FP32 spills out of L2/L3, INT8's smaller footprint keeps more of the working set hot. The BERT run shows a larger speedup than SIFT even though the corpus is 20× smaller, because the 384-dim vectors are larger individually — each cache line goes further with INT8.

Ingest is 2.0–2.5× slower for INT8 (the per-vector `max_abs → scale → round` computation on the append path). For streaming workloads where ingest rate matters, weigh this tradeoff explicitly.

## Interpretation guide

| Your measured Recall@10 vs FP32 | Verdict |
|---|---|
| ≥ 0.99 | INT8 is essentially free — enable it |
| 0.95 – 0.99 | Great tradeoff for 4× memory savings; safe for retrieval and coarse ranking; use FP32 rerank for final ordering |
| 0.85 – 0.95 | Use INT8 as coarse filter only; always rerank top-100 with FP32 originals |
| < 0.85 | Your embeddings don't quantize well. Stay on FP32, or use INT8 only for pre-filtering large candidate sets |

## Quantization scheme (technical)

- **Per-vector symmetric quantization.** For each vector `v`, compute `scale = max(|v|) / 127` and store `q[i] = round(v[i] / scale)`.
- **Zero-vector special case.** If `max(|v|) == 0`, store `q[i] = 0` and `scale = 0`; distance kernels return 0 for these.
- **On-disk format.** `int8_t q[dim]` + `float scale` + `float norm` per vector. Roughly `1 * dim + 8` bytes per stored vector, plus fixed engine overhead.

The scheme is deliberately simple. Alternatives considered but not implemented for v0.2: asymmetric quantization (would improve all-positive embedding cases), product quantization (would compress further at cost of complexity), scalar Matryoshka (variable precision per dimension). These are on the private roadmap for post-v0.2.

## Validating on your own embeddings

The `tests/int8_recall/` harness lets you measure the recall drop on your actual data before enabling INT8 in production. See `tests/int8_recall/README.md` for prep scripts (SIFT-1M, BERT) and how to plug in your own dataset.

If your measured recall is materially worse than the SIFT / BERT reference numbers above, please open a GitHub issue with:

- The number
- A description of the embedding model (BERT-family / CLIP / autoencoder / custom / etc.)
- Rough dimensionality and whether vectors are unit-normalized

That's the class of feedback that would improve the quantization scheme.

## Explicitly not covered

- **INT8 in the RocksDB cold tier** — the block codec supports it (see `test_block_codec.cpp`), but the recall/perf story above is measured against in-memory-only engines. Real cold-tier round-trips add serialization overhead that isn't the quantization scheme's fault.
- **INT4 or binary quantization** — see the private roadmap; binary in particular offers 32× compression with dedicated ARM NEON kernels. Not yet implemented.
- **Multi-modal embeddings (CLIP, image-text joint)** — not measured yet. Would likely fall between SIFT and BERT since they're normalized like BERT but have longer-tail distributions like visual features.
