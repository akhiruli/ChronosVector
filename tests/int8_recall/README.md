# INT8 recall validation

Measures how much accuracy you lose by using `CHRONOSV_DTYPE_INT8` storage instead of `CHRONOSV_DTYPE_FLOAT32`, on real (not synthetic) embedding data. Not a pass/fail unit test — a manual harness for engineering-quality validation before enabling INT8 in production.

## Why this exists

Our internal INT8 unit tests use synthetic uniform-random vectors, which quantize artificially well. Real embeddings — BERT, CLIP, sensor autoencoder outputs — have distinct statistical properties (unit-normalization, sparsity, outlier dimensions) that affect quantization differently. This harness lets you measure the actual recall drop on data you care about.

See [`docs/INT8.md`](../../docs/INT8.md) for interpretation and shipping recommendations.

## Reference numbers (measured on Apple Silicon)

| Dataset | Dim | Corpus | Metric | FP32 Recall@10 | INT8 Recall@10 | Drop |
|---|---|---|---|---|---|---|
| SIFT-1M (visual descriptors) | 128 | 1,000,000 | euclidean | 0.9992 | 0.9683 | -3.09 pp |
| BERT/MiniLM (20-newsgroups sentences) | 384 | 50,000 | cosine | 0.9982 | 0.9915 | **-0.66 pp** |

**Takeaway:** for text/document embeddings (RAG, semantic search, BERT-family models), INT8 loses <1% recall — essentially free. For raw visual descriptors (SIFT, ORB), INT8 loses ~3% — still usable, but rerank the top-K with FP32 originals if final ranking matters.

## Build

The harness is opt-in. Enable both flags together:

```sh
cmake -S . -B build-int8 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCHRONOSV_ENABLE_INT8=ON \
      -DCHRONOSV_BUILD_INT8_RECALL=ON
cmake --build build-int8 --target test_int8_recall -j
```

Produces `build-int8/tests/int8_recall/test_int8_recall`.

## Prepare a dataset

### Option A — SIFT-1M (fast, no ML dependencies)

```sh
./tests/int8_recall/prepare_sift1m.sh /tmp/chronosv-sift1m
```

Downloads ~500 MB from ann-benchmarks.com, creates a scoped Python venv with `numpy` + `h5py`, converts the HDF5 to the harness's `.fbin` / `.ibin` format. Takes 2–5 min depending on network.

### Option B — BERT sentence embeddings (~30 min, real ML stack)

```sh
python3 tests/int8_recall/prepare_bert.py /tmp/chronosv-bert
```

Downloads 20 Newsgroups (~14 MB via sklearn), fetches `all-MiniLM-L6-v2` from HuggingFace on first run (~90 MB), embeds 55k sentences, computes cosine top-100 ground truth. Total ~30 min on M-series (mostly embedding).

Requirements: `pip install numpy sentence-transformers scikit-learn`. Note: on corporate networks with MITM SSL inspection the model download disables SSL verification for that request — see the script header.

### Option C — Your own embeddings

Any dataset that produces three files with this format:

- `train.fbin` — `uint32 count | uint32 dim | count*dim float32`
- `test.fbin` — same layout, 100–10,000 queries typical
- `gt.ibin` — `uint32 count | uint32 dim | count*dim int32` where `gt[q]` is the ground-truth top-K neighbor indices (into `train`) for query `q`, sorted best-first. `dim` here is K.

Ground truth must be computed with the same metric you'll pass to the harness.

## Run

```sh
# SIFT (Euclidean metric)
./build-int8/tests/int8_recall/test_int8_recall /tmp/chronosv-sift1m euclidean

# BERT (cosine metric)
./build-int8/tests/int8_recall/test_int8_recall /tmp/chronosv-bert cosine

# Your own dataset
./build-int8/tests/int8_recall/test_int8_recall /path/to/dir cosine
```

## Reading the output

The harness prints a summary block:

```
=========== RESULTS ===========
  FP32  Recall@10  = 0.9982   mean latency 2202.4 us
  INT8  Recall@10  = 0.9915   mean latency 491.8 us
  Recall drop: 0.67 percentage points (0.67% relative)

  Hot memory:
    FP32: 96.8 MiB
    INT8: 25.0 MiB  (3.9x smaller)
===============================
```

Interpretation rules of thumb:

| Recall@10 | Verdict |
|---|---|
| ≥ 0.99 | INT8 is essentially free — enable it |
| 0.95 – 0.99 | Great tradeoff for 4× memory savings; safe for retrieval, may want FP32 rerank for final ranking |
| 0.85 – 0.95 | Use INT8 as coarse filter, always rerank top-100 with FP32 originals |
| < 0.85 | Your embeddings don't quantize well. Stay on FP32, or use INT8 only for coarse pre-filter |

If your recall on your own embeddings is materially worse than the SIFT / BERT reference numbers above, [open an issue](https://github.com/) with the number and a description of the embedding model — that's exactly the data point that would help us improve the quantization scheme.

## What the harness does NOT test

- **Streaming / sliding-window behavior.** All queries run against a static corpus. The quantization decision is independent of the eviction / hot-tier behavior — those are exercised by the soak test.
- **Ingest throughput impact.** The harness prints ingest rates but doesn't stress-test them. INT8 ingest is ~2.5× slower than FP32 because of the per-vector quantization computation. If your workload is ingest-bound rather than query-bound, weigh accordingly.
- **RocksDB persistence.** In-memory only; no `cold_path` configured.
