# ChronosVector

An embeddable C++23 **sliding-window vector store** for streaming time-series embeddings. Lock-free ring buffer in RAM, SIMD distance math via Eigen, RocksDB-backed cold tier. Bounded memory, predictable latency, no garbage collection.

- **Status:** v0.1 preview. Core engine, RocksDB persistence + recovery, INT8 quantized storage (validated on SIFT-1M + BERT — see [`docs/INT8.md`](docs/INT8.md)), competitor benchmarks vs hnswlib, CI perf visibility. 140 tests green on macOS ARM64 and Linux ARM64 under Debug / Release / ASan / TSan / UBSan. 6-hour soak validated at 10 kHz sustained ingest.
- **License:** Apache-2.0 — permanent, no BSL/SSPL rug-pull ever (see [design decisions](#license-commitments)).

---

## What it's for

Workloads where vector embeddings **stream in continuously** and only the **recent window** matters:

- On-device video / audio understanding — "have I seen this action in the last 10 minutes?"
- Industrial telemetry with learned signatures — vibration-embedding drift detection at kHz rates
- Edge / robotics where RAM is bounded and you can't afford GC pauses

The core primitive: **first-class sliding-window time-eviction** as a design contract, not a background chore layered on a general-purpose index. RAM stays flat regardless of runtime length. Query latency is predictable.

## What it's not

Not a general-purpose vector database. If you need HNSW / IVF / graph-index k-NN over a fixed 100 M-vector corpus, use [USearch](https://github.com/unum-cloud/usearch), [hnswlib](https://github.com/nmslib/hnswlib), [Qdrant](https://github.com/qdrant/qdrant), or [Milvus](https://github.com/milvus-io/milvus). They're better at that specific job.


## When to reach for ChronosVector

*For a deep-dive by industry (industrial IoT, video anomaly detection, wearables, HVAC, satellite telemetry, robotics, cybersecurity, e-commerce, plus explicit non-fits) see [`docs/USE_CASES.md`](docs/USE_CASES.md).*

Pick ChronosVector when **all** of these apply:

- Your data is a **stream** (append rate > query rate on hot data).
- **Only a recent window** is queried in the hot path; older data can go to disk or vanish.
- You need **bounded RAM** and **predictable per-call latency** — GC pauses, index-rebuild pauses, or unbounded memory growth are unacceptable.
- Your corpus per sensor / partition fits comfortably in a ring of tens of thousands to low millions of vectors.
- You want to **embed** the store in a C / C++ / Rust / Go / Python process, not run a server.

Pick something else when any of these apply:

- You have a **static corpus** of millions or billions of vectors and want log-N k-NN via a graph index → [USearch](https://github.com/unum-cloud/usearch), [hnswlib](https://github.com/nmslib/hnswlib), [Qdrant](https://github.com/qdrant/qdrant), [Milvus](https://github.com/milvus-io/milvus).
- You need **hybrid search** (vectors + payloads + filters) as first-class → Qdrant.
- You need a **managed service** with a control plane → Qdrant Cloud, Pinecone.

## How it compares

| Project | Index | Latency profile | Time-eviction |
|---|---|---|---|
| **ChronosVector** | Brute-force over bounded window | O(N·D), N is bounded; exact | **First-class** (design contract, not layered) |
| **Qdrant Edge** | HNSW | O(log N) approx; graph traversal is cache-unfriendly | Layered as background TTL |
| **USearch** | HNSW | O(log N) approx | Not built-in |
| **hnswlib** | HNSW | O(log N) approx | Not built-in |
| **ObjectBox** | HNSW | O(log N) approx | Not built-in |
| **Milvus Lite** | HNSW / IVF | O(log N) approx | TTL via collection config |

**Compact 15-function C ABI**, semver-stable from v1.0. No index-graph machinery.

### Head-to-head vs hnswlib (top-10 kNN, dim=128, Apple Silicon)

| Corpus size | ChronosVector (brute, exact) | hnswlib (HNSW, approx: M=16, efC=200, efQ=50) |
|---|---|---|
| 10,000 | 111 µs | 94 µs |
| 60,000 | 747 µs | 118 µs |
| 100,000 | 1,281 µs | 110 µs |

hnswlib is faster at every corpus size — that is expected and by design. HNSW is O(log N) approximate; ChronosVector is O(N) exact. Two things this table does NOT show:

1. **Build cost.** hnswlib pays O(N log N) to build the graph before the first query. For a sliding-window workload where evicted entries need to disappear from the index, that build cost is paid **again on every window shift** — a cost ChronosVector doesn't have because there's no index to rebuild.
2. **Recall.** hnswlib returns approximate neighbors tunable via `efQ`; ChronosVector returns the exact top-10 every time. Neither is universally "better" — they solve different problems.

Reproduce with `./bench/bench_compare` (needs `-DCHRONOSV_BUILD_BENCH=ON`).

## Quick start (5 minutes)

Requirements: CMake ≥ 3.25, C++23 compiler (AppleClang 15+, Clang 17+, GCC 13+), RocksDB (`brew install rocksdb` / `apt install librocksdb-dev`).

```sh
git clone --recursive <this-repo>   # --recursive pulls the Eigen submodule
cd ChronosVector
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCHRONOSV_BUILD_EXAMPLES=ON
cmake --build build -j
./build/examples/chronosv_cli
```

At the `>` prompt:

```
> create dim=4
engine created (dim=4)
> append s1 100 1,0,0,0
ok
> append s1 200 0.9,0.1,0,0
ok
> query s1 1,0,0,0 n=2
  ts=100  score=1.000000
  ts=200  score=0.993884
> quit
```

Reading this session:

- **`s1`** is a stream identifier (one sensor, one video feed, one user session — pick your semantics). **`100`** and **`200`** are timestamps in milliseconds, chosen by you.
- **`query ... n=2`** asks for the top-2 stored vectors most similar to `[1, 0, 0, 0]`. Results are sorted best-first.
- **Scores** are cosine similarity (the default metric): `1.000000` means the stored vector matches the query exactly; `0.993884` = `0.9 / √0.82` is the cosine of the angle between `[1, 0, 0, 0]` and `[0.9, 0.1, 0, 0]`.

For a walkthrough that uses the C++ RAII wrapper and shows anomaly detection with a metrics sink:

```sh
./build/examples/anomaly_stream
```

Full build / test / bench guide: [`docs/BUILDING.md`](docs/BUILDING.md).

## The 15 primitives

```c
/* Lifecycle */
chronosv_engine_t* chronosv_create(const chronosv_config_t* cfg, chronosv_error_t* err_out);
chronosv_error_t   chronosv_open(const char* path, chronosv_engine_t** out);
chronosv_error_t   chronosv_close(chronosv_engine_t* engine);
void               chronosv_destroy(chronosv_engine_t* engine);           /* NULL-safe */
chronosv_error_t   chronosv_preallocate_sensor(chronosv_engine_t*, const char* sensor_id);

/* Ingest */
chronosv_error_t chronosv_append(chronosv_engine_t*, const char* sensor_id,
                                 int64_t ts_ms, const float* vec, size_t dim,
                                 const void* payload);
chronosv_error_t chronosv_append_batch(...);

/* Query — hot window only in v0.1 */
chronosv_error_t chronosv_query_nearest_n(...);
chronosv_error_t chronosv_query_range(...);
chronosv_error_t chronosv_detect_anomaly(...);

/* Maintenance */
chronosv_error_t chronosv_maintain_sliding_window(chronosv_engine_t*, int64_t window_ms);
chronosv_error_t chronosv_drop_sensor(chronosv_engine_t*, const char* sensor_id);
chronosv_error_t chronosv_flush(chronosv_engine_t*);

/* Observability */
chronosv_error_t chronosv_list_sensors(...);
chronosv_error_t chronosv_get_stats(...);
```

C++ users get a header-only RAII wrapper via `chronosv/chronos_vector.hpp` — zero runtime cost, `chronosv::Engine`, `chronosv::Result<T>`, `std::span`-based inputs.

## Performance (measured, Apple Silicon M-series, Release)

| Metric | Target | Measured |
|---|---|---|
| Per-call P99 `append` at dim=128 (`BM_AppendLatencyDist`) | < 1 ms | **84 ns** |
| Per-call P99 `append` at dim=512 (post-prefetch fix) | < 1 ms | **250 ns** |
| Per-call P99 `append` at dim=1024 | < 1 ms | **3.5 µs** |
| Mean `query_nearest_n` at 60k × 128 (10-min × 100 Hz reference) | < 1 ms | **~720 µs** ✓ |
| Mean `query_nearest_n` at 100k × 128 (memory-bandwidth-bound at float32) | < 1 ms | **~1.3 ms** — see below |
| Sustained append throughput at dim=128 | ≥ 50k/sec | **~21 M/sec** |
| RSS growth over 6-hour soak vs steady baseline | ≤ 5% | **+2.26%** ✓ (at 10 kHz × 128-dim sustained ingest; see [`soak/README.md`](soak/README.md)) |

At 100k × 128 float32 the query is memory-bandwidth-bound (~1.3 ms end-to-end vs the < 1 ms target); INT8 quantization (`cfg.storage_dtype = CHRONOSV_DTYPE_INT8`, compiled by default) cuts memory ~4× and query latency 2.6–4.5× at 0.7–3 pp recall drop depending on embedding type — see [`docs/INT8.md`](docs/INT8.md) for measured numbers. Full raw JSON in `bench/baselines/`.

**Append tail latency was a specific hardening focus.** Prefetching the next ring slot in `SensorRing::Append` dropped dim=512 P99 from 16× median to 1.6× median; see `bench/bench_engine.cpp::BM_AppendLatencyDist` and `bench/baselines/README.md` for the diagnostic pattern.

## Architecture

Three layers:

1. **Hot** — Lock-free SoA ring buffer per sensor. Fixed capacity allocated once. SPSC-per-sensor producer, multi-consumer query. O(1) append.
2. **Math** — Eigen expression templates → NEON on Apple Silicon, AVX2/AVX-512 on x86.
3. **Cold** — RocksDB background eviction. Blocks flushed asynchronously (or on-demand via `chronosv_flush`). Survives power loss.

## Repo layout

```
include/chronosv/            Public C ABI (chronos_vector.h) + C++ wrapper (.hpp) + metrics_sink.h
src/                         Engine, kernels, block codec, RocksDB backend (internal)
tests/                       140 unit + integration tests, Catch2 (incl. INT8 kernels)
tests/int8_recall/           Opt-in INT8 recall validation harness (SIFT-1M + BERT)
bench/                       bench_ring, bench_kernels, bench_engine, bench_compare (google/benchmark)
bench/baselines/             Committed macOS + Linux JSON baselines
examples/chronosv_cli.cpp    Debug REPL
examples/anomaly_stream.cpp  Walkthrough: streaming anomaly detection + metrics sink
soak/                        Long-running load + flat-RSS check
docs/BUILDING.md             Full build / test / bench / soak guide
docs/INT8.md                 INT8 quantized storage: when to use + measured recall/perf
docs/USE_CASES.md            Industry deep-dives, fit criteria, deployment patterns
docker/                      Linux verification via Docker Desktop
third_party/eigen/           Vendored git submodule, pinned to 3.4.0
```

## Things to know

A few facts about the current release that don't fit anywhere else, in decreasing order of "you might trip over this":

- **`chronosv_query_range` reads the hot window only.** If you ask for a time range that extends before the oldest in-memory entry, you get back the partial hot result plus a `CHRONOSV_WARN_RANGE_TRUNCATED` warning. Older data lives on disk but there's no built-in cold-tier range reader yet — planned post-v0.1.

- **INT8 storage** compiles in by default and is opt-in at runtime via `cfg.storage_dtype = CHRONOSV_DTYPE_INT8`. Measured on real embeddings: **~0.7 pp recall drop** on BERT/MiniLM at 384-dim (essentially free) and ~3.1 pp drop on SIFT-1M at 128-dim (safe with FP32 rerank). ~4× memory reduction and ~2.5–4.5× query speedup depending on corpus size. Full measured numbers, when-to-use guidance, and a validation harness for your own embeddings in [`docs/INT8.md`](docs/INT8.md).

- **Reading memory numbers:** `phys_footprint` (macOS) / `RssAnon` (Linux) oscillates in a bounded band around a steady baseline — the memtable fill/flush cycle is inherent to RocksDB. The soak PASS gate at +5% growth vs baseline (measured +2.26% in the shipping build) reflects that reality; expect ~30-50 MiB swings within the band on any typical workload.

- **Overwrites are a first-class observable.** The ring buffer will overwrite its oldest entry when a producer outpaces eviction — design-permitted behavior, detectable via `chronosv_stats_t::total_overwrite_events` or the `on_overwrite` metrics-sink callback. Validated at **10 kHz sustained ingest for 6 hours**; see [`soak/README.md`](soak/README.md) for measured numbers.


## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). TL;DR: build under all four sanitizer configs locally, then verify against Linux via `./docker/verify_linux.sh`; PR should be TSan + ASan clean.

## License commitments

- **Apache-2.0 for the core, permanent.** No BSL, no SSPL, no Commons Clause pivot later.
- If a commercial layer ever materializes it lives in a **separate repo** with a separate license. The OSS core never gains feature flags that hide functionality behind a paid tier.

## Non-goals (worth stating)

- Distributed clustering, replication, consensus — use Qdrant or Milvus.
- SQL parser, transactions, query planner.
- Multi-tenancy beyond sensor-ID isolation.
- Cloud-hosted SaaS control plane.
- LLM / agent memory abstractions — the primitive is general; an "agent memory" library on top would live in a separate repo.

If a feature request smells like any of these, please open an issue rather than a PR — likely out of scope.
