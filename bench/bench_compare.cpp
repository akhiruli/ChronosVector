/*
 * Competitor comparison bench — ChronosVector vs hnswlib on the reference
 * workload (top-10 kNN over N vectors at dim=128).
 *
 * READ THIS BEFORE INTERPRETING THE NUMBERS
 * -----------------------------------------
 * These libraries make DIFFERENT tradeoffs, not just faster-vs-slower:
 *
 *   ChronosVector:  BRUTE-FORCE over a bounded sliding window.
 *                   Query is O(N * D) but N is bounded (design intent).
 *                   Results are EXACT.
 *                   Zero index build cost — just append and query.
 *
 *   hnswlib (HNSW): APPROXIMATE nearest neighbor via graph index.
 *                   Query is O(log N) but with build cost O(N log N).
 *                   Results are APPROXIMATE (tunable via ef parameter).
 *                   Assumes the corpus is roughly stable.
 *
 * Neither is "better" — they solve different problems. This bench shows the
 * crossover: at what N does HNSW's log-N query beat our linear scan? For
 * ChronosVector's target workload (10-min × 100 Hz = 60k) we expect brute-
 * force to be competitive because:
 *   (a) the constant factor on Eigen-SIMD dot products is tiny
 *   (b) HNSW's graph traversal has bad cache locality
 *   (c) we don't pay the O(N log N) build cost on every eviction cycle
 *
 * At corpus sizes hnswlib is actually optimized for (millions of vectors),
 * ChronosVector is deliberately out of scope — use hnswlib.
 *
 * Runs in Release only. Not part of ctest. Manual invocation.
 */
#include <benchmark/benchmark.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "chronosv/chronos_vector.h"

#include "hnswlib/hnswlib.h"

namespace {

/* Shared random corpus + query — regenerated per (count, dim) combination,
 * cached in a static so the two library benches see identical data. */
struct Corpus {
    std::vector<float> data;
    std::vector<float> query;
    std::size_t count;
    std::size_t dim;
};

Corpus& get_corpus(std::size_t count, std::size_t dim) {
    static std::vector<std::unique_ptr<Corpus>> cache;
    for (auto& c : cache) {
        if (c->count == count && c->dim == dim) return *c;
    }
    auto c = std::make_unique<Corpus>();
    c->count = count;
    c->dim = dim;
    c->data.resize(count * dim);
    c->query.resize(dim);
    std::mt19937 rng(0xC0DE5EED);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : c->data)  x = dist(rng);
    for (auto& x : c->query) x = dist(rng);
    cache.push_back(std::move(c));
    return *cache.back();
}

/* Preload ChronosVector with the corpus. Uses cold_path=null so we use
 * NullStorageBackend — no persistence overhead, matches hnswlib's
 * in-memory-only semantics for fair comparison. Also sizes ring_capacity
 * to hold the whole corpus so no eviction happens during the bench. */
struct ChronosFixture {
    chronosv_engine_t* eng = nullptr;
    explicit ChronosFixture(const Corpus& c) {
        chronosv_config_t cfg{};
        cfg.abi_version = CHRONOSV_ABI_VERSION;
        cfg.dim = static_cast<std::uint32_t>(c.dim);
        /* Round ring_capacity up to next power of two >= count. */
        std::uint64_t cap = 64;
        while (cap < c.count) cap <<= 1;
        cfg.ring_capacity = cap;
        /* Huge window so no eviction fires during preload or bench. */
        cfg.window_duration_ms = 24LL * 3600 * 1000;
        cfg.eviction_interval_ms = 3600 * 1000;
        chronosv_error_t err = CHRONOSV_OK;
        eng = chronosv_create(&cfg, &err);
        if (!eng) std::abort();
        for (std::size_t i = 0; i < c.count; ++i) {
            chronosv_append(eng, "s", static_cast<std::int64_t>(i),
                            c.data.data() + i * c.dim,
                            c.dim, nullptr);
        }
    }
    ~ChronosFixture() { chronosv_destroy(eng); }
};

/* hnswlib tuning parameters. These are the hnswlib README defaults for
 * "reasonable recall on random uniform data" — NOT adversarially tuned
 * in either direction. They're emitted as bench counters (see
 * BM_Hnswlib_Query below) so anyone reading the numbers can see what
 * config produced them without opening the source.
 *
 * If you want to argue hnswlib should look faster, raise kHnswEfConstruction
 * (better graph, slower build) or raise kHnswEfQuery (better recall, more
 * per-query work). If you want it to look faster still at cost of recall,
 * lower kHnswEfQuery. Neither of those changes ChronosVector's story:
 * we're the exact / bounded-window / no-index-build alternative to HNSW,
 * not "just faster HNSW". */
constexpr std::size_t kHnswM              = 16;
constexpr std::size_t kHnswEfConstruction = 200;
constexpr std::size_t kHnswEfQuery        = 50;

/* Preload hnswlib with the corpus + build the graph. Build time is NOT
 * timed — the bench measures query cost only, which is what matters at
 * steady state. If you care about build cost, that's a separate story
 * (and one where ChronosVector wins trivially: 0 index build). */
struct HnswFixture {
    std::unique_ptr<hnswlib::L2Space>                    space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>>     alg;
    explicit HnswFixture(const Corpus& c) {
        space = std::make_unique<hnswlib::L2Space>(c.dim);
        alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space.get(), c.count, kHnswM, kHnswEfConstruction);
        for (std::size_t i = 0; i < c.count; ++i) {
            alg->addPoint(c.data.data() + i * c.dim,
                          static_cast<hnswlib::labeltype>(i));
        }
        alg->setEf(kHnswEfQuery);
    }
};

}  // namespace

/* -------------------------------------------------------------------------- *
 * ChronosVector top-10 query
 * -------------------------------------------------------------------------- */

static void BM_ChronosVector_Query(benchmark::State& state) {
    const auto count = static_cast<std::size_t>(state.range(0));
    const auto dim   = static_cast<std::size_t>(state.range(1));
    const auto& c = get_corpus(count, dim);
    ChronosFixture fx(c);

    constexpr int kTopN = 10;
    std::vector<std::int64_t> out_ts(kTopN);
    std::vector<float>        out_sc(kTopN);
    int result_count = 0;

    for (auto _ : state) {
        auto err = chronosv_query_nearest_n(fx.eng, "s",
                                            c.query.data(), dim, kTopN,
                                            out_ts.data(), out_sc.data(), &result_count);
        benchmark::DoNotOptimize(err);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * count);
    state.counters["count"] = static_cast<double>(count);
    state.counters["dim"]   = static_cast<double>(dim);
    state.counters["ns_per_pair"] =
        benchmark::Counter(static_cast<double>(count),
                           benchmark::Counter::kIsIterationInvariantRate
                           | benchmark::Counter::kInvert);
}
BENCHMARK(BM_ChronosVector_Query)
    ->Args({10000,  128})
    ->Args({60000,  128})   // 10-min × 100 Hz reference workload
    ->Args({100000, 128});

/* -------------------------------------------------------------------------- *
 * hnswlib top-10 query
 * -------------------------------------------------------------------------- */

static void BM_Hnswlib_Query(benchmark::State& state) {
    const auto count = static_cast<std::size_t>(state.range(0));
    const auto dim   = static_cast<std::size_t>(state.range(1));
    const auto& c = get_corpus(count, dim);
    HnswFixture fx(c);

    constexpr int kTopN = 10;
    for (auto _ : state) {
        auto result = fx.alg->searchKnn(c.query.data(), kTopN);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * count);
    state.counters["count"] = static_cast<double>(count);
    state.counters["dim"]   = static_cast<double>(dim);
    /* Emit tuning params so the numbers are self-documenting. */
    state.counters["hnsw_M"]    = static_cast<double>(kHnswM);
    state.counters["hnsw_efC"]  = static_cast<double>(kHnswEfConstruction);
    state.counters["hnsw_efQ"]  = static_cast<double>(kHnswEfQuery);
    /* ns_per_pair here is misleading — hnswlib does NOT visit every pair,
     * that's the point of the graph. Report it anyway for column alignment
     * with the brute-force numbers. */
    state.counters["ns_per_pair"] =
        benchmark::Counter(static_cast<double>(count),
                           benchmark::Counter::kIsIterationInvariantRate
                           | benchmark::Counter::kInvert);
}
BENCHMARK(BM_Hnswlib_Query)
    ->Args({10000,  128})
    ->Args({60000,  128})
    ->Args({100000, 128});

BENCHMARK_MAIN();
