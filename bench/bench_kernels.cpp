/*
 * Distance-kernel microbenchmarks.
 *
 * Sweeps (count, dim) matrix to understand where Eigen's auto-vectorization
 * pays off vs. loop / cache effects. These are the numbers we'll compare
 * against USearch / hnswlib streaming kernels in Phase 3.
 *
 * Run in Release mode:
 *   cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release -DCHRONOSV_BUILD_BENCH=ON
 *   cmake --build build-rel -j
 *   ./build-rel/bench/bench_kernels --benchmark_min_time=0.5s
 *
 * A "typical" ChronosVector workload: dim=128, count in the 1k..100k range
 * (window of ~10 minutes at 100 Hz = 60,000 entries). Look at that band.
 */
#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>
#include <vector>

#include "kernels.h"

using chronosv::internal::CosineF32Chunk;
using chronosv::internal::EuclideanSqF32Chunk;
using chronosv::internal::EuclideanSqF32Chunk_Direct;
using chronosv::internal::L2NormF32;

namespace {

// Deterministic random data shared across benches within one process.
struct RandomVectors {
    std::vector<float> vecs;
    std::vector<float> norms;
    std::vector<float> q;
    float              qn;
    std::size_t        count;
    std::size_t        dim;

    RandomVectors(std::size_t c, std::size_t d) : count(c), dim(d) {
        std::mt19937 rng(0xC5057E5);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        vecs.resize(count * dim);
        for (auto& x : vecs) x = dist(rng);
        q.resize(dim);
        for (auto& x : q) x = dist(rng);
        norms.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            norms[i] = L2NormF32(&vecs[i * dim], dim);
        }
        qn = L2NormF32(q.data(), dim);
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// L2 norm (per-vector cost — what we pay on the Append path)
// ---------------------------------------------------------------------------

static void BM_L2Norm(benchmark::State& state) {
    const auto dim = static_cast<std::size_t>(state.range(0));
    std::vector<float> v(dim, 0.5f);
    for (auto _ : state) {
        benchmark::DoNotOptimize(L2NormF32(v.data(), dim));
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * dim * sizeof(float));
    state.counters["dim"] = static_cast<double>(dim);
}
BENCHMARK(BM_L2Norm)
    ->Arg(16)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048);

// ---------------------------------------------------------------------------
// Cosine — the default metric — across a (count, dim) matrix
// ---------------------------------------------------------------------------

static void BM_CosineF32Chunk(benchmark::State& state) {
    const auto count = static_cast<std::size_t>(state.range(0));
    const auto dim   = static_cast<std::size_t>(state.range(1));
    RandomVectors rv(count, dim);
    std::vector<float> out(count);

    for (auto _ : state) {
        CosineF32Chunk(rv.vecs.data(), rv.norms.data(),
                       count, dim, rv.q.data(), rv.qn, out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * count);
    state.SetBytesProcessed(state.iterations() * count * dim * sizeof(float));
    state.counters["count"] = static_cast<double>(count);
    state.counters["dim"]   = static_cast<double>(dim);
    state.counters["ns_per_pair"] =
        benchmark::Counter(static_cast<double>(count),
                           benchmark::Counter::kIsIterationInvariantRate
                           | benchmark::Counter::kInvert);
}
BENCHMARK(BM_CosineF32Chunk)
    // (count, dim) — spans the typical workload band and edges.
    ->Args({100,    64})
    ->Args({100,   128})
    ->Args({100,   512})
    ->Args({1000,   64})
    ->Args({1000,  128})
    ->Args({1000,  512})
    ->Args({10000,  64})
    ->Args({10000, 128})
    ->Args({10000, 512})
    ->Args({60000, 128})   // 10 min * 100 Hz window
    ->Args({100000,128});

// ---------------------------------------------------------------------------
// Euclidean squared — same matrix
// ---------------------------------------------------------------------------

static void BM_EuclideanSqF32Chunk(benchmark::State& state) {
    const auto count = static_cast<std::size_t>(state.range(0));
    const auto dim   = static_cast<std::size_t>(state.range(1));
    RandomVectors rv(count, dim);
    std::vector<float> out(count);

    for (auto _ : state) {
        EuclideanSqF32Chunk(rv.vecs.data(), rv.norms.data(),
                            count, dim, rv.q.data(), rv.qn, out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * count);
    state.SetBytesProcessed(state.iterations() * count * dim * sizeof(float));
    state.counters["count"] = static_cast<double>(count);
    state.counters["dim"]   = static_cast<double>(dim);
}
BENCHMARK(BM_EuclideanSqF32Chunk)
    ->Args({100,    64})
    ->Args({100,   128})
    ->Args({1000,   64})
    ->Args({1000,  128})
    ->Args({10000, 128})
    ->Args({60000, 128})
    ->Args({100000,128});

// Direct reference kernel — kept as a benchmark comparison so we can
// document the speedup and verify the fast path is worth its numerical
// caveats. If this ever gets FASTER than the identity path, something has
// changed and we should reconsider.
static void BM_EuclideanSqF32Chunk_Direct(benchmark::State& state) {
    const auto count = static_cast<std::size_t>(state.range(0));
    const auto dim   = static_cast<std::size_t>(state.range(1));
    RandomVectors rv(count, dim);
    std::vector<float> out(count);

    for (auto _ : state) {
        EuclideanSqF32Chunk_Direct(rv.vecs.data(), count, dim,
                                   rv.q.data(), out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * count);
    state.counters["count"] = static_cast<double>(count);
    state.counters["dim"]   = static_cast<double>(dim);
}
BENCHMARK(BM_EuclideanSqF32Chunk_Direct)
    ->Args({1000, 128})
    ->Args({10000, 128})
    ->Args({60000, 128});

BENCHMARK_MAIN();
