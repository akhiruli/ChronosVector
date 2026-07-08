/*
 * Engine-level benchmarks — end-to-end query & append latency measured
 * through the extern "C" wall.
 *
 * The kernel bench (bench_kernels) measures just the SIMD math; this bench
 * catches the additional cost of sensor-map lookup, top-N selection,
 * timestamp writeback, and metrics-sink calls.
 *
 * The critical number: BM_QueryNearestN_100k_dim128 must be < 1 ms to meet
 * the definition-of-done.
 */
#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "chronosv/chronos_vector.h"
#include "chronosv/types.h"

namespace {

struct EngineFx {
    chronosv_engine_t* h = nullptr;
    std::uint32_t dim = 0;
    std::string sensor = "s";

    EngineFx(std::uint32_t d, std::uint64_t ring_cap = 0) : dim(d) {
        chronosv_config_t cfg{};
        cfg.abi_version = CHRONOSV_ABI_VERSION;
        cfg.dim = d;
        cfg.ring_capacity = ring_cap;
        // Very large window so background eviction never touches our data
        // during the benchmark run.
        cfg.window_duration_ms = 24LL * 3600 * 1000;
        cfg.eviction_interval_ms = 3600 * 1000;
        chronosv_error_t err = CHRONOSV_OK;
        h = chronosv_create(&cfg, &err);
        if (!h) std::abort();
    }
    ~EngineFx() { chronosv_destroy(h); }

    void FillRandom(std::size_t count, unsigned seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> v(dim);
        for (std::size_t i = 0; i < count; ++i) {
            for (auto& x : v) x = dist(rng);
            chronosv_append(h, sensor.c_str(),
                            static_cast<std::int64_t>(i),
                            v.data(), dim, nullptr);
        }
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// End-to-end query (this is the number Phase 1 lives or dies on)
// ---------------------------------------------------------------------------

static void BM_QueryNearestN(benchmark::State& state) {
    const auto count = static_cast<std::size_t>(state.range(0));
    const auto dim   = static_cast<std::uint32_t>(state.range(1));
    // Round ring capacity up to next power of two >= count.
    std::uint64_t cap = 64;
    while (cap < count) cap <<= 1;

    EngineFx fx(dim, cap);
    fx.FillRandom(count);

    // Query vector — reused across iterations to keep the bench focused on
    // kernel + top-N + engine overhead, not on RNG.
    std::vector<float> q(dim);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : q) x = dist(rng);

    // Result buffers (top-10 is the typical ask).
    constexpr int kTopN = 10;
    std::array<std::int64_t, kTopN> ts{};
    std::array<float, kTopN>        sc{};
    int result_count = 0;

    for (auto _ : state) {
        auto err = chronosv_query_nearest_n(fx.h, fx.sensor.c_str(),
                                            q.data(), dim, kTopN,
                                            ts.data(), sc.data(), &result_count);
        benchmark::DoNotOptimize(err);
        benchmark::DoNotOptimize(sc);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * count);
    state.counters["count"] = static_cast<double>(count);
    state.counters["dim"]   = static_cast<double>(dim);
}
BENCHMARK(BM_QueryNearestN)
    ->Args({1000,   128})
    ->Args({10000,  128})
    ->Args({60000,  128})   // 10 min × 100 Hz reference workload
    ->Args({100000, 128})   // Phase 1 definition-of-done target
    ->Args({10000,  512})
    ->Args({60000,  512});

// ---------------------------------------------------------------------------
// End-to-end append (kernel bench doesn't include the sensor-map fast path,
// heterogeneous lookup, metrics sink hop, or extern "C" translation).
// ---------------------------------------------------------------------------

static void BM_AppendEngine(benchmark::State& state) {
    const auto dim = static_cast<std::uint32_t>(state.range(0));
    EngineFx fx(dim, 1u << 20);
    std::vector<float> v(dim, 0.5f);
    std::int64_t ts = 0;
    for (auto _ : state) {
        chronosv_append(fx.h, fx.sensor.c_str(), ts++, v.data(), dim, nullptr);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());
    state.counters["dim"] = static_cast<double>(dim);
}
BENCHMARK(BM_AppendEngine)->Arg(64)->Arg(128)->Arg(512)->Arg(1024);

// ---------------------------------------------------------------------------
// Per-call APPEND latency distribution (P50 / P99 / P999 / max)
//
// BM_AppendEngine above reports the *average* of many iterations per rep.
// That number smooths per-call jitter into invisibility, which hides the
// metric users actually care about — worst-case latency of a single
// chronosv_append() call — behind mean throughput.
//
// This bench times each call individually, holds the samples, and reports
// percentiles as bench counters. Use it to answer: "does append at
// dim=1024 have a P99 tail from cache-miss stalls on the ring slot?"
//
// Sampling overhead: two steady_clock::now() calls per iteration add
// ~30-50 ns on ARM. At dim=1024 (~2.5 µs mean) that's ~2% distortion —
// acceptable. At dim=64 (~0.04 µs) the overhead swamps the signal, so
// only run this for dims where per-call cost >> clock overhead.
// ---------------------------------------------------------------------------

static void BM_AppendLatencyDist(benchmark::State& state) {
    const auto dim = static_cast<std::uint32_t>(state.range(0));
    EngineFx fx(dim, 1u << 20);
    std::vector<float> v(dim, 0.5f);
    std::int64_t ts = 0;

    // Pre-reserve enough for a full 1s run at ~1 µs per call. Sizing high
    // avoids reallocation mid-loop; excess capacity is freed on scope exit.
    std::vector<std::int64_t> samples_ns;
    samples_ns.reserve(2'000'000);

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        chronosv_append(fx.h, fx.sensor.c_str(), ts++, v.data(), dim, nullptr);
        auto t1 = std::chrono::steady_clock::now();
        benchmark::ClobberMemory();
        samples_ns.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    state.SetItemsProcessed(state.iterations());
    state.counters["dim"] = static_cast<double>(dim);

    // Percentiles via nth_element (O(n)). Sort would work but we only need
    // a few points, so partial partitioning is cheaper.
    auto pct = [&samples_ns](double q) -> double {
        if (samples_ns.empty()) return 0.0;
        auto idx = static_cast<std::size_t>(
            (samples_ns.size() - 1) * q);
        std::nth_element(samples_ns.begin(),
                         samples_ns.begin() + idx,
                         samples_ns.end());
        return static_cast<double>(samples_ns[idx]);
    };
    // Order matters — nth_element mutates. Compute high-to-low so each pass
    // narrows the search space (nth_element leaves elements above idx
    // unsorted but with values > pivot, which is fine for later lower
    // quantiles that also live in the [0, idx) prefix).
    double max_ns = *std::max_element(samples_ns.begin(), samples_ns.end());
    double p999   = pct(0.999);
    double p99    = pct(0.99);
    double p95    = pct(0.95);
    double p50    = pct(0.50);

    state.counters["p50_ns"]  = p50;
    state.counters["p95_ns"]  = p95;
    state.counters["p99_ns"]  = p99;
    state.counters["p999_ns"] = p999;
    state.counters["max_ns"]  = max_ns;
    // Tail-to-median ratio is the "is there a cache-miss story here?"
    // signal. Ratio < 2 = smooth. Ratio > 3 = real latency tail worth
    // investigating (candidate fixes: __builtin_prefetch on next ring slot,
    // slot-write alignment).
    state.counters["p99_over_p50"]  = (p50 > 0) ? p99 / p50 : 0.0;
    state.counters["p999_over_p50"] = (p50 > 0) ? p999 / p50 : 0.0;
}
BENCHMARK(BM_AppendLatencyDist)->Arg(128)->Arg(512)->Arg(1024);

BENCHMARK_MAIN();
