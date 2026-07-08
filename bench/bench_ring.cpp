/*
 * Ring-buffer microbenchmarks.
 *
 * Focus:
 *   1. Single-thread Append latency vs. dim  — is the P99 < 1 ms claim
 *      realistic without any allocations on hot path?
 *   2. Sustained Append throughput            — vectors/sec on one core.
 *   3. SPSC concurrent throughput             — with a passive reader.
 *
 * Numbers are only meaningful in Release. Debug will be 10-100x slower.
 * Suggested invocation:
 *   cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release -DCHRONOSV_BUILD_BENCH=ON
 *   cmake --build build-rel -j
 *   ./build-rel/bench/bench_ring --benchmark_min_time=0.5s
 */
#include <benchmark/benchmark.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "chronosv/ring_buffer.h"
#include "chronosv/types.h"

using chronosv::internal::SensorRing;

// -- Helper: preallocated vector + fake precomputed norm --------------------

namespace {

struct AppendFixture {
    std::unique_ptr<SensorRing> ring;
    std::vector<float>          vec;

    AppendFixture(std::uint64_t cap, std::uint32_t dim) {
        ring = std::make_unique<SensorRing>(
            cap, dim, /*payload_size=*/0, CHRONOSV_DTYPE_FLOAT32);
        vec.assign(dim, 1.0f);
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// 1. Append latency vs. dim
// ---------------------------------------------------------------------------

static void BM_Append_Latency(benchmark::State& state) {
    const auto dim = static_cast<std::uint32_t>(state.range(0));
    AppendFixture fx(/*cap=*/1u << 16, dim);

    std::int64_t ts = 0;
    for (auto _ : state) {
        fx.ring->Append(ts++, fx.vec.data(), /*norm=*/1.0f, /*payload=*/nullptr);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * dim * sizeof(float));
    state.counters["dim"] = static_cast<double>(dim);
}
BENCHMARK(BM_Append_Latency)
    ->Arg(16)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024);

// ---------------------------------------------------------------------------
// 2. Sustained Append throughput at a fixed dim
// ---------------------------------------------------------------------------

static void BM_Append_Throughput(benchmark::State& state) {
    constexpr std::uint32_t kDim = 128;
    AppendFixture fx(/*cap=*/1u << 20, kDim);

    std::int64_t ts = 0;
    for (auto _ : state) {
        // Batch 100 appends per benchmark iteration to reduce loop overhead
        // and better reflect the hot inner loop.
        for (int i = 0; i < 100; ++i) {
            fx.ring->Append(ts++, fx.vec.data(), 1.0f, nullptr);
        }
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 100);
    state.SetBytesProcessed(state.iterations() * 100 * kDim * sizeof(float));
}
BENCHMARK(BM_Append_Throughput);

// ---------------------------------------------------------------------------
// 3. SPSC producer + passive reader — measures producer throughput WHILE
//    a concurrent reader hammers Snapshot() on another core. Approximates
//    the real workload of a query thread running during ingest.
// ---------------------------------------------------------------------------

static void BM_Append_WithReader(benchmark::State& state) {
    constexpr std::uint32_t kDim = 128;
    constexpr std::uint64_t kCap = 1u << 16;
    AppendFixture fx(kCap, kDim);

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> reads{0};

    std::thread reader([&] {
        while (!stop.load(std::memory_order_acquire)) {
            auto [b, e] = fx.ring->Snapshot();
            if (e > 0) {
                (void) fx.ring->timestamp_at(e - 1);
                (void) fx.ring->vector_at(e - 1);
                reads.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    std::int64_t ts = 0;
    for (auto _ : state) {
        for (int i = 0; i < 100; ++i) {
            fx.ring->Append(ts++, fx.vec.data(), 1.0f, nullptr);
        }
        benchmark::ClobberMemory();
    }
    stop.store(true, std::memory_order_release);
    reader.join();

    state.SetItemsProcessed(state.iterations() * 100);
    state.counters["reader_snapshots"] =
        benchmark::Counter(static_cast<double>(reads.load()),
                           benchmark::Counter::kAvgIterations);
    state.counters["overwrite_events"] =
        static_cast<double>(fx.ring->overwrite_events());
}
BENCHMARK(BM_Append_WithReader);

// ---------------------------------------------------------------------------
// 4. Snapshot() latency — how cheap is the query-path prelude?
// ---------------------------------------------------------------------------

static void BM_Snapshot(benchmark::State& state) {
    AppendFixture fx(/*cap=*/1u << 12, /*dim=*/128);
    // Prefill with a modest number of entries so Snapshot returns non-trivial.
    for (int i = 0; i < 1000; ++i) {
        fx.ring->Append(i, fx.vec.data(), 1.0f, nullptr);
    }
    for (auto _ : state) {
        auto [b, e] = fx.ring->Snapshot();
        benchmark::DoNotOptimize(b);
        benchmark::DoNotOptimize(e);
    }
}
BENCHMARK(BM_Snapshot);

BENCHMARK_MAIN();
