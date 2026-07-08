/**
 * @file anomaly_stream.cpp
 * @brief Walkthrough: streaming anomaly detection with a metrics sink.
 *
 * What this demonstrates:
 *   1. Setting up an Engine with a chronosv_metrics_sink_t wired to stdout.
 *   2. Feeding a synthetic sensor stream that alternates between "normal"
 *      readings (jitter around a fixed centroid) and injected anomalies
 *      (large displacement).
 *   3. Calling chronosv_detect_anomaly on each sample and printing the
 *      classification alongside sink events.
 *
 * This is the pattern the target audience (embedded / IoT / real-time
 * monitoring) will actually use — an append + anomaly-check loop, with
 * observability piped somewhere useful. Kept minimal on purpose; production
 * callers will replace the printf sink with their metrics system of choice
 * (Prometheus exporter, OpenTelemetry span, whatever).
 *
 * Build (from repo root):
 *     cmake -S . -B build -DCHRONOSV_BUILD_EXAMPLES=ON
 *     cmake --build build --target anomaly_stream
 *     ./build/examples/anomaly_stream
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "chronosv/chronos_vector.hpp"

namespace {

constexpr std::size_t kDim              = 32;
constexpr int         kWarmupSamples    = 200;   // populate the window
constexpr int         kStreamSamples    = 100;   // detection phase
constexpr float       kAnomalyThreshold = 0.35f; // cosine distance
constexpr int         kAnomalyEveryN    = 15;    // inject one anomaly per N normal samples

// ---- Metrics sink callbacks (C linkage, noexcept, non-blocking) ----------
//
// These fire from engine-internal threads. Keep them fast — printf is fine
// for a demo; a production sink would enqueue to a lock-free ring and drain
// from a worker.

extern "C" void OnAppend(void* /*ud*/, const char* sensor_id, std::int64_t latency_ns) {
    // Silence: on_append fires on every ingest and would flood output.
    // Uncomment to see per-append latency.
    (void)sensor_id; (void)latency_ns;
}

extern "C" void OnAnomaly(void* /*ud*/, const char* sensor_id, float distance) {
    std::printf("  [sink] on_anomaly_detected: sensor=%s distance=%.4f\n",
                sensor_id, static_cast<double>(distance));
}

extern "C" void OnOverwrite(void* /*ud*/, const char* sensor_id,
                            std::uint64_t entries_lost) {
    std::printf("  [sink] on_overwrite: sensor=%s lost=%llu (producer outpaced eviction)\n",
                sensor_id, static_cast<unsigned long long>(entries_lost));
}

extern "C" void OnEviction(void* /*ud*/, const char* sensor_id,
                           std::int64_t evicted_count, std::int64_t block_bytes) {
    std::printf("  [sink] on_eviction: sensor=%s count=%lld bytes=%lld\n",
                sensor_id,
                static_cast<long long>(evicted_count),
                static_cast<long long>(block_bytes));
}

// ---- Synthetic stream ----------------------------------------------------

// Fill `out` with a "normal" reading — small gaussian jitter around a fixed
// centroid. Represents a well-behaved sensor at steady state.
void FillNormal(std::span<float> out, std::mt19937& rng) {
    std::normal_distribution<float> jitter(0.0f, 0.05f);
    // Centroid: a fixed direction. Real workloads would derive this from
    // whatever "normal" means in the domain (nominal joint angle, expected
    // spectral shape, etc.).
    for (std::size_t i = 0; i < out.size(); ++i) {
        const float centroid = (i < out.size() / 2) ? 0.8f : 0.2f;
        out[i] = centroid + jitter(rng);
    }
}

// Fill `out` with an "anomalous" reading — a large displacement in a
// different direction. Represents a sensor fault, spike, or genuine outlier.
void FillAnomaly(std::span<float> out, std::mt19937& rng) {
    std::normal_distribution<float> jitter(0.0f, 0.1f);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const float centroid = (i < out.size() / 2) ? -0.6f : 0.9f;
        out[i] = centroid + jitter(rng);
    }
}

}  // namespace

int main() {
    // ---- Engine setup ----------------------------------------------------
    ::chronosv_metrics_sink_t sink{};
    sink.on_append           = &OnAppend;
    sink.on_anomaly_detected = &OnAnomaly;
    sink.on_overwrite        = &OnOverwrite;
    sink.on_eviction         = &OnEviction;

    ::chronosv_config_t cfg{};
    cfg.abi_version          = CHRONOSV_ABI_VERSION;
    cfg.dim                  = kDim;
    cfg.ring_capacity        = 512;
    cfg.window_duration_ms   = 60 * 1000;  // 60s hot window
    cfg.eviction_interval_ms = 30 * 1000;
    cfg.distance_metric      = CHRONOSV_METRIC_COSINE;
    cfg.metrics_sink         = &sink;
    // cfg.cold_path left NULL: in-memory only, no RocksDB required.

    auto engine_result = chronosv::Engine::Create(cfg);
    if (!engine_result) {
        std::fprintf(stderr, "chronosv::Engine::Create failed: %.*s\n",
                     static_cast<int>(chronosv::error_string(engine_result.error()).size()),
                     chronosv::error_string(engine_result.error()).data());
        return EXIT_FAILURE;
    }
    auto& engine = *engine_result;

    const std::string sensor = "gripper_torque";
    std::mt19937 rng(0xC0FFEE);
    std::vector<float> v(kDim);

    // ---- Phase 1: warm up the window with normal readings ---------------
    std::printf("Warmup: appending %d normal samples to '%s'...\n",
                kWarmupSamples, sensor.c_str());
    auto now_ms = []() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    std::int64_t ts = now_ms();
    for (int i = 0; i < kWarmupSamples; ++i) {
        FillNormal(v, rng);
        if (auto e = engine.Append(sensor, ts++, v); chronosv::is_error(e)) {
            std::fprintf(stderr, "Append failed: %.*s\n",
                         static_cast<int>(chronosv::error_string(e).size()),
                         chronosv::error_string(e).data());
            return EXIT_FAILURE;
        }
    }

    // ---- Phase 2: stream + detect ---------------------------------------
    std::printf("\nStream: %d samples, injecting an anomaly every %d.\n",
                kStreamSamples, kAnomalyEveryN);
    std::printf("        (threshold=%.2f cosine distance)\n\n", static_cast<double>(kAnomalyThreshold));

    int flagged = 0, injected = 0;
    for (int i = 0; i < kStreamSamples; ++i) {
        const bool is_injected_anomaly = (i > 0 && i % kAnomalyEveryN == 0);
        if (is_injected_anomaly) {
            FillAnomaly(v, rng);
            ++injected;
        } else {
            FillNormal(v, rng);
        }

        auto anomaly_result = engine.DetectAnomaly(sensor, v, kAnomalyThreshold);
        if (!anomaly_result) {
            std::fprintf(stderr, "DetectAnomaly failed: %.*s\n",
                         static_cast<int>(chronosv::error_string(anomaly_result.error()).size()),
                         chronosv::error_string(anomaly_result.error()).data());
            return EXIT_FAILURE;
        }
        const bool flag = *anomaly_result;
        if (flag) ++flagged;

        std::printf("sample %3d %s%s\n",
                    i,
                    is_injected_anomaly ? "[INJECTED] " : "           ",
                    flag ? "flagged=ANOMALY" : "flagged=normal");

        // Also append to keep the window rolling forward. In production, the
        // producer would append every reading; whether to also detect on
        // every reading is a policy choice.
        (void)engine.Append(sensor, ts++, v);
    }

    // ---- Summary ---------------------------------------------------------
    std::printf("\nSummary: injected=%d flagged=%d\n", injected, flagged);

    auto stats = engine.GetStats();
    if (stats) {
        std::printf("Engine stats: appends=%llu queries=%llu anomaly_checks=%llu "
                    "overwrites=%llu hot_bytes=%llu\n",
                    static_cast<unsigned long long>(stats->total_appends),
                    static_cast<unsigned long long>(stats->total_queries),
                    static_cast<unsigned long long>(stats->total_anomaly_checks),
                    static_cast<unsigned long long>(stats->total_overwrite_events),
                    static_cast<unsigned long long>(stats->hot_bytes));
    }

    return EXIT_SUCCESS;
}
