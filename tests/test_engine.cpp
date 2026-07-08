/*
 * Engine unit tests — exercises every one of the fifteen public primitives
 * plus the C++ RAII wrapper. Focus: correctness, error paths, cross-sensor
 * isolation, closed-engine semantics.
 *
 * Threading is exercised in test_ring_buffer.cpp; here we focus on the
 * engine-level state machine.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "chronosv/chronos_vector.h"
#include "chronosv/chronos_vector.hpp"

using Catch::Matchers::WithinAbs;

namespace {

/* Minimal valid config for tests. Zero-init then set required fields;
 * the engine substitutes documented defaults for the rest. */
chronosv_config_t basic_config(std::uint32_t dim = 4) {
    chronosv_config_t c{};
    c.abi_version = CHRONOSV_ABI_VERSION;
    c.dim         = dim;
    // cold_path is nullable in Phase 1 (no RocksDB yet).
    return c;
}

// Simple helper to create + auto-destroy an engine handle for RAII in tests.
struct EngineGuard {
    chronosv_engine_t* h = nullptr;
    ~EngineGuard() { chronosv_destroy(h); }
};

}  // namespace

/* ==========================================================================
 * Utility strings
 * ========================================================================== */

TEST_CASE("chronosv_version_string returns a stable string", "[engine][util]") {
    const char* v = chronosv_version_string();
    REQUIRE(v != nullptr);
    // We don't want to assert the exact value — just that it's non-empty and
    // that repeat calls return the same static.
    REQUIRE(std::string_view(v).size() > 0);
    REQUIRE(v == chronosv_version_string());
}

TEST_CASE("chronosv_error_string covers every documented code",
          "[engine][util]") {
    const chronosv_error_t codes[] = {
        CHRONOSV_OK,
        CHRONOSV_WARN_RANGE_TRUNCATED,
        CHRONOSV_WARN_PARTIAL_RESULT,
        CHRONOSV_ERR_INVALID_ARG,
        CHRONOSV_ERR_NOT_FOUND,
        CHRONOSV_ERR_DIM_MISMATCH,
        CHRONOSV_ERR_IO,
        CHRONOSV_ERR_OOM,
        CHRONOSV_ERR_UNSUPPORTED,
        CHRONOSV_ERR_CLOSED,
        CHRONOSV_ERR_CAPACITY,
        CHRONOSV_ERR_CORRUPTION,
        CHRONOSV_ERR_INTERNAL,
    };
    for (auto c : codes) {
        const char* s = chronosv_error_string(c);
        REQUIRE(s != nullptr);
        REQUIRE(std::string_view(s) != "unknown");
    }
    // Some out-of-range value returns "unknown".
    REQUIRE(std::string_view(chronosv_error_string(-31337)) == "unknown");
}

/* ==========================================================================
 * chronosv_create / destroy / close
 * ========================================================================== */

TEST_CASE("chronosv_create succeeds with minimal valid config",
          "[engine][lifecycle]") {
    EngineGuard g;
    auto cfg = basic_config();
    chronosv_error_t err = CHRONOSV_ERR_INTERNAL;
    g.h = chronosv_create(&cfg, &err);
    REQUIRE(g.h != nullptr);
    REQUIRE(err == CHRONOSV_OK);
}

TEST_CASE("chronosv_create rejects invalid configs",
          "[engine][lifecycle][error]") {
    // ABI version mismatch
    {
        auto cfg = basic_config();
        cfg.abi_version = 0;
        chronosv_error_t err = CHRONOSV_OK;
        REQUIRE(chronosv_create(&cfg, &err) == nullptr);
        REQUIRE(err == CHRONOSV_ERR_INVALID_ARG);
    }
    // dim = 0
    {
        auto cfg = basic_config();
        cfg.dim = 0;
        chronosv_error_t err = CHRONOSV_OK;
        REQUIRE(chronosv_create(&cfg, &err) == nullptr);
        REQUIRE(err == CHRONOSV_ERR_INVALID_ARG);
    }
    // Non-power-of-2 ring_capacity
    {
        auto cfg = basic_config();
        cfg.ring_capacity = 100;
        chronosv_error_t err = CHRONOSV_OK;
        REQUIRE(chronosv_create(&cfg, &err) == nullptr);
        REQUIRE(err == CHRONOSV_ERR_INVALID_ARG);
    }
    // Sub-minimum ring_capacity
    {
        auto cfg = basic_config();
        cfg.ring_capacity = 32;
        chronosv_error_t err = CHRONOSV_OK;
        REQUIRE(chronosv_create(&cfg, &err) == nullptr);
        REQUIRE(err == CHRONOSV_ERR_INVALID_ARG);
    }
    // Bad distance metric
    {
        auto cfg = basic_config();
        cfg.distance_metric = 99;
        chronosv_error_t err = CHRONOSV_OK;
        REQUIRE(chronosv_create(&cfg, &err) == nullptr);
        REQUIRE(err == CHRONOSV_ERR_INVALID_ARG);
    }
    // INT8 without the compile flag: rejected as UNSUPPORTED. With the
    // flag on, creation succeeds — this test case only exercises the
    // rejection path; the success path is covered by the dedicated INT8
    // engine tests.
#ifndef CHRONOSV_ENABLE_INT8
    {
        auto cfg = basic_config();
        cfg.storage_dtype = CHRONOSV_DTYPE_INT8;
        chronosv_error_t err = CHRONOSV_OK;
        REQUIRE(chronosv_create(&cfg, &err) == nullptr);
        REQUIRE(err == CHRONOSV_ERR_UNSUPPORTED);
    }
#endif
    // Null config
    {
        chronosv_error_t err = CHRONOSV_OK;
        REQUIRE(chronosv_create(nullptr, &err) == nullptr);
        REQUIRE(err == CHRONOSV_ERR_INVALID_ARG);
    }
}

TEST_CASE("chronosv_destroy is null-safe", "[engine][lifecycle]") {
    chronosv_destroy(nullptr);   // must not crash
}

TEST_CASE("chronosv_close then Append returns CHRONOSV_ERR_CLOSED",
          "[engine][lifecycle]") {
    auto cfg = basic_config();
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    REQUIRE(g.h != nullptr);
    REQUIRE(chronosv_close(g.h) == CHRONOSV_OK);

    std::array<float, 4> v = {1, 2, 3, 4};
    REQUIRE(chronosv_append(g.h, "s1", 0, v.data(), 4, nullptr)
            == CHRONOSV_ERR_CLOSED);
    REQUIRE(chronosv_close(g.h) == CHRONOSV_OK);  // idempotent
}

TEST_CASE("chronosv_open at nonexistent path fails with IO error",
          "[engine][lifecycle][phase2]") {
    chronosv_engine_t* h = nullptr;
    /* Path can't be created (root not writable + unusual segments). */
    REQUIRE(chronosv_open("/nonexistent/cannot/create/here", &h) == CHRONOSV_ERR_IO);
    REQUIRE(h == nullptr);
}

TEST_CASE("chronosv_flush on Phase-1 style engine (no cold_path) is OK",
          "[engine][lifecycle][phase2][flush]") {
    /* When cfg.cold_path is null, backend is NullStorageBackend — Flush
     * still succeeds (WriteBlocks were no-ops, Flush is a no-op). */
    auto cfg = basic_config();
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    REQUIRE(g.h != nullptr);
    REQUIRE(chronosv_flush(g.h) == CHRONOSV_OK);
}

/* ==========================================================================
 * Append + argument validation
 * ========================================================================== */

TEST_CASE("chronosv_append happy path", "[engine][append]") {
    auto cfg = basic_config(4);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);

    std::array<float, 4> v = {1, 2, 3, 4};
    REQUIRE(chronosv_append(g.h, "sensor_a", 100, v.data(), 4, nullptr) == CHRONOSV_OK);

    chronosv_stats_t s{};
    REQUIRE(chronosv_get_stats(g.h, &s) == CHRONOSV_OK);
    REQUIRE(s.total_appends == 1);
    REQUIRE(s.sensor_count == 1);
    REQUIRE(s.hot_bytes > 0);
}

TEST_CASE("chronosv_append rejects wrong dim", "[engine][append][error]") {
    auto cfg = basic_config(4);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    std::array<float, 3> v = {1, 2, 3};
    REQUIRE(chronosv_append(g.h, "s", 0, v.data(), 3, nullptr)
            == CHRONOSV_ERR_DIM_MISMATCH);
}

TEST_CASE("chronosv_append rejects null / empty / bad sensor_id",
          "[engine][append][error]") {
    auto cfg = basic_config();
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    std::array<float, 4> v = {1, 2, 3, 4};

    REQUIRE(chronosv_append(g.h, nullptr, 0, v.data(), 4, nullptr) == CHRONOSV_ERR_INVALID_ARG);
    REQUIRE(chronosv_append(g.h, "", 0, v.data(), 4, nullptr) == CHRONOSV_ERR_INVALID_ARG);
    // Reserved separator 0x01 in ID
    const char bad[] = {'s', 0x01, 'x', '\0'};
    REQUIRE(chronosv_append(g.h, bad, 0, v.data(), 4, nullptr) == CHRONOSV_ERR_INVALID_ARG);
    // Null vector
    REQUIRE(chronosv_append(g.h, "s", 0, nullptr, 4, nullptr) == CHRONOSV_ERR_INVALID_ARG);
    // Dim = 0
    REQUIRE(chronosv_append(g.h, "s", 0, v.data(), 0, nullptr) == CHRONOSV_ERR_INVALID_ARG);
}

TEST_CASE("chronosv_append rejects overlong sensor_id",
          "[engine][append][error]") {
    auto cfg = basic_config();
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    std::string huge(300, 'x');
    std::array<float, 4> v = {1, 2, 3, 4};
    REQUIRE(chronosv_append(g.h, huge.c_str(), 0, v.data(), 4, nullptr)
            == CHRONOSV_ERR_INVALID_ARG);
}

TEST_CASE("chronosv_append_batch happy path + count=0 is no-op",
          "[engine][append_batch]") {
    auto cfg = basic_config(4);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);

    // count=0 must be OK (short-circuit).
    REQUIRE(chronosv_append_batch(g.h, "s", nullptr, nullptr, 0, 4, nullptr)
            == CHRONOSV_OK);

    constexpr int N = 10;
    std::vector<std::int64_t> ts(N);
    std::vector<float> vecs(N * 4);
    for (int i = 0; i < N; ++i) {
        ts[i] = i * 10;
        for (int j = 0; j < 4; ++j) vecs[i * 4 + j] = float(i + j);
    }
    REQUIRE(chronosv_append_batch(g.h, "s", ts.data(), vecs.data(),
                                  N, 4, nullptr) == CHRONOSV_OK);
    chronosv_stats_t st{};
    chronosv_get_stats(g.h, &st);
    REQUIRE(st.total_appends == N);
}

/* ==========================================================================
 * QueryNearestN
 * ========================================================================== */

TEST_CASE("chronosv_query_nearest_n returns the exact match first",
          "[engine][query][cosine]") {
    auto cfg = basic_config(4);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);

    // Insert several unit-ish vectors along different directions.
    std::vector<std::array<float, 4>> vs = {
        {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1},
        {0.9f, 0.1f, 0, 0}, {0.5f, 0.5f, 0.5f, 0.5f},
    };
    for (int i = 0; i < int(vs.size()); ++i) {
        REQUIRE(chronosv_append(g.h, "s", i, vs[i].data(), 4, nullptr) == CHRONOSV_OK);
    }

    // Query for (1,0,0,0) — the exact match at index 0 (ts=0) should win.
    std::array<float, 4> q = {1, 0, 0, 0};
    std::array<std::int64_t, 3> ts{};
    std::array<float, 3>        sc{};
    int count = 0;
    auto err = chronosv_query_nearest_n(g.h, "s", q.data(), 4, 3,
                                        ts.data(), sc.data(), &count);
    REQUIRE(err == CHRONOSV_OK);
    REQUIRE(count == 3);
    REQUIRE(ts[0] == 0);                                  // exact match first
    REQUIRE_THAT(sc[0], WithinAbs(1.0f, 1e-6f));          // cosine of self
    // Second-best is (0.9, 0.1, 0, 0) at ts=4.
    REQUIRE(ts[1] == 4);
}

TEST_CASE("chronosv_query_nearest_n on empty sensor returns PARTIAL",
          "[engine][query][edge]") {
    auto cfg = basic_config(4);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    // Preallocate but never append — sensor exists, ring is empty.
    REQUIRE(chronosv_preallocate_sensor(g.h, "empty") == CHRONOSV_OK);

    std::array<float, 4> q = {1, 0, 0, 0};
    std::array<std::int64_t, 5> ts{};
    std::array<float, 5>        sc{};
    int count = -1;
    auto err = chronosv_query_nearest_n(g.h, "empty", q.data(), 4, 5,
                                        ts.data(), sc.data(), &count);
    REQUIRE(err == CHRONOSV_WARN_PARTIAL_RESULT);
    REQUIRE(count == 0);
}

TEST_CASE("chronosv_query_nearest_n with n > available returns PARTIAL",
          "[engine][query][edge]") {
    auto cfg = basic_config(4);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    std::array<float, 4> v = {1, 2, 3, 4};
    chronosv_append(g.h, "s", 0, v.data(), 4, nullptr);
    chronosv_append(g.h, "s", 1, v.data(), 4, nullptr);

    std::array<float, 4> q = {1, 2, 3, 4};
    std::array<std::int64_t, 10> ts{};
    std::array<float, 10>        sc{};
    int count = -1;
    auto err = chronosv_query_nearest_n(g.h, "s", q.data(), 4, 10,
                                        ts.data(), sc.data(), &count);
    REQUIRE(err == CHRONOSV_WARN_PARTIAL_RESULT);
    REQUIRE(count == 2);
}

TEST_CASE("chronosv_query_nearest_n unknown sensor returns NOT_FOUND",
          "[engine][query][error]") {
    auto cfg = basic_config(4);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    std::array<float, 4> q = {1, 2, 3, 4};
    std::array<std::int64_t, 3> ts{};
    std::array<float, 3>        sc{};
    int count = 42;
    auto err = chronosv_query_nearest_n(g.h, "nope", q.data(), 4, 3,
                                        ts.data(), sc.data(), &count);
    REQUIRE(err == CHRONOSV_ERR_NOT_FOUND);
    REQUIRE(count == 0);
}

TEST_CASE("query_nearest_n works when the ring wrapped (chunked path)",
          "[engine][query][wrap]") {
    // Use the minimum ring capacity (64) so we can force a wrap with a
    // modest number of appends. After wrapping, the query must correctly
    // span the two physical chunks — this is what FillScoresForRange does
    // when start_slot + N exceeds capacity.
    auto cfg = basic_config(4);
    cfg.ring_capacity = 64;
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);

    // Insert 100 entries with vectors that are NOT colinear — we vary the
    // last three components so each direction is distinct (cosine-wise).
    // Using vector[i] = (i, 1, i%7, i%11) gives us unique directions per i.
    for (int i = 0; i < 100; ++i) {
        std::array<float, 4> v = {float(i), 1.f, float(i % 7), float(i % 11)};
        chronosv_append(g.h, "s", i, v.data(), 4, nullptr);
    }
    // After 100 appends into cap=64, the surviving entries are for ts=36..99.
    // Query for the vector at ts=99 exactly — cosine=1.0 uniquely.
    std::array<float, 4> q = {99.0f, 1.0f, float(99 % 7), float(99 % 11)};
    std::array<std::int64_t, 3> ts{};
    std::array<float, 3>        sc{};
    int count = 0;
    auto err = chronosv_query_nearest_n(g.h, "s", q.data(), 4, 3,
                                        ts.data(), sc.data(), &count);
    REQUIRE(err == CHRONOSV_OK);
    REQUIRE(count == 3);
    REQUIRE(ts[0] == 99);
    REQUIRE_THAT(sc[0], WithinAbs(1.0f, 1e-5f));
    // All results must be from the surviving window [36, 99].
    for (int i = 0; i < count; ++i) {
        REQUIRE(ts[i] >= 36);
        REQUIRE(ts[i] <= 99);
    }
}

/* ==========================================================================
 * QueryRange
 * ========================================================================== */

TEST_CASE("chronosv_query_range filters by timestamp",
          "[engine][query_range]") {
    auto cfg = basic_config(2);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    for (int i = 0; i < 20; ++i) {
        std::array<float, 2> v = {float(i), float(i) * 2};
        chronosv_append(g.h, "s", i * 100, v.data(), 2, nullptr);
    }
    // Query [500, 1200] — matches ts in {500,600,700,800,900,1000,1100,1200} → 8.
    std::array<std::int64_t, 20> ts{};
    std::array<float, 40>        vecs{};
    int count = 0;
    auto err = chronosv_query_range(g.h, "s", 500, 1200,
                                    ts.data(), vecs.data(), 20, &count);
    REQUIRE(err == CHRONOSV_OK);
    REQUIRE(count == 8);
    // Verify the vectors round-tripped correctly.
    for (int i = 0; i < count; ++i) {
        const std::int64_t expected_ts = ts[i];
        const int          idx         = int(expected_ts / 100);
        REQUIRE_THAT(vecs[i * 2],     WithinAbs(float(idx),          1e-6f));
        REQUIRE_THAT(vecs[i * 2 + 1], WithinAbs(float(idx) * 2,      1e-6f));
    }
}

TEST_CASE("chronosv_query_range warns TRUNCATED when start is before window",
          "[engine][query_range][edge]") {
    auto cfg = basic_config(1);
    cfg.ring_capacity = 64;
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    // Fill past capacity so the oldest surviving ts is 36.
    for (int i = 0; i < 100; ++i) {
        float v = float(i);
        chronosv_append(g.h, "s", i, &v, 1, nullptr);
    }
    std::array<std::int64_t, 100> ts{};
    std::array<float, 100>        vecs{};
    int count = 0;
    // Ask for [0, 99] — starts BEFORE the surviving window (oldest is 36).
    auto err = chronosv_query_range(g.h, "s", 0, 99,
                                    ts.data(), vecs.data(), 100, &count);
    REQUIRE(err == CHRONOSV_WARN_RANGE_TRUNCATED);
    // Should return the 64 that are still in the hot ring.
    REQUIRE(count == 64);
}

/* ==========================================================================
 * DetectAnomaly
 * ========================================================================== */

TEST_CASE("chronosv_detect_anomaly flags an outlier under cosine metric",
          "[engine][anomaly][cosine]") {
    auto cfg = basic_config(4);
    // Explicit — default is cosine but let's be clear for the test intent.
    cfg.distance_metric = CHRONOSV_METRIC_COSINE;
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);

    // Build a window of very-similar vectors all pointing in +x.
    for (int i = 0; i < 50; ++i) {
        std::array<float, 4> v = {1.0f + 0.01f * i, 0.0f, 0.0f, 0.0f};
        chronosv_append(g.h, "s", i, v.data(), 4, nullptr);
    }

    // Near-mean vector: also +x direction. Should NOT be anomalous.
    int is_anom = -1;
    std::array<float, 4> normal = {1.5f, 0.0f, 0.0f, 0.0f};
    REQUIRE(chronosv_detect_anomaly(g.h, "s", normal.data(), 4,
                                    /*threshold=*/0.5f, &is_anom) == CHRONOSV_OK);
    REQUIRE(is_anom == 0);

    // Orthogonal vector: cosine distance ≈ 1.0, well above threshold.
    is_anom = -1;
    std::array<float, 4> outlier = {0.0f, 1.0f, 0.0f, 0.0f};
    REQUIRE(chronosv_detect_anomaly(g.h, "s", outlier.data(), 4,
                                    /*threshold=*/0.5f, &is_anom) == CHRONOSV_OK);
    REQUIRE(is_anom == 1);
}

TEST_CASE("chronosv_detect_anomaly on empty sensor returns not-anomaly",
          "[engine][anomaly][edge]") {
    auto cfg = basic_config(2);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    REQUIRE(chronosv_preallocate_sensor(g.h, "empty") == CHRONOSV_OK);
    int is_anom = -1;
    std::array<float, 2> v = {1, 0};
    REQUIRE(chronosv_detect_anomaly(g.h, "empty", v.data(), 2,
                                    0.5f, &is_anom) == CHRONOSV_OK);
    REQUIRE(is_anom == 0);
}

TEST_CASE("chronosv_detect_anomaly under euclidean metric",
          "[engine][anomaly][euclidean]") {
    auto cfg = basic_config(2);
    cfg.distance_metric = CHRONOSV_METRIC_EUCLIDEAN;
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);

    // Cluster around (0, 0).
    for (int i = 0; i < 20; ++i) {
        const float off = 0.1f * (i - 10);
        std::array<float, 2> v = {off, off};
        chronosv_append(g.h, "s", i, v.data(), 2, nullptr);
    }
    int is_anom = -1;
    // Point near the mean → not anomaly.
    std::array<float, 2> near = {0.05f, 0.05f};
    REQUIRE(chronosv_detect_anomaly(g.h, "s", near.data(), 2,
                                    /*threshold=*/1.0f, &is_anom) == CHRONOSV_OK);
    REQUIRE(is_anom == 0);
    // Far outlier.
    std::array<float, 2> far = {50.0f, 50.0f};
    REQUIRE(chronosv_detect_anomaly(g.h, "s", far.data(), 2,
                                    /*threshold=*/1.0f, &is_anom) == CHRONOSV_OK);
    REQUIRE(is_anom == 1);
}

/* ==========================================================================
 * Sensor lifecycle (drop, preallocate, list)
 * ========================================================================== */

TEST_CASE("chronosv_drop_sensor removes and NOT_FOUND thereafter",
          "[engine][drop]") {
    auto cfg = basic_config(4);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    std::array<float, 4> v = {1, 2, 3, 4};
    chronosv_append(g.h, "s", 0, v.data(), 4, nullptr);
    REQUIRE(chronosv_drop_sensor(g.h, "s") == CHRONOSV_OK);
    REQUIRE(chronosv_drop_sensor(g.h, "s") == CHRONOSV_ERR_NOT_FOUND);

    // Verify sensor_count went to zero.
    chronosv_stats_t st{};
    chronosv_get_stats(g.h, &st);
    REQUIRE(st.sensor_count == 0);
    REQUIRE(st.total_dropped_sensors == 1);
}

TEST_CASE("chronosv_preallocate_sensor is idempotent", "[engine][preallocate]") {
    auto cfg = basic_config(4);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    REQUIRE(chronosv_preallocate_sensor(g.h, "p") == CHRONOSV_OK);
    REQUIRE(chronosv_preallocate_sensor(g.h, "p") == CHRONOSV_OK);

    chronosv_stats_t st{};
    chronosv_get_stats(g.h, &st);
    REQUIRE(st.sensor_count == 1);
}

TEST_CASE("chronosv_list_sensors returns all registered ids",
          "[engine][list]") {
    auto cfg = basic_config(4);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    for (const char* id : {"alpha", "beta", "gamma"}) {
        chronosv_preallocate_sensor(g.h, id);
    }
    char* ids[10] = {};
    std::size_t count = 0;
    REQUIRE(chronosv_list_sensors(g.h, ids, 10, &count) == CHRONOSV_OK);
    REQUIRE(count == 3);
    // Collect names into a set for order-independent check.
    std::vector<std::string> got;
    for (std::size_t i = 0; i < count; ++i) {
        got.emplace_back(ids[i]);
        std::free(ids[i]);
    }
    std::sort(got.begin(), got.end());
    REQUIRE(got == std::vector<std::string>{"alpha", "beta", "gamma"});
}

TEST_CASE("chronosv_list_sensors with max=0 returns count=0 without touching out_ids",
          "[engine][list][edge]") {
    auto cfg = basic_config(4);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    chronosv_preallocate_sensor(g.h, "a");
    std::size_t count = 42;
    REQUIRE(chronosv_list_sensors(g.h, nullptr, 0, &count) == CHRONOSV_OK);
    REQUIRE(count == 0);
}

TEST_CASE("max_sensors cap enforced", "[engine][capacity]") {
    auto cfg = basic_config(4);
    cfg.max_sensors = 2;
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    REQUIRE(chronosv_preallocate_sensor(g.h, "a") == CHRONOSV_OK);
    REQUIRE(chronosv_preallocate_sensor(g.h, "b") == CHRONOSV_OK);
    REQUIRE(chronosv_preallocate_sensor(g.h, "c") == CHRONOSV_ERR_CAPACITY);
}

/* ==========================================================================
 * Cross-sensor isolation
 * ========================================================================== */

TEST_CASE("Two sensors are isolated", "[engine][multi_sensor]") {
    auto cfg = basic_config(2);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);

    std::array<float, 2> a = {1, 0};
    std::array<float, 2> b = {0, 1};
    chronosv_append(g.h, "A", 100, a.data(), 2, nullptr);
    chronosv_append(g.h, "B", 100, b.data(), 2, nullptr);

    std::array<std::int64_t, 5> ts_a{}, ts_b{};
    std::array<float, 5>        sc_a{}, sc_b{};
    int c_a = 0, c_b = 0;

    std::array<float, 2> qa = {1, 0};
    std::array<float, 2> qb = {0, 1};
    chronosv_query_nearest_n(g.h, "A", qa.data(), 2, 5,
                             ts_a.data(), sc_a.data(), &c_a);
    chronosv_query_nearest_n(g.h, "B", qb.data(), 2, 5,
                             ts_b.data(), sc_b.data(), &c_b);
    REQUIRE(c_a == 1);
    REQUIRE(c_b == 1);
    REQUIRE_THAT(sc_a[0], WithinAbs(1.0f, 1e-6f));
    REQUIRE_THAT(sc_b[0], WithinAbs(1.0f, 1e-6f));
}

/* ==========================================================================
 * GetStats accounting
 * ========================================================================== */

TEST_CASE("chronosv_get_stats accumulates append and query counts",
          "[engine][stats]") {
    auto cfg = basic_config(4);
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    std::array<float, 4> v = {1, 2, 3, 4};
    for (int i = 0; i < 25; ++i) {
        chronosv_append(g.h, "s", i, v.data(), 4, nullptr);
    }
    std::array<std::int64_t, 3> ts{};
    std::array<float, 3>        sc{};
    int count = 0;
    for (int i = 0; i < 5; ++i) {
        chronosv_query_nearest_n(g.h, "s", v.data(), 4, 3,
                                 ts.data(), sc.data(), &count);
    }
    chronosv_stats_t st{};
    REQUIRE(chronosv_get_stats(g.h, &st) == CHRONOSV_OK);
    REQUIRE(st.total_appends == 25);
    REQUIRE(st.total_queries == 5);
    REQUIRE(st.sensor_count == 1);
}

TEST_CASE("Stats overwrite counters increase when producer laps ring",
          "[engine][stats][overwrite]") {
    auto cfg = basic_config(1);
    cfg.ring_capacity = 64;
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    for (int i = 0; i < 200; ++i) {
        float f = float(i);
        chronosv_append(g.h, "s", i, &f, 1, nullptr);
    }
    chronosv_stats_t st{};
    chronosv_get_stats(g.h, &st);
    // 200 appends into capacity=64 → 200-64 = 136 overwrites.
    REQUIRE(st.total_overwrite_events == 136);
    REQUIRE(st.total_overwritten_entries == 136);
}

/* ==========================================================================
 * Metrics sink
 * ========================================================================== */

TEST_CASE("Metrics sink receives append events", "[engine][metrics]") {
    // Counter user_data captures how many events fired.
    struct Counters { std::atomic<int> appends{0}; std::atomic<int> queries{0}; };
    Counters cnt;

    chronosv_metrics_sink_t sink{};
    sink.user_data = &cnt;
    sink.on_append = [](void* ud, const char*, std::int64_t) {
        static_cast<Counters*>(ud)->appends.fetch_add(1);
    };
    sink.on_query = [](void* ud, const char*, std::int64_t, int) {
        static_cast<Counters*>(ud)->queries.fetch_add(1);
    };

    auto cfg = basic_config(4);
    cfg.metrics_sink = &sink;
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);

    std::array<float, 4> v = {1, 2, 3, 4};
    for (int i = 0; i < 7; ++i) chronosv_append(g.h, "s", i, v.data(), 4, nullptr);

    std::array<std::int64_t, 3> ts{}; std::array<float, 3> sc{}; int count = 0;
    chronosv_query_nearest_n(g.h, "s", v.data(), 4, 3,
                             ts.data(), sc.data(), &count);

    REQUIRE(cnt.appends.load() == 7);
    REQUIRE(cnt.queries.load() == 1);
}

/* ==========================================================================
 * C++ wrapper (chronos_vector.hpp)
 * ========================================================================== */

TEST_CASE("chronosv::Engine RAII wrapper: create / append / query / stats",
          "[engine][cpp_wrapper]") {
    auto cfg = basic_config(4);
    auto res = chronosv::Engine::Create(cfg);
    REQUIRE(res.has_value());
    chronosv::Engine e = std::move(*res);

    std::vector<float> v = {1, 0, 0, 0};
    REQUIRE(e.Append("s", 0, v) == chronosv::Error::Ok);

    auto q = e.QueryNearestN("s", v, /*n=*/1);
    REQUIRE(q.has_value());
    REQUIRE(q->size() == 1);
    REQUIRE(q->front().timestamp_ms == 0);
    REQUIRE_THAT(q->front().score, WithinAbs(1.0f, 1e-6f));

    auto stats = e.GetStats();
    REQUIRE(stats.has_value());
    REQUIRE(stats->total_appends == 1);
    REQUIRE(stats->total_queries == 1);
}

TEST_CASE("chronosv::Engine move construction transfers ownership",
          "[engine][cpp_wrapper]") {
    auto cfg = basic_config(4);
    auto res = chronosv::Engine::Create(cfg);
    REQUIRE(res.has_value());
    chronosv::Engine a = std::move(*res);
    REQUIRE(a.raw() != nullptr);
    chronosv::Engine b = std::move(a);
    REQUIRE(a.raw() == nullptr);
    REQUIRE(b.raw() != nullptr);
}

TEST_CASE("chronosv::Engine wrapper surfaces errors via Result",
          "[engine][cpp_wrapper][error]") {
    // Invalid config → Create fails.
    chronosv_config_t bad{};
    bad.abi_version = 0;   // wrong
    bad.dim = 4;
    auto res = chronosv::Engine::Create(bad);
    REQUIRE_FALSE(res.has_value());
    REQUIRE(res.error() == chronosv::Error::InvalidArg);
}

/* ==========================================================================
 * Eviction (background jthread + explicit maintain trigger)
 * ========================================================================== */

TEST_CASE("MaintainSlidingWindow triggers immediate tail advance",
          "[engine][eviction][maintain]") {
    // Timestamps span 0..999 in steps of 10 (100 entries). After maintain
    // with window_ms=200, the cutoff = max_ts(990) - 200 = 790.
    // All entries with ts < 790 should be evicted → surviving ts in [790, 990]
    // → 21 entries left (indices 79..99).
    auto cfg = basic_config(1);
    cfg.ring_capacity = 128;
    cfg.window_duration_ms = 1'000'000;  // huge so background thread doesn't fire yet
    cfg.eviction_interval_ms = 60000;
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    REQUIRE(g.h != nullptr);

    for (int i = 0; i < 100; ++i) {
        float v = float(i);
        chronosv_append(g.h, "s", i * 10, &v, 1, nullptr);
    }
    // Confirm baseline: 100 entries, no evictions yet.
    chronosv_stats_t st{};
    chronosv_get_stats(g.h, &st);
    REQUIRE(st.total_evictions == 0);

    // Tighten window and trigger eviction pass.
    REQUIRE(chronosv_maintain_sliding_window(g.h, 200) == CHRONOSV_OK);

    // Query the range to see what survived.
    std::array<std::int64_t, 200> ts{};
    std::array<float, 200>        vecs{};
    int count = 0;
    auto err = chronosv_query_range(g.h, "s", 0, 10000,
                                    ts.data(), vecs.data(), 200, &count);
    // Truncation warning is EXPECTED here: we asked for [0, 10000] but the
    // oldest surviving entry is 790 (post-eviction), so t_start_ms=0 is
    // before the hot window.
    REQUIRE((err == CHRONOSV_OK || err == CHRONOSV_WARN_RANGE_TRUNCATED));
    REQUIRE(count == 21);
    // Oldest surviving must be ≥ cutoff (790).
    for (int i = 0; i < count; ++i) REQUIRE(ts[i] >= 790);

    // Stats reflect the eviction call.
    chronosv_get_stats(g.h, &st);
    REQUIRE(st.total_evictions >= 1);
}

TEST_CASE("Background eviction thread fires on interval",
          "[engine][eviction][background]") {
    // Deterministic wait via the on_eviction sink: the sink notifies a
    // condvar, the test waits with a bounded timeout. Much less flaky than
    // polling stats in a sleep loop.
    struct Waiter {
        std::mutex mu;
        std::condition_variable cv;
        std::atomic<int> events{0};
    };
    Waiter waiter;

    chronosv_metrics_sink_t sink{};
    sink.user_data   = &waiter;
    sink.on_eviction = [](void* ud, const char*, std::int64_t, std::int64_t) {
        auto* w = static_cast<Waiter*>(ud);
        w->events.fetch_add(1);
        {
            std::lock_guard<std::mutex> lock(w->mu);
        }
        w->cv.notify_all();
    };

    auto cfg = basic_config(1);
    cfg.ring_capacity = 128;
    cfg.window_duration_ms = 100;
    cfg.eviction_interval_ms = 100;   // must be >= 100 per validator
    cfg.metrics_sink = &sink;
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    REQUIRE(g.h != nullptr);

    // Seed with 50 old entries then one far-future entry — the mismatch is
    // what makes the older entries evictable (cutoff = max_ts - window).
    for (int i = 0; i < 50; ++i) {
        float v = float(i);
        chronosv_append(g.h, "s", i, &v, 1, nullptr);
    }
    float far_val = 999.0f;
    chronosv_append(g.h, "s", /*ts=*/1000, &far_val, 1, nullptr);

    // Wait deterministically for the background thread to fire on_eviction
    // at least once. 2 s is generous — expected within ~150 ms.
    {
        std::unique_lock<std::mutex> lock(waiter.mu);
        const bool got = waiter.cv.wait_for(
            lock, std::chrono::seconds(2),
            [&] { return waiter.events.load() >= 1; });
        REQUIRE(got);
    }

    // Verify only the recent entry survives.
    std::array<std::int64_t, 100> ts{};
    std::array<float, 100>        vecs{};
    int count = 0;
    chronosv_query_range(g.h, "s", -10, 5000,
                         ts.data(), vecs.data(), 100, &count);
    REQUIRE(count == 1);
    REQUIRE(ts[0] == 1000);
}

TEST_CASE("Eviction on_eviction sink callback fires",
          "[engine][eviction][sink]") {
    struct Cnt { std::atomic<int> events{0}; std::atomic<std::int64_t> total_evicted{0}; };
    Cnt cnt;
    chronosv_metrics_sink_t sink{};
    sink.user_data   = &cnt;
    sink.on_eviction = [](void* ud, const char*, std::int64_t n, std::int64_t) {
        auto* c = static_cast<Cnt*>(ud);
        c->events.fetch_add(1);
        c->total_evicted.fetch_add(n);
    };

    auto cfg = basic_config(1);
    cfg.ring_capacity = 128;
    cfg.window_duration_ms = 100;
    cfg.metrics_sink = &sink;
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);

    for (int i = 0; i < 30; ++i) {
        float v = float(i);
        chronosv_append(g.h, "s", i, &v, 1, nullptr);
    }
    // Bump ts far forward and trigger eviction synchronously.
    float far = 999.0f;
    chronosv_append(g.h, "s", 1000, &far, 1, nullptr);
    chronosv_maintain_sliding_window(g.h, 100);

    REQUIRE(cnt.events.load() >= 1);
    // We had 30 old + 1 new. Cutoff = 1000 - 100 = 900. All 30 old evictable.
    REQUIRE(cnt.total_evicted.load() == 30);
}

TEST_CASE("Stats: overwrite counters survive DropSensor",
          "[engine][stats][overwrite][drop]") {
    // Regression guard: an earlier draft read overwrite counts only from
    // live rings, so dropping a sensor silently rewound the "we lost data"
    // canary. GetStats must report a monotonic lifetime total.
    auto cfg = basic_config(1);
    cfg.ring_capacity = 64;
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);

    // Force overwrites on sensor "s" by appending well past capacity.
    for (int i = 0; i < 200; ++i) {
        float v = float(i);
        chronosv_append(g.h, "s", i, &v, 1, nullptr);
    }
    chronosv_stats_t before{};
    chronosv_get_stats(g.h, &before);
    REQUIRE(before.total_overwrite_events == 136);   // 200 - capacity(64)
    REQUIRE(before.total_overwritten_entries == 136);

    // Drop the sensor. Its live-ring overwrite counts vanish, but the
    // engine must retain them in dropped-sensor cumulative atomics.
    REQUIRE(chronosv_drop_sensor(g.h, "s") == CHRONOSV_OK);
    chronosv_stats_t after{};
    chronosv_get_stats(g.h, &after);
    REQUIRE(after.sensor_count == 0);
    REQUIRE(after.total_overwrite_events == 136);    // MUST NOT decrease
    REQUIRE(after.total_overwritten_entries == 136);

    // Any subsequent activity on a NEW sensor should add on top.
    for (int i = 0; i < 100; ++i) {
        float v = float(i);
        chronosv_append(g.h, "t", i, &v, 1, nullptr);
    }
    chronosv_stats_t final{};
    chronosv_get_stats(g.h, &final);
    REQUIRE(final.total_overwrite_events == 136 + (100 - 64));
    REQUIRE(final.total_overwritten_entries == 136 + (100 - 64));
}

/* ==========================================================================
 * Phase 2 persistence — cold_path routing, flush, open/recovery
 * ========================================================================== */

TEST_CASE("Phase 2: engine with cold_path persists blocks; reopen sees them",
          "[engine][phase2][persistence]") {
    /* Use a per-test tempdir so this is hermetic. */
    namespace fs = std::filesystem;
    static std::atomic<int> counter{0};
    const fs::path cold = fs::temp_directory_path() /
                          ("chronosv_engine_persist_" +
                           std::to_string(counter.fetch_add(1)));
    fs::create_directories(cold);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code e; fs::remove_all(p, e); } } cleanup{cold};

    /* Phase A: create engine with cold_path, ingest, flush, close. */
    {
        auto cfg = basic_config(4);
        std::string cold_str = cold.string();
        cfg.cold_path = cold_str.c_str();
        cfg.window_duration_ms = 100;   /* short window so eviction acts fast */
        cfg.eviction_interval_ms = 100;
        EngineGuard g;
        g.h = chronosv_create(&cfg, nullptr);
        REQUIRE(g.h != nullptr);

        /* Ingest 20 old entries then force them out via chronosv_flush. */
        for (int i = 0; i < 20; ++i) {
            std::array<float, 4> v = {float(i), 0.f, 0.f, 0.f};
            chronosv_append(g.h, "sA", i, v.data(), 4, nullptr);
        }
        REQUIRE(chronosv_flush(g.h) == CHRONOSV_OK);

        chronosv_stats_t st{};
        chronosv_get_stats(g.h, &st);
        REQUIRE(st.total_evictions > 0);
        REQUIRE(st.cold_bytes_estimate >= 0);
        REQUIRE(st.flush_errors_total == 0);
    }

    /* Phase B: chronosv_open at the same path; verify sensor is recovered. */
    {
        chronosv_engine_t* h = nullptr;
        REQUIRE(chronosv_open(cold.string().c_str(), &h) == CHRONOSV_OK);
        REQUIRE(h != nullptr);
        EngineGuard g;
        g.h = h;

        char* ids[10] = {};
        std::size_t count = 0;
        REQUIRE(chronosv_list_sensors(g.h, ids, 10, &count) == CHRONOSV_OK);
        REQUIRE(count == 1);
        REQUIRE(std::string(ids[0]) == "sA");
        std::free(ids[0]);
    }
}

TEST_CASE("Phase 2: chronosv_flush with cold_path returns OK",
          "[engine][phase2][flush]") {
    namespace fs = std::filesystem;
    static std::atomic<int> counter{0};
    const fs::path cold = fs::temp_directory_path() /
                          ("chronosv_engine_flush_" +
                           std::to_string(counter.fetch_add(1)));
    fs::create_directories(cold);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code e; fs::remove_all(p, e); } } cleanup{cold};

    auto cfg = basic_config(4);
    std::string cold_str = cold.string();
    cfg.cold_path = cold_str.c_str();
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    REQUIRE(g.h != nullptr);

    /* Empty flush should succeed (no blocks to write, WAL sync no-op). */
    REQUIRE(chronosv_flush(g.h) == CHRONOSV_OK);

    /* Append then flush — everything ages out and is persisted. */
    for (int i = 0; i < 5; ++i) {
        std::array<float, 4> v = {float(i), 0.f, 0.f, 0.f};
        chronosv_append(g.h, "s", i, v.data(), 4, nullptr);
    }
    REQUIRE(chronosv_flush(g.h) == CHRONOSV_OK);
}

TEST_CASE("Phase 2: metadata roundtrip — Open recovers non-default schema",
          "[engine][phase2][metadata][recovery]") {
    /* Regression guard for concerns #1 + #4 from session 4.
     *
     * Before the metadata fix: chronosv_open always used default cfg
     * (dim=128). If Create used dim=32, reopening then trying to Append
     * a dim=32 vector would fail with DIM_MISMATCH because the recovered
     * engine "believed" it was dim=128. This test proves that the
     * persisted metadata now round-trips schema fields end-to-end. */
    namespace fs = std::filesystem;
    static std::atomic<int> counter{0};
    const fs::path cold = fs::temp_directory_path() /
                          ("chronosv_engine_meta_" +
                           std::to_string(counter.fetch_add(1)));
    fs::create_directories(cold);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code e; fs::remove_all(p, e); } } cleanup{cold};

    std::array<std::uint8_t, 16> original_uuid{};

    /* Phase A: create with NON-DEFAULT schema (dim=32, custom ring_capacity,
     * euclidean metric). Ingest something so a subsequent open has data. */
    {
        auto cfg = basic_config(32);   /* not 128 */
        std::string cold_str = cold.string();
        cfg.cold_path = cold_str.c_str();
        cfg.ring_capacity = 512;       /* non-default */
        cfg.window_duration_ms = 300000; /* 5 min, non-default */
        cfg.distance_metric = CHRONOSV_METRIC_EUCLIDEAN;

        EngineGuard g;
        g.h = chronosv_create(&cfg, nullptr);
        REQUIRE(g.h != nullptr);

        std::array<float, 32> v{};
        for (float& x : v) x = 0.5f;
        REQUIRE(chronosv_append(g.h, "s1", 100, v.data(), 32, nullptr) == CHRONOSV_OK);
        REQUIRE(chronosv_flush(g.h) == CHRONOSV_OK);

        /* Grab the UUID so we can prove Open recovers it. */
        chronosv_stats_t st{};
        chronosv_get_stats(g.h, &st);
        std::memcpy(original_uuid.data(), st.uuid, 16);
    }

    /* Phase B: open at the same path. Recovered engine must have dim=32,
     * euclidean metric, AND the same UUID (identity survives close/reopen). */
    {
        chronosv_engine_t* h = nullptr;
        REQUIRE(chronosv_open(cold.string().c_str(), &h) == CHRONOSV_OK);
        REQUIRE(h != nullptr);
        EngineGuard g;
        g.h = h;

        chronosv_stats_t st{};
        REQUIRE(chronosv_get_stats(g.h, &st) == CHRONOSV_OK);
        REQUIRE(std::memcmp(st.uuid, original_uuid.data(), 16) == 0);

        /* Prove the dim was recovered: Appending a dim=32 vector succeeds,
         * appending anything else fails with DIM_MISMATCH. */
        std::array<float, 32> v_ok{};
        REQUIRE(chronosv_append(g.h, "s1", 200, v_ok.data(), 32, nullptr) == CHRONOSV_OK);
        std::array<float, 128> v_bad{};
        REQUIRE(chronosv_append(g.h, "s1", 300, v_bad.data(), 128, nullptr)
                == CHRONOSV_ERR_DIM_MISMATCH);
    }
}

TEST_CASE("Phase 2: Create at existing path — schema match reuses UUID, mismatch refuses",
          "[engine][phase2][metadata][idempotent]") {
    /* Regression guard for the "Create silently overwrites persisted
     * metadata" concern from the session 4 fix pass. Contract:
     *   - Fresh path: metadata is written, UUID assigned.
     *   - Existing path with SAME schema (dim/dtype/metric/payload/ring_cap):
     *     Create succeeds, UUID preserved (identity survives). Runtime
     *     fields like window_duration_ms can differ and are updated.
     *   - Existing path with DIFFERENT schema: Create refuses with
     *     INVALID_ARG. No silent block-shape corruption possible. */
    namespace fs = std::filesystem;
    static std::atomic<int> counter{0};
    const fs::path cold = fs::temp_directory_path() /
                          ("chronosv_engine_reccreate_" +
                           std::to_string(counter.fetch_add(1)));
    fs::create_directories(cold);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code e; fs::remove_all(p, e); } } cleanup{cold};

    std::array<std::uint8_t, 16> uuid_first{};
    std::string cold_str = cold.string();

    /* Phase A: fresh Create with dim=32. Capture UUID. */
    {
        auto cfg = basic_config(32);
        cfg.cold_path = cold_str.c_str();
        EngineGuard g;
        g.h = chronosv_create(&cfg, nullptr);
        REQUIRE(g.h != nullptr);
        chronosv_stats_t st{};
        chronosv_get_stats(g.h, &st);
        std::memcpy(uuid_first.data(), st.uuid, 16);
    }

    /* Phase B: Create at same path with SAME schema but different window.
     * Must succeed AND preserve UUID. */
    {
        auto cfg = basic_config(32);
        cfg.cold_path = cold_str.c_str();
        cfg.window_duration_ms = 1'234'567;   /* different from default */
        EngineGuard g;
        chronosv_error_t err = CHRONOSV_ERR_INTERNAL;
        g.h = chronosv_create(&cfg, &err);
        REQUIRE(err == CHRONOSV_OK);
        REQUIRE(g.h != nullptr);
        chronosv_stats_t st{};
        chronosv_get_stats(g.h, &st);
        REQUIRE(std::memcmp(st.uuid, uuid_first.data(), 16) == 0);
    }

    /* Phase C: Create at same path with a DIFFERENT dim. Must refuse. */
    {
        auto cfg = basic_config(64);          /* was 32 in phase A */
        cfg.cold_path = cold_str.c_str();
        chronosv_error_t err = CHRONOSV_OK;
        chronosv_engine_t* h = chronosv_create(&cfg, &err);
        REQUIRE(h == nullptr);
        REQUIRE(err == CHRONOSV_ERR_INVALID_ARG);
    }

    /* Phase D: Create at same path with a DIFFERENT metric. Must refuse. */
    {
        auto cfg = basic_config(32);
        cfg.cold_path = cold_str.c_str();
        cfg.distance_metric = CHRONOSV_METRIC_EUCLIDEAN;  /* was COSINE by default */
        chronosv_error_t err = CHRONOSV_OK;
        chronosv_engine_t* h = chronosv_create(&cfg, &err);
        REQUIRE(h == nullptr);
        REQUIRE(err == CHRONOSV_ERR_INVALID_ARG);
    }
}

TEST_CASE("Phase 2: recover_hot_window=0 (default) — Open leaves rings empty",
          "[engine][phase2][recovery]") {
    /* Contract: default recovery does NOT rehydrate the hot rings from
     * persisted blocks. The recovered sensor exists in the map but its
     * ring is empty; queries return WARN_PARTIAL_RESULT with count=0. */
    namespace fs = std::filesystem;
    static std::atomic<int> counter{0};
    const fs::path cold = fs::temp_directory_path() /
                          ("chronosv_rhw_off_" +
                           std::to_string(counter.fetch_add(1)));
    fs::create_directories(cold);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code e; fs::remove_all(p, e); } } cleanup{cold};

    /* Phase A: create, ingest, flush, close. */
    {
        auto cfg = basic_config(4);
        std::string cold_str = cold.string();
        cfg.cold_path = cold_str.c_str();
        cfg.recover_hot_window = 0;   /* explicit: default behavior */
        EngineGuard g;
        g.h = chronosv_create(&cfg, nullptr);
        for (int i = 0; i < 5; ++i) {
            std::array<float, 4> v = {float(i), 1, 2, 3};
            chronosv_append(g.h, "s", i, v.data(), 4, nullptr);
        }
        chronosv_flush(g.h);
    }

    /* Phase B: reopen. Sensor "s" is registered, but the ring is empty
     * because default recovery skips rehydration. */
    {
        chronosv_engine_t* h = nullptr;
        REQUIRE(chronosv_open(cold.string().c_str(), &h) == CHRONOSV_OK);
        EngineGuard g;
        g.h = h;

        std::array<float, 4> q = {0, 1, 2, 3};
        std::array<std::int64_t, 5> ts{};
        std::array<float, 5>        sc{};
        int count = -1;
        const auto err = chronosv_query_nearest_n(g.h, "s", q.data(), 4, 5,
                                                  ts.data(), sc.data(), &count);
        REQUIRE(err == CHRONOSV_WARN_PARTIAL_RESULT);
        REQUIRE(count == 0);
    }
}

TEST_CASE("Phase 2: recover_hot_window=1 with recent data — rehydrates into ring",
          "[engine][phase2][recovery]") {
    /* Contract: with recover_hot_window=1 AND persisted entries fresh
     * enough (ts >= now - window_duration_ms), the ring is rehydrated and
     * queries return the persisted entries. */
    namespace fs = std::filesystem;
    static std::atomic<int> counter{0};
    const fs::path cold = fs::temp_directory_path() /
                          ("chronosv_rhw_on_" +
                           std::to_string(counter.fetch_add(1)));
    fs::create_directories(cold);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code e; fs::remove_all(p, e); } } cleanup{cold};

    /* Use current wall-clock timestamps so entries are inside the window
     * on reopen. */
    using namespace std::chrono;
    const std::int64_t now_ms =
        duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

    /* Phase A: create with recover_hot_window=1, wide window, ingest
     * "recent" entries, flush. */
    {
        auto cfg = basic_config(4);
        std::string cold_str = cold.string();
        cfg.cold_path = cold_str.c_str();
        cfg.recover_hot_window = 1;
        cfg.window_duration_ms = 24LL * 3600 * 1000;   /* 24 hours */
        EngineGuard g;
        g.h = chronosv_create(&cfg, nullptr);
        for (int i = 0; i < 5; ++i) {
            std::array<float, 4> v = {float(i), 1, 2, 3};
            chronosv_append(g.h, "s", now_ms + i, v.data(), 4, nullptr);
        }
        chronosv_flush(g.h);
    }

    /* Phase B: reopen — ring should be rehydrated from persisted blocks. */
    {
        chronosv_engine_t* h = nullptr;
        REQUIRE(chronosv_open(cold.string().c_str(), &h) == CHRONOSV_OK);
        EngineGuard g;
        g.h = h;

        std::array<float, 4> q = {0, 1, 2, 3};
        std::array<std::int64_t, 10> ts{};
        std::array<float, 10>        sc{};
        int count = 0;
        const auto err = chronosv_query_nearest_n(g.h, "s", q.data(), 4, 10,
                                                  ts.data(), sc.data(), &count);
        /* Should find our 5 recent entries (WARN_PARTIAL_RESULT since we asked for 10). */
        REQUIRE(err == CHRONOSV_WARN_PARTIAL_RESULT);
        REQUIRE(count == 5);
        /* Timestamps must match what we wrote. */
        std::vector<std::int64_t> got(ts.begin(), ts.begin() + count);
        std::sort(got.begin(), got.end());
        for (int i = 0; i < 5; ++i) REQUIRE(got[i] == now_ms + i);
    }
}

TEST_CASE("Phase 2: recover_hot_window=1 with stale data — entries filtered out",
          "[engine][phase2][recovery]") {
    /* Contract: entries whose original ts < (now - window_duration_ms)
     * are dropped during rehydration. A reopen after a long gap with only
     * old data yields empty rings even with recover_hot_window=1. */
    namespace fs = std::filesystem;
    static std::atomic<int> counter{0};
    const fs::path cold = fs::temp_directory_path() /
                          ("chronosv_rhw_stale_" +
                           std::to_string(counter.fetch_add(1)));
    fs::create_directories(cold);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code e; fs::remove_all(p, e); } } cleanup{cold};

    /* Phase A: write entries with timestamps FAR in the past. */
    {
        auto cfg = basic_config(4);
        std::string cold_str = cold.string();
        cfg.cold_path = cold_str.c_str();
        cfg.recover_hot_window = 1;
        cfg.window_duration_ms = 60'000;   /* 60 second window */
        EngineGuard g;
        g.h = chronosv_create(&cfg, nullptr);
        for (int i = 0; i < 5; ++i) {
            std::array<float, 4> v = {float(i), 1, 2, 3};
            /* ts = 1000 + i means these are from ~1970 — well outside any
             * "60 seconds ago" cutoff. */
            chronosv_append(g.h, "s", 1000 + i, v.data(), 4, nullptr);
        }
        chronosv_flush(g.h);
    }

    /* Phase B: reopen — cutoff is (now - 60s), all persisted entries
     * are much older, so rehydration filters them all out. Ring is empty. */
    {
        chronosv_engine_t* h = nullptr;
        REQUIRE(chronosv_open(cold.string().c_str(), &h) == CHRONOSV_OK);
        EngineGuard g;
        g.h = h;

        std::array<float, 4> q = {0, 1, 2, 3};
        std::array<std::int64_t, 5> ts{};
        std::array<float, 5>        sc{};
        int count = -1;
        const auto err = chronosv_query_nearest_n(g.h, "s", q.data(), 4, 5,
                                                  ts.data(), sc.data(), &count);
        REQUIRE(err == CHRONOSV_WARN_PARTIAL_RESULT);
        REQUIRE(count == 0);
    }
}

TEST_CASE("Phase 2: DropSensor removes persisted blocks too",
          "[engine][phase2][drop]") {
    namespace fs = std::filesystem;
    static std::atomic<int> counter{0};
    const fs::path cold = fs::temp_directory_path() /
                          ("chronosv_engine_drop_" +
                           std::to_string(counter.fetch_add(1)));
    fs::create_directories(cold);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code e; fs::remove_all(p, e); } } cleanup{cold};

    {
        auto cfg = basic_config(4);
        std::string cold_str = cold.string();
        cfg.cold_path = cold_str.c_str();
        EngineGuard g;
        g.h = chronosv_create(&cfg, nullptr);
        for (int i = 0; i < 5; ++i) {
            std::array<float, 4> v = {float(i), 0.f, 0.f, 0.f};
            chronosv_append(g.h, "sA", i, v.data(), 4, nullptr);
            chronosv_append(g.h, "sB", i, v.data(), 4, nullptr);
        }
        chronosv_flush(g.h);
        /* Drop sA — its persisted blocks should be gone; sB survives. */
        REQUIRE(chronosv_drop_sensor(g.h, "sA") == CHRONOSV_OK);
    }
    /* Reopen; only sB should be recovered. */
    {
        chronosv_engine_t* h = nullptr;
        REQUIRE(chronosv_open(cold.string().c_str(), &h) == CHRONOSV_OK);
        EngineGuard g;
        g.h = h;
        char* ids[10] = {};
        std::size_t count = 0;
        chronosv_list_sensors(g.h, ids, 10, &count);
        REQUIRE(count == 1);
        REQUIRE(std::string(ids[0]) == "sB");
        std::free(ids[0]);
    }
}

#ifdef CHRONOSV_ENABLE_INT8
TEST_CASE("Phase 1.5: INT8 engine end-to-end — quantize on Append, query returns close ranking",
          "[engine][int8]") {
    /* Contract: creating with storage_dtype=INT8 quantizes on Append and
     * dispatches to the INT8 kernels on Query. Results should match the
     * float32 engine's ranking closely — top-K identity + scores within
     * design tolerance. */
    auto cfg_f = basic_config(64);
    auto cfg_i = basic_config(64);
    cfg_i.storage_dtype = CHRONOSV_DTYPE_INT8;
    EngineGuard gf, gi;
    gf.h = chronosv_create(&cfg_f, nullptr);
    gi.h = chronosv_create(&cfg_i, nullptr);
    REQUIRE(gf.h != nullptr);
    REQUIRE(gi.h != nullptr);

    /* Ingest identical data into both engines. */
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    constexpr int kCount = 50;
    for (int i = 0; i < kCount; ++i) {
        std::array<float, 64> v{};
        for (auto& x : v) x = dist(rng);
        chronosv_append(gf.h, "s", i, v.data(), 64, nullptr);
        chronosv_append(gi.h, "s", i, v.data(), 64, nullptr);
    }

    /* Query with the same target. Compare top-5 IDs and scores. */
    std::array<float, 64> q{};
    for (auto& x : q) x = dist(rng);

    std::array<std::int64_t, 5> ts_f{}, ts_i{};
    std::array<float, 5>        sc_f{}, sc_i{};
    int nf = 0, ni = 0;
    REQUIRE(chronosv_query_nearest_n(gf.h, "s", q.data(), 64, 5,
                                     ts_f.data(), sc_f.data(), &nf) == CHRONOSV_OK);
    REQUIRE(chronosv_query_nearest_n(gi.h, "s", q.data(), 64, 5,
                                     ts_i.data(), sc_i.data(), &ni) == CHRONOSV_OK);
    REQUIRE(nf == 5);
    REQUIRE(ni == 5);

    /* Top-1 must match — quantization error should be small enough at dim=64
     * that the argmax is preserved. If this fails on random inputs, INT8
     * accuracy has regressed and is a real concern. */
    REQUIRE(ts_i[0] == ts_f[0]);

    /* Scores for top-1 should agree within the design tolerance (~2%). */
    REQUIRE(std::abs(sc_i[0] - sc_f[0]) < 0.03f);
}
#endif

TEST_CASE("Close stops the eviction thread; no crash on subsequent destroy",
          "[engine][eviction][lifecycle]") {
    auto cfg = basic_config(1);
    cfg.eviction_interval_ms = 100;
    cfg.window_duration_ms = 100;
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    // Let the thread spin at least once.
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    REQUIRE(chronosv_close(g.h) == CHRONOSV_OK);
    // Idempotent close.
    REQUIRE(chronosv_close(g.h) == CHRONOSV_OK);
    // Destroy runs via EngineGuard — must not deadlock or crash.
}
