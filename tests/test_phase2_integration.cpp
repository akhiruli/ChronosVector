/*
 * Phase 2 integration tests — exercise multi-component flows that unit
 * tests can't cover in isolation.
 *
 * Coverage:
 *   1. Multi-producer + eviction under sustained load, 5-10 s per test.
 *      Verifies no crash, no data race under TSan (test is included in
 *      the TSan build), and stats reflect the actual work done.
 *   2. Concurrent flush from multiple threads — must be idempotent, no
 *      deadlock, no corruption.
 *   3. Corrupt-block-on-open — persist a bad block, chronosv_open still
 *      succeeds (recovery is best-effort).
 *
 * These tests use real RocksDB via cold_path — slower than pure-unit but
 * that's the whole point of "integration."
 */
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/version.h>

#include "chronosv/chronos_vector.h"

namespace {

/* Same TempDir helper we use in test_storage_rocksdb.cpp. Duplicated
 * intentionally to keep tests self-contained; the helper is tiny. */
class TempDir {
public:
    TempDir() {
        static std::atomic<int> counter{0};
        auto base = std::filesystem::temp_directory_path();
        std::random_device rd;
        std::string name = "chronosv_integ_" +
                           std::to_string(rd()) + "_" +
                           std::to_string(counter.fetch_add(1));
        path_ = base / name;
        std::filesystem::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    TempDir(const TempDir&)            = delete;
    TempDir& operator=(const TempDir&) = delete;
    std::string str() const { return path_.string(); }
private:
    std::filesystem::path path_;
};

struct EngineGuard {
    chronosv_engine_t* h = nullptr;
    ~EngineGuard() { chronosv_destroy(h); }
};

chronosv_config_t basic_config_cold(std::uint32_t dim, const char* cold_path) {
    chronosv_config_t c{};
    c.abi_version = CHRONOSV_ABI_VERSION;
    c.dim         = dim;
    c.cold_path   = cold_path;
    return c;
}

}  // namespace

/* ==========================================================================
 * (1) Multi-producer + eviction under sustained load
 * ========================================================================== */

TEST_CASE("Integration: N producers + eviction + cold-tier writes (5 s)",
          "[phase2][integration][concurrency]") {
    /* One producer per sensor per the SPSC contract. Each producer runs
     * its own thread appending as fast as possible. Meanwhile the
     * background eviction thread ages entries out and persists them.
     * Test succeeds if: no crash, all producers finished, cold_bytes > 0,
     * flush_errors == 0. Under TSan this also proves no data races. */
    TempDir dir;
    std::string cold_str = dir.str();   /* keep alive across chronosv_create */
    auto cfg = basic_config_cold(4, cold_str.c_str());
    cfg.window_duration_ms   = 200;    /* tight window forces eviction */
    cfg.eviction_interval_ms = 100;
    /* Capacity chosen so kAppendsEach < ring_capacity — under TSan, if the
     * producer wraps and overwrites a slot the eviction thread is scanning,
     * TSan (correctly) flags it as a race. This is the design-permitted
     * "producer laps consumer" scenario (§5.3), but this test's purpose is
     * to verify the write-path plumbing, not the wrap contract. Sized to
     * avoid wrap during the test window. */
    cfg.ring_capacity        = 32768;

    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    REQUIRE(g.h != nullptr);

    constexpr int    kNumProducers = 4;
    constexpr int    kAppendsEach  = 20000;
    std::atomic<int> completed{0};
    std::vector<std::thread> producers;
    producers.reserve(kNumProducers);
    for (int p = 0; p < kNumProducers; ++p) {
        producers.emplace_back([&, p] {
            const std::string sensor = "s" + std::to_string(p);
            std::array<float, 4> v = {1.0f, 2.0f, 3.0f, 4.0f};
            for (int i = 0; i < kAppendsEach; ++i) {
                v[0] = static_cast<float>(i);
                chronosv_append(g.h, sensor.c_str(), i,
                                v.data(), 4, nullptr);
            }
            completed.fetch_add(1);
        });
    }
    for (auto& t : producers) t.join();
    REQUIRE(completed.load() == kNumProducers);

    /* Force a final flush so cold_bytes is populated deterministically. */
    REQUIRE(chronosv_flush(g.h) == CHRONOSV_OK);

    chronosv_stats_t st{};
    REQUIRE(chronosv_get_stats(g.h, &st) == CHRONOSV_OK);
    REQUIRE(st.total_appends == static_cast<std::uint64_t>(kNumProducers) * kAppendsEach);
    REQUIRE(st.sensor_count == kNumProducers);
    REQUIRE(st.total_evictions >= 1);
    REQUIRE(st.flush_errors_total == 0);
    REQUIRE(st.cold_bytes_estimate > 0);
}

/* ==========================================================================
 * (2) Concurrent chronosv_flush from many threads
 * ========================================================================== */

TEST_CASE("Integration: concurrent chronosv_flush is idempotent (no deadlock)",
          "[phase2][integration][flush]") {
    TempDir dir;
    std::string cold_str = dir.str();
    auto cfg = basic_config_cold(4, cold_str.c_str());
    EngineGuard g;
    g.h = chronosv_create(&cfg, nullptr);
    REQUIRE(g.h != nullptr);

    /* Seed one sensor with something to flush. */
    std::array<float, 4> v = {1, 2, 3, 4};
    for (int i = 0; i < 100; ++i) {
        chronosv_append(g.h, "s", i, v.data(), 4, nullptr);
    }

    /* Multiple threads call flush concurrently. All should return OK. */
    constexpr int kFlushers = 8;
    std::atomic<int> ok_count{0};
    std::atomic<int> other_count{0};
    std::vector<std::thread> flushers;
    for (int i = 0; i < kFlushers; ++i) {
        flushers.emplace_back([&] {
            const auto err = chronosv_flush(g.h);
            if (err == CHRONOSV_OK) ok_count.fetch_add(1);
            else                    other_count.fetch_add(1);
        });
    }
    for (auto& t : flushers) t.join();

    /* All should succeed. Any other return code means we hit a real error. */
    REQUIRE(ok_count.load() == kFlushers);
    REQUIRE(other_count.load() == 0);

    chronosv_stats_t st{};
    chronosv_get_stats(g.h, &st);
    REQUIRE(st.flush_errors_total == 0);
}

/* ==========================================================================
 * (3) Corrupt-block-on-open — recovery is best-effort
 * ========================================================================== */

TEST_CASE("Integration: chronosv_open succeeds when a persisted block is corrupt",
          "[phase2][integration][recovery][corruption]") {
    TempDir dir;

    /* Phase A: create, ingest, flush so blocks are on disk. */
    {
        std::string cold_str = dir.str();
        auto cfg = basic_config_cold(4, cold_str.c_str());
        EngineGuard g;
        g.h = chronosv_create(&cfg, nullptr);
        REQUIRE(g.h != nullptr);
        std::array<float, 4> v = {1, 2, 3, 4};
        for (int i = 0; i < 20; ++i) {
            chronosv_append(g.h, "s", i, v.data(), 4, nullptr);
        }
        REQUIRE(chronosv_flush(g.h) == CHRONOSV_OK);
    }

    /* Phase B: reach into RocksDB directly and corrupt a value byte.
     * Explicitly reset iterators BEFORE Close (RocksDB requires no live
     * iterators / snapshots when Close is called). Also let unique_ptr
     * destroy the DB naturally rather than double-calling Close. */
    {
        rocksdb::Options opts;
        opts.create_if_missing = false;
        rocksdb::DB* raw = nullptr;
#if defined(ROCKSDB_MAJOR) && ROCKSDB_MAJOR >= 10
        std::unique_ptr<rocksdb::DB> tmp;
        REQUIRE(rocksdb::DB::Open(opts, dir.str(), &tmp).ok());
        raw = tmp.release();
#else
        REQUIRE(rocksdb::DB::Open(opts, dir.str(), &raw).ok());
#endif
        std::unique_ptr<rocksdb::DB> owned(raw);
        {
            /* Find any sensor block key (key = "s" + \x01 + 8-byte-be block_id).
             * Scope the iterator so it's destroyed BEFORE the DB. */
            std::unique_ptr<rocksdb::Iterator> it(owned->NewIterator(rocksdb::ReadOptions{}));
            for (it->SeekToFirst(); it->Valid(); it->Next()) {
                const rocksdb::Slice k = it->key();
                /* Skip the metadata key (first byte is \x00). */
                if (k.size() > 0 && k.data()[0] == '\x00') continue;
                std::string val = it->value().ToString();
                if (val.size() < 50) continue;   /* need enough bytes to safely stomp */
                val[val.size() / 2] ^= 0x5A;     /* corrupt the middle */
                REQUIRE(owned->Put(rocksdb::WriteOptions{}, k, val).ok());
                break;
            }
        }
        /* Flush the WAL so the corrupted value is persisted before we close. */
        REQUIRE(owned->FlushWAL(/*sync=*/true).ok());
        /* Note: NOT calling owned->Close() explicitly — unique_ptr's
         * destructor calls RocksDB's DB destructor which handles close
         * cleanly. Double-close via explicit Close() then destructor has
         * caused issues in some RocksDB versions. */
    }

    /* Phase C: chronosv_open must still succeed. Recovery is best-effort:
     * the corrupt block is skipped but the sensor is registered and other
     * blocks (if any) are readable. */
    {
        chronosv_engine_t* h = nullptr;
        REQUIRE(chronosv_open(dir.str().c_str(), &h) == CHRONOSV_OK);
        REQUIRE(h != nullptr);
        EngineGuard g;
        g.h = h;

        /* Sensor "s" should still be listed. */
        char* ids[10] = {};
        std::size_t count = 0;
        REQUIRE(chronosv_list_sensors(g.h, ids, 10, &count) == CHRONOSV_OK);
        REQUIRE(count == 1);
        REQUIRE(std::string(ids[0]) == "s");
        std::free(ids[0]);
    }
}
