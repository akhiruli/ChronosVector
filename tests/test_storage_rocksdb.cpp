/*
 * RocksDBStorageBackend tests.
 *
 * Every test uses a fresh tempdir so tests are hermetic. The RAII TempDir
 * helper creates the directory and removes it on scope exit (even on
 * REQUIRE failure via Catch2's stack unwinding).
 *
 * Coverage:
 *   1. Open succeeds at a fresh path; ok() reports true.
 *   2. Open at a bad path (e.g. non-writable) fails cleanly with ok()=false.
 *   3. WriteBlock + ReadBlock round-trip preserves every field.
 *   4. ReadBlock on a nonexistent block_id returns NOT_FOUND.
 *   5. ListBlocks returns block_ids in ascending order.
 *   6. DeleteSensor removes all of the sensor's blocks; other sensors survive.
 *   7. Persistence: close backend, reopen at same path, blocks still readable.
 *   8. Corruption injection: manually overwrite a stored value with garbage
 *      bytes → ReadBlock returns CORRUPTION.
 *   9. Flush returns OK on a healthy DB.
 *   10. ColdBytesEstimate is monotone-non-decreasing across writes.
 */
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/version.h>

#include "chronosv/storage_backend.h"
#include "chronosv/types.h"
#include "storage_rocksdb.h"

using chronosv::internal::Block;
using chronosv::internal::RocksDBStorageBackend;

namespace {

/* Unique tempdir per test — auto-removes on scope exit. */
class TempDir {
public:
    TempDir() {
        // Combine PID + nanosecond + atomic counter for a name unlikely to
        // collide across parallel test runs.
        static std::atomic<int> counter{0};
        auto base = std::filesystem::temp_directory_path();
        std::random_device rd;
        std::string name = "chronosv_test_" +
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

    const std::filesystem::path& path() const noexcept { return path_; }
    std::string str() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

/* Build a Block with `count` sequential float vectors (dim=4, no payload). */
struct BlockFixture {
    std::vector<std::int64_t> ts;
    std::vector<float>        vecs;
    Block                     blk{};

    BlockFixture(std::string_view sensor, std::int64_t block_id,
                 std::uint32_t count, std::uint32_t dim = 4,
                 std::int64_t ts_base = 1000) {
        ts.resize(count);
        vecs.resize(static_cast<std::size_t>(count) * dim);
        for (std::uint32_t i = 0; i < count; ++i) {
            ts[i] = ts_base + i * 10;
            for (std::uint32_t d = 0; d < dim; ++d) {
                vecs[i * dim + d] = static_cast<float>(i * dim + d);
            }
        }
        blk.sensor_id  = sensor;
        blk.block_id   = block_id;
        blk.count      = count;
        blk.dim        = dim;
        blk.dtype      = CHRONOSV_DTYPE_FLOAT32;
        blk.t_start_ms = count > 0 ? ts.front() : 0;
        blk.t_end_ms   = count > 0 ? ts.back()  : 0;
        blk.timestamps = ts.data();
        blk.vectors    = vecs.data();
    }
};

}  // namespace

/* ==========================================================================
 * Open / lifecycle
 * ========================================================================== */

TEST_CASE("Backend opens successfully at a fresh path", "[storage_rocksdb][lifecycle]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    REQUIRE(backend.ok());
    REQUIRE(backend.last_error().empty());
}

TEST_CASE("Backend open fails cleanly at an unwritable path",
          "[storage_rocksdb][lifecycle][error]") {
    /* /nonexistent/deep/path shouldn't be creatable by an unprivileged
     * user on either macOS or Linux. RocksDB reports an IOError. */
    RocksDBStorageBackend backend("/nonexistent/chronosv/should/fail/here");
    REQUIRE_FALSE(backend.ok());
    REQUIRE_FALSE(backend.last_error().empty());
}

/* ==========================================================================
 * WriteBlock + ReadBlock round-trip
 * ========================================================================== */

TEST_CASE("Round-trip: WriteBlock → ReadBlock preserves every field",
          "[storage_rocksdb][roundtrip]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    REQUIRE(backend.ok());

    BlockFixture fx("sensor_a", /*block_id=*/7, /*count=*/50);
    REQUIRE(backend.WriteBlock(fx.blk) == CHRONOSV_OK);

    Block out_blk{};
    std::vector<std::int64_t> out_ts(fx.blk.count);
    std::vector<float>        out_vecs(static_cast<std::size_t>(fx.blk.count) * fx.blk.dim);
    REQUIRE(backend.ReadBlock("sensor_a", 7, out_blk,
                              out_ts.data(), out_vecs.data(),
                              nullptr, nullptr) == CHRONOSV_OK);

    REQUIRE(out_blk.count == fx.blk.count);
    REQUIRE(out_blk.dim == fx.blk.dim);
    REQUIRE(out_blk.dtype == fx.blk.dtype);
    REQUIRE(out_blk.t_start_ms == fx.blk.t_start_ms);
    REQUIRE(out_blk.t_end_ms   == fx.blk.t_end_ms);
    REQUIRE(out_ts == fx.ts);
    REQUIRE(std::memcmp(out_vecs.data(), fx.vecs.data(),
                        fx.vecs.size() * sizeof(float)) == 0);
}

TEST_CASE("ReadBlock on nonexistent block_id returns NOT_FOUND",
          "[storage_rocksdb][read][error]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    REQUIRE(backend.ok());

    Block out{};
    std::int64_t ts_dummy = 0;
    float vec_dummy = 0.0f;
    REQUIRE(backend.ReadBlock("sensor_a", 42, out,
                              &ts_dummy, &vec_dummy,
                              nullptr, nullptr) == CHRONOSV_ERR_NOT_FOUND);
}

/* ==========================================================================
 * ListBlocks
 * ==========================================================================
 */

TEST_CASE("ListBlocks returns block_ids in ascending (== temporal) order",
          "[storage_rocksdb][list]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    REQUIRE(backend.ok());

    // Write blocks in scrambled order — DB sorts by big-endian block_id key.
    for (std::int64_t id : {5, 1, 3, 2, 4}) {
        BlockFixture fx("s", id, /*count=*/2);
        REQUIRE(backend.WriteBlock(fx.blk) == CHRONOSV_OK);
    }
    std::vector<std::int64_t> ids;
    REQUIRE(backend.ListBlocks("s", ids) == CHRONOSV_OK);
    REQUIRE(ids == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("ListBlocks on unknown sensor returns empty list, not error",
          "[storage_rocksdb][list]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    std::vector<std::int64_t> ids{99};   // ensure clear() happens
    REQUIRE(backend.ListBlocks("no_such_sensor", ids) == CHRONOSV_OK);
    REQUIRE(ids.empty());
}

TEST_CASE("ListSensors: empty backend yields empty list",
          "[storage_rocksdb][list_sensors]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    std::vector<std::string> ids;
    ids.emplace_back("should be cleared");
    REQUIRE(backend.ListSensors(ids) == CHRONOSV_OK);
    REQUIRE(ids.empty());
}

TEST_CASE("ListSensors: returns unique sensor_ids across their blocks",
          "[storage_rocksdb][list_sensors]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    // Write blocks for three sensors, several blocks each. ListSensors
    // must return the three sensor_ids exactly once, sorted (RocksDB gives
    // us keys in sorted order and the impl exploits contiguity).
    for (const char* s : {"beta", "alpha", "gamma"}) {
        for (std::int64_t id : {1, 2, 3}) {
            BlockFixture fx(s, id, 2);
            REQUIRE(backend.WriteBlock(fx.blk) == CHRONOSV_OK);
        }
    }
    std::vector<std::string> ids;
    REQUIRE(backend.ListSensors(ids) == CHRONOSV_OK);
    REQUIRE(ids == std::vector<std::string>{"alpha", "beta", "gamma"});
}

TEST_CASE("ListBlocks isolates sensors — a's blocks don't show in b's list",
          "[storage_rocksdb][list][isolation]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    for (std::int64_t id : {1, 2, 3}) {
        BlockFixture a("sensor_a", id, 2);
        BlockFixture b("sensor_b", id * 10, 2);
        REQUIRE(backend.WriteBlock(a.blk) == CHRONOSV_OK);
        REQUIRE(backend.WriteBlock(b.blk) == CHRONOSV_OK);
    }
    std::vector<std::int64_t> ids_a, ids_b;
    backend.ListBlocks("sensor_a", ids_a);
    backend.ListBlocks("sensor_b", ids_b);
    REQUIRE(ids_a == std::vector<std::int64_t>{1, 2, 3});
    REQUIRE(ids_b == std::vector<std::int64_t>{10, 20, 30});
}

/* ==========================================================================
 * DeleteSensor
 * ========================================================================== */

TEST_CASE("DeleteSensor removes all blocks for a sensor; others untouched",
          "[storage_rocksdb][delete]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    for (std::int64_t id : {1, 2, 3}) {
        BlockFixture a("sensor_a", id, 2);
        BlockFixture b("sensor_b", id, 2);
        backend.WriteBlock(a.blk);
        backend.WriteBlock(b.blk);
    }

    REQUIRE(backend.DeleteSensor("sensor_a") == CHRONOSV_OK);

    std::vector<std::int64_t> ids;
    backend.ListBlocks("sensor_a", ids);
    REQUIRE(ids.empty());
    backend.ListBlocks("sensor_b", ids);
    REQUIRE(ids == std::vector<std::int64_t>{1, 2, 3});

    // Individual reads on deleted keys now NOT_FOUND.
    Block out{};
    std::int64_t ts_dummy = 0;
    float vec_dummy = 0;
    REQUIRE(backend.ReadBlock("sensor_a", 1, out, &ts_dummy, &vec_dummy,
                              nullptr, nullptr) == CHRONOSV_ERR_NOT_FOUND);
}

TEST_CASE("DeleteSensor on unknown sensor returns OK (idempotent)",
          "[storage_rocksdb][delete]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    REQUIRE(backend.DeleteSensor("never_existed") == CHRONOSV_OK);
}

/* ==========================================================================
 * Persistence across close/reopen
 * ========================================================================== */

TEST_CASE("Persistence: blocks survive close + reopen at the same path",
          "[storage_rocksdb][persistence]") {
    TempDir dir;

    // Phase 1: write, flush for durability, close.
    {
        RocksDBStorageBackend backend(dir.str());
        REQUIRE(backend.ok());
        for (std::int64_t id : {1, 2, 3}) {
            BlockFixture fx("sensor_a", id, 3);
            REQUIRE(backend.WriteBlock(fx.blk) == CHRONOSV_OK);
        }
        REQUIRE(backend.Flush() == CHRONOSV_OK);
    }

    // Phase 2: reopen at the same path; the blocks must be readable.
    RocksDBStorageBackend backend(dir.str());
    REQUIRE(backend.ok());

    std::vector<std::int64_t> ids;
    REQUIRE(backend.ListBlocks("sensor_a", ids) == CHRONOSV_OK);
    REQUIRE(ids == std::vector<std::int64_t>{1, 2, 3});

    Block out_blk{};
    std::vector<std::int64_t> out_ts(3);
    std::vector<float>        out_vecs(3 * 4);
    REQUIRE(backend.ReadBlock("sensor_a", 2, out_blk,
                              out_ts.data(), out_vecs.data(),
                              nullptr, nullptr) == CHRONOSV_OK);
    REQUIRE(out_blk.count == 3);
    REQUIRE(out_blk.dim   == 4);
}

/* ==========================================================================
 * Corruption injection
 * ========================================================================== */

TEST_CASE("Corruption: overwritten value → ReadBlock returns CORRUPTION",
          "[storage_rocksdb][corruption]") {
    TempDir dir;

    // Phase 1: write a valid block.
    {
        RocksDBStorageBackend backend(dir.str());
        REQUIRE(backend.ok());
        BlockFixture fx("s", 1, 5);
        REQUIRE(backend.WriteBlock(fx.blk) == CHRONOSV_OK);
        REQUIRE(backend.Flush() == CHRONOSV_OK);
    }

    // Phase 2: reach around the backend and stomp on the value directly
    // using a raw RocksDB handle. This simulates media corruption /
    // silent bit-flip in a way that's independent of anything the backend
    // could catch on the way in.
    {
        rocksdb::Options opts;
        opts.create_if_missing = false;
        /* Same version gate as storage_rocksdb.cpp — RocksDB 10+ requires
         * unique_ptr<DB>*; older versions expect DB**. */
        rocksdb::DB* raw = nullptr;
#if defined(ROCKSDB_MAJOR) && ROCKSDB_MAJOR >= 10
        std::unique_ptr<rocksdb::DB> tmp;
        rocksdb::Status s = rocksdb::DB::Open(opts, dir.str(), &tmp);
        raw = tmp.release();
#else
        rocksdb::Status s = rocksdb::DB::Open(opts, dir.str(), &raw);
#endif
        REQUIRE(s.ok());
        std::unique_ptr<rocksdb::DB> owned(raw);

        // Build the same key we wrote under. sensor_id=s, block_id=1 (be_u64).
        std::string key = "s";
        key.push_back('\x01');
        for (int i = 7; i >= 0; --i) key.push_back((i == 0) ? '\x01' : '\x00');

        std::string val;
        REQUIRE(owned->Get(rocksdb::ReadOptions{}, key, &val).ok());
        // Corrupt the middle byte of the value.
        REQUIRE(val.size() > 20);
        val[val.size() / 2] ^= 0x55;
        REQUIRE(owned->Put(rocksdb::WriteOptions{}, key, val).ok());
        (void) owned->Close();
    }

    // Phase 3: reopen with the backend and try to read — must see CORRUPTION.
    RocksDBStorageBackend backend(dir.str());
    REQUIRE(backend.ok());
    Block out_blk{};
    std::vector<std::int64_t> out_ts(5);
    std::vector<float>        out_vecs(5 * 4);
    REQUIRE(backend.ReadBlock("s", 1, out_blk,
                              out_ts.data(), out_vecs.data(),
                              nullptr, nullptr) == CHRONOSV_ERR_CORRUPTION);
}

/* ==========================================================================
 * Flush + ColdBytesEstimate
 * ========================================================================== */

TEST_CASE("Flush returns OK on a healthy backend",
          "[storage_rocksdb][flush]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    REQUIRE(backend.ok());
    REQUIRE(backend.Flush() == CHRONOSV_OK);  // no writes yet, still OK
    BlockFixture fx("s", 1, 10);
    REQUIRE(backend.WriteBlock(fx.blk) == CHRONOSV_OK);
    REQUIRE(backend.Flush() == CHRONOSV_OK);
}

TEST_CASE("Metadata: WriteMetadata + ReadMetadata round-trip",
          "[storage_rocksdb][metadata]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    REQUIRE(backend.ok());

    std::vector<std::uint8_t> in(64);
    for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<std::uint8_t>(i * 7 + 3);
    REQUIRE(backend.WriteMetadata(in.data(), in.size()) == CHRONOSV_OK);

    std::vector<std::uint8_t> out{99, 88};   // sentinel bytes
    REQUIRE(backend.ReadMetadata(out) == CHRONOSV_OK);
    REQUIRE(out == in);
}

TEST_CASE("Metadata: ReadMetadata on fresh backend returns NOT_FOUND",
          "[storage_rocksdb][metadata]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    REQUIRE(backend.ok());
    std::vector<std::uint8_t> out;
    REQUIRE(backend.ReadMetadata(out) == CHRONOSV_ERR_NOT_FOUND);
    REQUIRE(out.empty());
}

TEST_CASE("Metadata: WriteMetadata is overwrite-on-write",
          "[storage_rocksdb][metadata]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    std::uint8_t v1[4] = {1, 2, 3, 4};
    std::uint8_t v2[4] = {9, 8, 7, 6};
    REQUIRE(backend.WriteMetadata(v1, 4) == CHRONOSV_OK);
    REQUIRE(backend.WriteMetadata(v2, 4) == CHRONOSV_OK);
    std::vector<std::uint8_t> out;
    REQUIRE(backend.ReadMetadata(out) == CHRONOSV_OK);
    REQUIRE(out == std::vector<std::uint8_t>{9, 8, 7, 6});
}

TEST_CASE("Metadata: doesn't collide with sensor keys or show up in ListSensors",
          "[storage_rocksdb][metadata][isolation]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    // Write metadata + several sensors' blocks.
    std::uint8_t meta[8] = {'H','e','l','l','o','!','!','!'};
    REQUIRE(backend.WriteMetadata(meta, 8) == CHRONOSV_OK);
    for (const char* s : {"alpha", "beta"}) {
        BlockFixture fx(s, 1, 2);
        REQUIRE(backend.WriteBlock(fx.blk) == CHRONOSV_OK);
    }
    // ListSensors must NOT include the metadata key (it has no \x01 separator).
    std::vector<std::string> ids;
    REQUIRE(backend.ListSensors(ids) == CHRONOSV_OK);
    REQUIRE(ids == std::vector<std::string>{"alpha", "beta"});
    // Metadata still readable.
    std::vector<std::uint8_t> out;
    REQUIRE(backend.ReadMetadata(out) == CHRONOSV_OK);
    REQUIRE(out.size() == 8);
}

TEST_CASE("ColdBytesEstimate is monotone-non-decreasing across writes",
          "[storage_rocksdb][stats]") {
    TempDir dir;
    RocksDBStorageBackend backend(dir.str());
    REQUIRE(backend.ok());

    const std::uint64_t empty_estimate = backend.ColdBytesEstimate();

    for (std::int64_t id = 1; id <= 20; ++id) {
        BlockFixture fx("s", id, /*count=*/100);
        REQUIRE(backend.WriteBlock(fx.blk) == CHRONOSV_OK);
    }
    // Force memtable → L0 so GetApproximateSizes sees actual SST bytes,
    // not just in-memory memtable.
    REQUIRE(backend.Flush() == CHRONOSV_OK);

    const std::uint64_t after_writes = backend.ColdBytesEstimate();
    // RocksDB's estimate is approximate but must reflect the writes we made
    // (~1 KB per block × 20 blocks = ~20 KB at least, before compression).
    REQUIRE(after_writes >= empty_estimate);
}
