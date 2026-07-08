/*
 * Storage-backend interface smoke test.
 *
 * Real end-to-end coverage lives in test_storage_rocksdb.cpp; this file
 * exercises only the abstract-interface seam that lets alternate backends
 * be swapped in. These tests prove:
 *   1. The abstract class + Block POD compile and can be included from
 *      external translation units without pulling in library internals.
 *   2. NullStorageBackend can be instantiated and all methods return their
 *      documented no-op values.
 *   3. Consumers can implement a custom backend by subclassing — this is
 *      the pattern the commercial layer will use.
 */
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "chronosv/storage_backend.h"
#include "chronosv/types.h"

using chronosv::internal::Block;
using chronosv::internal::NullStorageBackend;
using chronosv::internal::StorageBackend;

TEST_CASE("NullStorageBackend: WriteBlock always succeeds",
          "[storage_backend][null]") {
    NullStorageBackend backend;
    Block b{};
    b.sensor_id          = "s1";
    b.block_id           = 0;
    b.count              = 0;
    b.dim                = 4;
    b.dtype              = CHRONOSV_DTYPE_FLOAT32;
    b.payload_size_bytes = 0;
    REQUIRE(backend.WriteBlock(b) == CHRONOSV_OK);
}

TEST_CASE("NullStorageBackend: ReadBlock returns NOT_FOUND",
          "[storage_backend][null]") {
    NullStorageBackend backend;
    Block out{};
    REQUIRE(backend.ReadBlock(/*sensor=*/"s1", /*block_id=*/0, out,
                              nullptr, nullptr, nullptr, nullptr)
            == CHRONOSV_ERR_NOT_FOUND);
}

TEST_CASE("NullStorageBackend: ListBlocks yields empty",
          "[storage_backend][null]") {
    NullStorageBackend backend;
    std::vector<std::int64_t> ids;
    ids.push_back(42);  // ensure ListBlocks clears
    REQUIRE(backend.ListBlocks("s1", ids) == CHRONOSV_OK);
    REQUIRE(ids.empty());
}

TEST_CASE("NullStorageBackend: DeleteSensor + Flush are no-op successful",
          "[storage_backend][null]") {
    NullStorageBackend backend;
    REQUIRE(backend.DeleteSensor("s1") == CHRONOSV_OK);
    REQUIRE(backend.Flush()             == CHRONOSV_OK);
    REQUIRE(backend.ColdBytesEstimate() == 0);
}

TEST_CASE("NullStorageBackend: ListSensors yields empty",
          "[storage_backend][null]") {
    NullStorageBackend backend;
    std::vector<std::string> ids;
    ids.emplace_back("stale");
    REQUIRE(backend.ListSensors(ids) == CHRONOSV_OK);
    REQUIRE(ids.empty());
}

TEST_CASE("NullStorageBackend: WriteMetadata is no-op, ReadMetadata is NOT_FOUND",
          "[storage_backend][null][metadata]") {
    NullStorageBackend backend;
    std::uint8_t data[4] = {1, 2, 3, 4};
    REQUIRE(backend.WriteMetadata(data, sizeof(data)) == CHRONOSV_OK);
    std::vector<std::uint8_t> out;
    out.emplace_back(9);
    REQUIRE(backend.ReadMetadata(out) == CHRONOSV_ERR_NOT_FOUND);
    REQUIRE(out.empty());
}

/* -------------------------------------------------------------------------- *
 * Custom backend example — proves the vtable + polymorphic use work.
 * -------------------------------------------------------------------------- */

class CountingBackend final : public StorageBackend {
public:
    std::atomic<int> writes{0};
    std::atomic<int> flushes{0};
    std::atomic<std::int64_t> bytes_estimate{0};

    chronosv_error_t WriteBlock(const Block& blk) noexcept override {
        writes.fetch_add(1);
        // Estimate: header (~48 B) + timestamps (8 × count) + vectors + CRC.
        bytes_estimate.fetch_add(
            48 + blk.count * (8 + blk.dim * 4) + 4);
        return CHRONOSV_OK;
    }
    chronosv_error_t ReadBlock(std::string_view, std::int64_t, Block&,
                               std::int64_t*, void*, float*,
                               std::uint8_t*) noexcept override {
        return CHRONOSV_ERR_NOT_FOUND;
    }
    chronosv_error_t ListBlocks(std::string_view,
                                std::vector<std::int64_t>& out) noexcept override {
        out.clear();
        return CHRONOSV_OK;
    }
    chronosv_error_t ListSensors(std::vector<std::string>& out) noexcept override {
        out.clear();
        return CHRONOSV_OK;
    }
    chronosv_error_t DeleteSensor(std::string_view) noexcept override {
        return CHRONOSV_OK;
    }
    chronosv_error_t Flush() noexcept override {
        flushes.fetch_add(1);
        return CHRONOSV_OK;
    }
    chronosv_error_t WriteMetadata(const std::uint8_t*, std::size_t) noexcept override {
        return CHRONOSV_OK;
    }
    chronosv_error_t ReadMetadata(std::vector<std::uint8_t>& out) noexcept override {
        out.clear();
        return CHRONOSV_ERR_NOT_FOUND;
    }
    std::uint64_t ColdBytesEstimate() const noexcept override {
        return static_cast<std::uint64_t>(bytes_estimate.load());
    }
};

TEST_CASE("Custom StorageBackend subclass: polymorphic dispatch works",
          "[storage_backend][custom]") {
    std::unique_ptr<StorageBackend> b = std::make_unique<CountingBackend>();
    Block blk{};
    blk.sensor_id = "s";
    blk.count     = 100;
    blk.dim       = 128;
    blk.dtype     = CHRONOSV_DTYPE_FLOAT32;
    REQUIRE(b->WriteBlock(blk) == CHRONOSV_OK);
    REQUIRE(b->WriteBlock(blk) == CHRONOSV_OK);
    REQUIRE(b->Flush()          == CHRONOSV_OK);
    REQUIRE(b->ColdBytesEstimate() > 0);

    auto* cb = static_cast<CountingBackend*>(b.get());
    REQUIRE(cb->writes.load()  == 2);
    REQUIRE(cb->flushes.load() == 1);
}
