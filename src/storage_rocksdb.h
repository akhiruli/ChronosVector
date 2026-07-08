/*
 * ChronosVector — RocksDB-backed StorageBackend implementation (internal).
 *
 * The shipping cold-tier backend for Phase 2. Uses RocksDB (system install
 * storage.
 *
 * NOT part of the public C ABI. Consumers only ever see the abstract
 * StorageBackend interface; the RocksDB dependency is entirely confined to
 * this translation unit + src/storage_rocksdb.cpp.
 *
 * <sensor_id>\x01<block_id_be_u64>
 *
 * Threading contract: matches StorageBackend base class — WriteBlock is
 * called only from the engine's eviction thread; ReadBlock / ListBlocks /
 * DeleteSensor / Flush may be called from any thread. RocksDB is thread-safe
 * for these operations.
 */
#ifndef CHRONOSV_STORAGE_ROCKSDB_H
#define CHRONOSV_STORAGE_ROCKSDB_H

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "chronosv/storage_backend.h"
#include "chronosv/types.h"

/* Forward-declare so we don't leak the RocksDB header into anyone that
 * includes this internal header. */
namespace rocksdb { class DB; }

namespace chronosv::internal {

class RocksDBStorageBackend final : public StorageBackend {
public:
    /* Open (or create) a RocksDB database at `path`. On failure, ok()
     * returns false and Status()/Message() explain why — the engine then
     * refuses to construct with this backend. */
    explicit RocksDBStorageBackend(std::string_view path) noexcept;
    ~RocksDBStorageBackend() override;

    RocksDBStorageBackend(const RocksDBStorageBackend&)            = delete;
    RocksDBStorageBackend& operator=(const RocksDBStorageBackend&) = delete;
    RocksDBStorageBackend(RocksDBStorageBackend&&)                 = delete;
    RocksDBStorageBackend& operator=(RocksDBStorageBackend&&)      = delete;

    /* True if the RocksDB open succeeded; ~false means construction failed
     * and the backend is unusable. */
    bool ok() const noexcept { return db_ != nullptr; }

    /* Human-readable status of the last failed open / operation, or "" if
     * everything's been fine. Not thread-safe; call only after ok() is
     * inspected during construction. */
    const std::string& last_error() const noexcept { return last_error_; }

    /* ---- StorageBackend interface --------------------------------------- */

    chronosv_error_t WriteBlock(const Block& blk) noexcept override;

    chronosv_error_t ReadBlock(std::string_view sensor_id,
                               std::int64_t block_id,
                               Block& out_blk,
                               std::int64_t* out_timestamps,
                               void*         out_vectors,
                               float*        out_scales,
                               std::uint8_t* out_payloads) noexcept override;

    chronosv_error_t ListBlocks(std::string_view sensor_id,
                                std::vector<std::int64_t>& out_ids) noexcept override;

    chronosv_error_t ListSensors(std::vector<std::string>& out_ids) noexcept override;

    chronosv_error_t DeleteSensor(std::string_view sensor_id) noexcept override;

    chronosv_error_t Flush() noexcept override;

    chronosv_error_t WriteMetadata(const std::uint8_t* data,
                                   std::size_t len) noexcept override;
    chronosv_error_t ReadMetadata(std::vector<std::uint8_t>& out) noexcept override;

    std::uint64_t ColdBytesEstimate() const noexcept override;

private:
    /* The rocksdb::DB* is heap-owned; we keep a unique_ptr with a custom
     * deleter because RocksDB's DB destructor is non-virtual and lives
     * behind Close(). */
    struct DbDeleter { void operator()(rocksdb::DB* db) const noexcept; };
    std::unique_ptr<rocksdb::DB, DbDeleter> db_;

    /* Reusable scratch for WriteBlock serialization. WriteBlock is
     * single-threaded per the interface contract (called only from the
     * eviction thread), so a per-instance buffer is safe. Grows only up. */
    std::vector<std::uint8_t> write_scratch_;

    std::string last_error_;
};

}  // namespace chronosv::internal

#endif  // CHRONOSV_STORAGE_ROCKSDB_H
