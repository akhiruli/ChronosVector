/*
 * ChronosVector — cold-tier StorageBackend interface (internal).
 *
 * NOT part of the public C ABI. Internal C++ seam used by the engine's
 * eviction path to persist aged-out data. The interface is deliberately
 * kept abstract even though only one implementation ships — this seam
 * lets alternate backends (encrypted-at-rest, S3-compatible, managed
 * cloud) swap in without forking the core.
 *
 * Wiring:
 *   - RocksDBStorageBackend is the shipped default
 *   - Engine holds a std::unique_ptr<StorageBackend>
 *   - EvictSensor: snapshot-copy the evicted range, hand to WriteBlock,
 *     THEN advance the ring tail (race-safety — the copy must complete
 *     before consumers observe the tail advance)
 */
#ifndef CHRONOSV_STORAGE_BACKEND_H
#define CHRONOSV_STORAGE_BACKEND_H

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <string_view>
#include <vector>

#include "chronosv/types.h"

namespace chronosv::internal {

/* -------------------------------------------------------------------------- */
/* Block — a borrowed view over an evicted range from one sensor.             */
/* -------------------------------------------------------------------------- */

/* Handed to StorageBackend::WriteBlock. Pointers are BORROWED from the
 * eviction-thread's scratch buffer; the backend must copy anything it wants
 * to persist. This is a zero-copy hand-off — the backend controls when data
 * hits disk / network / wherever.
 *
 * The physical layout the backend should write is:
 *   header(magic, version, flags, dim, count, t_start_ms, t_end_ms,
 *          payload_size_bytes, reserved)
 *   timestamps[count]
 *   vectors[count * dim * dtype_size(dtype)]
 *   scales[count]                     -- only if dtype == INT8
 *   payloads[count * payload_size]    -- only if payload_size > 0
 *   crc32(everything above)
 *
 * The Block struct below carries the raw pointers; the backend is
 * responsible for serialization. This decoupling keeps the format decision
 * inside the storage layer where it belongs. */
struct Block {
    /* Identity */
    std::string_view sensor_id;
    std::int64_t     block_id;                   /* monotonic per-sensor counter */

    /* Content bounds */
    std::int64_t     t_start_ms;                 /* min timestamp in this block */
    std::int64_t     t_end_ms;                   /* max timestamp in this block */
    std::uint32_t    count;                      /* number of entries */

    /* Schema (matches the engine's config) */
    std::uint32_t    dim;
    std::uint32_t    payload_size_bytes;         /* 0 = no payloads */
    chronosv_dtype_t dtype;                      /* FLOAT32 or INT8 */

    /* Borrowed pointers — valid only during the WriteBlock call. */
    const std::int64_t*  timestamps;             /* [count] */
    const void*          vectors;                /* [count * dim * dtype_size] */
    const float*         scales;                 /* [count], only if dtype == INT8; else nullptr */
    const std::uint8_t*  payloads;               /* [count * payload_size_bytes] or nullptr */
};

/* -------------------------------------------------------------------------- */
/* StorageBackend — abstract cold-tier interface.                              */
/* -------------------------------------------------------------------------- */

/* Every method returns chronosv_error_t (matches the engine's error model).
 * OK == 0, negative == error; warnings are not expected from storage
 * operations in Phase 2's initial scope.
 *
 * Threading contract: WriteBlock is called ONLY from the engine's eviction
 * thread — implementations may assume single-threaded writes. ReadBlock,
 * ListBlocks, and DeleteSensor are called from the engine's Open (recovery),
 * DropSensor, and Close paths respectively. Implementations should be safe
 * against concurrent reads from any thread. Flush is called from
 * chronosv_flush (Phase 2), which is caller-initiated and may run on any
 * thread. */
class StorageBackend {
public:
    virtual ~StorageBackend() = default;

    /* Persist the block. Must copy any data from Block's borrowed pointers
     * that the backend wants to retain — the buffers are freed / reused
     * after this call returns. */
    virtual chronosv_error_t WriteBlock(const Block& blk) noexcept = 0;

    /* Read a specific (sensor_id, block_id) block into caller-owned buffers.
     * The caller supplies pre-allocated timestamps / vectors / scales /
     * payloads buffers sized for at least `blk.count` entries. `blk` is
     * populated with the actual content bounds and count on success. */
    virtual chronosv_error_t ReadBlock(std::string_view sensor_id,
                                       std::int64_t block_id,
                                       Block& out_blk,
                                       std::int64_t* out_timestamps,
                                       void*         out_vectors,
                                       float*        out_scales,
                                       std::uint8_t* out_payloads) noexcept = 0;

    /* Enumerate block_ids for a sensor, sorted ascending (== temporal
     * order since block_id is monotonic per sensor). */
    virtual chronosv_error_t ListBlocks(std::string_view sensor_id,
                                        std::vector<std::int64_t>& out_ids) noexcept = 0;

    /* Enumerate all sensor_ids that have persisted blocks. Order is
     * implementation-defined (RocksDB backend returns them sorted). Called
     * by chronosv_open recovery to reconstruct the sensor map. Empty
     * output + CHRONOSV_OK is the valid result for a fresh backend. */
    virtual chronosv_error_t ListSensors(std::vector<std::string>& out_ids) noexcept = 0;

    /* Remove all persisted state for a sensor. Called from
     * chronosv_drop_sensor. Idempotent — must not error if the sensor has
     * no persisted blocks. */
    virtual chronosv_error_t DeleteSensor(std::string_view sensor_id) noexcept = 0;

    /* Block until any in-flight writes are durable (fsync, WAL flush, etc.).
     * Called from chronosv_flush. */
    virtual chronosv_error_t Flush() noexcept = 0;

    /* Engine-level metadata blob (~50-60 bytes typically). Persisted at
     * a reserved key that cannot collide with any sensor block key.
     * Called by chronosv_create (once, at engine creation) and by
     * chronosv_open (to recover the schema).
     *
     * ReadMetadata returns CHRONOSV_ERR_NOT_FOUND if no metadata has been
     * written yet — Open uses this as the "legacy path, use defaults"
     * signal. WriteMetadata overwrites any existing metadata unconditionally. */
    virtual chronosv_error_t WriteMetadata(const std::uint8_t* data,
                                           std::size_t len) noexcept = 0;
    virtual chronosv_error_t ReadMetadata(std::vector<std::uint8_t>& out) noexcept = 0;

    /* Optional: report cold-tier bytes used, or 0 if the backend can't
     * cheaply compute this. Read by chronosv_get_stats. */
    virtual std::uint64_t ColdBytesEstimate() const noexcept { return 0; }
};

/* -------------------------------------------------------------------------- */
/* NullStorageBackend — no-op default.                                         */
/* -------------------------------------------------------------------------- */

/* Every operation succeeds; nothing is persisted. Used by Phase 1 (where
 * eviction discards) and by tests that don't want a real cold tier. */
class NullStorageBackend final : public StorageBackend {
public:
    chronosv_error_t WriteBlock(const Block&) noexcept override {
        return CHRONOSV_OK;
    }
    chronosv_error_t ReadBlock(std::string_view,
                               std::int64_t,
                               Block&,
                               std::int64_t*,
                               void*,
                               float*,
                               std::uint8_t*) noexcept override {
        // No blocks exist in the null backend; reading any specific id
        // is a valid lookup that returns not-found.
        return CHRONOSV_ERR_NOT_FOUND;
    }
    chronosv_error_t ListBlocks(std::string_view,
                                std::vector<std::int64_t>& out_ids) noexcept override {
        out_ids.clear();
        return CHRONOSV_OK;
    }
    chronosv_error_t ListSensors(std::vector<std::string>& out_ids) noexcept override {
        out_ids.clear();
        return CHRONOSV_OK;
    }
    chronosv_error_t DeleteSensor(std::string_view) noexcept override {
        return CHRONOSV_OK;
    }
    chronosv_error_t Flush() noexcept override {
        return CHRONOSV_OK;
    }
    chronosv_error_t WriteMetadata(const std::uint8_t*, std::size_t) noexcept override {
        return CHRONOSV_OK;
    }
    chronosv_error_t ReadMetadata(std::vector<std::uint8_t>& out) noexcept override {
        out.clear();
        return CHRONOSV_ERR_NOT_FOUND;
    }
};

}  // namespace chronosv::internal

#endif  // CHRONOSV_STORAGE_BACKEND_H
