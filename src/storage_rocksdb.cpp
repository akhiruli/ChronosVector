/*
 * RocksDBStorageBackend implementation. See storage_rocksdb.h for the
 * contract and design links.
 */
#include "storage_rocksdb.h"

#include <algorithm>
#include <cstring>
#include <string>

#include <memory>

#include <rocksdb/cache.h>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/status.h>
#include <rocksdb/table.h>
#include <rocksdb/version.h>
#include <rocksdb/write_batch.h>
#include <rocksdb/write_buffer_manager.h>

#include "block_codec.h"

namespace chronosv::internal {

/* -------------------------------------------------------------------------- */
/* Key encoding: <sensor_id>\x01<block_id_big_endian_u64>                     */
/* Big-endian block_id so RocksDB's lex order == temporal order.              */
/* -------------------------------------------------------------------------- */

namespace {

constexpr char kSep = '\x01';  /* reserved separator, validated in engine */

void put_u64_be(std::uint8_t buf[8], std::uint64_t v) noexcept {
    buf[0] = static_cast<std::uint8_t>((v >> 56) & 0xFFu);
    buf[1] = static_cast<std::uint8_t>((v >> 48) & 0xFFu);
    buf[2] = static_cast<std::uint8_t>((v >> 40) & 0xFFu);
    buf[3] = static_cast<std::uint8_t>((v >> 32) & 0xFFu);
    buf[4] = static_cast<std::uint8_t>((v >> 24) & 0xFFu);
    buf[5] = static_cast<std::uint8_t>((v >> 16) & 0xFFu);
    buf[6] = static_cast<std::uint8_t>((v >>  8) & 0xFFu);
    buf[7] = static_cast<std::uint8_t>((v      ) & 0xFFu);
}

std::uint64_t get_u64_be(const std::uint8_t buf[8]) noexcept {
    return (static_cast<std::uint64_t>(buf[0]) << 56)
         | (static_cast<std::uint64_t>(buf[1]) << 48)
         | (static_cast<std::uint64_t>(buf[2]) << 40)
         | (static_cast<std::uint64_t>(buf[3]) << 32)
         | (static_cast<std::uint64_t>(buf[4]) << 24)
         | (static_cast<std::uint64_t>(buf[5]) << 16)
         | (static_cast<std::uint64_t>(buf[6]) <<  8)
         | (static_cast<std::uint64_t>(buf[7])      );
}

/* Encode `<sensor_id>\x01<block_id_be_u64>` into `out`. Reserves the right
 * amount + returns the key size. `out` must have room for
 * sensor_id.size() + 9 bytes. */
std::string encode_key(std::string_view sensor_id, std::int64_t block_id) {
    std::string k;
    k.reserve(sensor_id.size() + 1 + 8);
    k.append(sensor_id);
    k.push_back(kSep);
    std::uint8_t be[8];
    put_u64_be(be, static_cast<std::uint64_t>(block_id));
    k.append(reinterpret_cast<const char*>(be), 8);
    return k;
}

/* DeleteRange helpers — inclusive begin, exclusive end.
 *   begin = sensor_id + \x01                  (matches block_id = 0 and up)
 *   end   = sensor_id + \x02                  (smallest key past all valid block ids)
 * \x01 is the reserved separator and no sensor_id may contain it, so
 * anything with a different sensor_id sorts strictly outside this range.
 *
 * INVARIANT: this end-key math assumes `\x01` is the ONLY reserved
 * separator byte, and specifically that no valid key starts with
 * `sensor_id + \x02`. If reserved bytes are ever expanded (see
 * validate_sensor_id in engine.cpp), this helper needs a re-derivation
 * of the sentinel. */
std::string range_begin_key(std::string_view sensor_id) {
    std::string k(sensor_id);
    k.push_back(kSep);
    return k;
}
std::string range_end_key(std::string_view sensor_id) {
    std::string k(sensor_id);
    k.push_back(static_cast<char>(kSep + 1));
    return k;
}

/* Map a RocksDB Status to our error codes. Kept liberal — anything that isn't
 * "ok" and isn't a known category becomes IO. */
chronosv_error_t map_status(const rocksdb::Status& s) noexcept {
    if (s.ok())            return CHRONOSV_OK;
    if (s.IsNotFound())    return CHRONOSV_ERR_NOT_FOUND;
    if (s.IsCorruption())  return CHRONOSV_ERR_CORRUPTION;
    if (s.IsInvalidArgument()) return CHRONOSV_ERR_INVALID_ARG;
    /* IOError, NoSpace, MemoryLimit, and everything else fold to IO. */
    return CHRONOSV_ERR_IO;
}

}  // namespace

/* -------------------------------------------------------------------------- */
/* DB deleter (matches the forward-declared struct in the header).            */
/* -------------------------------------------------------------------------- */

void RocksDBStorageBackend::DbDeleter::operator()(rocksdb::DB* db) const noexcept {
    if (db) {
        /* Ignore the Close status — we're tearing down; nothing to do with
         * an error at this point. */
        (void) db->Close();
        delete db;
    }
}

/* -------------------------------------------------------------------------- */
/* Construct / destruct                                                        */
/* -------------------------------------------------------------------------- */

RocksDBStorageBackend::RocksDBStorageBackend(std::string_view path) noexcept {
    rocksdb::Options opts;
    opts.create_if_missing = true;
    /* Snappy compression is a reasonable default; every RocksDB build
     * ships with it. If Snappy is unavailable on some odd platform,
     * RocksDB will fall back gracefully. */
    opts.compression = rocksdb::kSnappyCompression;

    /* --------------------------------------------------------------------
     * Bounded-memory tuning — third attempt, WriteBufferManager-based.
     *
     * Prior failed attempts (see git history / soak-6h*.log):
     *   1. write_buffer_size=32MiB + strict_capacity_limit block cache:
     *      compaction thrash → eviction starved → 245M ring overwrites.
     *   2. max_open_files=500 alone: SST reader cache thrash → I/O
     *      contention → 83M overwrites in 3.5h.
     *
     * Both attempts capped memory *reactively* — they blocked writes
     * once a hard limit was hit. That was the wrong shape. The correct
     * mechanism per the RocksDB wiki is WriteBufferManager (WBM), which
     * caps memtable memory *proactively* by triggering a flush when the
     * total approaches the limit. This is what Kafka Streams and Flink
     * use for exactly this workload shape.
     *
     * Budget:
     *   WBM (memtable):        128 MiB   — proactive flush at ~90%
     *   Block cache:            64 MiB   — includes index/filter blocks
     *   Table reader cache:  ~ 20 MiB   — max_open_files=2000 (soft cap)
     *                       ─────────
     *   Predictable ceiling: ~215 MiB steady state
     *
     * Key differences from prior attempts:
     *   - allow_stall=false on WBM: writers get an error if memory
     *     is over budget, rather than blocking. This prevents the
     *     eviction thread from stalling during compaction bursts.
     *     RocksDB's own retry keeps the write moving.
     *   - cache_index_and_filter_blocks=true: index/filter memory now
     *     counted against block_cache instead of growing unbounded.
     *   - max_open_files=2000: comfortable ceiling for the SST count
     *     produced at 5 MiB/s ingest over 6h (~1700 files), avoids
     *     the reader-cache thrash of attempt 2's 500 cap.
     *   - Left write_buffer_size at RocksDB's 64 MiB default: don't
     *     make individual memtables smaller than RocksDB expects.
     * ------------------------------------------------------------------ */
    static const std::shared_ptr<rocksdb::WriteBufferManager> kWriteBufferManager =
        std::make_shared<rocksdb::WriteBufferManager>(
            /*buffer_size=*/128ULL * 1024 * 1024,
            /*cache=*/nullptr,          /* independent of block cache */
            /*allow_stall=*/false);
    opts.write_buffer_manager = kWriteBufferManager;

    opts.max_open_files = 2000;

    rocksdb::BlockBasedTableOptions table_opts;
    table_opts.cache_index_and_filter_blocks = true;
    table_opts.pin_l0_filter_and_index_blocks_in_cache = true;   /* prevents thrash on hot L0 */
    table_opts.block_cache = rocksdb::NewLRUCache(
        /*capacity=*/64ULL * 1024 * 1024,
        /*num_shard_bits=*/-1,
        /*strict_capacity_limit=*/false);   /* SOFT cap; prior attempt showed strict causes stalls */
    opts.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_opts));

    /* RocksDB API compatibility:
     *   - Ubuntu 24.04's librocksdb (~8.x) has only DB** Open overload
     *   - Homebrew's RocksDB 11.x removed DB** and requires unique_ptr<DB>*
     *   - Raw DB** was removed in RocksDB 10.0 per the changelog.
     *
     * Verified working against:
     *   - macOS Homebrew rocksdb 11.1.2 (ROCKSDB_MAJOR == 11)
     *   - Ubuntu 24.04 librocksdb-dev 8.x (ROCKSDB_MAJOR == 8)
     *
     * If someone hits a compile error on the 9.x boundary, either overload
     * exists there — flip the threshold to `>= 9` and re-test. */
    const std::string path_str(path);
    rocksdb::Status s;
    rocksdb::DB* raw_db = nullptr;
#if defined(ROCKSDB_MAJOR) && ROCKSDB_MAJOR >= 10
    {
        std::unique_ptr<rocksdb::DB> tmp;
        s = rocksdb::DB::Open(opts, path_str, &tmp);
        raw_db = tmp.release();
    }
#else
    s = rocksdb::DB::Open(opts, path_str, &raw_db);
#endif
    if (!s.ok()) {
        last_error_ = s.ToString();
        delete raw_db;   /* defensive; Open never returns non-null on failure but no harm */
        return;   /* db_ stays null; ok() returns false */
    }
    db_.reset(raw_db);
}

RocksDBStorageBackend::~RocksDBStorageBackend() = default;

/* -------------------------------------------------------------------------- */
/* WriteBlock                                                                  */
/* -------------------------------------------------------------------------- */

chronosv_error_t RocksDBStorageBackend::WriteBlock(const Block& blk) noexcept {
    if (!db_) return CHRONOSV_ERR_IO;
    if (blk.sensor_id.empty()) return CHRONOSV_ERR_INVALID_ARG;

    /* Grow scratch to fit; single-threaded per interface contract so no lock. */
    const std::size_t needed = block_serialized_size(blk);
    if (needed == 0) return CHRONOSV_ERR_INVALID_ARG;
    try {
        if (write_scratch_.size() < needed) write_scratch_.resize(needed);
    } catch (const std::bad_alloc&) {
        return CHRONOSV_ERR_OOM;
    }

    const std::size_t written = serialize_block(blk, write_scratch_.data());
    if (written == 0 || written != needed) return CHRONOSV_ERR_INVALID_ARG;

    const std::string key = encode_key(blk.sensor_id, blk.block_id);
    const rocksdb::Slice key_slice(key);
    const rocksdb::Slice val_slice(
        reinterpret_cast<const char*>(write_scratch_.data()), written);

    rocksdb::WriteOptions wopts;
    /* sync=false — WAL sync is Flush()'s job
     * and the Phase 2 kickoff decision. This
     * keeps the eviction hot path fast; durability comes from either the
     * next flush cycle or an explicit chronosv_flush call. */
    wopts.sync = false;

    return map_status(db_->Put(wopts, key_slice, val_slice));
}

/* -------------------------------------------------------------------------- */
/* ReadBlock                                                                   */
/* -------------------------------------------------------------------------- */

chronosv_error_t RocksDBStorageBackend::ReadBlock(
        std::string_view sensor_id,
        std::int64_t block_id,
        Block& out_blk,
        std::int64_t* out_timestamps,
        void*         out_vectors,
        float*        out_scales,
        std::uint8_t* out_payloads) noexcept {
    if (!db_) return CHRONOSV_ERR_IO;
    if (sensor_id.empty()) return CHRONOSV_ERR_INVALID_ARG;

    const std::string key = encode_key(sensor_id, block_id);
    std::string value;
    try {
        const rocksdb::Status s = db_->Get(rocksdb::ReadOptions{}, key, &value);
        if (!s.ok()) return map_status(s);
    } catch (const std::bad_alloc&) {
        return CHRONOSV_ERR_OOM;
    }

    DecodedBlockHeader hdr{};
    const chronosv_error_t rc = decode_block(
        reinterpret_cast<const std::uint8_t*>(value.data()),
        value.size(),
        &hdr,
        out_timestamps,
        out_vectors,
        out_scales,
        out_payloads);

    if (rc != CHRONOSV_OK) {
        /* CORRUPTION from the codec propagates unchanged — the engine's
         * ReadBlock call site fires the on_corruption sink event. */
        return rc;
    }

    /* Populate out_blk with what we recovered. The pointer members remain
     * caller-provided (or nullptr) — Block is a view, not owner. */
    out_blk.sensor_id          = sensor_id;
    out_blk.block_id           = block_id;
    out_blk.t_start_ms         = hdr.t_start_ms;
    out_blk.t_end_ms           = hdr.t_end_ms;
    out_blk.count              = hdr.count;
    out_blk.dim                = hdr.dim;
    out_blk.payload_size_bytes = hdr.payload_size_bytes;
    out_blk.dtype              = hdr.dtype;
    out_blk.timestamps         = out_timestamps;
    out_blk.vectors            = out_vectors;
    out_blk.scales             = out_scales;
    out_blk.payloads           = out_payloads;
    return CHRONOSV_OK;
}

/* -------------------------------------------------------------------------- */
/* ListBlocks                                                                  */
/* -------------------------------------------------------------------------- */

chronosv_error_t RocksDBStorageBackend::ListBlocks(
        std::string_view sensor_id,
        std::vector<std::int64_t>& out_ids) noexcept {
    out_ids.clear();
    if (!db_) return CHRONOSV_ERR_IO;
    if (sensor_id.empty()) return CHRONOSV_ERR_INVALID_ARG;

    const std::string begin = range_begin_key(sensor_id);
    const std::string end   = range_end_key(sensor_id);

    rocksdb::ReadOptions ropts;
    rocksdb::Slice upper_bound_slice(end);
    ropts.iterate_upper_bound = &upper_bound_slice;
    /* std::unique_ptr for RAII on the iterator — RocksDB requires explicit
     * delete after use. */
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ropts));
    if (!it) return CHRONOSV_ERR_IO;

    try {
        for (it->Seek(begin); it->Valid(); it->Next()) {
            const rocksdb::Slice k = it->key();
            /* Key layout: sensor_id + '\x01' + 8-byte-be block_id. Sanity
             * check the size before extracting; a shorter key means the
             * DB is corrupted (shouldn't happen if we're the only writer). */
            if (k.size() != sensor_id.size() + 1 + 8) continue;
            const std::uint8_t* be =
                reinterpret_cast<const std::uint8_t*>(k.data()) + sensor_id.size() + 1;
            out_ids.push_back(static_cast<std::int64_t>(get_u64_be(be)));
        }
        if (!it->status().ok()) return map_status(it->status());
    } catch (const std::bad_alloc&) {
        return CHRONOSV_ERR_OOM;
    }
    return CHRONOSV_OK;
}

/* -------------------------------------------------------------------------- */
/* ListSensors                                                                 */
/* -------------------------------------------------------------------------- */

chronosv_error_t RocksDBStorageBackend::ListSensors(
        std::vector<std::string>& out_ids) noexcept {
    out_ids.clear();
    if (!db_) return CHRONOSV_ERR_IO;

    /* Full-keyspace scan. For each key, the sensor_id is everything before
     * the first \x01. We dedupe by tracking the last emitted prefix; since
     * RocksDB returns keys in sorted order, all blocks for a given sensor
     * appear contiguously. That means we only need a "last seen" comparison,
     * not a full set — cheaper and allocation-light. */
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(rocksdb::ReadOptions{}));
    if (!it) return CHRONOSV_ERR_IO;

    try {
        std::string_view last_seen;
        std::string      last_seen_buf;
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            const rocksdb::Slice k = it->key();
            const char* sep = static_cast<const char*>(
                std::memchr(k.data(), kSep, k.size()));
            if (!sep) continue;  /* malformed key, skip defensively */
            const std::string_view prefix(k.data(), static_cast<std::size_t>(sep - k.data()));
            if (prefix != last_seen) {
                out_ids.emplace_back(prefix);
                last_seen_buf.assign(prefix);
                last_seen = last_seen_buf;
            }
        }
        if (!it->status().ok()) return map_status(it->status());
    } catch (const std::bad_alloc&) {
        return CHRONOSV_ERR_OOM;
    }
    return CHRONOSV_OK;
}

/* -------------------------------------------------------------------------- */
/* DeleteSensor                                                                */
/* -------------------------------------------------------------------------- */

chronosv_error_t RocksDBStorageBackend::DeleteSensor(
        std::string_view sensor_id) noexcept {
    if (!db_) return CHRONOSV_ERR_IO;
    if (sensor_id.empty()) return CHRONOSV_ERR_INVALID_ARG;

    /* DeleteRange over [<sensor_id>\x01, <sensor_id>\x02) drops all blocks
     * for the sensor in one shot. \x02 is a sentinel — no valid key starts
     * with sensor_id + \x02 because \x01 is our reserved separator and
     * sensor_id validation rejects both. */
    const std::string begin = range_begin_key(sensor_id);
    const std::string end   = range_end_key(sensor_id);

    rocksdb::WriteOptions wopts;
    wopts.sync = false;  /* consistent with WriteBlock; sync on Flush */

    /* DeleteRange targets the default column family. */
    const rocksdb::Status s = db_->DeleteRange(
        wopts, db_->DefaultColumnFamily(), begin, end);
    return map_status(s);
}

/* -------------------------------------------------------------------------- */
/* Flush                                                                       */
/* -------------------------------------------------------------------------- */

chronosv_error_t RocksDBStorageBackend::Flush() noexcept {
    if (!db_) return CHRONOSV_ERR_IO;

    /* Two-step durability per the Phase 2 kickoff decision:
     *   1. Flush memtable to L0 SST files.
     *   2. Sync the WAL — this is what makes prior sync=false Puts durable
     *      against machine-level crashes. */
    rocksdb::FlushOptions fopts;
    fopts.wait = true;
    rocksdb::Status s = db_->Flush(fopts);
    if (!s.ok()) return map_status(s);

    s = db_->SyncWAL();
    return map_status(s);
}

/* -------------------------------------------------------------------------- */
/* Engine metadata                                                             */
/* -------------------------------------------------------------------------- */

namespace {
/* Reserved metadata key. Starts with \x00 which no C-string sensor_id can
 * ever contain (null terminates), so this key can't collide with any
 * WriteBlock/DeleteRange key and won't show up in ListSensors (which
 * filters on the \x01 separator being present). */
const std::string kMetadataKey("\0meta", 5);
}  // namespace

chronosv_error_t RocksDBStorageBackend::WriteMetadata(
        const std::uint8_t* data, std::size_t len) noexcept {
    if (!db_) return CHRONOSV_ERR_IO;
    if (!data || len == 0) return CHRONOSV_ERR_INVALID_ARG;
    const rocksdb::Slice val(reinterpret_cast<const char*>(data), len);
    rocksdb::WriteOptions wopts;
    wopts.sync = false;  /* consistent with WriteBlock; Flush syncs the WAL */
    return map_status(db_->Put(wopts, kMetadataKey, val));
}

chronosv_error_t RocksDBStorageBackend::ReadMetadata(
        std::vector<std::uint8_t>& out) noexcept {
    out.clear();
    if (!db_) return CHRONOSV_ERR_IO;
    std::string value;
    try {
        const rocksdb::Status s = db_->Get(rocksdb::ReadOptions{}, kMetadataKey, &value);
        if (!s.ok()) return map_status(s);
        out.assign(value.begin(), value.end());
    } catch (const std::bad_alloc&) {
        return CHRONOSV_ERR_OOM;
    }
    return CHRONOSV_OK;
}

/* -------------------------------------------------------------------------- */
/* ColdBytesEstimate                                                           */
/* -------------------------------------------------------------------------- */

std::uint64_t RocksDBStorageBackend::ColdBytesEstimate() const noexcept {
    if (!db_) return 0;
    /* Approximate size across the entire keyspace. Passing a single-range
     * covering [\x00, \xff*) is a common idiom for "everything."
     *
     * ASSUMPTION: single default column family. If we ever add per-sensor
     * or per-purpose CFs (v0.2+ consideration), this call under-counts
     * because GetApproximateSizes without a CF
     * handle only measures the default CF. Fix at that time by summing
     * GetApproximateSizes(cf_handle, ...) across every CF. */
    const std::string begin(1, '\x00');
    const std::string end(1, '\xff');
    rocksdb::Range ranges[1] = { rocksdb::Range(begin, end) };
    std::uint64_t sizes[1] = {0};
    /* GetApproximateSizes doesn't return a Status; if the DB is dead we
     * just report 0. */
    db_->GetApproximateSizes(ranges, 1, sizes);
    return sizes[0];
}

}  // namespace chronosv::internal
