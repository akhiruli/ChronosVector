/*
 * ChronosVector — Engine core + extern "C" wall.
 *
 * Implements the fifteen public primitives declared in chronos_vector.h.
 * The Engine C++ class lives in an anonymous namespace inside this
 * translation unit; it is never exposed as a symbol. The opaque C handle
 * (chronosv_engine_t*) is a reinterpret_cast of Engine*.
 *
 * Phase 2 status: RocksDB cold tier wired via StorageBackend abstract
 * interface (see storage_backend.h / storage_rocksdb.h). When cfg.cold_path
 * is non-null, the engine constructs a RocksDBStorageBackend; otherwise a
 * NullStorageBackend (in-memory only, matches Phase 1 behavior for tests
 * that don't want a real cold tier).
 */
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "chronosv/chronos_vector.h"
#include "chronosv/metrics_sink.h"
#include "chronosv/ring_buffer.h"
#include "chronosv/storage_backend.h"
#include "chronosv/types.h"

#include "block_codec.h"
#include "kernels.h"
#include "storage_rocksdb.h"

namespace {

using chronosv::internal::Block;
using chronosv::internal::CosineF32Chunk;
using chronosv::internal::CosineI8Chunk;
using chronosv::internal::EuclideanSqF32Chunk;
using chronosv::internal::EuclideanSqI8Chunk;
using chronosv::internal::L2NormF32;
using chronosv::internal::NullStorageBackend;
using chronosv::internal::QuantizeF32ToI8;
using chronosv::internal::RocksDBStorageBackend;
using chronosv::internal::SensorRing;
using chronosv::internal::StorageBackend;
using chronosv::internal::dtype_size;

/* -------------------------------------------------------------------------- */
/* Defaults                                                                   */
/* -------------------------------------------------------------------------- */

constexpr std::uint64_t kDefaultRingCapacity      = 65536;
constexpr std::int64_t  kDefaultWindowDurationMs  = 600000;    // 10 min
constexpr std::int64_t  kDefaultEvictionIntervalMs = 60000;   // 60 s
constexpr std::size_t   kMaxSensorIdLen           = 256;
constexpr std::uint32_t kMaxDim                   = 65536;
constexpr std::uint32_t kMaxPayloadSize           = 4096;

/* -------------------------------------------------------------------------- */
/* Thread-local scratch buffers for query paths                               */
/* -------------------------------------------------------------------------- */

/* These grow monotonically to the largest ever seen for the thread; after
 * warmup, query paths do zero allocation. Each thread holds its own copy —
 * memory is bounded by (concurrent-query-threads × largest-window-size). */
thread_local std::vector<float>                                     tls_scores;  // per-slot scores
thread_local std::vector<std::pair<float, std::uint32_t>>           tls_topk;    // fused top-K heap
thread_local std::vector<float>                                     tls_mean;    // anomaly mean vector
thread_local std::vector<std::int8_t>                               tls_i8_query;    // quantized query buffer (INT8 dispatch)
thread_local std::vector<std::int8_t>                               tls_i8_append;   // quantized input buffer (INT8 Append)

/* -------------------------------------------------------------------------- */
/* Engine metadata codec                                                       */
/*                                                                            */
/* Persisted at StorageBackend metadata slot to allow chronosv_open to        */
/* recover the engine's schema (dim/dtype/payload/etc.) rather than           */
/* forcing the caller to remember it. Little-endian, fixed layout.            */
/*                                                                            */
/* Layout (59 bytes):                                                          */
/*   0    4  magic              = 'CEVM' (0x4D564543 LE)                       */
/*   4    2  version            = 1                                            */
/*   6    2  reserved            = 0                                            */
/*   8    4  dim                                                               */
/*   12   1  storage_dtype                                                     */
/*   13   1  distance_metric                                                   */
/*   14   4  payload_size_bytes                                                */
/*   18   8  ring_capacity                                                     */
/*   26   8  window_duration_ms                                                */
/*   34   8  eviction_interval_ms                                              */
/*   42   16 uuid                                                              */
/*   58   1  recover_hot_window                                                */
/*                                                                            */
/* -------------------------------------------------------------------------- */

constexpr std::uint32_t kEngineMetaMagic   = 0x4D564543u;  /* 'CEVM' LE */
constexpr std::uint16_t kEngineMetaVersion = 1;
constexpr std::size_t   kEngineMetaSize    = 59;

void serialize_engine_metadata(const chronosv_config_t& cfg,
                               const std::uint8_t       uuid[16],
                               std::uint8_t             out[kEngineMetaSize]) noexcept {
    auto put_u16 = [](std::uint8_t* p, std::uint16_t v) { std::memcpy(p, &v, 2); };
    auto put_u32 = [](std::uint8_t* p, std::uint32_t v) { std::memcpy(p, &v, 4); };
    auto put_u64 = [](std::uint8_t* p, std::uint64_t v) { std::memcpy(p, &v, 8); };
    auto put_i64 = [](std::uint8_t* p, std::int64_t  v) { std::memcpy(p, &v, 8); };

    put_u32(out +  0, kEngineMetaMagic);
    put_u16(out +  4, kEngineMetaVersion);
    put_u16(out +  6, 0);  /* reserved */
    put_u32(out +  8, cfg.dim);
    out[12] = cfg.storage_dtype;
    out[13] = cfg.distance_metric;
    put_u32(out + 14, cfg.payload_size_bytes);
    put_u64(out + 18, cfg.ring_capacity);
    put_i64(out + 26, cfg.window_duration_ms);
    put_i64(out + 34, cfg.eviction_interval_ms);
    std::memcpy(out + 42, uuid, 16);
    out[58] = cfg.recover_hot_window;
}

/* Decode metadata into cfg fields + uuid. Returns CHRONOSV_OK on success or
 * CHRONOSV_ERR_CORRUPTION on bad magic/version/length. */
chronosv_error_t deserialize_engine_metadata(const std::uint8_t* data,
                                             std::size_t         len,
                                             chronosv_config_t&  cfg,
                                             std::uint8_t        uuid[16]) noexcept {
    if (len < kEngineMetaSize)  return CHRONOSV_ERR_CORRUPTION;
    auto get_u16 = [](const std::uint8_t* p) { std::uint16_t v; std::memcpy(&v, p, 2); return v; };
    auto get_u32 = [](const std::uint8_t* p) { std::uint32_t v; std::memcpy(&v, p, 4); return v; };
    auto get_u64 = [](const std::uint8_t* p) { std::uint64_t v; std::memcpy(&v, p, 8); return v; };
    auto get_i64 = [](const std::uint8_t* p) { std::int64_t  v; std::memcpy(&v, p, 8); return v; };

    if (get_u32(data + 0) != kEngineMetaMagic)       return CHRONOSV_ERR_CORRUPTION;
    if (get_u16(data + 4) != kEngineMetaVersion)     return CHRONOSV_ERR_CORRUPTION;

    cfg.dim                  = get_u32(data +  8);
    cfg.storage_dtype        = data[12];
    cfg.distance_metric      = data[13];
    cfg.payload_size_bytes   = get_u32(data + 14);
    cfg.ring_capacity        = get_u64(data + 18);
    cfg.window_duration_ms   = get_i64(data + 26);
    cfg.eviction_interval_ms = get_i64(data + 34);
    std::memcpy(uuid, data + 42, 16);
    cfg.recover_hot_window   = data[58];
    return CHRONOSV_OK;
}

/* -------------------------------------------------------------------------- */
/* UUID v4 generator                                                          */
/* -------------------------------------------------------------------------- */

void generate_uuid_v4(std::uint8_t out[16]) noexcept {
    // Use a single per-process seeded RNG. We don't need cryptographic quality
    // for engine identity; std::random_device seeded mt19937_64 is fine.
    static std::mt19937_64 rng([]{
        std::random_device rd;
        std::seed_seq seq{rd(), rd(), rd(), rd()};
        return std::mt19937_64(seq);
    }());
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);

    const std::uint64_t a = rng();
    const std::uint64_t b = rng();
    std::memcpy(out,     &a, 8);
    std::memcpy(out + 8, &b, 8);
    // Version 4 (random) & RFC 4122 variant bits.
    out[6] = static_cast<std::uint8_t>((out[6] & 0x0f) | 0x40);
    out[8] = static_cast<std::uint8_t>((out[8] & 0x3f) | 0x80);
}

bool is_zero_uuid(const std::uint8_t u[16]) noexcept {
    for (int i = 0; i < 16; ++i) if (u[i] != 0) return false;
    return true;
}

/* -------------------------------------------------------------------------- */
/* Config validation + defaults                                               */
/* -------------------------------------------------------------------------- */

chronosv_error_t validate_and_default(chronosv_config_t& c) noexcept {
    if (c.abi_version != CHRONOSV_ABI_VERSION) return CHRONOSV_ERR_INVALID_ARG;
    if (c.dim == 0 || c.dim > kMaxDim)         return CHRONOSV_ERR_INVALID_ARG;
    if (c.payload_size_bytes > kMaxPayloadSize) return CHRONOSV_ERR_INVALID_ARG;
    if (c.storage_dtype > CHRONOSV_DTYPE_INT8) return CHRONOSV_ERR_INVALID_ARG;
    if (c.distance_metric > CHRONOSV_METRIC_EUCLIDEAN) return CHRONOSV_ERR_INVALID_ARG;

    if (c.storage_dtype == CHRONOSV_DTYPE_INT8) {
#ifndef CHRONOSV_ENABLE_INT8
        return CHRONOSV_ERR_UNSUPPORTED;
#endif
    }

    if (c.ring_capacity == 0)          c.ring_capacity = kDefaultRingCapacity;
    if (c.window_duration_ms == 0)     c.window_duration_ms = kDefaultWindowDurationMs;
    if (c.eviction_interval_ms == 0)   c.eviction_interval_ms = kDefaultEvictionIntervalMs;

    if (c.ring_capacity < 64)          return CHRONOSV_ERR_INVALID_ARG;
    if (!std::has_single_bit(c.ring_capacity)) return CHRONOSV_ERR_INVALID_ARG;
    if (c.window_duration_ms < 0)      return CHRONOSV_ERR_INVALID_ARG;
    if (c.eviction_interval_ms < 100)  return CHRONOSV_ERR_INVALID_ARG;

    return CHRONOSV_OK;
}

/* Validate a caller-provided sensor_id. Enforces UTF-8-safe reserved
 * separators. Length is bounded so it doesn't dominate the hot-map lookup. */
chronosv_error_t validate_sensor_id(const char* id) noexcept {
    if (!id) return CHRONOSV_ERR_INVALID_ARG;
    std::size_t n = 0;
    for (; n <= kMaxSensorIdLen && id[n]; ++n) {
        const unsigned char c = static_cast<unsigned char>(id[n]);
        if (c == 0x01) return CHRONOSV_ERR_INVALID_ARG;  // reserved separator
    }
    if (n == 0)                return CHRONOSV_ERR_INVALID_ARG;
    if (n > kMaxSensorIdLen)   return CHRONOSV_ERR_INVALID_ARG;
    return CHRONOSV_OK;
}

/* -------------------------------------------------------------------------- */
/* Heterogeneous string hasher — enables std::string_view lookup on a         */
/* std::string-keyed unordered_map without allocating (C++20 P0919).           */
/* -------------------------------------------------------------------------- */

struct StringHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view v)  const noexcept { return std::hash<std::string_view>{}(v); }
    std::size_t operator()(const std::string& s) const noexcept { return std::hash<std::string_view>{}(s); }
    std::size_t operator()(const char* c)       const noexcept { return std::hash<std::string_view>{}(c); }
};
struct StringEq {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const noexcept { return a == b; }
    bool operator()(const std::string& a, std::string_view b) const noexcept { return a == b; }
    bool operator()(std::string_view a, const std::string& b) const noexcept { return a == b; }
    bool operator()(const std::string& a, const std::string& b) const noexcept { return a == b; }
    bool operator()(const char* a, std::string_view b) const noexcept { return std::string_view(a) == b; }
    bool operator()(std::string_view a, const char* b) const noexcept { return a == std::string_view(b); }
};
/* Per-sensor engine state — the ring plus the persistence-related counter.
 * Held in the sensor map so all sensor-local state travels together. */
struct SensorSlot {
    std::unique_ptr<SensorRing> ring;
    /* Monotonic block_id counter for the RocksDB cold-tier key. Advances
     * once per successful WriteBlock. Recovered from
     * StorageBackend::ListBlocks on chronosv_open. */
    std::atomic<std::int64_t>   next_block_id{0};
};
using SensorMap = std::unordered_map<std::string, std::unique_ptr<SensorSlot>,
                                     StringHash, StringEq>;

/* Reusable scratch for the eviction copy-then-serialize path. Race safety:
 * eviction copies the evicted range into engine-owned scratch BEFORE
 * advancing tail, so the producer is free to overwrite the ring slots
 * without corrupting a block that's still being persisted.
 *
 * INVARIANTS:
 *   - SINGLE-WRITER: only the engine's eviction thread touches this. It
 *     handles multiple sensors sequentially in one pass. If eviction ever
 *     goes parallel-per-sensor, this needs to become per-sensor scratch or
 *     grow a mutex. Neither is on the roadmap.
 *   - CAPACITY MONOTONIC: EvictSensor uses `if (v.size() < needed) v.resize(needed)`,
 *     which grows size and never shrinks it. std::vector::resize() only
 *     grows capacity, never shrinks. So the RSS attributable to this
 *     scratch reaches a high-water-mark determined by the largest single
 *     eviction and stays there — a 24 h soak test with bounded-size
 *     evictions will see flat memory for these vectors.
 */
struct EvictScratch {
    std::vector<std::int64_t>  timestamps;
    std::vector<std::uint8_t>  vectors;    /* raw bytes, dim*count*dtype_size */
    std::vector<float>         scales;     /* INT8 only */
    std::vector<std::uint8_t>  payloads;   /* payload_size > 0 only */
};

/* -------------------------------------------------------------------------- */
/* Engine class                                                                */
/* -------------------------------------------------------------------------- */

class Engine {
public:
    /* Recovery factory used by chronosv_open. Opens an existing engine at
     * a cold-tier path, reads the persisted schema metadata (dim / dtype /
     * payload / metric / ring_capacity / window / eviction interval / uuid),
     * enumerates persisted sensors, populates the map with empty rings and
     * seeded next_block_id counters.
     *
     * Ring rehydration (recover_hot_window path) is deferred to a follow-up
     * session; recovered sensors get empty rings that allocate on first
     * Append.
     *
     * Legacy path (metadata NOT_FOUND): falls back to documented defaults
     * (dim=128 float32) so pre-metadata engines can still be opened. Log via
     * on_warning if a sink is ever plumbed here; for now silent. */
    static Engine* Open(const char* path, chronosv_error_t* err_out) noexcept {
        if (!path || path[0] == '\0') { *err_out = CHRONOSV_ERR_INVALID_ARG; return nullptr; }
        chronosv_config_t cfg{};
        cfg.abi_version = CHRONOSV_ABI_VERSION;
        cfg.cold_path   = path;
        cfg.dim         = 128;   /* placeholder; overwritten by metadata read below */
        try {
            std::unique_ptr<Engine> e(new Engine(cfg));
            /* Open the backend first — we can't read metadata without it. */
            const auto be = e->InitBackend();
            if (be != CHRONOSV_OK) { *err_out = be; return nullptr; }

            /* Read persisted metadata (if any). */
            std::vector<std::uint8_t> meta_bytes;
            const auto me = e->backend_->ReadMetadata(meta_bytes);
            if (me == CHRONOSV_OK) {
                /* Metadata present — replace our placeholder cfg wholesale
                 * with what was persisted. UUID also comes from metadata so
                 * the engine's identity is preserved across close/reopen. */
                const auto de = deserialize_engine_metadata(
                    meta_bytes.data(), meta_bytes.size(), e->cfg_, e->uuid_);
                if (de != CHRONOSV_OK) { *err_out = de; return nullptr; }
                e->cfg_.abi_version = CHRONOSV_ABI_VERSION;
                e->cfg_.cold_path   = path;
                /* metrics_sink is per-process, not persisted. */
                e->cfg_.metrics_sink = nullptr;
            } else if (me != CHRONOSV_ERR_NOT_FOUND) {
                *err_out = me;
                return nullptr;
            }
            /* If NOT_FOUND, keep the placeholder cfg above. */

            /* Fill defaults / validate the config we ended up with. */
            chronosv_config_t validated = e->cfg_;
            const auto ve = validate_and_default(validated);
            if (ve != CHRONOSV_OK) { *err_out = ve; return nullptr; }
            e->cfg_ = validated;

            /* Enumerate persisted sensors + seed next_block_id per sensor. */
            std::vector<std::string> sensor_ids;
            const auto lse = e->backend_->ListSensors(sensor_ids);
            if (lse != CHRONOSV_OK) { *err_out = lse; return nullptr; }

            /* If recover_hot_window is set, compute the wall-clock cutoff
             * once for all sensors. Entries older than this are filtered
             * out during rehydration. */
            const bool do_rehydrate = (e->cfg_.recover_hot_window != 0);
            std::int64_t rehydrate_cutoff_ms = 0;
            if (do_rehydrate) {
                using namespace std::chrono;
                const auto now_ms = duration_cast<milliseconds>(
                    system_clock::now().time_since_epoch()).count();
                rehydrate_cutoff_ms = now_ms - e->cfg_.window_duration_ms;
            }

            std::vector<std::int64_t> ids_buf;
            for (const auto& sid : sensor_ids) {
                ids_buf.clear();
                const auto lbe = e->backend_->ListBlocks(sid, ids_buf);
                if (lbe != CHRONOSV_OK) { *err_out = lbe; return nullptr; }
                auto slot = std::make_unique<SensorSlot>();
                slot->next_block_id.store(
                    ids_buf.empty() ? 0 : ids_buf.back() + 1,
                    std::memory_order_relaxed);
                if (do_rehydrate) {
                    /* Rehydrate the hot ring from persisted blocks. On any
                     * per-block error (corruption, IO), skip the block —
                     * the recovery is best-effort, engine still opens. */
                    e->RehydrateSensorRing(sid, *slot, ids_buf, rehydrate_cutoff_ms);
                }
                e->sensors_.emplace(sid, std::move(slot));
            }

            e->StartEvictionThread();
            *err_out = CHRONOSV_OK;
            return e.release();
        } catch (const std::bad_alloc&) {
            *err_out = CHRONOSV_ERR_OOM;
            return nullptr;
        } catch (...) {
            *err_out = CHRONOSV_ERR_INTERNAL;
            return nullptr;
        }
    }

    /* Factory: validates config, opens the backend, starts the eviction
     * thread. On failure, returns nullptr and writes the error to *err_out. */
    static Engine* Create(const chronosv_config_t& cfg_in,
                          chronosv_error_t* err_out) noexcept {
        chronosv_config_t cfg = cfg_in;
        const auto ve = validate_and_default(cfg);
        if (ve != CHRONOSV_OK) {
            if (err_out) *err_out = ve;
            return nullptr;
        }
        try {
            std::unique_ptr<Engine> e(new Engine(cfg));
            const auto be = e->InitBackend();
            if (be != CHRONOSV_OK) {
                if (err_out) *err_out = be;
                return nullptr;
            }
            /* Reconcile with any existing engine metadata at this path.
             * Three cases:
             *   1) NOT_FOUND (fresh path or NullBackend) — write ours.
             *   2) OK + schema matches — adopt persisted UUID (engine
             *      identity is stable across close/reopen), refresh the
             *      runtime-adjustable fields (window / eviction interval).
             *   3) OK + schema mismatch — refuse with INVALID_ARG. The
             *      user is trying to Create with a different dim / dtype /
             *      metric / payload_size / ring_capacity than what's
             *      persisted, which would silently corrupt existing blocks.
             *      They should either fix their config, use chronosv_open
             *      to inherit the persisted config, or wipe the path.
             *   4) Other error — propagate. */
            std::vector<std::uint8_t> existing;
            const auto re = e->backend_->ReadMetadata(existing);
            if (re == CHRONOSV_OK) {
                chronosv_config_t persisted{};
                std::uint8_t persisted_uuid[16] = {0};
                const auto de = deserialize_engine_metadata(
                    existing.data(), existing.size(), persisted, persisted_uuid);
                if (de != CHRONOSV_OK) { if (err_out) *err_out = de; return nullptr; }

                const bool schema_matches =
                    (persisted.dim == e->cfg_.dim)
                 && (persisted.storage_dtype == e->cfg_.storage_dtype)
                 && (persisted.distance_metric == e->cfg_.distance_metric)
                 && (persisted.payload_size_bytes == e->cfg_.payload_size_bytes)
                 && (persisted.ring_capacity == e->cfg_.ring_capacity);
                if (!schema_matches) {
                    if (err_out) *err_out = CHRONOSV_ERR_INVALID_ARG;
                    return nullptr;
                }
                /* Adopt persisted identity. */
                std::memcpy(e->uuid_, persisted_uuid, 16);
            } else if (re != CHRONOSV_ERR_NOT_FOUND) {
                if (err_out) *err_out = re;
                return nullptr;
            }

            /* Write (or refresh) the metadata blob so runtime fields
             * (window_ms, interval_ms) reflect what the user asked for. */
            std::uint8_t meta[kEngineMetaSize];
            serialize_engine_metadata(e->cfg_, e->uuid_, meta);
            const auto me = e->backend_->WriteMetadata(meta, kEngineMetaSize);
            if (me != CHRONOSV_OK && me != CHRONOSV_ERR_NOT_FOUND) {
                if (err_out) *err_out = me;
                return nullptr;
            }
            e->StartEvictionThread();
            if (err_out) *err_out = CHRONOSV_OK;
            return e.release();
        } catch (const std::bad_alloc&) {
            if (err_out) *err_out = CHRONOSV_ERR_OOM;
            return nullptr;
        } catch (...) {
            if (err_out) *err_out = CHRONOSV_ERR_INTERNAL;
            return nullptr;
        }
    }

    ~Engine() {
        // Ensure the eviction thread is stopped and joined before members
        // (sensors_ especially) start destructing. Close is idempotent.
        Close();
    }

    /* ---- Lifecycle ---- */

    chronosv_error_t Close() noexcept {
        // Signal the eviction thread to stop and wake it. Close is idempotent
        // — we can be called multiple times. Once closed, subsequent API
        // entry points reject with CLOSED.
        if (!closed_.exchange(true, std::memory_order_acq_rel)) {
            stop_requested_.store(true, std::memory_order_release);
            {
                std::unique_lock<std::mutex> lock(eviction_mu_);
                eviction_cv_.notify_all();
            }
            if (eviction_thread_.joinable()) eviction_thread_.join();
        }
        return CHRONOSV_OK;
    }
    bool IsClosed() const noexcept {
        return closed_.load(std::memory_order_acquire);
    }

    /* ---- Sensor lifecycle ---- */

    chronosv_error_t PreallocateSensor(std::string_view sensor_id) {
        if (IsClosed()) return CHRONOSV_ERR_CLOSED;
        chronosv_error_t err = CHRONOSV_OK;
        (void) GetOrCreateRing(sensor_id, &err);
        return err;
    }

    chronosv_error_t DropSensor(std::string_view sensor_id) {
        if (IsClosed()) return CHRONOSV_ERR_CLOSED;
        std::string sid_copy;   /* for backend call after we release lock */
        {
            std::unique_lock<std::shared_mutex> lock(map_mutex_);
            auto it = sensors_.find(sensor_id);
            if (it == sensors_.end()) return CHRONOSV_ERR_NOT_FOUND;
            // Roll the doomed ring's overwrite counters into the cumulative
            // "dropped" atomics BEFORE erasing, so GetStats can still report
            // an honest lifetime overwrite total after this sensor is gone.
            // Otherwise the operator's canary ("if overwrites > 0, we lost data")
            // silently rewinds when sensors get dropped.
            SensorRing* ring = it->second->ring.get();
            if (ring) {
                dropped_overwrite_events_.fetch_add(
                    ring->overwrite_events(), std::memory_order_relaxed);
                dropped_overwritten_entries_.fetch_add(
                    ring->overwritten_entries(), std::memory_order_relaxed);
            }
            sid_copy = it->first;   /* preserve id for backend call */
            sensors_.erase(it);
            total_dropped_sensors_.fetch_add(1, std::memory_order_relaxed);
        }
        /* Delete any persisted blocks for this sensor. Called outside the
         * map lock — the backend has its own thread-safety and this can be
         * slow (I/O). Return code is best-effort; the in-memory drop
         * already happened successfully. */
        if (backend_) {
            (void) backend_->DeleteSensor(sid_copy);
        }
        return CHRONOSV_OK;
    }

    /* ---- Ingest ---- */

    chronosv_error_t Append(std::string_view sensor_id,
                            std::int64_t ts_ms,
                            const float* vec,
                            std::size_t dim,
                            const void* payload) {
        if (IsClosed())         return CHRONOSV_ERR_CLOSED;
        if (dim != cfg_.dim)    return CHRONOSV_ERR_DIM_MISMATCH;

        chronosv_error_t err = CHRONOSV_OK;
        SensorRing* ring = GetOrCreateRing(sensor_id, &err);
        if (!ring) return err;

        // Compute L2 norm of the ORIGINAL float vec (what we'd get if we
        // didn't quantize). Kept in the ring's norms_ column for both dtypes.
        const float norm = L2NormF32(vec, dim);

        if (ring->dtype() == CHRONOSV_DTYPE_INT8) {
            /* Quantize to int8 + scale; feed the ring the quantized bytes. */
            if (tls_i8_append.size() < dim) tls_i8_append.resize(dim);
            float scale = 0.0f;
            QuantizeF32ToI8(vec, dim, tls_i8_append.data(), &scale);
            ring->Append(ts_ms, tls_i8_append.data(), norm, payload, scale);
        } else {
            ring->Append(ts_ms, vec, norm, payload);
        }

        total_appends_.fetch_add(1, std::memory_order_relaxed);
        if (sink_ && sink_->on_append) {
            // sink_id_buf is a small stack-scoped null-terminated copy; the
            // callback contract requires a C string. Using string_view.data()
            // directly is only safe if the caller passed a null-terminated
            // buffer, which C ABI callers always do (they passed const char*).
            sink_->on_append(sink_->user_data, sensor_id.data(), /*latency_ns=*/0);
        }
        // Detect overwrite events per-append and forward to sink. The ring
        // itself maintains cumulative counters; we snapshot the delta.
        if (sink_ && sink_->on_overwrite) {
            static thread_local std::uint64_t last_overwrites_seen = 0;
            const auto cur = ring->overwritten_entries();
            if (cur > last_overwrites_seen) {
                sink_->on_overwrite(sink_->user_data, sensor_id.data(),
                                    cur - last_overwrites_seen);
                last_overwrites_seen = cur;
            }
        }
        return CHRONOSV_OK;
    }

    chronosv_error_t AppendBatch(std::string_view sensor_id,
                                 const std::int64_t* ts_ms,
                                 const float* vecs,
                                 std::size_t count,
                                 std::size_t dim,
                                 const void* payloads) {
        if (IsClosed())         return CHRONOSV_ERR_CLOSED;
        if (dim != cfg_.dim)    return CHRONOSV_ERR_DIM_MISMATCH;
        if (count == 0)         return CHRONOSV_OK;
        if (!ts_ms || !vecs)    return CHRONOSV_ERR_INVALID_ARG;

        chronosv_error_t err = CHRONOSV_OK;
        SensorRing* ring = GetOrCreateRing(sensor_id, &err);
        if (!ring) return err;

        const auto* payload_bytes = static_cast<const std::uint8_t*>(payloads);
        const bool is_int8 = (ring->dtype() == CHRONOSV_DTYPE_INT8);
        if (is_int8 && tls_i8_append.size() < dim) tls_i8_append.resize(dim);
        for (std::size_t i = 0; i < count; ++i) {
            const float* v = vecs + i * dim;
            const float norm = L2NormF32(v, dim);
            const void* p = (payloads && cfg_.payload_size_bytes > 0)
                            ? static_cast<const void*>(payload_bytes + i * cfg_.payload_size_bytes)
                            : nullptr;
            if (is_int8) {
                float scale = 0.0f;
                QuantizeF32ToI8(v, dim, tls_i8_append.data(), &scale);
                ring->Append(ts_ms[i], tls_i8_append.data(), norm, p, scale);
            } else {
                ring->Append(ts_ms[i], v, norm, p);
            }
        }
        total_appends_.fetch_add(count, std::memory_order_relaxed);
        return CHRONOSV_OK;
    }

    /* ---- Query ---- */

    chronosv_error_t QueryNearestN(std::string_view sensor_id,
                                   const float* target,
                                   std::size_t dim,
                                   int n,
                                   std::int64_t* out_ts,
                                   float* out_scores,
                                   int* out_count) {
        if (IsClosed())         return CHRONOSV_ERR_CLOSED;
        if (dim != cfg_.dim)    return CHRONOSV_ERR_DIM_MISMATCH;
        if (n <= 0 || !target || !out_ts || !out_scores || !out_count) {
            return CHRONOSV_ERR_INVALID_ARG;
        }

        std::shared_lock<std::shared_mutex> lock(map_mutex_);
        auto it = sensors_.find(sensor_id);
        if (it == sensors_.end()) { *out_count = 0; return CHRONOSV_ERR_NOT_FOUND; }
        SensorRing* ring = it->second->ring.get();
        if (!ring) { *out_count = 0; return CHRONOSV_WARN_PARTIAL_RESULT; }

        auto [begin, end] = ring->Snapshot();
        const std::size_t N = static_cast<std::size_t>(end - begin);
        if (N == 0) {
            *out_count = 0;
            return CHRONOSV_WARN_PARTIAL_RESULT;
        }

        // Ensure thread-local scratch is large enough.
        if (tls_scores.size() < N)  tls_scores.resize(N);

        const bool is_cosine = (cfg_.distance_metric == CHRONOSV_METRIC_COSINE);
        const float qn = L2NormF32(target, dim);

        // For INT8 ring storage, quantize the query into tls_i8_query and
        // dispatch to the INT8 kernel. The float `qn` stays as the original
        // norm (needed for the cosine denominator).
        float q_scale = 0.0f;
        const std::int8_t* q_i8 = nullptr;
        if (ring->dtype() == CHRONOSV_DTYPE_INT8) {
            if (tls_i8_query.size() < dim) tls_i8_query.resize(dim);
            QuantizeF32ToI8(target, dim, tls_i8_query.data(), &q_scale);
            q_i8 = tls_i8_query.data();
        }

        // Split the [begin, end) logical range into up to two contiguous
        // physical chunks so we can call the kernel on each without a copy.
        FillScoresForRange(ring, begin, N, dim, target, qn,
                           q_i8, q_scale,
                           is_cosine, tls_scores.data());

        const int take = std::min<int>(n, static_cast<int>(N));

        // Fused top-K selection via a size-K heap. Avoids allocating and
        // sorting an O(N) index array — critical at N=100k where the naive
        // nth_element + sort pass adds hundreds of microseconds.
        //
        // Heap orientation: heap_cmp(a,b) = "a is BETTER than b" makes
        // std::push_heap place the WORST of the current top-K at front(),
        // so we can cheaply check "does new score displace the worst?".
        using Pair = std::pair<float, std::uint32_t>;
        auto heap_cmp = [is_cosine](const Pair& a, const Pair& b) {
            return is_cosine ? (a.first > b.first) : (a.first < b.first);
        };
        tls_topk.clear();
        if (static_cast<int>(tls_topk.capacity()) < take) tls_topk.reserve(take);

        for (std::uint32_t i = 0; i < N; ++i) {
            const float s = tls_scores[i];
            if (static_cast<int>(tls_topk.size()) < take) {
                tls_topk.emplace_back(s, i);
                std::push_heap(tls_topk.begin(), tls_topk.end(), heap_cmp);
            } else {
                const float worst = tls_topk.front().first;
                const bool s_better = is_cosine ? (s > worst) : (s < worst);
                if (s_better) {
                    std::pop_heap(tls_topk.begin(), tls_topk.end(), heap_cmp);
                    tls_topk.back() = {s, i};
                    std::push_heap(tls_topk.begin(), tls_topk.end(), heap_cmp);
                }
            }
        }
        // Final sort of the K survivors, best-first for the caller.
        std::sort(tls_topk.begin(), tls_topk.end(),
                  [is_cosine](const Pair& a, const Pair& b) {
                      return is_cosine ? (a.first > b.first) : (a.first < b.first);
                  });

        for (int i = 0; i < take; ++i) {
            const std::uint32_t local = tls_topk[static_cast<std::size_t>(i)].second;
            out_ts[i]     = ring->timestamp_at(begin + local);
            out_scores[i] = tls_topk[static_cast<std::size_t>(i)].first;
        }
        *out_count = take;

        total_queries_.fetch_add(1, std::memory_order_relaxed);
        if (sink_ && sink_->on_query) {
            sink_->on_query(sink_->user_data, sensor_id.data(),
                            /*latency_ns=*/0, take);
        }
        return (take < n) ? CHRONOSV_WARN_PARTIAL_RESULT : CHRONOSV_OK;
    }

    chronosv_error_t QueryRange(std::string_view sensor_id,
                                std::int64_t t_start_ms,
                                std::int64_t t_end_ms,
                                std::int64_t* out_ts,
                                float* out_vecs,
                                int max,
                                int* out_count) {
        if (IsClosed())         return CHRONOSV_ERR_CLOSED;
        if (max <= 0 || !out_ts || !out_vecs || !out_count) {
            return CHRONOSV_ERR_INVALID_ARG;
        }
        if (t_end_ms < t_start_ms) return CHRONOSV_ERR_INVALID_ARG;

        std::shared_lock<std::shared_mutex> lock(map_mutex_);
        auto it = sensors_.find(sensor_id);
        if (it == sensors_.end()) { *out_count = 0; return CHRONOSV_ERR_NOT_FOUND; }
        SensorRing* ring = it->second->ring.get();
        if (!ring) { *out_count = 0; return CHRONOSV_OK; }

        auto [begin, end] = ring->Snapshot();
        const std::size_t N = static_cast<std::size_t>(end - begin);
        const std::size_t dim = cfg_.dim;
        const std::size_t vsz = dim * sizeof(float);

        int written = 0;
        std::int64_t oldest_ts_in_ring = INT64_MAX;
        for (std::uint64_t idx = begin; idx < end && written < max; ++idx) {
            const std::int64_t ts = ring->timestamp_at(idx);
            if (ts < oldest_ts_in_ring) oldest_ts_in_ring = ts;
            if (ts < t_start_ms || ts > t_end_ms) continue;
            out_ts[written] = ts;
            std::memcpy(out_vecs + static_cast<std::size_t>(written) * dim,
                        ring->vector_at(idx), vsz);
            ++written;
        }
        *out_count = written;
        total_queries_.fetch_add(1, std::memory_order_relaxed);

        chronosv_error_t warn = CHRONOSV_OK;
        // If the caller asked for data older than what's currently in the
        // hot window, warn — they got a truncated view.
        if (N > 0 && t_start_ms < oldest_ts_in_ring) {
            warn = CHRONOSV_WARN_RANGE_TRUNCATED;
            if (sink_ && sink_->on_warning) {
                sink_->on_warning(sink_->user_data, warn);
            }
        }
        return warn;
    }

    chronosv_error_t DetectAnomaly(std::string_view sensor_id,
                                   const float* v,
                                   std::size_t dim,
                                   float threshold,
                                   int* out_is_anomaly) {
        if (IsClosed())         return CHRONOSV_ERR_CLOSED;
        if (dim != cfg_.dim)    return CHRONOSV_ERR_DIM_MISMATCH;
        if (!v || !out_is_anomaly) return CHRONOSV_ERR_INVALID_ARG;

        std::shared_lock<std::shared_mutex> lock(map_mutex_);
        auto it = sensors_.find(sensor_id);
        if (it == sensors_.end()) { *out_is_anomaly = 0; return CHRONOSV_ERR_NOT_FOUND; }
        SensorRing* ring = it->second->ring.get();
        if (!ring) { *out_is_anomaly = 0; return CHRONOSV_OK; }

        auto [begin, end] = ring->Snapshot();
        const std::size_t N = static_cast<std::size_t>(end - begin);
        if (N == 0) {
            *out_is_anomaly = 0;
            total_anomaly_checks_.fetch_add(1, std::memory_order_relaxed);
            return CHRONOSV_OK;
        }

        // Compute mean of window vectors — Eigen accumulator into thread-local buffer.
        if (tls_mean.size() < dim) tls_mean.resize(dim);
        std::memset(tls_mean.data(), 0, dim * sizeof(float));
        {
            using Eigen::Map;
            using Eigen::MatrixXf;
            using Eigen::VectorXf;
            using Idx = Eigen::Index;
            Map<VectorXf> mean(tls_mean.data(), static_cast<Idx>(dim));

            const std::uint64_t mask = ring->mask();
            const std::uint64_t start_slot = begin & mask;
            const auto* base = static_cast<const float*>(ring->vectors_raw());

            const std::size_t first_len =
                std::min<std::size_t>(N, ring->capacity() - start_slot);
            {
                const auto W = Map<const MatrixXf>(
                    base + start_slot * dim,
                    static_cast<Idx>(dim),
                    static_cast<Idx>(first_len));
                mean += W.rowwise().sum();
            }
            if (first_len < N) {
                const std::size_t second_len = N - first_len;
                const auto W = Map<const MatrixXf>(
                    base,
                    static_cast<Idx>(dim),
                    static_cast<Idx>(second_len));
                mean += W.rowwise().sum();
            }
            mean /= static_cast<float>(N);
        }

        // Compute distance(mean, v) using the engine metric.
        // For cosine: distance = 1 - cos_sim; anomaly if > threshold.
        // For euclid: distance = ||v - mean||;  anomaly if > threshold.
        float distance;
        if (cfg_.distance_metric == CHRONOSV_METRIC_COSINE) {
            const float mean_norm = L2NormF32(tls_mean.data(), dim);
            const float v_norm    = L2NormF32(v, dim);
            float score = 0.0f;
            CosineF32Chunk(tls_mean.data(), &mean_norm, /*count=*/1, dim,
                           v, v_norm, &score);
            distance = 1.0f - score;
        } else {
            const float mean_norm = L2NormF32(tls_mean.data(), dim);
            const float v_norm    = L2NormF32(v, dim);
            float sq = 0.0f;
            EuclideanSqF32Chunk(tls_mean.data(), &mean_norm, /*count=*/1, dim,
                                v, v_norm, &sq);
            distance = std::sqrt(std::max(0.0f, sq));
        }

        const bool is_anom = distance > threshold;
        *out_is_anomaly = is_anom ? 1 : 0;
        total_anomaly_checks_.fetch_add(1, std::memory_order_relaxed);
        if (is_anom && sink_ && sink_->on_anomaly_detected) {
            sink_->on_anomaly_detected(sink_->user_data, sensor_id.data(), distance);
        }
        return CHRONOSV_OK;
    }

    /* ---- Maintenance ---- */

    /* Force everything currently in the hot rings out to the cold tier,
     * then WAL-sync. Blocks until durable. */
    chronosv_error_t Flush() {
        if (IsClosed()) return CHRONOSV_ERR_CLOSED;
        /* Pass window_ms=0 explicitly so EVERYTHING in the ring is aged
         * out. Doesn't mutate cfg_ — a concurrent chronosv_get_stats or
         * chronosv_maintain_sliding_window reader always sees a coherent
         * window value. */
        EvictOnce(/*window_ms_override=*/0);

        if (!backend_) return CHRONOSV_OK;
        return backend_->Flush();
    }

    chronosv_error_t MaintainSlidingWindow(std::int64_t window_ms) {
        if (IsClosed()) return CHRONOSV_ERR_CLOSED;
        if (window_ms < 0) return CHRONOSV_ERR_INVALID_ARG;
        // Update the config, then trigger one immediate eviction pass so
        // callers who tightened the window see the effect right away.
        // Callers who want to trigger eviction WITHOUT changing the window
        // can call this with the current cfg_.window_duration_ms value.
        cfg_.window_duration_ms = window_ms;
        EvictOnce();
        return CHRONOSV_OK;
    }

    /* ---- Observability ---- */

    chronosv_error_t ListSensors(char** out_ids,
                                 std::size_t max,
                                 std::size_t* out_count) const {
        if (!out_count) return CHRONOSV_ERR_INVALID_ARG;
        if (max > 0 && !out_ids) return CHRONOSV_ERR_INVALID_ARG;

        std::shared_lock<std::shared_mutex> lock(map_mutex_);
        std::size_t written = 0;
        for (const auto& [id, _] : sensors_) {
            if (written >= max) break;
            char* copy = static_cast<char*>(std::malloc(id.size() + 1));
            if (!copy) {
                // Roll back what we've already malloc'd.
                for (std::size_t i = 0; i < written; ++i) std::free(out_ids[i]);
                *out_count = 0;
                return CHRONOSV_ERR_OOM;
            }
            std::memcpy(copy, id.data(), id.size());
            copy[id.size()] = '\0';
            out_ids[written++] = copy;
        }
        *out_count = written;
        return CHRONOSV_OK;
    }

    chronosv_error_t GetStats(chronosv_stats_t* out) const {
        if (!out) return CHRONOSV_ERR_INVALID_ARG;
        std::memset(out, 0, sizeof(*out));
        std::memcpy(out->uuid, uuid_, 16);
        out->abi_version              = CHRONOSV_ABI_VERSION;
        out->sensor_cap               = cfg_.max_sensors;
        out->total_appends            = total_appends_.load(std::memory_order_relaxed);
        out->total_queries            = total_queries_.load(std::memory_order_relaxed);
        out->total_anomaly_checks     = total_anomaly_checks_.load(std::memory_order_relaxed);
        out->total_evictions          = total_evictions_.load(std::memory_order_relaxed);
        out->total_dropped_sensors    = total_dropped_sensors_.load(std::memory_order_relaxed);

        out->flush_errors_total = flush_errors_total_.load(std::memory_order_relaxed);
        // Cold-tier size estimate — NullStorageBackend returns 0; RocksDB
        // returns an approximate SST byte count.
        if (backend_) {
            out->cold_bytes_estimate = backend_->ColdBytesEstimate();
        }
        // Sensor-scoped accounting. Seed the overwrite totals with the
        // cumulative counts from ALL previously-dropped sensors so the
        // number is a true "lifetime overwrites since engine creation"
        // regardless of drops (see DropSensor above).
        out->total_overwrite_events    = dropped_overwrite_events_.load(std::memory_order_relaxed);
        out->total_overwritten_entries = dropped_overwritten_entries_.load(std::memory_order_relaxed);
        std::shared_lock<std::shared_mutex> lock(map_mutex_);
        out->sensor_count = static_cast<std::uint32_t>(sensors_.size());
        for (const auto& [_, slot] : sensors_) {
            if (!slot->ring) continue;   /* slot without ring — post-open, pre-first-append */
            out->hot_bytes                 += slot->ring->hot_bytes();
            out->total_overwrite_events    += slot->ring->overwrite_events();
            out->total_overwritten_entries += slot->ring->overwritten_entries();
        }
        // Latency percentiles: not measured internally yet — deferred to a
        // future lock-free histogram. Zero-filled by the memset above.
        return CHRONOSV_OK;
    }

private:
    /* Two-phase construction: the fallible parts (backend open) happen in
     * InitBackend() so the ctor itself can't leave a partially-built
     * engine. Create() calls the ctor + InitBackend() + starts the
     * eviction thread only after both succeed. */
    explicit Engine(const chronosv_config_t& cfg) : cfg_(cfg) {
        std::memcpy(uuid_, cfg.uuid, 16);
        if (is_zero_uuid(uuid_)) generate_uuid_v4(uuid_);
        sink_ = cfg.metrics_sink;
    }

    /* Construct the cold-tier backend based on cfg. If cold_path is null,
     * uses NullStorageBackend (in-memory only). Otherwise opens RocksDB.
     * Returns OK on success or the RocksDB error on failure. */
    chronosv_error_t InitBackend() {
        if (!cfg_.cold_path || cfg_.cold_path[0] == '\0') {
            backend_ = std::make_unique<NullStorageBackend>();
            return CHRONOSV_OK;
        }
        auto b = std::make_unique<RocksDBStorageBackend>(cfg_.cold_path);
        if (!b->ok()) return CHRONOSV_ERR_IO;
        backend_ = std::move(b);
        return CHRONOSV_OK;
    }

    void StartEvictionThread() {
        // One background thread per engine, wakes every eviction_interval_ms,
        // trims tail past entries older than (max_ts_seen - window_duration_ms),
        // and copies the evicted range to the backend before advancing tail.
        //
        // Uses std::thread + std::atomic<bool> instead of std::jthread /
        // std::stop_token: those are C++20 but libc++ on Apple platforms
        // has been slow to ship them (missing in AppleClang as of Xcode
        // 16.x). std::thread is portable across every C++20 stdlib.
        eviction_thread_ = std::thread([this]() {
            EvictionLoop();
        });
    }

    void EvictionLoop() noexcept {
        while (!stop_requested_.load(std::memory_order_acquire)) {
            {
                std::unique_lock<std::mutex> lock(eviction_mu_);
                // Wait for either the interval to elapse OR Close() to
                // signal stop_requested_ (notify_all is called from Close
                // right after the store, so the predicate will observe it).
                eviction_cv_.wait_for(
                    lock,
                    std::chrono::milliseconds(cfg_.eviction_interval_ms),
                    [this] {
                        return stop_requested_.load(std::memory_order_acquire);
                    });
            }
            if (stop_requested_.load(std::memory_order_acquire)) return;
            try {
                EvictOnce();
            } catch (...) {
                // Never let eviction failure propagate — engine health is
                // reported via sink->on_flush_error / stats fields.
            }
        }
    }

    /* Run one eviction pass across all sensors. Callable from the background
     * thread, from MaintainSlidingWindow, or from tests via ForceEvictOnce.
     *
     * Scaling note: this is O(sensors) per pass, and for each sensor it does
     * two O(window) scans (max_ts + prefix cutoff). Fine for the target
     * embedded use case (typically 1..dozens of sensors per device). If a
     * deployment ever has >1000 sensors, the right optimization is a
     * per-sensor "next-due" min-heap so we only visit sensors whose window
     * boundary has actually shifted. Deferred until real workloads demand it. */
    /* Run one eviction pass. `window_ms_override < 0` (the default) uses
     * cfg_.window_duration_ms; passing 0 forces "evict everything" (used
     * by Flush). Any positive override is honored as-is.
     *
     * SERIALIZATION: eviction_pass_mu_ ensures only one EvictOnce runs at
     * a time across (a) the background eviction thread and (b) concurrent
     * chronosv_flush calls from user threads. Both share evict_scratch_,
     * which is single-writer by contract — the mutex enforces it.
     * Contention is bounded (eviction is cold path, flush is rare). */
    void EvictOnce(std::int64_t window_ms_override = -1) {
        std::lock_guard<std::mutex> pass_lock(eviction_pass_mu_);
        const std::int64_t effective_window =
            (window_ms_override >= 0) ? window_ms_override : cfg_.window_duration_ms;
        std::shared_lock<std::shared_mutex> lock(map_mutex_);
        // Early return before iteration in the common "empty engine
        // waiting for first Append" state. Cheaper than paying for the
        // iterator machinery on an empty map, and clearer intent.
        if (sensors_.empty()) return;
        for (const auto& [id, slot_ptr] : sensors_) {
            EvictSensor(id, *slot_ptr, effective_window);
        }
    }

    void EvictSensor(const std::string& id, SensorSlot& slot,
                     std::int64_t window_ms) noexcept {
        SensorRing* ring = slot.ring.get();
        if (!ring) return;
        auto [begin, end] = ring->Snapshot();
        if (begin == end) return;

        std::uint64_t new_tail = begin;
        if (window_ms == 0) {
            /* window_ms == 0 is the Flush sentinel: evict EVERYTHING,
             * including the entry at max_ts. Normal windowed eviction
             * uses "ts < cutoff" which would leave the max_ts entry in
             * the ring — right for windowing, wrong for flush.
             * Config validation defaults window_ms=0 in cfg to
             * kDefaultWindowDurationMs, so this branch is only reached
             * from the internal Flush() code path. */
            new_tail = end;
        } else {
            // Find max ts observed in the snapshot. This is our reference for
            // cutoff — max_ts_seen defines the eviction frontier; out-of-order
            // late arrivals older than cutoff are eligible.
            std::int64_t max_ts = std::numeric_limits<std::int64_t>::min();
            for (std::uint64_t idx = begin; idx < end; ++idx) {
                const auto t = ring->timestamp_at(idx);
                if (t > max_ts) max_ts = t;
            }
            const std::int64_t cutoff = max_ts - window_ms;

            // Advance tail past the leading run of entries older than cutoff.
            // Non-leading old entries (from out-of-order appends) will be
            // evicted on subsequent passes as head advances — this is the
            // "prefix-eviction" contract documented in the header comment for
            // maintain_sliding_window.
            while (new_tail < end && ring->timestamp_at(new_tail) < cutoff) {
                new_tail++;
            }
        }
        if (new_tail == begin) return;   /* nothing to evict this pass */

        const std::uint64_t evicted = new_tail - begin;
        const std::size_t   count   = static_cast<std::size_t>(evicted);
        const std::uint32_t dim     = ring->dim();
        const chronosv_dtype_t dtype = ring->dtype();
        const std::uint32_t payload_size = ring->payload_size();
        const std::size_t   vec_bytes    = count * dim * dtype_size(dtype);

        /* --- Race-safety step: copy the evicted range into engine-owned
         * scratch BEFORE advancing tail. After tail advances,
         * the producer is free to overwrite these slots. Copy uses
         * chunk-aware ring reads because [begin, new_tail) may wrap the
         * physical buffer. --- */
        try {
            if (evict_scratch_.timestamps.size() < count) evict_scratch_.timestamps.resize(count);
            if (evict_scratch_.vectors.size() < vec_bytes) evict_scratch_.vectors.resize(vec_bytes);
            if (dtype == CHRONOSV_DTYPE_INT8 &&
                evict_scratch_.scales.size() < count) {
                evict_scratch_.scales.resize(count);
            }
            if (payload_size > 0 &&
                evict_scratch_.payloads.size() < count * payload_size) {
                evict_scratch_.payloads.resize(count * payload_size);
            }
        } catch (const std::bad_alloc&) {
            /* If we can't allocate scratch, we can't safely evict. Skip this
             * pass; ring will keep growing until a subsequent pass succeeds
             * or the producer overwrites. */
            return;
        }

        const float* scales_base = ring->scales_raw();  /* nullptr unless INT8 */
        const std::uint64_t mask = ring->mask();
        for (std::size_t i = 0; i < count; ++i) {
            const std::uint64_t idx = begin + i;
            evict_scratch_.timestamps[i] = ring->timestamp_at(idx);
            std::memcpy(evict_scratch_.vectors.data() + i * dim * dtype_size(dtype),
                        ring->vector_at(idx),
                        dim * dtype_size(dtype));
            if (dtype == CHRONOSV_DTYPE_INT8 && scales_base) {
                evict_scratch_.scales[i] = scales_base[idx & mask];
            }
            if (payload_size > 0) {
                std::memcpy(evict_scratch_.payloads.data() + i * payload_size,
                            ring->payload_at(idx),
                            payload_size);
            }
        }

        /* Now safe to advance tail — the scratch is a full deep copy. */
        ring->AdvanceTail(new_tail);
        total_evictions_.fetch_add(1, std::memory_order_relaxed);

        /* Persist to cold tier. NullStorageBackend swallows this;
         * RocksDBStorageBackend writes a block and returns OK/error. */
        std::int64_t block_bytes = 0;
        if (backend_) {
            Block blk{};
            blk.sensor_id          = id;
            blk.block_id           = slot.next_block_id.load(std::memory_order_relaxed);
            blk.count              = static_cast<std::uint32_t>(count);
            blk.dim                = dim;
            blk.dtype              = dtype;
            blk.payload_size_bytes = payload_size;
            blk.t_start_ms         = evict_scratch_.timestamps.front();
            blk.t_end_ms           = evict_scratch_.timestamps.back();
            blk.timestamps         = evict_scratch_.timestamps.data();
            blk.vectors            = evict_scratch_.vectors.data();
            blk.scales             = (dtype == CHRONOSV_DTYPE_INT8)
                                     ? evict_scratch_.scales.data() : nullptr;
            blk.payloads           = payload_size > 0
                                     ? evict_scratch_.payloads.data() : nullptr;

            const chronosv_error_t rc = backend_->WriteBlock(blk);
            if (rc == CHRONOSV_OK) {
                slot.next_block_id.fetch_add(1, std::memory_order_relaxed);
                /* Approximate on-disk bytes = block header + data + CRC.
                 * For the sink callback we report a rough size, not the
                 * post-compression size. */
                block_bytes = static_cast<std::int64_t>(
                    40 + count * (8 + dim * dtype_size(dtype)
                                  + (payload_size > 0 ? payload_size : 0)) + 4);
            } else {
                /* Persistence failed — data is already gone from the ring
                 * (we advanced tail above). Increment failure counter,
                 * notify the sink, but do NOT block the eviction thread
                 * on retry. Recovery is the caller's problem (e.g., they
                 * chose the wrong cold_path or disk is full). */
                flush_errors_total_.fetch_add(1, std::memory_order_relaxed);
                if (sink_ && sink_->on_flush_error) {
                    sink_->on_flush_error(sink_->user_data, id.c_str(), rc);
                }
            }
        }

        if (sink_ && sink_->on_eviction) {
            sink_->on_eviction(sink_->user_data, id.c_str(),
                               static_cast<std::int64_t>(evicted), block_bytes);
        }
    }

    /* Look up an existing ring; return nullptr if absent. R-lock only.
     * Currently unused externally — the direct find() pattern is used at
     * every call site to also access the slot's next_block_id if needed. */
    SensorRing* LookupRing(std::string_view sensor_id) {
        std::shared_lock<std::shared_mutex> lock(map_mutex_);
        auto it = sensors_.find(sensor_id);
        if (it == sensors_.end()) return nullptr;
        return it->second->ring.get();
    }

    /* recover_hot_window path: replay persisted blocks (newest
     * first) into a fresh ring, filtering entries older than
     * (now - window_duration_ms). Stops when ring hits capacity or an
     * older block's t_end_ms falls before the cutoff.
     *
     * Best-effort: per-block errors (corruption, IO) skip that block and
     * continue. The engine still opens successfully. */
    void RehydrateSensorRing(const std::string& sensor_id,
                             SensorSlot& slot,
                             const std::vector<std::int64_t>& block_ids,
                             std::int64_t cutoff_ms) noexcept {
        if (block_ids.empty()) return;
        if (!backend_) return;

        /* Allocate a ring for this sensor now — normally deferred to first
         * Append, but rehydration needs somewhere to put the entries. */
        std::unique_ptr<SensorRing> ring;
        try {
            ring = std::make_unique<SensorRing>(
                cfg_.ring_capacity, cfg_.dim, cfg_.payload_size_bytes,
                static_cast<chronosv_dtype_t>(cfg_.storage_dtype));
        } catch (const std::bad_alloc&) {
            return;   /* ring stays null, sensor is registered but empty */
        }
        if (!ring->ok()) return;

        /* Scratch for one block's data. Sized to the largest block we'll
         * see (bounded by the persisted block sizes). */
        std::vector<std::int64_t>  ts_buf;
        std::vector<std::uint8_t>  vec_buf;
        std::vector<float>         scale_buf;    /* INT8 only */
        std::vector<std::uint8_t>  payload_buf;
        std::size_t filled_slots = 0;

        /* Iterate newest → oldest. Once we hit a block whose t_end_ms is
         * older than cutoff, everything older is also stale — stop. */
        for (auto it = block_ids.rbegin(); it != block_ids.rend(); ++it) {
            /* Peek header to size scratch and check t_end_ms. Read the
             * full block into scratch, then filter and replay. */
            Block hdr_only{};
            const auto peek_err = backend_->ReadBlock(sensor_id, *it, hdr_only,
                                                     nullptr, nullptr, nullptr, nullptr);
            if (peek_err == CHRONOSV_ERR_CORRUPTION) continue;   /* skip bad block */
            if (peek_err != CHRONOSV_OK) return;                 /* IO or missing — stop */

            if (hdr_only.t_end_ms < cutoff_ms) return;           /* everything past is stale */

            /* Full read. */
            try {
                if (ts_buf.size() < hdr_only.count) ts_buf.resize(hdr_only.count);
                const std::size_t vsz = hdr_only.count * hdr_only.dim
                                        * dtype_size(hdr_only.dtype);
                if (vec_buf.size() < vsz) vec_buf.resize(vsz);
                if (hdr_only.dtype == CHRONOSV_DTYPE_INT8 &&
                    scale_buf.size() < hdr_only.count) {
                    scale_buf.resize(hdr_only.count);
                }
                if (hdr_only.payload_size_bytes > 0 &&
                    payload_buf.size() < hdr_only.count * hdr_only.payload_size_bytes) {
                    payload_buf.resize(hdr_only.count * hdr_only.payload_size_bytes);
                }
            } catch (const std::bad_alloc&) {
                return;
            }
            Block full{};
            const auto read_err = backend_->ReadBlock(
                sensor_id, *it, full,
                ts_buf.data(),
                vec_buf.data(),
                hdr_only.dtype == CHRONOSV_DTYPE_INT8 ? scale_buf.data() : nullptr,
                hdr_only.payload_size_bytes > 0 ? payload_buf.data() : nullptr);
            if (read_err == CHRONOSV_ERR_CORRUPTION) continue;
            if (read_err != CHRONOSV_OK) return;

            /* Replay entries in ts order. Skip anything older than cutoff. */
            const std::size_t vec_stride = static_cast<std::size_t>(full.dim)
                                          * dtype_size(full.dtype);
            for (std::uint32_t i = 0; i < full.count; ++i) {
                if (ts_buf[i] < cutoff_ms) continue;
                if (filled_slots >= cfg_.ring_capacity) return;
                /* Reconstruct the original float L2 norm from what we have. */
                float norm = 0.0f;
                float scale_i = 0.0f;
                if (full.dtype == CHRONOSV_DTYPE_FLOAT32) {
                    norm = L2NormF32(
                        reinterpret_cast<const float*>(vec_buf.data() + i * vec_stride),
                        full.dim);
                } else {  /* INT8 */
                    /* norm_original = scale * sqrt(sum(q_i^2)). */
                    scale_i = scale_buf[i];
                    const std::int8_t* q = reinterpret_cast<const std::int8_t*>(
                        vec_buf.data() + i * vec_stride);
                    std::int32_t acc = 0;
                    for (std::uint32_t j = 0; j < full.dim; ++j) {
                        const std::int16_t v = q[j];
                        acc += static_cast<std::int32_t>(v) * v;
                    }
                    norm = scale_i * std::sqrt(static_cast<float>(acc));
                }
                const void* payload_ptr = (full.payload_size_bytes > 0)
                    ? static_cast<const void*>(payload_buf.data()
                        + i * full.payload_size_bytes)
                    : nullptr;
                ring->Append(ts_buf[i],
                             vec_buf.data() + i * vec_stride,
                             norm,
                             payload_ptr,
                             scale_i);
                ++filled_slots;
            }
        }

        slot.ring = std::move(ring);
    }

    /* Fast path: R-lock lookup; slow path: upgrade to W-lock and insert. */
    SensorRing* GetOrCreateRing(std::string_view sensor_id,
                                chronosv_error_t* err) {
        // Fast path — reader lock, hope it exists. Uses heterogeneous
        // string_view lookup so no allocation happens here.
        {
            std::shared_lock<std::shared_mutex> lock(map_mutex_);
            auto it = sensors_.find(sensor_id);
            if (it != sensors_.end() && it->second->ring) {
                if (err) *err = CHRONOSV_OK;
                return it->second->ring.get();
            }
        }

        // Slow path — writer lock, double-checked insert.
        std::unique_lock<std::shared_mutex> lock(map_mutex_);
        auto it = sensors_.find(sensor_id);
        if (it != sensors_.end() && it->second->ring) {
            if (err) *err = CHRONOSV_OK;
            return it->second->ring.get();
        }

        if (it == sensors_.end()) {
            if (cfg_.max_sensors > 0 && sensors_.size() >= cfg_.max_sensors) {
                if (err) *err = CHRONOSV_ERR_CAPACITY;
                return nullptr;
            }
        }

        std::unique_ptr<SensorRing> ring;
        try {
            ring = std::make_unique<SensorRing>(
                cfg_.ring_capacity, cfg_.dim, cfg_.payload_size_bytes,
                static_cast<chronosv_dtype_t>(cfg_.storage_dtype));
        } catch (const std::bad_alloc&) {
            if (err) *err = CHRONOSV_ERR_OOM;
            return nullptr;
        }
        if (!ring->ok()) {
            if (err) *err = CHRONOSV_ERR_OOM;
            return nullptr;
        }
        auto* raw = ring.get();

        if (it == sensors_.end()) {
            /* Fresh slot. Insertion allocates a string key + a SensorSlot;
             * one-time per sensor (slow path), not per Append. */
            auto slot = std::make_unique<SensorSlot>();
            slot->ring = std::move(ring);
            sensors_.emplace(std::string(sensor_id), std::move(slot));
        } else {
            /* Slot exists from chronosv_open recovery but had no ring
             * because we defer ring allocation to first Append. Attach now. */
            it->second->ring = std::move(ring);
        }
        if (err) *err = CHRONOSV_OK;
        return raw;
    }

    /* Compute cosine/euclidean scores across [begin, begin+N) of the ring,
     * splitting the physical range into up to two contiguous chunks and
     * calling the kernel on each. Writes N floats into `scores_out`. */
    void FillScoresForRange(SensorRing*         ring,
                            std::uint64_t       begin,
                            std::size_t         N,
                            std::size_t         dim,
                            const float*        q_f32,
                            float               qn,
                            const std::int8_t*  q_i8,      /* non-null only when INT8 */
                            float               q_scale,   /* only used for INT8 */
                            bool                is_cosine,
                            float*              scores_out) noexcept {
        const std::uint64_t mask = ring->mask();
        const std::uint64_t start_slot = begin & mask;
        const float* norms_base = ring->norms_raw();
        const bool is_int8 = (ring->dtype() == CHRONOSV_DTYPE_INT8);

        const std::size_t first_len =
            std::min<std::size_t>(N, ring->capacity() - start_slot);

        if (is_int8) {
            const auto*  vecs_base_i8 = static_cast<const std::int8_t*>(ring->vectors_raw());
            const float* scales_base  = ring->scales_raw();

            auto run = [&](const std::int8_t* vecs, const float* scales,
                           const float* norms, std::size_t count, float* out) {
                if (count == 0) return;
                if (is_cosine) {
                    CosineI8Chunk(vecs, scales, norms, count, dim,
                                  q_i8, q_scale, qn, out);
                } else {
                    EuclideanSqI8Chunk(vecs, scales, norms, count, dim,
                                       q_i8, q_scale, qn, out);
                }
            };
            run(vecs_base_i8 + start_slot * dim,
                scales_base + start_slot,
                norms_base + start_slot,
                first_len,
                scores_out);
            if (first_len < N) {
                const std::size_t second_len = N - first_len;
                run(vecs_base_i8, scales_base, norms_base, second_len,
                    scores_out + first_len);
            }
            return;
        }

        /* FLOAT32 path. */
        const auto* vecs_base = static_cast<const float*>(ring->vectors_raw());
        auto run = [&](const float* vecs, const float* norms,
                       std::size_t count, float* out) {
            if (count == 0) return;
            if (is_cosine) {
                CosineF32Chunk(vecs, norms, count, dim, q_f32, qn, out);
            } else {
                EuclideanSqF32Chunk(vecs, norms, count, dim, q_f32, qn, out);
            }
        };
        run(vecs_base + start_slot * dim,
            norms_base + start_slot,
            first_len,
            scores_out);
        if (first_len < N) {
            const std::size_t second_len = N - first_len;
            run(vecs_base, norms_base, second_len,
                scores_out + first_len);
        }
    }

    chronosv_config_t cfg_;
    std::uint8_t      uuid_[16];

    mutable std::shared_mutex map_mutex_;
    SensorMap                 sensors_;   // heterogeneous string_view lookup

    const chronosv_metrics_sink_t* sink_ = nullptr;

    /* Cold-tier storage. NullStorageBackend when no cold_path is configured,
     * RocksDBStorageBackend otherwise. Always non-null after successful
     * construction. */
    std::unique_ptr<StorageBackend> backend_;

    /* Scratch used by EvictSensor to copy the evicted range out of the ring
     * before advancing tail. Single-writer (eviction thread) so no lock. */
    EvictScratch evict_scratch_;

    std::atomic<std::uint64_t> total_appends_{0};
    std::atomic<std::uint64_t> total_queries_{0};
    std::atomic<std::uint64_t> total_anomaly_checks_{0};
    std::atomic<std::uint64_t> total_evictions_{0};
    std::atomic<std::uint64_t> total_dropped_sensors_{0};
    std::atomic<std::uint64_t> flush_errors_total_{0};

    /* Overwrite counts absorbed from dropped sensors so the lifetime total
     * in stats survives DropSensor. Live rings contribute their current
     * counts via GetStats iteration; dropped sensors contribute here. */
    std::atomic<std::uint64_t> dropped_overwrite_events_{0};
    std::atomic<std::uint64_t> dropped_overwritten_entries_{0};

    std::atomic<bool> closed_{false};

    /* Serializes EvictOnce calls (background thread + chronosv_flush).
     * evict_scratch_ is single-writer by contract — this mutex enforces it. */
    std::mutex eviction_pass_mu_;

    /* Eviction worker. Must be declared LAST so it destructs FIRST — the
     * thread dtor terminates on a joinable thread, so Close() (which joins)
     * must run before sensors_ / map_mutex_ / backend_ start tearing down.
     * The ~Engine destructor calls Close() to enforce this ordering. */
    std::mutex                 eviction_mu_;
    std::condition_variable    eviction_cv_;
    std::atomic<bool>          stop_requested_{false};
    std::thread                eviction_thread_;
};

inline Engine* to_engine(chronosv_engine_t* h) noexcept {
    return reinterpret_cast<Engine*>(h);
}
inline const Engine* to_engine(const chronosv_engine_t* h) noexcept {
    return reinterpret_cast<const Engine*>(h);
}
inline chronosv_engine_t* from_engine(Engine* e) noexcept {
    return reinterpret_cast<chronosv_engine_t*>(e);
}

/* -------------------------------------------------------------------------- */
/* Exception-catching macro for the extern "C" wall                            */
/* -------------------------------------------------------------------------- */

}  // anonymous namespace

#define CHRONOSV_CATCH_ALL                                                     \
    catch (const std::bad_alloc&)        { return CHRONOSV_ERR_OOM;         } \
    catch (const std::system_error&)     { return CHRONOSV_ERR_IO;          } \
    catch (const std::invalid_argument&) { return CHRONOSV_ERR_INVALID_ARG; } \
    catch (const std::exception&)        { return CHRONOSV_ERR_INTERNAL;    } \
    catch (...)                          { return CHRONOSV_ERR_INTERNAL;    }

/* ========================================================================== */
/* extern "C" wall                                                             */
/* ========================================================================== */

extern "C" {

/* ---- Utility ---- */

const char* chronosv_error_string(chronosv_error_t err) {
    switch (err) {
        case CHRONOSV_OK:                    return "ok";
        case CHRONOSV_WARN_RANGE_TRUNCATED:  return "warning: range truncated to hot window";
        case CHRONOSV_WARN_PARTIAL_RESULT:   return "warning: fewer results than requested";
        case CHRONOSV_ERR_INVALID_ARG:       return "error: invalid argument";
        case CHRONOSV_ERR_NOT_FOUND:         return "error: sensor not found";
        case CHRONOSV_ERR_DIM_MISMATCH:      return "error: vector dim does not match engine dim";
        case CHRONOSV_ERR_IO:                return "error: I/O failure";
        case CHRONOSV_ERR_OOM:               return "error: out of memory";
        case CHRONOSV_ERR_UNSUPPORTED:       return "error: feature not compiled in / not yet implemented";
        case CHRONOSV_ERR_CLOSED:            return "error: engine is closed";
        case CHRONOSV_ERR_CAPACITY:          return "error: max_sensors exceeded";
        case CHRONOSV_ERR_CORRUPTION:        return "error: persisted block failed magic/version/CRC/format validation";
        case CHRONOSV_ERR_INTERNAL:          return "error: internal";
        default:                             return "unknown";
    }
}

const char* chronosv_version_string(void) {
    return "0.1.0";
}

/* ---- Lifecycle ---- */

chronosv_engine_t* chronosv_create(const chronosv_config_t* cfg,
                                   chronosv_error_t* err_out) {
    if (!cfg) {
        if (err_out) *err_out = CHRONOSV_ERR_INVALID_ARG;
        return nullptr;
    }
    try {
        return from_engine(Engine::Create(*cfg, err_out));
    } catch (const std::bad_alloc&)   { if (err_out) *err_out = CHRONOSV_ERR_OOM;      return nullptr; }
      catch (const std::exception&)   { if (err_out) *err_out = CHRONOSV_ERR_INTERNAL; return nullptr; }
      catch (...)                     { if (err_out) *err_out = CHRONOSV_ERR_INTERNAL; return nullptr; }
}

chronosv_error_t chronosv_open(const char* path, chronosv_engine_t** out) {
    if (!path || !out) return CHRONOSV_ERR_INVALID_ARG;
    *out = nullptr;
    chronosv_error_t err = CHRONOSV_OK;
    try {
        Engine* e = Engine::Open(path, &err);
        if (!e) return err;
        *out = from_engine(e);
        return CHRONOSV_OK;
    }
    catch (const std::bad_alloc&)   { return CHRONOSV_ERR_OOM;      }
    catch (const std::exception&)   { return CHRONOSV_ERR_INTERNAL; }
    catch (...)                     { return CHRONOSV_ERR_INTERNAL; }
}

chronosv_error_t chronosv_close(chronosv_engine_t* engine) {
    if (!engine) return CHRONOSV_ERR_INVALID_ARG;
    try { return to_engine(engine)->Close(); } CHRONOSV_CATCH_ALL
}

void chronosv_destroy(chronosv_engine_t* engine) {
    if (!engine) return;
    /* Non-throwing dtors expected; if anything escapes, swallow — a
     * destructor throwing across extern "C" would be worse. */
    try { delete to_engine(engine); } catch (...) { /* noreturn-safe swallow */ }
}

chronosv_error_t chronosv_preallocate_sensor(chronosv_engine_t* engine,
                                             const char* sensor_id) {
    if (!engine) return CHRONOSV_ERR_INVALID_ARG;
    const auto vid = validate_sensor_id(sensor_id);
    if (vid != CHRONOSV_OK) return vid;
    try {
        return to_engine(engine)->PreallocateSensor(sensor_id);
    } CHRONOSV_CATCH_ALL
}

/* ---- Ingest ---- */

chronosv_error_t chronosv_append(chronosv_engine_t* engine,
                                 const char* sensor_id,
                                 int64_t ts_ms,
                                 const float* vec,
                                 size_t dim,
                                 const void* payload) {
    if (!engine || !vec || dim == 0) return CHRONOSV_ERR_INVALID_ARG;
    const auto vid = validate_sensor_id(sensor_id);
    if (vid != CHRONOSV_OK) return vid;
    try {
        return to_engine(engine)->Append(sensor_id, ts_ms, vec, dim, payload);
    } CHRONOSV_CATCH_ALL
}

chronosv_error_t chronosv_append_batch(chronosv_engine_t* engine,
                                       const char* sensor_id,
                                       const int64_t* ts_ms,
                                       const float* vecs_row_major,
                                       size_t count,
                                       size_t dim,
                                       const void* payloads) {
    if (!engine || dim == 0) return CHRONOSV_ERR_INVALID_ARG;
    const auto vid = validate_sensor_id(sensor_id);
    if (vid != CHRONOSV_OK) return vid;
    if (count == 0) return CHRONOSV_OK;
    if (!ts_ms || !vecs_row_major) return CHRONOSV_ERR_INVALID_ARG;
    try {
        return to_engine(engine)->AppendBatch(sensor_id, ts_ms, vecs_row_major,
                                              count, dim, payloads);
    } CHRONOSV_CATCH_ALL
}

/* ---- Query ---- */

chronosv_error_t chronosv_query_nearest_n(chronosv_engine_t* engine,
                                          const char* sensor_id,
                                          const float* target,
                                          size_t dim,
                                          int n,
                                          int64_t* out_ts,
                                          float* out_scores,
                                          int* out_count) {
    if (!engine) return CHRONOSV_ERR_INVALID_ARG;
    const auto vid = validate_sensor_id(sensor_id);
    if (vid != CHRONOSV_OK) return vid;
    try {
        return to_engine(engine)->QueryNearestN(sensor_id, target, dim, n,
                                                out_ts, out_scores, out_count);
    } CHRONOSV_CATCH_ALL
}

chronosv_error_t chronosv_query_range(chronosv_engine_t* engine,
                                      const char* sensor_id,
                                      int64_t t_start_ms,
                                      int64_t t_end_ms,
                                      int64_t* out_ts,
                                      float* out_vecs,
                                      int max,
                                      int* out_count) {
    if (!engine) return CHRONOSV_ERR_INVALID_ARG;
    const auto vid = validate_sensor_id(sensor_id);
    if (vid != CHRONOSV_OK) return vid;
    try {
        return to_engine(engine)->QueryRange(sensor_id, t_start_ms, t_end_ms,
                                             out_ts, out_vecs, max, out_count);
    } CHRONOSV_CATCH_ALL
}

chronosv_error_t chronosv_detect_anomaly(chronosv_engine_t* engine,
                                         const char* sensor_id,
                                         const float* v,
                                         size_t dim,
                                         float threshold,
                                         int* out_is_anomaly) {
    if (!engine) return CHRONOSV_ERR_INVALID_ARG;
    const auto vid = validate_sensor_id(sensor_id);
    if (vid != CHRONOSV_OK) return vid;
    try {
        return to_engine(engine)->DetectAnomaly(sensor_id, v, dim, threshold, out_is_anomaly);
    } CHRONOSV_CATCH_ALL
}

/* ---- Maintenance ---- */

chronosv_error_t chronosv_maintain_sliding_window(chronosv_engine_t* engine,
                                                  int64_t window_ms) {
    if (!engine) return CHRONOSV_ERR_INVALID_ARG;
    try { return to_engine(engine)->MaintainSlidingWindow(window_ms); } CHRONOSV_CATCH_ALL
}

chronosv_error_t chronosv_drop_sensor(chronosv_engine_t* engine,
                                      const char* sensor_id) {
    if (!engine) return CHRONOSV_ERR_INVALID_ARG;
    const auto vid = validate_sensor_id(sensor_id);
    if (vid != CHRONOSV_OK) return vid;
    try { return to_engine(engine)->DropSensor(sensor_id); } CHRONOSV_CATCH_ALL
}

chronosv_error_t chronosv_flush(chronosv_engine_t* engine) {
    if (!engine) return CHRONOSV_ERR_INVALID_ARG;
    try { return to_engine(engine)->Flush(); } CHRONOSV_CATCH_ALL
}

/* ---- Observability ---- */

chronosv_error_t chronosv_list_sensors(const chronosv_engine_t* engine,
                                       char** out_ids,
                                       size_t max,
                                       size_t* out_count) {
    if (!engine) return CHRONOSV_ERR_INVALID_ARG;
    try { return to_engine(engine)->ListSensors(out_ids, max, out_count); } CHRONOSV_CATCH_ALL
}

chronosv_error_t chronosv_get_stats(const chronosv_engine_t* engine,
                                    chronosv_stats_t* out) {
    if (!engine) return CHRONOSV_ERR_INVALID_ARG;
    try { return to_engine(engine)->GetStats(out); } CHRONOSV_CATCH_ALL
}

}  // extern "C"
