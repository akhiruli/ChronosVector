/**
 * @file types.h
 * @brief Shared C types for the ChronosVector public ABI.
 *
 * Defines PODs, enums, and error codes shared between ::chronos_vector.h,
 * ::metrics_sink.h, and internal C++ implementation code. Consumers should
 * include `chronos_vector.h` — this header is pulled in transitively.
 */
#ifndef CHRONOSV_TYPES_H
#define CHRONOSV_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------- */
/* ABI version                                                                */
/* -------------------------------------------------------------------------- */

/** Current ABI version. Bumped on any breaking change to the C ABI.
 *  `chronosv_create` rejects configs with a mismatching `abi_version`. */
#define CHRONOSV_ABI_VERSION 1

/* -------------------------------------------------------------------------- */
/* Error / warning codes                                                      */
/* -------------------------------------------------------------------------- */

/**
 * @brief Signed 32-bit return code.
 *
 * Sign convention (locked in — do not change post-1.0):
 * - `0` = success
 * - positive = warning (out-params valid, caller may want to log)
 * - negative = error (out-params undefined, caller must not read them)
 */
typedef int32_t chronosv_error_t;

#define CHRONOSV_OK                        0    /**< Success. */

/** @name Warnings (positive; out-params are valid)
 * @{ */
#define CHRONOSV_WARN_RANGE_TRUNCATED      1    /**< `query_range`: `t_start_ms` fell before oldest hot entry. */
#define CHRONOSV_WARN_PARTIAL_RESULT       2    /**< Requested `n > window population`. */
/** @} */

/** @name Errors (negative; out-params are undefined)
 * @{ */
#define CHRONOSV_ERR_INVALID_ARG          -1    /**< Null pointer, zero dim, malformed config. */
#define CHRONOSV_ERR_NOT_FOUND            -2    /**< Unknown `sensor_id`. */
#define CHRONOSV_ERR_DIM_MISMATCH         -3    /**< Vector `dim` != engine `dim`. */
#define CHRONOSV_ERR_IO                   -4    /**< RocksDB or filesystem error. */
#define CHRONOSV_ERR_OOM                  -5    /**< `std::bad_alloc` caught at boundary. */
#define CHRONOSV_ERR_UNSUPPORTED          -6    /**< Feature disabled at compile time (e.g. INT8). */
#define CHRONOSV_ERR_CLOSED               -7    /**< Operation on a closed engine. */
#define CHRONOSV_ERR_CAPACITY             -8    /**< `max_sensors` exceeded. */
/**
 * Persisted block failed magic/version/CRC/format validation. Distinct from
 * ::CHRONOSV_ERR_INVALID_ARG (which means the *caller* passed something bad);
 * corruption means the bytes on disk are wrong. Storage backends translate
 * this into `on_corruption` sink events.
 */
#define CHRONOSV_ERR_CORRUPTION           -9
#define CHRONOSV_ERR_INTERNAL            -99    /**< Caught-but-unclassified exception. */
/** @} */

/* -------------------------------------------------------------------------- */
/* Enums                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * @brief Storage dtype for the ring buffer.
 *
 * ::CHRONOSV_DTYPE_INT8 is compiled in by default (`CHRONOSV_ENABLE_INT8=ON`).
 * If the library was built with the flag explicitly OFF, requesting INT8
 * at engine construction returns ::CHRONOSV_ERR_UNSUPPORTED.
 *
 * INT8 trades a small accuracy loss (measured 0.7 pp Recall@10 on BERT
 * embeddings, 3.1 pp on SIFT-1M) for ~4× memory reduction and 2.5–4.5×
 * query speedup on memory-bandwidth-bound workloads. See `docs/INT8.md`
 * for when-to-use guidance and the validation harness at
 * `tests/int8_recall/` for measuring recall on your own embeddings.
 */
typedef enum chronosv_dtype_t {
    CHRONOSV_DTYPE_FLOAT32 = 0,  /**< 32-bit float. Default. */
    CHRONOSV_DTYPE_INT8    = 1   /**< 8-bit int with per-vector scale factor. */
} chronosv_dtype_t;

/** Distance metric, pinned per engine (not per query). */
typedef enum chronosv_metric_t {
    CHRONOSV_METRIC_COSINE    = 0,  /**< Cosine similarity. Default. */
    CHRONOSV_METRIC_EUCLIDEAN = 1   /**< Euclidean distance. */
} chronosv_metric_t;

/* -------------------------------------------------------------------------- */
/* Config (chronosv_create input)                                             */
/* -------------------------------------------------------------------------- */

/* Forward decl — full definition in metrics_sink.h. */
typedef struct chronosv_metrics_sink_t chronosv_metrics_sink_t;

/**
 * @brief Configuration passed to ::chronosv_create.
 *
 * Zero-initialize with `{0}` and set the required fields (`abi_version`,
 * `dim`); everything else picks up documented defaults.
 */
typedef struct chronosv_config_t {
    /** @name Required
     * @{ */
    uint32_t     abi_version;          /**< Must equal ::CHRONOSV_ABI_VERSION. */
    const char*  cold_path;            /**< RocksDB directory. NULL = in-memory only (no persistence). */
    uint32_t     dim;                  /**< Vector dimension, fixed per engine, must be > 0. */
    /** @} */

    /** @name Optional (zero-initialized = default)
     * @{ */
    uint8_t      uuid[16];             /**< All-zero => engine auto-generates a v4 UUID. */
    uint64_t     ring_capacity;        /**< Per-sensor; power of 2; default 65536. */
    int64_t      window_duration_ms;   /**< Sliding window duration. Default 600000 (10 min). */
    int64_t      eviction_interval_ms; /**< Background eviction cadence. Default 60000 (60 s). */
    uint32_t     payload_size_bytes;   /**< Fixed-size opaque payload per vector. 0 disables. */
    uint8_t      storage_dtype;        /**< ::chronosv_dtype_t; default ::CHRONOSV_DTYPE_FLOAT32. */
    uint8_t      distance_metric;      /**< ::chronosv_metric_t; default ::CHRONOSV_METRIC_COSINE. */
    uint32_t     max_sensors;          /**< 0 = unbounded (not recommended in production). */
    /** @} */

    /** @name Observability (see `metrics_sink.h`)
     * @{ */
    const chronosv_metrics_sink_t* metrics_sink;  /**< Nullable; not owned by engine. */
    /** @} */

    /** @name Recovery
     * @{ */
    uint8_t      recover_hot_window;   /**< 0 = fresh ring on open; 1 = rehydrate ring from cold tier. */
    /** @} */
} chronosv_config_t;

/* -------------------------------------------------------------------------- */
/* Stats (chronosv_get_stats output)                                          */
/* -------------------------------------------------------------------------- */

/**
 * @brief Snapshot of engine health, returned by ::chronosv_get_stats.
 *
 * Reads are lock-free and eventually consistent — a snapshot may show one
 * field reflecting N appends and another reflecting N+1. Callers must not
 * rely on cross-field consistency.
 */
typedef struct chronosv_stats_t {
    uint8_t  uuid[16];                    /**< Engine identity. */
    uint32_t abi_version;                 /**< Copy of ::CHRONOSV_ABI_VERSION at build. */

    /** @name Sensor accounting
     * @{ */
    uint32_t sensor_count;                /**< Currently registered sensors. */
    uint32_t sensor_cap;                  /**< Config `max_sensors`. */
    /** @} */

    /** @name Cumulative counters
     * @{ */
    uint64_t total_appends;
    uint64_t total_queries;
    uint64_t total_anomaly_checks;
    uint64_t total_evictions;
    uint64_t total_dropped_sensors;
    /** @} */

    /**
     * @name Data-loss telemetry
     * @brief Non-zero here means the producer is outpacing eviction and data
     *        is silently overwritten (design-permitted ring behavior). **Watch this.**
     * @{
     */
    uint64_t total_overwrite_events;      /**< Number of append calls that overwrote a live entry. */
    uint64_t total_overwritten_entries;   /**< Total entries lost (currently == events; may differ if batching). */
    /** @} */

    /** @name Memory
     * @{ */
    uint64_t hot_bytes;                   /**< Sum of ring allocations. */
    uint64_t cold_bytes_estimate;         /**< Approximate; 0 in Phase 1. */
    /** @} */

    /**
     * @name Latency (nanoseconds; approximate percentiles from lock-free buckets)
     *       Callers that need latency numbers should measure externally and
     *       feed a ::chronosv_metrics_sink_t for aggregation.
     * @{
     */
    uint64_t append_p50_ns;
    uint64_t append_p99_ns;
    uint64_t query_p50_ns;
    uint64_t query_p99_ns;
    /** @} */

    /** @name Health
     * @{ */
    uint64_t last_flush_unix_ms;
    uint64_t flush_errors_total;
    uint64_t corruption_events_total;
    /** @} */
} chronosv_stats_t;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CHRONOSV_TYPES_H */
