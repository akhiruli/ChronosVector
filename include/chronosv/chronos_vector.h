/**
 * @file chronos_vector.h
 * @brief ChronosVector public C ABI — the stable, semver-locked contract.
 *
 * This header defines the entire public surface of ChronosVector. Everything
 * else (internal C++ implementation, RocksDB schema, ring buffer layout) is
 * allowed to change between minor versions. This header is not, except across
 * a major version bump.
 *
 * @section api_grouping The fifteen primitives
 *
 * Grouped by concern, in the order they appear below:
 *
 * | Group         | Count | Functions                                                      |
 * |---------------|-------|----------------------------------------------------------------|
 * | Lifecycle     | 5     | create, open, close, destroy, preallocate_sensor               |
 * | Ingest        | 2     | append, append_batch                                           |
 * | Query         | 3     | query_nearest_n, query_range, detect_anomaly                   |
 * | Maintenance   | 3     | maintain_sliding_window, drop_sensor, flush                    |
 * | Observability | 2     | list_sensors, get_stats                                        |
 *
 * @section api_errors Error handling
 *
 * Every function returns a ::chronosv_error_t code. `0` is success; positive
 * values are warnings (out-params are valid, caller may want to log); negative
 * values are errors (out-params are undefined and must not be read). See
 * `types.h` for the full code table and ::chronosv_error_string() for
 * human-readable messages.
 *
 * @section api_threading Threading model
 *
 * Per-sensor SPSC: **one producer thread per sensor_id** for ::chronosv_append
 * / ::chronosv_append_batch / ::chronosv_preallocate_sensor. Query functions
 * are multi-consumer safe and may be called concurrently from any thread.
 * Lifecycle functions (::chronosv_create, ::chronosv_close,
 * ::chronosv_destroy) must not be called concurrently with anything else on
 * the same handle.
 */
#ifndef CHRONOS_VECTOR_H
#define CHRONOS_VECTOR_H

#include <stddef.h>
#include <stdint.h>

#include "chronosv/types.h"
#include "chronosv/metrics_sink.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------- */
/* Opaque handle                                                              */
/* -------------------------------------------------------------------------- */

/** Opaque engine handle. Created by ::chronosv_create or ::chronosv_open,
 *  freed by ::chronosv_destroy. Do not dereference. */
typedef struct chronosv_engine chronosv_engine_t;

/* -------------------------------------------------------------------------- */
/* Utility                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * @brief Human-readable string for an error or warning code.
 * @param err Any ::chronosv_error_t value.
 * @return Static string; caller must not free. Returns `"unknown"` for codes
 *         not defined by this ABI.
 */
const char* chronosv_error_string(chronosv_error_t err);

/**
 * @brief Library semantic version string, e.g. `"0.1.0"`.
 * @return Static string; caller must not free.
 */
const char* chronosv_version_string(void);

/* -------------------------------------------------------------------------- */
/* Lifecycle (5)                                                              */
/* -------------------------------------------------------------------------- */

/**
 * @brief Create a fresh engine with the given config.
 *
 * @param cfg     Configuration; must not be NULL. Zero-initialize with `{0}`
 *                and set at minimum `abi_version` (to ::CHRONOSV_ABI_VERSION)
 *                and `dim`. See ::chronosv_config_t for field semantics.
 * @param err_out Optional. If non-NULL, receives ::CHRONOSV_OK on success or
 *                the specific error code on failure.
 * @return Owning handle on success, `NULL` on failure. Caller must eventually
 *         call ::chronosv_destroy.
 * @retval NULL + ::CHRONOSV_ERR_INVALID_ARG  Null cfg, zero dim, bad abi_version.
 * @retval NULL + ::CHRONOSV_ERR_OOM          Backing allocation failed.
 * @retval NULL + ::CHRONOSV_ERR_IO           `cold_path` provided but RocksDB open failed.
 * @retval NULL + ::CHRONOSV_ERR_UNSUPPORTED  `storage_dtype = INT8` without CHRONOSV_ENABLE_INT8.
 * @note If `cfg->cold_path` points to a directory containing a previous
 *       engine's metadata, the schema (dim / dtype / metric / payload_size /
 *       ring_capacity) must match. Mismatch returns ::CHRONOSV_ERR_INVALID_ARG.
 */
chronosv_engine_t* chronosv_create(const chronosv_config_t* cfg,
                                   chronosv_error_t* err_out);

/**
 * @brief Open an engine at an existing cold-tier path.
 *
 * Enumerates sensors that already have persisted blocks, registers them in
 * the engine with empty rings (allocated on first append per sensor), and
 * seeds each sensor's next block id from the maximum persisted id + 1.
 *
 * Config is read from persisted metadata (dim, dtype, metric, payload_size,
 * ring_capacity). If your workload needs a different schema, use
 * ::chronosv_create with the new config against a fresh directory instead.
 *
 * @param path Path to a directory previously used as an engine's `cold_path`.
 * @param[out] out Non-NULL. Receives an owning handle on success. Caller must
 *                 eventually call ::chronosv_destroy.
 * @return ::CHRONOSV_OK on success.
 * @retval ::CHRONOSV_ERR_INVALID_ARG Null path or null out.
 * @retval ::CHRONOSV_ERR_IO          RocksDB open failed or metadata missing.
 * @retval ::CHRONOSV_ERR_CORRUPTION  Persisted metadata failed magic/CRC checks.
 */
chronosv_error_t chronosv_open(const char* path, chronosv_engine_t** out);

/**
 * @brief Flush hot data, join background threads, mark the engine closed.
 *
 * Idempotent. After this returns, all API calls on this handle return
 * ::CHRONOSV_ERR_CLOSED. The handle itself is not freed — use
 * ::chronosv_destroy for that.
 *
 * @param engine Handle. NULL is a no-op (returns ::CHRONOSV_OK).
 * @return ::CHRONOSV_OK on success, or the underlying storage error if the
 *         final flush failed.
 */
chronosv_error_t chronosv_close(chronosv_engine_t* engine);

/**
 * @brief Free the handle, implicitly closing if not already closed.
 *
 * Safe on NULL. After return, the handle is invalid and must not be used.
 *
 * @param engine Handle (may be NULL).
 */
void chronosv_destroy(chronosv_engine_t* engine);

/**
 * @brief Pre-allocate a sensor's ring buffer so the first ::chronosv_append
 *        does not include a cold-path allocation.
 *
 * Useful when the caller needs strict steady-state append latency from cycle
 * zero (e.g. hard real-time producers). Idempotent — calling twice on the
 * same sensor is a successful no-op.
 *
 * @param engine    Handle.
 * @param sensor_id Sensor identifier (NUL-terminated).
 * @return ::CHRONOSV_OK on success.
 * @retval ::CHRONOSV_ERR_CAPACITY  `max_sensors` already reached.
 */
chronosv_error_t chronosv_preallocate_sensor(chronosv_engine_t* engine,
                                             const char* sensor_id);

/* -------------------------------------------------------------------------- */
/* Ingest (2)                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * @brief Append one vector for a sensor. Hot-path operation.
 *
 * @param engine    Handle.
 * @param sensor_id Sensor identifier (NUL-terminated). New sensors are
 *                  registered on first append.
 * @param ts_ms     Caller-provided wall-clock timestamp in milliseconds.
 *                  Monotonicity across appends is NOT required.
 * @param vec       Pointer to `dim` floats.
 * @param dim       Must equal the engine's configured `dim`; otherwise
 *                  ::CHRONOSV_ERR_DIM_MISMATCH.
 * @param payload   Optional. If non-NULL and the engine was configured with
 *                  `payload_size_bytes > 0`, exactly that many bytes are
 *                  copied. Ignored otherwise.
 * @return ::CHRONOSV_OK on success.
 * @retval ::CHRONOSV_ERR_DIM_MISMATCH  `dim` does not match engine config.
 * @retval ::CHRONOSV_ERR_INVALID_ARG   Null sensor_id or null vec.
 * @retval ::CHRONOSV_ERR_CAPACITY      New sensor and `max_sensors` reached.
 *
 * @warning SPSC contract: exactly one producer thread per `sensor_id`. Calls
 *          from multiple threads on the same sensor are undefined behavior
 *          in release builds; debug builds assert.
 * @note When the ring is full, the oldest entry is overwritten and the
 *       overwrite counters (::chronosv_stats_t::total_overwrite_events) are
 *       bumped. Silent data loss is by design when the producer outpaces
 *       eviction — monitor the counter.
 */
chronosv_error_t chronosv_append(chronosv_engine_t* engine,
                                 const char* sensor_id,
                                 int64_t ts_ms,
                                 const float* vec,
                                 size_t dim,
                                 const void* payload);

/**
 * @brief Batched append. Equivalent to a loop of ::chronosv_append but avoids
 *        per-call sensor lookup and error-checking overhead.
 *
 * @param engine          Handle.
 * @param sensor_id       Sensor identifier (NUL-terminated).
 * @param ts_ms           Array of `count` timestamps.
 * @param vecs_row_major  Contiguous `[count][dim]` float array.
 * @param count           Number of vectors to append.
 * @param dim             Must equal the engine's configured `dim`.
 * @param payloads        Optional `[count][payload_size_bytes]` array. NULL to skip.
 * @return ::CHRONOSV_OK on success. Same error semantics as ::chronosv_append.
 * @warning Same SPSC contract as ::chronosv_append.
 */
chronosv_error_t chronosv_append_batch(chronosv_engine_t* engine,
                                       const char* sensor_id,
                                       const int64_t* ts_ms,
                                       const float* vecs_row_major,
                                       size_t count,
                                       size_t dim,
                                       const void* payloads);

/* -------------------------------------------------------------------------- */
/* Query (3) — all hot-window only in v0.1                                    */
/* -------------------------------------------------------------------------- */

/**
 * @brief k-nearest-neighbor query over the current hot window.
 *
 * Distance metric is the one pinned at engine construction
 * (`cfg.distance_metric`). Higher scores are more similar for cosine; lower
 * are more similar for euclidean. Results are sorted best-first.
 *
 * @param engine    Handle.
 * @param sensor_id Sensor identifier.
 * @param target    Query vector, `dim` floats.
 * @param dim       Must equal the engine's configured `dim`.
 * @param n         Number of results requested. Output buffers must be at
 *                  least this large.
 * @param[out] out_ts     Buffer of `n` `int64_t` — timestamps of results.
 * @param[out] out_scores Buffer of `n` floats — distances or similarities.
 * @param[out] out_count  Actual number of results written (may be < n).
 * @return ::CHRONOSV_OK on success.
 * @retval ::CHRONOSV_WARN_PARTIAL_RESULT  The window had fewer than `n` entries.
 *                                          Out-params are valid; `*out_count < n`.
 * @retval ::CHRONOSV_ERR_NOT_FOUND        Unknown sensor_id.
 * @retval ::CHRONOSV_ERR_DIM_MISMATCH     `dim` mismatch.
 */
chronosv_error_t chronosv_query_nearest_n(chronosv_engine_t* engine,
                                          const char* sensor_id,
                                          const float* target,
                                          size_t dim,
                                          int n,
                                          int64_t* out_ts,
                                          float* out_scores,
                                          int* out_count);

/**
 * @brief Timestamp-range query over the current hot window.
 *
 * Returns entries with `ts_ms` in the closed interval `[t_start_ms,
 * t_end_ms]`, in insertion order.
 *
 * @param engine    Handle.
 * @param sensor_id Sensor identifier.
 * @param t_start_ms Start of range (inclusive).
 * @param t_end_ms   End of range (inclusive).
 * @param[out] out_ts    Buffer of at least `max` `int64_t`s for timestamps.
 * @param[out] out_vecs  Buffer of at least `max * dim` floats, row-major.
 * @param max            Capacity of the output buffers.
 * @param[out] out_count Number of entries actually written (<= max).
 * @return ::CHRONOSV_OK on success.
 * @retval ::CHRONOSV_WARN_RANGE_TRUNCATED  `t_start_ms` falls before the oldest
 *   hot entry. **Older data lives in the cold tier and is NOT returned by
 *   this call in v0.1**. Callers should fall back to their own cold-tier
 *   query, or upgrade to a future `chronosv_query_cold_range` if/when added.
 * @retval ::CHRONOSV_ERR_NOT_FOUND         Unknown sensor_id.
 *
 * @warning This call is hot-window only. Silent truncation would be the worst
 *          possible outcome; the warning code exists specifically so callers
 *          can detect the boundary.
 */
chronosv_error_t chronosv_query_range(chronosv_engine_t* engine,
                                      const char* sensor_id,
                                      int64_t t_start_ms,
                                      int64_t t_end_ms,
                                      int64_t* out_ts,
                                      float* out_vecs,
                                      int max,
                                      int* out_count);

/**
 * @brief Threshold-based anomaly detection against the rolling window centroid.
 *
 * Semantics:
 *   1. Compute the mean vector across all entries currently in the sensor's
 *      hot window.
 *   2. Compute `distance(mean, v)` using the engine's configured metric.
 *   3. Flag as anomaly if `distance > threshold`.
 *
 * Threshold units depend on `cfg.distance_metric`:
 *
 * | Metric                        | Threshold units                          | Typical values                     |
 * |-------------------------------|------------------------------------------|------------------------------------|
 * | ::CHRONOSV_METRIC_COSINE      | Cosine DISTANCE = `1 - cos_sim`, range [0, 2] | 0.3–0.5 = "far from typical dir"   |
 * | ::CHRONOSV_METRIC_EUCLIDEAN   | Euclidean distance (NOT squared), same units as input | Scale to your signal magnitude      |
 *
 * @param engine    Handle.
 * @param sensor_id Sensor identifier.
 * @param v         Candidate vector, `dim` floats.
 * @param dim       Must equal the engine's configured `dim`.
 * @param threshold Distance above which the vector is flagged as anomalous.
 * @param[out] out_is_anomaly `0` = normal, `1` = anomaly.
 * @return ::CHRONOSV_OK. Empty sensors return OK with `*out_is_anomaly = 0`.
 */
chronosv_error_t chronosv_detect_anomaly(chronosv_engine_t* engine,
                                         const char* sensor_id,
                                         const float* v,
                                         size_t dim,
                                         float threshold,
                                         int* out_is_anomaly);

/* -------------------------------------------------------------------------- */
/* Maintenance (3)                                                            */
/* -------------------------------------------------------------------------- */

/**
 * @brief Update the age-out threshold and trigger one immediate eviction pass.
 *
 * Semantics:
 *   1. `cfg.window_duration_ms = window_ms`.
 *   2. Synchronously scan every registered sensor and advance its tail past
 *      the leading run of entries older than `max_ts_seen - window_ms`.
 *   3. Fire `on_eviction` metrics-sink callbacks for any advanced tails.
 *
 * This is the caller's hook for "shrink the window NOW" — it does not wait
 * for the next background eviction interval.
 *
 * @param engine    Handle.
 * @param window_ms New window duration in milliseconds. `0` evicts everything
 *                  (equivalent to what ::chronosv_flush does before syncing).
 * @return ::CHRONOSV_OK on success.
 * @note Does NOT force a flush to the cold tier. Use ::chronosv_flush for
 *       durability.
 */
chronosv_error_t chronosv_maintain_sliding_window(chronosv_engine_t* engine,
                                                  int64_t window_ms);

/**
 * @brief Remove all data for a sensor (hot ring + cold blocks).
 *
 * @param engine    Handle.
 * @param sensor_id Sensor identifier.
 * @return ::CHRONOSV_OK on success.
 * @retval ::CHRONOSV_ERR_NOT_FOUND Unknown sensor_id.
 */
chronosv_error_t chronosv_drop_sensor(chronosv_engine_t* engine,
                                      const char* sensor_id);

/**
 * @brief Block until all hot data is durable in the cold tier.
 *
 * Sequence:
 *   1. Trigger a synchronous eviction pass with `window_ms = 0` (evicts
 *      EVERYTHING regardless of the configured sliding window).
 *   2. Call `StorageBackend::Flush()` — for RocksDB, this flushes the
 *      memtable to L0 SSTs and then calls `SyncWAL()`.
 *
 * @param engine Handle.
 * @return ::CHRONOSV_OK on success, or the underlying storage error.
 * @note Latency: ~1–10 ms typical on SSD depending on WAL size (matches
 *       SQLite fsync cost). ChronosVector is optimized for the periodic-
 *       eviction path, not per-append durability. Use this on shutdown or
 *       at coarse checkpoints, not in the append hot path.
 */
chronosv_error_t chronosv_flush(chronosv_engine_t* engine);

/* -------------------------------------------------------------------------- */
/* Observability (2)                                                          */
/* -------------------------------------------------------------------------- */

/**
 * @brief Enumerate sensor IDs currently registered.
 *
 * @param engine   Handle.
 * @param[out] out_ids Array of `max` `char*` slots. Each returned slot points
 *                     to a newly `malloc`'d NUL-terminated copy of the
 *                     sensor_id. **Caller must `free()` each string.**
 * @param max      Capacity of `out_ids`.
 * @param[out] out_count `min(max, registered_sensors)`.
 * @return ::CHRONOSV_OK on success.
 * @retval ::CHRONOSV_ERR_OOM Allocation of a returned id string failed.
 */
chronosv_error_t chronosv_list_sensors(const chronosv_engine_t* engine,
                                       char** out_ids,
                                       size_t max,
                                       size_t* out_count);

/**
 * @brief Snapshot of engine stats.
 *
 * @param engine Handle.
 * @param[out] out Non-NULL. Fields are populated per ::chronosv_stats_t.
 * @return ::CHRONOSV_OK on success.
 * @note Reads are lock-free and eventually consistent. A snapshot may show
 *       one field reflecting `N` appends and another reflecting `N+1`.
 *       Callers must not rely on cross-field consistency.
 */
chronosv_error_t chronosv_get_stats(const chronosv_engine_t* engine,
                                    chronosv_stats_t* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CHRONOS_VECTOR_H */
