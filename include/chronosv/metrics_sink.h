/**
 * @file metrics_sink.h
 * @brief Push-based metrics sink (C function pointers).
 *
 * Deliberately plain C, not a C++ virtual class. A C++ interface would be
 * uncallable from Rust / Go / Python bindings via FFI, defeating the whole
 * purpose of the C ABI in `chronos_vector.h`. Function pointers work
 * identically from every language.
 *
 * @section sink_contract Callback contract
 *
 * Every callback is invoked from a hot path (producer thread, query thread,
 * or eviction thread). Callbacks MUST be:
 * - **non-blocking** — no I/O, no mutex, no allocation on a slow path
 * - **thread-safe** — may be invoked concurrently
 * - **noexcept** — an exception here has undefined behavior at the C boundary
 *
 * A slow sink slows down every append. For richer processing, buffer events
 * into a lock-free queue in the callback and drain from a worker thread.
 */
#ifndef CHRONOSV_METRICS_SINK_H
#define CHRONOSV_METRICS_SINK_H

#include <stdint.h>

#include "chronosv/types.h"  /* chronosv_error_t */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Function-pointer struct describing an observability sink.
 *
 * Any function pointer may be NULL — the engine skips that event. Partial
 * implementations are fine (e.g. a caller that only cares about append
 * latency sets `on_append` and leaves the rest NULL).
 */
struct chronosv_metrics_sink_t {
    /** Opaque to the engine; passed back on every callback so implementers
     *  can carry a `this` pointer / channel handle / logger. */
    void* user_data;

    /** @name Hot-path events (producer / consumer threads)
     * @{ */
    /** Fires after every successful ::chronosv_append. `latency_ns` is the
     *  end-to-end call latency measured inside the extern "C" wall. */
    void (*on_append)(void* ud, const char* sensor_id, int64_t latency_ns);

    /** Fires after every successful ::chronosv_query_nearest_n. */
    void (*on_query)(void* ud, const char* sensor_id, int64_t latency_ns, int result_count);

    /** Fires after every ::chronosv_detect_anomaly call whose result is
     *  `is_anomaly = 1`. `distance` is the value that exceeded the threshold. */
    void (*on_anomaly_detected)(void* ud, const char* sensor_id, float distance);
    /** @} */

    /** @name Data-loss events
     * @{ */
    /** Producer outpaced eviction; data was silently overwritten.*/
    void (*on_overwrite)(void* ud, const char* sensor_id, uint64_t entries_lost);
    /** @} */

    /** @name Cold-path events (eviction thread)
     * @{ */
    /** Fires when a block is persisted to the cold tier. `block_bytes` is
     *  the serialized on-disk size (post-compression if applicable). */
    void (*on_eviction)(void* ud, const char* sensor_id, int64_t evicted_count, int64_t block_bytes);

    /** Fires when a cold-tier write fails. Engine keeps running; the affected
     *  eviction is retried on the next pass. */
    void (*on_flush_error)(void* ud, const char* sensor_id, chronosv_error_t err);
    /** @} */

    /** @name Rare events
     * @{ */
    /** A persisted block failed CRC / magic / version validation on read. */
    void (*on_corruption)(void* ud, const char* sensor_id, int64_t block_id);

    /** A warning code was returned to a caller. Useful for centralised
     *  logging without wrapping every API call. */
    void (*on_warning)(void* ud, chronosv_error_t warn);
    /** @} */
};

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CHRONOSV_METRICS_SINK_H */
