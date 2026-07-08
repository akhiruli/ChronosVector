/*
 * ChronosVector — lock-free SPSC ring buffer (internal, header-only).
 *
 * NOT part of the public ABI. Included via internal engine.cpp only.
 *
 * Key contract:
 *   - Single Producer / Single-consumer-for-eviction / Multi-consumer-for-queries.
 *   - Producer thread bumps head (release) after payload writes (relaxed).
 *   - Eviction thread bumps tail (release) after copying evicted range out.
 *   - Query threads acquire-load head and tail, iterate the snapshot.
 *   - Overwrite is by design: when head - tail >= capacity, the producer laps
 *     the consumer and the oldest slot is clobbered. An overwrite counter is
 *     incremented so operators can detect it (see chronosv_stats_t).
 */
#ifndef CHRONOSV_RING_BUFFER_H
#define CHRONOSV_RING_BUFFER_H

#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <new>
#include <thread>
#include <utility>

#include "chronosv/types.h"

namespace chronosv::internal {

/* Cacheline size assumed here. std::hardware_destructive_interference_size is
 * the "correct" answer but libc++ still doesn't ship it uniformly. 64 is right
 * on all our current targets (x86_64, ARM64). */
inline constexpr std::size_t kCacheline = 64;

/* Round up `n` bytes to the next multiple of `align`. std::aligned_alloc
 * requires size to be a multiple of alignment. */
inline constexpr std::size_t round_up(std::size_t n, std::size_t align) noexcept {
    return (n + align - 1) & ~(align - 1);
}

/* Allocate `bytes` (rounded up) aligned to kCacheline. Returns nullptr on OOM
 * (never throws). Free with aligned_free below. */
inline void* aligned_alloc_bytes(std::size_t bytes) noexcept {
    if (bytes == 0) return nullptr;
    const std::size_t rounded = round_up(bytes, kCacheline);
#if defined(_WIN32)
    return _aligned_malloc(rounded, kCacheline);
#else
    return std::aligned_alloc(kCacheline, rounded);
#endif
}
inline void aligned_free(void* p) noexcept {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

/* Cheap hash of the current thread id — used only for the debug-mode SPSC
 * guard. `0` is reserved as "no producer yet claimed", so we OR in a bit to
 * ensure we never return zero. */
inline std::uint64_t current_thread_id_hash() noexcept {
    const auto h = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return static_cast<std::uint64_t>(h) | 0x1ULL;
}

/* Size of one element of a given dtype, in bytes. */
inline constexpr std::size_t dtype_size(chronosv_dtype_t dt) noexcept {
    switch (dt) {
        case CHRONOSV_DTYPE_FLOAT32: return sizeof(float);
        case CHRONOSV_DTYPE_INT8:    return sizeof(std::int8_t);
    }
    return 0;
}

/* -------------------------------------------------------------------------- */
/* SensorRing                                                                 */
/* -------------------------------------------------------------------------- */

class alignas(kCacheline) SensorRing {
public:
    /* Not copyable, not movable — pinned in place because it owns aligned
     * allocations and has atomics whose addresses matter (queries and
     * eviction hold references). The engine stores rings via
     * std::unique_ptr<SensorRing> in a map keyed by sensor_id. */
    SensorRing(const SensorRing&)            = delete;
    SensorRing& operator=(const SensorRing&) = delete;
    SensorRing(SensorRing&&)                 = delete;
    SensorRing& operator=(SensorRing&&)      = delete;

    /* Construct a ring. `capacity` MUST be a power of two >= 64; the caller
     * (engine validator) enforces this before we get here.
     *
     * All backing allocations happen here. Fails via `ok()` returning false
     * (never throws — hot path is noexcept end-to-end). */
    SensorRing(std::uint64_t capacity,
               std::uint32_t dim,
               std::uint32_t payload_size,
               chronosv_dtype_t dtype) noexcept
        : capacity_(capacity),
          mask_(capacity - 1),
          dim_(dim),
          payload_size_(payload_size),
          dtype_(dtype) {
        assert(capacity_ >= 64 && "capacity too small");
        assert((capacity_ & mask_) == 0 && "capacity must be power of two");
        assert(dim_ > 0 && "dim must be > 0");

        timestamps_ = static_cast<std::int64_t*>(
            aligned_alloc_bytes(capacity_ * sizeof(std::int64_t)));
        vectors_ = aligned_alloc_bytes(capacity_ * dim_ * dtype_size(dtype_));
        norms_ = static_cast<float*>(
            aligned_alloc_bytes(capacity_ * sizeof(float)));

        if (dtype_ == CHRONOSV_DTYPE_INT8) {
            scales_ = static_cast<float*>(
                aligned_alloc_bytes(capacity_ * sizeof(float)));
        }
        if (payload_size_ > 0) {
            payloads_ = static_cast<std::uint8_t*>(
                aligned_alloc_bytes(capacity_ * payload_size_));
        }
        ok_ = timestamps_ && vectors_ && norms_
              && (dtype_ != CHRONOSV_DTYPE_INT8 || scales_)
              && (payload_size_ == 0 || payloads_);
    }

    ~SensorRing() {
        aligned_free(timestamps_);
        aligned_free(vectors_);
        aligned_free(norms_);
        aligned_free(scales_);
        aligned_free(payloads_);
    }

    bool ok() const noexcept { return ok_; }

    /* ---- Accessors (immutable after construction) ---- */
    std::uint64_t     capacity()      const noexcept { return capacity_; }
    std::uint32_t     dim()           const noexcept { return dim_; }
    std::uint32_t     payload_size()  const noexcept { return payload_size_; }
    chronosv_dtype_t  dtype()         const noexcept { return dtype_; }

    /* ---- Producer path (SPSC, hot) --------------------------------------
     *
     * Called by exactly one thread per sensor. Bytes referenced by `vec` and
     * `payload` are read fully before head advances; caller may reuse those
     * buffers immediately on return. `precomputed_norm` is the L2 norm of
     * `vec` — computed by the engine core (single O(D) pass) so the ring
     * doesn't need dtype-specific norm logic.
     *
     * `precomputed_scale` is only meaningful when dtype == INT8 (per-vector
     * quantization scale). Ignored for FLOAT32; callers may pass 0. */
    void Append(std::int64_t ts,
                const void* vec,
                float precomputed_norm,
                const void* payload,
                float precomputed_scale = 0.0f) noexcept {
        DebugCheckSpsc();

        const std::uint64_t h = head_.load(std::memory_order_relaxed);

        /* Overwrite detection: if the slot at head is still "live" from the
         * consumer's view (head - tail >= capacity), we are about to clobber
         * an entry that has not yet been evicted. Bump the counters. This is
         * design-permitted but must be observable. */
        const std::uint64_t t = tail_.load(std::memory_order_acquire);
        if (h - t >= capacity_) {
            overwrite_events_.fetch_add(1, std::memory_order_relaxed);
            overwritten_entries_.fetch_add(1, std::memory_order_relaxed);
        }

        const std::uint64_t idx = h & mask_;
        const std::size_t   vsz = static_cast<std::size_t>(dim_) * dtype_size(dtype_);

        /* Prefetch the NEXT slot's vector region into L1 with write intent.
         * Ring appends cycle through slots so per-slot access is inherently
         * cold — without prefetch, each memcpy below stalls on first-line
         * miss at large dim. Placing the hint here lets the ~50 ns memory
         * latency overlap with the current call's memcpy + norm write, which
         * hides the miss and flattens the P99/P999 tail. Hardware prefetch
         * takes over for subsequent lines once the first is loading.
         *
         * Measured on Apple Silicon (bench/bench_engine.cpp BM_AppendLatencyDist):
         *   dim=128:  no measurable change (already fits in one cacheline)
         *   dim=512:  P99 1375 ns → 250 ns (5.5×), P999 3583 ns → 1333 ns (2.7×)
         *   dim=1024: marginal — residual tail attributable to OS scheduler
         *             at that call length, not cache misses.
         *
         * A strided/multi-line prefetch was tried and did not measurably beat
         * this single-line version. The hardware prefetcher does the rest. */
        const std::uint64_t next_idx = (h + 1) & mask_;
        __builtin_prefetch(
            static_cast<std::uint8_t*>(vectors_) + next_idx * vsz,
            /*rw=*/1, /*locality=*/3);

        timestamps_[idx] = ts;
        std::memcpy(static_cast<std::uint8_t*>(vectors_) + idx * vsz, vec, vsz);
        norms_[idx] = precomputed_norm;
        if (dtype_ == CHRONOSV_DTYPE_INT8 && scales_) {
            scales_[idx] = precomputed_scale;
        }
        if (payload_size_ > 0 && payload) {
            std::memcpy(payloads_ + idx * payload_size_, payload, payload_size_);
        }

        /* Release: any reader that observes the new head also observes the
         * payload writes above. */
        head_.store(h + 1, std::memory_order_release);
    }

    /* ---- Reader path (query / eviction) --------------------------------
     *
     * Returns [begin, end) — a half-open interval of monotonically-increasing
     * "virtual" indices (not masked). Use idx & mask() to map to a slot.
     *
     * If the producer has lapped the reader (head - tail > capacity), begin is
     * clamped to head - capacity, i.e. we return the newest `capacity` entries
     * only. Older entries were overwritten and are gone. */
    std::pair<std::uint64_t, std::uint64_t> Snapshot() const noexcept {
        const std::uint64_t t = tail_.load(std::memory_order_acquire);
        const std::uint64_t h = head_.load(std::memory_order_acquire);
        const std::uint64_t begin = (h - t > capacity_) ? (h - capacity_) : t;
        return {begin, h};
    }

    /* Called by eviction after it has copied out the range [tail, new_tail).
     * new_tail must be >= current tail and <= current head. */
    void AdvanceTail(std::uint64_t new_tail) noexcept {
        tail_.store(new_tail, std::memory_order_release);
    }

    /* ---- Slot readers (const, for query kernels) ------------------------ */

    std::int64_t timestamp_at(std::uint64_t idx) const noexcept {
        return timestamps_[idx & mask_];
    }
    const void* vector_at(std::uint64_t idx) const noexcept {
        const std::size_t vsz = static_cast<std::size_t>(dim_) * dtype_size(dtype_);
        return static_cast<const std::uint8_t*>(vectors_) + (idx & mask_) * vsz;
    }
    float norm_at(std::uint64_t idx) const noexcept {
        return norms_[idx & mask_];
    }
    const void* payload_at(std::uint64_t idx) const noexcept {
        if (payload_size_ == 0) return nullptr;
        return payloads_ + (idx & mask_) * payload_size_;
    }

    /* Contiguous accessors — useful for kernels that want to Eigen-map the
     * whole underlying array. Callers must respect the mask themselves. */
    const std::int64_t* timestamps_raw() const noexcept { return timestamps_; }
    const void*         vectors_raw()    const noexcept { return vectors_; }
    const float*        norms_raw()      const noexcept { return norms_; }
    const float*        scales_raw()     const noexcept { return scales_; }  /* nullptr unless INT8 */
    std::uint64_t       mask()           const noexcept { return mask_; }

    /* ---- Observability -------------------------------------------------- */

    std::uint64_t overwrite_events() const noexcept {
        return overwrite_events_.load(std::memory_order_relaxed);
    }
    std::uint64_t overwritten_entries() const noexcept {
        return overwritten_entries_.load(std::memory_order_relaxed);
    }
    std::uint64_t hot_bytes() const noexcept {
        const std::size_t vsz = static_cast<std::size_t>(dim_) * dtype_size(dtype_);
        std::uint64_t b = capacity_ * (sizeof(std::int64_t) + vsz + sizeof(float));
        if (dtype_ == CHRONOSV_DTYPE_INT8) b += capacity_ * sizeof(float);
        if (payload_size_ > 0)             b += capacity_ * payload_size_;
        return b;
    }

private:
    void DebugCheckSpsc() noexcept {
#ifndef NDEBUG
        const std::uint64_t tid = current_thread_id_hash();
        std::uint64_t expected = 0;
        if (!producer_tid_hash_.compare_exchange_strong(
                expected, tid, std::memory_order_relaxed)) {
            assert(expected == tid
                && "SPSC contract violated: multiple producer threads on one sensor");
        }
#endif
    }

    /* --- Cacheline 0: producer-hot head_ --- */
    alignas(kCacheline) std::atomic<std::uint64_t> head_{0};

    /* --- Cacheline 1: consumer-hot tail_ --- */
    alignas(kCacheline) std::atomic<std::uint64_t> tail_{0};

    /* --- Cacheline 2: overwrite counters (written by producer, read by observer) --- */
    alignas(kCacheline) std::atomic<std::uint64_t> overwrite_events_{0};
    std::atomic<std::uint64_t> overwritten_entries_{0};

#ifndef NDEBUG
    std::atomic<std::uint64_t> producer_tid_hash_{0};
#endif

    /* Immutable after construction */
    std::uint64_t     capacity_;
    std::uint64_t     mask_;
    std::uint32_t     dim_;
    std::uint32_t     payload_size_;
    chronosv_dtype_t  dtype_;
    bool              ok_ = false;

    /* Backing storage */
    std::int64_t* timestamps_ = nullptr;
    void*         vectors_    = nullptr;
    float*        norms_      = nullptr;
    float*        scales_     = nullptr;  /* INT8 only */
    std::uint8_t* payloads_   = nullptr;
};

}  // namespace chronosv::internal

#endif  // CHRONOSV_RING_BUFFER_H
