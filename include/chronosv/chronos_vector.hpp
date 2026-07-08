/**
 * @file chronos_vector.hpp
 * @brief C++23 convenience wrapper over the C ABI.
 *
 * Header-only, inline, zero runtime cost. **This is not the stability
 * contract** — the C ABI in `chronos_vector.h` is. This wrapper is sugar for
 * C++ users who prefer RAII, `std::span`, and `std::expected`-based error
 * handling.
 *
 * Requires C++23 (`<expected>`, `<span>`). If your build is C++20 or older,
 * include `chronos_vector.h` directly and use the C API.
 *
 */
#ifndef CHRONOS_VECTOR_HPP
#define CHRONOS_VECTOR_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "chronosv/chronos_vector.h"

namespace chronosv {

/**
 * @brief Strongly-typed error enum mirroring the C `CHRONOSV_ERR_*` /
 *        `CHRONOSV_WARN_*` codes.
 *
 * Positive values are warnings; negative are errors; `Ok` is `0`. See
 * ::is_error / ::is_warning for classification helpers.
 */
enum class Error : ::chronosv_error_t {
    Ok                   = CHRONOSV_OK,
    WarnRangeTruncated   = CHRONOSV_WARN_RANGE_TRUNCATED,
    WarnPartialResult    = CHRONOSV_WARN_PARTIAL_RESULT,
    InvalidArg           = CHRONOSV_ERR_INVALID_ARG,
    NotFound             = CHRONOSV_ERR_NOT_FOUND,
    DimMismatch          = CHRONOSV_ERR_DIM_MISMATCH,
    Io                   = CHRONOSV_ERR_IO,
    Oom                  = CHRONOSV_ERR_OOM,
    Unsupported          = CHRONOSV_ERR_UNSUPPORTED,
    Closed               = CHRONOSV_ERR_CLOSED,
    Capacity             = CHRONOSV_ERR_CAPACITY,
    Corruption           = CHRONOSV_ERR_CORRUPTION,
    Internal             = CHRONOSV_ERR_INTERNAL,
};

/** @return `true` if `e` denotes a hard error (out-params undefined). */
inline bool is_error(Error e) noexcept {
    return static_cast<::chronosv_error_t>(e) < 0;
}

/** @return `true` if `e` denotes a warning (out-params valid, caller may want to log). */
inline bool is_warning(Error e) noexcept {
    return static_cast<::chronosv_error_t>(e) > 0;
}

/** @return Human-readable string for `e`; static storage, do not free. */
inline std::string_view error_string(Error e) noexcept {
    return ::chronosv_error_string(static_cast<::chronosv_error_t>(e));
}

/** Convenience alias for the `std::expected<T, Error>` return type used by
 *  fallible ::Engine methods. */
template <typename T>
using Result = std::expected<T, Error>;

/** One nearest-neighbor match returned by ::Engine::QueryNearestN. */
struct Match {
    std::int64_t timestamp_ms;  /**< Timestamp of the matched entry. */
    float        score;         /**< Distance or similarity per engine's metric. */
};

/**
 * @brief RAII wrapper around ::chronosv_engine_t.
 *
 * Constructed via the ::Create factory, released automatically on destruction
 * (calls ::chronosv_destroy). Move-only. Not thread-safe for lifecycle ops;
 * ::Append / ::QueryNearestN follow the same threading contract as the
 * underlying C API (per-sensor SPSC for append, MT-safe for queries).
 */
class Engine {
public:
    /**
     * @brief Create a fresh engine.
     * @param cfg Configuration. See ::chronosv_config_t.
     * @return An owning ::Engine, or the specific ::Error on failure.
     */
    static Result<Engine> Create(const ::chronosv_config_t& cfg) noexcept {
        ::chronosv_error_t err = CHRONOSV_OK;
        auto* h = ::chronosv_create(&cfg, &err);
        if (!h) return std::unexpected(static_cast<Error>(err));
        return Engine(h);
    }

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    Engine(Engine&& other) noexcept : h_(std::exchange(other.h_, nullptr)) {}
    Engine& operator=(Engine&& other) noexcept {
        if (this != &other) {
            reset();
            h_ = std::exchange(other.h_, nullptr);
        }
        return *this;
    }

    ~Engine() { reset(); }

    /**
     * @brief Explicit close, useful if you want to observe a close error.
     * @return `Error::Ok` on success or the underlying storage error. Further
     *         calls on this engine return `Error::Closed`.
     */
    Error Close() noexcept {
        if (!h_) return Error::Closed;
        return static_cast<Error>(::chronosv_close(h_));
    }

    /**
     * @brief Append one vector for a sensor. Hot-path operation.
     * @param sensor_id NUL-terminated view. **Must be null-terminated** — the
     *                  C ABI reads a `const char*`. If the caller passes a
     *                  `string_view` over a non-terminated buffer, they must
     *                  wrap it first; we do not silently copy on the hot path.
     * @param ts_ms Timestamp in milliseconds.
     * @param vec Vector, whose size must equal the engine's configured `dim`.
     * @param payload Optional fixed-size payload.
     * @return `Error::Ok` on success or a specific error code.
     * @see chronosv_append
     */
    Error Append(std::string_view sensor_id, std::int64_t ts_ms,
                 std::span<const float> vec,
                 const void* payload = nullptr) noexcept {
        return static_cast<Error>(::chronosv_append(
            h_, sensor_id.data(), ts_ms, vec.data(), vec.size(), payload));
    }

    /**
     * @brief k-NN query over the current hot window.
     * @param sensor_id Sensor identifier (NUL-terminated).
     * @param target Query vector, size must equal engine's `dim`.
     * @param n Number of results requested.
     * @return Vector of up to `n` ::Match entries, sorted best-first, or an
     *         ::Error on failure. Warnings (::Error::WarnPartialResult) are
     *         swallowed — the caller sees `result.size() < n` instead.
     * @see chronosv_query_nearest_n
     */
    Result<std::vector<Match>>
    QueryNearestN(std::string_view sensor_id,
                  std::span<const float> target, int n) const noexcept {
        std::vector<std::int64_t> ts(n);
        std::vector<float>        sc(n);
        int count = 0;
        auto err = ::chronosv_query_nearest_n(
            h_, sensor_id.data(), target.data(), target.size(), n,
            ts.data(), sc.data(), &count);
        if (err < 0) return std::unexpected(static_cast<Error>(err));
        std::vector<Match> out;
        out.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) out.push_back({ts[i], sc[i]});
        return out;
    }

    /**
     * @brief Anomaly check against the rolling window centroid.
     * @param sensor_id Sensor identifier.
     * @param v Candidate vector.
     * @param threshold Distance above which the vector is flagged.
     * @return `true` if anomalous, `false` if normal, or an ::Error.
     * @see chronosv_detect_anomaly for threshold-unit semantics.
     */
    Result<bool> DetectAnomaly(std::string_view sensor_id,
                               std::span<const float> v,
                               float threshold) const noexcept {
        int is_anomaly = 0;
        auto err = ::chronosv_detect_anomaly(
            h_, sensor_id.data(), v.data(), v.size(), threshold, &is_anomaly);
        if (err < 0) return std::unexpected(static_cast<Error>(err));
        return is_anomaly != 0;
    }

    /**
     * @brief Snapshot of engine stats.
     * @return Filled ::chronosv_stats_t or an ::Error. See ::chronosv_get_stats
     *         for consistency guarantees.
     */
    Result<::chronosv_stats_t> GetStats() const noexcept {
        ::chronosv_stats_t s{};
        auto err = ::chronosv_get_stats(h_, &s);
        if (err < 0) return std::unexpected(static_cast<Error>(err));
        return s;
    }

    /** @name Raw handle escape hatch
     *
     * For callers who need to invoke C-only APIs not yet wrapped. Ownership
     * is not transferred; do not call ::chronosv_destroy on the returned
     * pointer.
     * @{ */
    ::chronosv_engine_t* raw() noexcept { return h_; }
    const ::chronosv_engine_t* raw() const noexcept { return h_; }
    /** @} */

private:
    explicit Engine(::chronosv_engine_t* h) noexcept : h_(h) {}

    void reset() noexcept {
        if (h_) {
            ::chronosv_destroy(h_);
            h_ = nullptr;
        }
    }

    ::chronosv_engine_t* h_ = nullptr;
};

}  // namespace chronosv

#endif  // CHRONOS_VECTOR_HPP
