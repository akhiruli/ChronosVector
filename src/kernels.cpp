/*
 * ChronosVector — SIMD distance kernel implementations.
 *
 * We map Eigen over the caller-provided buffers. Eigen's expression templates
 * auto-vectorize the matvec into NEON on Apple Silicon and AVX2 / AVX-512 on
 * x86. The compiler-emitted assembly should be inspected (`objdump -d`) once
 * the benchmark harness lands in Phase 3, to confirm we're getting the
 * vectorized paths we expect.
 */
#include "kernels.h"

#include <cmath>       /* std::lround for INT8 quantizer */
#include <cstdint>

#include <Eigen/Core>

namespace chronosv::internal {

/* -------------------------------------------------------------------------- */
/* L2 norm                                                                     */
/* -------------------------------------------------------------------------- */

float L2NormF32(const float* v, std::size_t dim) noexcept {
    using Eigen::Map;
    using Eigen::VectorXf;
    const auto vv = Map<const VectorXf>(v, static_cast<Eigen::Index>(dim));
    return vv.norm();
}

/* -------------------------------------------------------------------------- */
/* Cosine (float32)                                                            */
/* -------------------------------------------------------------------------- */

void CosineF32Chunk(const float* vecs,
                    const float* norms,
                    std::size_t  count,
                    std::size_t  dim,
                    const float* q,
                    float        qn,
                    float*       scores_out) noexcept {
    using Eigen::Map;
    using Eigen::MatrixXf;
    using Eigen::VectorXf;
    using Eigen::ArrayXf;
    using Idx = Eigen::Index;

    /* Map `vecs` as a [dim x count] column-major matrix. `vecs` is stored
     * row-major (each row is one vector of length `dim`), which Eigen treats
     * as `dim` rows of `count` cols under the default column-major mapping —
     * exactly what we want for `W.transpose() * q` to yield [count] dots. */
    const auto W  = Map<const MatrixXf>(vecs,
                                        static_cast<Idx>(dim),
                                        static_cast<Idx>(count));
    const auto qv = Map<const VectorXf>(q, static_cast<Idx>(dim));
    const auto rn = Map<const ArrayXf>(norms, static_cast<Idx>(count));

    /* Scratch for the dot products; written directly into the output buffer. */
    Map<ArrayXf> out(scores_out, static_cast<Idx>(count));
    out = (W.transpose() * qv).array();

    /* Vectorized normalize. Guard against zero row-norms and zero q-norm
     * (both would produce NaN); those slots get score 0, matching the
     * design's documented behavior for degenerate zero vectors. */
    if (qn > 0.0f) {
        out = (rn > 0.0f).select(out / (rn * qn), 0.0f);
    } else {
        out.setZero();
    }
}

/* -------------------------------------------------------------------------- */
/* Euclidean squared (float32) — fast path via norm identity                   */
/* -------------------------------------------------------------------------- */

void EuclideanSqF32Chunk(const float* vecs,
                         const float* row_norms,
                         std::size_t  count,
                         std::size_t  dim,
                         const float* q,
                         float        qn,
                         float*       scores_out) noexcept {
    using Eigen::Map;
    using Eigen::MatrixXf;
    using Eigen::VectorXf;
    using Eigen::ArrayXf;
    using Idx = Eigen::Index;

    const auto W  = Map<const MatrixXf>(vecs,
                                        static_cast<Idx>(dim),
                                        static_cast<Idx>(count));
    const auto qv = Map<const VectorXf>(q, static_cast<Idx>(dim));
    const auto rn = Map<const ArrayXf>(row_norms, static_cast<Idx>(count));

    Map<ArrayXf> out(scores_out, static_cast<Idx>(count));

    /* Step 1: dot products via the same matvec cosine uses. */
    out = (W.transpose() * qv).array();

    /* Step 2: ||W_i||^2 + ||q||^2 - 2 * (W_i . q). */
    const float qn2 = qn * qn;
    out = rn.square() + qn2 - 2.0f * out;

    /* Step 3: clamp small negative values (from catastrophic cancellation
     * when W_i ≈ q) up to zero. Distance is non-negative by definition. */
    out = out.max(0.0f);
}

/* -------------------------------------------------------------------------- */
/* INT8 quantization + kernels                                                 */
/*                                                                            */
/* Scalar loops for now (dim=128 int8 fits well in registers; the compiler    */
/* auto-vectorizes with -O3 on both NEON and AVX2). NEON vmull_s8 /            */
/* AVX-VNNI accelerated paths are a future optimization if the profile        */
/* justifies it.                                                              */
/* -------------------------------------------------------------------------- */

void QuantizeF32ToI8(const float* v,
                     std::size_t  dim,
                     std::int8_t* out_q,
                     float*       out_scale) noexcept {
    /* Two-pass: find max_abs, then scale. */
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        const float a = v[i] < 0 ? -v[i] : v[i];
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) {
        /* Zero vector — represented as all-zero q with scale=0. Kernels
         * handle this explicitly by producing score=0. */
        for (std::size_t i = 0; i < dim; ++i) out_q[i] = 0;
        *out_scale = 0.0f;
        return;
    }
    const float scale     = max_abs / 127.0f;
    const float inv_scale = 127.0f / max_abs;
    for (std::size_t i = 0; i < dim; ++i) {
        float scaled = v[i] * inv_scale;
        /* Clamp to [-127, 127] — INT8_MIN (-128) is deliberately excluded
         * to keep symmetric quantization. */
        if (scaled >  127.0f) scaled =  127.0f;
        if (scaled < -127.0f) scaled = -127.0f;
        /* Round-half-away-from-zero via lround (avoids compiler-dependent
         * banker's rounding). */
        out_q[i] = static_cast<std::int8_t>(std::lround(scaled));
    }
    *out_scale = scale;
}

/* Scalar int8 dot product with int32 accumulator. dim * 127*127 fits in
 * int32 up to dim ~131k, well beyond any realistic value. */
static inline std::int32_t dot_i8(const std::int8_t* a,
                                  const std::int8_t* b,
                                  std::size_t dim) noexcept {
    std::int32_t acc = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        acc += static_cast<std::int16_t>(a[i])
             * static_cast<std::int16_t>(b[i]);
    }
    return acc;
}

void CosineI8Chunk(const std::int8_t* vecs,
                   const float*       scales,
                   const float*       row_norms,
                   std::size_t        count,
                   std::size_t        dim,
                   const std::int8_t* q,
                   float              q_scale,
                   float              q_norm,
                   float*             scores_out) noexcept {
    /* Shortcut: zero query norm → all scores zero. */
    if (q_norm <= 0.0f || q_scale <= 0.0f) {
        for (std::size_t i = 0; i < count; ++i) scores_out[i] = 0.0f;
        return;
    }
    const float q_norm_inv = 1.0f / q_norm;
    for (std::size_t i = 0; i < count; ++i) {
        if (row_norms[i] <= 0.0f || scales[i] <= 0.0f) {
            scores_out[i] = 0.0f;
            continue;
        }
        const std::int32_t d = dot_i8(vecs + i * dim, q, dim);
        /* dot_float = d * scales[i] * q_scale; normalized by norms product. */
        scores_out[i] = (static_cast<float>(d) * scales[i] * q_scale)
                        * (q_norm_inv / row_norms[i]);
    }
}

void EuclideanSqI8Chunk(const std::int8_t* vecs,
                        const float*       scales,
                        const float*       row_norms,
                        std::size_t        count,
                        std::size_t        dim,
                        const std::int8_t* q,
                        float              q_scale,
                        float              q_norm,
                        float*             scores_out) noexcept {
    /* ||a-b||^2 = ||a||^2 + ||b||^2 - 2*(a.b), where a.b in float space
     * is (dot_i8 * scale_a * scale_b). Same numerical caveats as the
     * float32 fast path: clamp negative results to zero. */
    const float qn2 = q_norm * q_norm;
    for (std::size_t i = 0; i < count; ++i) {
        const std::int32_t d       = dot_i8(vecs + i * dim, q, dim);
        const float        dot_flt = static_cast<float>(d) * scales[i] * q_scale;
        const float        rn2     = row_norms[i] * row_norms[i];
        float sq = rn2 + qn2 - 2.0f * dot_flt;
        if (sq < 0.0f) sq = 0.0f;
        scores_out[i] = sq;
    }
}

/* -------------------------------------------------------------------------- */
/* Euclidean squared (float32) — direct reference path                         */
/* -------------------------------------------------------------------------- */

void EuclideanSqF32Chunk_Direct(const float* vecs,
                                std::size_t  count,
                                std::size_t  dim,
                                const float* q,
                                float*       scores_out) noexcept {
    using Eigen::Map;
    using Eigen::MatrixXf;
    using Eigen::VectorXf;
    using Idx = Eigen::Index;

    const auto W  = Map<const MatrixXf>(vecs,
                                        static_cast<Idx>(dim),
                                        static_cast<Idx>(count));
    const auto qv = Map<const VectorXf>(q, static_cast<Idx>(dim));

    Map<Eigen::VectorXf> out(scores_out, static_cast<Idx>(count));

    /* Broadcast subtract + squaredNorm. Numerically robust but slower —
     * Eigen doesn't vectorize the colwise-subtract as well as matvec. */
    out = (W.colwise() - qv).colwise().squaredNorm();
}

}  // namespace chronosv::internal
