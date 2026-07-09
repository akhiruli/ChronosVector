/*
 * ChronosVector — SIMD distance kernels (internal).
 *
 * Not part of the public ABI. Consumed by engine.cpp.
 *
 * Kernels operate on a *contiguous chunk* of vectors, not the full ring
 * snapshot. The caller (engine) is responsible for splitting a snapshot
 * that wraps around the ring boundary into two contiguous chunks and
 * invoking the kernel twice.
 *
 * All kernels are noexcept and never allocate; they map Eigen expressions
 * over caller-provided buffers.
 */
#ifndef CHRONOSV_KERNELS_H
#define CHRONOSV_KERNELS_H

#include <cstddef>

#include <cstdint>

namespace chronosv::internal {

/* Compute the L2 norm of a dim-length float vector. Used on the append path
 * to cache per-row norms in the ring buffer. */
float L2NormF32(const float* v, std::size_t dim) noexcept;

/* --------------------------------------------------------------------------
 * INT8 quantization + kernels.
 *
 * Per-vector symmetric quantization: `q[i] = round(v[i] / scale)` where
 * `scale = max(|v[i]|) / 127`. Zero vectors get scale=0 and are handled
 * specially (all-zero q, cosine=0 result).
 *
 * Memory: `int8_t q[dim]` + `float scale` per vector — ~4x smaller than
 * float32. The primary benefit is bounded-RAM reduction and 2.5-4.5x
 * query-time speedup from reduced memory bandwidth; a hand-tuned SIMD
 * int8 dot-product path (for further compute speedup) is a future
 * optimization.
 *
 * Accuracy: measured Recall@10 drop vs FP32:
 *   -0.66 pp on 384-dim unit-normalized text embeddings (MiniLM/BERT)
 *   -3.09 pp on 128-dim raw visual descriptors (SIFT-1M)
 * See docs/INT8.md for the full measured story and tests/int8_recall/
 * for the validation harness (works on user's own embeddings).
 * -------------------------------------------------------------------------- */

/* Quantize a float32 vector to int8 + scale. Writes `dim` int8 values to
 * `out_q` and one float to `*out_scale`. Zero-vector special case:
 * out_q[i]=0 for all i, *out_scale=0. */
void QuantizeF32ToI8(const float* v,
                     std::size_t  dim,
                     std::int8_t* out_q,
                     float*       out_scale) noexcept;

/* Cosine similarity, INT8 storage.
 *
 * For each of `count` rows of `vecs` (packed `int8_t[count][dim]`), compute
 * `cosine(row, q)` and write to `scores_out[i]`.
 *
 * The math:
 *   cos(a, b) = (a . b) / (||a|| * ||b||)
 *   where a = q_a * scale_a (float reconstruction from int8)
 *         a . b = scale_a * scale_b * sum(q_a[i] * q_b[i])
 *
 * The `norms` array holds ||original_vec|| (i.e., the float L2 norm before
 * quantization); this is what SensorRing already caches at Append time.
 *
 * Zero norms (from zero vectors) produce score=0. */
void CosineI8Chunk(const std::int8_t* vecs,
                   const float*       scales,      /* [count] per-row scale */
                   const float*       row_norms,   /* [count] original-float L2 norm */
                   std::size_t        count,
                   std::size_t        dim,
                   const std::int8_t* q,
                   float              q_scale,
                   float              q_norm,      /* original-float L2 norm of q */
                   float*             scores_out) noexcept;

/* Squared euclidean distance, INT8 storage. Same identity trick as the
 * fast float32 kernel: `||a-b||^2 = ||a||^2 + ||b||^2 - 2*(a.b)`. Uses
 * cached row_norms and q_norm (both float32, original-vector L2 norms).
 * Clamps to zero for numerical safety on near-identical vectors. */
void EuclideanSqI8Chunk(const std::int8_t* vecs,
                        const float*       scales,
                        const float*       row_norms,
                        std::size_t        count,
                        std::size_t        dim,
                        const std::int8_t* q,
                        float              q_scale,
                        float              q_norm,
                        float*             scores_out) noexcept;

/* Cosine similarity: for each of `count` rows in `vecs` (row-major, [count][dim]),
 * writes `(row · q) / (row_norm * qn)` to scores_out[i]. `norms` is the
 * pre-computed per-row L2 norm array (from Append time). If a row_norm or qn
 * is zero, the score is 0.
 *
 * Preconditions (not checked): count > 0, dim > 0, qn >= 0. */
void CosineF32Chunk(const float* vecs,
                    const float* norms,
                    std::size_t  count,
                    std::size_t  dim,
                    const float* q,
                    float        qn,
                    float*       scores_out) noexcept;

/* Squared euclidean distance: for each of `count` rows in `vecs`, writes
 * `||row - q||^2` to scores_out[i]. Squared (not sqrt'd) because it preserves
 * ordering and saves the sqrt on the hot path. Callers that need actual
 * distance take the sqrt after top-N selection.
 *
 * Uses the identity ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * (a . b), which
 * lets us reuse the same fast Eigen matvec that cosine uses instead of the
 * far-slower broadcast-subtract form. `row_norms` and `qn` must be the
 * precomputed L2 norms of `vecs` and `q` respectively (same values the
 * ring buffer already caches).
 *
 * Numerical note: the identity is subject to catastrophic cancellation when
 * a ≈ b (subtraction of two near-equal large numbers). The kernel clamps
 * negative results to zero for safety. Absolute values may differ from a
 * direct computation by ~1e-5 relative on typical dim=128 vectors, but the
 * ORDERING of results (which is what top-N selection cares about) is
 * preserved. */
void EuclideanSqF32Chunk(const float* vecs,
                         const float* row_norms,
                         std::size_t  count,
                         std::size_t  dim,
                         const float* q,
                         float        qn,
                         float*       scores_out) noexcept;

/* Reference implementation of squared euclidean using the direct
 * (b - a)-subtract form. Slower (does not use precomputed norms, does not
 * share the matvec) but numerically robust against cancellation. Kept for
 * tests / accuracy comparisons and available if a future caller needs the
 * higher-precision path. NOT used by the engine hot path. */
void EuclideanSqF32Chunk_Direct(const float* vecs,
                                std::size_t  count,
                                std::size_t  dim,
                                const float* q,
                                float*       scores_out) noexcept;

}  // namespace chronosv::internal

#endif  // CHRONOSV_KERNELS_H
